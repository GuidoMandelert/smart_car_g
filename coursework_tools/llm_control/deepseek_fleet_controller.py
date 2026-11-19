"""Centralized, batched DeepSeek controller for a simulated vehicle fleet."""

import json
import logging
import math
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATOR_DIR = PROJECT_ROOT / "car-following-simulation"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from vehicle_idm import IDMVehicle


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
logger = logging.getLogger(__name__)


def get_api_key():
    """Load the project-local key, while allowing an environment override."""
    env_path = SIMULATOR_DIR / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Set DEEPSEEK_API_KEY in car-following-simulation\\.env "
            "or in your environment before selecting the DeepSeek fleet controller."
        )
    return api_key


class DeepSeekFleetController:
    """Request fleet-wide speed targets and safety-filter every step."""

    def __init__(
        self,
        model=DEFAULT_MODEL,
        timeout_seconds=60.0,
        decision_interval_steps=200,
        api_key=None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if decision_interval_steps < 1:
            raise ValueError("decision_interval_steps must be positive")

        self.model = model
        self.timeout_seconds = timeout_seconds
        self.decision_interval_steps = decision_interval_steps
        self.api_key = api_key if api_key is not None else get_api_key()
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")

        self.request_count = 0
        self.successful_batch_count = 0
        self.failed_batch_count = 0
        self.idm_fallback_vehicle_steps = 0
        self.safety_filter_interventions = 0
        self.minimum_gap_unachievable_steps = 0
        self.clamped_target_speeds = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.usage_reported_batch_count = 0
        self.usage_missing_batch_count = 0
        self._cached_target_speeds = {}
        self._effective_target_speeds = {}

    def actions_for_step(self, vehicles, step):
        """Return cached target speeds, refreshing the fleet on schedule."""
        active_vehicles = [vehicle for vehicle in vehicles if vehicle.active]
        if not active_vehicles:
            self._cached_target_speeds = {}
            self._effective_target_speeds = {}
            return {}

        if step % self.decision_interval_steps == 0:
            self.request_count += 1
            self._current_step = step
            try:
                self._cached_target_speeds = self._request_target_speeds(
                    active_vehicles
                )
                self.successful_batch_count += 1
            except HTTPError as exc:
                self._record_batch_failure(f"HTTP {exc.code}")
            except (
                URLError,
                TimeoutError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                UnicodeError,
            ) as exc:
                self._record_batch_failure(f"{type(exc).__name__}: {exc}")

        active_ids = {vehicle.id for vehicle in active_vehicles}
        self._effective_target_speeds = {
            vehicle_id: speed
            for vehicle_id, speed in self._effective_target_speeds.items()
            if vehicle_id in active_ids
        }
        effective_targets = {}
        for vehicle in active_vehicles:
            if vehicle.id not in self._cached_target_speeds:
                self._effective_target_speeds.pop(vehicle.id, None)
                continue
            requested_speed = self._cached_target_speeds.get(
                vehicle.id,
                vehicle.current_speed_limit,
            )
            previous_target = self._effective_target_speeds.get(
                vehicle.id,
                vehicle.speed,
            )
            target_change_rate = (
                vehicle.comfortable_deceleration
                if requested_speed < previous_target
                else vehicle.maximum_acceleration
            )
            maximum_change = target_change_rate * vehicle.time_step
            target_change = max(
                -maximum_change,
                min(requested_speed - previous_target, maximum_change),
            )
            effective_speed = previous_target + target_change
            self._effective_target_speeds[vehicle.id] = effective_speed
            effective_targets[vehicle.id] = effective_speed
        return effective_targets

    def record_idm_fallback(self):
        """Count a vehicle-step that lacked a fleet speed target."""
        self.idm_fallback_vehicle_steps += 1

    def apply_safety_filter(self, vehicle, recommended_acceleration):
        """Cap acceleration to preserve the minimum gap on the next step."""
        acceleration = max(
            -vehicle.maximum_deceleration,
            min(recommended_acceleration, vehicle.maximum_acceleration),
        )
        leader = vehicle.vehicle_front
        if leader is None or not leader.active:
            return acceleration

        lower = -vehicle.maximum_deceleration
        upper = vehicle.maximum_acceleration
        lower_margin = self._predicted_minimum_gap_margin(
            vehicle,
            lower,
            leader,
        )
        if lower_margin < 0.0:
            safe_upper_bound = lower
            self.minimum_gap_unachievable_steps += 1
        elif self._predicted_minimum_gap_margin(vehicle, upper, leader) >= 0.0:
            safe_upper_bound = upper
        else:
            low = lower
            high = upper
            for _ in range(40):
                midpoint = 0.5 * (low + high)
                if self._predicted_minimum_gap_margin(
                    vehicle,
                    midpoint,
                    leader,
                ) >= 0.0:
                    low = midpoint
                else:
                    high = midpoint
            safe_upper_bound = low

        safe_acceleration = min(acceleration, safe_upper_bound)
        if safe_acceleration < acceleration - 1e-9:
            self.safety_filter_interventions += 1
        return safe_acceleration

    @staticmethod
    def _predicted_minimum_gap_margin(vehicle, acceleration, leader):
        """Return predicted next-step gap above the required minimum spacing."""
        time_step = vehicle.time_step
        ego_acceleration = max(
            -vehicle.maximum_deceleration,
            min(acceleration, vehicle.maximum_acceleration),
        )
        leader_acceleration = max(
            -leader.maximum_deceleration,
            min(leader.acceleration, leader.maximum_acceleration),
        )
        ego_next_speed = max(
            0.0,
            vehicle.speed + ego_acceleration * time_step,
        )
        leader_next_speed = max(
            0.0,
            leader.speed + leader_acceleration * time_step,
        )
        ego_displacement = 0.5 * (
            vehicle.speed + ego_next_speed
        ) * time_step
        leader_displacement = 0.5 * (
            leader.speed + leader_next_speed
        ) * time_step
        ego_next_position = min(
            vehicle.road_length,
            vehicle.position + ego_displacement,
        )
        leader_next_position = min(
            leader.road_length,
            leader.position + leader_displacement,
        )
        next_gap = (
            leader_next_position
            - ego_next_position
            - vehicle.vehicle_length
        )
        return next_gap - vehicle.minimum_spacing

    def _request_target_speeds(self, vehicles):
        vehicle_by_id = {vehicle.id: vehicle for vehicle in vehicles}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a fleet-level speed planner for a single-lane "
                        "vehicle simulation. For each vehicle, choose a "
                        "desired target speed in meters per second to promote "
                        "smooth traffic flow and avoid unnecessary stop-and-go "
                        "driving. Use the full fleet state, leader, gap, "
                        "relative speed, and speed limit. A local IDM "
                        "controller recalculates acceleration every simulation "
                        "step and enforces car-following behavior; your target "
                        "is not a direct acceleration command. Choose a target "
                        "between zero and the vehicle's maximum_target_speed. "
                        "Return only a JSON object with a targets array; "
                        "each element must have exactly vehicle_id and "
                        "target_speed fields. Include every observed vehicle "
                        "exactly once."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "decision_step": self._current_step,
                            "vehicles": [
                                self._vehicle_observation(vehicle)
                                for vehicle in vehicles
                            ],
                        }
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": min(393216, max(128, 32 * len(vehicles))),
            "stream": False,
        }
        request = Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            response_body = response.read(1_048_577)
        if len(response_body) > 1_048_576:
            raise ValueError("DeepSeek response exceeded the 1 MiB limit")

        completion = json.loads(response_body)
        if not isinstance(completion, dict):
            raise ValueError("DeepSeek response must be a JSON object")
        self._record_token_usage(completion.get("usage"))
        content = completion["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek returned an empty message")
        result = json.loads(content)
        targets = result["targets"]
        if not isinstance(targets, list):
            raise ValueError("DeepSeek targets must be a list")

        expected_ids = set(vehicle_by_id)
        target_speeds = {}
        for target in targets:
            if not isinstance(target, dict):
                raise ValueError("Each DeepSeek target must be an object")
            vehicle_id = target["vehicle_id"]
            target_speed = target["target_speed"]
            if (
                isinstance(vehicle_id, bool)
                or not isinstance(vehicle_id, int)
                or vehicle_id not in expected_ids
                or vehicle_id in target_speeds
            ):
                raise ValueError("DeepSeek returned an invalid vehicle ID")
            if (
                isinstance(target_speed, bool)
                or not isinstance(target_speed, (int, float))
            ):
                raise ValueError("DeepSeek target speed must be numeric")
            try:
                target_speed = float(target_speed)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    "DeepSeek target speed must be finite"
                ) from exc
            if not math.isfinite(target_speed):
                raise ValueError("DeepSeek target speed must be finite")

            vehicle = vehicle_by_id[vehicle_id]
            bounded_target_speed = max(
                0.0,
                min(target_speed, vehicle.current_speed_limit),
            )
            if bounded_target_speed != target_speed:
                self.clamped_target_speeds += 1
            target_speeds[vehicle_id] = bounded_target_speed

        if set(target_speeds) != expected_ids:
            raise ValueError(
                "DeepSeek must return exactly one target for every active vehicle"
            )
        return target_speeds

    def _record_token_usage(self, usage):
        if not isinstance(usage, dict):
            self.usage_missing_batch_count += 1
            return

        values = (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            self.usage_missing_batch_count += 1
            return

        self.prompt_tokens += values[0]
        self.completion_tokens += values[1]
        self.total_tokens += values[2]
        self.usage_reported_batch_count += 1

    @staticmethod
    def _vehicle_observation(vehicle):
        leader = vehicle.vehicle_front
        if leader is None or not leader.active:
            leader_id = None
            gap = None
            relative_speed = None
            leader_speed = None
        else:
            leader_id = leader.id
            gap = leader.position - vehicle.position - vehicle.vehicle_length
            relative_speed = vehicle.speed - leader.speed
            leader_speed = leader.speed

        return {
            "vehicle_id": vehicle.id,
            "position": vehicle.position,
            "speed": vehicle.speed,
            "acceleration": vehicle.acceleration,
            "current_speed_limit": vehicle.current_speed_limit,
            "minimum_target_speed": 0.0,
            "maximum_target_speed": vehicle.current_speed_limit,
            "minimum_gap": vehicle.minimum_spacing,
            "leader_id": leader_id,
            "gap": gap,
            "relative_speed": relative_speed,
            "leader_speed": leader_speed,
        }

    def _record_batch_failure(self, reason):
        self._cached_target_speeds = {}
        self._effective_target_speeds = {}
        self.failed_batch_count += 1
        logger.warning(
            "DeepSeek fleet batch failed (%s); using IDM until the next batch.",
            reason,
        )
