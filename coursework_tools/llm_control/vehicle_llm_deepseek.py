"""DeepSeek-backed longitudinal controller for the car-following simulation."""

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
            "or in your environment before selecting the DeepSeek controller."
        )
    return api_key


class DeepSeekLLMVehicle(IDMVehicle):
    """Use cached DeepSeek acceleration recommendations with IDM fallback."""

    def __init__(
        self,
        config,
        vehicle_id,
        vehicle_front=None,
        model=DEFAULT_MODEL,
        timeout_seconds=60.0,
        decision_interval_steps=50,
        api_key=None,
    ):
        super().__init__(config, vehicle_id, vehicle_front)
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

        self.llm_request_count = 0
        self.idm_fallback_count = 0
        self.safety_filter_interventions = 0
        self.safety_margin_unachievable_steps = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.usage_reported_request_count = 0
        self.usage_missing_request_count = 0
        self._steps_until_request = 0
        self._cached_acceleration = None

    def update_acceleration(self):
        if self._steps_until_request <= 0:
            self.llm_request_count += 1
            self._steps_until_request = self.decision_interval_steps - 1
            try:
                self._cached_acceleration = self._request_acceleration()
            except HTTPError as exc:
                self._use_idm_after_failure(
                    f"HTTP {exc.code}"
                )
            except (
                URLError,
                TimeoutError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                UnicodeError,
            ) as exc:
                self._use_idm_after_failure(
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            self._steps_until_request -= 1

        if self._cached_acceleration is None:
            super().update_acceleration()
            recommended_acceleration = self.acceleration
        else:
            recommended_acceleration = self._cached_acceleration

        self.acceleration = self._apply_safety_filter(recommended_acceleration)

    def get_observation(self):
        leader = self.vehicle_front
        if leader is None or not leader.active:
            gap = None
            relative_speed = None
            front_speed = None
        else:
            gap = leader.position - self.position - self.vehicle_length
            relative_speed = self.speed - leader.speed
            front_speed = leader.speed

        return {
            "vehicle_id": self.id,
            "position": self.position,
            "speed": self.speed,
            "acceleration": self.acceleration,
            "current_speed_limit": self.current_speed_limit,
            "gap": gap,
            "relative_speed": relative_speed,
            "front_speed": front_speed,
            "minimum_acceleration": -self.maximum_deceleration,
            "maximum_acceleration": self.maximum_acceleration,
            "minimum_gap": self.minimum_spacing,
        }

    def _apply_safety_filter(self, recommended_acceleration):
        """Cap actions that would violate the predicted stopping-gap condition."""
        acceleration = max(
            -self.maximum_deceleration,
            min(recommended_acceleration, self.maximum_acceleration),
        )
        leader = self.vehicle_front
        if leader is None or not leader.active:
            return acceleration

        lower = -self.maximum_deceleration
        upper = self.maximum_acceleration
        lower_margin = self._predicted_safety_margin(lower, leader)
        if lower_margin < 0.0:
            safe_upper_bound = lower
            self.safety_margin_unachievable_steps += 1
        elif self._predicted_safety_margin(upper, leader) >= 0.0:
            safe_upper_bound = upper
        else:
            low = lower
            high = upper
            for _ in range(40):
                midpoint = 0.5 * (low + high)
                if self._predicted_safety_margin(midpoint, leader) >= 0.0:
                    low = midpoint
                else:
                    high = midpoint
            safe_upper_bound = low

        safe_acceleration = min(acceleration, safe_upper_bound)
        if safe_acceleration < acceleration - 1e-9:
            self.safety_filter_interventions += 1
        return safe_acceleration

    def _predicted_safety_margin(self, acceleration, leader):
        """Return next-step gap minus a conservative stopping-distance gap."""
        time_step = self.time_step
        ego_acceleration = max(
            -self.maximum_deceleration,
            min(acceleration, self.maximum_acceleration),
        )
        leader_acceleration = max(
            -leader.maximum_deceleration,
            min(leader.acceleration, leader.maximum_acceleration),
        )
        ego_next_speed = max(
            0.0,
            self.speed + ego_acceleration * time_step,
        )
        leader_next_speed = max(
            0.0,
            leader.speed + leader_acceleration * time_step,
        )
        ego_displacement = 0.5 * (
            self.speed + ego_next_speed
        ) * time_step
        leader_displacement = 0.5 * (
            leader.speed + leader_next_speed
        ) * time_step
        ego_next_position = min(
            self.road_length,
            self.position + ego_displacement,
        )
        leader_next_position = min(
            leader.road_length,
            leader.position + leader_displacement,
        )
        next_gap = (
            leader_next_position
            - ego_next_position
            - self.vehicle_length
        )
        reaction_distance = ego_next_speed * time_step
        ego_stopping_distance = (
            ego_next_speed**2 / (2.0 * self.maximum_deceleration)
        )
        leader_stopping_distance = (
            leader_next_speed**2 / (2.0 * leader.maximum_deceleration)
        )
        required_gap = self.minimum_spacing + max(
            0.0,
            reaction_distance
            + ego_stopping_distance
            - leader_stopping_distance,
        )
        return next_gap - required_gap

    def _request_acceleration(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a longitudinal controller in a single-lane "
                        "vehicle simulation. Choose a safe acceleration in "
                        "meters per second squared using the observation. "
                        "Avoid collisions and remain within the supplied "
                        "acceleration bounds. Reply only with a JSON object "
                        'of the form {"acceleration": 0.0}.'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(self.get_observation()),
                },
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 64,
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
        acceleration = result["acceleration"]
        if (
            isinstance(acceleration, bool)
            or not isinstance(acceleration, (int, float))
        ):
            raise ValueError("DeepSeek acceleration must be a finite number")
        try:
            acceleration = float(acceleration)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "DeepSeek acceleration must be a finite number"
            ) from exc
        if not math.isfinite(acceleration):
            raise ValueError("DeepSeek acceleration must be a finite number")

        return max(
            -self.maximum_deceleration,
            min(acceleration, self.maximum_acceleration),
        )

    def _record_token_usage(self, usage):
        if not isinstance(usage, dict):
            self.usage_missing_request_count += 1
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
            self.usage_missing_request_count += 1
            return

        self.prompt_tokens += values[0]
        self.completion_tokens += values[1]
        self.total_tokens += values[2]
        self.usage_reported_request_count += 1

    def _use_idm_after_failure(self, reason):
        self._cached_acceleration = None
        self.idm_fallback_count += 1
        logger.warning(
            "DeepSeek request failed for vehicle %s (%s); using IDM until "
            "the next request.",
            self.id,
            reason,
        )
