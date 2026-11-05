"""Experimental Ollama-backed vehicle controller, isolated from the simulator.

This module intentionally lives outside ``car-following-simulation``. Nothing
in the existing simulator imports or activates this class. It is an experiment
for learning about an LLM control interface, not a safety-certified controller.
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


# The teaching project uses local imports (for example, ``from vehicle import
# Vehicle``). Add its directory so this isolated module can reuse the existing
# IDM class without copying or editing the simulator's base files.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_DIR = PROJECT_ROOT / "car-following-simulation"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

from vehicle_idm import IDMVehicle


LOGGER = logging.getLogger(__name__)


class OllamaLLMVehicle(IDMVehicle):
    """Ask a local Ollama model for acceleration, with IDM as a fallback.

    The model is called only once per ``decision_interval_steps`` simulation
    updates. Its most recent valid recommendation is reused between calls.
    If a request or response fails validation, IDM is recalculated at each
    simulation update until the next LLM decision is attempted.

    This class is not wired into ``Simulator``. Calling the model synchronously
    can pause a simulation while the model responds; the interval reduces call
    frequency but does not make inference real-time or guarantee safety.
    """

    def __init__(
        self,
        config,
        vehicle_id,
        vehicle_front=None,
        *,
        model=None,
        host=None,
        timeout_seconds=120.0,
        decision_interval_steps=10,
    ):
        # Reuse IDMVehicle initialization so IDM fallback has all the exact
        # parameters expected by the existing implementation.
        super().__init__(config, vehicle_id, vehicle_front)

        # Require the model to be named explicitly or configured in the
        # environment. This avoids silently assuming which local model to use.
        self.model = model or os.environ.get("OLLAMA_MODEL")
        if not self.model:
            raise ValueError(
                "Set OLLAMA_MODEL or pass model='your-model-name' when "
                "constructing OllamaLLMVehicle."
            )

        # Ollama's local HTTP server defaults to port 11434. Allow a custom
        # server address without adding a package dependency.
        self.host = (host or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )).rstrip("/")

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if (
            isinstance(decision_interval_steps, bool)
            or not isinstance(decision_interval_steps, int)
            or decision_interval_steps < 1
        ):
            raise ValueError("decision_interval_steps must be a positive integer")

        self.timeout_seconds = timeout_seconds
        self.decision_interval_steps = decision_interval_steps

        # Zero makes the controller request its first recommendation on its
        # first update. Subsequent requests follow the configured interval.
        self._steps_until_decision = 0
        self._cached_acceleration = 0.0
        self._using_idm_fallback = False

        # These counters let a later experiment report how often the LLM was
        # called and how often the controller had to fall back to IDM.
        self.llm_request_count = 0
        self.idm_fallback_count = 0

    def get_observation(self):
        """Return the same named traffic fields as the original placeholder.

        Speeds are in m/s, positions and gap are in metres, and acceleration is
        in m/s^2. A missing active leader is represented by ``None`` for the
        leader-dependent fields.
        """
        leader = self.vehicle_front
        if leader is None or not leader.active:
            gap = None
            relative_speed = None
            front_speed = None
        else:
            gap = leader.position - self.position - self.vehicle_length
            # Positive relative speed means this vehicle is closing on a
            # slower leader; negative means the leader is pulling away.
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
        }

    def update_acceleration(self):
        """Update acceleration using Ollama or the established IDM fallback."""
        if self._steps_until_decision > 0:
            # Recalculate IDM on each intermediate step after a failed LLM
            # request; otherwise hold the latest bounded LLM recommendation.
            if self._using_idm_fallback:
                super().update_acceleration()
            else:
                self.acceleration = self._cached_acceleration
            self._steps_until_decision -= 1
            return

        self.llm_request_count += 1
        try:
            recommendation = self._request_acceleration()
        except (
            OSError,
            TimeoutError,
            URLError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            # The failure is visible in the log and counter. The vehicle then
            # gets a fresh IDM acceleration instead of a success-shaped guess.
            self.idm_fallback_count += 1
            self._using_idm_fallback = True
            LOGGER.warning(
                "Ollama recommendation failed for vehicle %s; using IDM "
                "until the next decision (%s).",
                self.id,
                error,
            )
            super().update_acceleration()
        else:
            # The simulator also bounds acceleration during motion. Bound the
            # LLM value here so the recommendation is constrained immediately.
            bounded = max(
                -self.maximum_deceleration,
                min(recommendation, self.maximum_acceleration),
            )
            if bounded != recommendation:
                LOGGER.warning(
                    "Clamped Ollama acceleration for vehicle %s from %.3f "
                    "to %.3f m/s^2.",
                    self.id,
                    recommendation,
                    bounded,
                )
            self._cached_acceleration = bounded
            self.acceleration = bounded
            self._using_idm_fallback = False

        # An interval of 10 means: decide now, reuse/fallback for the next
        # nine updates, then ask the model again on update number eleven.
        self._steps_until_decision = self.decision_interval_steps - 1

    def _request_acceleration(self):
        """Send one structured chat request and return a validated number.

        Ollama's ``/api/chat`` endpoint accepts JSON mode. The response is
        checked independently because JSON mode controls formatting, not
        correctness, physical plausibility, or safety.
        """
        observation = self.get_observation()
        user_content = (
            "Choose longitudinal acceleration for the observed vehicle. "
            "Return one JSON object with exactly this numeric field: "
            '{"acceleration_m_s2": number}. Positive values accelerate and '
            "negative values brake. The simulator will enforce physical "
            "bounds. Observation (SI units): "
            f"{json.dumps(observation, allow_nan=False)}"
        )
        request_body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an experimental car-following policy in a "
                        "toy, single-lane simulator. Consider the speed limit, "
                        "speed, gap, and relative speed. Return only valid "
                        "JSON with one finite numeric acceleration_m_s2 field. "
                        "Do not include Markdown or additional text."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        }
        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # A finite timeout prevents the local simulation from waiting forever.
        # The call is synchronous, so it can still pause the simulation for up
        # to this duration; this is a known limitation of the experiment.
        with urlopen(request, timeout=self.timeout_seconds) as response:
            response_body = json.loads(response.read().decode("utf-8"))

        message = response_body["message"]
        model_text = message["content"]
        if not isinstance(model_text, str):
            raise TypeError("Ollama message content must be a string")

        model_output = json.loads(model_text)
        if not isinstance(model_output, dict):
            raise TypeError("Ollama response content must be a JSON object")
        if "acceleration_m_s2" not in model_output:
            raise KeyError("Ollama response is missing acceleration_m_s2")

        acceleration = model_output["acceleration_m_s2"]
        # Do not accept booleans or numeric strings; JSON must contain a
        # genuine number so output mistakes cannot be silently coerced.
        if isinstance(acceleration, bool) or not isinstance(
            acceleration, (int, float)
        ):
            raise TypeError("acceleration_m_s2 must be a JSON number")
        acceleration = float(acceleration)
        if not math.isfinite(acceleration):
            raise ValueError("acceleration_m_s2 must be finite")
        return acceleration
