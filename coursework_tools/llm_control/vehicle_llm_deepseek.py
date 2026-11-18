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
        else:
            self.acceleration = self._cached_acceleration

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
        }

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

    def _use_idm_after_failure(self, reason):
        self._cached_acceleration = None
        self.idm_fallback_count += 1
        logger.warning(
            "DeepSeek request failed for vehicle %s (%s); using IDM until "
            "the next request.",
            self.id,
            reason,
        )
