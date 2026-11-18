"""Offline tests for the DeepSeek vehicle controller."""

import io
import json
import unittest
from unittest.mock import patch
from urllib.request import Request

from coursework_tools.llm_control.vehicle_llm_deepseek import (
    API_URL,
    DeepSeekLLMVehicle,
)
from config import Config


def fake_response(acceleration):
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"acceleration": acceleration})
                }
            }
        ]
    }
    return io.BytesIO(json.dumps(body).encode("utf-8"))


class DeepSeekVehicleTests(unittest.TestCase):
    def make_vehicle(self, **kwargs):
        config = Config(time_max=1.0)
        return DeepSeekLLMVehicle(
            config,
            vehicle_id=2,
            api_key="unit-test-key",
            **kwargs,
        )

    @patch("coursework_tools.llm_control.vehicle_llm_deepseek.urlopen")
    def test_request_uses_bearer_auth_and_json_contract(self, urlopen_mock):
        urlopen_mock.return_value = fake_response(0.5)
        vehicle = self.make_vehicle()

        self.assertAlmostEqual(vehicle._request_acceleration(), 0.5)

        request = urlopen_mock.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertEqual(request.full_url, API_URL)
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    @patch("coursework_tools.llm_control.vehicle_llm_deepseek.urlopen")
    def test_recommendation_is_clamped_and_cached(self, urlopen_mock):
        urlopen_mock.side_effect = lambda *args, **kwargs: fake_response(99)
        vehicle = self.make_vehicle(decision_interval_steps=3)

        vehicle.update_acceleration()
        self.assertEqual(vehicle.acceleration, vehicle.maximum_acceleration)
        vehicle.update_acceleration()
        vehicle.update_acceleration()

        self.assertEqual(urlopen_mock.call_count, 1)
        vehicle.update_acceleration()
        self.assertEqual(urlopen_mock.call_count, 2)

    @patch("coursework_tools.llm_control.vehicle_llm_deepseek.urlopen")
    def test_invalid_response_uses_idm_and_logs_fallback(self, urlopen_mock):
        urlopen_mock.return_value = io.BytesIO(
            b'{"choices":[{"message":{"content":"not json"}}]}'
        )
        vehicle = self.make_vehicle()

        with self.assertLogs(
            "coursework_tools.llm_control.vehicle_llm_deepseek",
            level="WARNING",
        ):
            vehicle.update_acceleration()

        self.assertEqual(vehicle.idm_fallback_count, 1)
        self.assertEqual(vehicle.acceleration, 0.0)

    def test_invalid_decision_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "decision_interval_steps"):
            self.make_vehicle(decision_interval_steps=0)


if __name__ == "__main__":
    unittest.main()
