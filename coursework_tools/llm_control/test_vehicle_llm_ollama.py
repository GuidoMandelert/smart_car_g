"""Unit tests for the isolated Ollama controller; no Ollama server is needed."""

import json
import unittest
from unittest.mock import patch

from coursework_tools.llm_control.vehicle_llm_ollama import OllamaLLMVehicle
from config import Config


class FakeHttpResponse:
    """Provide the small context-manager interface used by urllib.urlopen."""

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def read(self):
        return self.body


class OllamaVehicleTests(unittest.TestCase):
    def make_vehicle(self, **kwargs):
        config = Config(seed=1, experiment=1, time_max=10.0)
        return OllamaLLMVehicle(
            config,
            vehicle_id=1,
            model="test-model",
            **kwargs,
        )

    def test_observation_keeps_original_fields_and_si_values(self):
        vehicle = self.make_vehicle()
        observation = vehicle.get_observation()

        self.assertEqual(observation["vehicle_id"], 1)
        self.assertEqual(observation["speed"], 30.0)
        self.assertIsNone(observation["gap"])
        self.assertIsNone(observation["relative_speed"])

    def test_http_response_is_parsed_as_a_numeric_acceleration(self):
        body = json.dumps(
            {
                "message": {
                    "content": '{"acceleration_m_s2": 0.75}',
                }
            }
        ).encode("utf-8")
        vehicle = self.make_vehicle()

        with patch(
            "coursework_tools.llm_control.vehicle_llm_ollama.urlopen",
            return_value=FakeHttpResponse(body),
        ) as mocked_urlopen:
            acceleration = vehicle._request_acceleration()

        self.assertEqual(acceleration, 0.75)
        request = mocked_urlopen.call_args.args[0]
        request_data = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_data["model"], "test-model")
        self.assertEqual(request_data["format"], "json")
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")

    def test_non_numeric_model_output_is_rejected(self):
        body = json.dumps(
            {
                "message": {
                    "content": '{"acceleration_m_s2": "brake"}',
                }
            }
        ).encode("utf-8")
        vehicle = self.make_vehicle()

        with patch(
            "coursework_tools.llm_control.vehicle_llm_ollama.urlopen",
            return_value=FakeHttpResponse(body),
        ):
            with self.assertRaisesRegex(TypeError, "JSON number"):
                vehicle._request_acceleration()

    def test_action_is_clamped_and_reused_until_next_decision(self):
        vehicle = self.make_vehicle(decision_interval_steps=3)

        with patch.object(
            vehicle,
            "_request_acceleration",
            side_effect=[99.0, -99.0],
        ) as request:
            vehicle.update_acceleration()
            self.assertEqual(vehicle.acceleration, 1.5)

            vehicle.update_acceleration()
            vehicle.update_acceleration()
            self.assertEqual(request.call_count, 1)
            self.assertEqual(vehicle.acceleration, 1.5)

            vehicle.update_acceleration()
            self.assertEqual(request.call_count, 2)
            self.assertEqual(vehicle.acceleration, -8.0)

    def test_failed_request_uses_idm_and_counts_fallback(self):
        vehicle = self.make_vehicle(decision_interval_steps=2)

        with patch.object(
            vehicle,
            "_request_acceleration",
            side_effect=ValueError("test response failure"),
        ):
            with self.assertLogs(
                "coursework_tools.llm_control.vehicle_llm_ollama",
                level="WARNING",
            ):
                vehicle.update_acceleration()

        self.assertEqual(vehicle.idm_fallback_count, 1)
        self.assertTrue(vehicle._using_idm_fallback)
        self.assertAlmostEqual(vehicle.acceleration, 0.0)

    def test_invalid_decision_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.make_vehicle(decision_interval_steps=0)

    def test_invalid_timeout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive finite"):
            self.make_vehicle(timeout_seconds=0)

    def test_missing_model_name_is_rejected(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OLLAMA_MODEL"):
                OllamaLLMVehicle(Config(), vehicle_id=1)


if __name__ == "__main__":
    unittest.main()
