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
from simulator import Simulator
from vehicle_idm import IDMVehicle


def fake_response(acceleration, include_usage=True):
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"acceleration": acceleration})
                }
            }
        ]
    }
    if include_usage:
        body["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
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
    def test_usage_metadata_is_accumulated_without_changing_action(
        self,
        urlopen_mock,
    ):
        urlopen_mock.return_value = fake_response(0.5)
        vehicle = self.make_vehicle()

        self.assertAlmostEqual(vehicle._request_acceleration(), 0.5)

        self.assertEqual(vehicle.prompt_tokens, 10)
        self.assertEqual(vehicle.completion_tokens, 2)
        self.assertEqual(vehicle.total_tokens, 12)
        self.assertEqual(vehicle.usage_reported_request_count, 1)

    @patch("coursework_tools.llm_control.vehicle_llm_deepseek.urlopen")
    def test_missing_usage_metadata_is_counted(self, urlopen_mock):
        urlopen_mock.return_value = fake_response(0.5, include_usage=False)
        vehicle = self.make_vehicle()

        vehicle._request_acceleration()

        self.assertEqual(vehicle.usage_missing_request_count, 1)

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

    @patch("coursework_tools.llm_control.vehicle_llm_deepseek.urlopen")
    def test_safety_filter_brakes_for_close_follower_behind_braking_leader(
        self,
        urlopen_mock,
    ):
        urlopen_mock.return_value = fake_response(1.5)
        config = Config(time_max=1.0)
        leader = IDMVehicle(config, vehicle_id=1)
        leader.position = 20.0
        leader.speed = 15.0
        leader.acceleration = -8.0
        vehicle = DeepSeekLLMVehicle(
            config,
            vehicle_id=2,
            vehicle_front=leader,
            api_key="unit-test-key",
        )
        vehicle.speed = 20.0
        vehicle.position = 0.0

        vehicle.update_acceleration()

        self.assertLess(vehicle.acceleration, 1.5)
        self.assertEqual(vehicle.safety_filter_interventions, 1)
        self.assertGreaterEqual(
            vehicle._predicted_safety_margin(vehicle.acceleration, leader),
            -1e-8,
        )

    def test_safety_filter_leaves_free_road_recommendation_unchanged(self):
        vehicle = self.make_vehicle()

        self.assertEqual(vehicle._apply_safety_filter(1.0), 1.0)
        self.assertEqual(vehicle.safety_filter_interventions, 0)

    def test_aggressive_llm_fleets_use_safety_filter_without_overlaps(self):
        class MaximumAccelerationVehicle(DeepSeekLLMVehicle):
            def _request_acceleration(self):
                return self.maximum_acceleration

        for controlled_count in (10, 50):
            with self.subTest(controlled_count=controlled_count):
                config = Config(seed=1, experiment=1, time_max=400.0)
                simulator = Simulator(
                    config,
                    llm_vehicle_factory=MaximumAccelerationVehicle,
                    llm_vehicle_id=2,
                    llm_vehicle_count=controlled_count,
                    safe_entry_gap=True,
                    llm_vehicle_kwargs={
                        "api_key": "offline-test-key",
                        "decision_interval_steps": 200,
                    },
                )

                simulator.run(verbose=False)

                self.assertEqual(
                    len(simulator.llm_vehicles),
                    controlled_count,
                )
                self.assertGreater(
                    sum(
                        v.safety_filter_interventions
                        for v in simulator.llm_vehicles
                    ),
                    0,
                )
                self.assertEqual(simulator.collision_events, [])
                self.assertGreaterEqual(
                    simulator.minimum_gap_observed,
                    -1e-9,
                )


if __name__ == "__main__":
    unittest.main()
