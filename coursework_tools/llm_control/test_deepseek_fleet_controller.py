"""Offline tests for the centralized DeepSeek fleet controller."""

import io
import json
import unittest
from unittest.mock import patch
from urllib.request import Request

from coursework_tools.llm_control.deepseek_fleet_controller import (
    API_URL,
    DeepSeekFleetController,
)
from config import Config
from simulator import Simulator
from vehicle_idm import IDMVehicle


def fake_response(actions):
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"targets": actions})
                }
            }
        ],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 8,
            "total_tokens": 48,
        },
    }
    return io.BytesIO(json.dumps(body).encode("utf-8"))


class DeepSeekFleetControllerTests(unittest.TestCase):
    def make_controller(self, **kwargs):
        return DeepSeekFleetController(
            api_key="unit-test-key",
            **kwargs,
        )

    def make_vehicles(self, count):
        config = Config(time_max=1.0)
        vehicles = []
        leader = None
        for vehicle_id in range(1, count + 1):
            vehicle = IDMVehicle(config, vehicle_id, leader)
            if leader is not None:
                leader.position = vehicle_id * 50.0
            vehicles.append(vehicle)
            leader = vehicle
        return vehicles

    @patch("coursework_tools.llm_control.deepseek_fleet_controller.urlopen")
    def test_batch_request_contains_all_vehicle_states_and_returns_targets(
        self,
        urlopen_mock,
    ):
        vehicles = self.make_vehicles(2)
        vehicles[0].speed = 20.0
        vehicles[1].speed = 15.0
        urlopen_mock.side_effect = lambda *args, **kwargs: fake_response(
            [
                {"vehicle_id": 1, "target_speed": 20.0},
                {"vehicle_id": 2, "target_speed": 15.0},
            ]
        )
        controller = self.make_controller(decision_interval_steps=3)

        actions = controller.actions_for_step(vehicles, step=0)

        self.assertEqual(actions, {1: 20.0, 2: 15.0})
        request = urlopen_mock.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertEqual(request.full_url, API_URL)
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer unit-test-key",
        )
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-flash")
        observations = json.loads(payload["messages"][1]["content"])["vehicles"]
        self.assertEqual([item["vehicle_id"] for item in observations], [1, 2])
        self.assertEqual(observations[0]["maximum_target_speed"], 30.0)
        self.assertEqual(controller.total_tokens, 48)
        self.assertEqual(controller.usage_reported_batch_count, 1)

    @patch("coursework_tools.llm_control.deepseek_fleet_controller.urlopen")
    def test_batch_response_must_cover_every_active_vehicle(self, urlopen_mock):
        vehicles = self.make_vehicles(2)
        urlopen_mock.return_value = fake_response(
            [{"vehicle_id": 1, "target_speed": 10.0}]
        )
        controller = self.make_controller()

        with self.assertLogs(
            "coursework_tools.llm_control.deepseek_fleet_controller",
            level="WARNING",
        ):
            actions = controller.actions_for_step(vehicles, step=0)

        self.assertEqual(actions, {})
        self.assertEqual(controller.failed_batch_count, 1)
        self.assertEqual(controller.successful_batch_count, 0)

    def test_missing_usage_metadata_is_reported_as_unavailable(self):
        controller = self.make_controller()

        controller._record_token_usage(None)

        self.assertEqual(controller.usage_missing_batch_count, 1)
        self.assertEqual(controller.total_tokens, 0)

    @patch("coursework_tools.llm_control.deepseek_fleet_controller.urlopen")
    def test_cached_batch_is_reused_between_decision_steps(self, urlopen_mock):
        vehicles = self.make_vehicles(1)
        urlopen_mock.side_effect = lambda *args, **kwargs: fake_response(
            [{"vehicle_id": 1, "target_speed": 25.0}]
        )
        controller = self.make_controller(decision_interval_steps=3)

        self.assertEqual(
            controller.actions_for_step(vehicles, 0),
            {1: 29.8},
        )
        self.assertEqual(
            controller.actions_for_step(vehicles, 1),
            {1: 29.6},
        )
        self.assertEqual(urlopen_mock.call_count, 1)
        self.assertEqual(controller.request_count, 1)

    def test_safety_filter_brakes_for_close_follower(self):
        config = Config(time_max=1.0)
        leader = IDMVehicle(config, vehicle_id=1)
        leader.position = 7.005
        leader.speed = 20.0
        follower = IDMVehicle(config, vehicle_id=2, vehicle_front=leader)
        follower.position = 0.0
        follower.speed = 20.0
        controller = self.make_controller()

        acceleration = controller.apply_safety_filter(follower, 1.5)

        self.assertLess(acceleration, 1.5)
        self.assertEqual(controller.safety_filter_interventions, 1)
        self.assertGreaterEqual(
            controller._predicted_minimum_gap_margin(
                follower,
                acceleration,
                leader,
            ),
            -1e-8,
        )

    def test_fleet_target_speeds_use_local_idm_and_stay_overlap_free(self):
        class AlternatingTargetSpeedController(DeepSeekFleetController):
            def _request_target_speeds(self, vehicles):
                desired_speed = (
                    30.0
                    if (self._current_step // self.decision_interval_steps) % 2
                    == 0
                    else 0.0
                )
                return {
                    vehicle.id: min(
                        desired_speed,
                        vehicle.current_speed_limit,
                    )
                    for vehicle in vehicles
                }

        config = Config(seed=1, experiment=1, time_max=180.0)
        config.road_length = 1000.0
        config.bottleneck_x_start = 600.0
        config.bottleneck_x_end = 800.0
        config.bottleneck_t_start = 10.0
        config.bottleneck_t_end = 50.0
        config.bottleneck_speed_limit = 2.0
        controller = AlternatingTargetSpeedController(
            api_key="offline-test-key",
            decision_interval_steps=200,
        )
        simulator = Simulator(
            config,
            fleet_controller=controller,
            safe_entry_gap=True,
        )

        simulator.run(verbose=False)

        self.assertGreater(len(simulator.vehicles), 40)
        self.assertEqual(simulator.collision_events, [])
        self.assertGreaterEqual(simulator.minimum_gap_observed, -1e-9)
        self.assertGreater(controller.request_count, 0)


if __name__ == "__main__":
    unittest.main()
