"""Unit and invariant tests for the corrected simulation."""

import unittest

from config import Config
from simulator import Simulator
from vehicle_idm import IDMVehicle


class SimulationTests(unittest.TestCase):
    def test_invalid_experiment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "experiment"):
            Config(experiment=5)

    def test_free_vehicle_at_desired_speed_has_zero_acceleration(self) -> None:
        config = Config(time_max=1.0)
        vehicle = IDMVehicle(config, vehicle_id=1)
        vehicle.update_acceleration()
        self.assertAlmostEqual(vehicle.acceleration, 0.0)

    def test_close_follower_decelerates(self) -> None:
        config = Config(time_max=1.0)
        leader = IDMVehicle(config, vehicle_id=1)
        leader.position = 40.0
        follower = IDMVehicle(config, vehicle_id=2, vehicle_front=leader)
        follower.update_acceleration()
        self.assertLess(follower.acceleration, 0.0)

    def test_vehicle_already_inside_when_bottleneck_starts_is_exempt(self) -> None:
        config = Config()
        vehicle = IDMVehicle(config, vehicle_id=1)
        vehicle.influenced_by_bottleneck = True
        vehicle.position = 0.5 * (
            config.bottleneck_x_start + config.bottleneck_x_end
        )

        vehicle.check_road(99.0)
        vehicle.check_road(100.0)

        self.assertEqual(vehicle.bottleneck_entry_time, 99.0)
        self.assertEqual(vehicle.current_speed_limit, config.speed_limit)

    def test_vehicle_entering_after_bottleneck_starts_slows_down(self) -> None:
        config = Config()
        vehicle = IDMVehicle(config, vehicle_id=1)
        vehicle.influenced_by_bottleneck = True
        vehicle.position = config.bottleneck_x_start - 10.0
        vehicle.check_road(99.0)

        vehicle.position = config.bottleneck_x_start
        vehicle.check_road(100.0)

        self.assertEqual(vehicle.bottleneck_entry_time, 100.0)
        self.assertEqual(
            vehicle.current_speed_limit,
            config.bottleneck_speed_limit,
        )

    def test_motion_update_is_consistent(self) -> None:
        config = Config(time_max=1.0)
        vehicle = IDMVehicle(config, vehicle_id=1)
        vehicle.position = 0.0
        vehicle.speed = 10.0
        vehicle.acceleration = 2.0
        vehicle.update_motion()
        self.assertAlmostEqual(vehicle.speed, 10.15)
        self.assertAlmostEqual(vehicle.position, 1.0075)
        self.assertAlmostEqual(vehicle.acceleration, 1.5)

    def test_simulation_is_reproducible_and_collision_free(self) -> None:
        first = self._run_signature()
        second = self._run_signature()
        self.assertEqual(first, second)

    def test_selected_vehicle_factory_is_opt_in_and_other_vehicles_use_idm(self):
        class FakeLLMVehicle(IDMVehicle):
            pass

        config = Config(seed=1, experiment=1, time_max=5.0)
        simulator = Simulator(
            config,
            llm_vehicle_factory=FakeLLMVehicle,
            llm_vehicle_id=2,
        )

        simulator._generate_vehicle(current_time=0.0)
        simulator.vehicles[0].position = 50.0
        simulator._generate_vehicle(current_time=2.0)

        self.assertIsInstance(simulator.vehicles[0], IDMVehicle)
        self.assertNotIsInstance(simulator.vehicles[0], FakeLLMVehicle)
        self.assertIsInstance(simulator.vehicles[1], FakeLLMVehicle)
        self.assertEqual(simulator.llm_vehicles, [simulator.vehicles[1]])
        self.assertIs(simulator.vehicles[1].vehicle_front, simulator.vehicles[0])

    def test_default_simulator_still_creates_only_idm_vehicles(self):
        config = Config(seed=1, experiment=1, time_max=5.0)
        simulator = Simulator(config)
        simulator._generate_vehicle(current_time=0.0)

        self.assertIs(type(simulator.vehicles[0]), IDMVehicle)
        self.assertEqual(simulator.llm_vehicles, [])

    def _run_signature(self):
        config = Config(seed=7, experiment=4, time_max=250.0)
        simulator = Simulator(config)
        simulator.run(verbose=False)

        for vehicle in simulator.vehicles:
            self.assertTrue(all(record["speed"] >= 0.0 for record in vehicle.history))
            positions = [record["position"] for record in vehicle.history]
            self.assertEqual(positions, sorted(positions))

            leader = vehicle.vehicle_front
            if leader is None:
                continue
            leader_positions = {
                record["t"]: record["position"] for record in leader.history
            }
            for record in vehicle.history:
                if record["t"] in leader_positions:
                    gap = (
                        leader_positions[record["t"]]
                        - record["position"]
                        - vehicle.vehicle_length
                    )
                    self.assertGreaterEqual(gap, -1e-9)

        return [
            (
                vehicle.id,
                round(vehicle.position, 8),
                round(vehicle.speed, 8),
                len(vehicle.history),
            )
            for vehicle in simulator.vehicles
        ]


if __name__ == "__main__":
    unittest.main()
