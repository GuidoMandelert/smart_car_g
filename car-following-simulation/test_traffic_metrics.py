"""Tests for post-run metrics and machine-readable run summaries."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from config import Config
from main import main as run_main
from simulator import Simulator
from traffic_metrics import collect_traffic_metrics
from vehicle_idm import IDMVehicle


class TrafficMetricsTests(unittest.TestCase):
    def test_completed_trip_metrics_and_throughput(self):
        config = Config(time_max=0.2)
        config.road_length = 4.0
        simulator = Simulator(config)
        simulator.run(verbose=False)

        metrics = collect_traffic_metrics(simulator, config)

        self.assertEqual(metrics["generated_vehicle_count"], 1)
        self.assertEqual(metrics["completed_vehicle_count"], 1)
        self.assertAlmostEqual(
            metrics["throughput_vehicles_per_hour"],
            18000.0,
        )
        self.assertAlmostEqual(
            metrics["mean_completed_travel_time_s"],
            0.2,
        )
        self.assertEqual(metrics["stopped_time_fraction"], 0.0)

    def test_slow_vehicle_count_and_span_are_time_aligned(self):
        config = Config(time_max=1.0)
        first = IDMVehicle(config, vehicle_id=1)
        second = IDMVehicle(config, vehicle_id=2)
        first.history = [
            {"t": 0.0, "position": 20.0, "speed": 30.0},
            {"t": 0.1, "position": 20.0, "speed": 0.0},
        ]
        second.history = [
            {"t": 0.0, "position": 10.0, "speed": 30.0},
            {"t": 0.1, "position": 10.0, "speed": 4.0},
        ]
        simulator = Simulator(config)
        simulator.vehicles = [first, second]

        metrics = collect_traffic_metrics(simulator, config)

        self.assertEqual(metrics["maximum_slow_vehicle_count"], 2)
        self.assertAlmostEqual(
            metrics["maximum_slow_vehicle_span_m"],
            15.0,
        )
        self.assertEqual(metrics["stopped_time_fraction"], 0.5)
        self.assertEqual(metrics["slow_speed_threshold_mps"], 5.0)

    def test_cli_writes_seeded_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "run.json"
            arguments = [
                "main.py",
                "--controller",
                "idm",
                "--experiment",
                "1",
                "--time-max",
                "0.1",
                "--seed",
                "23",
                "--metrics-output",
                str(metrics_path),
                "--no-show",
            ]
            with patch.object(sys, "argv", arguments):
                with patch("main.plot_time_space_diagram"):
                    run_main()

            summary = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["run"]["seed"], 23)
        self.assertFalse(summary["run"]["safe_entry_gap"])
        self.assertEqual(summary["run"]["controller"], "idm")
        self.assertGreaterEqual(
            summary["run"]["simulation_wall_time_s"],
            0.0,
        )
        self.assertEqual(summary["traffic"]["generated_vehicle_count"], 1)
        self.assertEqual(summary["safety"]["observed_overlap_count"], 0)
        self.assertIsNone(summary["controller_metrics"])


if __name__ == "__main__":
    unittest.main()
