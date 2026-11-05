"""Run one car-following experiment."""

import argparse
import sys
from pathlib import Path

from config import Config
from plotting import plot_time_space_diagram
from simulator import Simulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run a car-following simulation.")
    parser.add_argument(
        "--controller",
        choices=("idm", "ollama"),
        default="idm",
        help="Controller for one selected vehicle; all other vehicles use IDM.",
    )
    parser.add_argument(
        "--llm-vehicle-id",
        type=int,
        default=2,
        help="Vehicle ID controlled by Ollama (default: 2, which has a leader).",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3:4b",
        help="Local Ollama model name (default: qwen3:4b).",
    )
    parser.add_argument(
        "--llm-decision-interval-steps",
        type=int,
        default=50,
        help="Simulation steps between Ollama requests (default: 50).",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum wait for a local Ollama response (default: 300 seconds).",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=Config.DEFAULT_TIME_MAX,
        help="Simulation duration in seconds.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    config = Config(
        experiment=args.experiment,
        time_max=args.time_max,
    )
    llm_vehicle_factory = None
    llm_vehicle_kwargs = None
    if args.controller == "ollama":
        if args.llm_vehicle_id < 1:
            parser.error("--llm-vehicle-id must be a positive integer")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from coursework_tools.llm_control.vehicle_llm_ollama import (
            OllamaLLMVehicle,
        )

        llm_vehicle_factory = OllamaLLMVehicle
        llm_vehicle_kwargs = {
            "model": args.llm_model,
            "timeout_seconds": args.llm_timeout_seconds,
            "decision_interval_steps": args.llm_decision_interval_steps,
        }

    simulator = Simulator(
        config,
        llm_vehicle_factory=llm_vehicle_factory,
        llm_vehicle_id=args.llm_vehicle_id,
        llm_vehicle_kwargs=llm_vehicle_kwargs,
    )
    simulator.run()
    if args.controller == "ollama":
        if simulator.llm_vehicles:
            for vehicle in simulator.llm_vehicles:
                print(
                    f"Ollama vehicle {vehicle.id}: "
                    f"{vehicle.llm_request_count} requests, "
                    f"{vehicle.idm_fallback_count} IDM fallbacks, "
                    f"final position {vehicle.position:.1f} m, "
                    f"final speed {vehicle.speed:.2f} m/s, "
                    f"final acceleration {vehicle.acceleration:.3f} m/s^2"
                )
        else:
            print(
                f"Warning: vehicle {args.llm_vehicle_id} was not generated; "
                "no Ollama control was applied."
            )
    plot_time_space_diagram(
        simulator,
        config,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
