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
        choices=("idm", "deepseek"),
        default="idm",
        help="Controller for selected vehicles; unselected vehicles use IDM.",
    )
    parser.add_argument(
        "--llm-first-vehicle-id",
        "--llm-vehicle-id",
        dest="llm_first_vehicle_id",
        type=int,
        default=2,
        help=(
            "First consecutive vehicle ID controlled by DeepSeek "
            "(default: 2; --llm-vehicle-id is an alias)."
        ),
    )
    parser.add_argument(
        "--llm-vehicle-count",
        type=int,
        default=1,
        help=(
            "Number of consecutive vehicles controlled by DeepSeek "
            "starting at --llm-first-vehicle-id (default: 1)."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default="deepseek-flash",
        help="DeepSeek API model name (default: deepseek-flash).",
    )
    parser.add_argument(
        "--llm-decision-interval-steps",
        type=int,
        default=50,
        help="Simulation steps between DeepSeek requests (default: 50).",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=60.0,
        help="Maximum wait for a DeepSeek response (default: 60 seconds).",
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
    if args.controller == "deepseek":
        if args.llm_first_vehicle_id < 1:
            parser.error("--llm-first-vehicle-id must be a positive integer")
        if args.llm_vehicle_count < 1:
            parser.error("--llm-vehicle-count must be a positive integer")
        if args.llm_decision_interval_steps < 1:
            parser.error("--llm-decision-interval-steps must be positive")
        if args.llm_timeout_seconds <= 0:
            parser.error("--llm-timeout-seconds must be positive")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from coursework_tools.llm_control.vehicle_llm_deepseek import (
            DeepSeekLLMVehicle,
            get_api_key,
        )

        try:
            get_api_key()
        except RuntimeError as exc:
            parser.error(str(exc))
        llm_vehicle_factory = DeepSeekLLMVehicle
        llm_vehicle_kwargs = {
            "model": args.llm_model,
            "timeout_seconds": args.llm_timeout_seconds,
            "decision_interval_steps": args.llm_decision_interval_steps,
        }

    simulator = Simulator(
        config,
        llm_vehicle_factory=llm_vehicle_factory,
        llm_vehicle_id=args.llm_first_vehicle_id,
        llm_vehicle_count=args.llm_vehicle_count,
        llm_vehicle_kwargs=llm_vehicle_kwargs,
    )
    simulator.run()
    if args.controller == "deepseek":
        expected_last_id = (
            args.llm_first_vehicle_id + args.llm_vehicle_count - 1
        )
        if simulator.llm_vehicles:
            total_requests = sum(
                vehicle.llm_request_count for vehicle in simulator.llm_vehicles
            )
            total_fallbacks = sum(
                vehicle.idm_fallback_count for vehicle in simulator.llm_vehicles
            )
            print(
                f"DeepSeek vehicles {args.llm_first_vehicle_id}-"
                f"{expected_last_id}: {len(simulator.llm_vehicles)} generated, "
                f"{total_requests} requests, {total_fallbacks} IDM fallbacks."
            )
            for vehicle in simulator.llm_vehicles:
                print(
                    f"DeepSeek vehicle {vehicle.id}: "
                    f"{vehicle.llm_request_count} requests, "
                    f"{vehicle.idm_fallback_count} IDM fallback events, "
                    f"final position {vehicle.position:.1f} m, "
                    f"final speed {vehicle.speed:.2f} m/s"
                )
        else:
            print(
                f"Warning: no vehicles in the selected ID range "
                f"{args.llm_first_vehicle_id}-{expected_last_id} were "
                "generated; no DeepSeek control was applied."
            )
    plot_time_space_diagram(
        simulator,
        config,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
