"""Run one car-following experiment."""

import argparse
import json
import sys
import time
from pathlib import Path

from config import Config
from plotting import plot_time_space_diagram
from simulator import Simulator
from traffic_metrics import collect_traffic_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run a car-following simulation.")
    parser.add_argument(
        "--controller",
        choices=("idm", "deepseek", "deepseek-fleet"),
        default="idm",
        help=(
            "IDM, per-vehicle DeepSeek, or one centralized DeepSeek batch "
            "for the active fleet."
        ),
    )
    parser.add_argument(
        "--llm-first-vehicle-id",
        "--llm-vehicle-id",
        dest="llm_first_vehicle_id",
        type=int,
        default=2,
        help=(
            "First consecutive vehicle ID controlled by DeepSeek "
            "in per-vehicle mode (default: 2; --llm-vehicle-id is an alias)."
        ),
    )
    parser.add_argument(
        "--llm-vehicle-count",
        type=int,
        default=1,
        help=(
            "Number of consecutive vehicles controlled in per-vehicle mode "
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
        "--fleet-decision-interval-steps",
        type=int,
        default=200,
        help=(
            "Simulation steps between centralized fleet requests "
            "(default: 200)."
        ),
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
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.DEFAULT_SEED,
        help=f"Random seed for reproducible runs (default: {Config.DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--safe-entry-gap",
        action="store_true",
        help=(
            "Use emergency-braking clearance when admitting a vehicle. "
            "Default preserves the original scenario's entry rule."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional JSON file for traffic, safety, and controller metrics.",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        experiment=args.experiment,
        time_max=args.time_max,
    )
    llm_vehicle_factory = None
    llm_vehicle_kwargs = None
    fleet_controller = None
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
    elif args.controller == "deepseek-fleet":
        if args.fleet_decision_interval_steps < 1:
            parser.error("--fleet-decision-interval-steps must be positive")
        if args.llm_timeout_seconds <= 0:
            parser.error("--llm-timeout-seconds must be positive")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from coursework_tools.llm_control.deepseek_fleet_controller import (
            DeepSeekFleetController,
            get_api_key,
        )

        try:
            api_key = get_api_key()
        except RuntimeError as exc:
            parser.error(str(exc))
        fleet_controller = DeepSeekFleetController(
            model=args.llm_model,
            timeout_seconds=args.llm_timeout_seconds,
            decision_interval_steps=args.fleet_decision_interval_steps,
            api_key=api_key,
        )

    simulator = Simulator(
        config,
        llm_vehicle_factory=llm_vehicle_factory,
        llm_vehicle_id=args.llm_first_vehicle_id,
        llm_vehicle_count=args.llm_vehicle_count,
        llm_vehicle_kwargs=llm_vehicle_kwargs,
        fleet_controller=fleet_controller,
        safe_entry_gap=args.safe_entry_gap,
    )
    simulation_started_at = time.perf_counter()
    simulator.run()
    simulation_wall_time_s = time.perf_counter() - simulation_started_at
    print(f"Simulation wall-clock time: {simulation_wall_time_s:.2f} s")
    controller_metrics = None
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
            total_prompt_tokens = sum(
                vehicle.prompt_tokens for vehicle in simulator.llm_vehicles
            )
            total_completion_tokens = sum(
                vehicle.completion_tokens for vehicle in simulator.llm_vehicles
            )
            total_tokens = sum(
                vehicle.total_tokens for vehicle in simulator.llm_vehicles
            )
            usage_reported = sum(
                vehicle.usage_reported_request_count
                for vehicle in simulator.llm_vehicles
            )
            usage_missing = sum(
                vehicle.usage_missing_request_count
                for vehicle in simulator.llm_vehicles
            )
            total_safety_interventions = sum(
                v.safety_filter_interventions for v in simulator.llm_vehicles
            )
            total_unachievable_steps = sum(
                v.safety_margin_unachievable_steps
                for v in simulator.llm_vehicles
            )
            controller_metrics = {
                "controlled_vehicle_count": len(simulator.llm_vehicles),
                "request_count": total_requests,
                "idm_fallback_count": total_fallbacks,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "usage_reported_request_count": usage_reported,
                "usage_missing_request_count": usage_missing,
                "safety_filter_interventions": total_safety_interventions,
                "unachievable_stopping_margin_vehicle_steps": (
                    total_unachievable_steps
                ),
            }
            print(
                f"DeepSeek vehicles {args.llm_first_vehicle_id}-"
                f"{expected_last_id}: {len(simulator.llm_vehicles)} generated, "
                f"{total_requests} requests, {total_fallbacks} IDM fallbacks, "
                f"{total_safety_interventions} "
                "safety-filter interventions, "
                f"{total_unachievable_steps} "
                "steps where the stopping-distance margin was infeasible."
            )
            print(
                f"Per-vehicle DeepSeek token usage: {total_tokens} tokens "
                f"({total_prompt_tokens} input, {total_completion_tokens} output; "
                f"usage reported for {usage_reported} requests, "
                f"missing for {usage_missing})."
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
            controller_metrics = {
                "controlled_vehicle_count": 0,
                "request_count": 0,
                "idm_fallback_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "usage_reported_request_count": 0,
                "usage_missing_request_count": 0,
            }
            print(
                f"Warning: no vehicles in the selected ID range "
                f"{args.llm_first_vehicle_id}-{expected_last_id} were "
                "generated; no DeepSeek control was applied."
            )
    elif args.controller == "deepseek-fleet":
        print(
            "Centralized DeepSeek fleet: "
            f"{fleet_controller.request_count} batch requests, "
            f"{fleet_controller.successful_batch_count} successful batches, "
            f"{fleet_controller.failed_batch_count} failed batches, "
            f"{fleet_controller.idm_fallback_vehicle_steps} road-speed-limit "
            "vehicle-steps, "
            f"{fleet_controller.total_tokens} tokens "
            f"({fleet_controller.prompt_tokens} input, "
            f"{fleet_controller.completion_tokens} output; usage reported "
            f"for {fleet_controller.usage_reported_batch_count} batches, "
            f"missing for {fleet_controller.usage_missing_batch_count}), "
            f"{fleet_controller.clamped_target_speeds} clamped "
            "target speeds, "
            f"{fleet_controller.safety_filter_interventions} safety-filter "
            "interventions, "
            f"{fleet_controller.minimum_gap_unachievable_steps} "
            "unachievable next-step minimum-gap vehicle-steps."
        )

        controller_metrics = {
            "batch_request_count": fleet_controller.request_count,
            "successful_batch_count": fleet_controller.successful_batch_count,
            "failed_batch_count": fleet_controller.failed_batch_count,
            "road_speed_limit_fallback_vehicle_steps": (
                fleet_controller.idm_fallback_vehicle_steps
            ),
            "prompt_tokens": fleet_controller.prompt_tokens,
            "completion_tokens": fleet_controller.completion_tokens,
            "total_tokens": fleet_controller.total_tokens,
            "usage_reported_batch_count": (
                fleet_controller.usage_reported_batch_count
            ),
            "usage_missing_batch_count": (
                fleet_controller.usage_missing_batch_count
            ),
            "clamped_target_speed_count": (
                fleet_controller.clamped_target_speeds
            ),
            "safety_filter_interventions": (
                fleet_controller.safety_filter_interventions
            ),
            "unachievable_next_step_gap_vehicle_steps": (
                fleet_controller.minimum_gap_unachievable_steps
            ),
        }

    traffic_metrics = collect_traffic_metrics(simulator, config)
    print(
        f"Completed vehicles: {traffic_metrics['completed_vehicle_count']} "
        f"({traffic_metrics['throughput_vehicles_per_hour']:.0f} veh/h)"
    )
    if traffic_metrics["mean_completed_travel_time_s"] is None:
        print("Completed-trip time: unavailable")
    else:
        print(
            "Completed-trip time (mean / median / p95): "
            f"{traffic_metrics['mean_completed_travel_time_s']:.1f} / "
            f"{traffic_metrics['median_completed_travel_time_s']:.1f} / "
            f"{traffic_metrics['p95_completed_travel_time_s']:.1f} s"
        )
    if traffic_metrics["mean_speed_mps"] is None:
        print("Mean speed and stopped-time fraction: unavailable")
    else:
        print(
            f"Mean vehicle-time speed: {traffic_metrics['mean_speed_mps']:.2f} "
            f"m/s; stopped-time fraction "
            f"(<= {traffic_metrics['stopped_speed_threshold_mps']:.1f} m/s): "
            f"{traffic_metrics['stopped_time_fraction']:.3f}"
        )
    print(
        f"Maximum slow vehicles (<= "
        f"{traffic_metrics['slow_speed_threshold_mps']:.1f} m/s): "
        f"{traffic_metrics['maximum_slow_vehicle_count']}; maximum "
        f"slow-vehicle span: "
        f"{traffic_metrics['maximum_slow_vehicle_span_m']:.1f} m"
    )

    run_summary = {
        "run": {
            "controller": args.controller,
            "model": args.llm_model if args.controller != "idm" else None,
            "seed": args.seed,
            "experiment": args.experiment,
            "duration_s": config.time_max,
            "simulation_wall_time_s": simulation_wall_time_s,
            "time_step_s": config.time_step,
            "road_length_m": config.road_length,
            "speed_limit_mps": config.speed_limit,
            "vehicle_arrival_min_interval_s": config.vehicle_min_interval,
            "vehicle_arrival_extra_interval_s": (
                config.vehicle_extra_interval
            ),
            "bottleneck": {
                "start_x_m": config.bottleneck_x_start,
                "end_x_m": config.bottleneck_x_end,
                "start_time_s": config.bottleneck_t_start,
                "end_time_s": config.bottleneck_t_end,
                "speed_limit_mps": config.bottleneck_speed_limit,
                "influence_rate": config.bottleneck_influence_rate,
            },
            "safe_entry_gap": args.safe_entry_gap,
            "llm_first_vehicle_id": (
                args.llm_first_vehicle_id
                if args.controller == "deepseek"
                else None
            ),
            "llm_vehicle_count": (
                args.llm_vehicle_count
                if args.controller == "deepseek"
                else None
            ),
            "decision_interval_steps": (
                args.llm_decision_interval_steps
                if args.controller == "deepseek"
                else args.fleet_decision_interval_steps
                if args.controller == "deepseek-fleet"
                else None
            ),
        },
        "traffic": traffic_metrics,
        "safety": {
            "observed_overlap_count": len(simulator.collision_events),
            "minimum_observed_bumper_gap_m": (
                simulator.minimum_gap_observed
            ),
            "minimum_gap_details": simulator.minimum_gap_details,
            "gap_observations_below_minimum_spacing": (
                simulator.gap_below_minimum_observations
            ),
        },
        "controller_metrics": controller_metrics,
    }
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(run_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Run metrics saved to: {args.metrics_output}")

    plot_time_space_diagram(
        simulator,
        config,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
