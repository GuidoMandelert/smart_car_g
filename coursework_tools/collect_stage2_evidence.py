"""Run an IDM scenario and save a diagram plus an editable evidence note."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_DIR = PROJECT_ROOT / "car-following-simulation"
sys.path.insert(0, str(SIMULATION_DIR))

from config import Config
from plotting import plot_time_space_diagram
from simulator import Simulator


def describe_arrivals(config):
    if config.experiment in (1, 3):
        return f"Fixed interval: {config.vehicle_min_interval:g} s"

    return (
        f"Random interval: {config.vehicle_min_interval:g} s minimum plus "
        f"exponential interval (mean {config.vehicle_extra_interval:g} s)"
    )


def describe_bottleneck(config):
    if config.time_max <= config.bottleneck_t_start:
        period = (
            "not active during this run; simulation ends at "
            f"{config.time_max:g} s, before scheduled start at "
            f"{config.bottleneck_t_start:g} s"
        )
    else:
        end_time = min(config.bottleneck_t_end, config.time_max)
        period = f"{config.bottleneck_t_start:g}-{end_time:g} s"

    return (
        f"{period}; position {config.bottleneck_x_start:g}-"
        f"{config.bottleneck_x_end:g} m; speed limit "
        f"{config.bottleneck_speed_limit:g} m/s"
    )


def display_path(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_report(path, config, simulator, diagram_path):
    vehicle_count = len(simulator.vehicles)
    average_inflow = vehicle_count / config.time_max * 3600
    content = f"""# Car-Following Simulation Run Evidence

## Run settings

- **Controller:** IDM
- **Experiment:** {config.experiment}
- **Seed:** {config.seed}
- **Duration:** {config.time_max:g} seconds
- **Arrival process:** {describe_arrivals(config)}
- **Bottleneck:** {describe_bottleneck(config)}
- **Diagram:** {display_path(diagram_path)}

## Simulation output

- **Vehicles generated:** {vehicle_count}
- **Average admitted inflow:** {average_inflow:.1f} vehicles/hour

## Visual observations

After opening the diagram, replace this prompt with your own observations.
Describe where vehicles slow down, whether a queue or stop-and-go wave forms,
and whether traffic recovers after the bottleneck ends.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run an existing IDM experiment and save the plot and evidence note. "
            "This helper does not modify the simulator."
        )
    )
    parser.add_argument("--experiment", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-max", type=float, default=Config.DEFAULT_TIME_MAX)
    parser.add_argument("--output-dir", type=Path, default=Path("evidence"))
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        experiment=args.experiment,
        time_max=args.time_max,
    )
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    stem = f"stage2-idm-exp{args.experiment}-seed{args.seed}"
    diagram_path = output_dir / f"{stem}.png"
    report_path = output_dir / f"{stem}.md"

    simulator = Simulator(config)
    simulator.run()
    plot_time_space_diagram(
        simulator,
        config,
        output=diagram_path,
        show=False,
    )
    write_report(report_path, config, simulator, diagram_path)
    print(f"Evidence note saved to: {display_path(report_path)}")


if __name__ == "__main__":
    main()
