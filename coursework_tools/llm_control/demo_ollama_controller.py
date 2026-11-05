"""Ask Ollama for one acceleration recommendation on a synthetic vehicle."""

import argparse
import logging

from .vehicle_llm_ollama import OllamaLLMVehicle
from config import Config


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Make one isolated Ollama controller request. This does not run "
            "or modify the traffic simulator."
        )
    )
    parser.add_argument("--model", help="Ollama model name; defaults to OLLAMA_MODEL.")
    parser.add_argument(
        "--decision-interval-steps",
        type=int,
        default=10,
        help="Simulation updates between model decisions.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Maximum wait for a local Ollama response (default: 120 seconds).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # This configuration only supplies the vehicle parameters the controller
    # inherits from IDM. No simulator loop is started by this demonstration.
    config = Config(seed=1, experiment=1, time_max=1.0)
    vehicle = OllamaLLMVehicle(
        config,
        vehicle_id=1,
        model=args.model,
        decision_interval_steps=args.decision_interval_steps,
        timeout_seconds=args.timeout_seconds,
    )
    vehicle.check_road(current_time=0.0)
    print("Synthetic observation:")
    print(vehicle.get_observation())

    # The first update contacts Ollama synchronously. If the service fails or
    # returns invalid output, the copied controller logs and uses IDM instead.
    vehicle.update_acceleration()
    print(f"Acceleration selected: {vehicle.acceleration:.3f} m/s^2")
    print(f"Ollama requests: {vehicle.llm_request_count}")
    print(f"IDM fallbacks: {vehicle.idm_fallback_count}")


if __name__ == "__main__":
    main()
