"""Post-run performance metrics derived from recorded vehicle histories."""


STOPPED_SPEED_THRESHOLD_MPS = 0.5
SLOW_SPEED_THRESHOLD_MPS = 5.0


def collect_traffic_metrics(simulator, config):
    """Summarize completed trips, vehicle-time speeds, and slow-traffic extent."""
    completed_vehicles = [
        vehicle
        for vehicle in simulator.vehicles
        if not vehicle.active and vehicle.position >= config.road_length
    ]
    travel_times = [
        vehicle.history[-1]["t"] - vehicle.history[0]["t"]
        for vehicle in completed_vehicles
        if len(vehicle.history) >= 2
    ]

    speed_samples = []
    slow_vehicles_by_time = {}
    for vehicle in simulator.vehicles:
        for state in vehicle.history[1:]:
            if state["position"] >= config.road_length:
                continue
            speed_samples.append(state["speed"])
            if state["speed"] <= SLOW_SPEED_THRESHOLD_MPS:
                slow_vehicles_by_time.setdefault(state["t"], []).append(
                    state["position"]
                )

    maximum_slow_vehicle_count = max(
        (len(positions) for positions in slow_vehicles_by_time.values()),
        default=0,
    )
    maximum_slow_vehicle_span_m = max(
        (
            max(positions) - min(positions) + config.vehicle_length
            for positions in slow_vehicles_by_time.values()
        ),
        default=0.0,
    )
    stopped_samples = sum(
        speed <= STOPPED_SPEED_THRESHOLD_MPS for speed in speed_samples
    )

    return {
        "generated_vehicle_count": len(simulator.vehicles),
        "completed_vehicle_count": len(completed_vehicles),
        "vehicles_remaining_at_end": (
            len(simulator.vehicles) - len(completed_vehicles)
        ),
        "throughput_vehicles_per_hour": (
            len(completed_vehicles) / config.time_max * 3600.0
        ),
        "mean_completed_travel_time_s": _mean(travel_times),
        "median_completed_travel_time_s": _percentile(travel_times, 50.0),
        "p95_completed_travel_time_s": _percentile(travel_times, 95.0),
        "mean_speed_mps": _mean(speed_samples),
        "stopped_time_fraction": (
            stopped_samples / len(speed_samples)
            if speed_samples
            else None
        ),
        "stopped_speed_threshold_mps": STOPPED_SPEED_THRESHOLD_MPS,
        "slow_speed_threshold_mps": SLOW_SPEED_THRESHOLD_MPS,
        "maximum_slow_vehicle_count": maximum_slow_vehicle_count,
        "maximum_slow_vehicle_span_m": maximum_slow_vehicle_span_m,
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def _percentile(values, percentile):
    if not values:
        return None

    ordered_values = sorted(values)
    rank = (len(ordered_values) - 1) * percentile / 100.0
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    fraction = rank - lower_index
    return (
        ordered_values[lower_index] * (1.0 - fraction)
        + ordered_values[upper_index] * fraction
    )
