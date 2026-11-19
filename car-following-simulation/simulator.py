"""The main simulation loop."""

import random

from vehicle_idm import IDMVehicle


class Simulator:
    def __init__(
        self,
        config,
        llm_vehicle_factory=None,
        llm_vehicle_id=2,
        llm_vehicle_count=1,
        llm_vehicle_kwargs=None,
        fleet_controller=None,
        safe_entry_gap=False,
    ):
        if llm_vehicle_id < 1:
            raise ValueError("llm_vehicle_id must be positive")
        if llm_vehicle_count < 1:
            raise ValueError("llm_vehicle_count must be positive")
        self.config = config
        self.llm_vehicle_factory = llm_vehicle_factory
        self.llm_vehicle_id = llm_vehicle_id
        self.llm_vehicle_count = llm_vehicle_count
        self.llm_vehicle_kwargs = llm_vehicle_kwargs or {}
        self.fleet_controller = fleet_controller
        self.safe_entry_gap = safe_entry_gap
        self.llm_vehicles = []
        self.vehicles = []
        self.next_generation_time = 0.0
        self.minimum_gap_observed = None
        self.minimum_gap_details = None
        self.gap_below_minimum_observations = 0
        self.collision_events = []
        self._colliding_pairs = set()

    def run(self, verbose=True):
        """Generate vehicles and update their states one time step at a time."""
        time_step = self.config.time_step
        number_of_steps = int(round(self.config.time_max / time_step))

        for step in range(number_of_steps):
            current_time = step * time_step

            if verbose and step > 0 and step % int(100 / time_step) == 0:
                print(f"step: {step}, time: {current_time:.0f}")

            # 1. Add a vehicle when its arrival time has been reached.
            self._generate_vehicle(current_time)

            active_vehicles = [vehicle for vehicle in self.vehicles if vehicle.active]

            # 2. Set the desired speed from the current road conditions.
            for vehicle in active_vehicles:
                vehicle.check_road(current_time)

            # 3. Calculate all accelerations before changing any vehicle state.
            if self.fleet_controller is None:
                for vehicle in active_vehicles:
                    vehicle.update_acceleration()
            else:
                fleet_target_speeds = self.fleet_controller.actions_for_step(
                    active_vehicles,
                    step,
                )
                for vehicle in active_vehicles:
                    target_speed = fleet_target_speeds.get(vehicle.id)
                    if target_speed is None:
                        target_speed = vehicle.current_speed_limit
                        self.fleet_controller.record_idm_fallback()
                    vehicle.update_acceleration(desired_speed=target_speed)

                for vehicle in sorted(
                    active_vehicles,
                    key=lambda item: item.position,
                    reverse=True,
                ):
                    vehicle.acceleration = (
                        self.fleet_controller.apply_safety_filter(
                            vehicle,
                            vehicle.acceleration,
                        )
                    )

            # 4. Update all speeds and positions.
            for vehicle in active_vehicles:
                vehicle.update_motion()

            # 5. Record only vehicles that are still on the road, plus the
            # final state of a vehicle that has just reached the road end.
            next_time = (step + 1) * time_step
            self._record_gap_metrics(active_vehicles, next_time)
            for vehicle in active_vehicles:
                vehicle.record_state(next_time)

        if verbose:
            inflow_rate = len(self.vehicles) / self.config.time_max * 3600
            print(f"\nVehicle number: {len(self.vehicles)}")
            print(f"Average admitted inflow: {inflow_rate:.0f} veh/h\n")
            print(f"Observed overlaps: {len(self.collision_events)}")
            if self.minimum_gap_details is None:
                print("Minimum observed bumper gap: unavailable")
            else:
                gap = self.minimum_gap_details
                print(
                    f"Minimum observed bumper gap: {gap['gap']:.3f} m "
                    f"(vehicles {gap['rear_vehicle_id']} and "
                    f"{gap['front_vehicle_id']} at {gap['time']:.1f} s)"
                )
            print(
                f"Gap observations below IDM minimum spacing "
                f"({self.config.idm_minimum_spacing:.1f} m): "
                f"{self.gap_below_minimum_observations}\n"
            )

    def _record_gap_metrics(self, vehicles, current_time):
        on_road = [
            vehicle
            for vehicle in vehicles
            if vehicle.position < self.config.road_length
        ]
        ordered = sorted(on_road, key=lambda vehicle: vehicle.position)
        colliding_pairs = set()

        for rear, front in zip(ordered, ordered[1:]):
            gap = front.position - rear.position - rear.vehicle_length
            pair = (rear.id, front.id)

            if (
                self.minimum_gap_observed is None
                or gap < self.minimum_gap_observed
            ):
                self.minimum_gap_observed = gap
                self.minimum_gap_details = {
                    "gap": gap,
                    "rear_vehicle_id": rear.id,
                    "front_vehicle_id": front.id,
                    "time": current_time,
                }

            if gap < self.config.idm_minimum_spacing:
                self.gap_below_minimum_observations += 1

            if gap < 0.0:
                colliding_pairs.add(pair)
                if pair not in self._colliding_pairs:
                    self.collision_events.append(
                        {
                            "rear_vehicle_id": rear.id,
                            "front_vehicle_id": front.id,
                            "time": current_time,
                            "gap": gap,
                        }
                    )

        self._colliding_pairs = colliding_pairs

    def _generate_vehicle(self, current_time):
        """Generate at most one vehicle when the road entrance is clear."""
        if current_time + 1e-12 < self.next_generation_time:
            return

        active_vehicles = [vehicle for vehicle in self.vehicles if vehicle.active]
        vehicle_front = active_vehicles[-1] if active_vehicles else None

        # Do not insert a new vehicle on top of a queue at the entrance.
        if vehicle_front is not None:
            if self.safe_entry_gap:
                initial_speed = self.config.initial_speed
                stopping_gap = (
                    initial_speed * self.config.time_step
                    + initial_speed**2
                    / (2.0 * self.config.idm_maximum_deceleration)
                )
                following_gap = (
                    initial_speed * self.config.idm_time_headway
                )
                entry_gap = (
                    self.config.vehicle_length
                    + self.config.idm_minimum_spacing
                    + max(stopping_gap, following_gap)
                )
            else:
                entry_gap = (
                    self.config.vehicle_length
                    + self.config.idm_minimum_spacing
                    + self.config.initial_speed * self.config.idm_time_headway
                )
            if vehicle_front.position < entry_gap:
                return

        vehicle_id = len(self.vehicles) + 1
        if (
            self.llm_vehicle_factory is not None
            and self.llm_vehicle_id
            <= vehicle_id
            < self.llm_vehicle_id + self.llm_vehicle_count
        ):
            vehicle = self.llm_vehicle_factory(
                self.config,
                vehicle_id,
                vehicle_front,
                **self.llm_vehicle_kwargs,
            )
            self.llm_vehicles.append(vehicle)
        else:
            vehicle = IDMVehicle(self.config, vehicle_id, vehicle_front)
        vehicle.record_state(current_time)
        self.vehicles.append(vehicle)

        interval = self.config.vehicle_min_interval
        if self.config.vehicle_extra_interval > 0:
            interval += random.expovariate(
                1.0 / self.config.vehicle_extra_interval
            )
        self.next_generation_time = self.next_generation_time + interval
