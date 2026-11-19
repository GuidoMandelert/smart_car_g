# Car-Following Simulation — Teaching Version

This directory contains a readable, corrected car-following simulation for
teaching IDM, traffic bottlenecks, and stop-and-go traffic.

## Program structure

The simulation follows the same sequence at every time step:

```text
Generate vehicle
    -> check road conditions
    -> calculate every vehicle's acceleration
    -> update every vehicle's speed and position
    -> record the results
```

| File | Purpose |
| --- | --- |
| `config.py` | Road, IDM, bottleneck, and experiment parameters |
| `vehicle.py` | Shared vehicle state and motion update |
| `vehicle_idm.py` | IDM acceleration calculation |
| `simulator.py` | The five-step simulation loop |
| `plotting.py` | Speed-colored time-space diagram |
| `vehicle_knn.py` | Empty extension point for a future KNN policy |
| `vehicle_llm.py` | Empty extension point for a future LLM policy |
| `test_simulation.py` | Optional correctness tests; not part of a normal run |

## Run

From the repository root:

```bash
.venv/bin/python car-following-simulation/main.py --experiment 1
```

The default duration is defined only once, as `DEFAULT_TIME_MAX` in
`config.py`. Override it for one run with:

```bash
.venv/bin/python car-following-simulation/main.py \
  --experiment 1 --time-max 500
```

Save the plot without opening a window:

```bash
.venv/bin/python car-following-simulation/main.py \
  --experiment 1 \
  --output car-following-simulation/results/experiment-1.png \
  --no-show
```

## Experiments

| Experiment | Arrival process | Bottleneck duration |
| --- | --- | --- |
| 1 | Fixed arrival interval | 100–300 seconds |
| 2 | Random interval | 100–300 seconds |
| 3 | Fixed arrival interval | 100 seconds to the end |
| 4 | Random interval | 100 seconds to the end |

The temporary low-speed region is configured relative to the road end in
`config.py`. It is a simple single-lane representation of a temporary highway
bottleneck; it does not model an actual lane drop or lane-changing behavior.

A vehicle is affected only if it enters the low-speed region after the
bottleneck has started and before it ends. A vehicle that entered the region
before the start time continues through without adopting the bottleneck speed.

## Important correction

The original program updated speed first and then used the updated speed in a
second kinematic formula for position. This version updates speed and position
together using the average of the old and new speeds:

```text
new speed = old speed + acceleration x time step
distance  = (old speed + new speed) / 2 x time step
```

All accelerations are calculated before any vehicle moves, so every vehicle
uses the same pre-update traffic state.

## Optional tests

```bash
cd car-following-simulation
../.venv/bin/python -m unittest -v test_simulation.py test_traffic_metrics.py
```

The tests are separate from the teaching workflow and do not run when
`main.py` is executed.

## Reproducible comparison metrics

Use the original scenario for the primary comparison: do not pass
`--safe-entry-gap`. The default entrance rule and experiment settings are
preserved. `--seed` defaults to the configured seed and can be set explicitly
for matched runs. The stricter `--safe-entry-gap` is a separate sensitivity
experiment and must be applied to all controllers if used. The section 16
results in the experiment log used that stricter rule (131 vehicles); they
should not be compared directly with original-entry runs (177 vehicles).

Use 400 seconds for the next matched comparison: the 300-second original-entry
IDM run completed only 6 trips, while the 400-second run completed 61 and
includes 100 seconds after the bottleneck ends. The commands below use the
same experiment, duration, seed, and original entry rule. IDM requires no API
key; the two DeepSeek runs make billable API calls. Run those commands only
when ready to incur the charges.

```powershell
.\.venv\Scripts\python.exe car-following-simulation\main.py `
  --controller idm `
  --experiment 1 `
  --time-max 400 `
  --seed 1 `
  --metrics-output evidence\idm-default-entry-seed-1-400s-metrics.json `
  --output evidence\idm-default-entry-seed-1-400s.png `
  --no-show

.\.venv\Scripts\python.exe car-following-simulation\main.py `
  --controller deepseek `
  --experiment 1 `
  --time-max 400 `
  --seed 1 `
  --llm-first-vehicle-id 2 `
  --llm-vehicle-count 50 `
  --llm-model deepseek-flash `
  --llm-decision-interval-steps 200 `
  --metrics-output evidence\deepseek-50-default-entry-seed-1-400s-metrics.json `
  --output evidence\deepseek-50-default-entry-seed-1-400s.png `
  --no-show

.\.venv\Scripts\python.exe car-following-simulation\main.py `
  --controller deepseek-fleet `
  --experiment 1 `
  --time-max 400 `
  --seed 1 `
  --metrics-output evidence\deepseek-fleet-default-entry-seed-1-400s-metrics.json `
  --output evidence\deepseek-fleet-default-entry-seed-1-400s.png `
  --no-show
```

`--metrics-output` writes JSON with run settings, completed-trip throughput
and travel times, vehicle-time mean speed and stopped-time fraction, the
maximum number and spatial span of slow vehicles, observed gaps and overlaps,
controller request/token/fallback statistics when available, and measured
wall-clock time spent in the simulation loop. Mean speed
and stopped-time fraction are calculated across recorded on-road vehicle-time
samples; stopped time is defined as speed at or below 0.5 m/s. Slow traffic
for the maximum count and spatial span is defined as speed at or below 5 m/s,
which includes vehicles moving at the configured 2 m/s bottleneck limit. The
reported spatial span is the distance from the furthest-forward to
furthest-back slow vehicle at a sampled time, including one vehicle length.
These are comparative simulation measures, not safety guarantees. For
stronger evidence, repeat each condition across multiple seeds and compare
the per-run JSON outputs rather than relying on a single plot or seed.

## Optional DeepSeek vehicle-control experiment

The simulation uses IDM by default and makes no API calls unless a DeepSeek
controller is selected. With `--controller deepseek`, DeepSeek controls vehicle 2
only; all other vehicles continue using IDM. To control a consecutive range,
set `--llm-vehicle-count`. The default first vehicle ID is 2, leaving vehicle 1
under IDM as a lead vehicle.

1. Create an API key in the [DeepSeek platform](https://platform.deepseek.com/api_keys).
   Do not paste the key into chat or commit it to source control.
2. Put the key in `car-following-simulation\.env`:

   ```text
   DEEPSEEK_API_KEY=your_key_goes_here
   ```

   The root `.gitignore` excludes `.env` files. A shell environment variable
   named `DEEPSEEK_API_KEY` can also be used and takes precedence.
3. From the repository root, install the simulator dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r car-following-simulation\requirements.txt
   ```
4. Run a short opt-in test:

   ```powershell
   .\.venv\Scripts\python.exe car-following-simulation\main.py `
     --experiment 1 `
     --time-max 200 `
     --controller deepseek `
     --llm-first-vehicle-id 2 `
     --llm-vehicle-count 10 `
     --llm-model deepseek-flash `
     --llm-decision-interval-steps 200 `
     --output evidence\deepseek-vehicles-2-to-11-exp1.png `
     --no-show
   ```

To select 50 consecutive vehicles instead, use
`--llm-vehicle-count 50`; with the default first ID of 2 this selects IDs 2
through 51. The old `--llm-vehicle-id` option remains as an alias for choosing
the first selected ID.

### Centralized fleet-batch experiment

To test fleet-level coordination without changing the existing per-vehicle
controller, select `--controller deepseek-fleet`. One request contains the
current state of every active vehicle and must return exactly one desired
target speed per vehicle. The default batch interval is 200 steps (20 simulated
seconds). Target speeds are cached between batches, while IDM recalculates
each vehicle's acceleration from its current state every simulation step. The
target speed is rate-limited by the vehicle's acceleration and comfortable
deceleration settings. A local filter then limits acceleration to preserve the
minimum bumper spacing on the next step when feasible; it is an experimental
guard, not a formal safety guarantee. Vehicles created between batches, or
after a failed batch, follow the road speed limit using IDM until their first
fleet target is received. By default, vehicle admission uses the original
scenario's entry-gap rule. Add `--safe-entry-gap` to opt into the stricter
initial-speed and emergency-braking clearance rule for controlled comparisons.

```powershell
.\.venv\Scripts\python.exe car-following-simulation\main.py `
  --experiment 1 `
  --time-max 300 `
  --controller deepseek-fleet `
  --llm-model deepseek-flash `
  --fleet-decision-interval-steps 200 `
  --output evidence\deepseek-central-fleet-targetspeed-exp1-300s.png `
  --no-show 2>&1 | Tee-Object -FilePath evidence\deepseek-central-fleet-targetspeed-exp1-300s.log
```

The centralized mode is a separate implementation in
`coursework_tools/llm_control/deepseek_fleet_controller.py`; the original
single/per-vehicle control logic in `vehicle_llm_deepseek.py` remains separate;
token-usage reporting there is telemetry-only and does not change its control
decisions.
The API response is validated as a complete fleet target-speed map; on batch
failure or for a vehicle not in the latest snapshot, IDM uses the road speed
limit until a target is available. The centralized mode's local filter checks
next-step minimum spacing; the existing per-vehicle mode retains its separate
stopping-distance filter. Neither is a formal safety guarantee.
Centralization reduces request count, but sending the whole fleet and
receiving a target per vehicle may not reduce total token use. When DeepSeek
returns usage metadata, the run summary reports prompt, completion, and total
tokens so request-count and token-cost efficiency can be compared.

The controller requests fleet target speeds every 200 simulation steps in the
example (20 simulated seconds at a 0.1-second timestep); local car-following
acceleration is still updated every 0.1 seconds. API responses are synchronous
and usage is billed by tokens; consult
[DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing) before
long experiments. As a rough planning estimate for a 400-second Experiment 1
run at that interval, the IDM baseline vehicle lifetimes imply about 147
requests for IDs 2-11 and about 785 for IDs 2-51. These are estimates, not
guaranteed totals; DeepSeek decisions may change vehicle lifetimes and traffic
generation. Each DeepSeek action passes a per-step safety filter that checks
the predicted gap against the current leader, including when that leader is
another DeepSeek-controlled vehicle. The per-vehicle controller uses its
stopping-distance filter; the fleet controller uses its next-step
minimum-spacing filter. If the respective constraint is already infeasible,
the filter requests maximum braking and reports that condition. The simulator
also reports minimum observed bumper gap, gap observations below the IDM
minimum spacing, and detected overlaps. These checks are simplified
simulation safeguards, not proof of collision avoidance or a safety-certified
real-vehicle controller. Both controllers fall back to IDM when an LLM request
or response fails.

To turn DeepSeek off, omit `--controller deepseek` (IDM is the default). The
offline tests for the controller use mocked API responses and do not incur API
charges.
