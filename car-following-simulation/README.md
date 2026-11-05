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
../.venv/bin/python -m unittest -v test_simulation.py
```

The tests are separate from the teaching workflow and do not run when
`main.py` is executed.

## Optional local Ollama vehicle-control experiment

The standard run remains IDM. To use the isolated local Ollama controller for
vehicle 2 while every other vehicle remains IDM, run from the repository root:

```powershell
.\.venv\Scripts\python.exe car-following-simulation\main.py `
  --experiment 1 `
  --time-max 5 `
  --controller ollama `
  --llm-vehicle-id 2 `
  --llm-model qwen3:4b `
  --llm-decision-interval-steps 50 `
  --llm-timeout-seconds 300 `
  --output evidence\ollama-vehicle-2-exp1.png `
  --no-show
```

Ollama must be installed, running locally, and have the selected model
downloaded. This short run gives vehicle 2 a leader and normally makes one
model request. Increase `--time-max` to study a longer trajectory; a request is
made every 50 simulation steps by default. Local model responses are
synchronous, so each request pauses the simulation until it returns or times
out. Every other vehicle remains IDM to keep inference costs manageable.

The console reports the selected vehicle's request count, IDM fallback count,
and final state. IDM remains the default controller; omitting
`--controller ollama` preserves the usual simulation behavior. Use the same
experiment, duration, and seed when comparing IDM and Ollama runs.
