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

## Optional DeepSeek vehicle-control experiment

The simulation uses IDM by default and makes no API calls unless
`--controller deepseek` is selected. By default DeepSeek controls vehicle 2
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

The controller requests an acceleration recommendation every 200 simulation
steps in the example (20 simulated seconds at a 0.1-second timestep). API
responses are synchronous and usage is billed by tokens; consult
[DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing) before
long experiments. As a rough planning estimate for a 400-second Experiment 1
run at that interval, the IDM baseline vehicle lifetimes imply about 147
requests for IDs 2-11 and about 785 for IDs 2-51. These are estimates, not
guaranteed totals; DeepSeek decisions may change vehicle lifetimes and traffic
generation. The controller limits actions to the simulator's acceleration
bounds and falls back to IDM if a request or response fails. This is a
coursework simulation, not a safety-certified real-vehicle controller.

To turn DeepSeek off, omit `--controller deepseek` (IDM is the default). The
offline tests for the controller use mocked API responses and do not incur API
charges.
