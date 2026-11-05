# LLM Vehicle-Control Coursework Workflow

This workflow connects the coursework brief to the projects in this repository.
It is designed to produce a useful report even if a live LLM cannot be
integrated: learn the simulation, establish a conventional baseline, build up
to a safe LLM experiment, and document evidence and limitations at every step.

Word copy: [LLM_COURSEWORK_WORKFLOW_ISOLATED.docx](docs/LLM_COURSEWORK_WORKFLOW_ISOLATED.docx).

## What is already available

The main coursework project is `car-following-simulation/`:

- `vehicle_idm.py` implements the Intelligent Driver Model (IDM) baseline.
- `simulator.py` generates IDM vehicles, updates all accelerations before
  motion, and records each vehicle's state history.
- `plotting.py` draws speed-colored time-space diagrams.
- `vehicle_llm.py` provides an observation method but deliberately raises
  `NotImplementedError` for acceleration. It is not a working LLM controller.
- The simulator currently creates `IDMVehicle` directly, so adding an LLM
  vehicle also requires a deliberate way to select or inject a controller.

The RNN project is useful background for sequence prediction and model
evaluation, but it is not an LLM and is not connected to the traffic simulator.
Do not present its text generation as LLM-based vehicle control.

## Fit with the repository's recommended learning path

The [repository README's recommended learning path](https://github.com/gotrafficgo/teaching_artificial_intelligence#recommended-learning-path)
starts with the car-following simulation, then uses the CNN to learn about
model training, the RNN to extend those ideas to sequences, and the maze
project to explore Q-learning. This coursework workflow follows that order at
its core: understand the car-following simulator first, because it is the
environment the coursework asks you to extend.

The other projects are optional supporting study, not dependencies for the
vehicle-control experiments. If you have time, the CNN and RNN provide useful
context for learned models, and the reinforcement-learning project illustrates
a different way to learn policies through interaction. The projects are
independent: their models and outputs are not wired into the traffic simulator.
In particular, an RNN that generates text is not an LLM controller. Keep the
coursework focused on the simulator and describe any supporting project only
as background unless you actually connect and evaluate it.

## Stage 1 — Set up and understand the baseline

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m unittest discover -s car-following-simulation -p "test_simulation.py" -v
.\.venv\Scripts\python.exe coursework_tools\collect_stage2_evidence.py --experiment 1 --seed 1
```

If the virtual environment is not available or does not contain the
dependencies, follow the setup instructions in the root `README.md` and the
car-following project README.

Read `config.py`, `vehicle.py`, `vehicle_idm.py`, and `simulator.py` in that
order. Trace one time step: vehicle generation, road-condition update,
acceleration calculation, motion update, and history recording. Note the units
and limits in the configuration, including the 0.1-second simulation step and
the bottleneck speed.

The standalone helper in `coursework_tools/` saves a PNG diagram and an
editable Markdown evidence note under `evidence/`. It does not modify the
simulator. The note records the controller, experiment number, seed, duration,
arrival process, bottleneck settings, diagram path, vehicle count, and average
admitted inflow. Open the diagram and replace the visual-observations prompt
in the note with your own observations. Save the test output and baseline
diagram alongside the report.
Experiments 1 and 3 use fixed arrivals; experiments 2 and 4 use random
arrivals. Experiments 1–2 have a temporary bottleneck, while 3–4 keep it active
to the end of the run.

## Stage 2 — Establish a fair comparison

Use the IDM result as the reference before changing the simulator. Choose one
experiment and keep its settings, seed, road conditions, and duration fixed
for every controller being compared. Begin with experiment 1 for a relatively
simple repeatable setup. Use another experiment only to test whether the
behavior changes under random arrivals or a longer bottleneck.

For each run, record:

| Item | Record |
| --- | --- |
| Run identity | Controller, experiment, seed, simulation duration |
| Configuration | Time step, arrival process, bottleneck settings |
| Outcome | Diagram, vehicle count, observed queues or stop-and-go waves |
| Reliability | Invalid actions, fallback count, model errors, if applicable |
| Runtime | Wall-clock duration and any controller response delay |

The run report includes vehicle count and average admitted inflow. Vehicle
histories are available in memory during a run; if further quantitative
measures are needed, add a small, reproducible analysis step and document its
definition. Useful measures include mean speed, stopped-time fraction, minimum
following gap, and controller-call latency. Do not compare one attractive plot
alone or change several experiment settings at once.

## Stage 3 — Design a constrained LLM experiment

First define a narrow task: given the ego vehicle and its leader, recommend a
bounded longitudinal acceleration. The existing
`LLMVehicle.get_observation()` exposes vehicle ID, position, speed,
acceleration, speed limit, gap, relative speed, and leader speed. Establish
units and edge-case meanings (especially `None` when there is no active
leader) before using those values in a prompt.

Use this progression:

1. **Specify the interface.** Define the observation fields and a strict
   structured response, such as one numeric acceleration in m/s². Document
   the allowed range and behavior for missing or invalid values.
2. **Test without a live model.** Implement a deterministic mock response or
   recorded response fixture and verify the observation-to-action path. This
   separates simulator integration bugs from model/network problems.
3. **Add controller selection.** Introduce a small, explicit way for the
   simulator to create IDM or LLM vehicles. Keep IDM as the default so
   existing runs remain unchanged.
4. **Validate and limit every action.** Reject malformed, non-finite, or
   out-of-range responses; apply the simulation's acceleration and speed
   constraints; and use a documented IDM fallback on timeout or invalid
   output. Log each fallback rather than silently treating it as success.
5. **Account for latency.** Do not call a remote LLM for every 0.1-second
   simulation step. Start with a slower, explicit decision interval or cached
   response, define what happens between decisions, and report the interval,
   elapsed time, and stale-response policy.
6. **Only then try a real model, if available.** Keep credentials outside the
   repository (for example, environment variables), avoid committing secrets,
   and record the model/version and prompt format. Do not send personal or
   otherwise sensitive data; this experiment needs only synthetic vehicle
   observations.

An LLM output is a recommendation, not a safety guarantee. The simulator's
existing acceleration bounds and a tested fallback should remain authoritative.
If a live model, API access, or time is unavailable, the mock-policy experiment
and an honest integration analysis are valid coursework progress; label them
clearly and do not claim that a mock is an LLM.

## Stage 4 — Test, compare, and interpret

Before interpreting a plot, check that the controller can run through a short
simulation, returns finite actions, respects motion constraints, and handles
no-leader observations. Add tests for malformed responses, action clipping,
fallback behavior, and reproducibility where applicable. Re-run the existing
simulation tests to check that the IDM path remains unchanged.

Run IDM and the candidate controller with the same experiment and seed.
Compare their time-space diagrams and the same quantitative measures. Repeat
the comparison for multiple seeds if the controller or arrival process is
stochastic. Report failed runs and fallbacks as results, not as runs to omit.
Discuss the trade-offs among behavior, stability, inference delay, cost, and
reproducibility; avoid concluding that a controller is safer based only on a
visually smoother diagram.

## Stage 5 — Turn the work into the report

Use the coursework's required sections and build each from the saved evidence:

1. **Introduction and objectives:** state the intended LLM-control task and
   what success was feasible.
2. **Methodology and implementation:** describe the IDM framework, the
   observation/action interface, controller-selection changes, tools, and
   whether the policy was a mock or a live LLM.
3. **Experiments and results:** list exact settings, show labeled diagrams,
   summarize metrics and runtime, and include small relevant code excerpts.
4. **Challenges and problem-solving:** explain concrete integration obstacles
   (for example, direct IDM construction, strict action parsing, and inference
   latency), what you tried, and what worked or did not.
5. **Personal reflection and conclusion:** distinguish observed evidence from
   speculation, discuss limitations, and propose the next experiment.

For every figure, include its controller, experiment, seed, and duration.
Keep a brief dated experiment log while working; this makes unsuccessful
attempts and decisions easier to report accurately.

## Completion checklist

- [ ] Existing IDM tests pass and an unchanged baseline diagram is saved.
- [ ] The simulation loop and IDM behavior are understood well enough to
  explain them.
- [ ] Baseline and candidate settings are controlled and recorded.
- [ ] Any LLM integration is clearly identified as mock, recorded, or live.
- [ ] Actions are validated, bounded, and have a tested fallback.
- [ ] Latency, failures, and limitations are reported alongside behavior.
- [ ] The final report covers all five required sections and includes evidence.
