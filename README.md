# GLA Supply Simulator

A stock-and-flow simulation of the London housing supply pipeline. Homes move
through four stages — **in planning → approved → started → completed** — and the
model projects quarterly completions forward from a given starting position.

It exists to answer questions of the form *"if we changed X at time T, what
happens to completions in five years?"* — where X is an approval rate, a
build-out rate, an application volume, or the share of schemes that survive each
stage.

> ### Relationship to `home_pipeline`
>
> This is the **earlier** of two supply simulators. It models the pipeline as
> aggregate *volumes* flowing between compartments at average rates.
>
> `home_pipeline` is the later, more detailed reimplementation: it simulates
> **individual sites** with their own size, phase durations and stall states,
> and is calibrated against Planning London Datahub records.
>
> Neither supersedes the other outright. This one is faster to reason about,
> easier to explain in a meeting, and its parameters map directly onto things
> people already talk about (approval rates, build-out times). `home_pipeline`
> is better where site size distribution and stalling matter. Both are kept.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens with three development threads pre-configured. You will
need `data/historic.csv` to see the historic comparison charts — see
**Getting the data**.

```bash
python3 -m pytest tests/ -q      # 3 tests covering the container flow accounting
```

---

## How the model works

The pipeline is a chain of four SimPy `Container`s. Each quarter, every stage
passes a fraction of its stock to the next one:

```
                 application_rate (absolute homes/quarter)
                          │
                          ▼
                  ┌───────────────┐
                  │  IN PLANNING  │
                  └───────────────┘
                          │  level × (1 / planning_time) × planning_success_rate
                          ▼
                  ┌───────────────┐
                  │   APPROVED    │
                  └───────────────┘
                          │  level × (1 / start_time) × approved_to_start_rate
                          ▼
                  ┌───────────────┐
                  │    STARTED    │
                  └───────────────┘
                          │  level × (1 / completion_time) × start_to_completion_rate
                          ▼
                  ┌───────────────┐
                  │   COMPLETED   │
                  └───────────────┘
```

Two kinds of parameter govern each transition:

- **A duration** (`planning_time`, `start_time`, `completion_time`), in quarters.
  Its reciprocal is the share of the stock that moves on each quarter, so this
  behaves as a mean dwell time with an exponential tail — not a fixed lag.
- **A success rate** (`planning_success_rate`, `approved_to_start_rate`,
  `start_to_completion_rate`). The shortfall is attrition: homes that leave the
  stage without arriving at the next one.

### Threads

The simulation runs several **threads** in parallel and sums them. Each has its
own parameters and initial stocks. The three defaults are calibrated separately
because they behave quite differently:

| Thread | Application rate | Planning | Start | Completion | Notes |
|---|---|---|---|---|---|
| Large Private Sites | 6,455/q | 9.3q | 9.4q | 28.3q | The bulk of supply |
| Small Private Sites | 1,760/q | 9.3q | 10.8q | 24.9q | Slower to start, faster to build |
| Public Sites | 1,289/q | 9.3q | 15.5q | 25.6q | Near-100% approval and start rates, but slowest to get going |

Defaults live in `src/config.py` and are all editable in the sidebar.

### Policies

The **Policy Editor** tab applies timed changes to parameters. A policy has a
start quarter, a target parameter, a target thread (or all threads), and either
an `Absolute` increment or a `Multiply` factor. It takes effect from its start
quarter onward and stays in effect.

This is the mechanism for modelling interventions — e.g. "from 2026 Q3,
multiply `approved_to_start_rate` by 1.15 for Large Private Sites".

### Presets

Named parameter sets are saved to `data/presets.json`. The ones shipped include
several Molior-based calibrations and an `Emergency measures effect
(Business LDN)` scenario.

---

## Repository layout

```
app.py                  Streamlit UI — sidebar, policy editor, charts
src/
  config.py               DEFAULT_THREADS — the calibrated starting parameters
  containers.py           MonitoredContainer — SimPy Container that records
                          its own inflow/outflow/level history per quarter
  pipeline.py             HousingPipeline — the four containers and the
                          SimPy processes that move stock between them
  simulation.py           Quarter labelling, policy application, run loop
  presets.py              Save/load/delete named scenarios
  historic.py             Load and align the historic actuals series
tests/
  test_monitored_container.py   Flow accounting for MonitoredContainer
data/
  historic.csv            Historic actuals 2016– (GITIGNORED, see below)
  presets.json            Saved scenarios
_archive/                 Superseded caches (GITIGNORED)
```

### Why `MonitoredContainer` exists

SimPy's stock `Container` tracks only a current level. The dashboard needs
per-quarter *flows* as well as stocks, so `MonitoredContainer` overrides `get()`
and `put()` to append to `inflows`, `outflows` and `levels` lists as it goes.

Note the indexing convention in `get_quarterly_level()`: it reads
`levels[quarter * 2]`, because each quarter produces exactly two level entries
(one `get`, one `put`). This is fragile — it silently assumes every container is
touched exactly twice per quarter. If you add a flow into or out of a stage,
this breaks. The tests in `tests/test_monitored_container.py` pin the current
behaviour.

---

## Data

Everything needed to run the model is committed. There is nothing to fetch.

- `data/historic.csv` — quarterly applications, approvals, starts and
  completions from 2016, used for the historic comparison overlay and for
  seeding initial stocks from actuals:

  ```
  Year + Q, Applications, Approvals, Starts, Completions
  2016 Q1,  7619,         6053,      6233,   4705
  ```

- `data/presets.json` — saved scenarios.

> **This repository is internal.** The historic series is Molior-derived and
> Molior is a subscription product. That is fine inside the GLA, but it means
> this repository must not be pushed to a public remote or shared outside the
> organisation without checking the licence position first.

---

## Known issues and things to watch

- **The `levels[quarter * 2]` convention** described above is the most likely
  thing to break if anyone extends the flow structure.
- **Exponential dwell times.** Because each stage passes a *fraction* of its
  stock on each quarter, homes do not spend a fixed time in a stage — a few
  leave almost immediately and a tail stays for a long time. For questions
  sensitive to the shape of the build-out profile rather than its average, use
  `home_pipeline`, which samples explicit durations and applies an S-curve.
- **No stalling.** Sites cannot pause and resume; attrition is permanent loss
  only. This was a large part of the motivation for `home_pipeline`.
- **Calibration provenance is partly undocumented.** The default rates came from
  Molior and GLA planning data, but the derivation was not written up at the
  time. The preset names are the best surviving record of what was tried.
- The commit history is candid (`trying to fix the whole thing`,
  `finally fixed flows etc etc`). Those messages date from a period when the
  flow accounting was wrong; `78bf193 replaced simpy.Containers with
  MonitoredContainers` is where it was fixed.
