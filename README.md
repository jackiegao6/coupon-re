# CouponIM-experience

Python and C++ implementations of **coupon influence maximization (CIM)**,
including streaming SRR selection, baseline methods, independent evaluation,
and resumable experiment scheduling. The objective is to maximize the expected
number of distinct coupon consumers.

## Quick start

Requires Python 3.10+. Run from this directory:

```sh
python -m pip install -r requirements.txt
python -m couponim demo --output results/demo-python
```

The demo generates a small graph and runs CIM-RIS plus six baselines.
No external dataset is needed. Results are saved as `raw.csv`, `summary.csv`,
and per-job JSON records containing seeds and diagnostics.

For C++ acceleration, install a C++17 compiler (`g++` or `clang++` on PATH):

```sh
python -m couponim demo --backend cpp --output results/demo-cpp
```

The native kernels are compiled automatically.

## Custom experiments

Edit `examples/external-graph.template.json` to specify your graph,
methods, coupon budgets, sampling budgets, and evaluation trials:

```sh
python -m couponim run --config examples/external-graph.template.json --backend cpp --workers 2 --output results/my-study
```

Use `--workers 1` for serial execution. Repeating the same command reuses
completed jobs and saved selection checkpoints. Graph paths are relative to the
configuration file. Edge-list and CSR NPZ inputs are supported; benchmark
datasets and plotting code are not included.

## Full experiment suite

The runnable package supports **15 methods**, uniform/full-bank/conditioned/adaptive
sampling, parameter and budget sweeps, exact small-graph diagnosis, sampler
ablation, and independent re-evaluation of saved seeds.

```sh
python -m couponim methods
python -m couponim run --config examples/all-methods.json --backend cpp --workers 2 --output results/all-methods
```

See the [code and experiment catalog](docs/CATALOG.md) for every method and entry
point, and [extended configuration](docs/EXTENDED_CONFIG.md) for parameter grids.

## Core code

- `couponim/rr.py`, `streaming.py`: RR sampling and streaming SRR selection.
- `couponim/cpp/`: native sampling, selection, and evaluation kernels.
- `couponim/forward.py`, `heuristics.py`: baseline implementations.
- `couponim/runner.py`: scheduling, checkpoints, and CSV aggregation.

## Tests

```sh
python -m unittest discover -s tests -v
```

Set `COUPONIM_TEST_CPP=1` to also test Python/C++ equivalence.

See [algorithm details](docs/ALGORITHM.md),
[experiment settings](docs/EXPERIMENTS.md), and
[中文说明](README.zh-CN.md).
