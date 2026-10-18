# Code and experiment catalog

The public `couponim` package is the runnable implementation. `legacy/` contains
71 source snapshots from the earlier project for reference. These two scopes
are deliberately documented separately: archived scripts still refer to their
original paths/dependencies and are not standalone commands.

## Implementations

| Method / capability | Entry point | Backend |
| --- | --- | --- |
| Uniform-root SRR, one component at a time | `streaming.py`; `CIM-RIS` | Python / C++ |
| Uniform-root SRR, full row-major bank | `rr.py`; `CIM-RIS-Bank` | Python / C++ |
| Conditioned roots and nonempty root gates | `conditioned.py`; `CIM-Conditioned` | Python |
| Incremental conditioned samples, validation stopping | `adaptive.py`; `CIM-Adaptive` | Python |
| Fixed forward coverage and lazy greedy | `forward.py`; `Forward-CELF` | Python / C++ sampling |
| Fixed forward coverage and stochastic greedy | `forward.py`; `Forward-Stochastic` | Python / C++ sampling |
| Monte Carlo consumption matrix + greedy | `comparators.py`; `MC-Greedy` | Python / C++ sampling |
| Rank total single-coupon consumption probability | `comparators.py`; `CUM-Rank` | Python / C++ sampling |
| Ordinary IC RR coverage, evaluated under CIM | `comparators.py`; `IC-RIS` | Python |
| Random, Degree, PageRank, Alpha, 1Hop | `selection.py`, `heuristics.py` | Python |
| Visited-set exact DP and greedy | `exact.py`; `Exact-Greedy` | Python, small graphs |
| Exact subset optimum and uniform-random expectation | `exact.py`; `diagnose` | Python, small graphs |
| Fixed-seed sampling bias/variance/empty-rate diagnosis | `diagnostics.py`; `sampler-ablation` | Python |
| Independent forward evaluation and redemptions | `simulation.py`, `native.py` | Python / C++ |
| Verified full-bank seed-prefix extension | `NativeModel.extend_bank_prefix` | C++ |
| Graph statistics and input conversion | `diagnostics.py`, `convert.py` | Python; optional SciPy for MAT |
| Job isolation, parallelism, resume, timeout, memory monitoring | `runner.py` | Python process scheduler |
| Fresh evaluation of stored seed sets | `reevaluate.py` | Python / C++, parallel |

`python -m couponim methods` prints the 15 accepted method names.
The default demo still runs seven methods; it remains a quick installation check.
With `--backend cpp`, unsupported selection methods explicitly use Python.
Every job records `selection_backend`; final evaluation can still use C++.
No C++ version of conditioned/adaptive or IC selection is claimed.

## Experiment families

The original figure-number-specific drivers are represented by reusable
configurations, with no plotting stage. All supplied configurations generate
small graphs at runtime. Replace their graph lists with your own input specs
for substantive experiments.

| Experiment | Configuration / command | Earlier scripts |
| --- | --- | --- |
| All available methods | `examples/all-methods.json` | submission and baseline pilot suites |
| Coupon-budget sweep | `examples/budget-sweep.json` | budget_extension, figure2/3 budget extensions |
| Behavior environments | `examples/behavior-sweep.json` | figure3, uniform_ssr_directed |
| Heterogeneity and degree/permuted assignment | `examples/heterogeneity.json` | heterogeneity_pilot, figure2_heterogeneity |
| Transfer/redemption-share grid | `examples/parameter-grid.json` | appendix_parameter_sweep |
| Adaptive sample growth | `examples/adaptive.json` | adaptive_parameter_sweep |
| Sampling-budget / time / memory sweep | `examples/sampling-sensitivity.json` | figure6 and streaming_sensitivity suites |
| Streaming vs full bank | `examples/streaming-vs-bank.json` | streaming and full-bank validators |
| Selection repetition/stability | `examples/selection-stability.json` | selection_seed_stability |
| Graph-size scaling | `examples/scalability.json` | validated_scalability, large_graph_quality |
| Fixed-seed sampler comparison | `sampler-ablation --config examples/sampler-ablation.json` | real_sampler_ablation |
| Exact diagnosis | `diagnose --config examples/exact-diagnosis.json` | exact_selection_diagnosis |
| Distinct consumers vs redemptions | Any `run`: sigma, U and U-sigma are evaluated together | figure4/5 suites |
| Re-evaluate existing selections | `reevaluate --source ... --trials ...` | saved-seed evaluation extensions |
| Per-run source/protocol/resource audit | `protocol.json`, `jobs/`, `execution/` | audit_sample_scaling, record_figure2_costs |

Run a configuration using:

```sh
python -m couponim run --config examples/all-methods.json --backend cpp --workers 2 --output results/all-methods
python -m couponim diagnose --config examples/exact-diagnosis.json --output results/exact
python -m couponim sampler-ablation --config examples/sampler-ablation.json --output results/ablation
python -m couponim inspect --config examples/scalability.json --output results/graph-stats
python -m couponim reevaluate --source results/all-methods --output results/new-evaluation --trials 5000 --backend cpp --workers 2
```

Ablation uses a fixed Alpha-selected assignment for each graph/behavior/k, then
repeats independent uniform and conditioned estimates. `empty_fraction`,
membership count, time and estimates are recorded per repetition. It does not
optimize a new assignment for each sample batch.

## Budget extension API

The native kernel can replay and check an existing **row-major full-bank**
selection and then generate extra component columns on the same roots:

```python
from couponim.native import NativeModel, build

build("build")
model = NativeModel(graph, alpha, discard, "build", "results/extension-work")
old_seeds, info = model.bank(k=5, theta=1000, seed=42)
new_seeds, audit = model.extend_bank_prefix(
    theta=1000, original_seed=42, selected=old_seeds,
    extra_component_seeds=[1001, 1002, 1003],
)
assert audit["prefix_reproduced"]
```

This extends the coupon budget, not the number of sample rows. It requires the
exact graph, probabilities, theta and old row-major seed. Passing a streaming
selection here is not a supported checkpoint conversion. Incomplete current
rounds are not resumable; completed selections are.

## Model boundaries

- Coupon trajectories discard on the first revisit. Exact DP carries a visited
  set; an ordinary absorbing-Markov solve is not substituted on cyclic graphs.
- Uniform and conditioned SRR are different sampling distributions with
  different normalization factors. Labels remain distinct in all CSVs.
- The adaptive stopping rule is empirical validation stability, not a proven
  sample-complexity stopping certificate. Final evaluation is independent.
- Dense MC matrices have an explicit byte limit (default 256 MiB). Exact DP has
  a node limit (default 16), and subset enumeration has a 100,000-subset limit.
- Streaming and row-major full-bank implementations use different RNG layouts.
  They are distribution-equivalent; the same scalar seed does not mean identical
  sampled worlds. Tests compare matched worlds when asserting exact equality.
- Python MC-matrix sampling retains historical per-source RNG streams. Native
  MC-matrix construction uses the sequential forward-bank stream; results are
  not required to be seedwise identical between those two estimators.
- Multiple coupons per seed are available in lower-level
  `forward_coverage_seeds(..., capacity_per_node=...)`,
  `cim_ris_seeds(..., capacity_per_node=...)` and `mc_greedy_order`.
  The public experiment runner enforces distinct seeds, as in the current paper.
- Numerical modifications to plotted results, forecasts, mock-result generators
  and all drawing scripts are outside this code distribution.

See [source inventory](source-inventory.json) for hashes and the disposition of
103 original source files, and [legacy notes](../legacy/README.md) before using
historical code.
