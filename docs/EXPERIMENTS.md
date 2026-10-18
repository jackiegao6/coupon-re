# Experiment protocol and output schema

## Configuration

All graph inputs are shared across methods. No graph or method results are
silently substituted. A trial is one complete forward campaign with one coupon
per selected seed. By default there are 5,000 independent trials per selected set.

| Key | Meaning |
| --- | --- |
| `graphs` | List of graph specifications; names must be unique |
| `scenarios` | Behavior settings applied identically to every method |
| `budgets` | Positive k values, at most the graph's node count |
| `theta_factors` | Positive integer multipliers f, with theta = f*n, for CIM-RIS |
| `methods` | Methods to execute; listed in the main README |
| `repeats` | Independent selection/evaluation repetitions per configuration |
| `evaluation_trials` | Forward campaigns for each selected set |
| `forward_batches` | Forward trajectories per source for Forward-CELF |
| `seed` | Master seed for deterministic, domain-separated stream derivation |

Forward-CELF uses one fixed empirical coverage bank over `(batch, consumer)` pairs.
CELF is ordinary greedy on that bank with lazy marginal evaluation. It is a
simulation-based baseline, not an exact oracle or a mathematical upper bound.
The runner does not claim its training coverage is an unbiased evaluation of a
sample-selected seed set. All selected sets receive fresh independent evaluation.

Changing k, theta, method, repetition, or behavior produces different job IDs and
streams. Repeated execution of the identical configuration is reproducible.
Different theta points are not nested samples. Their measured spread need not be
monotone with increasing theta. Default sampler budget bounds are
`1 <= theta < 2^32` and `1 <= k <= n < 2^31`.

## Behavior settings

Let `z[u] = clip(log(1+outdegree[u]) / max(max_v log(1+outdegree[v]), 1), 1e-5, 1)`.
This preserves the normalization in the original implementation.

For three base settings, `a[u] = a0 + a1*(1-z[u])`, `d[u] = d0 + d1*z[u]`:

| Scenario | a0 | a1 | d0 | d1 |
| --- | ---: | ---: | ---: | ---: |
| balanced-reference | 0.30 | 0.10 | 0.10 | 0.10 |
| adoption-heavy | 0.45 | 0.35 | 0.15 | 0.05 |
| forwarding-heavy | 0.01 | 0.05 | 0.01 | 0.05 |

The higher-variance `balanced` setting starts with the balanced-reference a[u]
and sets `a_new[u] = mean(a) + 3*(a[u]-mean(a))`, `d_new[u] = 0.5-a_new[u]`.
For non-sinks this keeps transfer at 0.5, retains mean consumption, and multiplies
consumption variance by nine. Invalid probabilities raise an error; they are not
clipped. In every setting sinks override dropping to `1-a[u]` and transfer to zero.

## Files

`protocol.json` freezes configuration, graph hashes, core source hashes, software
versions, and (for C++) compiler information. Resuming with changed inputs is
rejected. All job results remain in JSON, and only completed results enter CSVs.
No missing value is filled using forecasts or interpolated observations.

`raw.csv` has one row per completed job:

- `dataset`, `scenario`, `k`, `method`, `replicate`, `theta_factor`, `backend`;
- `spread`: sample mean distinct consumers, sigma(S);
- `sigma_over_k`: `spread/k`;
- `consumed_coupons`: sample mean redeemed coupons, U(S);
- `repeat_redemptions`: U(S)-sigma(S), measured in the same campaigns;
- `within_run_ci95_halfwidth`: 1.96 times the estimated standard error across
  evaluation campaigns (a normal approximation, not a between-run Student-t CI);
- `selection_seconds`: wall time for preparation, sampling and selection, excluding
  initial graph loading, compilation, and independent evaluation;
- `native_peak_private_bytes`: native process's OS peak private committed memory
  on Windows, blank on unsupported platforms/backends;
- `native_peak_rss_bytes`: native selector's peak resident set/working set, separate
  from private commit. Both native peaks include its graph representation.

`summary.csv` groups runs by dataset, scenario, k, method and theta multiplier.
It records the number of completed runs and arithmetic means of spread,
redemptions, and selection time. `std_across_run_means` is blank when only one run
exists. Incomplete repeated experiments therefore have fewer runs, visibly recorded.

`jobs/*.json` includes actual seeds, independent selection/evaluation seeds,
the selected training coverage, sampler counters, selected marginal counts,
per-stage timings, and native metrics where applicable. Training coverage is named
`training_estimate_not_quality`. Native algorithm time and graph-loading time are
also retained so they are not confused with end-to-end selection time.

`execution/*.json` records job exit status, return code, worker count, whole-job
wall time and sampled process-tree peak RSS. That RSS includes Python, any native
child, graph loading, selection, and evaluation, and can miss short-lived peaks;
it is not interchangeable with OS-reported native peak private memory.

Use serial runs on the same environment for runtime comparison. The coordinator
marks concurrent execution in `execution.json`. Timings are observations under
that execution regime, not asymptotic complexity claims.

## Scope

This standalone runner cleans up paths, packaging, and orchestration around the
existing core. It does not include old plotting pipelines, result modifications,
or automatic downloads. Source-only tests generate their own small graphs.
Reproducing a specific paper figure additionally requires the original graph,
exact protocol, and seeds for that figure.
