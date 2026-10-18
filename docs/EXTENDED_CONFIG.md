# Extended experiment configuration

Built-in scenario strings still work. A custom scenario is a JSON object with
a unique `name` and one of these specifications:

| kind | Parameters | Meaning |
| --- | --- | --- |
| `degree` | `a0, a1, d0, d1` | a=a0+a1(1-z), d=d0+d1*z |
| `heterogeneity` | `scale`, `association`, `permutation_seed` | Scale centered balanced-reference consumption probabilities; optionally permute them |
| `transfer-share` | `transfer`, `share`, optional `degree_effect=.2` | Appendix grid: a=(1-t)*r[u], d=(1-t)*(1-r[u]) |
| `constant` | `alpha`, `discard` | Identical node probabilities before sink adjustment |

For heterogeneity, `association` is `degree` or `permuted`. Permutations are
fixed by the declared seed, not redrawn between methods or selection repetitions.
Scale=3 multiplies variance by 9; it does not multiply the mean. Invalid
probabilities raise an error. The transfer-share grid explicitly clips
`r[u]=share+degree_effect*(.5-z[u])` to [.05,.95], matching the old appendix
protocol. Its z uses [0,1]; the other degree settings use [1e-5,1].
Sinks always have transfer zero and dropping 1-alpha.

The optional `algorithm_options` object accepts:

| Key | Default |
| --- | ---: |
| `stochastic_epsilon` | 0.1 |
| `matrix_max_bytes` | 268435456 |
| `exact_max_nodes` | 16 |
| `adaptive_initial_samples` | 100000 |
| `adaptive_minimum_samples` | 100000 |
| `adaptive_maximum_samples` | 3000000 |
| `adaptive_validation_trials` | 5000 |
| `adaptive_tolerance_percent` | 0.5 |

The adaptive sampler appends to a shared sample prefix, doubles the target,
reselects, and compares previous/current assignments on paired validation
campaigns. It stops at stability after the minimum, or at the maximum.
The former graph-derived sample-floor heuristic is retained in the historical
driver; the portable interface makes the minimum explicit and reproducible.
Small examples override these defaults to complete quickly.

`theta_factors` applies to CIM-RIS, CIM-RIS-Bank, CIM-Conditioned and IC-RIS.
CIM-Adaptive uses its own declared sample limits. `forward_batches` is the
trajectory count per candidate source for Forward-CELF, Forward-Stochastic,
MC-Greedy and CUM-Rank. Each applicable multiplier/configuration produces a
separate isolated job; repeated theta runs do not share nested samples.

## Import your graph

Use edge lists directly, or convert once:

```sh
python -m couponim convert --input your-edges.txt --output data/your-graph.npz --undirected --nodes 10000
python -m pip install ".[data]"
python -m couponim convert --input your-graph.mat --input-format mat --matrix-key network --output data/from-mat.npz --directed
```

Rows are source nodes and columns are target nodes. Input must be unweighted
binary adjacency. MAT loading uses SciPy and does not support MATLAB v7.3/HDF5.
No data is downloaded or included. Conversion refuses to overwrite a file.

## Results and timing

All selection records now identify `selection_backend`, including hybrid
C++-sampling/Python-greedy methods. Raw output also exposes sampled whole-job
process-tree RSS. That includes evaluation and may miss instantaneous peaks;
it is **not** the native selector's peak private committed memory.

`reevaluate` freezes its own source hashes and input-result hashes, checks graph
file integrity, and evaluates saved seeds on new deterministic random streams.
It writes `reevaluated.csv` and per-job JSON in a separate output directory.
No selection time is measured again and no selection is rerun. Resume with the
same reevaluation command reuses completed evaluations. The reevaluation command
supports process parallelism but not the main runner's STOP-file/time-limit
controls; use `run` for a scheduled selection-and-evaluation study.

Graph and exact diagnostics require a new/empty output directory. Only the
exact diagnostic enumerates subsets, and only below its stated size cap.
