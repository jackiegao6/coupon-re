# Algorithm and implementation

## Diffusion and reverse sets

On arrival at u, a coupon consumes with probability a[u], drops with d[u],
or transfers to exactly one outgoing neighbor. Sinks have zero transfer
probability. A first revisit discards the coupon. Different coupons make
independent choices, including when they start or arrive at the same node.

`CouponRR.sample(root, rng)` draws a cached action for each queried node in a
single coupon world. Consumption occupies `[0,a)`, dropping `[a,a+d)`, and
transfer edges have disjoint intervals partitioning the remaining unit interval.
The root draw is retained: if it does not consume, the RR set is empty. Otherwise
incoming edges are searched using the same cached draw whenever a node is queried
again. A consuming root cannot simultaneously select a transfer edge. Node
timestamps avoid clearing an n-element cache before each sample.

The RR set contains exactly the sources whose coupon would be consumed by this
root in that fixed world. Empty sets are valid samples and remain in the denominator.
The Python RR API accepts explicit edge probabilities for verification or custom
reverse sampling. The supplied forward evaluator and CLI use uniform-neighbor
forwarding; do not combine weighted RR input with that evaluator.

## SRR rows, indexed coverage, and streaming

Sample theta independent uniform roots `u[t]`. Row t represents an SRR tuple
`(R[1,t], ..., R[k,t])`. All its components have the same root and independent
coupon worlds. Root values can repeat across different independent rows.

At round j:

1. Sample the current component R[j,t] for every row t using its own cached-action world.
2. For each still available vertex v, store uncovered row IDs containing v.
3. Choose the unselected vertex with the most stored IDs. Break ties by smallest node ID.
4. Mark those rows covered in the shared persistent coverage array.
5. Release this component's entire inverted index and proceed to j+1.

Previous components are no longer needed; future components have not yet been
generated. A covered row contributes zero to every later gain. The current
implementation still generates its later RR sets before skipping their storage,
preserving the random-number consumption of the existing full-bank reference.
Skipping covered-row sampling is a separate optimization and is not enabled here.

Seeds cannot repeat. Fixed-sample coverage is indexed: seed s[j] tests only
R[j,t], and each SRR row contributes at most one unit. This is not k independent
ordinary maximum-coverage problems. The full-bank reference is useful for exact
implementation checks, not large-scale experiments.

## Cost

Let B[j] be the number of uncovered, available memberships retained at round j.
Resident algorithm storage is `O(n + m + theta + max_j B[j])`. It includes the
graph, incoming index, cached draws, persistent roots and coverage, and only one
component index. Native vector capacity and allocation overhead add constants.
It does not scale with the sum of all k component indices, but is not constant
in theta. Python objects have larger constants than the native arrays.

There are exactly `theta * k` RR calls. Worst-case sampling takes
`O(theta * k * (n + m))`; selecting the largest gain and freeing the index add
`O(k*n + sum_j B[j])`. The actual scanned-edge and membership counters are saved.
Streaming reduces retained storage; it does not remove the k factor from sampling.

The helper `sufficient_budget` implements the existing Bernstein/union-bound
sufficient budget. Its `lower_bound` must be a valid positive lower bound on the
optimum for the caller's model. The normal experiment runner uses explicit
`theta_factor` values and does not claim those choices meet a theoretical guarantee.

## Native implementation and verification

The C++ kernel reads the root and component MT19937 states initialized by
CPython, reproduces its 53-bit uniform draws and `randrange` rejection sampling,
and uses the same interval endpoints and traversal order as Python. No fast-math
or floating-point contraction is enabled by the provided build configuration.

`coupon_stream` can write completed prefix checkpoints, but the public scheduler
currently requests only each job's final k. It can reuse a completed selection
for evaluation after interruption. It does not resume inside an unfinished
component or import historical workstation result directories.

Independent validation includes exhaustive small-world forward/reverse equivalence,
weighted intervals, branch merging, cyclic walks, exact full-bank/streaming choices,
and Python/native forward evaluation. The native bridge currently requires
little-endian CPython hosts. Windows native compilation and execution are tested
locally; POSIX memory handling is implemented but should be tested on its target host.
