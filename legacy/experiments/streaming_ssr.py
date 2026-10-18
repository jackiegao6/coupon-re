"""SRR greedy selection retaining only the current coupon component.

Roots persist across rounds. Every RR call has a fresh cached-action world.
The input streams also allow exact comparison with a fully stored reference.
"""
from array import array
from collections import defaultdict
import random
import time

import numpy as np

from uniform_ssr import CouponRR


def stream_ssr_greedy(sampler: CouponRR, k: int, theta: int,
                      root_seed: int, component_seeds):
    """Return distinct seeds and diagnostics, without materializing an SRR bank.

    All theta RR calls are made in each round, including already covered rows.
    Only uncovered memberships of available candidates enter the current index.
    The index is released after the selected node's covered flags are updated.
    """
    if not 1 <= k <= sampler.graph.n or not 0 < theta <= 2**32:
        raise ValueError("Require 1 <= k <= n and 0 < theta <= 2**32")
    if len(component_seeds) != k:
        raise ValueError("One independent stream seed per coupon is required")
    start = time.perf_counter()
    rng = random.Random(root_seed)
    roots = array('I', (rng.randrange(sampler.graph.n) for _ in range(theta)))
    covered = bytearray(theta)
    available = np.ones(sampler.graph.n, dtype=bool)
    selected, chosen_gains = [], []
    max_current_memberships = 0
    sampling_seconds = greedy_seconds = 0.0
    for seed in component_seeds:
        stage = time.perf_counter()
        rng = random.Random(seed)
        index = defaultdict(lambda: array('I'))
        memberships = 0
        for t, root in enumerate(roots):
            rr = sampler.sample(root, rng)
            if not covered[t]:
                for v in rr:
                    if available[v]:
                        index[v].append(t)
                        memberships += 1
        sampling_seconds += time.perf_counter() - stage
        max_current_memberships = max(max_current_memberships, memberships)
        stage = time.perf_counter()
        best = min((int(v) for v in np.flatnonzero(available)),
                   key=lambda v: (-len(index.get(v, ())), v))
        selected.append(best)
        chosen_gains.append(len(index.get(best, ())))
        available[best] = False
        for t in index.get(best, ()):
            covered[t] = 1
        # No references to older component lists survive this round.
        del index
        greedy_seconds += time.perf_counter() - stage
    return selected, dict(selected_marginal_counts=chosen_gains,
                          covered_count=sum(covered),
                          training_estimate_not_quality=sampler.graph.n * sum(covered) / theta,
                          sampling_seconds=sampling_seconds, greedy_seconds=greedy_seconds,
                          selection_seconds=time.perf_counter() - start,
                          max_current_memberships=max_current_memberships,
                          retained_component_count=1, stats=sampler.stats())
