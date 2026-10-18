"""Uniform-root, unconditional RR/SSR sampler for paper-online-v1 Section 5.

One cached uniform variate per node and coupon world; independent worlds
per RR call; empty components remain in the sample count. The historical
conditioned-root implementation in run_real_submission.py is not changed.
"""
from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass
import math
import random
import time

import numpy as np

from run_real_submission import Graph


class CouponRR:
    """Lazy possible worlds with CSR incoming edge IDs and cache timestamps.

    Optional edge probabilities are unconditional transfer probabilities,
    aligned with graph.indices; by default forwarding is uniform over neighbors.
    Real-valued uniforms use [0, 1), which is distributionally equivalent to
    continuous Uniform[0, 1] in the manuscript, with disjoint half-open bins.
    """

    def __init__(self, graph: Graph, alpha, discard, edge_probabilities=None):
        self.graph = graph
        self.alpha = np.asarray(alpha, dtype=float).copy()
        self.discard = np.asarray(discard, dtype=float).copy()
        if self.alpha.shape != (graph.n,) or self.discard.shape != (graph.n,):
            raise ValueError("One consumption and dropping probability per node required")
        if (not np.isfinite(self.alpha).all() or not np.isfinite(self.discard).all()
                or np.any(self.alpha < 0) or np.any(self.discard < 0)
                or np.any(self.alpha + self.discard > 1 + 1e-12)):
            raise ValueError("Invalid node probabilities")
        transfer = np.maximum(0., 1. - self.alpha - self.discard)
        if np.any(transfer[graph.degrees == 0] > 1e-12):
            raise ValueError("Nodes without outgoing edges must have zero transfer probability")
        self.sources = np.repeat(np.arange(graph.n, dtype=np.int32), graph.degrees)
        if edge_probabilities is None:
            self.probabilities = transfer[self.sources] / graph.degrees[self.sources]
        else:
            self.probabilities = np.asarray(edge_probabilities, dtype=float).copy()
        if (self.probabilities.shape != (graph.m,)
                or not np.isfinite(self.probabilities).all()
                or np.any(self.probabilities < 0)):
            raise ValueError("Invalid edge probabilities")
        row_sums = np.bincount(self.sources, weights=self.probabilities, minlength=graph.n)
        if not np.allclose(row_sums, transfer, atol=1e-12, rtol=1e-12):
            raise ValueError("Outgoing edge probabilities must sum to transfer probability")
        self.low = np.empty(graph.m)
        self.high = np.empty(graph.m)
        for v in range(graph.n):
            lo, hi = int(graph.indptr[v]), int(graph.indptr[v + 1])
            if lo == hi:
                continue
            ends = self.alpha[v] + self.discard[v] + np.cumsum(self.probabilities[lo:hi])
            self.low[lo:hi] = np.r_[self.alpha[v] + self.discard[v], ends[:-1]]
            self.high[lo:hi] = ends
            # Avoid a floating-point gap at the last positive interval.
            positive = np.flatnonzero(self.probabilities[lo:hi] > 0)
            if len(positive):
                self.high[lo + positive[-1]] = 1.
        counts = np.bincount(graph.indices, minlength=graph.n)
        self.inptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self.inedges = np.empty(graph.m, dtype=np.int64)
        cursor = self.inptr[:-1].copy()
        for edge, target in enumerate(graph.indices):
            self.inedges[cursor[target]] = edge
            cursor[target] += 1
        self.cache_epoch = np.zeros(graph.n, dtype=np.int64)
        self.member_epoch = np.zeros(graph.n, dtype=np.int64)
        self.draws = np.empty(graph.n)
        self.epoch = 0
        self.calls = self.nonempty = self.memberships = self.scanned_edges = 0
        self.draw_count = self.max_size = 0

    def sample(self, root: int, rng: random.Random, fixed_draws=None) -> list[int]:
        """Generate exactly one unconditioned RR set, including an empty result.

        fixed_draws is only for exhaustive world-level verification. A fixed
        world supplies a draw for every node; it does not change traversal.
        """
        if not 0 <= root < self.graph.n:
            raise ValueError("Root is outside graph")
        self.calls += 1
        self.epoch += 1
        epoch = self.epoch
        draw = rng.random() if fixed_draws is None else float(fixed_draws[root])
        self.draw_count += 1
        self.draws[root] = draw
        self.cache_epoch[root] = epoch
        if draw >= self.alpha[root]:
            return []
        queue = [root]
        self.member_epoch[root] = epoch
        self.nonempty += 1
        cursor = 0
        while cursor < len(queue):
            v = queue[cursor]
            cursor += 1
            self.scanned_edges += int(self.inptr[v + 1] - self.inptr[v])
            for pos in range(int(self.inptr[v]), int(self.inptr[v + 1])):
                edge = self.inedges[pos]
                w = int(self.sources[edge])
                if self.cache_epoch[w] != epoch:
                    self.draws[w] = rng.random() if fixed_draws is None else fixed_draws[w]
                    self.cache_epoch[w] = epoch
                    self.draw_count += 1
                r = self.draws[w]
                if (self.low[edge] <= r < self.high[edge]
                        and self.member_epoch[w] != epoch):
                    self.member_epoch[w] = epoch
                    queue.append(w)
        size = len(queue)
        self.memberships += size
        self.max_size = max(self.max_size, size)
        return queue

    def stats(self):
        return {key: int(getattr(self, key)) for key in (
            "calls", "nonempty", "memberships", "scanned_edges", "draw_count", "max_size")}


@dataclass
class SSRBank:
    n: int
    k: int
    theta: int
    index: list
    root_counts: np.ndarray
    empty_ssr: int
    stats: dict
    sampling_seconds: float

    def estimate(self, assignment) -> float:
        if len(assignment) > self.k or len(set(assignment)) != len(assignment):
            raise ValueError("Assignment must use distinct nodes and available coupon positions")
        covered = set()
        for j, v in enumerate(assignment):
            covered.update(self.index[j].get(int(v), ()))
        return self.n * len(covered) / self.theta

    def greedy(self):
        started = time.perf_counter()
        covered = bytearray(self.theta)
        selected = []
        available = np.ones(self.n, dtype=bool)
        gains_chosen = []
        for j in range(self.k):
            gains = np.zeros(self.n, dtype=np.int64)
            for v, ids in self.index[j].items():
                if available[v]:
                    gains[v] = sum(not covered[t] for t in ids)
            gains[~available] = -1
            v = int(np.argmax(gains))
            selected.append(v)
            available[v] = False
            gains_chosen.append(int(gains[v]))
            for t in self.index[j].get(v, ()):
                covered[t] = 1
        return selected, {
            "greedy_seconds": time.perf_counter() - started,
            "training_estimate_not_quality": self.n * sum(covered) / self.theta,
            "selected_marginal_counts": gains_chosen,
        }


def sample_ssr(sampler: CouponRR, k: int, theta: int, seed: int) -> SSRBank:
    if not 1 <= k <= sampler.graph.n or theta <= 0:
        raise ValueError("Require 1 <= k <= n and positive sample count")
    started = time.perf_counter()
    rng = random.Random(seed)
    code = 'I' if theta <= 2**32 else 'Q'
    index = [defaultdict(lambda: array(code)) for _ in range(k)]
    root_counts = np.zeros(sampler.graph.n, dtype=np.int64)
    empty = 0
    before = sampler.stats()
    for t in range(theta):
        root = rng.randrange(sampler.graph.n)
        root_counts[root] += 1
        any_nonempty = False
        for j in range(k):
            rr = sampler.sample(root, rng)
            any_nonempty |= bool(rr)
            for v in rr:
                index[j][v].append(t)
        empty += not any_nonempty
    stats = {key: value - before[key] for key, value in sampler.stats().items()
             if key != "max_size"}
    stats["sampler_max_size_so_far"] = sampler.max_size
    return SSRBank(sampler.graph.n, k, theta, index, root_counts, empty,
                   stats, time.perf_counter() - started)


def sufficient_budget(n: int, k: int, lower_bound: float, epsilon: float, delta: float) -> int:
    """The Bernstein sufficient budget, not a necessary or minimal sample size."""
    if not (1 <= k <= n and lower_bound > 0 and 0 < epsilon < 1 and 0 < delta < 1):
        raise ValueError("Invalid theoretical-budget arguments")
    logs = []
    log_product = 0.
    for j in range(1, k + 1):
        log_product += math.log(n - j + 1)
        logs.append(log_product)
    top = max(logs)
    log_count = top + math.log(sum(math.exp(x - top) for x in logs))
    factor = 8*n*k*k/(epsilon*epsilon*lower_bound*lower_bound) + 4*n*k/(3*epsilon*lower_bound)
    return math.ceil(factor * (math.log(2/delta) + log_count))
