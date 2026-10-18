"""Forward sampled coverage baselines for the fixed-action coupon model.

Each capacity slot has independent coupon outcomes in each batch. Fixing this
bank gives a monotone submodular coverage objective, so lazy bounds remain
valid. Its training value is NOT an independent estimate of selected quality.
"""
from __future__ import annotations

import heapq
import math
import random
import time

import numpy as np

import run_real_submission as core


def sample_coverage(graph, alpha, discard, batches, seed, capacity_per_node=1):
    if batches <= 0 or capacity_per_node <= 0:
        raise ValueError("batches and capacity must be positive")
    rng = random.Random(seed)
    bank = []
    for node in range(graph.n):
        for _ in range(capacity_per_node):
            outcomes = []
            for batch in range(batches):
                adopter = core.single_coupon_adopter(graph, node, alpha, discard, rng)
                if adopter >= 0:
                    outcomes.append(batch * graph.n + adopter)
            bank.append(np.asarray(outcomes, dtype=np.int64))
    return bank


def select_coverage(bank, k, mode="lazy", epsilon=0.1, seed=0):
    if not 0 <= k <= len(bank):
        raise ValueError("budget must fit the slot ground set")
    if mode not in {"lazy", "stochastic"} or not 0 < epsilon < 1:
        raise ValueError("invalid mode or epsilon")
    selected, covered = [], set()
    evaluations = 0

    def gain(slot):
        nonlocal evaluations
        evaluations += 1
        return sum(int(outcome) not in covered for outcome in bank[slot])

    if mode == "lazy":
        heap = [(-len(outcomes), slot, 0) for slot, outcomes in enumerate(bank)]
        heapq.heapify(heap)
        while len(selected) < k:
            negative_gain, slot, evaluated_round = heapq.heappop(heap)
            if evaluated_round != len(selected):
                heapq.heappush(heap, (-gain(slot), slot, len(selected)))
                continue
            selected.append(slot)
            covered.update(map(int, bank[slot]))
    else:
        rng = random.Random(seed)
        remaining = set(range(len(bank)))
        sample_size = math.ceil(len(bank) / max(k, 1) * math.log(1.0 / epsilon))
        while len(selected) < k:
            candidates = rng.sample(sorted(remaining), min(len(remaining), sample_size))
            slot = max(candidates, key=lambda item: (gain(item), -item))
            selected.append(slot)
            covered.update(map(int, bank[slot]))
            remaining.remove(slot)
    return selected, len(covered), evaluations


def forward_coverage_seeds(graph, alpha, discard, k, batches, seed,
                           mode="lazy", epsilon=0.1, capacity_per_node=1):
    if not 0 <= k <= graph.n * capacity_per_node:
        raise ValueError("budget must fit seed capacity")
    started = time.perf_counter()
    bank = sample_coverage(graph, alpha, discard, batches, seed, capacity_per_node)
    sampled = time.perf_counter()
    slots, count, evaluations = select_coverage(bank, k, mode, epsilon, seed + 1)
    ended = time.perf_counter()
    return [slot // capacity_per_node for slot in slots], {
        "selection_seconds": ended - started,
        "sampling_seconds": sampled - started,
        "greedy_seconds": ended - sampled,
        "training_estimate_not_quality": count / batches,
        "forward_trajectories": graph.n * capacity_per_node * batches,
        "outcome_array_bytes_not_peak_memory": sum(row.nbytes for row in bank),
        "marginal_recomputations": evaluations,
    }
