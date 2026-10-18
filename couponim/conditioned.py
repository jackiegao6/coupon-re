"""Conditioned sampler adapted from experiments/run_real_submission.py.

Roots have weight 1-(1-alpha[root])**k. The k consumption gates are
conditioned on at least one success. This variant is separate from the
uniform-root streaming sampler; both use discard-on-revisit diffusion.
"""
from __future__ import annotations
from collections import defaultdict
import math
import random
import time
import numpy as np
from .graph import Graph

def conditioned_gates(probability: float, k: int, rng: random.Random) -> list[bool]:
    if probability <= 0.0:
        return [False] * k
    if probability >= 1.0:
        return [True] * k

    normalizer = -math.expm1(k * math.log1p(-probability))
    target = rng.random() * normalizer
    mass = probability
    cumulative = 0.0
    first = k - 1
    for index in range(k):
        cumulative += mass
        if target <= cumulative:
            first = index
            break
        mass *= 1.0 - probability

    gates = [False] * k
    gates[first] = True
    for index in range(first + 1, k):
        gates[index] = rng.random() < probability
    return gates

def reverse_coupon_set(
    graph: Graph,
    root: int,
    alpha: np.ndarray,
    discard: np.ndarray,
    rng: random.Random,
) -> set[int]:
    rr_set = {root}
    queue = [root]
    # The root gate has already fixed the root action to adoption. Marking it
    # terminal keeps the lazy possible world internally consistent on cycles.
    choices: dict[int, int] = {root: -1}
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for predecessor in graph.in_neighbors[current]:
            if predecessor not in choices:
                draw = rng.random()
                if draw < alpha[predecessor] + discard[predecessor]:
                    choices[predecessor] = -1
                else:
                    neighbors = graph.out_neighbors(predecessor)
                    choices[predecessor] = int(neighbors[rng.randrange(len(neighbors))])
            if choices[predecessor] == current and predecessor not in rr_set:
                rr_set.add(predecessor)
                queue.append(predecessor)
    return rr_set

def cim_ris_seeds(
    graph: Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    k: int,
    samples: int,
    seed: int,
    capacity_per_node: int = 1,
) -> tuple[list[int], float, float, int]:
    if capacity_per_node < 1:
        raise ValueError("capacity_per_node must be positive")
    if k < 0 or k > graph.n * capacity_per_node:
        raise ValueError("coupon budget must be nonnegative and fit seed capacity")
    if k > 0 and samples <= 0:
        raise ValueError("samples must be positive for a nonempty allocation")
    started = time.perf_counter()
    if k == 0:
        return [], time.perf_counter() - started, 0.0, 0
    rng = random.Random(seed)
    root_weights = 1.0 - np.power(1.0 - alpha, k)
    # Preserve historical positive weights, recovering only cancellation to zero.
    tiny_positive = (alpha > 0.0) & (alpha < 1.0) & (root_weights == 0.0)
    root_weights[tiny_positive] = -np.expm1(k * np.log1p(-alpha[tiny_positive]))
    weight_sum = float(np.sum(root_weights))
    if weight_sum == 0.0:
        # All outcomes have zero adoption; return a deterministic feasible allocation.
        selected = [index // capacity_per_node for index in range(k)]
        return selected, time.perf_counter() - started, 0.0, 0
    roots = rng.choices(range(graph.n), weights=root_weights, k=samples)
    sample_seeds = [rng.getrandbits(64) for _ in range(samples)]
    coverage: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(k)]
    memberships = 0

    for sample_id, (root, sample_seed) in enumerate(zip(roots, sample_seeds)):
        sample_rng = random.Random(sample_seed)
        gates = conditioned_gates(float(alpha[root]), k, sample_rng)
        for coupon_index, gate in enumerate(gates):
            if not gate:
                continue
            rr_set = reverse_coupon_set(graph, root, alpha, discard, sample_rng)
            memberships += len(rr_set)
            for node in rr_set:
                coverage[coupon_index][node].append(sample_id)

    selected: list[int] = []
    selected_counts = np.zeros(graph.n, dtype=np.int32)
    covered = bytearray(samples)
    covered_count = 0
    for coupon_index in range(k):
        best_node = -1
        best_gain = -1
        for node, sample_ids in coverage[coupon_index].items():
            if selected_counts[node] >= capacity_per_node:
                continue
            gain = sum(1 for sample_id in sample_ids if not covered[sample_id])
            if gain > best_gain or (gain == best_gain and node < best_node):
                best_node = node
                best_gain = gain
        if best_node < 0:
            best_node = next(
                node
                for node in range(graph.n)
                if selected_counts[node] < capacity_per_node
            )
            best_gain = 0
        selected.append(best_node)
        selected_counts[best_node] += 1
        for sample_id in coverage[coupon_index].get(best_node, ()):
            if not covered[sample_id]:
                covered[sample_id] = 1
                covered_count += 1

    estimated_spread = weight_sum * covered_count / samples
    return selected, time.perf_counter() - started, estimated_spread, memberships
