"""Additional comparison methods from the original experimental suite."""
from __future__ import annotations
from collections import defaultdict
import random
import time
import numpy as np
from .graph import Graph
from .simulation import single_coupon_adopter

def reverse_ic_set(
    graph: Graph,
    root: int,
    transfer: np.ndarray,
    rng: random.Random,
) -> set[int]:
    rr_set = {root}
    queue = [root]
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for predecessor in graph.in_neighbors[current]:
            probability = float(transfer[predecessor]) / graph.degrees[predecessor]
            if predecessor not in rr_set and rng.random() < probability:
                rr_set.add(predecessor)
                queue.append(predecessor)
    return rr_set

def ic_ris_order(
    graph: Graph,
    transfer: np.ndarray,
    max_k: int,
    samples: int,
    seed: int,
) -> tuple[list[int], float]:
    started = time.perf_counter()
    master_rng = random.Random(seed)
    coverage: dict[int, list[int]] = defaultdict(list)
    for sample_id in range(samples):
        root = master_rng.randrange(graph.n)
        rr_set = reverse_ic_set(
            graph,
            root,
            transfer,
            random.Random(master_rng.getrandbits(64)),
        )
        for node in rr_set:
            coverage[node].append(sample_id)

    selected: list[int] = []
    selected_set: set[int] = set()
    covered = bytearray(samples)
    for _ in range(max_k):
        best_node = -1
        best_gain = -1
        for node, sample_ids in coverage.items():
            if node in selected_set:
                continue
            gain = sum(1 for sample_id in sample_ids if not covered[sample_id])
            if gain > best_gain or (gain == best_gain and node < best_node):
                best_node = node
                best_gain = gain
        if best_node < 0:
            best_node = next(node for node in range(graph.n) if node not in selected_set)
        selected.append(best_node)
        selected_set.add(best_node)
        for sample_id in coverage.get(best_node, ()):
            covered[sample_id] = 1
    return selected, time.perf_counter() - started

def estimate_single_coupon_matrix(
    graph: Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    trajectories: int,
    seed: int,
) -> np.ndarray:
    matrix = np.zeros((graph.n, graph.n), dtype=np.float32)
    master_rng = random.Random(seed)
    for start in range(graph.n):
        counts: dict[int, int] = defaultdict(int)
        rng = random.Random(master_rng.getrandbits(64))
        for _ in range(trajectories):
            adopter = single_coupon_adopter(graph, start, alpha, discard, rng)
            if adopter >= 0:
                counts[adopter] += 1
        for adopter, count in counts.items():
            matrix[start, adopter] = count / trajectories
    return matrix

def mc_greedy_order(
    q_matrix: np.ndarray, max_k: int, capacity_per_node: int = 1
) -> list[int]:
    if capacity_per_node < 1 or capacity_per_node * q_matrix.shape[0] < max_k:
        raise ValueError("Infeasible capacity")
    residual = np.ones(q_matrix.shape[1], dtype=np.float64)
    selected: list[int] = []
    counts = np.zeros(q_matrix.shape[0], dtype=np.int32)
    for _ in range(max_k):
        gains = q_matrix.dot(residual)
        gains[counts >= capacity_per_node] = -1.0
        node = int(np.argmax(gains))
        selected.append(node)
        counts[node] += 1
        residual *= 1.0 - q_matrix[node]
    return selected
