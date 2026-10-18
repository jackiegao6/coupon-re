"""Independent forward evaluation under discard-on-revisit diffusion."""
from __future__ import annotations
import math
import random
from typing import Sequence
import numpy as np
from .graph import Graph

def single_coupon_adopter(
    graph: Graph,
    start: int,
    alpha: np.ndarray,
    discard: np.ndarray,
    rng: random.Random,
) -> int:
    current = start
    visited: set[int] = set()
    while current not in visited:
        visited.add(current)
        draw = rng.random()
        if draw < alpha[current]:
            return current
        if draw < alpha[current] + discard[current]:
            return -1
        neighbors = graph.out_neighbors(current)
        if not len(neighbors):
            return -1
        current = int(neighbors[rng.randrange(len(neighbors))])
    return -1


def evaluate(
    graph: Graph,
    seeds: Sequence[int],
    alpha: np.ndarray,
    discard: np.ndarray,
    simulations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if simulations <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive evaluations and distinct nonempty seeds required")
    if any(not 0 <= s < graph.n for s in seeds):
        raise ValueError("Seed outside graph")
    rng = random.Random(seed)
    count = 0
    mean = 0.0
    m2 = 0.0
    redemption_mean = 0.0
    for _ in range(simulations):
        adopters: set[int] = set()
        redemptions = 0
        for start in seeds:
            adopter = single_coupon_adopter(graph, start, alpha, discard, rng)
            if adopter >= 0:
                adopters.add(adopter)
                redemptions += 1
        value = float(len(adopters))
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
        redemption_mean += (redemptions - redemption_mean) / count
    variance = m2 / (count - 1) if count > 1 else 0.0
    ci95 = 1.96 * math.sqrt(variance / count) if count else 0.0
    return mean, ci95, variance, redemption_mean


