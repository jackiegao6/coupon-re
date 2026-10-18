"""Small-graph exact diagnosis under discard-on-revisit, including cycles.

Unlike an absorbing Markov matrix with fresh decisions on revisits, this
dynamic program carries the visited set. Its state space is exponential.
"""
from functools import lru_cache
from itertools import combinations
from math import comb
import numpy as np


def consumption_matrix(graph, alpha, discard, max_nodes=16):
    if graph.n > max_nodes:
        raise ValueError(f"Exact visited-set DP is limited to {max_nodes} nodes")
    n = graph.n
    @lru_cache(None)
    def visit(u, seen):
        result = np.zeros(n)
        result[u] = alpha[u]
        neighbors = graph.out_neighbors(u)
        if len(neighbors):
            weight = (1-alpha[u]-discard[u])/len(neighbors)
            if weight > 0:
                for v in neighbors:
                    v = int(v)
                    if not seen & (1 << v):
                        result += weight * visit(v, seen | (1 << v))
        return result
    return np.asarray([visit(u, 1 << u) for u in range(n)])


def spread(q, seeds):
    return float(np.sum(1-np.prod(1-q[list(seeds)], axis=0)))


def greedy(q, k):
    if not 1 <= k <= len(q):
        raise ValueError("Require 1 <= k <= n")
    residual = np.ones(q.shape[1])
    selected, marginals = [], []
    for _ in range(k):
        gains = q @ residual
        gains[selected] = -np.inf
        chosen = int(np.argmax(gains))
        selected.append(chosen)
        marginals.append(float(gains[chosen]))
        residual *= 1-q[chosen]
    return selected, marginals


def diagnose(graph, alpha, discard, k, max_nodes=16, max_subsets=100000):
    q = consumption_matrix(graph, alpha, discard, max_nodes)
    selected, gains = greedy(q, k)
    result = {"n": graph.n, "k": k, "greedy_seeds": selected,
              "greedy_spread": spread(q, selected), "greedy_marginals": gains,
              "single_coupon_consumption_probabilities": q.sum(axis=1).tolist()}
    count = comb(graph.n, k)
    if count > max_subsets:
        result["enumeration"] = {"status": "skipped", "subsets": count, "limit": max_subsets}
    else:
        values, best, best_value = [], None, -1
        for seeds in combinations(range(graph.n), k):
            value = spread(q, seeds)
            values.append(value)
            if value > best_value:
                best, best_value = seeds, value
        result["enumeration"] = {"status": "complete", "subsets": count,
                                 "optimal_seeds": best, "optimal_spread": best_value,
                                 "uniform_random_mean": float(np.mean(values)),
                                 "uniform_random_std": float(np.std(values))}
    return result
