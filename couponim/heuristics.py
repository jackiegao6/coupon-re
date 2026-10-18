"""Structural and local one-coupon ranking baselines."""
import random
import numpy as np
from .graph import Graph

def pagerank_order(graph: Graph, iterations: int = 50, damping: float = 0.85) -> list[int]:
    rank = np.full(graph.n, 1.0 / graph.n)
    nonisolated = graph.degrees > 0
    for _ in range(iterations):
        dangling = float(np.sum(rank[~nonisolated]))
        next_rank = np.full(graph.n, (1.0 - damping + damping * dangling) / graph.n)
        contributions = damping * rank[nonisolated] / graph.degrees[nonisolated]
        edge_contributions = np.repeat(contributions, graph.degrees[nonisolated])
        np.add.at(next_rank, graph.indices, edge_contributions)
        if float(np.max(np.abs(next_rank - rank))) < 1e-11:
            rank = next_rank
            break
        rank = next_rank
    return np.argsort(-rank, kind="stable").tolist()


def static_orders(
    graph: Graph,
    alpha: np.ndarray,
    transfer: np.ndarray,
    seed: int,
) -> dict[str, list[int]]:
    degree_order = np.argsort(-graph.degrees, kind="stable").tolist()
    alpha_order = np.argsort(-alpha, kind="stable").tolist()
    one_hop = alpha.copy()
    for node in range(graph.n):
        neighbors = graph.out_neighbors(node)
        if len(neighbors):
            one_hop[node] += transfer[node] * float(np.mean(alpha[neighbors]))
    one_hop_order = np.argsort(-one_hop, kind="stable").tolist()
    random_order = list(range(graph.n))
    random.Random(seed).shuffle(random_order)
    return {
        "DegreeTopM": degree_order,
        "Alpha-Sort": alpha_order,
        "1Hop-Sort": one_hop_order,
        "PageRank": pagerank_order(graph),
        "Random": random_order,
    }


