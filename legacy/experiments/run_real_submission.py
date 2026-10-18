"""Run traceable coupon-diffusion experiments without SciPy at runtime.

The repository stores graphs as pickled SciPy CSR matrices. This runner
loads their raw CSR state through a minimal compatibility class, then uses
the exact model defined in ``paper-v2 copy.tex``. It never reads planning
or mock result files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "Netscience": ROOT / "dataset/network/network.netscience-adj.pkl",
    "NetFacebookEgo": ROOT / "dataset/network/network.netfacebookego-adj.pkl",
    "DoubanRandom": ROOT / "dataset/network/network.doubanrandom-adj.pkl",
    "EmailEnron": ROOT / "dataset/network/network.EmailEnron-adj.pkl",
    "network.douban": ROOT / "dataset/network/network.douban-adj.pkl",
}
SCENARIOS = {
    "balanced": (0.30, 0.10, 0.10, 0.10),
    "adoption-heavy": (0.45, 0.35, 0.15, 0.05),
    "forwarding-heavy": (0.01, 0.05, 0.01, 0.05),
}


class _CSRState:
    """Enough of scipy.sparse.csr_matrix for unpickling stored arrays."""

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)


@dataclass
class Graph:
    name: str
    n: int
    m: int
    indices: np.ndarray
    indptr: np.ndarray
    in_neighbors: list[list[int]]
    degrees: np.ndarray

    def out_neighbors(self, node: int) -> np.ndarray:
        return self.indices[self.indptr[node] : self.indptr[node + 1]]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _load_raw_csr(path: Path) -> _CSRState:
    # Resolve pickle names locally rather than replacing global NumPy modules.
    # Global aliases recurse on NumPy 2 and can corrupt already-imported SciPy.
    numpy_prefix = "numpy._core" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core"

    class CSRReader(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("scipy.sparse") and name == "csr_matrix":
                return _CSRState
            for prefix in ("numpy._core", "numpy.core"):
                if module == prefix or module.startswith(prefix + "."):
                    return getattr(importlib.import_module(numpy_prefix + module[len(prefix):]), name)
            return super().find_class(module, name)

    with path.open("rb") as handle:
        return CSRReader(handle).load()


def load_graph(name: str) -> Graph:
    path = DATASETS[name]
    matrix = _load_raw_csr(path)
    n, width = matrix._shape
    if n != width:
        raise ValueError(f"{name} is not square: {matrix._shape}")

    indices = np.asarray(matrix.indices, dtype=np.int32)
    indptr = np.asarray(matrix.indptr, dtype=np.int64)
    degrees = np.diff(indptr).astype(np.int32, copy=False)
    in_neighbors: list[list[int]] = [[] for _ in range(n)]
    for source in range(n):
        for target in indices[indptr[source] : indptr[source + 1]]:
            in_neighbors[int(target)].append(source)

    return Graph(
        name=name,
        n=n,
        m=len(indices),
        indices=indices,
        indptr=indptr,
        in_neighbors=in_neighbors,
        degrees=degrees,
    )


def node_probabilities(
    graph: Graph,
    scenario: str,
    degree_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_a, slope_a, base_d, slope_d = SCENARIOS[scenario]
    log_degree = np.log1p(graph.degrees.astype(np.float64))
    denominator = max(float(np.max(log_degree)), 1.0)
    normalized = np.clip(log_degree / denominator, 1e-5, 1.0) ** degree_power
    alpha = base_a + slope_a * (1.0 - normalized)
    discard = base_d + slope_d * normalized
    transfer = 1.0 - alpha - discard
    isolated = graph.degrees == 0
    discard[isolated] += transfer[isolated]
    transfer[isolated] = 0.0
    if np.any(alpha < 0) or np.any(discard < 0) or np.any(transfer < 0):
        raise ValueError(f"Invalid probabilities for scenario {scenario}")
    if not np.allclose(alpha + discard + transfer, 1.0):
        raise AssertionError("Node probabilities do not sum to one")
    return alpha, discard, transfer


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


def evaluate_with_streams(
    graph: Graph,
    seeds: Sequence[int],
    alpha: np.ndarray,
    discard: np.ndarray,
    simulation_seeds: Sequence[int],
) -> tuple[float, float, float, float]:
    """Evaluate one allocation using reusable realization-level streams.

    Allocations compared in the same configuration start each forward
    realization from the same random-generator state. Path-dependent random
    consumption means that this is a partial common-random-number coupling,
    while every method retains the correct marginal simulation distribution.
    """
    count = 0
    mean = 0.0
    m2 = 0.0
    redemption_mean = 0.0
    for simulation_seed in simulation_seeds:
        rng = random.Random(simulation_seed)
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


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    budgets = sorted({int(item) for item in args.budgets.split(",") if item})
    output_dir = Path(args.output_dir)
    result_rows: list[dict[str, object]] = []
    seed_records: list[dict[str, object]] = []

    for dataset in datasets:
        print(f"Loading {dataset}...", flush=True)
        graph = load_graph(dataset)
        alpha, discard, transfer = node_probabilities(graph, args.scenario)
        if max(budgets) > graph.n:
            raise ValueError(f"Budget exceeds node count for {dataset}")

        static_started = time.perf_counter()
        orders = static_orders(
            graph,
            alpha,
            transfer,
            _stable_seed(args.seed, dataset, "static"),
        )
        static_seconds = time.perf_counter() - static_started

        ic_order, ic_seconds = ic_ris_order(
            graph,
            transfer,
            max(budgets),
            args.rr_samples,
            _stable_seed(args.seed, dataset, "ic-ris"),
        )
        orders["IC-RIS"] = ic_order

        if dataset == "Netscience" and args.oracle_trajectories > 0:
            oracle_started = time.perf_counter()
            q_matrix = estimate_single_coupon_matrix(
                graph,
                alpha,
                discard,
                args.oracle_trajectories,
                _stable_seed(args.seed, dataset, "oracle"),
            )
            orders["MC-Greedy"] = mc_greedy_order(q_matrix, max(budgets))
            oracle_seconds = time.perf_counter() - oracle_started
        else:
            oracle_seconds = 0.0

        for budget in budgets:
            print(f"  {dataset}: k={budget}", flush=True)
            cim_seeds, cim_seconds, estimated, memberships = cim_ris_seeds(
                graph,
                alpha,
                discard,
                budget,
                args.rr_samples,
                _stable_seed(args.seed, dataset, args.scenario, budget, "cim-ris"),
            )
            method_seeds = {name: order[:budget] for name, order in orders.items()}
            method_seeds["CIM-RIS"] = cim_seeds

            for method, seeds in method_seeds.items():
                mean, ci95, variance, redemptions = evaluate(
                    graph,
                    seeds,
                    alpha,
                    discard,
                    args.eval_simulations,
                    _stable_seed(args.seed, dataset, args.scenario, budget, method, "eval"),
                )
                if method == "CIM-RIS":
                    selection_seconds = cim_seconds
                elif method == "IC-RIS":
                    selection_seconds = ic_seconds
                elif method == "MC-Greedy":
                    selection_seconds = oracle_seconds
                else:
                    selection_seconds = static_seconds
                result_rows.append({
                    "dataset": dataset,
                    "scenario": args.scenario,
                    "nodes": graph.n,
                    "edges": graph.m,
                    "k": budget,
                    "method": method,
                    "mean_adopters": f"{mean:.8f}",
                    "ci95": f"{ci95:.8f}",
                    "variance": f"{variance:.8f}",
                    "mean_redemptions": f"{redemptions:.8f}",
                    "selection_seconds": f"{selection_seconds:.8f}",
                    "rr_samples": args.rr_samples if method in {"CIM-RIS", "IC-RIS"} else "",
                    "eval_simulations": args.eval_simulations,
                    "cim_estimated_spread": f"{estimated:.8f}" if method == "CIM-RIS" else "",
                    "rr_memberships": memberships if method == "CIM-RIS" else "",
                    "status": "REAL_EXPERIMENT",
                })
                seed_records.append({
                    "dataset": dataset,
                    "scenario": args.scenario,
                    "k": budget,
                    "method": method,
                    "seeds": seeds,
                })

    result_path = output_dir / f"real_quality_{args.scenario}.csv"
    write_rows(
        result_path,
        [
            "dataset", "scenario", "nodes", "edges", "k", "method",
            "mean_adopters", "ci95", "variance", "mean_redemptions",
            "selection_seconds", "rr_samples", "eval_simulations",
            "cim_estimated_spread", "rr_memberships", "status",
        ],
        result_rows,
    )
    with (output_dir / f"real_seeds_{args.scenario}.json").open("w", encoding="utf-8") as handle:
        json.dump(seed_records, handle, indent=2)
    metadata = {
        "status": "REAL_EXPERIMENT",
        "datasets": datasets,
        "budgets": budgets,
        "scenario": args.scenario,
        "rr_samples": args.rr_samples,
        "eval_simulations": args.eval_simulations,
        "oracle_trajectories_per_source": args.oracle_trajectories,
        "seed": args.seed,
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "python": sys.version,
        "numpy": np.__version__,
    }
    with (output_dir / f"real_metadata_{args.scenario}.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote {result_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="Netscience,NetFacebookEgo,DoubanRandom,EmailEnron")
    parser.add_argument("--budgets", default="10,25,50,100,150,200")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="balanced")
    parser.add_argument("--rr-samples", type=int, default=5000)
    parser.add_argument("--eval-simulations", type=int, default=5000)
    parser.add_argument("--oracle-trajectories", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/results"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
