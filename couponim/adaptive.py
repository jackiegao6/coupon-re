"""Incremental conditioned SRR sampling and heuristic validation stopping.

Ported from run_adaptive_parameter_sweep.py. The stopping rule is empirical,
not a finite-sample approximation certificate. Final quality evaluation must
use a separate stream (the experiment runner does this).
"""
from __future__ import annotations
import bisect
from collections import defaultdict
import random
import time
import numpy as np
from .graph import Graph
from .conditioned import conditioned_gates, reverse_coupon_set
from .simulation import single_coupon_adopter

class IncrementalCIMSamplePool:
    """Append conditioned joint RR samples while preserving one stream prefix."""

    def __init__(
        self,
        graph: Graph,
        alpha: np.ndarray,
        discard: np.ndarray,
        k: int,
        seed: int,
    ) -> None:
        self.graph = graph
        self.alpha = alpha
        self.discard = discard
        self.k = k
        self.rng = random.Random(seed)
        root_weights = np.ones_like(alpha, dtype=float)
        interior = alpha < 1
        root_weights[interior] = -np.expm1(k*np.log1p(-alpha[interior]))
        self.weight_sum = float(root_weights.sum())
        self.root_cumulative = np.cumsum(root_weights).tolist()
        self.coverage: list[dict[int, list[int]]] = [
            defaultdict(list) for _ in range(k)
        ]
        self.samples = 0
        self.memberships = 0
        self.total_seconds = 0.0

    def extend(self, target_samples: int) -> None:
        if target_samples < self.samples:
            raise ValueError("Incremental sample count cannot decrease")
        if self.weight_sum <= 0.0:
            self.samples = target_samples
            return
        for sample_id in range(self.samples, target_samples):
            root = bisect.bisect(
                self.root_cumulative,
                self.rng.random() * self.weight_sum,
                0,
                self.graph.n - 1,
            )
            sample_rng = random.Random(self.rng.getrandbits(64))
            gates = conditioned_gates(
                float(self.alpha[root]), self.k, sample_rng
            )
            for coupon_index, gate in enumerate(gates):
                if not gate:
                    continue
                rr_set = reverse_coupon_set(
                    self.graph, root, self.alpha, self.discard, sample_rng
                )
                self.memberships += len(rr_set)
                for node in rr_set:
                    self.coverage[coupon_index][node].append(sample_id)
        self.samples = target_samples

    def select(self) -> tuple[list[int], float]:
        if self.samples <= 0:
            raise ValueError("Cannot select seeds without RR samples")
        selected: list[int] = []
        selected_counts = np.zeros(self.graph.n, dtype=np.int32)
        covered = bytearray(self.samples)
        covered_count = 0
        for coupon_index in range(self.k):
            best_node = -1
            best_gain = -1
            for node, sample_ids in self.coverage[coupon_index].items():
                if selected_counts[node] >= 1:
                    continue
                gain = sum(1 for sample_id in sample_ids if not covered[sample_id])
                if gain > best_gain or (gain == best_gain and node < best_node):
                    best_node = node
                    best_gain = gain
            if best_node < 0:
                best_node = next(
                    node for node in range(self.graph.n) if selected_counts[node] < 1
                )
            selected.append(best_node)
            selected_counts[best_node] += 1
            for sample_id in self.coverage[coupon_index].get(best_node, ()):
                if not covered[sample_id]:
                    covered[sample_id] = 1
                    covered_count += 1
        estimate = self.weight_sum * covered_count / self.samples
        return selected, estimate

    def extend_and_select(
        self, target_samples: int
    ) -> tuple[list[int], float, float, float, int]:
        started = time.perf_counter()
        self.extend(target_samples)
        allocation, estimate = self.select()
        stage_seconds = time.perf_counter() - started
        self.total_seconds += stage_seconds
        return (
            allocation,
            stage_seconds,
            self.total_seconds,
            estimate,
            self.memberships,
        )

def paired_validation(graph, alpha, discard, previous, current, trials, seed):
    rng = random.Random(seed)
    totals = [0, 0]
    for _ in range(trials):
        campaign_seed = rng.getrandbits(64)
        for i, seeds in enumerate((previous, current)):
            stream = random.Random(campaign_seed)
            adopters = {single_coupon_adopter(graph, s, alpha, discard, stream) for s in seeds}
            adopters.discard(-1)
            totals[i] += len(adopters)
    return tuple(v/trials for v in totals)


def adaptive_select(graph, alpha, discard, k, seed, *, initial_samples=100000,
                    minimum_samples=100000, maximum_samples=3000000,
                    validation_trials=5000, tolerance_percent=.5):
    if (not 1 <= k <= graph.n or not 0 < initial_samples <= maximum_samples
            or not 0 < minimum_samples <= maximum_samples or validation_trials < 1
            or tolerance_percent < 0):
        raise ValueError("Invalid adaptive settings")
    pool = IncrementalCIMSamplePool(graph, alpha, discard, k, seed)
    validation_rng = random.Random(seed ^ 0xA4D415)
    previous, traces, target = None, [], initial_samples
    while True:
        seeds, stage_time, total_time, estimate, memberships = pool.extend_and_select(target)
        trace = {"samples": target, "seeds": seeds, "training_estimate_not_quality": estimate,
                 "memberships": memberships, "stage_seconds": stage_time,
                 "pool_seconds": total_time, "stable": False}
        if previous is not None:
            validation_seed = validation_rng.getrandbits(64)
            old, new = paired_validation(graph, alpha, discard, previous, seeds,
                                         validation_trials, validation_seed)
            relative = (100*(new-old)/old if old else (0 if new == 0 else None))
            trace.update(validation_seed=validation_seed, validation_previous=old,
                         validation_current=new, relative_change_percent=relative,
                         stable=relative is not None and abs(relative) <= tolerance_percent)
        traces.append(trace)
        if target >= minimum_samples and trace["stable"]:
            reason = "validation_stability"
            break
        if target == maximum_samples:
            reason = "sample_cap"
            break
        previous, target = seeds, min(maximum_samples, target*2)
    return seeds, {"adaptive_trace": traces, "final_samples": target, "stop_reason": reason,
                   "stopping_rule": "empirical_paired_validation_not_a_certificate",
                   "training_estimate_not_quality": estimate, "memberships": memberships}
