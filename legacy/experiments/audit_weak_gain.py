"""Decompose weak CIM-RIS gains without rerunning or changing experiments."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_real_submission as core


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "experiments/results/validated-v2/validated_raw.csv"
OUTPUT_PATH = ROOT / "experiments/results/diagnostics/weak_gain_decomposition.csv"
HOLDOUT_PATH = ROOT / "experiments/results/diagnostics/weak_gain_holdout.csv"
BASELINES = ["IC-RIS", "1Hop-Sort", "Alpha-Sort", "DegreeTopM", "PageRank"]
Q_PATHS = {
    ("Netscience", scenario): ROOT
    / f"experiments/results/validated-v2/oracle/netscience_{scenario}_100000.npz"
    for scenario in core.SCENARIOS
}
Q_PATHS.update(
    {
        ("NetFacebookEgo", scenario): ROOT
        / (
            "experiments/results/validated-v2/extensions/oracle/"
            f"netfacebookego_{scenario}_50000.npz"
        )
        for scenario in core.SCENARIOS
    }
)


def spread(q_matrix: np.ndarray, allocation: list[int]) -> float:
    residual = np.ones(q_matrix.shape[1], dtype=np.float64)
    for source in allocation:
        residual *= 1.0 - q_matrix[source]
    return float(np.sum(1.0 - residual))


def relative_gap(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference if reference > 0.0 else 0.0


def independent_rr_estimate(
    graph: core.Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    allocation: list[int],
    samples: int,
    seed: int,
) -> float:
    """Estimate one fixed allocation on RR samples unused for selection."""
    rng = random.Random(seed)
    k = len(allocation)
    root_weights = 1.0 - np.power(1.0 - alpha, k)
    weight_sum = float(root_weights.sum())
    covered = 0
    for root in rng.choices(range(graph.n), weights=root_weights, k=samples):
        sample_rng = random.Random(rng.getrandbits(64))
        gates = core.conditioned_gates(float(alpha[root]), k, sample_rng)
        for coupon_index, gate in enumerate(gates):
            if gate and allocation[coupon_index] in core.reverse_coupon_set(
                graph, root, alpha, discard, sample_rng
            ):
                covered += 1
                break
    return weight_sum * covered / samples


def load_rows() -> dict[tuple[str, str, int, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    with RAW_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["scenario"], int(row["k"]), row["method"])
            grouped[key].append(row)
    return grouped


def write_rows(rows: list[dict[str, object]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    grouped = load_rows()
    output: list[dict[str, object]] = []
    for (dataset, scenario), q_path in sorted(Q_PATHS.items()):
        graph = core.load_graph(dataset)
        alpha, discard, transfer = core.node_probabilities(graph, scenario)
        q_matrix = np.load(q_path)["q"].astype(np.float64)
        if q_matrix.shape != (graph.n, graph.n):
            raise AssertionError(f"Unexpected matrix shape: {q_path}")
        row_redemption = q_matrix.sum(axis=1)
        if np.any(row_redemption + 1e-9 < alpha):
            raise AssertionError("Eventual redemption cannot be below immediate adoption")

        reference_order = core.mc_greedy_order(q_matrix, 200)
        for k in [10, 25, 50, 100, 150, 200]:
            reference_allocation = reference_order[:k]
            reference_spread = spread(q_matrix, reference_allocation)
            method_spreads: dict[str, list[float]] = {}
            for method in ["CIM-RIS", *BASELINES]:
                method_rows = grouped[(dataset, scenario, k, method)]
                if len(method_rows) != 5:
                    raise AssertionError(
                        f"Expected five rows for {dataset}/{scenario}/{k}/{method}"
                    )
                values = [
                    spread(q_matrix, json.loads(row["seeds"])) for row in method_rows
                ]
                method_spreads[method] = values

            mean_spreads = {
                method: statistics.mean(values)
                for method, values in method_spreads.items()
            }
            best_baseline = max(BASELINES, key=mean_spreads.__getitem__)
            cim_spread = mean_spreads["CIM-RIS"]
            baseline_spread = mean_spreads[best_baseline]

            optimism = []
            for row, independent_spread in zip(
                grouped[(dataset, scenario, k, "CIM-RIS")],
                method_spreads["CIM-RIS"],
            ):
                rr_estimate = float(row["cim_estimated_spread"])
                optimism.append(
                    100.0 * (rr_estimate - independent_spread) / independent_spread
                    if independent_spread > 0.0
                    else 0.0
                )

            selected = np.asarray(reference_allocation, dtype=np.int64)
            immediate_fraction = float(
                alpha[selected].sum() / row_redemption[selected].sum()
            )
            rr_samples = int(
                grouped[(dataset, scenario, k, "CIM-RIS")][0]["rr_samples"]
            )
            root_weight_sum = float(np.sum(1.0 - np.power(1.0 - alpha, k)))
            effective_per_index = rr_samples * float(alpha.sum()) / root_weight_sum
            output.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "k": k,
                    "reference_spread_q": reference_spread,
                    "cim_spread_q_mean": cim_spread,
                    "best_baseline": best_baseline,
                    "best_baseline_spread_q_mean": baseline_spread,
                    "cim_gap_to_reference_pct": relative_gap(
                        reference_spread, cim_spread
                    ),
                    "baseline_gap_to_reference_pct": relative_gap(
                        reference_spread, baseline_spread
                    ),
                    "cim_gain_over_baseline_pct": 100.0
                    * (cim_spread - baseline_spread)
                    / baseline_spread,
                    "rr_estimate_optimism_pct_mean": statistics.mean(optimism),
                    "rr_estimate_optimism_pct_std": statistics.stdev(optimism),
                    "reference_immediate_redemption_fraction": immediate_fraction,
                    "mean_adoption_probability": float(alpha.mean()),
                    "mean_transfer_probability": float(transfer.mean()),
                    "degree_one_fraction": float(np.mean(graph.degrees == 1)),
                    "rr_samples": rr_samples,
                    "expected_nonempty_rr_sets_per_index": effective_per_index,
                    "expected_nonempty_rr_sets_per_index_per_candidate": (
                        effective_per_index / graph.n
                    ),
                    "status": "POST_HOC_DIAGNOSTIC",
                }
            )

    write_rows(output)
    for scenario in core.SCENARIOS:
        selected = [row for row in output if row["scenario"] == scenario]
        print(
            scenario,
            "cim_gap=%.3f%%" % statistics.mean(
                float(row["cim_gap_to_reference_pct"]) for row in selected
            ),
            "baseline_gap=%.3f%%" % statistics.mean(
                float(row["baseline_gap_to_reference_pct"]) for row in selected
            ),
            "cim_vs_baseline=%+.3f%%" % statistics.mean(
                float(row["cim_gain_over_baseline_pct"]) for row in selected
            ),
            "rr_optimism=%+.3f%%" % statistics.mean(
                float(row["rr_estimate_optimism_pct_mean"]) for row in selected
            ),
            "immediate_fraction=%.3f" % statistics.mean(
                float(row["reference_immediate_redemption_fraction"])
                for row in selected
            ),
        )

    # A fixed-allocation holdout distinguishes estimator bias from adaptive
    # overfitting to the RR samples used by greedy selection.
    graph = core.load_graph("NetFacebookEgo")
    alpha, discard, _ = core.node_probabilities(graph, "balanced")
    q_matrix = np.load(Q_PATHS[("NetFacebookEgo", "balanced")])["q"].astype(
        np.float64
    )
    fixed = core.mc_greedy_order(q_matrix, 100)
    fixed_q_spread = spread(q_matrix, fixed)
    holdout = [
        independent_rr_estimate(
            graph,
            alpha,
            discard,
            fixed,
            samples=20_000,
            seed=core._stable_seed("weak-gain-holdout", repeat),
        )
        for repeat in range(10)
    ]
    with HOLDOUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "repeat",
                "q_spread",
                "independent_rr_estimate",
                "rr_samples",
                "status",
            ],
        )
        writer.writeheader()
        for repeat, estimate in enumerate(holdout):
            writer.writerow(
                {
                    "repeat": repeat,
                    "q_spread": fixed_q_spread,
                    "independent_rr_estimate": estimate,
                    "rr_samples": 20_000,
                    "status": "POST_HOC_DIAGNOSTIC",
                }
            )
    print(
        "fixed-allocation holdout:",
        "q_spread=%.3f" % fixed_q_spread,
        "rr_mean=%.3f" % statistics.mean(holdout),
        "rr_std=%.3f" % statistics.stdev(holdout),
    )
    print(f"Wrote {OUTPUT_PATH} and {HOLDOUT_PATH}")


if __name__ == "__main__":
    main()
