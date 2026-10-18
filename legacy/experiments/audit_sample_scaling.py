"""Post-hoc high-sample diagnostic for the hardest validated configuration."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import numpy as np

import run_real_submission as core
from audit_weak_gain import spread


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/results/diagnostics/high_sample_scaling.csv"
SAMPLE_COUNTS = [50_000, 100_000, 250_000, 500_000]
SEEDS = [20260721, 20260722, 20260723]


def main() -> None:
    graph = core.load_graph("NetFacebookEgo")
    alpha, discard, _ = core.node_probabilities(graph, "forwarding-heavy")
    q = np.load(
        ROOT
        / (
            "experiments/results/validated-v2/extensions/oracle/"
            "netfacebookego_forwarding-heavy_50000.npz"
        )
    )["q"].astype(np.float64)
    k = 100
    reference = spread(q, core.mc_greedy_order(q, k))
    rows: list[dict[str, object]] = []
    for samples in SAMPLE_COUNTS:
        for seed in SEEDS:
            allocation, elapsed, estimate, memberships = core.cim_ris_seeds(
                graph,
                alpha,
                discard,
                k,
                samples,
                core._stable_seed("high-sample-audit", samples, seed),
            )
            independent = spread(q, allocation)
            rows.append(
                {
                    "dataset": graph.name,
                    "scenario": "forwarding-heavy",
                    "k": k,
                    "rr_samples": samples,
                    "selection_seed": seed,
                    "independent_q_spread": independent,
                    "reference_q_spread": reference,
                    "gap_to_reference_pct": 100.0
                    * (reference - independent)
                    / reference,
                    "training_rr_estimate": estimate,
                    "training_optimism_pct": 100.0
                    * (estimate - independent)
                    / independent,
                    "rr_memberships": memberships,
                    "selection_seconds": elapsed,
                    "status": "POST_HOC_DIAGNOSTIC",
                }
            )
            print(samples, seed, "q=%.4f" % independent, "gap=%.2f%%" % rows[-1]["gap_to_reference_pct"], flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for samples in SAMPLE_COUNTS:
        selected = [row for row in rows if row["rr_samples"] == samples]
        print(
            "summary",
            samples,
            "spread=%.4f" % statistics.mean(float(row["independent_q_spread"]) for row in selected),
            "gap=%.2f%%" % statistics.mean(float(row["gap_to_reference_pct"]) for row in selected),
            "optimism=%.1f%%" % statistics.mean(float(row["training_optimism_pct"]) for row in selected),
        )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
