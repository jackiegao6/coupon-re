"""Verify the post-hoc weak-gain diagnostics and their narrow conclusions."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_DIR = ROOT / "experiments/results/diagnostics"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DIAGNOSTIC_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    decomposition = read_csv("weak_gain_decomposition.csv")
    holdout = read_csv("weak_gain_holdout.csv")
    scaling = read_csv("high_sample_scaling.csv")
    if len(decomposition) != 36 or len(holdout) != 10 or len(scaling) != 12:
        raise AssertionError("Unexpected diagnostic row count")
    for rows in [decomposition, holdout, scaling]:
        if {row["status"] for row in rows} != {"POST_HOC_DIAGNOSTIC"}:
            raise AssertionError("Diagnostic status boundary was lost")

    holdout_q = float(holdout[0]["q_spread"])
    holdout_values = [float(row["independent_rr_estimate"]) for row in holdout]
    if abs(statistics.mean(holdout_values) - holdout_q) > 2.0:
        raise AssertionError("Fixed-allocation RR estimate is unexpectedly biased")

    by_samples: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in scaling:
        by_samples[int(row["rr_samples"])].append(row)
    if set(by_samples) != {50_000, 100_000, 250_000, 500_000}:
        raise AssertionError("Missing high-sample diagnostic setting")
    gaps = {
        samples: statistics.mean(
            float(row["gap_to_reference_pct"]) for row in rows
        )
        for samples, rows in by_samples.items()
    }
    optimism = {
        samples: statistics.mean(
            float(row["training_optimism_pct"]) for row in rows
        )
        for samples, rows in by_samples.items()
    }
    if not gaps[500_000] < gaps[50_000]:
        raise AssertionError("More samples did not reduce the measured reference gap")
    if not optimism[500_000] < optimism[50_000]:
        raise AssertionError("More samples did not reduce training optimism")

    hard = next(
        row
        for row in decomposition
        if row["dataset"] == "NetFacebookEgo"
        and row["scenario"] == "forwarding-heavy"
        and int(row["k"]) == 100
    )
    if float(hard["expected_nonempty_rr_sets_per_index_per_candidate"]) >= 1.0:
        raise AssertionError("Expected sparse-sample diagnostic no longer holds")
    print(
        "WEAK-GAIN AUDIT CHECK: OK",
        f"holdout_mean={statistics.mean(holdout_values):.3f}",
        f"q={holdout_q:.3f}",
        f"gap_50k={gaps[50_000]:.2f}%",
        f"gap_500k={gaps[500_000]:.2f}%",
    )


if __name__ == "__main__":
    main()
