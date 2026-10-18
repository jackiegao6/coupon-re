"""Verify completeness and manuscript-level claims for v2.3 extensions."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results/validated-v2/extensions"
PROTOCOL = "validated-v2.3-extension"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        output = list(csv.DictReader(handle))
    if not output:
        raise AssertionError(f"Empty extension file: {name}")
    if any(row["status"] != "REAL_EXPERIMENT" for row in output):
        raise AssertionError(f"Non-real row in {name}")
    if any(row["protocol_version"] != PROTOCOL for row in output):
        raise AssertionError(f"Protocol mismatch in {name}")
    return output


def close(observed: float, expected: float, tolerance: float = 0.06) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, observed {observed}")


def main() -> None:
    metadata = json.loads((RESULTS / "extension_metadata.json").read_text())
    if metadata["status"] != "REAL_EXPERIMENT":
        raise AssertionError("Extension metadata is incomplete")
    expected_counts = {
        "oracle_matrices": 6,
        "strong_benchmark_raw_rows": 180,
        "capacity_raw_rows": 90,
        "sensitivity_raw_rows": 120,
    }
    for key, value in expected_counts.items():
        if metadata.get(key) != value:
            raise AssertionError(f"Unexpected {key}: {metadata.get(key)}")

    oracle = rows("oracle_diagnostics.csv")
    benchmark_raw = rows("strong_benchmark_raw.csv")
    benchmark = rows("strong_benchmark_summary.csv")
    capacity_raw = rows("capacity_raw.csv")
    capacity = rows("capacity_summary.csv")
    sensitivity_raw = rows("sensitivity_raw.csv")
    sensitivity = rows("sensitivity_summary.csv")
    if [len(value) for value in [oracle, benchmark_raw, benchmark, capacity_raw,
                                  capacity, sensitivity_raw, sensitivity]] != [
        6, 180, 36, 90, 18, 120, 24
    ]:
        raise AssertionError("Unexpected extension CSV row counts")
    for summary in [benchmark, capacity, sensitivity]:
        if any(row["selection_repeats"] != "5" for row in summary):
            raise AssertionError("A summary is missing five selection repeats")
    facebook_oracles = [row for row in oracle if row["dataset"] == "NetFacebookEgo"]
    if len(facebook_oracles) != 3 or any(
        row["trajectories_per_source"] != "50000" for row in facebook_oracles
    ):
        raise AssertionError("NetFacebookEgo oracle coverage is incomplete")

    def benchmark_values(dataset: str, scenario: str) -> list[float]:
        return [
            float(row["mean_gap_to_mc_greedy_percent"])
            for row in benchmark
            if row["dataset"] == dataset and row["scenario"] == scenario
        ]

    close(statistics.mean(benchmark_values("NetFacebookEgo", "balanced")), 1.37)
    close(
        statistics.mean(benchmark_values("NetFacebookEgo", "adoption-heavy")),
        0.38,
    )
    forwarding = benchmark_values("NetFacebookEgo", "forwarding-heavy")
    close(min(forwarding), 4.40)
    close(max(forwarding), 11.10)

    by_configuration: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in sensitivity:
        key = (row["dataset"], int(row["k"]))
        by_configuration.setdefault(key, []).append(
            (
                int(row["rr_samples"]),
                float(row["mean_gap_to_mc_greedy_percent"]),
            )
        )
    if len(by_configuration) != 6:
        raise AssertionError("Unexpected sensitivity configuration count")
    for values in by_configuration.values():
        ordered = [gap for _, gap in sorted(values)]
        if any(later >= earlier for earlier, later in zip(ordered, ordered[1:])):
            raise AssertionError("A sensitivity gap does not decrease with samples")
    mean_by_samples = {}
    time_by_samples = {}
    for sample_count in [5_000, 20_000, 50_000, 100_000]:
        values = [
            float(row["mean_gap_to_mc_greedy_percent"])
            for row in sensitivity
            if int(row["rr_samples"]) == sample_count
        ]
        mean_by_samples[sample_count] = statistics.mean(values)
        time_by_samples[sample_count] = statistics.mean(
            float(row["mean_selection_seconds"])
            for row in sensitivity
            if int(row["rr_samples"]) == sample_count
        )
    close(mean_by_samples[5_000], 8.90)
    close(mean_by_samples[20_000], 7.10)
    close(mean_by_samples[50_000], 6.22)
    close(mean_by_samples[100_000], 5.09)
    close(time_by_samples[5_000], 0.266, tolerance=0.006)
    close(time_by_samples[20_000], 1.016, tolerance=0.006)
    close(time_by_samples[50_000], 2.435, tolerance=0.006)
    close(time_by_samples[100_000], 4.741, tolerance=0.006)

    capacity_index = {
        (row["dataset"], row["scenario"], row["capacity_policy"]): row
        for row in capacity
    }
    expected_capacity = {
        ("Netscience", "capacity-2"): (1.60, 49.6),
        ("Netscience", "unrestricted"): (2.09, 63.4),
        ("NetFacebookEgo", "capacity-2"): (0.24, 7.8),
        ("NetFacebookEgo", "unrestricted"): (0.25, 6.8),
    }
    for (dataset, policy), (expected_gain, expected_repeats) in expected_capacity.items():
        base = float(
            capacity_index[(dataset, "forwarding-heavy", "distinct")][
                "mean_cim_mean_adopters"
            ]
        )
        row = capacity_index[(dataset, "forwarding-heavy", policy)]
        observed = float(row["mean_cim_mean_adopters"])
        close(100.0 * (observed - base) / base, expected_gain)
        close(float(row["mean_cim_repeated_placements"]), expected_repeats)
    print("VALIDATED-V2.3 EXTENSION CLAIM CHECK: OK")


if __name__ == "__main__":
    main()
