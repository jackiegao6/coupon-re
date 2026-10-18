"""Verify completeness and all rounded manuscript claims for validated-v2."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results/validated-v2"
DATASETS = ["Netscience", "NetFacebookEgo", "DoubanRandom", "EmailEnron"]
SCENARIOS = ["balanced", "adoption-heavy", "forwarding-heavy"]
BUDGETS = [10, 25, 50, 100, 150, 200]


def close(value: float, expected: float, tolerance: float = 0.06) -> None:
    if abs(value - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, observed {value}")


def main() -> None:
    metadata = json.loads((RESULTS / "study_metadata.json").read_text())
    jobs = list((RESULTS / "jobs").glob("*/*/*.json"))
    raw = list(csv.DictReader((RESULTS / "validated_raw.csv").open()))
    summary = list(csv.DictReader((RESULTS / "validated_summary.csv").open()))
    if metadata["status"] != "REAL_EXPERIMENT" or metadata["completed_jobs"] != 360:
        raise AssertionError("Study metadata is incomplete")
    if len(jobs) != 360 or len(raw) != 2610 or len(summary) != 522:
        raise AssertionError("Unexpected validated-v2 artifact counts")
    if Counter(row["status"] for row in raw) != {"REAL_EXPERIMENT": 2610}:
        raise AssertionError("Non-real row detected")
    if Counter(row["protocol_version"] for row in raw) != {"validated-v2.2": 2610}:
        raise AssertionError("Protocol mismatch")
    if Counter(row["selection_repeats"] for row in summary) != {"5": 522}:
        raise AssertionError("Missing repeated selections")

    index = {
        (row["dataset"], row["scenario"], int(row["k"]), row["method"]): row
        for row in summary
    }
    expected_table = {
        "balanced": (-2.3, 5.8, -0.1, 12.6, 13.3),
        "adoption-heavy": (-2.3, -0.6, 1.4, 9.5, 8.9),
        "forwarding-heavy": (-3.4, 8.5, -0.7, 25.5, 30.2),
    }
    baselines = ["MC-Greedy", "1Hop-Sort", "IC-RIS", "DegreeTopM", "PageRank"]
    for scenario in SCENARIOS:
        for baseline, expected in zip(baselines, expected_table[scenario]):
            datasets = ["Netscience"] if baseline == "MC-Greedy" else DATASETS
            differences = []
            for dataset in datasets:
                for k in BUDGETS:
                    cim = float(index[(dataset, scenario, k, "CIM-RIS")]["mean_adopters"])
                    other = float(index[(dataset, scenario, k, baseline)]["mean_adopters"])
                    differences.append(100.0 * (cim - other) / other)
            close(statistics.mean(differences), expected)

    oracle_gaps = []
    degree_wins = 0
    pagerank_wins = 0
    cim_cvs = []
    raw_cim: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in raw:
        if row["method"] == "CIM-RIS":
            raw_cim[(row["dataset"], row["scenario"], int(row["k"]))].append(
                float(row["mean_adopters"])
            )
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            for k in BUDGETS:
                cim = float(index[(dataset, scenario, k, "CIM-RIS")]["mean_adopters"])
                cim_std = float(
                    index[(dataset, scenario, k, "CIM-RIS")]["std_across_selection_runs"]
                )
                cim_cvs.append(cim_std / cim)
                degree_wins += cim > float(
                    index[(dataset, scenario, k, "DegreeTopM")]["mean_adopters"]
                )
                pagerank_wins += cim > float(
                    index[(dataset, scenario, k, "PageRank")]["mean_adopters"]
                )
                if dataset == "Netscience":
                    oracle = float(
                        index[(dataset, scenario, k, "MC-Greedy")]["mean_adopters"]
                    )
                    oracle_gaps.append(100.0 * (oracle - cim) / oracle)
    close(min(oracle_gaps), 1.1)
    close(max(oracle_gaps), 4.4)
    if degree_wins != 72 or pagerank_wins != 72:
        raise AssertionError("Topology-baseline win count changed")
    close(100.0 * statistics.mean(cim_cvs), 0.51)
    close(100.0 * max(cim_cvs), 3.07)
    ranges = [(max(v) - min(v)) / statistics.mean(v) for v in raw_cim.values()]
    close(100.0 * statistics.mean(ranges), 1.27)

    scalability = list(
        csv.DictReader((RESULTS / "validated_scalability.csv").open())
    )
    douban_200 = [
        float(row["selection_seconds"])
        for row in scalability
        if int(row["k"]) == 200
    ]
    close(statistics.mean(douban_200), 31.2)
    email_200 = float(
        index[("EmailEnron", "balanced", 200, "CIM-RIS")]["mean_selection_seconds"]
    )
    close(email_200, 51.9)

    sampler = list(
        csv.DictReader((RESULTS / "validated_sampler_ablation.csv").open())
    )
    for dataset, expected in [("Netscience", 1.30), ("EmailEnron", 1.53)]:
        uniform = next(
            row
            for row in sampler
            if row["dataset"] == dataset
            and int(row["k"]) == 10
            and row["sampler"] == "Uniform root"
        )
        conditioned = next(
            row
            for row in sampler
            if row["dataset"] == dataset
            and int(row["k"]) == 10
            and row["sampler"] == "Conditioned root"
        )
        uniform_cost = float(uniform["std_spread_estimate"]) ** 2 * float(
            uniform["mean_batch_seconds"]
        )
        conditioned_cost = float(conditioned["std_spread_estimate"]) ** 2 * float(
            conditioned["mean_batch_seconds"]
        )
        close(uniform_cost / conditioned_cost, expected, tolerance=0.015)
    print("VALIDATED-V2 CLAIM CHECK: OK")


if __name__ == "__main__":
    main()
