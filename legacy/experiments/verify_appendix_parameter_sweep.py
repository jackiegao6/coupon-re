"""Independently verify the validated-v2.4 appendix parameter sweep."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import run_appendix_parameter_sweep as spec


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results/validated-v2/appendix-grid"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{left} != {right}")


def main() -> None:
    metadata = json.loads((RESULTS / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != spec.STATUS:
        raise AssertionError("Appendix metadata is not complete")
    if metadata["protocol_version"] != spec.PROTOCOL_VERSION:
        raise AssertionError("Appendix metadata protocol mismatch")

    jobs = sorted((RESULTS / "jobs").glob("*/*.json"))
    expected_jobs = (
        len(spec.DATASETS)
        * len(spec.TRANSFER_VALUES)
        * len(spec.REDEMPTION_SHARES)
    )
    if len(jobs) != expected_jobs:
        raise AssertionError(f"Expected {expected_jobs} jobs, found {len(jobs)}")
    for path in jobs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"] != spec.STATUS:
            raise AssertionError(f"Invalid job status: {path}")
        key = payload["key"]
        expected = spec.cell_key(
            key["dataset"],
            float(key["transfer_probability"]),
            float(key["central_redemption_share"]),
        )
        if key != expected:
            raise AssertionError(f"Job protocol mismatch: {path}")

    raw = read_csv("raw.csv")
    methods = read_csv("method_summary.csv")
    comparisons = read_csv("comparison_summary.csv")
    expected_configurations = (
        len(spec.DATASETS)
        * (
            len(spec.TRANSFER_VALUES)
            * len(spec.REDEMPTION_SHARES)
            + len(spec.TRANSFER_VALUES) * (len(spec.BUDGET_SLICE) - 1)
        )
        * len(spec.SELECTION_SEEDS)
    )
    if len(raw) != expected_configurations * len(spec.METHODS):
        raise AssertionError(f"Unexpected raw row count: {len(raw)}")
    expected_cells = expected_configurations // len(spec.SELECTION_SEEDS)
    if len(methods) != expected_cells * len(spec.METHODS):
        raise AssertionError(f"Unexpected method-summary count: {len(methods)}")
    if len(comparisons) != expected_cells:
        raise AssertionError(f"Unexpected comparison count: {len(comparisons)}")

    for row in raw + methods + comparisons:
        if (
            row["status"] != spec.STATUS
            or row["protocol_version"] != spec.PROTOCOL_VERSION
        ):
            raise AssertionError("Unvalidated row in appendix results")

    raw_counts = Counter(
        (
            row["dataset"],
            float(row["transfer_probability"]),
            float(row["central_redemption_share"]),
            int(row["k"]),
            row["method"],
        )
        for row in raw
    )
    if set(raw_counts.values()) != {len(spec.SELECTION_SEEDS)}:
        raise AssertionError("Every method cell must contain five selection runs")
    if {row["method"] for row in raw} != set(spec.METHODS):
        raise AssertionError("Method set mismatch")

    grouped: dict[tuple[object, ...], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        numeric = [
            float(row["mean_adopters"]),
            float(row["mean_redemptions"]),
            float(row["duplicate_fraction"]),
            float(row["selection_seconds"]),
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise AssertionError("Non-finite raw value")
        if numeric[0] < 0.0 or numeric[1] < numeric[0] or not 0.0 <= numeric[2] <= 1.0:
            raise AssertionError("Invalid adoption/redemption metrics")
        key = (
            row["dataset"],
            float(row["transfer_probability"]),
            float(row["central_redemption_share"]),
            int(row["k"]),
            row["method"],
        )
        grouped[key].append(row)

    method_index = {
        (
            row["dataset"],
            float(row["transfer_probability"]),
            float(row["central_redemption_share"]),
            int(row["k"]),
            row["method"],
        ): row
        for row in methods
    }
    for key, rows in grouped.items():
        summary = method_index[key]
        for raw_field, summary_field in [
            ("mean_adopters", "mean_mean_adopters"),
            ("mean_redemptions", "mean_mean_redemptions"),
            ("duplicate_fraction", "mean_duplicate_fraction"),
            ("selection_seconds", "mean_selection_seconds"),
        ]:
            close(
                float(summary[summary_field]),
                statistics.mean(float(row[raw_field]) for row in rows),
            )

    comparison_index = {
        (
            row["dataset"],
            float(row["transfer_probability"]),
            float(row["central_redemption_share"]),
            int(row["k"]),
        ): row
        for row in comparisons
    }
    for cell, comparison in comparison_index.items():
        by_method = {
            method: method_index[(*cell, method)] for method in spec.METHODS
        }
        best_method = max(
            spec.BASELINES,
            key=lambda method: float(by_method[method]["mean_mean_adopters"]),
        )
        if comparison["best_baseline"] != best_method:
            raise AssertionError("Best-baseline identity mismatch")
        cim = float(by_method["CIM-RIS"]["mean_mean_adopters"])
        best = float(by_method[best_method]["mean_mean_adopters"])
        close(
            float(comparison["relative_gain_best_pct"]),
            100.0 * (cim - best) / best,
        )
        duplicate_delta = 100.0 * (
            float(by_method[best_method]["mean_duplicate_fraction"])
            - float(by_method["CIM-RIS"]["mean_duplicate_fraction"])
        )
        close(float(comparison["duplicate_reduction_pp"]), duplicate_delta)

    manifest = {}
    for line in (RESULTS / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        manifest[name] = digest
    for name in ["raw.csv", "method_summary.csv", "comparison_summary.csv"]:
        observed = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        if manifest.get(name) != observed:
            raise AssertionError(f"Checksum mismatch: {name}")

    grid = [row for row in comparisons if int(row["k"]) == spec.GRID_BUDGET]
    wins = Counter()
    for row in grid:
        wins[row["dataset"]] += int(float(row["relative_gain_best_pct"]) > 0.0)
    if wins != Counter({"NetFacebookEgo": 16, "Netscience": 1}):
        raise AssertionError(f"Unexpected k=100 win counts: {wins}")

    by_dataset = {
        dataset: [row for row in grid if row["dataset"] == dataset]
        for dataset in spec.DATASETS
    }
    facebook = by_dataset["NetFacebookEgo"]
    netscience = by_dataset["Netscience"]
    close(
        statistics.mean(float(row["relative_gain_best_pct"]) for row in facebook),
        0.5015673507215298,
    )
    close(
        statistics.mean(float(row["relative_gain_best_pct"]) for row in netscience),
        -1.6193430119109304,
    )
    close(
        max(float(row["relative_gain_best_pct"]) for row in facebook),
        2.2412262341234865,
    )
    close(
        min(float(row["relative_gain_best_pct"]) for row in facebook),
        -3.4804936992922832,
    )
    close(
        min(float(row["relative_gain_best_pct"]) for row in netscience),
        -4.743464306288405,
    )
    close(
        max(float(row["relative_gain_best_pct"]) for row in netscience),
        0.6329052011701038,
    )
    for dataset_rows in by_dataset.values():
        for baseline in ["DegreeTopM", "PageRank"]:
            if not all(float(row[f"gain_vs_{baseline}_pct"]) > 0.0 for row in dataset_rows):
                raise AssertionError(f"CIM-RIS does not beat {baseline} in every grid cell")

    expected_slice = {
        ("NetFacebookEgo", 25): (0, -1.0793726178762502),
        ("NetFacebookEgo", 100): (4, 0.8805384273837514),
        ("NetFacebookEgo", 200): (5, 1.2633072633721183),
        ("Netscience", 25): (0, -2.3131277528373415),
        ("Netscience", 100): (0, -0.8384733320931568),
        ("Netscience", 200): (0, -1.036720861007594),
    }
    for (dataset, budget), (expected_wins, expected_mean) in expected_slice.items():
        selected = [
            row
            for row in comparisons
            if row["dataset"] == dataset
            and int(row["k"]) == budget
            and math.isclose(
                float(row["central_redemption_share"]),
                spec.BUDGET_SLICE_SHARE,
            )
        ]
        observed = [float(row["relative_gain_best_pct"]) for row in selected]
        if sum(value > 0.0 for value in observed) != expected_wins:
            raise AssertionError("Budget-slice win count mismatch")
        close(statistics.mean(observed), expected_mean)

    gains = [float(row["relative_gain_best_pct"]) for row in grid]
    duplicate = [float(row["duplicate_reduction_pp"]) for row in grid]
    gain_mean = statistics.mean(gains)
    duplicate_mean = statistics.mean(duplicate)
    correlation = sum(
        (gain - gain_mean) * (overlap - duplicate_mean)
        for gain, overlap in zip(gains, duplicate)
    ) / math.sqrt(
        sum((gain - gain_mean) ** 2 for gain in gains)
        * sum((overlap - duplicate_mean) ** 2 for overlap in duplicate)
    )
    close(correlation, 0.03232018744262757)

    manuscript = (ROOT / "paper-v2 copy.tex").read_text(encoding="utf-8")
    required_claims = [
        "baseline in 16 of 20 cells",
        "has higher mean spread in only one cell",
        "$-1.62\\%$, and ranges",
        "has the higher mean in none of the five transfer",
        "Pearson correlation with the",
        "relative spread difference is only $0.03$",
    ]
    for claim in required_claims:
        if claim not in manuscript:
            raise AssertionError(f"Missing appendix claim text: {claim}")
    print(
        "APPENDIX PARAMETER SWEEP CHECK: OK "
        f"({len(raw)} raw rows; wins={dict(wins)})"
    )


if __name__ == "__main__":
    main()
