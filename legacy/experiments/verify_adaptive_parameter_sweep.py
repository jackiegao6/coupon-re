"""Independently verify the complete validated-v2.5 adaptive sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import run_adaptive_parameter_sweep as spec
import run_appendix_parameter_sweep as v24
import run_real_submission as core


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments/results/validated-v2/adaptive-grid"


def read_csv(results: Path, name: str) -> list[dict[str, str]]:
    with (results / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{left} != {right}")


def verify_trace(payload: dict[str, object]) -> None:
    key = payload["key"]
    trace = payload["trace"]
    dataset = str(key["dataset"])
    transfer_probability = float(key["transfer_probability"])
    redemption_share = float(key["central_redemption_share"])
    k = int(key["k"])
    selection_seed = int(key["selection_seed"])
    graph = core.load_graph(dataset)
    alpha, _, _ = v24.grid_probabilities(
        graph, transfer_probability, redemption_share
    )
    adoption_sum = float(alpha.sum())
    root_weight_sum = float((1.0 - (1.0 - alpha) ** k).sum())
    raw_floor = (
        spec.TARGET_OBSERVATIONS_PER_CANDIDATE
        * graph.n
        * root_weight_sum
        / adoption_sum
    )
    floor = min(
        spec.FLOOR_CAP,
        max(spec.INITIAL_SAMPLES, spec.rounded_up(raw_floor, spec.SAMPLE_ROUNDING)),
    )
    close(float(trace["raw_sample_floor"]), raw_floor)
    if int(trace["sample_floor"]) != floor:
        raise AssertionError("Sample floor mismatch")

    expected_rr_stream = spec.stream_id(
        dataset,
        f"{transfer_probability:.2f}",
        f"{redemption_share:.2f}",
        k,
        selection_seed,
        "cim-ris",
    )
    expected_final_stream = spec.stream_id(
        dataset,
        f"{transfer_probability:.2f}",
        f"{redemption_share:.2f}",
        k,
        selection_seed,
        "final-evaluation",
    )
    stages = trace["stages"]
    if not stages or int(stages[0]["rr_samples"]) != spec.INITIAL_SAMPLES:
        raise AssertionError("Invalid initial adaptive stage")
    validation_streams: set[int] = set()
    for index, stage in enumerate(stages):
        samples = int(stage["rr_samples"])
        if int(stage["rr_stream_id"]) != expected_rr_stream:
            raise AssertionError("RR stages do not share one nested stream")
        allocation = stage["allocation"]
        if len(allocation) != k or len(set(allocation)) != k:
            raise AssertionError("CIM allocation violates the unit-capacity policy")
        if not all(0 <= int(node) < graph.n for node in allocation):
            raise AssertionError("CIM allocation contains an invalid node")
        if index == 0:
            if stage["validation_stream_id"] is not None:
                raise AssertionError("First stage must not have validation")
        else:
            previous_samples = int(stages[index - 1]["rr_samples"])
            expected_samples = (
                min(previous_samples * 2, floor)
                if previous_samples < floor
                else min(previous_samples * 2, spec.HARD_CAP)
            )
            if samples != expected_samples:
                raise AssertionError("Adaptive sample sequence mismatch")
            expected_validation = spec.stream_id(
                dataset,
                f"{transfer_probability:.2f}",
                f"{redemption_share:.2f}",
                k,
                selection_seed,
                samples,
                "stability-validation",
            )
            if int(stage["validation_stream_id"]) != expected_validation:
                raise AssertionError("Validation stream mismatch")
            validation_streams.add(expected_validation)
            relative = float(stage["validation_relative_change_pct"])
            if bool(stage["stable"]) != (
                abs(relative) <= spec.STABILITY_TOLERANCE_PCT
            ):
                raise AssertionError("Stability flag mismatch")
        if index < len(stages) - 1 and (
            samples >= floor and index > 0 and bool(stage["stable"])
        ):
            raise AssertionError("Adaptive selection continued after convergence")

    final = stages[-1]
    terminal = (
        int(final["rr_samples"]) >= floor
        and len(stages) > 1
        and bool(final["stable"])
    ) or int(final["rr_samples"]) >= spec.HARD_CAP
    if not terminal:
        raise AssertionError("Adaptive selection stopped before a terminal condition")
    hard_cap_reached = int(final["rr_samples"]) >= spec.HARD_CAP
    unresolved = hard_cap_reached and not bool(final["stable"])
    if bool(trace["hard_cap_reached"]) != hard_cap_reached:
        raise AssertionError("Hard-cap flag mismatch")
    if bool(trace["unresolved_instability"]) != unresolved:
        raise AssertionError("Unresolved-instability flag mismatch")
    if int(trace["final_evaluation_stream_id"]) != expected_final_stream:
        raise AssertionError("Final evaluation stream mismatch")
    forbidden = validation_streams | {
        expected_rr_stream,
        int(trace["ic_rr_stream_id"]),
    }
    if expected_final_stream in forbidden:
        raise AssertionError("Final evaluation reuses a selection or validation stream")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = args.results
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != spec.STATUS:
        raise AssertionError("Adaptive metadata is not complete")
    if metadata.get("protocol_version") != spec.PROTOCOL_VERSION:
        raise AssertionError("Adaptive metadata protocol mismatch")

    jobs = spec.expected_jobs()
    paths = [spec.job_path(results, *job) for job in jobs]
    observed = set((results / "jobs").rglob("*.json"))
    if observed != set(paths):
        raise AssertionError(
            f"Expected {len(paths)} jobs, found {len(observed)} matching files"
        )
    for job, path in zip(jobs, paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != spec.STATUS:
            raise AssertionError(f"Invalid job status: {path}")
        if payload.get("key") != spec.job_key(*job):
            raise AssertionError(f"Job protocol mismatch: {path}")
        if len(payload.get("rows", [])) != len(spec.METHODS):
            raise AssertionError(f"Method row count mismatch: {path}")
        verify_trace(payload)

    raw = read_csv(results, "raw.csv")
    methods = read_csv(results, "method_summary.csv")
    comparisons = read_csv(results, "comparison_summary.csv")
    if len(raw) != 1_800 or len(methods) != 360 or len(comparisons) != 60:
        raise AssertionError(
            f"Unexpected aggregate sizes: {len(raw)}, {len(methods)}, {len(comparisons)}"
        )
    if metadata.get("completed_selection_jobs") != 300:
        raise AssertionError("Metadata selection-job count mismatch")

    raw_counts = Counter(
        (
            row["dataset"],
            float(row["transfer_probability"]),
            float(row["central_redemption_share"]),
            int(row["k"]),
            int(row["selection_seed"]),
            row["method"],
        )
        for row in raw
    )
    if set(raw_counts.values()) != {1} or len(raw_counts) != len(raw):
        raise AssertionError("Raw rows are missing or duplicated")
    if {row["method"] for row in raw} != set(spec.METHODS):
        raise AssertionError("Method set mismatch")

    grouped: dict[tuple[object, ...], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        if row["status"] != spec.STATUS or row["protocol_version"] != spec.PROTOCOL_VERSION:
            raise AssertionError("Unvalidated aggregate row")
        adopters = float(row["mean_adopters"])
        redemptions = float(row["mean_redemptions"])
        duplicate = float(row["duplicate_fraction"])
        if not all(math.isfinite(value) for value in (adopters, redemptions, duplicate)):
            raise AssertionError("Non-finite result")
        if adopters < 0.0 or redemptions < adopters or not 0.0 <= duplicate <= 1.0:
            raise AssertionError("Invalid spread or redemption metric")
        grouped[
            (
                row["dataset"],
                float(row["transfer_probability"]),
                float(row["central_redemption_share"]),
                int(row["k"]),
                row["method"],
            )
        ].append(row)

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
        if len(rows) != len(spec.SELECTION_SEEDS):
            raise AssertionError("Method summary does not have five selection repeats")
        summary = method_index[key]
        for raw_field, summary_field in (
            ("mean_adopters", "mean_mean_adopters"),
            ("mean_redemptions", "mean_mean_redemptions"),
            ("duplicate_fraction", "mean_duplicate_fraction"),
            ("selection_seconds", "mean_selection_seconds"),
            ("final_rr_samples", "mean_final_rr_samples"),
        ):
            close(
                float(summary[summary_field]),
                statistics.mean(float(row[raw_field]) for row in rows),
            )

    for row in comparisons:
        cell = (
            row["dataset"],
            float(row["transfer_probability"]),
            float(row["central_redemption_share"]),
            int(row["k"]),
        )
        by_method = {method: method_index[(*cell, method)] for method in spec.METHODS}
        best_method = max(
            spec.BASELINES,
            key=lambda method: float(by_method[method]["mean_mean_adopters"]),
        )
        if row["best_baseline"] != best_method:
            raise AssertionError("Best-baseline identity mismatch")
        cim = float(by_method["CIM-RIS"]["mean_mean_adopters"])
        best = float(by_method[best_method]["mean_mean_adopters"])
        close(float(row["relative_gain_best_pct"]), 100.0 * (cim - best) / best)

    manifest = {}
    for line in (results / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        manifest[name] = digest
    for name in ("raw.csv", "method_summary.csv", "comparison_summary.csv", "traces.json"):
        artifact_bytes = (results / name).read_bytes()
        observed_digest = hashlib.sha256(artifact_bytes).hexdigest()
        if manifest.get(name) != observed_digest:
            # A Windows checkout can expand LF in JSON without changing data.
            # Accept only this exact, reported transformation, never arbitrary
            # reparsing/reserialization that could hide content changes.
            lf_digest = hashlib.sha256(artifact_bytes.replace(b"\r\n", b"\n")).hexdigest()
            if name.endswith(".json") and manifest.get(name) == lf_digest:
                print(f"Checksum verified after CRLF-to-LF normalization: {name}")
            else:
                raise AssertionError(f"Checksum mismatch: {name}")

    print(
        "validated-v2.5 verified: 300 jobs, 1800 raw rows, "
        "60 complete budget configurations"
    )


if __name__ == "__main__":
    main()
