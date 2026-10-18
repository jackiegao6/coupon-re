"""Run the prespecified appendix parameter-regime sweep.

The sweep keeps every transfer-by-redemption-share cell and writes one
resumable JSON job per dataset and parameter pair. Only outputs marked with
PROTOCOL_VERSION and REAL_EXPERIMENT are accepted during aggregation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_real_submission as core


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "validated-v2.4-appendix-grid"
STATUS = "REAL_EXPERIMENT"
DATASETS = ["Netscience", "NetFacebookEgo"]
TRANSFER_VALUES = [0.30, 0.50, 0.70, 0.85, 0.93]
REDEMPTION_SHARES = [0.20, 0.40, 0.60, 0.80]
GRID_BUDGET = 100
BUDGET_SLICE_SHARE = 0.60
BUDGET_SLICE = [25, 100, 200]
SELECTION_SEEDS = [20260715, 20260716, 20260717, 20260718, 20260719]
RR_SAMPLES = 20_000
EVALUATION_SIMULATIONS = 2_000
DEGREE_CONTRAST = 0.20
T_CRITICAL_95_DF4 = 2.7764451051977987
METHODS = [
    "CIM-RIS",
    "IC-RIS",
    "1Hop-Sort",
    "Alpha-Sort",
    "DegreeTopM",
    "PageRank",
]
BASELINES = [method for method in METHODS if method != "CIM-RIS"]


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def hardware_metadata() -> dict[str, object]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
    }


def grid_probabilities(
    graph: core.Graph,
    transfer_probability: float,
    central_redemption_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_degree = np.log1p(graph.degrees.astype(np.float64))
    denominator = max(float(np.max(log_degree)), 1.0)
    normalized = np.clip(log_degree / denominator, 0.0, 1.0)
    redemption_share = np.clip(
        central_redemption_share + DEGREE_CONTRAST * (0.5 - normalized),
        0.05,
        0.95,
    )
    transfer = np.full(graph.n, transfer_probability, dtype=np.float64)
    alpha = (1.0 - transfer) * redemption_share
    discard = (1.0 - transfer) * (1.0 - redemption_share)
    isolated = graph.degrees == 0
    discard[isolated] += transfer[isolated]
    transfer[isolated] = 0.0
    if not np.allclose(alpha + discard + transfer, 1.0):
        raise AssertionError("Grid probabilities do not sum to one")
    if np.any(alpha < 0.0) or np.any(discard < 0.0) or np.any(transfer < 0.0):
        raise AssertionError("Grid probabilities must be nonnegative")
    return alpha, discard, transfer


def cell_budgets(redemption_share: float) -> list[int]:
    if math.isclose(redemption_share, BUDGET_SLICE_SHARE):
        return BUDGET_SLICE
    return [GRID_BUDGET]


def evaluation_streams(
    dataset: str,
    transfer_probability: float,
    redemption_share: float,
    k: int,
    selection_seed: int,
) -> list[int]:
    master = core._stable_seed(
        PROTOCOL_VERSION,
        dataset,
        f"{transfer_probability:.2f}",
        f"{redemption_share:.2f}",
        k,
        selection_seed,
        "forward-evaluation",
    )
    rng = random.Random(master)
    return [rng.getrandbits(64) for _ in range(EVALUATION_SIMULATIONS)]


def duplicate_fraction(mean_adopters: float, mean_redemptions: float) -> float:
    if mean_redemptions <= 0.0:
        return 0.0
    return max(0.0, (mean_redemptions - mean_adopters) / mean_redemptions)


def cell_key(
    dataset: str,
    transfer_probability: float,
    redemption_share: float,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "transfer_probability": transfer_probability,
        "central_redemption_share": redemption_share,
        "degree_contrast": DEGREE_CONTRAST,
        "budgets": cell_budgets(redemption_share),
        "selection_seeds": SELECTION_SEEDS,
        "rr_samples": RR_SAMPLES,
        "evaluation_simulations": EVALUATION_SIMULATIONS,
        "methods": METHODS,
        "capacity_per_node": 1,
    }


def job_path(
    output_dir: Path,
    dataset: str,
    transfer_probability: float,
    redemption_share: float,
) -> Path:
    return (
        output_dir
        / "jobs"
        / dataset
        / (
            f"transfer-{transfer_probability:.2f}_"
            f"redemption-{redemption_share:.2f}.json"
        )
    )


def run_cell(
    dataset: str,
    transfer_probability: float,
    redemption_share: float,
) -> dict[str, object]:
    graph = core.load_graph(dataset)
    alpha, discard, transfer = grid_probabilities(
        graph, transfer_probability, redemption_share
    )
    static_orders = core.static_orders(
        graph,
        alpha,
        transfer,
        core._stable_seed(
            PROTOCOL_VERSION,
            dataset,
            transfer_probability,
            redemption_share,
            "static",
        ),
    )
    static_orders = {
        method: order for method, order in static_orders.items() if method in METHODS
    }

    rows: list[dict[str, object]] = []
    budgets = cell_budgets(redemption_share)
    max_budget = max(budgets)
    for selection_seed in SELECTION_SEEDS:
        ic_order, ic_seconds = core.ic_ris_order(
            graph,
            transfer,
            max_budget,
            RR_SAMPLES,
            core._stable_seed(
                PROTOCOL_VERSION,
                dataset,
                transfer_probability,
                redemption_share,
                selection_seed,
                "ic-ris",
            ),
        )
        for k in budgets:
            cim_seeds, cim_seconds, estimated, memberships = core.cim_ris_seeds(
                graph,
                alpha,
                discard,
                k,
                RR_SAMPLES,
                core._stable_seed(
                    PROTOCOL_VERSION,
                    dataset,
                    transfer_probability,
                    redemption_share,
                    k,
                    selection_seed,
                    "cim-ris",
                ),
            )
            orders = {
                **static_orders,
                "IC-RIS": ic_order,
                "CIM-RIS": cim_seeds,
            }
            streams = evaluation_streams(
                dataset,
                transfer_probability,
                redemption_share,
                k,
                selection_seed,
            )
            for method in METHODS:
                allocation = orders[method][:k]
                mean, ci95, variance, redemptions = core.evaluate_with_streams(
                    graph, allocation, alpha, discard, streams
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "transfer_probability": transfer_probability,
                        "central_redemption_share": redemption_share,
                        "k": k,
                        "selection_seed": selection_seed,
                        "method": method,
                        "mean_adopters": mean,
                        "ci95": ci95,
                        "forward_variance": variance,
                        "mean_redemptions": redemptions,
                        "duplicate_fraction": duplicate_fraction(mean, redemptions),
                        "selection_seconds": (
                            cim_seconds
                            if method == "CIM-RIS"
                            else ic_seconds
                            if method == "IC-RIS"
                            else 0.0
                        ),
                        "cim_estimated_spread": (
                            estimated if method == "CIM-RIS" else None
                        ),
                        "rr_memberships": (
                            memberships if method == "CIM-RIS" else None
                        ),
                        "rr_samples": RR_SAMPLES if method in {"CIM-RIS", "IC-RIS"} else 0,
                        "evaluation_simulations": EVALUATION_SIMULATIONS,
                        "status": STATUS,
                        "protocol_version": PROTOCOL_VERSION,
                    }
                )
    return {
        "status": STATUS,
        "key": cell_key(dataset, transfer_probability, redemption_share),
        "probability_diagnostics": {
            "min_adoption": float(np.min(alpha)),
            "max_adoption": float(np.max(alpha)),
            "min_discard": float(np.min(discard)),
            "max_discard": float(np.max(discard)),
            "mean_transfer": float(np.mean(transfer)),
        },
        "rows": rows,
    }


def summarize_methods(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "dataset",
        "transfer_probability",
        "central_redemption_share",
        "k",
        "method",
    ]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    summary: list[dict[str, object]] = []
    value_fields = [
        "mean_adopters",
        "mean_redemptions",
        "duplicate_fraction",
        "selection_seconds",
    ]
    for group_key, group_rows in sorted(groups.items()):
        record = dict(zip(keys, group_key))
        record["selection_repeats"] = len(group_rows)
        for field in value_fields:
            values = [float(row[field]) for row in group_rows]
            record[f"mean_{field}"] = statistics.mean(values)
            record[f"std_{field}"] = statistics.stdev(values)
        record["status"] = STATUS
        record["protocol_version"] = PROTOCOL_VERSION
        summary.append(record)
    return summary


def summarize_comparisons(
    raw_rows: list[dict[str, object]],
    method_summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    cell_keys = [
        "dataset",
        "transfer_probability",
        "central_redemption_share",
        "k",
    ]
    summaries: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in method_summary:
        key = tuple(row[field] for field in cell_keys)
        summaries[key][str(row["method"])] = row

    raw_index: dict[
        tuple[object, ...], dict[int, dict[str, dict[str, object]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in raw_rows:
        key = tuple(row[field] for field in cell_keys)
        raw_index[key][int(row["selection_seed"])][str(row["method"])] = row

    output: list[dict[str, object]] = []
    for key, by_method in sorted(summaries.items()):
        cim = by_method["CIM-RIS"]
        best_method = max(
            BASELINES,
            key=lambda method: float(by_method[method]["mean_mean_adopters"]),
        )
        best = by_method[best_method]
        cim_spread = float(cim["mean_mean_adopters"])
        best_spread = float(best["mean_mean_adopters"])
        relative_gain = (
            100.0 * (cim_spread - best_spread) / best_spread
            if best_spread > 0.0
            else 0.0
        )
        paired_gains: list[float] = []
        for selection_seed in SELECTION_SEEDS:
            seed_rows = raw_index[key][selection_seed]
            cim_value = float(seed_rows["CIM-RIS"]["mean_adopters"])
            baseline_value = float(seed_rows[best_method]["mean_adopters"])
            paired_gains.append(
                100.0 * (cim_value - baseline_value) / baseline_value
                if baseline_value > 0.0
                else 0.0
            )
        record = dict(zip(cell_keys, key))
        record.update(
            {
                "best_baseline": best_method,
                "cim_mean_adopters": cim_spread,
                "best_baseline_mean_adopters": best_spread,
                "relative_gain_best_pct": relative_gain,
                "paired_gain_mean_pct": statistics.mean(paired_gains),
                "paired_gain_std_pct": statistics.stdev(paired_gains),
                "paired_gain_ci95_pct": (
                    T_CRITICAL_95_DF4
                    * statistics.stdev(paired_gains)
                    / math.sqrt(len(paired_gains))
                ),
                "cim_duplicate_fraction": float(
                    cim["mean_duplicate_fraction"]
                ),
                "best_baseline_duplicate_fraction": float(
                    best["mean_duplicate_fraction"]
                ),
                "duplicate_reduction_pp": 100.0
                * (
                    float(best["mean_duplicate_fraction"])
                    - float(cim["mean_duplicate_fraction"])
                ),
                "cim_wins": int(relative_gain > 0.0),
                "selection_repeats": len(SELECTION_SEEDS),
                "status": STATUS,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        for baseline in BASELINES:
            baseline_spread = float(by_method[baseline]["mean_mean_adopters"])
            record[f"gain_vs_{baseline}_pct"] = (
                100.0 * (cim_spread - baseline_spread) / baseline_spread
                if baseline_spread > 0.0
                else 0.0
            )
        output.append(record)
    return output


def write_manifest(output_dir: Path, paths: list[Path]) -> None:
    lines = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (output_dir / "manifest.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments/results/validated-v2/appendix-grid"),
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    write_json_atomic(
        metadata_path,
        {
            "status": "RUNNING",
            "protocol_version": PROTOCOL_VERSION,
            "datasets": DATASETS,
            "transfer_values": TRANSFER_VALUES,
            "central_redemption_shares": REDEMPTION_SHARES,
            "grid_budget": GRID_BUDGET,
            "budget_slice_share": BUDGET_SLICE_SHARE,
            "budget_slice": BUDGET_SLICE,
            "selection_seeds": SELECTION_SEEDS,
            "rr_samples": RR_SAMPLES,
            "evaluation_simulations": EVALUATION_SIMULATIONS,
            "degree_contrast": DEGREE_CONTRAST,
            "methods": METHODS,
            "workers": args.workers,
            "hardware": hardware_metadata(),
        },
    )

    started = time.perf_counter()
    payloads: list[dict[str, object]] = []
    pending: list[tuple[str, float, float]] = []
    for dataset in DATASETS:
        for transfer_probability in TRANSFER_VALUES:
            for redemption_share in REDEMPTION_SHARES:
                path = job_path(
                    output_dir, dataset, transfer_probability, redemption_share
                )
                expected = cell_key(dataset, transfer_probability, redemption_share)
                if path.exists() and not args.force:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload.get("status") != STATUS or payload.get("key") != expected:
                        raise RuntimeError(f"Protocol mismatch in existing job: {path}")
                    payloads.append(payload)
                else:
                    pending.append((dataset, transfer_probability, redemption_share))

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_cell, *task): task for task in pending
            }
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                payload = future.result()
                path = job_path(output_dir, *task)
                write_json_atomic(path, payload)
                payloads.append(payload)
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] {task[0]} "
                    f"t={task[1]:.2f} r={task[2]:.2f}",
                    flush=True,
                )

    raw_rows = [
        row
        for payload in payloads
        for row in payload["rows"]
    ]
    raw_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["transfer_probability"],
            row["central_redemption_share"],
            row["k"],
            row["selection_seed"],
            METHODS.index(str(row["method"])),
        )
    )
    method_summary = summarize_methods(raw_rows)
    comparison_summary = summarize_comparisons(raw_rows, method_summary)

    raw_path = output_dir / "raw.csv"
    method_path = output_dir / "method_summary.csv"
    comparison_path = output_dir / "comparison_summary.csv"
    write_csv_atomic(raw_path, raw_rows)
    write_csv_atomic(method_path, method_summary)
    write_csv_atomic(comparison_path, comparison_summary)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": STATUS,
            "completed_cells": len(payloads),
            "raw_rows": len(raw_rows),
            "method_summary_rows": len(method_summary),
            "comparison_summary_rows": len(comparison_summary),
            "elapsed_seconds_current_invocation": time.perf_counter() - started,
        }
    )
    write_json_atomic(metadata_path, metadata)
    write_manifest(output_dir, [raw_path, method_path, comparison_path])
    print(f"Wrote {len(raw_rows)} raw rows to {raw_path}", flush=True)


if __name__ == "__main__":
    main()
