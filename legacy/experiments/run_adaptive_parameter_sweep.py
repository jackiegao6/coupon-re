"""Run the locked v2.5 adaptive-sampling parameter sweep."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_appendix_parameter_sweep as v24
import run_real_submission as core


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "validated-v2.5-adaptive-grid"
STATUS = "REAL_EXPERIMENT"
DATASETS = v24.DATASETS
TRANSFER_VALUES = v24.TRANSFER_VALUES
REDEMPTION_SHARES = v24.REDEMPTION_SHARES
GRID_BUDGET = v24.GRID_BUDGET
BUDGET_SLICE_SHARE = v24.BUDGET_SLICE_SHARE
BUDGET_SLICE = v24.BUDGET_SLICE
SELECTION_SEEDS = v24.SELECTION_SEEDS
METHODS = v24.METHODS
BASELINES = v24.BASELINES
INITIAL_SAMPLES = 100_000
SAMPLE_ROUNDING = 50_000
TARGET_OBSERVATIONS_PER_CANDIDATE = 10.0
FLOOR_CAP = 1_500_000
HARD_CAP = 3_000_000
VALIDATION_SIMULATIONS = 5_000
STABILITY_TOLERANCE_PCT = 0.5
FINAL_EVALUATION_SIMULATIONS = 10_000
T_CRITICAL_95_DF4 = 2.7764451051977987


class IncrementalCIMSamplePool:
    """Append conditioned joint RR samples while preserving one stream prefix."""

    def __init__(
        self,
        graph: core.Graph,
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
        root_weights = 1.0 - np.power(1.0 - alpha, k)
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
            gates = core.conditioned_gates(
                float(self.alpha[root]), self.k, sample_rng
            )
            for coupon_index, gate in enumerate(gates):
                if not gate:
                    continue
                rr_set = core.reverse_coupon_set(
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


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def cell_budgets(redemption_share: float) -> list[int]:
    return BUDGET_SLICE if math.isclose(redemption_share, BUDGET_SLICE_SHARE) else [GRID_BUDGET]


def rounded_up(value: float, unit: int) -> int:
    return int(math.ceil(value / unit) * unit)


def sample_floor(graph: core.Graph, alpha: np.ndarray, k: int) -> tuple[int, float]:
    adoption_sum = float(alpha.sum())
    root_weight_sum = float(np.sum(1.0 - np.power(1.0 - alpha, k)))
    if adoption_sum <= 0.0 or root_weight_sum <= 0.0:
        return INITIAL_SAMPLES, 0.0
    raw = (
        TARGET_OBSERVATIONS_PER_CANDIDATE
        * graph.n
        * root_weight_sum
        / adoption_sum
    )
    floor = max(INITIAL_SAMPLES, rounded_up(raw, SAMPLE_ROUNDING))
    return min(floor, FLOOR_CAP), raw


def stream_batch(*parts: object, count: int) -> list[int]:
    rng = random.Random(core._stable_seed(PROTOCOL_VERSION, *parts))
    return [rng.getrandbits(64) for _ in range(count)]


def stream_id(*parts: object) -> int:
    return core._stable_seed(PROTOCOL_VERSION, *parts)


def validation_means(
    graph: core.Graph,
    previous: list[int],
    current: list[int],
    alpha: np.ndarray,
    discard: np.ndarray,
    streams: list[int],
) -> tuple[float, float, float]:
    previous_mean, _, _, _ = core.evaluate_with_streams(
        graph, previous, alpha, discard, streams
    )
    current_mean, _, _, _ = core.evaluate_with_streams(
        graph, current, alpha, discard, streams
    )
    relative = (
        100.0 * (current_mean - previous_mean) / previous_mean
        if previous_mean > 0.0
        else 0.0
    )
    return previous_mean, current_mean, relative


def adaptive_cim_selection(
    graph: core.Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    dataset: str,
    transfer_probability: float,
    redemption_share: float,
    k: int,
    selection_seed: int,
    verbose: bool = False,
) -> tuple[list[int], list[dict[str, object]], int, bool, float]:
    floor, raw_floor = sample_floor(graph, alpha, k)
    samples = INITIAL_SAMPLES
    previous: list[int] | None = None
    stages: list[dict[str, object]] = []
    final_stable = False
    rr_stream_id = stream_id(
        dataset,
        f"{transfer_probability:.2f}",
        f"{redemption_share:.2f}",
        k,
        selection_seed,
        "cim-ris",
    )
    pool = IncrementalCIMSamplePool(graph, alpha, discard, k, rr_stream_id)

    while True:
        (
            allocation,
            stage_seconds,
            cumulative_seconds,
            training_estimate,
            memberships,
        ) = pool.extend_and_select(
            samples
        )
        stage: dict[str, object] = {
            "rr_samples": samples,
            "allocation": allocation,
            "training_rr_estimate": training_estimate,
            "rr_memberships": memberships,
            "rr_stream_id": rr_stream_id,
            "stage_seconds": stage_seconds,
            "selection_seconds": cumulative_seconds,
            "validation_stream_id": None,
            "validation_previous_mean": None,
            "validation_current_mean": None,
            "validation_relative_change_pct": None,
            "stable": False,
        }
        if previous is not None:
            validation_stream_id = stream_id(
                dataset,
                f"{transfer_probability:.2f}",
                f"{redemption_share:.2f}",
                k,
                selection_seed,
                samples,
                "stability-validation",
            )
            streams = stream_batch(
                dataset,
                f"{transfer_probability:.2f}",
                f"{redemption_share:.2f}",
                k,
                selection_seed,
                samples,
                "stability-validation",
                count=VALIDATION_SIMULATIONS,
            )
            old_mean, new_mean, relative = validation_means(
                graph, previous, allocation, alpha, discard, streams
            )
            final_stable = abs(relative) <= STABILITY_TOLERANCE_PCT
            stage.update(
                {
                    "validation_stream_id": validation_stream_id,
                    "validation_previous_mean": old_mean,
                    "validation_current_mean": new_mean,
                    "validation_relative_change_pct": relative,
                    "stable": final_stable,
                }
            )
        stages.append(stage)
        if verbose:
            change = stage["validation_relative_change_pct"]
            print(
                f"  {dataset} t={transfer_probability:.2f} r={redemption_share:.2f} "
                f"k={k} seed={selection_seed} T={samples}: "
                f"change={change if change is not None else 'n/a'}, "
                f"floor={floor}",
                flush=True,
            )

        if samples >= floor and previous is not None and final_stable:
            break
        if samples >= HARD_CAP:
            break
        previous = allocation
        if samples < floor:
            samples = min(samples * 2, floor)
        else:
            samples = min(samples * 2, HARD_CAP)
        if samples <= stages[-1]["rr_samples"]:
            samples = min(int(stages[-1]["rr_samples"]) * 2, HARD_CAP)

    hard_cap_reached = samples >= HARD_CAP
    return allocation, stages, floor, hard_cap_reached, raw_floor


def job_key(
    dataset: str,
    transfer_probability: float,
    redemption_share: float,
    k: int,
    selection_seed: int,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "transfer_probability": transfer_probability,
        "central_redemption_share": redemption_share,
        "k": k,
        "selection_seed": selection_seed,
        "methods": METHODS,
        "initial_samples": INITIAL_SAMPLES,
        "target_observations_per_candidate": TARGET_OBSERVATIONS_PER_CANDIDATE,
        "floor_cap": FLOOR_CAP,
        "hard_cap": HARD_CAP,
        "validation_simulations": VALIDATION_SIMULATIONS,
        "stability_tolerance_pct": STABILITY_TOLERANCE_PCT,
        "final_evaluation_simulations": FINAL_EVALUATION_SIMULATIONS,
        "capacity_per_node": 1,
    }


def job_path(
    output_dir: Path,
    dataset: str,
    transfer_probability: float,
    redemption_share: float,
    k: int,
    selection_seed: int,
) -> Path:
    return output_dir / "jobs" / dataset / (
        f"transfer-{transfer_probability:.2f}_redemption-{redemption_share:.2f}"
    ) / (
        f"k-{k:04d}_seed-{selection_seed}.json"
    )


def run_job(
    dataset: str,
    transfer_probability: float,
    redemption_share: float,
    k: int,
    selection_seed: int,
    verbose: bool = False,
) -> dict[str, object]:
    graph = core.load_graph(dataset)
    alpha, discard, transfer = v24.grid_probabilities(
        graph, transfer_probability, redemption_share
    )
    static_orders = {
        method: order
        for method, order in core.static_orders(
            graph,
            alpha,
            transfer,
            core._stable_seed(PROTOCOL_VERSION, dataset, transfer_probability, redemption_share, "static"),
        ).items()
        if method in METHODS
    }
    cim_allocation, stages, floor, hard_cap_reached, raw_floor = adaptive_cim_selection(
        graph,
        alpha,
        discard,
        dataset,
        transfer_probability,
        redemption_share,
        k,
        selection_seed,
        verbose=verbose,
    )
    final_samples = int(stages[-1]["rr_samples"])
    ic_stream_id = stream_id(
        dataset,
        f"{transfer_probability:.2f}",
        f"{redemption_share:.2f}",
        k,
        selection_seed,
        "ic-ris",
    )
    ic_order, ic_seconds = core.ic_ris_order(
        graph,
        transfer,
        k,
        final_samples,
        ic_stream_id,
    )
    orders = {**static_orders, "IC-RIS": ic_order, "CIM-RIS": cim_allocation}
    final_evaluation_stream_id = stream_id(
        dataset,
        f"{transfer_probability:.2f}",
        f"{redemption_share:.2f}",
        k,
        selection_seed,
        "final-evaluation",
    )
    final_streams = stream_batch(
        dataset,
        f"{transfer_probability:.2f}",
        f"{redemption_share:.2f}",
        k,
        selection_seed,
        "final-evaluation",
        count=FINAL_EVALUATION_SIMULATIONS,
    )
    unresolved_instability = hard_cap_reached and not bool(stages[-1]["stable"])
    rows: list[dict[str, object]] = []
    for method in METHODS:
        allocation = orders[method][:k]
        mean, ci95, variance, redemptions = core.evaluate_with_streams(
            graph, allocation, alpha, discard, final_streams
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
                "duplicate_fraction": v24.duplicate_fraction(mean, redemptions),
                "selection_seconds": (
                    float(stages[-1]["selection_seconds"])
                    if method == "CIM-RIS"
                    else ic_seconds
                    if method == "IC-RIS"
                    else 0.0
                ),
                "final_rr_samples": final_samples if method in {"CIM-RIS", "IC-RIS"} else 0,
                "sample_floor": floor if method == "CIM-RIS" else 0,
                "stability_relative_change_pct": (
                    stages[-1]["validation_relative_change_pct"]
                    if method == "CIM-RIS"
                    else None
                ),
                "stable": bool(stages[-1]["stable"]) if method == "CIM-RIS" else None,
                "hard_cap_reached": hard_cap_reached if method == "CIM-RIS" else None,
                "unresolved_instability": unresolved_instability if method == "CIM-RIS" else None,
                "evaluation_simulations": FINAL_EVALUATION_SIMULATIONS,
                "evaluation_stream_id": final_evaluation_stream_id,
                "status": STATUS,
                "protocol_version": PROTOCOL_VERSION,
                "seeds": json.dumps(allocation, separators=(",", ":")),
            }
        )
    trace = {
        "dataset": dataset,
        "transfer_probability": transfer_probability,
        "central_redemption_share": redemption_share,
        "k": k,
        "selection_seed": selection_seed,
        "raw_sample_floor": raw_floor,
        "sample_floor": floor,
        "hard_cap_reached": hard_cap_reached,
        "unresolved_instability": unresolved_instability,
        "ic_rr_stream_id": ic_stream_id,
        "final_evaluation_stream_id": final_evaluation_stream_id,
        "stages": stages,
    }
    return {
        "status": STATUS,
        "key": job_key(
            dataset,
            transfer_probability,
            redemption_share,
            k,
            selection_seed,
        ),
        "probability_diagnostics": {
            "min_adoption": float(alpha.min()),
            "max_adoption": float(alpha.max()),
            "mean_adoption": float(alpha.mean()),
            "mean_transfer": float(transfer.mean()),
        },
        "trace": trace,
        "rows": rows,
    }


def summarize_methods(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ["dataset", "transfer_probability", "central_redemption_share", "k", "method"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        record = dict(zip(keys, key))
        record["selection_repeats"] = len(group)
        for field in ["mean_adopters", "mean_redemptions", "duplicate_fraction", "selection_seconds", "final_rr_samples"]:
            values = [float(row[field]) for row in group]
            record[f"mean_{field}"] = statistics.mean(values)
            record[f"std_{field}"] = statistics.stdev(values)
        record["stable_runs"] = sum(row["stable"] is True for row in group)
        record["hard_cap_runs"] = sum(
            row["hard_cap_reached"] is True for row in group
        )
        record["unresolved_instability_runs"] = sum(
            row["unresolved_instability"] is True for row in group
        )
        record["status"] = STATUS
        record["protocol_version"] = PROTOCOL_VERSION
        output.append(record)
    return output


def summarize_comparisons(raw_rows: list[dict[str, object]], method_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    cell_fields = ["dataset", "transfer_probability", "central_redemption_share", "k"]
    summaries: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    raw_index: dict[tuple[object, ...], dict[int, dict[str, dict[str, object]]]] = defaultdict(lambda: defaultdict(dict))
    for row in method_summary:
        summaries[tuple(row[field] for field in cell_fields)][str(row["method"])] = row
    for row in raw_rows:
        raw_index[tuple(row[field] for field in cell_fields)][int(row["selection_seed"])][str(row["method"])] = row
    output: list[dict[str, object]] = []
    for key, methods in sorted(summaries.items()):
        cim = methods["CIM-RIS"]
        best_method = max(BASELINES, key=lambda method: float(methods[method]["mean_mean_adopters"]))
        best = methods[best_method]
        cim_spread = float(cim["mean_mean_adopters"])
        best_spread = float(best["mean_mean_adopters"])
        gains = []
        for seed in SELECTION_SEEDS:
            seed_rows = raw_index[key][seed]
            cv = float(seed_rows["CIM-RIS"]["mean_adopters"])
            bv = float(seed_rows[best_method]["mean_adopters"])
            gains.append(100.0 * (cv - bv) / bv if bv > 0 else 0.0)
        record = dict(zip(cell_fields, key))
        record.update(
            {
                "best_baseline": best_method,
                "cim_mean_adopters": cim_spread,
                "best_baseline_mean_adopters": best_spread,
                "relative_gain_best_pct": 100.0 * (cim_spread - best_spread) / best_spread if best_spread > 0 else 0.0,
                "paired_gain_mean_pct": statistics.mean(gains),
                "paired_gain_std_pct": statistics.stdev(gains),
                "paired_gain_ci95_pct": T_CRITICAL_95_DF4 * statistics.stdev(gains) / math.sqrt(len(gains)),
                "cim_duplicate_fraction": float(cim["mean_duplicate_fraction"]),
                "best_baseline_duplicate_fraction": float(best["mean_duplicate_fraction"]),
                "duplicate_reduction_pp": 100.0 * (float(best["mean_duplicate_fraction"]) - float(cim["mean_duplicate_fraction"])),
                "cim_higher_mean": int(cim_spread > best_spread),
                "mean_final_rr_samples": float(cim["mean_final_rr_samples"]),
                "stable_runs": int(cim["stable_runs"]),
                "hard_cap_runs": int(cim["hard_cap_runs"]),
                "unresolved_instability_runs": int(
                    cim["unresolved_instability_runs"]
                ),
                "selection_repeats": len(SELECTION_SEEDS),
                "status": STATUS,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        for baseline in BASELINES:
            value = float(methods[baseline]["mean_mean_adopters"])
            record[f"gain_vs_{baseline}_pct"] = 100.0 * (cim_spread - value) / value if value > 0 else 0.0
        output.append(record)
    return output


def write_manifest(output_dir: Path, paths: list[Path]) -> None:
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in sorted(paths)]
    (output_dir / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_jobs() -> list[tuple[str, float, float, int, int]]:
    return [
        (dataset, transfer, redemption, k, selection_seed)
        for dataset in DATASETS
        for transfer in TRANSFER_VALUES
        for redemption in REDEMPTION_SHARES
        for k in cell_budgets(redemption)
        for selection_seed in SELECTION_SEEDS
    ]


def aggregate(output_dir: Path) -> None:
    raw: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    jobs = expected_jobs()
    expected_paths = {
        job_path(output_dir, dataset, transfer, redemption, k, selection_seed)
        for dataset, transfer, redemption, k, selection_seed in jobs
    }
    observed_paths = set((output_dir / "jobs").rglob("*.json"))
    if observed_paths != expected_paths:
        missing = len(expected_paths - observed_paths)
        extra = len(observed_paths - expected_paths)
        raise RuntimeError(
            f"Cannot aggregate incomplete grid: missing={missing}, extra={extra}"
        )
    for dataset, transfer, redemption, k, selection_seed in jobs:
        path = job_path(
            output_dir, dataset, transfer, redemption, k, selection_seed
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != STATUS or payload.get("key") != job_key(
            dataset, transfer, redemption, k, selection_seed
        ):
            raise RuntimeError(f"Invalid job: {path}")
        raw.extend(payload["rows"])
        traces.append(payload["trace"])
    method_summary = summarize_methods(raw)
    comparison_summary = summarize_comparisons(raw, method_summary)
    raw_path = output_dir / "raw.csv"
    method_path = output_dir / "method_summary.csv"
    comparison_path = output_dir / "comparison_summary.csv"
    trace_path = output_dir / "traces.json"
    write_csv_atomic(raw_path, raw)
    write_csv_atomic(method_path, method_summary)
    write_csv_atomic(comparison_path, comparison_summary)
    write_json_atomic(trace_path, {"status": STATUS, "protocol_version": PROTOCOL_VERSION, "traces": traces})
    write_manifest(output_dir, [raw_path, method_path, comparison_path, trace_path])
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": STATUS,
            "completed_cells": 40,
            "completed_budget_configurations": 60,
            "completed_selection_jobs": len(jobs),
            "raw_rows": len(raw),
            "method_summary_rows": len(method_summary),
            "comparison_summary_rows": len(comparison_summary),
        }
    )
    write_json_atomic(metadata_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "experiments/results/validated-v2/adaptive-grid"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--transfers", default=",".join(str(v) for v in TRANSFER_VALUES))
    parser.add_argument("--redemptions", default=",".join(str(v) for v in REDEMPTION_SHARES))
    parser.add_argument("--selection-seeds", default=",".join(str(v) for v in SELECTION_SEEDS))
    parser.add_argument(
        "--budgets",
        default="",
        help="Optional budget filter for smoke runs; the locked full run leaves this empty.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    transfers = [float(value) for value in args.transfers.split(",") if value.strip()]
    redemptions = [float(value) for value in args.redemptions.split(",") if value.strip()]
    selection_seeds = [
        int(value) for value in args.selection_seeds.split(",") if value.strip()
    ]
    budget_filter = {
        int(value) for value in args.budgets.split(",") if value.strip()
    }
    write_json_atomic(
        output_dir / "metadata.json",
        {
            "status": "RUNNING",
            "protocol_version": PROTOCOL_VERSION,
            "datasets": datasets,
            "transfer_values": transfers,
            "central_redemption_shares": redemptions,
            "selection_seeds": selection_seeds,
            "budget_filter": sorted(budget_filter),
            "initial_samples": INITIAL_SAMPLES,
            "target_observations_per_candidate": TARGET_OBSERVATIONS_PER_CANDIDATE,
            "floor_cap": FLOOR_CAP,
            "hard_cap": HARD_CAP,
            "validation_simulations": VALIDATION_SIMULATIONS,
            "stability_tolerance_pct": STABILITY_TOLERANCE_PCT,
            "final_evaluation_simulations": FINAL_EVALUATION_SIMULATIONS,
            "workers": args.workers,
            "hardware": v24.hardware_metadata(),
        },
    )
    jobs = [
        (dataset, transfer, redemption, k, selection_seed)
        for dataset in datasets
        for transfer in transfers
        for redemption in redemptions
        for k in cell_budgets(redemption)
        if not budget_filter or k in budget_filter
        for selection_seed in selection_seeds
    ]
    started = time.perf_counter()

    def consume(
        dataset: str,
        transfer: float,
        redemption: float,
        k: int,
        selection_seed: int,
        payload: dict[str, object],
    ) -> None:
        path = job_path(
            output_dir, dataset, transfer, redemption, k, selection_seed
        )
        write_json_atomic(path, payload)
        trace = payload["trace"]
        print(
            f"{dataset} t={transfer:.2f} r={redemption:.2f} k={k} "
            f"seed={selection_seed}: T={trace['stages'][-1]['rr_samples']}, "
            f"stable={trace['stages'][-1]['stable']}, "
            f"hard_cap={trace['hard_cap_reached']}",
            flush=True,
        )

    pending = []
    for dataset, transfer, redemption, k, selection_seed in jobs:
        path = job_path(
            output_dir, dataset, transfer, redemption, k, selection_seed
        )
        if path.exists() and not args.force:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("key") == job_key(
                dataset, transfer, redemption, k, selection_seed
            ) and payload.get("status") == STATUS:
                continue
            raise RuntimeError(f"Protocol mismatch: {path}")
        pending.append((dataset, transfer, redemption, k, selection_seed))
    if args.workers == 1:
        for dataset, transfer, redemption, k, selection_seed in pending:
            consume(
                dataset,
                transfer,
                redemption,
                k,
                selection_seed,
                run_job(
                    dataset,
                    transfer,
                    redemption,
                    k,
                    selection_seed,
                    verbose=True,
                ),
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, *job): job for job in pending}
            for future in as_completed(futures):
                dataset, transfer, redemption, k, selection_seed = futures[future]
                consume(
                    dataset,
                    transfer,
                    redemption,
                    k,
                    selection_seed,
                    future.result(),
                )
    full_run = (
        datasets == DATASETS
        and transfers == TRANSFER_VALUES
        and redemptions == REDEMPTION_SHARES
        and selection_seeds == SELECTION_SEEDS
        and not budget_filter
    )
    if full_run:
        aggregate(output_dir)
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata["elapsed_seconds_current_invocation"] = time.perf_counter() - started
        write_json_atomic(output_dir / "metadata.json", metadata)
    else:
        print("Partial smoke run complete; aggregation skipped.", flush=True)


if __name__ == "__main__":
    main()
