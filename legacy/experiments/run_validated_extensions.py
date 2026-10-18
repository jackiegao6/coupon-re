"""Run prespecified strong-reference, capacity, and sensitivity extensions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import run_real_submission as core


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "validated-v2.3-extension"
SELECTION_SEEDS = [20260715, 20260716, 20260717, 20260718, 20260719]
SCENARIOS = ["balanced", "adoption-heavy", "forwarding-heavy"]
DATASETS = ["Netscience", "NetFacebookEgo"]
MAIN_BUDGETS = [10, 25, 50, 100, 150, 200]
CAPACITY_BUDGETS = [200]
CAPACITY_POLICIES = [("distinct", 1), ("capacity-2", 2), ("unrestricted", 200)]
SENSITIVITY_SCENARIOS = ["forwarding-heavy"]
SENSITIVITY_BUDGETS = [10, 50, 200]
SENSITIVITY_SAMPLES = [5_000, 20_000, 50_000, 100_000]
ORACLE_TRAJECTORIES = {"Netscience": 100_000, "NetFacebookEgo": 50_000}
EVALUATION_SIMULATIONS = 10_000


_WORKER_CONTEXT: tuple[
    core.Graph, np.ndarray, np.ndarray, int, int, str
] | None = None


def _init_oracle_worker(
    graph: core.Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    trajectories: int,
    master_seed: int,
    scenario: str,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (
        graph,
        alpha,
        discard,
        trajectories,
        master_seed,
        scenario,
    )


def _estimate_oracle_chunk(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Oracle worker was not initialized")
    graph, alpha, discard, trajectories, master_seed, scenario = _WORKER_CONTEXT
    begin, end = bounds
    result = np.zeros((end - begin, graph.n), dtype=np.float32)
    for start in range(begin, end):
        counts: dict[int, int] = defaultdict(int)
        rng = random.Random(
            core._stable_seed(master_seed, graph.name, scenario, start, "oracle-row")
        )
        for _ in range(trajectories):
            adopter = core.single_coupon_adopter(
                graph, start, alpha, discard, rng
            )
            if adopter >= 0:
                counts[adopter] += 1
        for adopter, count in counts.items():
            result[start - begin, adopter] = count / trajectories
    return begin, result


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sample_budget(k: int) -> int:
    return 100_000 if k <= 50 else 50_000


def simulation_streams(master_seed: int) -> list[int]:
    rng = random.Random(master_seed)
    return [rng.getrandbits(64) for _ in range(EVALUATION_SIMULATIONS)]


def validate_q_matrix(q: np.ndarray, trajectories: int) -> dict[str, float]:
    if not np.all(np.isfinite(q)) or float(np.min(q)) < 0.0:
        raise AssertionError("Oracle matrix contains invalid probabilities")
    row_sums = np.sum(q, axis=1, dtype=np.float64)
    tolerance = 1.0 / trajectories
    if float(np.max(row_sums)) > 1.0 + tolerance:
        raise AssertionError("Oracle row redemption probability exceeds one")
    return {
        "min_entry": float(np.min(q)),
        "max_entry": float(np.max(q)),
        "min_row_sum": float(np.min(row_sums)),
        "max_row_sum": float(np.max(row_sums)),
    }


def build_oracle_parallel(
    graph: core.Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    trajectories: int,
    master_seed: int,
    scenario: str,
    workers: int,
) -> np.ndarray:
    matrix = np.zeros((graph.n, graph.n), dtype=np.float32)
    chunk_size = max(1, math.ceil(graph.n / max(1, workers * 4)))
    bounds = [
        (begin, min(begin + chunk_size, graph.n))
        for begin in range(0, graph.n, chunk_size)
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_oracle_worker,
        initargs=(
            graph,
            alpha,
            discard,
            trajectories,
            master_seed,
            scenario,
        ),
    ) as executor:
        futures = [executor.submit(_estimate_oracle_chunk, item) for item in bounds]
        for completed, future in enumerate(as_completed(futures), start=1):
            begin, rows = future.result()
            matrix[begin : begin + len(rows)] = rows
            print(
                f"    oracle chunks {completed}/{len(futures)}",
                flush=True,
            )
    return matrix


def load_or_build_oracle(
    output_dir: Path,
    graph: core.Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    scenario: str,
    workers: int,
) -> tuple[np.ndarray, dict[str, object]]:
    trajectories = ORACLE_TRAJECTORIES[graph.name]
    if graph.name == "Netscience":
        source_dir = ROOT / "experiments/results/validated-v2/oracle"
        matrix_path = source_dir / f"netscience_{scenario}_{trajectories}.npz"
        metadata_path = source_dir / f"netscience_{scenario}_{trajectories}.json"
        q = np.load(matrix_path)["q"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") != "REAL_EXPERIMENT"
            or metadata.get("trajectories_per_source") != trajectories
        ):
            raise AssertionError(f"Invalid validated oracle cache: {metadata_path}")
        return q, {
            "source": "validated-v2.2 cache",
            "elapsed_seconds": metadata["elapsed_seconds"],
            **validate_q_matrix(q, trajectories),
        }

    oracle_dir = output_dir / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = oracle_dir / f"netfacebookego_{scenario}_{trajectories}.npz"
    metadata_path = oracle_dir / f"netfacebookego_{scenario}_{trajectories}.json"
    master_seed = core._stable_seed(
        20260715, graph.name, scenario, trajectories, "extension-oracle"
    )
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": graph.name,
        "scenario": scenario,
        "trajectories_per_source": trajectories,
        "master_seed": master_seed,
        "cycle_semantics": "fixed action per node; repeated node terminates",
    }
    if matrix_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            q = np.load(matrix_path)["q"]
            return q, {
                "source": "validated-v2.3 cache",
                "elapsed_seconds": metadata["elapsed_seconds"],
                **validate_q_matrix(q, trajectories),
            }
        raise RuntimeError(f"Oracle cache protocol mismatch: {metadata_path}")

    started = time.perf_counter()
    q = build_oracle_parallel(
        graph,
        alpha,
        discard,
        trajectories,
        master_seed,
        scenario,
        workers,
    )
    elapsed = time.perf_counter() - started
    diagnostics = validate_q_matrix(q, trajectories)
    np.savez_compressed(matrix_path, q=q)
    write_json_atomic(
        metadata_path,
        {
            **expected,
            "workers": workers,
            "elapsed_seconds": elapsed,
            **diagnostics,
            "status": "REAL_EXPERIMENT",
        },
    )
    return q, {
        "source": "validated-v2.3 new run",
        "elapsed_seconds": elapsed,
        **diagnostics,
    }


def summarize(
    rows: list[dict[str, object]],
    keys: list[str],
    value_fields: list[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for group_key, group_rows in sorted(groups.items()):
        record = dict(zip(keys, group_key))
        record["selection_repeats"] = len(group_rows)
        for field in value_fields:
            values = [float(row[field]) for row in group_rows]
            record[f"mean_{field}"] = statistics.mean(values)
            record[f"std_{field}"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        record["status"] = "REAL_EXPERIMENT"
        record["protocol_version"] = PROTOCOL_VERSION
        output.append(record)
    return output


def load_validated_cim_rows() -> dict[tuple[str, str, int], list[dict[str, object]]]:
    source = ROOT / "experiments/results/validated-v2/validated_raw.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if (
            row["method"] == "CIM-RIS"
            and row["status"] == "REAL_EXPERIMENT"
            and row["protocol_version"] == "validated-v2.2"
            and row["dataset"] in DATASETS
        ):
            index[(row["dataset"], row["scenario"], int(row["k"]))].append(
                {
                    "selection_seed": int(row["selection_seed"]),
                    "allocation": json.loads(row["seeds"]),
                    "mean_adopters": float(row["mean_adopters"]),
                    "evaluation_seed": int(row["evaluation_seed"]),
                    "selection_seconds": float(row["selection_seconds"]),
                }
            )
    return index


def run_extensions(output_dir: Path, workers: int) -> None:
    metadata_path = output_dir / "extension_metadata.json"
    write_json_atomic(
        metadata_path,
        {
            "status": "RUNNING",
            "protocol_version": PROTOCOL_VERSION,
            "selection_seeds": SELECTION_SEEDS,
            "datasets": DATASETS,
            "scenarios": SCENARIOS,
            "main_budgets": MAIN_BUDGETS,
            "oracle_trajectories_per_source": ORACLE_TRAJECTORIES,
            "capacity_budgets": CAPACITY_BUDGETS,
            "capacity_policies": [label for label, _ in CAPACITY_POLICIES],
            "sensitivity_scenarios": SENSITIVITY_SCENARIOS,
            "sensitivity_budgets": SENSITIVITY_BUDGETS,
            "sensitivity_samples": SENSITIVITY_SAMPLES,
            "evaluation_simulations": EVALUATION_SIMULATIONS,
            "oracle_workers": workers,
        },
    )
    validated_rows = load_validated_cim_rows()
    benchmark_raw: list[dict[str, object]] = []
    capacity_raw: list[dict[str, object]] = []
    sensitivity_raw: list[dict[str, object]] = []
    oracle_diagnostics: list[dict[str, object]] = []

    completed = 0
    total = len(DATASETS) * len(SCENARIOS)
    for scenario in SCENARIOS:
        for dataset in DATASETS:
            graph = core.load_graph(dataset)
            alpha, discard, _ = core.node_probabilities(graph, scenario)
            print(f"Building/loading oracle: {dataset} / {scenario}", flush=True)
            q, diagnostics = load_or_build_oracle(
                output_dir, graph, alpha, discard, scenario, workers
            )
            completed += 1
            oracle_diagnostics.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "trajectories_per_source": ORACLE_TRAJECTORIES[dataset],
                    **diagnostics,
                    "status": "REAL_EXPERIMENT",
                    "protocol_version": PROTOCOL_VERSION,
                }
            )
            print(f"[{completed}/{total}] oracle ready", flush=True)

            distinct_reference = core.mc_greedy_order(q, 200, capacity_per_node=1)
            for k in MAIN_BUDGETS:
                rows = validated_rows[(dataset, scenario, k)]
                if len(rows) != len(SELECTION_SEEDS):
                    raise AssertionError(
                        f"Expected five CIM-RIS rows for {dataset}/{scenario}/k={k}"
                    )
                for item in rows:
                    streams = simulation_streams(item["evaluation_seed"])
                    reference_mean, _, _, _ = core.evaluate_with_streams(
                        graph,
                        distinct_reference[:k],
                        alpha,
                        discard,
                        streams,
                    )
                    cim_mean = item["mean_adopters"]
                    benchmark_raw.append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "k": k,
                            "selection_seed": item["selection_seed"],
                            "cim_mean_adopters": cim_mean,
                            "mc_greedy_mean_adopters": reference_mean,
                            "gap_to_mc_greedy_percent": 100.0
                            * (reference_mean - cim_mean)
                            / reference_mean,
                            "oracle_trajectories_per_source": ORACLE_TRAJECTORIES[
                                dataset
                            ],
                            "evaluation_simulations": EVALUATION_SIMULATIONS,
                            "status": "REAL_EXPERIMENT",
                            "protocol_version": PROTOCOL_VERSION,
                        }
                    )

            reference_orders = {
                policy: core.mc_greedy_order(
                    q, 200, capacity_per_node=capacity
                )
                for policy, capacity in CAPACITY_POLICIES
            }
            for k in CAPACITY_BUDGETS:
                validated_by_seed = {
                    int(row["selection_seed"]): row
                    for row in validated_rows[(dataset, scenario, k)]
                }
                for selection_seed in SELECTION_SEEDS:
                    evaluation_seed = core._stable_seed(
                        20260715,
                        dataset,
                        scenario,
                        k,
                        selection_seed,
                        "capacity-evaluation",
                    )
                    streams = simulation_streams(evaluation_seed)
                    for policy, capacity in CAPACITY_POLICIES:
                        if policy == "distinct":
                            prior = validated_by_seed[selection_seed]
                            allocation = prior["allocation"]
                            elapsed = prior["selection_seconds"]
                        else:
                            allocation, elapsed, _, _ = core.cim_ris_seeds(
                                graph,
                                alpha,
                                discard,
                                k,
                                sample_budget(k),
                                core._stable_seed(
                                    selection_seed,
                                    dataset,
                                    scenario,
                                    k,
                                    policy,
                                    "capacity-selection",
                                ),
                                capacity_per_node=capacity,
                            )
                        reference = reference_orders[policy][:k]
                        cim_mean, _, _, _ = core.evaluate_with_streams(
                            graph, allocation, alpha, discard, streams
                        )
                        reference_mean, _, _, _ = core.evaluate_with_streams(
                            graph, reference, alpha, discard, streams
                        )
                        capacity_raw.append(
                            {
                                "dataset": dataset,
                                "scenario": scenario,
                                "k": k,
                                "capacity_policy": policy,
                                "capacity_per_node": capacity,
                                "selection_seed": selection_seed,
                                "rr_samples": sample_budget(k),
                                "cim_mean_adopters": cim_mean,
                                "mc_greedy_mean_adopters": reference_mean,
                                "gap_to_mc_greedy_percent": 100.0
                                * (reference_mean - cim_mean)
                                / reference_mean,
                                "cim_repeated_placements": k
                                - len(set(allocation)),
                                "mc_greedy_repeated_placements": k
                                - len(set(reference)),
                                "selection_seconds": elapsed,
                                "evaluation_seed": evaluation_seed,
                                "status": "REAL_EXPERIMENT",
                                "protocol_version": PROTOCOL_VERSION,
                            }
                        )
                    print(
                        f"    capacity {dataset}/{scenario}/k={k}: "
                        f"seed {selection_seed}",
                        flush=True,
                    )

            if scenario in SENSITIVITY_SCENARIOS:
                for k in SENSITIVITY_BUDGETS:
                    reference = distinct_reference[:k]
                    for selection_seed in SELECTION_SEEDS:
                        evaluation_seed = core._stable_seed(
                            20260715,
                            dataset,
                            scenario,
                            k,
                            selection_seed,
                            "sensitivity-evaluation",
                        )
                        streams = simulation_streams(evaluation_seed)
                        reference_mean, _, _, _ = core.evaluate_with_streams(
                            graph, reference, alpha, discard, streams
                        )
                        for samples in SENSITIVITY_SAMPLES:
                            allocation, elapsed, _, _ = core.cim_ris_seeds(
                                graph,
                                alpha,
                                discard,
                                k,
                                samples,
                                core._stable_seed(
                                    selection_seed,
                                    dataset,
                                    scenario,
                                    k,
                                    samples,
                                    "sensitivity-selection",
                                ),
                            )
                            cim_mean, _, _, _ = core.evaluate_with_streams(
                                graph, allocation, alpha, discard, streams
                            )
                            sensitivity_raw.append(
                                {
                                    "dataset": dataset,
                                    "scenario": scenario,
                                    "k": k,
                                    "rr_samples": samples,
                                    "selection_seed": selection_seed,
                                    "cim_mean_adopters": cim_mean,
                                    "mc_greedy_mean_adopters": reference_mean,
                                    "gap_to_mc_greedy_percent": 100.0
                                    * (reference_mean - cim_mean)
                                    / reference_mean,
                                    "selection_seconds": elapsed,
                                    "evaluation_seed": evaluation_seed,
                                    "status": "REAL_EXPERIMENT",
                                    "protocol_version": PROTOCOL_VERSION,
                                }
                            )
                        print(
                            f"    sensitivity {dataset}/{scenario}/k={k}: "
                            f"seed {selection_seed}",
                            flush=True,
                        )

            del q
            write_csv(output_dir / "oracle_diagnostics.csv", oracle_diagnostics)
            write_csv(output_dir / "strong_benchmark_raw.csv", benchmark_raw)
            write_csv(output_dir / "capacity_raw.csv", capacity_raw)
            if sensitivity_raw:
                write_csv(output_dir / "sensitivity_raw.csv", sensitivity_raw)

    benchmark_summary = summarize(
        benchmark_raw,
        ["dataset", "scenario", "k"],
        ["cim_mean_adopters", "mc_greedy_mean_adopters", "gap_to_mc_greedy_percent"],
    )
    capacity_summary = summarize(
        capacity_raw,
        ["dataset", "scenario", "k", "capacity_policy", "capacity_per_node"],
        [
            "cim_mean_adopters",
            "mc_greedy_mean_adopters",
            "gap_to_mc_greedy_percent",
            "cim_repeated_placements",
            "mc_greedy_repeated_placements",
            "selection_seconds",
        ],
    )
    sensitivity_summary = summarize(
        sensitivity_raw,
        ["dataset", "scenario", "k", "rr_samples"],
        [
            "cim_mean_adopters",
            "mc_greedy_mean_adopters",
            "gap_to_mc_greedy_percent",
            "selection_seconds",
        ],
    )
    write_csv(output_dir / "strong_benchmark_summary.csv", benchmark_summary)
    write_csv(output_dir / "capacity_summary.csv", capacity_summary)
    write_csv(output_dir / "sensitivity_summary.csv", sensitivity_summary)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": "REAL_EXPERIMENT",
            "oracle_matrices": len(oracle_diagnostics),
            "strong_benchmark_raw_rows": len(benchmark_raw),
            "capacity_raw_rows": len(capacity_raw),
            "sensitivity_raw_rows": len(sensitivity_raw),
        }
    )
    write_json_atomic(metadata_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments/results/validated-v2/extensions"),
    )
    parser.add_argument("--oracle-workers", type=int, default=8)
    args = parser.parse_args()
    if args.oracle_workers < 1:
        raise ValueError("--oracle-workers must be positive")
    run_extensions(Path(args.output_dir), args.oracle_workers)


if __name__ == "__main__":
    main()
