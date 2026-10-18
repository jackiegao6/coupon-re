"""Run the preregistered, repeated-seed coupon experiment study.

Each configuration is written to an independent JSON file under
``experiments/results/validated-v2/jobs``. Existing jobs are reused only when
their complete protocol key matches, making interrupted runs resumable without
mixing rows from different experiment revisions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

import run_real_submission as core


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "validated-v2.2"


def sample_budget(k: int) -> int:
    """Budget locked from the pre-comparison convergence diagnostic."""
    return 100_000 if k <= 50 else 50_000


def hardware_metadata() -> dict[str, object]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    memory_kib = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_kib = int(line.split()[1])
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "memory_kib": memory_kib,
        "parallelism_per_job": 1,
    }


def job_key(
    dataset: str,
    scenario: str,
    k: int,
    selection_seed: int,
    rr_samples: int,
    eval_simulations: int,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "scenario": scenario,
        "k": k,
        "selection_seed": selection_seed,
        "rr_samples": rr_samples,
        "eval_simulations": eval_simulations,
    }


def job_path(output_dir: Path, key: dict[str, object]) -> Path:
    return (
        output_dir
        / "jobs"
        / str(key["scenario"])
        / str(key["dataset"])
        / f"k{key['k']}_seed{key['selection_seed']}.json"
    )


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_or_build_oracle(
    output_dir: Path,
    graph: core.Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    scenario: str,
    trajectories: int,
    master_seed: int,
) -> np.ndarray | None:
    if graph.name != "Netscience" or trajectories <= 0:
        return None
    oracle_dir = output_dir / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = oracle_dir / f"netscience_{scenario}_{trajectories}.npz"
    metadata_path = oracle_dir / f"netscience_{scenario}_{trajectories}.json"
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": graph.name,
        "scenario": scenario,
        "trajectories_per_source": trajectories,
        "seed": core._stable_seed(master_seed, graph.name, scenario, "oracle"),
    }
    if matrix_path.exists() and metadata_path.exists():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(recorded.get(k) == v for k, v in expected.items()):
            return np.load(matrix_path)["q"]

    started = time.perf_counter()
    q = core.estimate_single_coupon_matrix(
        graph, alpha, discard, trajectories, int(expected["seed"])
    )
    np.savez_compressed(matrix_path, q=q)
    write_json_atomic(
        metadata_path,
        {
            **expected,
            "elapsed_seconds": time.perf_counter() - started,
            "status": "REAL_EXPERIMENT",
        },
    )
    return q


def run_job(
    output_dir: Path,
    graph: core.Graph,
    scenario: str,
    alpha: np.ndarray,
    discard: np.ndarray,
    transfer: np.ndarray,
    k: int,
    selection_seed: int,
    eval_simulations: int,
    oracle_order: list[int] | None,
    force: bool,
) -> Path:
    rr_samples = sample_budget(k)
    key = job_key(
        graph.name, scenario, k, selection_seed, rr_samples, eval_simulations
    )
    path = job_path(output_dir, key)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("key") == key and existing.get("status") == "REAL_EXPERIMENT":
            return path
        raise RuntimeError(f"Protocol mismatch in existing job: {path}")

    static_started = time.perf_counter()
    orders = core.static_orders(
        graph,
        alpha,
        transfer,
        core._stable_seed(selection_seed, graph.name, scenario, k, "static"),
    )
    static_seconds = time.perf_counter() - static_started

    ic_order, ic_seconds = core.ic_ris_order(
        graph,
        transfer,
        k,
        rr_samples,
        core._stable_seed(selection_seed, graph.name, scenario, k, "ic-ris"),
    )
    orders["IC-RIS"] = ic_order

    cim_seeds, cim_seconds, estimated, memberships = core.cim_ris_seeds(
        graph,
        alpha,
        discard,
        k,
        rr_samples,
        core._stable_seed(selection_seed, graph.name, scenario, k, "cim-ris"),
    )
    orders["CIM-RIS"] = cim_seeds
    if oracle_order is not None:
        orders["MC-Greedy"] = oracle_order[:k]

    evaluation_master = core._stable_seed(
        20260715, graph.name, scenario, k, selection_seed, "forward-evaluation"
    )
    evaluation_rng = random.Random(evaluation_master)
    simulation_seeds = [evaluation_rng.getrandbits(64) for _ in range(eval_simulations)]

    methods: list[dict[str, object]] = []
    for method, order in orders.items():
        seeds = order[:k]
        mean, ci95, variance, redemptions = core.evaluate_with_streams(
            graph, seeds, alpha, discard, simulation_seeds
        )
        if method == "CIM-RIS":
            selection_seconds = cim_seconds
        elif method == "IC-RIS":
            selection_seconds = ic_seconds
        elif method == "MC-Greedy":
            selection_seconds = None
        else:
            selection_seconds = static_seconds
        methods.append(
            {
                "method": method,
                "seeds": seeds,
                "mean_adopters": mean,
                "ci95": ci95,
                "forward_variance": variance,
                "mean_redemptions": redemptions,
                "selection_seconds": selection_seconds,
                "cim_estimated_spread": estimated if method == "CIM-RIS" else None,
                "rr_memberships": memberships if method == "CIM-RIS" else None,
            }
        )

    write_json_atomic(
        path,
        {
            "status": "REAL_EXPERIMENT",
            "key": key,
            "evaluation_seed": evaluation_master,
            "methods": methods,
        },
    )
    return path


def parse_csv(value: str, cast) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="Netscience,NetFacebookEgo,DoubanRandom,EmailEnron",
    )
    parser.add_argument(
        "--scenarios",
        default="balanced,adoption-heavy,forwarding-heavy",
    )
    parser.add_argument("--budgets", default="10,25,50,100,150,200")
    parser.add_argument(
        "--selection-seeds", default="20260715,20260716,20260717,20260718,20260719"
    )
    parser.add_argument("--eval-simulations", type=int, default=10_000)
    parser.add_argument("--oracle-trajectories", type=int, default=100_000)
    parser.add_argument("--master-seed", type=int, default=20260715)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments/results/validated-v2"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv(args.datasets, str)
    scenarios = parse_csv(args.scenarios, str)
    budgets = sorted(set(parse_csv(args.budgets, int)))
    selection_seeds = parse_csv(args.selection_seeds, int)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / "study_metadata.json",
        {
            "status": "RUNNING",
            "protocol_version": PROTOCOL_VERSION,
            "datasets": datasets,
            "scenarios": scenarios,
            "budgets": budgets,
            "selection_seeds": selection_seeds,
            "eval_simulations": args.eval_simulations,
            "oracle_trajectories_per_source": args.oracle_trajectories,
            "sample_budget": {str(k): sample_budget(k) for k in budgets},
            "master_seed": args.master_seed,
            "hardware": hardware_metadata(),
        },
    )

    completed = 0
    total = len(datasets) * len(scenarios) * len(budgets) * len(selection_seeds)
    for scenario in scenarios:
        if scenario not in core.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        for dataset in datasets:
            graph = core.load_graph(dataset)
            alpha, discard, transfer = core.node_probabilities(graph, scenario)
            q = load_or_build_oracle(
                output_dir,
                graph,
                alpha,
                discard,
                scenario,
                args.oracle_trajectories,
                args.master_seed,
            )
            oracle_order = (
                core.mc_greedy_order(q, max(budgets)) if q is not None else None
            )
            for k in budgets:
                for selection_seed in selection_seeds:
                    path = run_job(
                        output_dir,
                        graph,
                        scenario,
                        alpha,
                        discard,
                        transfer,
                        k,
                        selection_seed,
                        args.eval_simulations,
                        oracle_order,
                        args.force,
                    )
                    completed += 1
                    print(f"[{completed}/{total}] {path}", flush=True)

    metadata_path = output_dir / "study_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "REAL_EXPERIMENT"
    metadata["completed_jobs"] = completed
    write_json_atomic(metadata_path, metadata)


if __name__ == "__main__":
    main()
