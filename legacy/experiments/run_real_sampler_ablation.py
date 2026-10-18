"""Compare uniform-root and conditioned-root estimators on real graphs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from run_real_submission import (
    ROOT,
    _stable_seed,
    conditioned_gates,
    load_graph,
    node_probabilities,
    reverse_coupon_set,
)


def estimate_batch(
    graph,
    seeds: list[int],
    alpha: np.ndarray,
    discard: np.ndarray,
    batch_size: int,
    conditioned: bool,
    seed: int,
) -> tuple[float, float, float, float]:
    batch_started = time.perf_counter()
    rng = random.Random(seed)
    k = len(seeds)
    if conditioned:
        root_weights = 1.0 - np.power(1.0 - alpha, k)
        scale = float(np.sum(root_weights))
        roots = rng.choices(range(graph.n), weights=root_weights, k=batch_size)
    else:
        scale = float(graph.n)
        roots = [rng.randrange(graph.n) for _ in range(batch_size)]

    covered = 0
    zero_contribution = 0
    started = time.perf_counter()
    for root in roots:
        sample_rng = random.Random(rng.getrandbits(64))
        if conditioned:
            gates = conditioned_gates(float(alpha[root]), k, sample_rng)
        else:
            gates = [sample_rng.random() < alpha[root] for _ in range(k)]
        if not any(gates):
            zero_contribution += 1
            continue
        for coupon_index, gate in enumerate(gates):
            if not gate:
                continue
            rr_set = reverse_coupon_set(graph, root, alpha, discard, sample_rng)
            if seeds[coupon_index] in rr_set:
                covered += 1
                break
    elapsed = time.perf_counter() - started
    return scale * covered / batch_size, zero_contribution / batch_size, elapsed, time.perf_counter() - batch_started


def run(args: argparse.Namespace) -> None:
    seed_records = None
    if not args.validated_job_dir:
        seed_records = json.load(open(args.seed_file, encoding="utf-8"))
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    budgets = [int(item) for item in args.budgets.split(",") if item]
    rows: list[dict[str, object]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch_output = output.with_name(output.stem + "_batches.csv")
    if output.exists() or batch_output.exists():
        raise FileExistsError(f"Choose a new output path to preserve prior evidence: {output}")
    batch_fields = ["dataset", "scenario", "k", "sampler", "repeat", "batch_seed", "batch_size", "selection_seed", "spread_estimate", "zero_fraction", "loop_seconds", "batch_seconds", "protocol_version", "status"]
    with batch_output.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=batch_fields).writeheader()
    inputs = {}

    for dataset in datasets:
        graph = load_graph(dataset)
        from run_real_submission import DATASETS
        inputs[str(DATASETS[dataset].relative_to(ROOT))] = hashlib.sha256(DATASETS[dataset].read_bytes()).hexdigest()
        alpha, discard, _ = node_probabilities(graph, args.scenario)
        for budget in budgets:
            if args.validated_job_dir:
                job_path = (
                    Path(args.validated_job_dir)
                    / args.scenario
                    / dataset
                    / f"k{budget}_seed{args.selection_seed}.json"
                )
                payload = json.loads(job_path.read_text(encoding="utf-8"))
                inputs[str(job_path)] = hashlib.sha256(job_path.read_bytes()).hexdigest()
                if payload.get("status") != "REAL_EXPERIMENT":
                    raise ValueError(f"Incomplete validated job: {job_path}")
                seeds = next(
                    method["seeds"]
                    for method in payload["methods"]
                    if method["method"] == "CIM-RIS"
                )
            else:
                seeds = next(
                    record["seeds"]
                    for record in seed_records
                    if record["dataset"] == dataset
                    and record["scenario"] == args.scenario
                    and record["k"] == budget
                    and record["method"] == "CIM-RIS"
                )
            for conditioned in (False, True):
                label = "Conditioned root" if conditioned else "Uniform root"
                estimates: list[float] = []
                zero_fractions: list[float] = []
                elapsed_values: list[float] = []
                loop_values: list[float] = []
                for repeat in range(args.repeats):
                    batch_seed = _stable_seed(args.seed, dataset, budget, label, repeat)
                    estimate, zero_fraction, loop_elapsed, elapsed = estimate_batch(
                        graph,
                        seeds,
                        alpha,
                        discard,
                        args.batch_size,
                        conditioned,
                        batch_seed,
                    )
                    estimates.append(estimate)
                    zero_fractions.append(zero_fraction)
                    elapsed_values.append(elapsed)
                    loop_values.append(loop_elapsed)
                    with batch_output.open("a", newline="", encoding="utf-8") as handle:
                        csv.DictWriter(handle, fieldnames=batch_fields).writerow(dict(zip(batch_fields, [dataset, args.scenario, budget, label, repeat, batch_seed, args.batch_size, args.selection_seed, estimate, zero_fraction, loop_elapsed, elapsed, args.protocol_version, "REAL_EXPERIMENT"])))
                    if (repeat + 1) % 10 == 0:
                        print(f"{dataset} k={budget} {label}: {repeat + 1}/{args.repeats} batches", flush=True)
                mean = statistics.mean(estimates)
                std = statistics.stdev(estimates) if len(estimates) > 1 else 0.0
                rows.append({
                    "dataset": dataset,
                    "scenario": args.scenario,
                    "k": budget,
                    "sampler": label,
                    "batch_size": args.batch_size,
                    "repeats": args.repeats,
                    "mean_spread_estimate": f"{mean:.8f}",
                    "std_spread_estimate": f"{std:.8f}",
                    "coefficient_of_variation": f"{(std / mean if mean else 0.0):.8f}",
                    "mean_zero_fraction": f"{statistics.mean(zero_fractions):.8f}",
                    "mean_batch_seconds": f"{statistics.mean(elapsed_values):.8f}",
                    "mean_loop_seconds": f"{statistics.mean(loop_values):.8f}",
                    "timing_scope": "full_batch_including_root_generation",
                    "selection_seed": args.selection_seed,
                    "protocol_version": args.protocol_version,
                    "status": "REAL_EXPERIMENT",
                })
            print(f"{dataset} k={budget} complete", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    metadata = {
        "status": "REAL_EXPERIMENT", "protocol_version": args.protocol_version,
        "arguments": vars(args), "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "inputs_sha256": inputs,
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(), ROOT / "experiments/run_real_submission.py"]},
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [output, batch_output]},
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-file",
        default=str(ROOT / "experiments/results/forwarding-heavy/real_seeds_forwarding-heavy.json"),
    )
    parser.add_argument(
        "--validated-job-dir",
        default="",
        help="Directory containing scenario/dataset/k*_seed*.json jobs.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260715)
    parser.add_argument("--protocol-version", default="validated-v2.2")
    parser.add_argument("--datasets", default="Netscience,EmailEnron")
    parser.add_argument("--budgets", default="10,50,200")
    parser.add_argument("--scenario", default="forwarding-heavy")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--output",
        default=str(ROOT / "experiments/results/sampler/real_sampler_ablation.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
