"""Measure CIM-RIS selection variability across independent master seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from run_real_submission import (
    ROOT,
    SCENARIOS,
    _stable_seed,
    cim_ris_seeds,
    evaluate,
    load_graph,
    node_probabilities,
    write_rows,
)


DEFAULT_DATASETS = "Netscience,NetFacebookEgo,DoubanRandom,EmailEnron"
DEFAULT_SCENARIOS = "balanced,adoption-heavy,forwarding-heavy"


def _read_original_seeds(scenario: str) -> dict[tuple[str, int], list[int]]:
    path = ROOT / f"experiments/results/{scenario}/real_seeds_{scenario}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return {
        (record["dataset"], int(record["k"])): record["seeds"]
        for record in records
        if record["method"] == "CIM-RIS"
    }


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["scenario"]), int(row["k"]))
        groups.setdefault(key, []).append(float(row["mean_adopters"]))

    output: list[dict[str, object]] = []
    for (dataset, scenario, budget), values in sorted(groups.items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        minimum = min(values)
        maximum = max(values)
        output.append({
            "dataset": dataset,
            "scenario": scenario,
            "k": budget,
            "selection_seeds": len(values),
            "mean_adopters_across_seeds": f"{mean:.8f}",
            "std_across_seeds": f"{std:.8f}",
            "coefficient_of_variation": f"{(std / mean if mean else 0.0):.8f}",
            "min_adopters": f"{minimum:.8f}",
            "max_adopters": f"{maximum:.8f}",
            "relative_range": f"{((maximum - minimum) / mean if mean else 0.0):.8f}",
            "status": "REAL_EXPERIMENT",
        })
    return output


def run(args: argparse.Namespace) -> None:
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    scenarios = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    budgets = sorted({int(value) for value in args.budgets.split(",") if value})
    selection_seeds = [int(value) for value in args.selection_seeds.split(",") if value]
    sample_budgets = {
        "balanced": args.balanced_samples,
        "adoption-heavy": args.other_samples,
        "forwarding-heavy": args.other_samples,
    }
    rows: list[dict[str, object]] = []

    for dataset in datasets:
        print(f"Loading {dataset}...", flush=True)
        graph = load_graph(dataset)
        for scenario in scenarios:
            if scenario not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario}")
            alpha, discard, _ = node_probabilities(graph, scenario)
            original = _read_original_seeds(scenario)
            rr_samples = sample_budgets[scenario]

            for budget in budgets:
                common_eval_seed = _stable_seed(
                    args.evaluation_seed,
                    dataset,
                    scenario,
                    budget,
                    "selection-stability-eval",
                )
                for selection_seed in selection_seeds:
                    if selection_seed == args.original_seed:
                        seeds = original[(dataset, budget)]
                        selection_seconds = ""
                    else:
                        seeds, selection_seconds, _, _ = cim_ris_seeds(
                            graph,
                            alpha,
                            discard,
                            budget,
                            rr_samples,
                            _stable_seed(
                                selection_seed,
                                dataset,
                                scenario,
                                budget,
                                "cim-ris",
                            ),
                        )
                    mean, ci95, variance, redemptions = evaluate(
                        graph,
                        seeds,
                        alpha,
                        discard,
                        args.eval_simulations,
                        common_eval_seed,
                    )
                    rows.append({
                        "dataset": dataset,
                        "scenario": scenario,
                        "nodes": graph.n,
                        "edges": graph.m,
                        "k": budget,
                        "selection_seed": selection_seed,
                        "mean_adopters": f"{mean:.8f}",
                        "ci95": f"{ci95:.8f}",
                        "evaluation_variance": f"{variance:.8f}",
                        "mean_redemptions": f"{redemptions:.8f}",
                        "selection_seconds": (
                            f"{selection_seconds:.8f}"
                            if isinstance(selection_seconds, float)
                            else selection_seconds
                        ),
                        "rr_samples": rr_samples,
                        "eval_simulations": args.eval_simulations,
                        "evaluation_seed": common_eval_seed,
                        "status": "REAL_EXPERIMENT",
                    })
                    print(
                        f"  {dataset} {scenario} k={budget} "
                        f"selection_seed={selection_seed} complete",
                        flush=True,
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_dir / "real_selection_seed_stability.csv",
        list(rows[0]),
        rows,
    )
    summary_rows = _summaries(rows)
    write_rows(
        output_dir / "real_selection_seed_stability_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    metadata = {
        "status": "REAL_EXPERIMENT",
        "runner": "experiments/run_selection_seed_stability.py",
        "datasets": datasets,
        "scenarios": scenarios,
        "budgets": budgets,
        "selection_seeds": selection_seeds,
        "original_seed_reuses_saved_allocation": args.original_seed,
        "evaluation_seed_master": args.evaluation_seed,
        "eval_simulations": args.eval_simulations,
        "rr_samples": sample_budgets,
    }
    (output_dir / "real_selection_seed_stability_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote stability artifacts to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--budgets", default="10,25,50,100,150,200")
    parser.add_argument(
        "--selection-seeds",
        default="20260715,20260716,20260717,20260718,20260719",
    )
    parser.add_argument("--original-seed", type=int, default=20260715)
    parser.add_argument("--evaluation-seed", type=int, default=20260715)
    parser.add_argument("--balanced-samples", type=int, default=10000)
    parser.add_argument("--other-samples", type=int, default=20000)
    parser.add_argument("--eval-simulations", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments/results/stability"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
