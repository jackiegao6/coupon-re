"""Measure CIM-RIS selection time on the full Douban graph."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import run_real_submission as core
from run_validated_study import PROTOCOL_VERSION, sample_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="network.douban")
    parser.add_argument("--scenario", default="balanced")
    parser.add_argument("--budgets", default="10,25,50,100,150,200")
    parser.add_argument(
        "--selection-seeds", default="20260715,20260716,20260717,20260718,20260719"
    )
    parser.add_argument(
        "--output",
        default="experiments/results/validated-v2/validated_scalability.csv",
    )
    args = parser.parse_args()
    budgets = [int(v) for v in args.budgets.split(",") if v]
    selection_seeds = [int(v) for v in args.selection_seeds.split(",") if v]
    graph = core.load_graph(args.dataset)
    alpha, discard, _ = core.node_probabilities(graph, args.scenario)
    rows: list[dict[str, object]] = []
    for k in budgets:
        for selection_seed in selection_seeds:
            samples = sample_budget(k)
            seeds, seconds, estimate, memberships = core.cim_ris_seeds(
                graph,
                alpha,
                discard,
                k,
                samples,
                core._stable_seed(
                    selection_seed, graph.name, args.scenario, k, "cim-ris"
                ),
            )
            rows.append(
                {
                    "dataset": graph.name,
                    "scenario": args.scenario,
                    "nodes": graph.n,
                    "edges": graph.m,
                    "k": k,
                    "selection_seed": selection_seed,
                    "rr_samples": samples,
                    "selection_seconds": f"{seconds:.8f}",
                    "estimated_spread": f"{estimate:.8f}",
                    "rr_memberships": memberships,
                    "selected_seed_count": len(seeds),
                    "protocol_version": PROTOCOL_VERSION,
                    "status": "REAL_EXPERIMENT",
                }
            )
            print(f"{graph.name} k={k} seed={selection_seed} complete", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
