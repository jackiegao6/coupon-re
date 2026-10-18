"""Independent forward evaluation on full Douban; preserve every allocation.

Protocol: plan/task-packets/2026-09-13-experiment-evidence-repair.md.
Run serially so measured selection times are not affected by our other jobs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

import run_real_submission as core
from run_validated_study import sample_budget

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/results/evidence-20260913/large-graph"
PROTOCOL = "evidence-20260913-large-graph"
BUDGETS = (10, 25, 50, 100, 150, 200)
SEEDS = tuple(range(20260715, 20260720))
METHODS = ("CIM-RIS", "IC-RIS", "1Hop-Sort", "Alpha-Sort", "DegreeTopM", "PageRank", "Random")
EVALUATIONS = 10_000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    graph = core.load_graph("network.douban")
    alpha, discard, transfer = core.node_probabilities(graph, "balanced")
    fingerprints = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (Path(__file__).resolve(), ROOT / "experiments/run_real_submission.py", ROOT / "experiments/run_validated_study.py", core.DATASETS[graph.name])
    }
    environment = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "processor": platform.processor(), "parallelism": 1}
    static = core.static_orders(graph, alpha, transfer, SEEDS[0])
    raw = []
    for k in BUDGETS:
        for selection_seed in SEEDS:
            key = {"protocol_version": PROTOCOL, "dataset": graph.name, "scenario": "balanced", "k": k, "selection_seed": selection_seed, "rr_samples": sample_budget(k), "eval_simulations": EVALUATIONS, "capacity_per_node": 1, "fingerprints": fingerprints, "environment": environment}
            path = OUTPUT / "jobs" / f"k{k}_seed{selection_seed}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("key") != key or payload.get("status") != "REAL_EXPERIMENT":
                    raise ValueError(f"Existing job differs from code/data/environment: {path}")
                raw.extend(payload["rows"])
                print(f"Reused k={k} seed={selection_seed}", flush=True)
                continue
            cim_seed = core._stable_seed(selection_seed, graph.name, "balanced", k, "cim-ris")
            ic_seed = core._stable_seed(selection_seed, graph.name, "balanced", k, "ic-ris")
            cim, cim_seconds, training_estimate, memberships = core.cim_ris_seeds(graph, alpha, discard, k, sample_budget(k), cim_seed)
            ic, ic_seconds = core.ic_ris_order(graph, transfer, k, sample_budget(k), ic_seed)
            allocations = {name: order[:k] for name, order in static.items() if name != "Random"}
            random_order = list(range(graph.n))
            random_seed = core._stable_seed(selection_seed, graph.name, "balanced", k, "static")
            random.Random(random_seed).shuffle(random_order)
            allocations.update({"CIM-RIS": cim, "IC-RIS": ic, "Random": random_order[:k]})
            eval_seed = core._stable_seed(PROTOCOL, graph.name, "balanced", k, selection_seed, "independent-final-forward")
            assert eval_seed not in (cim_seed, ic_seed, random_seed)
            rng = random.Random(eval_seed)
            streams = [rng.getrandbits(64) for _ in range(EVALUATIONS)]
            rows = []
            for method in METHODS:
                allocation = allocations[method]
                assert len(allocation) == k and len(set(allocation)) == k
                mean, ci, variance, redemptions = core.evaluate_with_streams(graph, allocation, alpha, discard, streams)
                assert 0 <= mean <= redemptions + 1e-9 <= k + 1e-9
                rows.append({"dataset": graph.name, "scenario": "balanced", "k": k, "selection_seed": selection_seed, "method": method, "mean_adopters": mean, "forward_ci95_normal": ci, "forward_variance": variance, "mean_redemptions": redemptions, "eval_simulations": EVALUATIONS, "evaluation_seed": eval_seed, "rr_samples": sample_budget(k) if method in ("CIM-RIS", "IC-RIS") else 0, "selection_seconds": cim_seconds if method == "CIM-RIS" else ic_seconds if method == "IC-RIS" else "", "training_estimate_not_quality": training_estimate if method == "CIM-RIS" else "", "rr_memberships": memberships if method == "CIM-RIS" else "", "seeds": json.dumps(allocation, separators=(",", ":")), "protocol_version": PROTOCOL, "status": "REAL_EXPERIMENT"})
            write_json(path, {"status": "REAL_EXPERIMENT", "key": key, "selection_streams": {"cim": cim_seed, "ic": ic_seed, "random": random_seed}, "rows": rows})
            raw.extend(rows)
            print(f"Completed k={k} seed={selection_seed}; CIM independent spread={rows[0]['mean_adopters']:.4f}", flush=True)
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["k"], row["method"])].append(row)
    summaries = []
    for (k, method), rows in sorted(grouped.items()):
        values = [row["mean_adopters"] for row in rows]
        assert len(values) == 5
        std = statistics.stdev(values)
        summaries.append({"dataset": graph.name, "scenario": "balanced", "k": k, "method": method, "selection_repeats": len(rows), "mean_adopters": statistics.mean(values), "std_across_selection_runs": std, "ci95_across_selection_runs": float(student_t.ppf(.975, len(values)-1)) * std / len(values)**.5, "ci_method": "student_t_two_sided_95", "mean_redemptions": statistics.mean(row["mean_redemptions"] for row in rows), "protocol_version": PROTOCOL, "status": "REAL_EXPERIMENT"})
    write_csv(OUTPUT / "raw.csv", raw)
    write_csv(OUTPUT / "summary.csv", summaries)
    write_json(OUTPUT / "metadata.json", {"status": "REAL_EXPERIMENT", "protocol_version": PROTOCOL, "environment": environment, "fingerprints": fingerprints, "completed_jobs": 30, "rows": len(raw), "summaries": len(summaries), "selection_seeds": SEEDS, "budgets": BUDGETS, "evaluation_simulations": EVALUATIONS, "limitations": "Controlled balanced probabilities; fixed RR budgets; no exact or MC-Greedy reference; timings are from this environment only."})
    paths = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "manifest.sha256")
    (OUTPUT / "manifest.sha256").write_text("".join(f"{digest(p)}  {p.relative_to(OUTPUT).as_posix()}\n" for p in paths), encoding="utf-8")
    print(f"Wrote {len(raw)} raw rows, {len(summaries)} summaries", flush=True)


if __name__ == "__main__":
    main()
