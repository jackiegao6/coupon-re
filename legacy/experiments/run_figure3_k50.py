"""Resumable k=50 extension of Figure 3, preserving all earlier evaluations."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_uniform_ssr_study as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/figure3-k50-20261015"
DATASETS = ["Netscience", "NetFacebookEgo", "DoubanRandom"]
SCENARIOS = ["adoption-heavy", "forwarding-heavy"]
METHODS = ["Uniform-SSR", "Forward-CELF", "1Hop-Sort", "Alpha-Sort",
           "PageRank", "DegreeTopM", "Random"]
SOURCES = study.SOURCE_FILES + ["experiments/run_figure3_k50.py"]


def specs():
    return [dict(id=f"figure3-{d}-{s}-k50-r{r}", dataset=d,
                 scenario=s, replicate=r, k=50, theta_factor=8,
                 forward_batches=256, evaluation_trials=5000)
            for d in DATASETS for s in SCENARIOS for r in range(5)]


def verify(result, job):
    assert result["status"] == "complete" and result["job"] == job
    assert len(result["rows"]) == len(METHODS)
    assert {r["method"] for r in result["rows"]} == set(METHODS)
    evaluation_seeds = set()
    selection_seeds = set()
    for row in result["rows"]:
        assert row["dataset"] == job["dataset"]
        assert row["scenario"] == job["scenario"]
        assert row["replicate"] == job["replicate"]
        assert len(row["seeds"]) == len(set(row["seeds"])) == row["k"] == job["k"]
        assert row["forward_simulations"] == job["evaluation_trials"]
        assert 0 <= row["spread"] <= row["consumed_coupons"] + 1e-9 <= job["k"] + 1e-9
        evaluation_seeds.add(row["evaluation_seed"])
        selection_seeds.add(row["selection_seed"])
        if row["method"] == "Uniform-SSR":
            assert row["theta"] == job["theta_factor"] * result["metadata"]["n"]
            assert row["rr_positions"] == row["theta"] * job["k"]
        if row["method"] == "Forward-CELF":
            assert row["forward_batches"] == job["forward_batches"]
    assert len(evaluation_seeds) == len(METHODS)
    assert not evaluation_seeds & selection_seeds


def run(job):
    started = time.perf_counter()
    g, a, d, tr, metadata = study.prepare(job)
    seed = study.core._stable_seed("figure3-k50", job["id"])
    orders = study.core.static_orders(g, a, tr, seed)
    infos = {m: {"selection_seed": seed} for m in orders}
    start_bank = time.perf_counter()
    bank = study.forward.sample_coverage(g, a, d, job["forward_batches"], seed + 1)
    sampling_seconds = time.perf_counter() - start_bank
    start_greedy = time.perf_counter()
    orders["Forward-CELF"], count, evaluations = study.forward.select_coverage(bank, job["k"], "lazy")
    infos["Forward-CELF"] = dict(
        selection_seed=seed + 1, forward_batches=job["forward_batches"],
        forward_sampling_seconds=sampling_seconds,
        selection_seconds=sampling_seconds + time.perf_counter() - start_greedy,
        forward_trajectories=job["forward_batches"] * g.n,
        marginal_evaluations=evaluations,
        training_estimate_not_quality=count / job["forward_batches"])
    del bank
    orders["Uniform-SSR"], infos["Uniform-SSR"] = study.uniform_selection(
        g, a, d, job["k"], job["theta_factor"] * g.n, seed + 2)
    infos["Uniform-SSR"]["theta_factor"] = job["theta_factor"]
    rows = []
    for method in METHODS:
        chosen = orders[method][:job["k"]]
        rows.append(study.evaluate_selection(g, a, d, chosen, job, method,
                    infos[method], simulations=job["evaluation_trials"]))
        print(f"{job['id']}: {method} evaluated", flush=True)
    result = dict(job=job, status="complete", metadata=metadata, rows=rows,
                  environment=study.environment(), wall_seconds=time.perf_counter() - started)
    verify(result, job)
    return result


def initialize():
    manifest = dict(jobs=specs(),
        source_sha256={p: study.hash_file(ROOT / p) for p in SOURCES},
        dataset_sha256={d: study.hash_file(study.core.DATASETS[d]) for d in DATASETS},
        protocol="Figure 3 extension: k=50; unchanged adoption-heavy and forwarding-heavy "
                 "parameters; seven displayed methods; five independent runs; CIM-RIS theta=8n; "
                 "Forward-CELF 256 trajectories/source; 5000 independent forward campaigns "
                 "per solution. All jobs retained. Parallel timings are not comparative "
                 "runtime measurements. Original k=5,25 runs reused without modification.")
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == manifest, "Protocol/source changed"
    else:
        study.write_json(path, manifest)
        for rel in SOURCES:
            dest = OUT / "source" / Path(rel).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
    return manifest


def launch(job):
    dest = OUT / "jobs" / f"{job['id']}.json"
    if dest.exists():
        verify(json.loads(dest.read_text(encoding="utf-8")), job)
        return job["id"] + " reused"
    log = OUT / "logs" / f"{job['id']}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run([sys.executable, __file__, "--job", job["id"]],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Job failed; see {log}")
    verify(json.loads(dest.read_text(encoding="utf-8")), job)
    return job["id"] + " complete"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.job:
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        for rel, digest in manifest["source_sha256"].items():
            assert study.hash_file(ROOT / rel) == digest, rel
        job = next(j for j in manifest["jobs"] if j["id"] == args.job)
        assert study.hash_file(study.core.DATASETS[job["dataset"]]) == manifest["dataset_sha256"][job["dataset"]]
        study.write_json(OUT / "jobs" / f"{job['id']}.json", run(job))
    else:
        manifest = initialize()
        print(f"Registered {len(manifest['jobs'])} jobs / 210 evaluated solutions.", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(launch, j) for j in manifest["jobs"]]):
                print(future.result(), flush=True)
