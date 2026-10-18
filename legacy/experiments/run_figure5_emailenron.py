"""Registered current-Balanced EmailEnron extension of Figure 5 (k=200)."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import psutil

import run_uniform_ssr_study as study
from run_figure4_balanced import parameters

ROOT = study.ROOT
OUT = ROOT / "experiments/results/figure5-emailenron-20261015"
METHODS = ["Uniform-SSR", "Forward-CELF", "1Hop-Sort", "Alpha-Sort",
           "PageRank", "DegreeTopM", "Random"]
SOURCES = study.SOURCE_FILES + ["experiments/run_figure4_balanced.py",
                              "experiments/run_uniform_ssr_directed.py",
                              "experiments/run_figure5_emailenron.py"]


def specs():
    return [dict(id=f"figure5-EmailEnron-balanced-k200-r{rep}", dataset="EmailEnron",
                 scenario="balanced", replicate=rep, k=200, scale=3,
                 theta_factor=8, forward_batches=256, evaluation_trials=5000)
            for rep in range(5)]


def initialize():
    manifest = dict(jobs=specs(),
        source_sha256={p: study.hash_file(ROOT / p) for p in SOURCES},
        dataset=study.core.DATASETS["EmailEnron"].relative_to(ROOT).as_posix(),
        dataset_sha256=study.hash_file(study.core.DATASETS["EmailEnron"]),
        protocol="Figure 5 EmailEnron extension: current Balanced with variance multiplier "
                 "9, unchanged consumption mean and transfer probabilities; k=200; "
                 "CIM-RIS theta=8n; Forward-CELF 256 trajectories per source; "
                 "all seven displayed methods; five independent runs evaluated with "
                 "5000 fresh forward campaigns per solution. Both redemptions and "
                 "distinct consumers measured on the same campaigns. All registered "
                 "runs retained. Original CSR graph used without further filtering. "
                 "Jobs run sequentially; timings are not used for comparative cost claims.")
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


def verify(result, job):
    assert result["status"] == "complete" and result["job"] == job
    assert len(result["rows"]) == len(METHODS)
    assert {r["method"] for r in result["rows"]} == set(METHODS)
    for row in result["rows"]:
        assert (row["dataset"], row["scenario"], row["replicate"]) == (
            job["dataset"], job["scenario"], job["replicate"])
        assert len(row["seeds"]) == len(set(row["seeds"])) == row["k"] == job["k"]
        assert row["forward_simulations"] == job["evaluation_trials"]
        assert row["evaluation_seed"] != row["selection_seed"]
        assert 0 <= row["spread"] <= row["consumed_coupons"] + 1e-9 <= job["k"] + 1e-9
        if row["method"] == "Uniform-SSR":
            assert row["theta"] == job["theta_factor"] * result["metadata"]["n"]
            assert row["rr_positions"] == row["theta"] * job["k"]
        if row["method"] == "Forward-CELF":
            assert row["forward_batches"] == job["forward_batches"]
    assert len({r["evaluation_seed"] for r in result["rows"]}) == len(METHODS)


def uniform_selection(graph, alpha, discard, job, seed):
    """Same sampler and greedy as study.uniform_selection, with progress logs."""
    started = time.perf_counter()
    sampler = study.CouponRR(graph, alpha, discard)
    preparation = time.perf_counter() - started
    total = job["k"] * job["theta_factor"] * graph.n
    finished = threading.Event()

    def progress():
        while not finished.wait(30):
            print(f"{job['id']}: RR components {sampler.calls:,}/{total:,}; "
                  f"RSS {psutil.Process().memory_info().rss/2**20:.0f} MiB; "
                  f"elapsed {time.perf_counter()-started:.0f}s", flush=True)

    monitor = threading.Thread(target=progress, daemon=True)
    monitor.start()
    try:
        bank = study.sample_ssr(sampler, job["k"], job["theta_factor"] * graph.n, seed)
        print(job["id"] + ": sampling complete; selecting seeds", flush=True)
        chosen, info = bank.greedy()
        info.update(selection_seed=seed, selection_seconds=time.perf_counter()-started,
            sampler_preprocessing_seconds=preparation, sampling_seconds=bank.sampling_seconds,
            theta=bank.theta, theta_factor=job["theta_factor"], rr_positions=total,
            empty_ssr_count=bank.empty_ssr, nonempty_rr_count=bank.stats["nonempty"],
            memberships=bank.stats["memberships"], scanned_edges=bank.stats["scanned_edges"],
            node_random_draws=bank.stats["draw_count"],
            max_rr_size=bank.stats["sampler_max_size_so_far"],
            root_count_sum=int(bank.root_counts.sum()))
        return chosen, info
    finally:
        finished.set()
        monitor.join()


def run(job):
    started = time.perf_counter()
    graph = study.core.load_graph(job["dataset"])
    assert (graph.n, graph.m) == (33696, 361622)
    alpha, discard, transfer, metadata = parameters(graph)
    metadata.update(n=graph.n, m=graph.m,
                    dataset_sha256=study.hash_file(study.core.DATASETS[job["dataset"]]))
    seed = study.core._stable_seed("figure5-emailenron", job["id"])
    checkpoint = OUT / "partial" / f"{job['id']}.json"
    rows = []
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert previous["job"] == job and previous["metadata"] == metadata
        rows = previous["rows"]
    completed = {r["method"] for r in rows}
    orders = study.core.static_orders(graph, alpha, transfer, seed)
    infos = {m: dict(selection_seed=seed) for m in orders}
    if "Forward-CELF" not in completed:
        begin = time.perf_counter()
        bank = study.forward.sample_coverage(graph, alpha, discard, job["forward_batches"], seed+1)
        sampling_seconds = time.perf_counter()-begin
        orders["Forward-CELF"], count, evaluations = study.forward.select_coverage(bank, job["k"], "lazy")
        infos["Forward-CELF"] = dict(selection_seed=seed+1,
            forward_batches=job["forward_batches"], forward_sampling_seconds=sampling_seconds,
            selection_seconds=time.perf_counter()-begin,
            forward_trajectories=graph.n*job["forward_batches"], marginal_evaluations=evaluations,
            training_estimate_not_quality=count/job["forward_batches"])
        del bank
        gc.collect()
    for method in METHODS[1:] + METHODS[:1]:
        if method in completed:
            continue
        if method == "Uniform-SSR":
            chosen, info = uniform_selection(graph, alpha, discard, job, seed+2)
        else:
            chosen, info = orders[method][:job["k"]], infos[method]
        info["selected_mean_consumption_probability"] = float(alpha[chosen].mean())
        rows.append(study.evaluate_selection(graph, alpha, discard, chosen, job, method,
                    info, simulations=job["evaluation_trials"]))
        study.write_json(checkpoint, dict(job=job, status="in_progress", metadata=metadata, rows=rows))
        print(f"{job['id']}: {method} evaluated; "
              f"sigma={rows[-1]['spread']:.4f}, U={rows[-1]['consumed_coupons']:.4f}", flush=True)
    rows.sort(key=lambda r: METHODS.index(r["method"]))
    result = dict(job=job, status="complete", rows=rows, metadata=metadata,
        environment=study.environment(), wall_seconds_current_attempt=time.perf_counter()-started)
    verify(result, job)
    study.write_json(checkpoint, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job")
    args = parser.parse_args()
    manifest = initialize()
    if args.job:
        job = next(j for j in manifest["jobs"] if j["id"] == args.job)
        study.write_json(OUT / "jobs" / f"{job['id']}.json", run(job))
    else:
        for job in manifest["jobs"]:
            dest = OUT / "jobs" / f"{job['id']}.json"
            if dest.exists():
                verify(json.loads(dest.read_text(encoding="utf-8")), job)
                print(job["id"] + " reused", flush=True)
                continue
            log = OUT / "logs" / f"{job['id']}.txt"
            log.parent.mkdir(parents=True, exist_ok=True)
            print("START " + job["id"], flush=True)
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run([sys.executable, __file__, "--job", job["id"]],
                                        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"Failed {job['id']}; see {log}")
            verify(json.loads(dest.read_text(encoding="utf-8")), job)
            print("COMPLETE " + job["id"], flush=True)
