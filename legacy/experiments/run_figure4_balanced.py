"""Rerun only Figure 4's Epinions Balanced panel with current parameters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import run_uniform_ssr_directed as directed
import run_uniform_ssr_study as study

ROOT = study.ROOT
OUT = ROOT / "experiments/results/figure4-balanced-20261015"
METHODS = ["Uniform-SSR", "Forward-CELF", "1Hop-Sort", "Alpha-Sort",
           "PageRank", "DegreeTopM", "Random"]
SOURCES = study.SOURCE_FILES + ["experiments/run_uniform_ssr_directed.py",
                              "experiments/run_figure4_balanced.py"]


def specs():
    return [dict(id=f"figure4-Epinions-balanced-r{rep}", dataset="SNAP-Epinions",
                 scenario="balanced", replicate=rep, k=10, scale=3,
                 theta_factor=4, forward_batches=32, evaluation_trials=5000)
            for rep in range(3)]


def parameters(graph):
    original, _, transfer = study.core.node_probabilities(graph, "balanced")
    alpha = original.mean() + 3 * (original - original.mean())
    discard = 1 - transfer - alpha
    assert np.all(alpha >= 0) and np.all(discard >= 0)
    assert np.allclose(alpha + discard + transfer, 1)
    assert np.all(transfer[graph.degrees == 0] == 0)
    assert np.allclose(transfer[graph.degrees > 0], .5)
    assert abs(float(alpha.mean() - original.mean())) < 1e-12
    assert abs(float(alpha.var() - 9 * original.var())) < 1e-12
    metadata = dict(
        formula="a=mean(a_reference)+3*(a_reference-mean(a_reference)); "
                "t=0.5 for non-sinks and 0 for sinks; d=1-t-a; no clipping",
        original_consumption_mean=float(original.mean()),
        original_consumption_variance=float(original.var()),
        consumption_mean=float(alpha.mean()), consumption_variance=float(alpha.var()),
        consumption_min=float(alpha.min()), consumption_max=float(alpha.max()),
        dropping_min=float(discard.min()), dropping_max=float(discard.max()),
        transfer_mean=float(transfer.mean()),
        zero_outdegree_nodes=int(np.sum(graph.degrees == 0)))
    return alpha, discard, transfer, metadata


def initialize():
    manifest = dict(jobs=specs(),
        source_sha256={p: study.hash_file(ROOT / p) for p in SOURCES},
        dataset_sha256=study.hash_file(directed.DATA),
        dataset=directed.DATA.relative_to(ROOT).as_posix(),
        protocol="Current Balanced definition, identical to Figure 2, with the existing "
                 "zero-outdegree rule. Epinions k=10, theta=4n, Forward-CELF 32 "
                 "trajectories/source, three independent runs, 5000 independent forward "
                 "evaluation campaigns per solution. Seven displayed methods; all runs "
                 "retained. Right forwarding-heavy panel is reused unchanged. "
                 "Directed edges preserved without symmetrization or component filtering.")
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


def run(job):
    started = time.perf_counter()
    graph, _, selfloops = directed.load_directed()
    alpha, discard, transfer, metadata = parameters(graph)
    metadata.update(n=graph.n, m=graph.m, self_loops=selfloops,
                    dataset_sha256=study.hash_file(directed.DATA))
    seed = study.core._stable_seed("figure4-balanced", job["id"])
    orders = study.core.static_orders(graph, alpha, transfer, seed)
    infos = {m: dict(selection_seed=seed) for m in orders}
    begin = time.perf_counter()
    bank = study.forward.sample_coverage(graph, alpha, discard, job["forward_batches"], seed + 1)
    sampling_seconds = time.perf_counter() - begin
    orders["Forward-CELF"], count, evaluations = study.forward.select_coverage(bank, job["k"], "lazy")
    infos["Forward-CELF"] = dict(selection_seed=seed + 1,
        forward_batches=job["forward_batches"], forward_sampling_seconds=sampling_seconds,
        selection_seconds=time.perf_counter() - begin,
        forward_trajectories=graph.n * job["forward_batches"], marginal_evaluations=evaluations,
        training_estimate_not_quality=count / job["forward_batches"])
    del bank
    print(job["id"] + ": forward selections ready", flush=True)
    orders["Uniform-SSR"], infos["Uniform-SSR"] = study.uniform_selection(
        graph, alpha, discard, job["k"], job["theta_factor"] * graph.n, seed + 2)
    infos["Uniform-SSR"]["theta_factor"] = job["theta_factor"]
    rows = []
    for method in METHODS:
        chosen = orders[method][:job["k"]]
        infos[method]["selected_mean_consumption_probability"] = float(alpha[chosen].mean())
        rows.append(study.evaluate_selection(graph, alpha, discard, chosen, job, method,
                    infos[method], simulations=job["evaluation_trials"]))
        print(job["id"] + ": " + method + " evaluated", flush=True)
    result = dict(job=job, status="complete", rows=rows, metadata=metadata,
                  environment=study.environment(), wall_seconds=time.perf_counter() - started)
    verify(result, job)
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
            path = OUT / "jobs" / f"{job['id']}.json"
            if path.exists():
                verify(json.loads(path.read_text(encoding="utf-8")), job)
                print(job["id"] + " reused", flush=True)
                continue
            log = OUT / "logs" / f"{job['id']}.txt"
            log.parent.mkdir(parents=True, exist_ok=True)
            print("START " + job["id"], flush=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run([sys.executable, __file__, "--job", job["id"]],
                                        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Job failed; see {log}")
            verify(json.loads(path.read_text(encoding="utf-8")), job)
            print("COMPLETE " + job["id"], flush=True)
