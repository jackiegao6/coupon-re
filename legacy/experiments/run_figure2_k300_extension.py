"""Extend the measured six-network Figure 2 to k=250 and k=300.

The first 200 SSR component streams are replayed exactly from the original
k=200 experiment; independent streams supply components 201--300. Existing
raw runs are read from z-assets-2 and never overwritten. New results and a
combined CSV/PDF are written to z-assets-21.
"""
from __future__ import annotations

from array import array
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import time

import numpy as np

import forward_coverage_baselines as forward
import run_real_submission as core
import run_figure2_six_networks as base
from uniform_ssr import CouponRR

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "z-assets-2"
OUT = ROOT / "z-assets-21"
NEW_KS = [250, 300]
ALL_KS = base.KS + NEW_KS
T_CRITICAL_DF2 = base.T_CRITICAL_DF2


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def old_job(name: str, replicate: int, old_manifest: dict) -> dict:
    path = OLD / "jobs" / f"{name}-r{replicate}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["dataset_sha256"] == old_manifest["datasets"][name]
    assert value["protocol_sha256"] == old_manifest["protocol_sha256"]
    assert len(value["rows"]) == len(base.KS) * len(base.METHODS)
    return value


def extended_ssr_order(graph, alpha, discard, original_seed: int,
                       additional_seed: int):
    """Replay old worlds exactly, adding 100 independent worlds per root."""
    sampler = CouponRR(graph, alpha, discard)
    original_rng = random.Random(original_seed)
    additional_rng = random.Random(additional_seed)
    code = "I" if graph.n <= 2**32 else "Q"
    index = [defaultdict(lambda: array(code)) for _ in range(300)]
    for sample_id in range(graph.n):
        root = original_rng.randrange(graph.n)
        for component in range(200):
            for node in sampler.sample(root, original_rng):
                index[component][node].append(sample_id)
        for component in range(200, 300):
            for node in sampler.sample(root, additional_rng):
                index[component][node].append(sample_id)
    bank = SimpleNamespace(n=graph.n, theta=graph.n, k=300, index=index)
    selected = base.greedy_prefixes(bank)
    return selected, sampler.stats()


def run_job(name: str, replicate: int, manifest: dict, old_manifest: dict):
    path = OUT / "jobs" / f"{name}-r{replicate}.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["protocol_sha256"] == manifest["protocol_sha256"]
        assert value["dataset_sha256"] == manifest["datasets"][name]
        assert len(value["rows"]) == len(NEW_KS) * len(base.METHODS)
        print(f"REUSED {name} r{replicate}", flush=True)
        return
    old = old_job(name, replicate, old_manifest)
    prior = {(row["k"], row["method"]): row for row in old["rows"]}
    started = time.perf_counter()
    graph = base.load_graph(name)
    alpha, discard, transfer = base.behavior(graph)
    old_seed = core._stable_seed("figure2-six-measured-20261015", name, replicate)
    orders = core.static_orders(graph, alpha, transfer, old_seed + 1)
    for method, order in orders.items():
        assert order[:200] == prior[(200, method)]["seeds"], method
    print(f"{name} r{replicate}: static orders verified", flush=True)

    started_forward = time.perf_counter()
    bank = forward.sample_coverage(graph, alpha, discard,
                                   base.FORWARD_BATCHES, old_seed + 2)
    orders["Forward-CELF"], _, _ = forward.select_coverage(bank, 300, "lazy")
    del bank
    assert orders["Forward-CELF"][:200] == prior[(200, "Forward-CELF")]["seeds"]
    forward_seconds = time.perf_counter() - started_forward
    print(f"{name} r{replicate}: Forward-CELF prefix verified ({forward_seconds:.1f}s)",
          flush=True)

    started_ssr = time.perf_counter()
    orders["Uniform-SSR"], ssr_stats = extended_ssr_order(
        graph, alpha, discard, old_seed + 3,
        core._stable_seed("figure2-six-extra-components", name, replicate))
    assert orders["Uniform-SSR"][:200] == prior[(200, "Uniform-SSR")]["seeds"]
    ssr_seconds = time.perf_counter() - started_ssr
    print(f"{name} r{replicate}: CIM-RIS prefix verified ({ssr_seconds:.1f}s)",
          flush=True)

    rows = []
    for k in NEW_KS:
        for method in base.METHODS:
            chosen = orders[method][:k]
            evaluation_seed = core._stable_seed("figure2-six-eval", name,
                                                 replicate, method, k)
            spread, within_ci, variance, redemptions = core.evaluate(
                graph, chosen, alpha, discard, base.EVALUATIONS, evaluation_seed)
            assert 0 <= spread <= redemptions + 1e-7 <= k + 1e-7
            rows.append(dict(dataset=name, replicate=replicate, k=k,
                             method=method, spread=spread,
                             consumed_coupons=redemptions,
                             within_run_ci95=within_ci,
                             evaluation_variance=variance,
                             evaluation_seed=evaluation_seed,
                             selection_seed=old_seed,
                             seeds=chosen))
        print(f"{name} r{replicate}: k={k} evaluated", flush=True)
    result = dict(dataset=name, replicate=replicate, n=graph.n, m=graph.m,
                  dataset_sha256=manifest["datasets"][name],
                  protocol_sha256=manifest["protocol_sha256"], rows=rows,
                  forward_seconds=forward_seconds, ssr_seconds=ssr_seconds,
                  ssr_stats=ssr_stats,
                  wall_seconds=time.perf_counter() - started)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)
    print(f"COMPLETE {name} r{replicate} ({result['wall_seconds']:.1f}s)", flush=True)


def summarize(manifest: dict, old_manifest: dict):
    raw = []
    for name in base.DATASETS:
        for replicate in range(base.REPEATS):
            raw.extend(old_job(name, replicate, old_manifest)["rows"])
            path = OUT / "jobs" / f"{name}-r{replicate}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            assert value["protocol_sha256"] == manifest["protocol_sha256"]
            assert value["dataset_sha256"] == manifest["datasets"][name]
            assert len(value["rows"]) == len(NEW_KS) * len(base.METHODS)
            raw.extend(value["rows"])
    assert len(raw) == len(base.DATASETS) * base.REPEATS * len(ALL_KS) * len(base.METHODS)
    summary = []
    for name in base.DATASETS:
        for k in ALL_KS:
            for method in base.METHODS:
                group = sorted((row for row in raw if row["dataset"] == name
                                and row["k"] == k and row["method"] == method),
                               key=lambda row: row["replicate"])
                assert [row["replicate"] for row in group] == list(range(base.REPEATS))
                spreads = np.asarray([row["spread"] for row in group])
                mean = float(spreads.mean())
                ci = float(T_CRITICAL_DF2 * spreads.std(ddof=1) / math.sqrt(base.REPEATS))
                summary.append(dict(dataset=base.DISPLAY.get(name, name),
                                    dataset_id=name, k=k, method=method,
                                    runs=base.REPEATS,
                                    mean_spread=mean, spread_ci95_halfwidth=ci,
                                    spread_per_coupon=mean/k,
                                    per_coupon_ci95_halfwidth=ci/k,
                                    mean_consumed_coupons=float(np.mean(
                                        [r["consumed_coupons"] for r in group])),
                                    theta_factor=base.THETA_FACTOR,
                                    forward_batches=base.FORWARD_BATCHES,
                                    evaluation_trials=base.EVALUATIONS))
    OUT.mkdir(exist_ok=True)
    with (OUT / "figure-2-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with (OUT / "figure-2-raw.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["dataset", "replicate", "k", "method", "spread",
                  "consumed_coupons", "within_run_ci95", "evaluation_variance",
                  "evaluation_seed", "selection_seed", "seeds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in raw:
            writer.writerow({**row, "seeds": json.dumps(row["seeds"])})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", nargs=2, metavar=("DATASET", "REPLICATE"))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    old_manifest = json.loads((OLD / "figure-2-protocol.json").read_text(encoding="utf-8"))
    assert old_manifest["source_sha256"] == sha(Path(base.__file__))
    assert old_manifest["budgets"] == base.KS
    assert old_manifest["repeats"] == base.REPEATS
    data_files = {name: base.EPINIONS if name == "SNAP-Epinions"
                  else core.DATASETS[name] for name in base.DATASETS}
    hashes = {name: sha(path) for name, path in data_files.items()}
    assert hashes == old_manifest["datasets"]
    protocol = dict(base_protocol_sha256=old_manifest["protocol_sha256"],
                    base_source_sha256=old_manifest["source_sha256"],
                    extension_source_sha256=sha(Path(__file__)),
                    base_raw_sha256=sha(OLD / "figure-2-raw.csv"),
                    datasets=hashes, budgets=ALL_KS, newly_run_budgets=NEW_KS,
                    repeats=base.REPEATS, theta_factor=base.THETA_FACTOR,
                    forward_batches=base.FORWARD_BATCHES,
                    evaluation_trials=base.EVALUATIONS,
                    behavior=old_manifest["behavior"],
                    ssr_extension="Replay original 200-component worlds exactly, "
                                  "then sample 100 independent additional coupon "
                                  "worlds for each of the same n roots; verify old "
                                  "k=200 selected prefixes before evaluation")
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest = {**protocol, "protocol_sha256": protocol_sha}
    OUT.mkdir(exist_ok=True)
    path = OUT / "figure-2-protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.job:
        name, replicate = args.job
        if name not in base.DATASETS or not 0 <= int(replicate) < base.REPEATS:
            parser.error("Unknown job")
        run_job(name, int(replicate), manifest, old_manifest)
    elif args.summarize:
        summarize(manifest, old_manifest)
        print("WROTE combined 9-budget CSV files", flush=True)
    else:
        for name in base.DATASETS:
            for replicate in range(base.REPEATS):
                run_job(name, replicate, manifest, old_manifest)
        summarize(manifest, old_manifest)
        print("WROTE combined 9-budget CSV files", flush=True)


if __name__ == "__main__":
    main()
