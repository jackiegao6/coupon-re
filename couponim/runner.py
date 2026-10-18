"""Isolated experiment jobs, deterministic streams, checkpoints and CSV output."""
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

from . import __version__, native
from .forward import sample_coverage, select_coverage
from .graph import load_graph
from .heuristics import pagerank_order
from .parameters import node_probabilities
from .rr import CouponRR
from .simulation import evaluate
from .streaming import stream_ssr_greedy

PACKAGE = Path(__file__).resolve().parent
METHODS = ("CIM-RIS", "Forward-CELF", "Random", "Degree", "PageRank", "Alpha", "1Hop")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], "big")


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def prepare(config_path, output, backend, build_dir):
    path = Path(config_path).resolve()
    spec = read(path)
    allowed = {"graphs", "scenarios", "budgets", "theta_factors", "methods", "repeats",
               "evaluation_trials", "forward_batches", "seed"}
    if set(spec) - allowed:
        raise ValueError(f"Unknown config keys: {set(spec) - allowed}")
    spec = {"scenarios": ["balanced-reference"], "budgets": [5], "theta_factors": [8],
            "methods": ["CIM-RIS", "Forward-CELF", "Random"], "repeats": 1,
            "evaluation_trials": 5000, "forward_batches": 256, "seed": 42, **spec}
    for key in ("repeats", "evaluation_trials", "forward_batches"):
        positive_int(spec[key], key)
    if not isinstance(spec["seed"], int) or isinstance(spec["seed"], bool):
        raise ValueError("seed must be an integer")
    for key in ("budgets", "theta_factors", "methods", "scenarios"):
        values = spec[key]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be a nonempty list of distinct values")
    for key in ("budgets", "theta_factors"):
        for value in spec[key]:
            positive_int(value, key)
    if set(spec["methods"]) - set(METHODS):
        raise ValueError(f"Methods must be drawn from {METHODS}")
    if not isinstance(spec.get("graphs"), list) or not spec["graphs"]:
        raise ValueError("graphs must be a nonempty list")
    graphs, seen = [], set()
    for graph_spec in spec["graphs"]:
        item = dict(graph_spec)
        if "path" in item:
            item["path"] = str((path.parent / item["path"]).resolve())
        g = load_graph(item, path.parent)
        name = item.get("name", g.name)
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Graph names must be nonempty and unique")
        seen.add(name)
        item["name"] = name
        if max(spec["budgets"]) > g.n or max(spec["theta_factors"]) * g.n >= 2**32:
            raise ValueError(f"Budget outside implementation limits for {name}")
        for scenario in spec["scenarios"]:
            node_probabilities(g, scenario)
        graphs.append({"spec": item, "n": g.n, "m": g.m,
                       "sha256": digest(item["path"]) if "path" in item else "generated-at-runtime"})
    spec["graphs"] = [g["spec"] for g in graphs]
    sources = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted(PACKAGE.rglob("*"))
               if p.suffix in {".py", ".cpp", ".hpp"}}
    identity = {"version": __version__, "config": spec, "graphs": graphs,
                "backend": backend, "sources": sources,
                "environment": {"python": sys.version, "numpy": np.__version__,
                                "psutil": psutil.__version__, "platform": platform.platform()}}
    if backend == "cpp":
        native.build(build_dir)
        identity["native_build"] = read(Path(build_dir) / "build.json")
    jobs = []
    for graph, scenario, k, method, replicate in itertools.product(
            graphs, spec["scenarios"], spec["budgets"], spec["methods"], range(spec["repeats"])):
        for factor in spec["theta_factors"] if method == "CIM-RIS" else [None]:
            job = {"dataset": graph["spec"]["name"], "graph": graph["spec"], "n": graph["n"],
                   "m": graph["m"], "scenario": scenario, "k": k, "method": method,
                   "replicate": replicate, "theta_factor": factor,
                   "evaluation_trials": spec["evaluation_trials"], "forward_batches": spec["forward_batches"]}
            job["id"] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
            job["selection_seed"] = stable(spec["seed"], job["id"], "selection")
            job["evaluation_seed"] = stable(spec["seed"], job["id"], "evaluation")
            jobs.append(job)
    protocol = {"identity": identity, "jobs": jobs}
    output = Path(output).resolve()
    target = output / "protocol.json"
    if target.exists():
        if read(target) != protocol:
            raise ValueError("Output protocol differs (data/config/code/environment). Use a new --output directory")
    else:
        write(target, protocol)
    return protocol


def choose_static(graph, alpha, transfer, method, seed, k):
    if method == "Random":
        return random.Random(seed).sample(range(graph.n), k)
    if method == "PageRank":
        return pagerank_order(graph)[:k]
    score = graph.degrees if method == "Degree" else alpha.copy()
    if method == "1Hop":
        for u in range(graph.n):
            neighbors = graph.out_neighbors(u)
            if len(neighbors):
                # A transfer back to self is discarded on revisit.
                score[u] += transfer[u] * float(alpha[neighbors[neighbors != u]].sum()) / len(neighbors)
    return np.argsort(-score, kind="stable")[:k].tolist()


def run_job(output, job_id, build_dir):
    output = Path(output).resolve()
    protocol = read(output / "protocol.json")
    job = next(j for j in protocol["jobs"] if j["id"] == job_id)
    backend = protocol["identity"]["backend"]
    folder = output / "work" / job_id
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    graph = load_graph(job["graph"], output)
    alpha, discard, transfer = node_probabilities(graph, job["scenario"])
    graph_load_seconds = time.perf_counter() - started
    seed, k = job["selection_seed"], job["k"]
    checkpoint = folder / "selection.json"
    model = None
    if checkpoint.exists():
        selected_result = read(checkpoint)
        if selected_result["job"] != job:
            raise ValueError("Selection checkpoint does not match job")
        seeds, info = selected_result["seeds"], selected_result["selection"]
    else:
        selection_start = time.perf_counter()
        info = {}
        if job["method"] in {"CIM-RIS", "Forward-CELF"} and backend == "cpp":
            model = native.NativeModel(graph, alpha, discard, build_dir, folder)
            info["backend_preparation_seconds"] = model.preparation_seconds
        if job["method"] == "CIM-RIS":
            theta = graph.n * job["theta_factor"]
            root_seed = stable(seed, "roots")
            component_seeds = [stable(seed, "component", j) for j in range(k)]
            if backend == "cpp":
                seeds, diagnostics = model.stream(k, theta, root_seed, component_seeds)
            else:
                seeds, diagnostics = stream_ssr_greedy(CouponRR(graph, alpha, discard), k,
                                                       theta, root_seed, component_seeds)
            info.update(diagnostics)
            info["root_seed"] = root_seed
            info["component_seeds"] = component_seeds
        elif job["method"] == "Forward-CELF":
            phase = time.perf_counter()
            bank = (model.forward_bank(job["forward_batches"], seed) if backend == "cpp" else
                    sample_coverage(graph, alpha, discard, job["forward_batches"], seed))
            info["sampling_seconds"] = time.perf_counter() - phase
            phase = time.perf_counter()
            seeds, count, evaluations = select_coverage(bank, k, mode="lazy")
            info.update(greedy_seconds=time.perf_counter()-phase,
                        training_estimate_not_quality=count/job["forward_batches"],
                        marginal_recomputations=evaluations,
                        outcome_array_bytes_not_peak_memory=sum(row.nbytes for row in bank))
            del bank
        else:
            seeds = choose_static(graph, alpha, transfer, job["method"], seed, k)
        if "selection_seconds" in info:
            info["algorithm_seconds"] = info.pop("selection_seconds")
        info["selection_seconds"] = time.perf_counter() - selection_start
        write(checkpoint, {"job": job, "seeds": seeds, "selection": info})
    if len(seeds) != k or len(set(seeds)) != k or any(not 0 <= s < graph.n for s in seeds):
        raise ValueError("Selection does not contain k distinct valid seeds")
    phase = time.perf_counter()
    if backend == "cpp":
        if model is None:
            model = native.NativeModel(graph, alpha, discard, build_dir, folder)
        evaluation = model.evaluate(seeds, job["evaluation_trials"], job["evaluation_seed"])
    else:
        mean, ci, variance, redeemed = evaluate(graph, seeds, alpha, discard,
                                              job["evaluation_trials"], job["evaluation_seed"])
        evaluation = {"spread": mean, "within_run_ci95_halfwidth": ci,
                      "evaluation_variance": variance, "consumed_coupons": redeemed}
    evaluation["evaluation_wall_seconds"] = time.perf_counter() - phase
    if not 0 <= evaluation["spread"] <= evaluation["consumed_coupons"] + 1e-9 <= k + 1e-9:
        raise ValueError("Evaluation violates 0 <= sigma <= U <= k")
    result = {"status": "complete", "job": job, "seeds": seeds, "selection": info,
              "evaluation": evaluation, "graph_load_seconds": graph_load_seconds, "completed_utc": now()}
    write(output / "jobs" / f"{job_id}.json", result)


def csv_write(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def publish(output, protocol):
    output = Path(output)
    rows, groups = [], {}
    for job in protocol["jobs"]:
        path = output / "jobs" / f'{job["id"]}.json'
        if not path.exists():
            continue
        record = read(path)
        if record.get("status") != "complete" or record["job"] != job:
            raise ValueError(f"Invalid completed record: {path}")
        selection, evaluation = record["selection"], record["evaluation"]
        row = {key: job[key] for key in ("id", "dataset", "scenario", "k", "method", "replicate",
                                        "theta_factor", "evaluation_trials")}
        row.update(backend=protocol["identity"]["backend"], spread=evaluation["spread"],
                   sigma_over_k=evaluation["spread"]/job["k"], consumed_coupons=evaluation["consumed_coupons"],
                   repeat_redemptions=evaluation["consumed_coupons"]-evaluation["spread"],
                   within_run_ci95_halfwidth=evaluation["within_run_ci95_halfwidth"],
                   selection_seconds=selection["selection_seconds"],
                   native_peak_private_bytes=selection.get("peak_private_bytes"),
                   native_peak_rss_bytes=selection.get("peak_working_set_bytes"))
        rows.append(row)
        key = (job["dataset"], job["scenario"], job["k"], job["method"], job["theta_factor"])
        groups.setdefault(key, []).append(row)
    fields = ["id", "dataset", "scenario", "k", "method", "replicate", "theta_factor", "evaluation_trials",
              "backend", "spread", "sigma_over_k", "consumed_coupons", "repeat_redemptions",
              "within_run_ci95_halfwidth", "selection_seconds", "native_peak_private_bytes", "native_peak_rss_bytes"]
    csv_write(output / "raw.csv", rows, fields)
    summaries = []
    for key, group in groups.items():
        summary = dict(zip(("dataset", "scenario", "k", "method", "theta_factor"), key))
        summary.update(runs=len(group), mean_spread=statistics.mean(r["spread"] for r in group),
                       std_across_run_means=statistics.stdev(r["spread"] for r in group) if len(group)>1 else None,
                       mean_consumed_coupons=statistics.mean(r["consumed_coupons"] for r in group),
                       mean_selection_seconds=statistics.mean(r["selection_seconds"] for r in group))
        summaries.append(summary)
    csv_write(output / "summary.csv", summaries,
              ["dataset", "scenario", "k", "method", "theta_factor", "runs", "mean_spread",
               "std_across_run_means", "mean_consumed_coupons", "mean_selection_seconds"])
    return len(rows)


def terminate_tree(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=3)
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    process.wait()


def run(config, output, backend="python", workers=1, build_dir="build", timeout=None):
    positive_int(workers, "workers")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "RUNNING.json"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_utc": now()}, stream)
    except FileExistsError:
        raise RuntimeError(f"{lock} exists; ensure its runner has stopped before removing this lock") from None
    active, failed = {}, 0
    protocol = None
    try:
        protocol = prepare(config, output, backend, build_dir)
        pending = [j for j in protocol["jobs"] if not (output / "jobs" / f'{j["id"]}.json').exists()]
        reused = len(protocol["jobs"]) - len(pending)
        print(f"{len(pending)} pending; {reused} completed jobs reused; workers={workers}", flush=True)
        write(output / "execution.json", {"workers": workers, "started_utc": now(),
              "timing_note": "Concurrent wall times are not serial benchmark times" if workers>1 else "Serial execution"})
        (output / "logs").mkdir(exist_ok=True)
        stopped = False
        last_publish = 0.0
        completed_count = publish(output, protocol)
        while pending or active:
            if (output / "STOP").exists():
                stopped = True
                break
            while pending and len(active) < workers:
                job = pending.pop(0)
                log = (output / "logs" / f'{job["id"]}.log').open("a", encoding="utf-8")
                env = os.environ.copy()
                env["PYTHONPATH"] = str(PACKAGE.parent) + os.pathsep + env.get("PYTHONPATH", "")
                env["OMP_NUM_THREADS"] = "1"
                env["OPENBLAS_NUM_THREADS"] = "1"
                process = subprocess.Popen([sys.executable, "-m", "couponim", "_job", "--output", str(output),
                    "--job-id", job["id"], "--build-dir", str(Path(build_dir).resolve())],
                    stdout=log, stderr=subprocess.STDOUT, env=env)
                active[job["id"]] = {"process": process, "log": log, "start": time.perf_counter(), "peak": 0,
                                     "job": job, "timed_out": False}
                print(f'START {job["dataset"]} {job["method"]} k={job["k"]} factor={job["theta_factor"]}', flush=True)
            for job_id, item in list(active.items()):
                proc = item["process"]
                try:
                    parent = psutil.Process(proc.pid)
                    processes = [parent, *parent.children(recursive=True)]
                    rss = 0
                    for process in processes:
                        try:
                            rss += process.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    item["peak"] = max(item["peak"], rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if timeout and time.perf_counter()-item["start"] > timeout and proc.poll() is None:
                    item["timed_out"] = True
                    terminate_tree(proc)
                if proc.poll() is None:
                    continue
                item["log"].close()
                complete = proc.returncode == 0 and (output / "jobs" / f"{job_id}.json").exists()
                status = "complete" if complete else "timeout" if item["timed_out"] else "failed"
                failed += not complete
                write(output / "execution" / f"{job_id}.json", {"status": status, "returncode": proc.returncode,
                      "workers": workers, "wall_seconds": time.perf_counter()-item["start"],
                      "process_tree_peak_rss_sampled_bytes": item["peak"],
                      "rss_scope": "Entire job, including graph loading, selection, evaluation and child processes"})
                del active[job_id]
                completed_count = publish(output, protocol)
                print(f"{status.upper()} {job_id}", flush=True)
            if time.perf_counter() - last_publish >= 1 or not active:
                write(output / "progress.json", {"updated_utc": now(), "completed": completed_count,
                      "total": len(protocol["jobs"]), "pending": len(pending), "active": list(active), "failed_this_run": failed})
                last_publish = time.perf_counter()
            if active:
                time.sleep(0.1)
        return 130 if stopped else 1 if failed else 0
    except KeyboardInterrupt:
        print("Stopping experiment jobs; completed results and selection checkpoints are retained.", flush=True)
        return 130
    finally:
        for job_id, item in active.items():
            terminate_tree(item["process"])
            item["log"].close()
            write(output / "execution" / f"{job_id}.json", {"status": "stopped"})
        if protocol:
            publish(output, protocol)
        lock.unlink(missing_ok=True)
