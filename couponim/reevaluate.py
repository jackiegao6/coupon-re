"""Re-evaluate saved selections without re-running seed selection."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
from . import native
from .graph import load_graph
from .parameters import node_probabilities
from .simulation import evaluate
from .runner import csv_write, digest, read, stable, write


def _evaluate(task):
    record, trials, backend, build_dir, output = task
    job = record["job"]
    graph = load_graph(job["graph"], ".")
    a, d, _ = node_probabilities(graph, job.get("behavior", job["scenario"]))
    seed = stable(job["evaluation_seed"], "independent-reevaluation", trials)
    begin = time.perf_counter()
    if backend == "cpp":
        model = native.NativeModel(graph, a, d, build_dir, Path(output)/"work"/job["id"])
        evaluation = model.evaluate(record["seeds"], trials, seed)
    else:
        mean, ci, variance, redeemed = evaluate(graph, record["seeds"], a, d, trials, seed)
        evaluation = {"spread": mean, "within_run_ci95_halfwidth": ci,
                      "evaluation_variance": variance, "consumed_coupons": redeemed}
    result = {"source_job": job, "seeds": record["seeds"], "trials": trials,
              "evaluation_seed": seed, "evaluation": evaluation,
              "wall_seconds": time.perf_counter()-begin}
    write(Path(output)/"jobs"/f'{job["id"]}.json', result)
    return result


def reevaluate(source, output, trials=5000, backend="python", workers=1, build_dir="build"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output in source.parents or trials < 1 or workers < 1:
        raise ValueError("Use a separate output directory and positive trials/workers")
    protocol = read(source/"protocol.json")
    # Ensure external graphs still match the measurements that selected these seeds.
    for graph in protocol["identity"]["graphs"]:
        if "path" in graph["spec"] and digest(graph["spec"]["path"]) != graph["sha256"]:
            raise ValueError("Source graph has changed; cannot re-evaluate that selection")
    paths = sorted((source/"jobs").glob("*.json"))
    records = [read(p) for p in paths]
    if not records or any(r.get("status") != "complete" for r in records):
        raise ValueError("Source must contain completed run records")
    for record in records:
        if record["job"] not in protocol["jobs"]:
            raise ValueError("Source record differs from its protocol")
    if backend == "cpp":
        native.build(build_dir)
    from . import __version__
    package = Path(__file__).parent
    identity = {"source": str(source), "source_protocol_sha256": digest(source/"protocol.json"),
                "source_jobs": {p.name: digest(p) for p in paths},
                "trials": trials, "backend": backend, "version": __version__,
                "sources": {str(p.relative_to(package)): digest(p) for p in package.rglob("*")
                            if p.suffix in {".py", ".cpp", ".hpp"}}}
    target = output/"protocol.json"
    if target.exists() and read(target) != identity:
        raise ValueError("Re-evaluation protocol changed; choose a new output")
    write(target, identity)
    tasks = [(r, trials, backend, str(Path(build_dir).resolve()), str(output)) for r in records
             if not (output/"jobs"/f'{r["job"]["id"]}.json').exists()]
    if workers == 1:
        for task in tasks:
            _evaluate(task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_evaluate, tasks))
    rows = []
    for record in records:
        job = record["job"]
        result = read(output/"jobs"/f'{job["id"]}.json')
        rows.append({**{k: job[k] for k in ("id", "dataset", "scenario", "k", "method", "theta_factor", "replicate")},
                     "evaluation_trials": trials, **{k: result["evaluation"][k] for k in
                     ("spread", "consumed_coupons", "within_run_ci95_halfwidth", "evaluation_variance")}})
    csv_write(output/"reevaluated.csv", rows, list(rows[0]))
    return len(rows)
