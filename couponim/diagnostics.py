"""Plot-free graph inspection, exact diagnosis and sampler ablation."""
import random
import time
from pathlib import Path
import numpy as np
from .conditioned import conditioned_gates, reverse_coupon_set
from .exact import diagnose
from .graph import load_graph
from .parameters import node_probabilities
from .rr import CouponRR
from .runner import csv_write, read, stable, write
from .selection import choose_static


def graph_statistics(graph):
    degrees = graph.degrees
    indegree = np.bincount(graph.indices, minlength=graph.n)
    arcs = {(u, int(v)) for u in range(graph.n) for v in graph.out_neighbors(u)}
    return {"n": graph.n, "m_directed_arcs": graph.m,
            "mean_out_degree": graph.m/graph.n, "max_out_degree": int(degrees.max()),
            "mean_in_degree": graph.m/graph.n, "max_in_degree": int(indegree.max()),
            "sink_nodes": int(np.sum(degrees == 0)),
            "isolated_nodes": int(np.sum((degrees == 0) & (indegree == 0))),
            "self_loops": sum(u == v for u, v in arcs),
            "reciprocal_arc_fraction": sum((v, u) in arcs for u, v in arcs)/max(1, len(arcs))}


def estimate_fixed_assignment(graph, alpha, discard, seeds, samples, seed, conditioned=False):
    """One unbiased estimate of a FIXED indexed assignment, not a fitted seed set."""
    if (samples < 1 or not seeds or len(set(seeds)) != len(seeds)
            or any(not 0 <= s < graph.n for s in seeds)):
        raise ValueError("Invalid sample count or assignment")
    rng = random.Random(seed)
    k = len(seeds)
    weights = np.ones(graph.n)
    mask = alpha < 1
    weights[mask] = -np.expm1(k*np.log1p(-alpha[mask]))
    normalizer = float(weights.sum()) if conditioned else graph.n
    if normalizer == 0:
        return {"estimate": 0., "empty_fraction": 1., "seconds": 0., "memberships": 0}
    sampler = CouponRR(graph, alpha, discard)
    covered, empty, memberships = 0, 0, 0
    started = time.perf_counter()
    for _ in range(samples):
        root = (rng.choices(range(graph.n), weights=weights)[0]
                if conditioned else rng.randrange(graph.n))
        gates = conditioned_gates(float(alpha[root]), k, rng) if conditioned else None
        hit, nonempty = False, False
        for j, source in enumerate(seeds):
            rr = (reverse_coupon_set(graph, root, alpha, discard, rng)
                  if gates is not None and gates[j] else
                  set() if gates is not None else sampler.sample(root, rng))
            memberships += len(rr)
            hit |= source in rr
            nonempty |= bool(rr)
        covered += hit
        empty += not nonempty
    return {"estimate": normalizer*covered/samples, "empty_fraction": empty/samples,
            "seconds": time.perf_counter()-started, "memberships": memberships}


def run_diagnostic(config, output, mode, max_nodes=16):
    path, output = Path(config).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output must be new or empty")
    spec = read(path)
    records, rows = [], []
    for graph_spec in spec["graphs"]:
        graph = load_graph(graph_spec, path.parent)
        name = graph_spec.get("name", graph.name)
        if mode == "inspect":
            records.append({"dataset": name, **graph_statistics(graph)})
            continue
        for behavior in spec.get("scenarios", ["balanced-reference"]):
            scenario = behavior.get("name") if isinstance(behavior, dict) else behavior
            alpha, discard, transfer = node_probabilities(graph, behavior)
            for k in spec.get("budgets", [2]):
                if mode == "diagnose":
                    records.append({"dataset": name, "scenario": scenario,
                                    **diagnose(graph, alpha, discard, k, max_nodes)})
                    continue
                seeds = choose_static(graph, alpha, transfer, "Alpha", 0, k)
                for factor in spec.get("theta_factors", [1, 4, 16]):
                    for repeat in range(spec.get("repeats", 3)):
                        for variant in ("uniform", "conditioned"):
                            stats = estimate_fixed_assignment(graph, alpha, discard, seeds,
                                graph.n*factor, stable(spec.get("seed", 42), name, scenario, k,
                                                       factor, repeat, variant),
                                conditioned=variant == "conditioned")
                            rows.append({"dataset": name, "scenario": scenario, "k": k,
                                         "theta_factor": factor, "replicate": repeat,
                                         "sampler": variant, **stats})
                records.append({"dataset": name, "scenario": scenario, "k": k, "fixed_seeds": seeds})
    output.mkdir(parents=True, exist_ok=True)
    write(output / "diagnostic.json", {"mode": mode, "config": spec, "records": records})
    if rows:
        csv_write(output / "sampler-ablation.csv", rows, list(rows[0]))
    return records
