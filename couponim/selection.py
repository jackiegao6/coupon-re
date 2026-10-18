"""Method registry and a common selection interface for experiment jobs."""
import random
import time
import numpy as np
from . import exact
from .adaptive import adaptive_select
from .comparators import ic_ris_order, estimate_single_coupon_matrix, mc_greedy_order
from .conditioned import cim_ris_seeds
from .forward import sample_coverage, select_coverage
from .heuristics import pagerank_order
from .rr import CouponRR, sample_ssr
from .streaming import stream_ssr_greedy

DEFAULT_METHODS = ("CIM-RIS", "Forward-CELF", "Random", "Degree", "PageRank", "Alpha", "1Hop")
METHODS = DEFAULT_METHODS + ("CIM-RIS-Bank", "CIM-Conditioned", "CIM-Adaptive",
                             "Forward-Stochastic", "MC-Greedy", "CUM-Rank", "IC-RIS", "Exact-Greedy")
FACTOR_METHODS = {"CIM-RIS", "CIM-RIS-Bank", "CIM-Conditioned", "IC-RIS"}
NATIVE_METHODS = {"CIM-RIS", "CIM-RIS-Bank", "Forward-CELF", "Forward-Stochastic",
                  "MC-Greedy", "CUM-Rank"}
OPTIONS = {"stochastic_epsilon", "matrix_max_bytes", "exact_max_nodes",
           "adaptive_initial_samples", "adaptive_minimum_samples", "adaptive_maximum_samples",
           "adaptive_validation_trials", "adaptive_tolerance_percent"}


def validate_options(options):
    if not isinstance(options, dict) or set(options)-OPTIONS:
        raise ValueError(f"algorithm_options must use only {sorted(OPTIONS)}")
    for key, value in options.items():
        if key in {"stochastic_epsilon", "adaptive_tolerance_percent"}:
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < options.get("stochastic_epsilon", .1) < 1:
        raise ValueError("stochastic_epsilon must be in (0,1)")
    if options.get("adaptive_tolerance_percent", .5) < 0:
        raise ValueError("adaptive_tolerance_percent must be nonnegative")


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
                score[u] += transfer[u]*float(alpha[neighbors[neighbors != u]].sum())/len(neighbors)
    return np.argsort(-score, kind="stable")[:k].tolist()


def select(graph, alpha, discard, transfer, method, k, theta_factor, seed, batches,
           options, stable_seed, model=None):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    validate_options(options)
    theta = graph.n*theta_factor if method in FACTOR_METHODS else None
    info = {"selection_backend": "cpp" if model else "python"}
    if method == "CIM-RIS":
        root_seed = stable_seed(seed, "roots")
        component_seeds = [stable_seed(seed, "component", j) for j in range(k)]
        if model:
            seeds, detail = model.stream(k, theta, root_seed, component_seeds)
        else:
            seeds, detail = stream_ssr_greedy(CouponRR(graph, alpha, discard), k,
                                             theta, root_seed, component_seeds)
        info.update(detail, root_seed=root_seed, component_seeds=component_seeds)
    elif method == "CIM-RIS-Bank":
        if model:
            seeds, detail = model.bank(k, theta, seed)
        else:
            bank = sample_ssr(CouponRR(graph, alpha, discard), k, theta, seed)
            seeds, detail = bank.greedy()
            detail.update(sampling_seconds=bank.sampling_seconds, stats=bank.stats,
                          empty_ssr_count=bank.empty_ssr)
        info.update(detail)
    elif method == "CIM-Conditioned":
        seeds, _, estimate, memberships = cim_ris_seeds(graph, alpha, discard, k, theta, seed)
        info.update(training_estimate_not_quality=estimate, memberships=memberships)
    elif method == "CIM-Adaptive":
        kwargs = {key.removeprefix("adaptive_"): value for key, value in options.items()
                  if key.startswith("adaptive_")}
        seeds, detail = adaptive_select(graph, alpha, discard, k, seed, **kwargs)
        info.update(detail)
    elif method in {"Forward-CELF", "Forward-Stochastic"}:
        start = time.perf_counter()
        bank = model.forward_bank(batches, seed) if model else sample_coverage(graph, alpha, discard, batches, seed)
        info["sampling_seconds"] = time.perf_counter()-start
        start = time.perf_counter()
        seeds, count, evaluations = select_coverage(bank, k,
            mode="lazy" if method == "Forward-CELF" else "stochastic",
            epsilon=options.get("stochastic_epsilon", .1), seed=stable_seed(seed, "stochastic"))
        info.update(greedy_seconds=time.perf_counter()-start, training_estimate_not_quality=count/batches,
                    marginal_recomputations=evaluations,
                    outcome_array_bytes_not_peak_memory=sum(row.nbytes for row in bank),
                    selection_backend="cpp_sampling_python_greedy" if model else "python")
    elif method in {"MC-Greedy", "CUM-Rank"}:
        required = 4*graph.n*graph.n
        if required > options.get("matrix_max_bytes", 256*1024*1024):
            raise ValueError(f"Dense matrix needs {required} bytes, above matrix_max_bytes")
        start = time.perf_counter()
        if model:
            bank = model.forward_bank(batches, seed)
            q = np.zeros((graph.n, graph.n), dtype=np.float32)
            for u, outcomes in enumerate(bank):
                np.add.at(q[u], outcomes % graph.n, 1)
            q /= batches
        else:
            q = estimate_single_coupon_matrix(graph, alpha, discard, batches, seed)
        info.update(matrix_seconds=time.perf_counter()-start, matrix_bytes=q.nbytes,
                    trajectories_per_source=batches,
                    selection_backend="cpp_sampling_python_matrix" if model else "python")
        # Python matrix sampling uses the historical per-source stream;
        # native bank sampling is distribution-equivalent, not seedwise identical.
        info["matrix_rng_layout"] = "native_sequential" if model else "historical_per_source"
        seeds = (mc_greedy_order(q, k) if method == "MC-Greedy"
                 else np.argsort(-q.sum(axis=1), kind="stable")[:k].tolist())
        info["matrix_estimate_not_quality"] = exact.spread(q, seeds)
    elif method == "IC-RIS":
        seeds, _ = ic_ris_order(graph, transfer, k, theta, seed)
        info["objective_used_for_selection"] = "independent_cascade_not_coupon_spread"
    elif method == "Exact-Greedy":
        q = exact.consumption_matrix(graph, alpha, discard, options.get("exact_max_nodes", 16))
        seeds, gains = exact.greedy(q, k)
        info.update(exact_spread=exact.spread(q, seeds), exact_marginals=gains)
    else:
        seeds = choose_static(graph, alpha, transfer, method, seed, k)
    return seeds, info
