"""Isolated real-graph pilot; outputs are not manuscript comparison evidence.

RR samples and forward batches are distinct work units. Timings include each
method's sampling and selection, exclude shared graph/probability loading and
final evaluation, and are not matched-resource comparisons.
"""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

import run_real_submission as core
from forward_coverage_baselines import forward_coverage_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=core.DATASETS, default='Netscience')
    parser.add_argument('--scenario', choices=core.SCENARIOS, default='balanced')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--batches', type=int, default=100)
    parser.add_argument('--rr-samples', type=int, default=2000)
    parser.add_argument('--eval-simulations', type=int, default=2000)
    parser.add_argument('--selection-seed', type=int, default=20260913)
    parser.add_argument('--output', type=Path,
                        default=core.ROOT / 'experiments/results/ccfa-baseline-pilot/netscience-balanced-k10.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; existing pilot records are immutable')
    if min(args.k, args.batches, args.rr_samples, args.eval_simulations) <= 0:
        parser.error('budgets and sample counts must be positive')
    loaded = time.perf_counter()
    graph = core.load_graph(args.dataset)
    alpha, discard, transfer = core.node_probabilities(graph, args.scenario)
    if args.k > graph.n:
        parser.error('k exceeds distinct-node capacity')
    graph_seconds = time.perf_counter() - loaded
    evaluation_seed = core._stable_seed(args.selection_seed, args.dataset, args.scenario, args.k, 'pilot-holdout')
    rng = random.Random(evaluation_seed)
    evaluation_streams = [rng.getrandbits(64) for _ in range(args.eval_simulations)]
    rows = []
    for method in ['CIM-RIS', 'IC-RIS', 'Forward-Lazy', 'Forward-Stochastic', 'Random']:
        # Forward methods share their frozen training bank for a controlled pilot.
        stream_name = 'forward-bank' if method.startswith('Forward-') else method
        stream = core._stable_seed(args.selection_seed, args.dataset, args.scenario, args.k, stream_name)
        assert stream != evaluation_seed
        if method.startswith('Forward-'):
            seeds, metrics = forward_coverage_seeds(
                graph, alpha, discard, args.k, args.batches, stream,
                mode='lazy' if method == 'Forward-Lazy' else 'stochastic')
        elif method == 'CIM-RIS':
            seeds, elapsed, estimated, memberships = core.cim_ris_seeds(
                graph, alpha, discard, args.k, args.rr_samples, stream)
            metrics = {'selection_seconds': elapsed, 'training_estimate_not_quality': estimated,
                       'rr_memberships': memberships}
        elif method == 'IC-RIS':
            seeds, elapsed = core.ic_ris_order(graph, transfer, args.k, args.rr_samples, stream)
            metrics = {'selection_seconds': elapsed}
        else:
            started = time.perf_counter()
            seeds = random.Random(stream).sample(range(graph.n), args.k)
            metrics = {'selection_seconds': time.perf_counter() - started}
        mean, ci, variance, redemptions = core.evaluate_with_streams(
            graph, seeds, alpha, discard, evaluation_streams)
        assert len(seeds) == len(set(seeds)) == args.k
        assert 0 <= mean <= redemptions + 1e-10 <= args.k + 1e-10
        rows.append({'method': method, 'seeds': seeds, 'selection_stream': stream,
                     'mean_adopters': mean, 'mean_redemptions': redemptions,
                     'within_run_normal_ci95': ci, 'forward_variance': variance, **metrics})
    sources = [Path(__file__).resolve(), core.ROOT / 'experiments/run_real_submission.py',
               core.ROOT / 'experiments/forward_coverage_baselines.py', core.DATASETS[args.dataset]]
    result = {
        'status': 'REAL_PILOT_NOT_PUBLICATION_EVIDENCE',
        'protocol_version': 'ccfa-forward-baseline-pilot-v1',
        'config': {key: value for key, value in vars(args).items() if key != 'output'},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'platform': platform.platform()},
        'graph_load_and_probabilities_seconds': graph_seconds,
        'evaluation_seed': evaluation_seed,
        'fingerprints': {str(path.relative_to(core.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sources},
        'limitations': ['One selection run; no cross-run interval or significance claim.',
                        'Different sample units; not a matched-time or matched-quality comparison.',
                        'Array bytes exclude Python objects and are not peak memory.'],
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print('Saved independent pilot:', args.output)
    for row in rows:
        print(row['method'], 'adopters=', row['mean_adopters'], 'seconds=', row['selection_seconds'])


if __name__ == '__main__':
    main()
