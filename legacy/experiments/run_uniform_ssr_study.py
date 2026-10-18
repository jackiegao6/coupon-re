"""Resumable evidence suite for the uniform-root Section 5 algorithm.

Each job runs in a fresh process, writes raw JSON atomically, and records
independent forward evaluation seeds. No historical results are overwritten.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import psutil

import run_real_submission as core
import forward_coverage_baselines as forward
from uniform_ssr import CouponRR, sample_ssr, sufficient_budget
from verify_uniform_ssr import fixtures, exact_matrix, spread

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'experiments/results/uniform-ssr-20261012'
SOURCE_FILES = ['experiments/uniform_ssr.py', 'experiments/run_uniform_ssr_study.py',
                'experiments/verify_uniform_ssr.py', 'experiments/run_real_submission.py',
                'experiments/forward_coverage_baselines.py']


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def environment():
    mem = psutil.Process().memory_info()
    return {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
            'cpu': platform.processor(), 'logical_cpus': psutil.cpu_count(),
            'physical_memory_bytes': psutil.virtual_memory().total,
            'rss_bytes': mem.rss, 'peak_working_set_bytes': getattr(mem, 'peak_wset', None),
            'timestamp_utc': datetime.now(timezone.utc).isoformat()}


def evaluate_selection(g, a, d, seeds, job, method, selection_info=None, simulations=5000):
    eval_seed = core._stable_seed('uniform-eval', job['id'], method, len(seeds))
    started = time.perf_counter()
    mean, ci, variance, redemptions = core.evaluate(g, seeds, a, d, simulations, eval_seed)
    return {'dataset': g.name, 'scenario': job.get('scenario', 'balanced'),
            'replicate': job.get('replicate', 0), 'k': len(seeds), 'method': method,
            'seeds': list(map(int, seeds)), 'forward_simulations': simulations,
            'evaluation_seed': eval_seed, 'spread': mean, 'mc_ci95_halfwidth': ci,
            'mc_variance': variance, 'consumed_coupons': redemptions,
            'evaluation_seconds': time.perf_counter()-started,
            **(selection_info or {})}


def prepare(job):
    started = time.perf_counter()
    g = core.load_graph(job['dataset'])
    a, d, tr = core.node_probabilities(g, job.get('scenario', 'balanced'))
    info = {'n': g.n, 'm': g.m, 'graph_load_and_parameters_seconds': time.perf_counter()-started,
            'dataset_sha256': hash_file(core.DATASETS[g.name]),
            'probability_model': 'degree-dependent synthetic probabilities from SCENARIOS',
            'scenario_parameters': core.SCENARIOS[job.get('scenario', 'balanced')]}
    return g, a, d, tr, info


def uniform_selection(g, a, d, k, theta, seed):
    started = time.perf_counter()
    sampler = CouponRR(g, a, d)
    prep = time.perf_counter()-started
    bank = sample_ssr(sampler, k, theta, seed)
    seeds, info = bank.greedy()
    info.update({'selection_seconds': time.perf_counter()-started,
                 'sampler_preprocessing_seconds': prep,
                 'sampling_seconds': bank.sampling_seconds, 'theta': theta,
                 'rr_positions': theta*k, 'empty_ssr_count': bank.empty_ssr,
                 'nonempty_rr_count': bank.stats['nonempty'],
                 'memberships': bank.stats['memberships'],
                 'scanned_edges': bank.stats['scanned_edges'],
                 'node_random_draws': bank.stats['draw_count'],
                 'max_rr_size': bank.stats['sampler_max_size_so_far'],
                 'root_count_sum': int(bank.root_counts.sum()),
                 'selection_seed': seed})
    return seeds, info


def quality_job(job):
    g, a, d, tr, metadata = prepare(job)
    rng_seed = core._stable_seed('uniform-quality', job['id'])
    ks = job['budgets']
    maximum = max(ks)
    started = time.perf_counter()
    orders = core.static_orders(g, a, tr, rng_seed)
    metadata['all_static_orders_seconds'] = time.perf_counter()-started
    selection_info = {method: {} for method in orders}
    # Both generic optimization baselines use exactly the same forward bank.
    started = time.perf_counter()
    bank = forward.sample_coverage(g, a, d, job['forward_batches'], rng_seed+1)
    sample_seconds = time.perf_counter()-started
    for method, mode in [('Forward-CELF', 'lazy'), ('Forward-Stochastic', 'stochastic')]:
        started = time.perf_counter()
        order, count, evaluations = forward.select_coverage(bank, maximum, mode, .1, rng_seed+2)
        orders[method] = order
        selection_info[method] = {'selection_seconds_for_max_k': sample_seconds+time.perf_counter()-started,
            'forward_sampling_seconds': sample_seconds, 'forward_batches': job['forward_batches'],
            'forward_trajectories': job['forward_batches']*g.n, 'marginal_evaluations': evaluations,
            'training_estimate_for_max_k_not_quality': count/job['forward_batches']}
    del bank
    gc.collect()
    if g.name == 'Netscience':
        started = time.perf_counter()
        q = core.estimate_single_coupon_matrix(g, a, d, 2048, rng_seed+3)
        orders['MC-Greedy'] = core.mc_greedy_order(q, maximum)
        selection_info['MC-Greedy'] = {'selection_seconds_for_max_k': time.perf_counter()-started,
                                     'trajectories_per_source': 2048,
                                     'not_exact_optimum': True}
        del q
    rows = []
    for k in ks:
        for method, order in orders.items():
            rows.append(evaluate_selection(g, a, d, order[:k], job, method, selection_info[method]))
        theta = job['theta_factor'] * g.n
        seeds, info = uniform_selection(g, a, d, k, theta, rng_seed+100+k)
        info['theta_factor'] = job['theta_factor']
        rows.append(evaluate_selection(g, a, d, seeds, job, 'Uniform-SSR', info))
    return {'metadata': metadata, 'rows': rows}


def sensitivity_job(job):
    g, a, d, tr, metadata = prepare(job)
    rows = []
    for factor in job['theta_factors']:
        selection_seed = core._stable_seed('uniform-sensitivity', job['id'], factor)
        seeds, info = uniform_selection(g, a, d, job['k'], factor*g.n, selection_seed)
        info['theta_factor'] = factor
        # A separate evaluation stream per selected solution, independent of selection.
        eval_job = {**job, 'id': job['id']+f'-factor{factor}'}
        rows.append(evaluate_selection(g, a, d, seeds, eval_job, 'Uniform-SSR', info))
    return {'metadata': metadata, 'rows': rows}


def scale_job(job):
    end_to_end = time.perf_counter()
    g, a, d, tr, metadata = prepare(job)
    seed = core._stable_seed('uniform-scale', job['id'])
    seeds, info = uniform_selection(g, a, d, job['k'], job['theta_factor']*g.n, seed)
    info['theta_factor'] = job['theta_factor']
    info['end_to_end_selection_seconds_including_graph_load'] = time.perf_counter()-end_to_end
    # Windows exposes the process-lifetime peak working set. Each scale job has
    # its own process; capture before evaluation to isolate selection's peak.
    mem = psutil.Process().memory_info()
    info['selection_process_peak_working_set_bytes'] = getattr(mem, 'peak_wset', None)
    info['selection_process_rss_bytes'] = mem.rss
    return {'metadata': metadata, 'rows': [evaluate_selection(g, a, d, seeds, job, 'Uniform-SSR', info)]}


def calibration_job(job):
    fixture = fixtures()[job['fixture']]
    g, a, d, weights = fixture
    sampler = CouponRR(g, a, d, weights)
    q = exact_matrix(sampler)
    assignment = [0, 1]
    truth = spread(q, assignment)
    optimum = max(spread(q, s) for s in itertools.combinations(range(g.n), 2))
    rows = []
    for theta in job['sample_counts']:
        for replicate in range(job['replicates']):
            bank = sample_ssr(sampler, 2, theta, core._stable_seed(job['id'], theta, replicate))
            chosen, info = bank.greedy()
            rows.append({'graph': g.name, 'theta': theta, 'replicate': replicate,
                'fixed_assignment': assignment, 'fixed_exact_spread': truth,
                'fixed_estimate': bank.estimate(assignment),
                'selected_seeds': chosen, 'selected_exact_spread': spread(q, chosen),
                'selected_training_estimate': info['training_estimate_not_quality'],
                'exact_optimum': optimum, 'mean_rr_size': bank.stats['memberships']/(2*theta),
                'mean_edge_scans': bank.stats['scanned_edges']/(2*theta),
                'empty_ssr_fraction': bank.empty_ssr/theta})
    return {'rows': rows, 'metadata': {'n': g.n, 'm': g.m, 'exact_matrix': q.tolist(),
        'expected_rr_size': float(q.sum()/g.n),
        'expected_edge_scans': float(np.dot(q.sum(axis=1), np.diff(sampler.inptr))/g.n),
        'expected_empty_ssr_fraction': float(np.mean((1-a)**2))}}


def theory_job(job):
    g, a, d, weights = fixtures()[job['fixture']]
    sampler = CouponRR(g, a, d, weights)
    q = exact_matrix(sampler)
    k, epsilon, delta = 2, .2, .05
    L = float(np.sort(a)[-k:].sum())
    theta = sufficient_budget(g.n, k, L, epsilon, delta)
    tau = epsilon*L/(2*k)
    optimum = max(spread(q, s) for s in itertools.combinations(range(g.n), k))
    rows = []
    for replicate in range(job['replicates']):
        bank = sample_ssr(sampler, k, theta, core._stable_seed(job['id'], replicate))
        largest_error = 0.
        for length in range(1, k+1):
            for assignment in itertools.permutations(range(g.n), length):
                empirical = bank.estimate(assignment)-bank.estimate(assignment[:-1])
                actual = spread(q, assignment)-spread(q, assignment[:-1])
                largest_error = max(largest_error, abs(empirical-actual))
        chosen, info = bank.greedy()
        actual = spread(q, chosen)
        rows.append({'graph': g.name, 'replicate': replicate, 'theta': theta,
            'epsilon': epsilon, 'delta': delta, 'lower_bound': L, 'tau': tau,
            'max_marginal_error_over_all_prefixes': largest_error,
            'uniform_accuracy_event': largest_error <= tau,
            'selected_exact_spread': actual, 'exact_optimum': optimum,
            'approximation_ratio': actual/optimum,
            'required_ratio': 1-1/math.e-epsilon,
            'guarantee_satisfied': actual >= (1-1/math.e-epsilon)*optimum})
    return {'rows': rows, 'metadata': {'n': g.n, 'm': g.m,
        'note': 'Finite empirical check with exact optimum; not a replacement for the theorem.'}}


def make_jobs():
    jobs = []
    for dataset in ['Netscience', 'NetFacebookEgo', 'DoubanRandom']:
        for scenario in core.SCENARIOS:
            for replicate in range(5):
                jobs.append({'suite': 'quality', 'id': f'q-{dataset}-{scenario}-r{replicate}',
                    'dataset': dataset, 'scenario': scenario, 'replicate': replicate,
                    'budgets': [5, 25], 'theta_factor': 8, 'forward_batches': 256})
    for dataset in ['Netscience', 'DoubanRandom']:
        for scenario in ['balanced', 'forwarding-heavy']:
            for replicate in range(5):
                jobs.append({'suite': 'sensitivity', 'id': f's-{dataset}-{scenario}-r{replicate}',
                    'dataset': dataset, 'scenario': scenario, 'replicate': replicate,
                    'k': 10, 'theta_factors': [1, 4, 16, 64]})
    for dataset in core.DATASETS:
        for factor in [1, 4]:
            for replicate in range(3):
                jobs.append({'suite': 'scalability', 'id': f'c-{dataset}-f{factor}-r{replicate}',
                    'dataset': dataset, 'scenario': 'balanced', 'replicate': replicate,
                    'k': 10, 'theta_factor': factor})
    for fixture in [0, 3]:
        jobs.append({'suite': 'calibration', 'id': f'calibration-{fixture}', 'fixture': fixture,
                     'sample_counts': [100, 400, 1600, 6400], 'replicates': 100})
    for fixture in [0, 2, 3]:
        jobs.append({'suite': 'theory', 'id': f'theory-{fixture}', 'fixture': fixture, 'replicates': 30})
    return jobs


def run_worker(args):
    manifest = json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    job = next(j for j in manifest['jobs'] if j['id'] == args.job)
    for rel, sha in manifest['source_sha256'].items():
        if hash_file(ROOT/rel) != sha:
            raise RuntimeError(f'Source changed after registration: {rel}')
    started = time.perf_counter()
    result = {'job': job, 'environment_start': environment(), 'status': 'running'}
    result.update({'quality': quality_job, 'sensitivity': sensitivity_job,
                   'scalability': scale_job, 'calibration': calibration_job,
                   'theory': theory_job}[job['suite']](job))
    result.update(status='complete', wall_seconds=time.perf_counter()-started, environment_end=environment())
    write_json(args.output/'jobs'/f'{job["id"]}.json', result)
    print(f'COMPLETE {job["id"]} {result["wall_seconds"]:.2f}s', flush=True)


def run_coordinator(args):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'manifest.json'
    if not path.exists():
        write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(),
            'protocol': 'uniform-ssr-v1', 'jobs': make_jobs(), 'environment': environment(),
            'source_sha256': {p: hash_file(ROOT/p) for p in SOURCE_FILES},
            'dataset_sha256': {name: hash_file(path) for name, path in core.DATASETS.items()},
            'notes': ['Five independent selections per quality configuration; 5000 independent forward trials each.',
                'All empirical budgets are declared, not claimed to satisfy the theoretical sufficient bound.',
                'Forward-CELF and Forward-Stochastic share 256 trajectories per source; compute budgets are not equal.',
                'MC-Greedy is a Monte Carlo reference on Netscience, not exact OPT.',
                'Scalability uses separate processes; peak working set includes interpreter and graph.',
                'Local graph file hashes identify data; original upstream provenance remains incomplete.']})
    manifest = json.loads(path.read_text(encoding='utf-8'))
    suite_set = set(args.suites.split(',')) if args.suites else None
    jobs = [j for j in manifest['jobs'] if suite_set is None or j['suite'] in suite_set]
    completed = failed = 0
    for j in jobs:
        dest = args.output/'jobs'/f'{j["id"]}.json'
        if dest.exists() and json.loads(dest.read_text(encoding='utf-8'))['status'] == 'complete':
            completed += 1
            continue
        if args.limit and completed + failed >= args.limit:
            break
        print('START '+j['id'], flush=True)
        log = args.output/'logs'/f'{j["id"]}.txt'
        log.parent.mkdir(exist_ok=True)
        with log.open('w', encoding='utf-8') as handle:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output',
                str(args.output.resolve()), '--job', j['id']], stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            failed += 1
            print('FAILED '+j['id']+' (see '+str(log)+')', flush=True)
            break  # Investigate before running more jobs with a potentially invalid implementation.
        completed += 1
        print(log.read_text(encoding='utf-8').strip(), flush=True)
    write_json(args.output/'progress.json', {'completed_in_requested_suites': completed,
        'failed': failed, 'total_registered_jobs': len(manifest['jobs']),
        'actual_completed_jobs': sum(1 for p in (args.output/'jobs').glob('*.json')),
        'timestamp_utc': datetime.now(timezone.utc).isoformat()})
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--suites', default='')
    parser.add_argument('--job')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run_worker(args) if args.job else run_coordinator(args)
