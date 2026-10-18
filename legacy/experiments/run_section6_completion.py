"""Locked supplementary protocol for the current Section 6; preserves old results.

Models: IC-RIS and estimated CUM ranking on the original 45 configurations.
Budgets: balanced k=10,50,100, with five independent selection runs.
Shuffle: paired permutation of non-sink behavior parameters (three runs).
Tradeoff: sequential quality/cost measurements on two graphs (three runs).
Every selected solution receives 5000 fresh coupon campaign evaluations.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import run_real_submission as core
import forward_coverage_baselines as forward
import run_uniform_ssr_study as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/section6-completion-20261012'
SMALL = ['Netscience', 'NetFacebookEgo', 'DoubanRandom']
SOURCES = study.SOURCE_FILES + ['experiments/run_section6_completion.py']


def specs():
    jobs = []
    for dataset in SMALL:
        for scenario in core.SCENARIOS:
            for rep in range(5):
                jobs.append(dict(suite='models', dataset=dataset, scenario=scenario,
                                 replicate=rep, budgets=[5, 25]))
        for rep in range(5):
            jobs.append(dict(suite='budgets', dataset=dataset, scenario='balanced',
                             replicate=rep, budgets=[10, 50, 100]))
    for dataset in ['Netscience', 'DoubanRandom']:
        for rep in range(3):
            jobs.append(dict(suite='shuffle', dataset=dataset, scenario='balanced',
                             replicate=rep, budgets=[25]))
            jobs.append(dict(suite='tradeoff', dataset=dataset, scenario='balanced',
                             replicate=rep, budgets=[10]))
    for job in jobs:
        job['id'] = f"{job['suite']}-{job['dataset']}-{job['scenario']}-r{job['replicate']}"
        job['evaluation_trials'] = 5000
    return jobs


def evaluate(g, a, d, seeds, job, method, info=None):
    row = study.evaluate_selection(g, a, d, seeds, job, method, info,
                                   simulations=job['evaluation_trials'])
    assert len(seeds) == len(set(seeds)) == row['k']
    assert 0 <= row['spread'] <= row['consumed_coupons'] + 1e-9 <= row['k'] + 1e-9
    return row


def run_job(job):
    started = time.perf_counter()
    g, a, d, tr, meta = study.prepare(job)
    seed = core._stable_seed('section6-completion', job['id'])
    if job['suite'] == 'shuffle':
        ids = np.flatnonzero(g.degrees > 0)
        perm = np.random.default_rng(seed).permutation(ids)
        a[ids], d[ids] = a[perm].copy(), d[perm].copy()
        tr = np.maximum(0., 1-a-d)
        meta['behavior_permutation_seed'] = seed
        meta['behavior_rule'] = 'paired shuffle among non-sinks; sinks unchanged'
    rows = []
    if job['suite'] == 'tradeoff':
        settings = [('Uniform-SSR', f) for f in [1, 4, 16, 64]]
        settings += [('Forward', b) for b in [64, 256, 1024]]
        random.Random(seed).shuffle(settings)
        meta['execution_order'] = settings
        for method, budget in settings:
            gc.collect()
            selection_seed = core._stable_seed(seed, method, budget)
            eval_job = {**job, 'id': job['id'] + f'-{method}-{budget}'}
            if method == 'Uniform-SSR':
                seeds, info = study.uniform_selection(g, a, d, 10, budget*g.n, selection_seed)
                info['theta_factor'] = budget
                rows.append(evaluate(g, a, d, seeds, eval_job, method, info))
            else:
                t = time.perf_counter()
                bank = forward.sample_coverage(g, a, d, budget, selection_seed)
                bank_seconds = time.perf_counter()-t
                for label, mode in [('Forward-CELF', 'lazy'), ('Forward-Stochastic', 'stochastic')]:
                    t = time.perf_counter()
                    seeds, _, _ = forward.select_coverage(bank, 10, mode, .1, selection_seed+1)
                    info = {'selection_seconds': bank_seconds+time.perf_counter()-t,
                            'forward_batches': budget, 'sampling_seconds': bank_seconds}
                    rows.append(evaluate(g, a, d, seeds, eval_job, label, info))
                del bank
    else:
        maximum = max(job['budgets'])
        t = time.perf_counter()
        bank = forward.sample_coverage(g, a, d, 256, seed+1)
        bank_seconds = time.perf_counter()-t
        counts = np.array([len(x) for x in bank])
        cum_order = sorted(range(g.n), key=lambda v: (-int(counts[v]), v))
        ic_order, ic_seconds = core.ic_ris_order(g, tr, maximum, 8*g.n, seed+2)
        orders = {'CUM-Rank': cum_order, 'IC-RIS': ic_order}
        if job['suite'] != 'models':
            orders.update(core.static_orders(g, a, tr, seed+3))
            lazy_order, _, _ = forward.select_coverage(bank, maximum, 'lazy')
            orders['Forward-CELF'] = lazy_order
            if g.name == 'Netscience':
                q = core.estimate_single_coupon_matrix(g, a, d, 2048, seed+4)
                orders['MC-Greedy'] = core.mc_greedy_order(q, maximum)
                del q
        for k in job['budgets']:
            for method, order in orders.items():
                info = {'forward_batches': 256} if method in ['CUM-Rank', 'Forward-CELF'] else {}
                if method == 'CUM-Rank':
                    info['estimated_total_consumption_not_quality'] = float(counts[order[:k]].sum()/256)
                if method == 'IC-RIS':
                    info.update(ic_rr_samples=8*g.n, ic_selection_seconds_for_max_k=ic_seconds)
                rows.append(evaluate(g, a, d, order[:k], job, method, info))
            if job['suite'] != 'models':
                order, _, _ = forward.select_coverage(bank, k, 'stochastic', .1, seed+10+k)
                rows.append(evaluate(g, a, d, order, job, 'Forward-Stochastic', {'forward_batches': 256}))
                order, info = study.uniform_selection(g, a, d, k, 8*g.n, seed+100+k)
                info['theta_factor'] = 8
                rows.append(evaluate(g, a, d, order, job, 'Uniform-SSR', info))
        meta['forward_bank_seconds'] = bank_seconds
    return dict(job=job, status='complete', metadata=meta, rows=rows,
                wall_seconds=time.perf_counter()-started, environment=study.environment())


def initialize():
    manifest = {'jobs': specs(), 'source_sha256': {p: study.hash_file(ROOT/p) for p in SOURCES},
                'dataset_sha256': {d: study.hash_file(core.DATASETS[d]) for d in SMALL},
                'protocol': 'Uniform unconditional coupon SRR; cached node-coupon actions; '
                '5000 independent forward evaluations. IC-RIS uses p(u,v) as IC edge probability. '
                'CUM-Rank orders per-source consumption frequency from 256 trajectories. '
                'Only tradeoff jobs are used for comparative timing and run sequentially. '
                'All registered configurations are retained, including unfavorable results.'}
    path = OUT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest, 'Protocol/source changed'
    else:
        study.write_json(path, manifest)
        for rel in SOURCES:
            dest = OUT/'source'/Path(rel).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/rel, dest)
    return manifest


def launch(job):
    dest = OUT/'jobs'/f"{job['id']}.json"
    if dest.exists():
        old = json.loads(dest.read_text(encoding='utf-8'))
        assert old['status'] == 'complete' and old['job'] == job
        return job['id'] + ' reused'
    log = OUT/'logs'/f"{job['id']}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as f:
        done = subprocess.run([sys.executable, __file__, '--job', job['id']], cwd=ROOT,
                              stdout=f, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(f"Failed {job['id']}; see {log}")
    return job['id'] + ' complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--suite', choices=['models', 'budgets', 'shuffle', 'tradeoff', 'all'], default='all')
    args = parser.parse_args()
    if args.job:
        manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        for p, expected in manifest['source_sha256'].items():
            assert study.hash_file(ROOT/p) == expected, p
        job = next(j for j in manifest['jobs'] if j['id'] == args.job)
        study.write_json(OUT/'jobs'/f"{args.job}.json", run_job(job))
    else:
        manifest = initialize()
        jobs = [j for j in manifest['jobs'] if args.suite == 'all' or j['suite'] == args.suite]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for message in pool.map(launch, [j for j in jobs if j['suite'] != 'tradeoff']):
                print(message, flush=True)
        for job in jobs:
            if job['suite'] == 'tradeoff':
                print(launch(job), flush=True)
        print('All requested jobs completed.', flush=True)
