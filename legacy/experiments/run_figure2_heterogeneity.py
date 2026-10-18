"""Registered Figure 2 rerun: mean-preserving ninefold behavior variance."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import run_uniform_ssr_study as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/figure2-heterogeneity-20261014'
DATASETS = ['Netscience', 'NetFacebookEgo', 'DoubanRandom']
KS = [5, 10, 25, 50, 100, 150, 200]
SOURCES = study.SOURCE_FILES+['experiments/run_figure2_heterogeneity.py']

def specs():
    return [dict(id=f'figure2-variance9-{name}-r{rep}', dataset=name,
                 scenario='balanced-variance9', replicate=rep, budgets=KS,
                 scale=3, theta_factor=8, forward_batches=256,
                 mc_trajectories=2048, evaluation_trials=5000)
            for name in DATASETS for rep in range(5)]

def parameters(job):
    g, a, _, _, meta = study.prepare({**job, 'scenario':'balanced'})
    assert not np.any(g.degrees == 0)
    mean = float(a.mean())
    changed = mean+job['scale']*(a-mean)
    d = .5-changed
    assert changed.min() >= 0 and d.min() >= 0
    assert abs(float(changed.mean())-mean) < 1e-12
    assert abs(float(changed.var())-job['scale']**2*float(a.var())) < 1e-12
    return g, changed, d, np.full(g.n, .5), dict(**meta,
        original_consumption_mean=mean, original_consumption_variance=float(a.var()),
        consumption_mean=float(changed.mean()), consumption_variance=float(changed.var()),
        consumption_min=float(changed.min()), consumption_max=float(changed.max()),
        transfer_probability=.5, association='original degree association retained',
        alpha_by_node=changed.tolist(), original_alpha_by_node=a.tolist())

def initialize():
    manifest = dict(jobs=specs(),
        source_sha256={p:study.hash_file(ROOT/p) for p in SOURCES},
        dataset_sha256={n:study.hash_file(study.core.DATASETS[n]) for n in DATASETS},
        protocol='All three Figure 2 datasets, all seven budgets, five runs, all nine '
        'existing methods (MC-Greedy on Netscience only). Mean-preserving scale=3, '
        'variance multiplier=9; a_new=mean(a)+3*(a-mean(a)); d_new=0.5-a_new, '
        't=0.5. No clipping or permutation. Same graph and parameters for all methods. '
        'CIM-RIS theta=8n; both forward methods share a 256-path/source bank; '
        'MC-Greedy 2048 paths/source. Static and lazy-greedy orders may be sliced; '
        'stochastic greedy and CIM-RIS rerun separately for each budget. '
        '5000 fresh independent forward campaigns per solution, unique evaluation streams. '
        'Retain all registered results, including negative findings. This setting was '
        'chosen after an exploratory pilot, and is not an independent confirmation '
        'of a prespecified variance hypothesis. Original balanced results are preserved. '
        'Concurrent execution times are not comparative cost measurements.')
    for name in DATASETS:
        parameters(dict(dataset=name, scale=3))
    path = OUT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest
    else:
        study.write_json(path, manifest)
        for rel in SOURCES:
            target = OUT/'source'/Path(rel).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/rel, target)
    return manifest

def verify(result, job):
    assert result['job'] == job and result['status'] == 'complete'
    assert len(result['rows']) == (63 if job['dataset'] == 'Netscience' else 56)
    for row in result['rows']:
        assert row['k'] in KS and row['forward_simulations'] == 5000
        assert len(row['seeds']) == len(set(row['seeds'])) == row['k']
        assert row['evaluation_seed'] != row['selection_seed']
        assert 0 <= row['spread'] <= row['consumed_coupons']+1e-9 <= row['k']+1e-9

def run(job):
    started = time.perf_counter()
    g, a, d, tr, meta = parameters(job)
    seed = study.core._stable_seed('figure2-variance9', job['id'])
    orders = study.core.static_orders(g, a, tr, seed+1)
    order_seeds = {method:seed+1 for method in orders}
    bank = study.forward.sample_coverage(g, a, d, 256, seed+2)
    orders['Forward-CELF'], _, _ = study.forward.select_coverage(bank, max(KS), 'lazy')
    order_seeds['Forward-CELF'] = seed+2
    if g.name == 'Netscience':
        q = study.core.estimate_single_coupon_matrix(g, a, d, 2048, seed+3)
        orders['MC-Greedy'] = study.core.mc_greedy_order(q, max(KS))
        order_seeds['MC-Greedy'] = seed+3
    rows = []
    def evaluate(method, chosen, info):
        info['selected_mean_consumption_probability'] = float(a[chosen].mean())
        row = study.evaluate_selection(g, a, d, chosen, job, method, info, simulations=5000)
        rows.append(row)
    for k in KS:
        for method, order in orders.items():
            info = dict(selection_seed=order_seeds[method])
            if method == 'Forward-CELF':
                info['forward_batches'] = 256
            if method == 'MC-Greedy':
                info['mc_trajectories'] = 2048
            evaluate(method, order[:k], info)
        chosen, _, _ = study.forward.select_coverage(bank, k, 'stochastic', .1, seed+10+k)
        evaluate('Forward-Stochastic', chosen, dict(selection_seed=seed+10+k,
            bank_seed=seed+2, forward_batches=256))
        chosen, info = study.uniform_selection(g, a, d, k, 8*g.n, seed+100+k)
        info.update(theta_factor=8, selection_seed=seed+100+k)
        evaluate('Uniform-SSR', chosen, info)
        print(f'{job["id"]}: k={k} complete', flush=True)
    result = dict(job=job, status='complete', metadata=meta, rows=rows,
        environment=study.environment(), wall_seconds=time.perf_counter()-started)
    verify(result, job)
    return result

def launch(job):
    path = OUT/'jobs'/f'{job["id"]}.json'
    if path.exists():
        verify(json.loads(path.read_text(encoding='utf-8')), job)
        return job['id']+' reused'
    log = OUT/'logs'/f'{job["id"]}.txt'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.run([sys.executable, __file__, '--job', job['id']],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'Failed {job["id"]}; see {log}')
    return job['id']+' complete'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.job:
        manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        for rel, digest in manifest['source_sha256'].items():
            assert study.hash_file(ROOT/rel) == digest, rel
        job = next(j for j in manifest['jobs'] if j['id'] == args.job)
        assert study.hash_file(study.core.DATASETS[job['dataset']]) == manifest['dataset_sha256'][job['dataset']]
        study.write_json(OUT/'jobs'/f'{job["id"]}.json', run(job))
    else:
        manifest = initialize()
        print('Registered 15 jobs / 875 evaluated solutions.', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(launch, job) for job in manifest['jobs']]):
                print(future.result(), flush=True)
        print('All Figure 2 reruns complete.', flush=True)
