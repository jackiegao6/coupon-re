"""Registered sampling-budget diagnosis at k=200; preserve main comparisons.

Reuse the existing five theta=8n runs on each graph as a fixed reference.
Run theta=32n and 64n for all graphs, five repeats and 5000 independent
forward evaluations each. Training coverage is diagnostic only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_uniform_ssr_study as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/sampling-diagnosis-20261013'
REFERENCE = ROOT / 'experiments/results/budget-extension-20261013'
DATASETS = ['Netscience', 'NetFacebookEgo', 'DoubanRandom']
SOURCES = study.SOURCE_FILES + ['experiments/run_sampling_diagnosis.py']


def specs():
    return [dict(id=f'sampling-{dataset}-k200-f{factor}-r{rep}',
                 dataset=dataset, scenario='balanced', k=200, theta_factor=factor,
                 replicate=rep, evaluation_trials=5000)
            for factor in [32, 64] for dataset in DATASETS for rep in range(5)]


def initialize():
    reference_paths = [REFERENCE / 'manifest.json'] + [
        REFERENCE / 'jobs' / f'extended-budgets-{dataset}-balanced-r{rep}.json'
        for dataset in DATASETS for rep in range(5)]
    manifest = dict(jobs=specs(),
        source_sha256={p: study.hash_file(ROOT / p) for p in SOURCES},
        dataset_sha256={name: study.hash_file(study.core.DATASETS[name]) for name in DATASETS},
        reference_sha256={p.relative_to(ROOT).as_posix(): study.hash_file(p) for p in reference_paths},
        protocol='Uniform unconditional SSR at k=200 and balanced behavior; '
        'theta/n=32,64 with five runs for every graph. Reuse all registered '
        'theta/n=8 runs and corresponding baselines from the prior budget extension. '
        'Keep selection and evaluation independent, with 5000 forward campaigns '
        'per solution. No change to main quality comparisons or algorithm. '
        'Compare training coverage with independent evaluation without treating '
        'training coverage as quality. Concurrent times are not efficiency evidence. '
        'Retain all configurations, including any with lower quality.')
    path = OUT / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest
    else:
        study.write_json(path, manifest)
        for rel in SOURCES:
            target = OUT / 'source' / Path(rel).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
    return manifest


def validate(result, job):
    assert result['job'] == job and result['status'] == 'complete'
    assert len(result['rows']) == 1
    row = result['rows'][0]
    assert row['method'] == 'Uniform-SSR' and row['k'] == 200
    assert row['theta'] == job['theta_factor'] * result['metadata']['n']
    assert len(row['seeds']) == len(set(row['seeds'])) == 200
    assert row['forward_simulations'] == 5000
    assert row['selection_seed'] != row['evaluation_seed']
    assert 0 <= row['spread'] <= row['consumed_coupons'] + 1e-9 <= 200 + 1e-9
    assert len(row['selected_marginal_counts']) == 200
    assert abs(sum(row['selected_marginal_counts']) / job['theta_factor']
               - row['training_estimate_not_quality']) < 1e-9


def run(job):
    start = time.perf_counter()
    graph, a, d, tr, meta = study.prepare(job)
    selection_seed = study.core._stable_seed('sampling-diagnosis', job['id'])
    seeds, info = study.uniform_selection(graph, a, d, job['k'],
                                         job['theta_factor'] * graph.n, selection_seed)
    info['theta_factor'] = job['theta_factor']
    row = study.evaluate_selection(graph, a, d, seeds, job, 'Uniform-SSR', info,
                                   simulations=job['evaluation_trials'])
    result = dict(job=job, status='complete', metadata=meta, rows=[row],
                  wall_seconds=time.perf_counter()-start, environment=study.environment())
    validate(result, job)
    return result


def launch(job):
    path = OUT / 'jobs' / f"{job['id']}.json"
    if path.exists():
        validate(json.loads(path.read_text(encoding='utf-8')), job)
        return job['id'] + ' reused'
    log = OUT / 'logs' / f"{job['id']}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, __file__, '--job', job['id']],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Failed {job["id"]}; see {log}')
    return job['id'] + ' complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.job:
        manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
        for rel, expected in manifest['source_sha256'].items():
            assert study.hash_file(ROOT / rel) == expected, rel
        job = next(j for j in manifest['jobs'] if j['id'] == args.job)
        assert study.hash_file(study.core.DATASETS[job['dataset']]) == manifest['dataset_sha256'][job['dataset']]
        study.write_json(OUT / 'jobs' / f"{job['id']}.json", run(job))
    else:
        manifest = initialize()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(launch, job) for job in manifest['jobs']]
            for future in as_completed(futures):
                print(future.result(), flush=True)
        print('All 30 registered jobs completed.', flush=True)
