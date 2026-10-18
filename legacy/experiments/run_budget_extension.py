"""Registered balanced-budget extension; keep the original selection protocol.

Every method is evaluated at k=150 and 200 on all three small graphs,
with five independent runs and 5000 fresh forward campaigns per solution.
This is a quality experiment: concurrent job times are not cost comparisons.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

import run_section6_completion as previous
import run_uniform_ssr_study as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/budget-extension-20261013'
SOURCES = previous.SOURCES + ['experiments/run_budget_extension.py']


def specs():
    return [dict(id=f'extended-budgets-{dataset}-balanced-r{rep}',
                 suite='extended-budgets', dataset=dataset, scenario='balanced',
                 replicate=rep, budgets=[150, 200], evaluation_trials=5000)
            for dataset in previous.SMALL for rep in range(5)]


def initialize():
    manifest = dict(jobs=specs(),
                    source_sha256={p: study.hash_file(ROOT / p) for p in SOURCES},
                    dataset_sha256={d: study.hash_file(previous.core.DATASETS[d])
                                    for d in previous.SMALL},
                    protocol='Same run_job as the Section 6 completion: theta=8n; '
                    'forward bank=256 trajectories/source; MC-Greedy=2048 '
                    'trajectories/source on Netscience only; stochastic greedy '
                    'rerun for each k. Uniform independent selection and evaluation '
                    'seeds. Record both distinct consumers and consumed coupons. '
                    'Retain every registered result regardless of relative quality. '
                    'Concurrent times are not comparative efficiency measurements.')
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


def verify_result(result, job):
    assert result['job'] == job and result['status'] == 'complete'
    assert len(result['rows']) == (22 if job['dataset'] == 'Netscience' else 20)
    for row in result['rows']:
        assert row['k'] in job['budgets']
        assert len(row['seeds']) == len(set(row['seeds'])) == row['k']
        assert row['forward_simulations'] == 5000
        assert row['evaluation_seed'] != row.get('selection_seed')
        assert 0 <= row['spread'] <= row['consumed_coupons'] + 1e-9 <= row['k'] + 1e-9


def launch(job):
    path = OUT / 'jobs' / f"{job['id']}.json"
    if path.exists():
        verify_result(json.loads(path.read_text(encoding='utf-8')), job)
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
        assert study.hash_file(previous.core.DATASETS[job['dataset']]) == manifest['dataset_sha256'][job['dataset']]
        result = previous.run_job(job)
        verify_result(result, job)
        study.write_json(OUT / 'jobs' / f"{job['id']}.json", result)
    else:
        manifest = initialize()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(launch, job) for job in manifest['jobs']]
            for future in as_completed(futures):
                print(future.result(), flush=True)
        print('All 15 registered jobs completed.', flush=True)
