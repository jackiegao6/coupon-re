"""Correct the k-dependent candidate budget of the small-k stochastic baseline.

The original study reused the max-k order. That is valid for ordinary greedy,
but stochastic greedy's candidate-pool size depends on the target k. Preserve
original JSON and supersede only its Forward-Stochastic k=5 row.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import run_real_submission as core
import forward_coverage_baselines as forward
from run_uniform_ssr_study import ROOT, DEFAULT_OUTPUT, hash_file, write_json, environment, evaluate_selection

OUT=DEFAULT_OUTPUT/'stochastic-k5-correction'


def worker(job_id):
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for source,expected in manifest['source_sha256'].items():
        assert hash_file(ROOT/source)==expected
    job=next(x for x in manifest['jobs'] if x['id']==job_id)
    original_path=DEFAULT_OUTPUT/'jobs'/f'{job_id}.json'
    original=json.loads(original_path.read_text(encoding='utf-8'))
    assert original['job']==job
    started=time.perf_counter()
    graph=core.load_graph(job['dataset'])
    a,d,_=core.node_probabilities(graph,job['scenario'])
    rng_seed=core._stable_seed('uniform-quality',job_id)
    select_started=time.perf_counter()
    bank=forward.sample_coverage(graph,a,d,job['forward_batches'],rng_seed+1)
    sampled=time.perf_counter()
    seeds,count,evaluations=forward.select_coverage(bank,5,'stochastic',.1,rng_seed+2)
    info={'selection_seconds':time.perf_counter()-select_started,
        'forward_sampling_seconds':sampled-select_started,'forward_batches':job['forward_batches'],
        'forward_trajectories':graph.n*job['forward_batches'],'marginal_evaluations':evaluations,
        'training_estimate_not_quality':count/job['forward_batches'],
        'algorithm_target_k':5,'correction':'candidate-pool size computed for k=5, not max_k=25',
        'selection_seed':rng_seed+2,'forward_bank_seed':rng_seed+1}
    row=evaluate_selection(graph,a,d,seeds,job,'Forward-Stochastic',info)
    write_json(OUT/'jobs'/f'{job_id}.json',{'status':'complete','job':job,'row':row,
        'supersedes_original_file_sha256':hash_file(original_path),'wall_seconds':time.perf_counter()-started,
        'environment':environment()})
    print('COMPLETE corrected-k5 '+job_id,flush=True)


def coordinator():
    path=OUT/'manifest.json'
    if not path.exists():
        main=json.loads((DEFAULT_OUTPUT/'manifest.json').read_text(encoding='utf-8'))
        sources=['experiments/rerun_uniform_stochastic_k5.py','experiments/forward_coverage_baselines.py',
                 'experiments/run_uniform_ssr_study.py','experiments/run_real_submission.py']
        write_json(path,{'jobs':[x for x in main['jobs'] if x['suite']=='quality'],
            'source_sha256':{s:hash_file(ROOT/s) for s in sources},'environment':environment(),
            'scope':'Supersede only Forward-Stochastic k=5; identical original forward-bank seed and independent evaluation seed.',
            'reason':'Stochastic greedy candidate count depends on target k; max-k prefix reuse is not the specified k=5 algorithm.'})
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for job in manifest['jobs']:
        dest=OUT/'jobs'/f'{job["id"]}.json'
        if dest.exists():continue
        log=OUT/(job['id']+'.log')
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',job['id']],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(encoding='utf-8'));raise SystemExit(result.returncode)
        print(log.read_text(encoding='utf-8').strip(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job');a=p.parse_args()
    worker(a.job) if a.job else coordinator()
