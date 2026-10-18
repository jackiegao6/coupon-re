"""Small registered mean-preserving heterogeneity pilot; no historical edits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np

import run_uniform_ssr_study as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/heterogeneity-pilot-20261014'
DATASETS = ['Netscience','NetFacebookEgo']
SOURCES = study.SOURCE_FILES+['experiments/run_heterogeneity_pilot.py']


def specs():
    return [dict(id=f'{name}-l{scale}-{association}-r{rep}',dataset=name,
                 scenario='balanced',scale=scale,association=association,
                 replicate=rep,k=25,theta_factor=8,forward_batches=256,evaluation_trials=5000)
            for name in DATASETS for scale in [1,3]
            for association in ['degree','permuted'] for rep in range(3)]


def parameters(job):
    g,a,_,_,meta=study.prepare(job)
    assert not np.any(g.degrees==0)
    mean=float(a.mean())
    base=a.copy()
    permutation=np.arange(g.n)
    permutation_seed=study.core._stable_seed('heterogeneity-permutation',g.name)
    if job['association']=='permuted':
        random.Random(permutation_seed).shuffle(permutation)
        base=base[permutation]
    modified=mean+job['scale']*(base-mean)
    d=.5-modified
    assert np.min(modified)>=0 and np.max(modified)<=.5
    assert abs(float(modified.mean())-mean)<1e-12
    assert abs(float(modified.var())-job['scale']**2*float(a.var()))<1e-12
    assert np.allclose(1-modified-d,.5,atol=1e-12)
    return g,modified,d,dict(**meta,original_mean=mean,original_variance=float(a.var()),
        consumption_mean=float(modified.mean()),consumption_variance=float(modified.var()),
        consumption_min=float(modified.min()),consumption_max=float(modified.max()),
        fixed_transfer_probability=.5,permutation_seed=permutation_seed,
        alpha_by_node=modified.tolist(),permutation=permutation.tolist())


def initialize():
    manifest=dict(jobs=specs(),source_sha256={p:study.hash_file(ROOT/p) for p in SOURCES},
        dataset_sha256={n:study.hash_file(study.core.DATASETS[n]) for n in DATASETS},
        protocol='Pilot only: two graphs, k=25, scales 1/3 (variance multipliers 1/9), '
        'degree-associated and one fixed node permutation per graph, three selection repeats. '
        'a_new=mean(a)+scale*(a-mean(a)); d_new=0.5-a_new. No clipping. '
        'Use the same fixed permutation at both scales and all repeats. '
        'Keep Random seed sets identical across parameter settings within each graph/replicate. '
        'Use matched selection RNG seeds for each nonrandom method within graph/replicate '
        '(not independent settings); evaluate each solution with a fresh independent stream. '
        'CIM-RIS theta=8n; Forward-CELF 256 paths/source is an empirical reference, not an upper bound. '
        '5000 forward campaigns per solution. Report all 24 jobs/72 solutions, including unfavorable '
        'or nonmonotone results. Three repeats and one permutation are exploratory, not a significance claim. '
        'No equal-cost claim; concurrent execution times are not benchmarks.')
    for name in DATASETS:
        for scale in [1,3]:
            for association in ['degree','permuted']:
                parameters(dict(dataset=name,scenario='balanced',scale=scale,association=association))
    path=OUT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==manifest
    else:
        study.write_json(path,manifest)
        for rel in SOURCES:
            dest=OUT/'source'/Path(rel).name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/rel,dest)
    return manifest


def run(job):
    g,a,d,meta=parameters(job)
    seeds={}
    details={}
    for method in ['CIM-RIS','Random','Forward-CELF']:
        seed=study.core._stable_seed('heterogeneity-selection',g.name,job['replicate'],method)
        if method=='CIM-RIS':
            chosen,info=study.uniform_selection(g,a,d,job['k'],job['theta_factor']*g.n,seed)
        elif method=='Random':
            order=list(range(g.n))
            random.Random(seed).shuffle(order)
            chosen,info=order[:job['k']],{}
        else:
            chosen,info=study.forward.forward_coverage_seeds(g,a,d,job['k'],256,seed)
        seeds[method]=chosen
        details[method]={**info,'selection_seed':seed,'selected_mean_consumption_probability':float(a[chosen].mean())}
    rows=[]
    for method,chosen in seeds.items():
        row=study.evaluate_selection(g,a,d,chosen,job,method,details[method],simulations=5000)
        assert row['evaluation_seed']!=row['selection_seed']
        assert len(chosen)==len(set(chosen))==25
        assert 0<=row['spread']<=row['consumed_coupons']+1e-10<=25+1e-10
        row.update(scale=job['scale'],association=job['association'])
        rows.append(row)
    return dict(job=job,status='complete',metadata=meta,rows=rows,environment=study.environment())


def launch(job):
    path=OUT/'jobs'/f'{job["id"]}.json'
    if path.exists():
        result=json.loads(path.read_text(encoding='utf-8'))
        assert result['job']==job and result['status']=='complete' and len(result['rows'])==3
        return job['id']+' reused'
    log=OUT/'logs'/f'{job["id"]}.txt'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as stream:
        r=subprocess.run([sys.executable,__file__,'--job',job['id']],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f'Failed {job["id"]}, see {log}')
    return job['id']+' complete'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.job:
        manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        for rel,digest in manifest['source_sha256'].items():
            assert study.hash_file(ROOT/rel)==digest,rel
        job=next(j for j in manifest['jobs'] if j['id']==args.job)
        assert study.hash_file(study.core.DATASETS[job['dataset']])==manifest['dataset_sha256'][job['dataset']]
        study.write_json(OUT/'jobs'/f'{job["id"]}.json',run(job))
    else:
        manifest=initialize()
        print('Registered 24 small jobs / 72 evaluated seed sets.',flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(launch,j) for j in manifest['jobs']]
            for future in as_completed(futures):
                print(future.result(),flush=True)
        print('Pilot complete.',flush=True)
