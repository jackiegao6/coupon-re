"""Measured two-panel Epinions study, preserving the original Figure 3 protocol.

One independent k_max=50 training bank per scenario/repeat supplies valid
selection prefixes for k=5,25,50. Evaluation uses independent forward campaigns.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

import forward_coverage_baselines as forward
import run_real_submission as core
import run_figure2_six_networks as graphs
import run_uniform_ssr_study as study
from uniform_ssr import CouponRR, sample_ssr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'z-assets-3'
SCENARIOS = ['adoption-heavy', 'forwarding-heavy']
KS = [5, 25, 50]
REPEATS = 5
THETA_FACTOR = 8
FORWARD_BATCHES = 256
EVALUATIONS = 5000
METHODS = ['Uniform-SSR', 'Forward-CELF', '1Hop-Sort', 'Alpha-Sort',
           'PageRank', 'DegreeTopM', 'Random']
SOURCES = list(dict.fromkeys(study.SOURCE_FILES + [
    'experiments/run_figure2_six_networks.py',
    'experiments/run_figure3_epinions.py']))


def job_specs():
    return [dict(id=f'figure3-Epinions-{scenario}-r{repeat}',
                 dataset='SNAP-Epinions', scenario=scenario, replicate=repeat)
            for repeat in range(REPEATS) for scenario in SCENARIOS]


def initialize():
    manifest = dict(
        data_status='measured', dataset='SNAP-Epinions', n=75879, m=508837,
        dataset_sha256=study.hash_file(graphs.EPINIONS),
        budgets=KS, repeats=REPEATS, theta_factor=THETA_FACTOR,
        forward_batches=FORWARD_BATCHES, evaluation_trials=EVALUATIONS,
        methods=METHODS, scenarios={s:list(core.SCENARIOS[s]) for s in SCENARIOS},
        behavior='Original Figure 3 degree-based parameters; sink transfer set to zero; discard on first revisit; coupons independent.',
        preprocessing='Directed SNAP Epinions arcs and self-loops preserved; observed node IDs compressed in sorted order; no symmetrization or component filtering.',
        selection='Within each scenario/repeat, prefixes of a k_max=50 bank are used for k=5,25,50. Each prefix has the intended marginal sampling distribution. Runs independent; budgets within a run correlated.',
        evaluation='5000 fresh forward campaigns per method and budget; evaluation RNG seeds independent of all training seeds; all registered runs retained.',
        intervals='Two-sided 95% Student-t confidence intervals over five run means; t critical at df=4 is 2.7764451051977987.',
        timing='Training-block wall times include sampling and greedy selection for k_max=50. Jobs may overlap; these are diagnostic costs, not isolated speed comparisons. Peak process memory is process-lifetime, not per-method incremental memory.',
        source_sha256={p:study.hash_file(ROOT/p) for p in SOURCES}, jobs=job_specs())
    path = OUT/'figure-3-protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest, 'Protocol or code changed'
    else:
        study.write_json(path, manifest)
        for source in SOURCES:
            target=OUT/'source'/Path(source).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/source, target)
    return manifest


def memory():
    info=psutil.Process().memory_info()
    return dict(rss_bytes=info.rss, process_lifetime_peak_bytes=getattr(info,'peak_wset',None))


def verify(result, job):
    assert result['status']=='complete' and result['job']==job
    assert (result['n'],result['m'])==(75879,508837)
    rows=result['rows']
    assert len(rows)==len(KS)*len(METHODS)
    assert {(r['k'],r['method']) for r in rows}=={(k,m) for k in KS for m in METHODS}
    training=set(result['training_seeds'].values())
    eval_seeds=set()
    for row in rows:
        assert row['dataset_id']=='SNAP-Epinions'
        assert row['scenario']==job['scenario'] and row['replicate']==job['replicate']
        assert len(set(row['seeds']))==len(row['seeds'])==row['k']
        assert all(0 <= s < result['n'] for s in row['seeds'])
        assert row['evaluation_trials']==EVALUATIONS
        assert 0 <= row['spread'] <= row['consumed_coupons']+1e-8 <= row['k']+1e-8
        assert row['evaluation_seed'] not in training
        eval_seeds.add(row['evaluation_seed'])
    assert len(eval_seeds)==len(rows)
    assert result['ssr_stats']['calls']==THETA_FACTOR*result['n']*max(KS)


def worker(job, manifest):
    for p,digest in manifest['source_sha256'].items():
        assert study.hash_file(ROOT/p)==digest, p
    assert study.hash_file(graphs.EPINIONS)==manifest['dataset_sha256']
    start=time.perf_counter()
    graph=graphs.load_graph('SNAP-Epinions')
    alpha,discard,transfer=core.node_probabilities(graph,job['scenario'])
    seed=core._stable_seed('figure3-epinions-measured-v1',job['id'])
    training=dict(static=seed,forward=seed+1,ssr=seed+2)
    stage=time.perf_counter()
    orders=core.static_orders(graph,alpha,transfer,training['static'])
    costs=dict(static_all_methods_seconds=time.perf_counter()-stage)
    print(f"{job['id']}: graph and static orders ready",flush=True)

    stage=time.perf_counter()
    bank=forward.sample_coverage(graph,alpha,discard,FORWARD_BATCHES,training['forward'])
    costs['forward_sampling_seconds']=time.perf_counter()-stage
    costs['forward_outcome_array_bytes']=sum(a.nbytes for a in bank)
    costs['forward_after_sampling_memory']=memory()
    greedy_start=time.perf_counter()
    orders['Forward-CELF'],count,evaluations=forward.select_coverage(bank,max(KS),'lazy')
    costs['forward_greedy_seconds']=time.perf_counter()-greedy_start
    costs['forward_selection_seconds']=time.perf_counter()-stage
    costs['forward_marginal_evaluations']=evaluations
    del bank
    gc.collect()
    print(f"{job['id']}: Forward-CELF ready ({costs['forward_selection_seconds']:.1f}s)",flush=True)

    stage=time.perf_counter()
    sampler=CouponRR(graph,alpha,discard)
    costs['ssr_preprocessing_seconds']=time.perf_counter()-stage
    bank=sample_ssr(sampler,max(KS),THETA_FACTOR*graph.n,training['ssr'])
    costs['ssr_sampling_seconds']=bank.sampling_seconds
    costs['ssr_after_sampling_memory']=memory()
    print(f"{job['id']}: SSR samples ready ({bank.sampling_seconds:.1f}s)",flush=True)
    orders['Uniform-SSR'],greedy_info=bank.greedy()
    costs['ssr_greedy_seconds']=greedy_info['greedy_seconds']
    costs['ssr_selection_seconds']=time.perf_counter()-stage
    ssr_stats=dict(bank.stats)
    del bank,sampler
    gc.collect()
    print(f"{job['id']}: CIM-RIS order ready ({costs['ssr_selection_seconds']:.1f}s)",flush=True)

    rows=[]
    for k in KS:
        for method in METHODS:
            chosen=orders[method][:k]
            evaluation_seed=core._stable_seed('figure3-epinions-forward-evaluation-v1',job['id'],method,k)
            stage=time.perf_counter()
            spread,ci,variance,redemptions=core.evaluate(graph,chosen,alpha,discard,EVALUATIONS,evaluation_seed)
            rows.append(dict(dataset='Epinions',dataset_id='SNAP-Epinions',
                scenario=job['scenario'],replicate=job['replicate'],k=k,method=method,
                spread=spread,consumed_coupons=redemptions,within_run_ci95_halfwidth=ci,
                evaluation_variance=variance,evaluation_trials=EVALUATIONS,
                evaluation_seed=evaluation_seed,
                selection_seed=training['ssr' if method=='Uniform-SSR' else 'forward' if method=='Forward-CELF' else 'static'],
                evaluation_seconds=time.perf_counter()-stage,seeds=list(map(int,chosen))))
        print(f"{job['id']}: k={k} evaluated",flush=True)
    result=dict(status='complete',job=job,n=graph.n,m=graph.m,
        protocol_sha256=study.hash_file(OUT/'figure-3-protocol.json'),
        dataset_sha256=manifest['dataset_sha256'],training_seeds=training,
        behavior_summary={key:dict(min=float(values.min()),mean=float(values.mean()),max=float(values.max()))
                          for key,values in [('consumption',alpha),('dropping',discard),('transfer',transfer)]},
        rows=rows,costs=costs,ssr_stats=ssr_stats,environment=study.environment(),
        wall_seconds=time.perf_counter()-start)
    verify(result,job)
    study.write_json(OUT/'jobs'/f"{job['id']}.json",result)
    print(f"COMPLETE {job['id']} ({result['wall_seconds']:.1f}s)",flush=True)


def launch(job):
    target=OUT/'jobs'/f"{job['id']}.json"
    if target.exists():
        verify(json.loads(target.read_text(encoding='utf-8')),job)
        return 'REUSED '+job['id']
    log=OUT/'logs'/f"{job['id']}.txt"
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.run([sys.executable,__file__,'--job',job['id']],cwd=ROOT,
                               stdout=stream,stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'Job failed; see {log}')
    verify(json.loads(target.read_text(encoding='utf-8')),job)
    return 'COMPLETE '+job['id']


def aggregate(manifest):
    raw=[]
    costs=[]
    hashes={}
    protocol_hash=study.hash_file(OUT/'figure-3-protocol.json')
    for job in manifest['jobs']:
        path=OUT/'jobs'/f"{job['id']}.json"
        result=json.loads(path.read_text(encoding='utf-8'))
        verify(result,job)
        assert result['protocol_sha256']==protocol_hash
        assert result['dataset_sha256']==manifest['dataset_sha256']
        hashes[path.relative_to(OUT).as_posix()]=study.hash_file(path)
        raw.extend({**row,'job_id':job['id']} for row in result['rows'])
        costs.append(dict(job_id=job['id'],scenario=job['scenario'],replicate=job['replicate'],
                          k_max=max(KS),**result['costs'],wall_seconds=result['wall_seconds'],
                          process_lifetime_peak_bytes=result['environment']['peak_working_set_bytes']))
    assert len(raw)==210 and len({row['evaluation_seed'] for row in raw})==210
    summary=[]
    for scenario in SCENARIOS:
        for k in KS:
            for method in METHODS:
                group=[r for r in raw if (r['scenario'],r['k'],r['method'])==(scenario,k,method)]
                assert sorted(r['replicate'] for r in group)==list(range(REPEATS))
                mean=statistics.mean(r['spread'] for r in group)
                ci=2.7764451051977987*statistics.stdev(r['spread'] for r in group)/math.sqrt(REPEATS)
                summary.append(dict(dataset='Epinions',dataset_id='SNAP-Epinions',
                    scenario=scenario,k=k,method=method,runs=REPEATS,
                    mean_spread=mean,spread_ci95_halfwidth=ci,spread_per_coupon=mean/k,
                    mean_consumed_coupons=statistics.mean(r['consumed_coupons'] for r in group),
                    theta_factor=THETA_FACTOR,forward_batches=FORWARD_BATCHES,evaluation_trials=EVALUATIONS))
    payload=dict(data_status='measured',protocol_sha256=protocol_hash,source_job_sha256=hashes,
                 raw_rows=raw,summary_rows=summary,cost_records=costs)
    study.write_json(OUT/'figure-3-results.json',payload)
    print(f'Aggregated {len(raw)} per-run evaluations into {len(summary)} plotted means.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.job:
        manifest=json.loads((OUT/'figure-3-protocol.json').read_text(encoding='utf-8'))
        worker(next(j for j in manifest['jobs'] if j['id']==args.job),manifest)
        return
    manifest=initialize()
    if not args.summarize_only:
        print('Registered 10 runs, 210 evaluated solutions, theta=8n, k=5/25/50.',flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(launch,j) for j in manifest['jobs']]):
                print(future.result(),flush=True)
    aggregate(manifest)


if __name__=='__main__':
    main()
