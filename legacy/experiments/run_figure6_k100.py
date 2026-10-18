"""Paired quality/time/memory sensitivity at k=100, run sequentially.

Uses the unchanged row-major compiled SSR bank from the k=10 experiment.
Every configuration is evaluated by 5000 independent forward campaigns.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import time
import numpy as np
import psutil
import run_real_submission as core
import run_uniform_ssr_study as study
import run_figure2_six_networks as graphs
from run_figure3_k300_extension import dump_graph, dump_state
from uniform_ssr import CouponRR, sample_ssr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'z-assets-51'
EXT=OUT/'experiments'
WORK=EXT/'work'
EXE=WORK/'figure6_kernel.exe'
DATASETS=['EmailEnron','network.douban','SNAP-Epinions']
DISPLAY={'network.douban':'Douban','SNAP-Epinions':'Epinions'}
FACTORS=[1,4,16,64]
K=100
REPEATS=5
TRIALS=5000
SOURCES=list(dict.fromkeys(study.SOURCE_FILES+[
    'experiments/run_figure2_six_networks.py','experiments/run_figure3_epinions.py',
    'experiments/run_figure3_k300_extension.py','experiments/figure3_extension_kernel.cpp',
    'experiments/figure6_sensitivity_kernel.cpp','experiments/run_figure6_k100.py']))

def jobs():
    return [dict(id=f'k100-{dataset}-balanced-r{r}',dataset=dataset,scenario='balanced',
                 replicate=r,k=K,theta_factors=FACTORS) for r in range(REPEATS) for dataset in DATASETS]

def invoke(mode,graphfile,inputs,output):
    before=psutil.virtual_memory().available
    started=time.perf_counter()
    proc=subprocess.Popen([str(EXE),mode,str(graphfile),str(inputs),str(output)])
    watched=psutil.Process(proc.pid)
    parent=psutil.Process()
    peaks={'child_peak_working_set_bytes':0,'child_peak_private_bytes':0,
           'parent_peak_rss_observed_bytes':0,'peak_combined_rss_observed_bytes':0}
    minimum_available=before; samples=0; last=0
    while proc.poll() is None:
        try:
            mem=watched.memory_info();pmem=parent.memory_info()
            peaks['child_peak_working_set_bytes']=max(peaks['child_peak_working_set_bytes'],mem.rss,getattr(mem,'peak_wset',0))
            peaks['child_peak_private_bytes']=max(peaks['child_peak_private_bytes'],getattr(mem,'private',0),getattr(mem,'peak_pagefile',0))
            peaks['parent_peak_rss_observed_bytes']=max(peaks['parent_peak_rss_observed_bytes'],pmem.rss)
            peaks['peak_combined_rss_observed_bytes']=max(peaks['peak_combined_rss_observed_bytes'],pmem.rss+mem.rss)
            minimum_available=min(minimum_available,psutil.virtual_memory().available);samples+=1
            elapsed=time.perf_counter()-started
            if elapsed-last>=20:
                study.write_json(EXT/'active.json',dict(mode=mode,job=output.parent.name,output=output.name,
                    elapsed_seconds=elapsed,**peaks,minimum_system_available_bytes=minimum_available))
                print(f'  ACTIVE {output.parent.name}/{output.stem}: {elapsed:.0f}s; private={peaks["child_peak_private_bytes"]/2**20:.1f} MiB',flush=True)
                last=elapsed
        except psutil.Error:pass
        time.sleep(.025)
    if proc.returncode:raise RuntimeError(f'{mode} failed: {proc.returncode}')
    if not samples or not peaks['child_peak_private_bytes']:raise RuntimeError('Memory measurements unavailable')
    return dict(wall_seconds=time.perf_counter()-started,**peaks,memory_poll_count=samples,
                initial_system_available_bytes=before,minimum_system_available_bytes=minimum_available)

def bank(graphfile,folder,k,theta,seed,factor):
    inputs=folder/f'ssr-f{factor}.bin';output=folder/f'ssr-f{factor}.json'
    with inputs.open('wb') as f:f.write(struct.pack('<II',theta,k));dump_state(f,seed)
    cost=invoke('ssrbank',graphfile,inputs,output)
    return json.loads(output.read_text()),cost

def evaluate(graphfile,folder,seeds,seed,factor,trials=TRIALS):
    inputs=folder/f'eval-f{factor}.bin';output=folder/f'eval-f{factor}.json'
    with inputs.open('wb') as f:
        f.write(struct.pack('<III',1,trials,len(seeds)))
        np.asarray(seeds,dtype='<i4').tofile(f);dump_state(f,seed)
    cost=invoke('evaluate',graphfile,inputs,output)
    return json.loads(output.read_text())[0],cost

def validate():
    path=EXT/'kernel-verification.json'
    if path.exists():
        record=json.loads(path.read_text());assert record['kernel_sha256']==study.hash_file(EXE) and record['passed'];return
    folder=WORK/'validation';folder.mkdir(parents=True,exist_ok=True)
    graph=core.load_graph('Netscience');a,d,_=core.node_probabilities(graph,'balanced')
    graphfile=folder/'graph.bin';dump_graph(graphfile,graph,a,d)
    seed=core._stable_seed('k100-validation','selection')
    sampler=CouponRR(graph,a,d);reference=sample_ssr(sampler,K,graph.n,seed)
    chosen,details=reference.greedy()
    actual,cost=bank(graphfile,folder,K,graph.n,seed,1)
    assert actual['selected']==chosen and actual['selected_marginal_counts']==details['selected_marginal_counts']
    assert actual['empty_ssr_count']==reference.empty_ssr and actual['stats']['calls']==K*graph.n
    for name in ['nonempty','memberships','scanned_edges','draw_count']:
        assert actual['stats'][name]==reference.stats[name],name
    eval_seed=core._stable_seed('k100-validation','evaluation')
    ref=core.evaluate(graph,chosen,a,d,500,eval_seed)
    observed,_=evaluate(graphfile,folder,chosen,eval_seed,1,500)
    for value,field in zip(ref,['spread','within_run_ci95_halfwidth','evaluation_variance','consumed_coupons']):
        assert math.isclose(value,observed[field],rel_tol=0,abs_tol=1e-10),(field,value,observed[field])
    study.write_json(path,dict(passed=True,k=K,dataset='Netscience',theta=graph.n,
       exact_python_selection_and_rr_counters=True,evaluation_trials=500,evaluation_absolute_tolerance=1e-10,
       kernel_sha256=study.hash_file(EXE),memory_meter_checked=True,validation_cost=cost))
    print('Verified k=100 compiled selection, counters and independent evaluation against Python.',flush=True)

def initialize():
    WORK.mkdir(parents=True,exist_ok=True)
    old=ROOT/'z-assets-4/six-network-experiments'
    old_manifest=json.loads((old/'protocol.json').read_text())
    for name in ['experiments/figure6_sensitivity_kernel.cpp','experiments/figure3_extension_kernel.cpp']:
        assert study.hash_file(ROOT/name)==old_manifest['source_sha256'][name]
    source_exe=old/'work/figure6_kernel.exe'
    if not EXE.exists():shutil.copy2(source_exe,EXE)
    assert study.hash_file(EXE)==study.hash_file(source_exe)
    manifest=dict(data_status='measured',datasets=DATASETS,display_names=DISPLAY,k=K,theta_factors=FACTORS,
        repeats=REPEATS,evaluation_trials=TRIALS,scenario='balanced',behavior_parameters=list(core.SCENARIOS['balanced']),
        balanced_setting='Original balanced reference, unchanged from z-assets-4.',
        algorithm='Uniform-root row-major SSR bank, k independent cached-action RR components per sample, discard-on-revisit, indexed roundwise greedy. Unchanged compiled implementation.',
        timing='Sequential jobs; selection_seconds = binary preparation + selection child-process wall time. Initial graph loading and independent evaluation recorded separately.',
        memory='Per selection process: OS peak private committed bytes and peak working set, polled every 25ms. Includes graph, SSR inverted index and greedy arrays. Parent and combined sampled RSS separately recorded.',
        seeds='stable_seed(k100-sensitivity, job_id, factor); independent stable_seed(k100-evaluation, job_id, factor)',
        source_sha256={name:study.hash_file(ROOT/name) for name in SOURCES},
        dataset_sha256={ds:study.hash_file(graphs.EPINIONS if ds=='SNAP-Epinions' else core.DATASETS[ds]) for ds in DATASETS},
        kernel_sha256=study.hash_file(EXE),original_kernel_sha256=study.hash_file(source_exe),jobs=jobs())
    target=EXT/'protocol.json'
    if target.exists():assert json.loads(target.read_text())==manifest,'Source/protocol changed'
    else:
        study.write_json(target,manifest)
        for name in SOURCES:
            dest=EXT/'source'/Path(name).name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/name,dest)
    return manifest

def check_row(row):
    assert row['k']==K and row['forward_simulations']==TRIALS
    assert len(row['seeds'])==len(set(row['seeds']))==K
    assert all(0<=v<row['n'] for v in row['seeds'])
    assert row['theta']==row['theta_factor']*row['n'] and row['rr_positions']==row['theta']*K
    assert row['root_count_sum']==row['theta'] and row['selection_seed']!=row['evaluation_seed']
    assert 0<=row['spread']<=row['consumed_coupons']+1e-8<=K+1e-8
    assert row['selection_peak_private_bytes']>0 and row['selection_peak_working_set_bytes']>0
    assert row['selection_seconds']>=row['sampling_seconds']+row['greedy_seconds']

def worker(job,manifest):
    for name,digest in manifest['source_sha256'].items():assert study.hash_file(ROOT/name)==digest,name
    folder=WORK/job['id'];folder.mkdir(parents=True,exist_ok=True)
    checkpoint=EXT/'checkpoints'/f"{job['id']}.json"
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text());assert saved['protocol_sha256']==study.hash_file(EXT/'protocol.json')
        rows=saved['rows'];costs=saved['costs'];prep=saved['graph_binary_preparation_seconds'];load=saved['graph_load_and_behavior_seconds']
        graphfile=folder/'graph.bin';assert graphfile.exists()
        n=saved['n'];m=saved['m']
    else:
        started=time.perf_counter();g=graphs.load_graph(job['dataset']);a,d,_=core.node_probabilities(g,'balanced')
        load=time.perf_counter()-started;n=g.n;m=g.m
        graphfile=folder/'graph.bin';started=time.perf_counter();dump_graph(graphfile,g,a,d);prep=time.perf_counter()-started
        del g,a,d;gc.collect();rows=[];costs=[]
    print(f"START {job['id']}: n={n}, m={m}, k={K}",flush=True)
    for factor in FACTORS:
        if any(r['theta_factor']==factor for r in rows):continue
        seed=core._stable_seed('k100-sensitivity',job['id'],factor)
        selected,selection_cost=bank(graphfile,folder,K,factor*n,seed,factor)
        eval_seed=core._stable_seed('k100-evaluation',job['id'],factor)
        quality,evaluation_cost=evaluate(graphfile,folder,selected['selected'],eval_seed,factor)
        stats=selected['stats']
        row=dict(dataset=DISPLAY.get(job['dataset'],job['dataset']),scenario='balanced',replicate=job['replicate'],k=K,
          method='Uniform-SSR',theta_factor=factor,theta=factor*n,n=n,m=m,job_id=job['id'],
          selection_seed=seed,evaluation_seed=eval_seed,seeds=selected['selected'],forward_simulations=TRIALS,
          spread=quality['spread'],mc_ci95_halfwidth=quality['within_run_ci95_halfwidth'],mc_variance=quality['evaluation_variance'],
          consumed_coupons=quality['consumed_coupons'],evaluation_seconds=quality['evaluation_seconds'],
          evaluation_process_seconds=evaluation_cost['wall_seconds'],
          selection_seconds=prep+selection_cost['wall_seconds'],selection_process_seconds=selection_cost['wall_seconds'],
          sampler_preprocessing_seconds=prep,graph_load_and_behavior_seconds=load,
          graph_inclusive_selection_seconds=load+prep+selection_cost['wall_seconds'],
          sampling_seconds=selected['sampling_seconds'],greedy_seconds=selected['greedy_seconds'],
          selection_peak_private_bytes=selection_cost['child_peak_private_bytes'],
          selection_peak_working_set_bytes=selection_cost['child_peak_working_set_bytes'],
          selection_peak_combined_rss_bytes=selection_cost['peak_combined_rss_observed_bytes'],
          initial_system_available_bytes=selection_cost['initial_system_available_bytes'],
          minimum_system_available_bytes=selection_cost['minimum_system_available_bytes'],
          rr_positions=stats['calls'],nonempty_rr_count=stats['nonempty'],memberships=stats['memberships'],
          scanned_edges=stats['scanned_edges'],node_random_draws=stats['draw_count'],max_rr_size=stats['max_size'],
          empty_ssr_count=selected['empty_ssr_count'],root_count_sum=selected['root_count_sum'],
          training_estimate_not_quality=selected['training_estimate_not_quality'],selected_marginal_counts=selected['selected_marginal_counts'])
        check_row(row);rows.append(row);costs.append(dict(theta_factor=factor,selection=selection_cost,evaluation=evaluation_cost))
        saved=dict(job=job,n=n,m=m,protocol_sha256=study.hash_file(EXT/'protocol.json'),rows=rows,costs=costs,
          graph_binary_preparation_seconds=prep,graph_load_and_behavior_seconds=load)
        study.write_json(checkpoint,saved)
        print(f"DONE {job['id']} f={factor}: spread={row['spread']:.4f}, time={row['selection_seconds']:.2f}s, private={row['selection_peak_private_bytes']/2**20:.1f} MiB, rss={row['selection_peak_working_set_bytes']/2**20:.1f} MiB",flush=True)
    assert len(rows)==4
    study.write_json(EXT/'jobs'/f"{job['id']}.json",dict(**saved,status='complete',
       kernel_sha256=study.hash_file(EXE),dataset_sha256=manifest['dataset_sha256'][job['dataset']],environment=study.environment()))

def aggregate():
    raw=[];hashes={};completed=0
    for job in jobs():
        p=EXT/'jobs'/f"{job['id']}.json"
        if p.exists():
            d=json.loads(p.read_text());assert d['status']=='complete' and d['job']==job
            for r in d['rows']:check_row(r)
            raw.extend(d['rows']);hashes[p.relative_to(ROOT).as_posix()]=study.hash_file(p);completed+=1
    raw.sort(key=lambda r:([DISPLAY.get(ds,ds) for ds in DATASETS].index(r['dataset']),r['theta_factor'],r['replicate']))
    study.write_json(OUT/'results.json',dict(status='complete' if completed==len(jobs()) else 'running',data_status='measured',
        k=K,datasets=[DISPLAY.get(ds,ds) for ds in DATASETS],scenario='balanced',theta_factors=FACTORS,
        repeats=REPEATS,evaluation_trials=TRIALS,completed_jobs=completed,total_jobs=len(jobs()),raw_rows=raw,
        source_sha256=hashes,protocol_sha256=study.hash_file(EXT/'protocol.json')))
    print(f'PROGRESS {completed}/{len(jobs())} jobs, {len(raw)}/60 evaluations',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job');parser.add_argument('--validate-only',action='store_true');parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.job:
        manifest=json.loads((EXT/'protocol.json').read_text());worker(next(j for j in jobs() if j['id']==args.job),manifest);return
    manifest=initialize()
    if args.summarize_only:aggregate();return
    validate()
    if args.validate_only:return
    for job in jobs():
        dest=EXT/'jobs'/f"{job['id']}.json"
        if not dest.exists():
            log=EXT/'logs'/f"{job['id']}.txt";log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a',encoding='utf-8') as f:
                subprocess.run([sys.executable,__file__,'--job',job['id']],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        aggregate()
    print('All k=100 experiments complete.',flush=True)

if __name__=='__main__':main()
