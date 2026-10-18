"""Directed-network extension using the publisher-verified SNAP Epinions file."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

import run_real_submission as core
import forward_coverage_baselines as forward
from run_uniform_ssr_study import ROOT, DEFAULT_OUTPUT, hash_file, write_json, environment, uniform_selection, evaluate_selection

OUT = DEFAULT_OUTPUT/'directed-extension'
DATA = ROOT/'dataset/downloaded_real_networks/soc-Epinions1.txt'


def load_directed():
    provenance = json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
    assert provenance['byte_identical']
    assert hash_file(DATA) == provenance['official_uncompressed_sha256']
    edges = np.loadtxt(DATA, comments='#', dtype=np.int64)
    original_ids = np.unique(edges)
    n = len(original_ids)
    compressed = np.searchsorted(original_ids, edges)
    assert len(np.unique(compressed[:,0]*n+compressed[:,1])) == len(edges), 'Duplicate edges need explicit handling'
    order = np.lexsort((compressed[:,1], compressed[:,0]))
    compressed = compressed[order]
    degrees = np.bincount(compressed[:,0], minlength=n).astype(np.int32)
    indices = compressed[:,1].astype(np.int32)
    indptr = np.r_[0,np.cumsum(degrees)]
    incoming = [[] for _ in range(n)]
    for u,v in compressed:
        incoming[int(v)].append(int(u))
    graph = core.Graph('SNAP-Epinions',n,len(edges),indices,indptr,incoming,degrees)
    assert (n,len(edges)) == (75879,508837)
    return graph, original_ids, int(np.sum(compressed[:,0]==compressed[:,1]))


def worker(job_id):
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for path, expected in manifest['source_sha256'].items():
        assert hash_file(ROOT/path) == expected
    job = next(x for x in manifest['jobs'] if x['id']==job_id)
    started = time.perf_counter()
    graph, ids, selfloops = load_directed()
    load_seconds = time.perf_counter()-started
    a,d,tr = core.node_probabilities(graph,job['scenario'])
    rng = core._stable_seed('directed-extension',job_id)
    seeds, info = uniform_selection(graph,a,d,10,4*graph.n,rng)
    info['end_to_end_selection_seconds_including_graph_load']=time.perf_counter()-started
    info['selection_process_peak_working_set_bytes']=getattr(psutil.Process().memory_info(),'peak_wset',None)
    rows=[evaluate_selection(graph,a,d,seeds,job,'Uniform-SSR',info)]
    orders=core.static_orders(graph,a,tr,rng+1)
    for method,order in orders.items():
        rows.append(evaluate_selection(graph,a,d,order[:10],job,method))
    base_started=time.perf_counter()
    bank=forward.sample_coverage(graph,a,d,32,rng+2)
    sampling_seconds=time.perf_counter()-base_started
    for method,mode in [('Forward-CELF','lazy'),('Forward-Stochastic','stochastic')]:
        start=time.perf_counter()
        order,count,evaluations=forward.select_coverage(bank,10,mode,.1,rng+3)
        rows.append(evaluate_selection(graph,a,d,order,job,method,{
            'selection_seconds':sampling_seconds+time.perf_counter()-start,
            'forward_batches':32,'forward_trajectories':graph.n*32,
            'training_estimate_not_quality':count/32,'marginal_evaluations':evaluations}))
    del bank;gc.collect()
    write_json(OUT/'jobs'/f'{job_id}.json',{'job':job,'status':'complete','rows':rows,
        'wall_seconds':time.perf_counter()-started,'environment':environment(),
        'metadata':{'n':graph.n,'m':graph.m,'self_loops':selfloops,
            'zero_outdegree_nodes':int(np.sum(graph.degrees==0)),
            'graph_load_seconds':load_seconds,'original_id_min':int(ids.min()),
            'original_id_max':int(ids.max()),'dataset_sha256':hash_file(DATA)}})
    print('COMPLETE '+job_id,flush=True)


def coordinator():
    path=OUT/'manifest.json'
    if not path.exists():
        jobs=[{'id':f'Epinions-{scenario}-r{rep}','scenario':scenario,'replicate':rep}
              for scenario in ['balanced','forwarding-heavy'] for rep in range(3)]
        sources=['experiments/run_uniform_ssr_directed.py','experiments/uniform_ssr.py',
                 'experiments/run_uniform_ssr_study.py','experiments/run_real_submission.py',
                 'experiments/forward_coverage_baselines.py']
        write_json(path,{'jobs':jobs,'environment':environment(),
            'source_sha256':{s:hash_file(ROOT/s) for s in sources},
            'purpose':'A separately registered directed-network extension; does not replace or filter main-study results.',
            'configuration':{'k':10,'theta_factor':4,'forward_batches':32,'evaluation_trials':5000},
            'preprocessing':'Preserve all directed edges and self-loops, no symmetrization or component filtering; sorted observed node IDs mapped to 0..n-1.'})
        _, ids, _=load_directed()
        np.save(OUT/'original_node_ids.npy',ids)
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for job in manifest['jobs']:
        dest=OUT/'jobs'/f'{job["id"]}.json'
        if dest.exists(): continue
        print('START '+job['id'],flush=True)
        log=OUT/(job['id']+'.log')
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',job['id']],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(encoding='utf-8'));raise SystemExit(result.returncode)
        print(log.read_text(encoding='utf-8').strip(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job');args=p.parse_args()
    worker(args.job) if args.job else coordinator()
