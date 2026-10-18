"""Add measured k=500 observations, retaining old k=100 data.

Select a storage implementation explicitly at initialization. Jobs are serial,
checkpointed individually, and independently evaluated by 5000 campaigns.
"""
from pathlib import Path
import argparse,gc,json,shutil,struct,subprocess,sys,time
import numpy as np
import psutil
import run_figure6_k100 as prior
import run_uniform_ssr_study as study
from run_figure3_k300_extension import dump_graph,dump_state

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'z-assets-52';EXT=OUT/'experiments';WORK=EXT/'work'
DATASETS=prior.DATASETS;DISPLAY=prior.DISPLAY;FACTORS=prior.FACTORS
NEW_KS=[500];TRIALS=5000
MAX_PRIVATE_BYTES=10*2**30
SOURCE='experiments/run_figure6_k_sweep.py'


def jobs():
    return [dict(id=f"k{k}-{ds}-balanced-f{factor}-r0",dataset=ds,
                 scenario='balanced',k=k,theta_factor=factor,replicate=0)
            for ds in DATASETS for factor in FACTORS for k in NEW_KS]


def setup(implementation):
    WORK.mkdir(parents=True,exist_ok=True)
    executable=(ROOT/'z-assets-3/extension-k300/work/figure3_kernel.exe'
                if implementation=='stream' else prior.EXE)
    sources=list(dict.fromkeys(prior.SOURCES+[SOURCE]))
    existing=json.loads((ROOT/'z-assets-51/results.json').read_text())
    assert existing['status']=='complete' and existing['repeats']==3
    protocol=dict(data_status='measured',implementation=implementation,
        original_k100_results_sha256=study.hash_file(ROOT/'z-assets-51/results.json'),
        datasets=DATASETS,budgets=[100]+NEW_KS,new_budgets=NEW_KS,
        theta_factors=FACTORS,new_repeats=1,old_repeats=3,evaluation_trials=TRIALS,
        behavior_parameters=list(prior.core.SCENARIOS['balanced']),
        root_and_component_seeds='stable_seed(k-sweep-root/job; k-sweep-component/job/j)',
        evaluation_seeds='stable_seed(k-sweep-evaluation/job)',
        execution='Serial fresh selection processes; no concurrent experiment jobs.',
        timing='Binary preparation plus whole selection-process wall time; graph loading and independent evaluation excluded.',
        memory='Selection-process peak private committed bytes and working set via Windows process peak counters.',
        private_memory_limit_bytes=MAX_PRIVATE_BYTES,
        preflight='For retained bank implementation, skip jobs whose k-scaled observed k=100 maximum exceeds the 10 GiB private-memory safety limit. Predictions are not measurements.',
        source_sha256={name:study.hash_file(ROOT/name) for name in sources},
        dataset_sha256={ds:study.hash_file(prior.graphs.EPINIONS if ds=='SNAP-Epinions' else prior.core.DATASETS[ds]) for ds in DATASETS},
        kernel_sha256=study.hash_file(executable),jobs=jobs())
    path=EXT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol or sources changed'
    else:
        study.write_json(path,protocol)
        for name in sources:
            dest=EXT/'source'/Path(name).name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/name,dest)
        shutil.copy2(executable,WORK/'kernel.exe')
    assert study.hash_file(WORK/'kernel.exe')==protocol['kernel_sha256']
    return protocol


def prepare_graph(ds):
    directory=WORK/'graphs'/ds;record=directory/'preparation.json'
    if record.exists():
        d=json.loads(record.read_text());assert study.hash_file(directory/'graph.bin')==d['graph_sha256'];return d
    directory.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();g=prior.graphs.load_graph(ds)
    a,d,_=prior.core.node_probabilities(g,'balanced');load=time.perf_counter()-started
    started=time.perf_counter();dump_graph(directory/'graph.bin',g,a,d);prep=time.perf_counter()-started
    result=dict(n=g.n,m=g.m,graph_load_and_behavior_seconds=load,
                graph_binary_preparation_seconds=prep,graph_sha256=study.hash_file(directory/'graph.bin'))
    study.write_json(record,result)
    return result


class ResourceLimit(RuntimeError):pass


def invoke(mode,graphfile,inputs,output):
    before=psutil.virtual_memory().available;started=time.perf_counter()
    proc=subprocess.Popen([str(WORK/'kernel.exe'),mode,str(graphfile),str(inputs),str(output)])
    watched=psutil.Process(proc.pid);parent=psutil.Process()
    peaks=dict(child_peak_working_set_bytes=0,child_peak_private_bytes=0,
               parent_peak_rss_observed_bytes=0,peak_combined_rss_observed_bytes=0)
    minimum=before;samples=0;last=0
    while proc.poll() is None:
        try:
            mem=watched.memory_info();pmem=parent.memory_info()
            peaks['child_peak_working_set_bytes']=max(peaks['child_peak_working_set_bytes'],mem.rss,getattr(mem,'peak_wset',0))
            peaks['child_peak_private_bytes']=max(peaks['child_peak_private_bytes'],getattr(mem,'private',0),getattr(mem,'peak_pagefile',0))
            peaks['parent_peak_rss_observed_bytes']=max(peaks['parent_peak_rss_observed_bytes'],pmem.rss)
            peaks['peak_combined_rss_observed_bytes']=max(peaks['peak_combined_rss_observed_bytes'],pmem.rss+mem.rss)
            minimum=min(minimum,psutil.virtual_memory().available);samples+=1
            elapsed=time.perf_counter()-started
            if peaks['child_peak_private_bytes']>MAX_PRIVATE_BYTES:
                proc.terminate();proc.wait(timeout=10)
                study.write_json(output.with_suffix('.resource-limit.json'),dict(elapsed_seconds=elapsed,**peaks))
                raise ResourceLimit('Selection process exceeded 10 GiB private-memory safety limit')
            if elapsed-last>=20:
                study.write_json(EXT/'active.json',dict(mode=mode,job=output.parent.name,elapsed_seconds=elapsed,**peaks))
                print(f"ACTIVE {output.parent.name}: {elapsed:.0f}s, private={peaks['child_peak_private_bytes']/2**20:.1f} MiB",flush=True)
                last=elapsed
        except psutil.NoSuchProcess:pass
        time.sleep(.025)
    if proc.returncode:raise RuntimeError(f'{mode} failed with code {proc.returncode}')
    assert samples and peaks['child_peak_private_bytes']>0
    return dict(wall_seconds=time.perf_counter()-started,**peaks,memory_poll_count=samples,
                initial_system_available_bytes=before,minimum_system_available_bytes=minimum)


def worker(job,protocol):
    for name,digest in protocol['source_sha256'].items():assert study.hash_file(ROOT/name)==digest,name
    ds=job['dataset']
    if protocol['implementation']=='bank':
        original=json.loads((ROOT/'z-assets-51/results.json').read_text())['raw_rows']
        matches=[r for r in original if r['dataset']==DISPLAY.get(ds,ds) and r['theta_factor']==job['theta_factor']]
        estimate=max(r['selection_peak_private_bytes'] for r in matches)*job['k']/100
        if estimate>MAX_PRIVATE_BYTES:
            study.write_json(EXT/'jobs'/(job['id']+'.json'),dict(status='not_run_resource_limit',job=job,
                predicted_private_bytes=estimate,limit_bytes=MAX_PRIVATE_BYTES,
                reason='k-scaled observed k=100 peak exceeds private-memory safety budget; no measurement made',
                protocol_sha256=study.hash_file(EXT/'protocol.json')))
            print(f"SKIP {job['id']}: predicted {estimate/2**30:.2f} GiB exceeds 10 GiB budget",flush=True)
            return
    info=prepare_graph(ds);gc.collect()
    graphfile=WORK/'graphs'/ds/'graph.bin';folder=WORK/job['id'];folder.mkdir(exist_ok=True)
    k=job['k'];theta=job['theta_factor']*info['n']
    rootseed=prior.core._stable_seed('k-sweep-root',job['id'])
    component_seeds=[prior.core._stable_seed('k-sweep-component',job['id'],j) for j in range(k)]
    eval_seed=prior.core._stable_seed('k-sweep-evaluation',job['id'])
    inputfile=folder/'selection.bin';outputfile=folder/'selection.json'
    with inputfile.open('wb') as f:
        if protocol['implementation']=='stream':
            f.write(struct.pack('<III',theta,0,k));dump_state(f,rootseed)
            for seed in component_seeds:dump_state(f,seed)
            mode='ssr'
        else:
            f.write(struct.pack('<II',theta,k));dump_state(f,rootseed);mode='ssrbank'
    print(f"START {job['id']}: theta={theta}, k={k}, storage={protocol['implementation']}",flush=True)
    try:costs=invoke(mode,graphfile,inputfile,outputfile)
    except ResourceLimit as exc:
        study.write_json(EXT/'jobs'/(job['id']+'.json'),dict(status='stopped_resource_limit',job=job,
            reason=str(exc),protocol_sha256=study.hash_file(EXT/'protocol.json')))
        print(f"STOP {job['id']}: {exc}",flush=True);return
    selected=json.loads(outputfile.read_text());chosen=selected['selected']
    assert len(chosen)==len(set(chosen))==k and all(0<=v<info['n'] for v in chosen)
    assert selected['stats']['calls']==theta*k
    evalinput=folder/'evaluation.bin';evaloutput=folder/'evaluation.json'
    with evalinput.open('wb') as f:
        f.write(struct.pack('<III',1,TRIALS,k));np.asarray(chosen,dtype='<i4').tofile(f);dump_state(f,eval_seed)
    evaluation_cost=invoke('evaluate',graphfile,evalinput,evaloutput)
    quality=json.loads(evaloutput.read_text())[0]
    assert 0<=quality['spread']<=quality['consumed_coupons']+1e-8<=k+1e-8
    stats=selected['stats'];prep=info['graph_binary_preparation_seconds'];load=info['graph_load_and_behavior_seconds']
    row=dict(dataset=DISPLAY.get(ds,ds),scenario='balanced',replicate=0,k=k,method='Uniform-SSR',
        implementation=protocol['implementation'],theta_factor=job['theta_factor'],theta=theta,
        n=info['n'],m=info['m'],job_id=job['id'],selection_seed=rootseed,component_seeds=component_seeds,
        evaluation_seed=eval_seed,seeds=chosen,forward_simulations=TRIALS,spread=quality['spread'],
        mc_ci95_halfwidth=quality['within_run_ci95_halfwidth'],mc_variance=quality['evaluation_variance'],
        consumed_coupons=quality['consumed_coupons'],evaluation_seconds=quality['evaluation_seconds'],
        evaluation_process_seconds=evaluation_cost['wall_seconds'],
        selection_seconds=prep+costs['wall_seconds'],selection_process_seconds=costs['wall_seconds'],
        sampler_preprocessing_seconds=prep,graph_load_and_behavior_seconds=load,
        graph_inclusive_selection_seconds=load+prep+costs['wall_seconds'],
        selection_peak_private_bytes=costs['child_peak_private_bytes'],
        selection_peak_working_set_bytes=costs['child_peak_working_set_bytes'],
        selection_peak_combined_rss_bytes=costs['peak_combined_rss_observed_bytes'],
        initial_system_available_bytes=costs['initial_system_available_bytes'],
        minimum_system_available_bytes=costs['minimum_system_available_bytes'],
        rr_positions=stats['calls'],nonempty_rr_count=stats['nonempty'],memberships=stats['memberships'],
        scanned_edges=stats['scanned_edges'],node_random_draws=stats['draw_count'],max_rr_size=stats['max_size'],
        root_count_sum=theta,selected_marginal_counts=selected['selected_marginal_counts'])
    record=dict(status='complete',job=job,row=row,costs=dict(selection=costs,evaluation=evaluation_cost),
        protocol_sha256=study.hash_file(EXT/'protocol.json'),kernel_sha256=protocol['kernel_sha256'],
        dataset_sha256=protocol['dataset_sha256'][ds],environment=study.environment())
    study.write_json(EXT/'jobs'/(job['id']+'.json'),record)
    print(f"DONE {job['id']}: spread={row['spread']:.4f}, time={row['selection_seconds']:.2f}s, private={row['selection_peak_private_bytes']/2**20:.1f} MiB",flush=True)


def aggregate(protocol):
    path=ROOT/'z-assets-51/results.json';assert study.hash_file(path)==protocol['original_k100_results_sha256']
    original=json.loads(path.read_text());rows=[dict(r,implementation='bank',source_result='z-assets-51/results.json') for r in original['raw_rows']]
    hashes={'z-assets-51/results.json':study.hash_file(path)};count=0;unavailable=[]
    for job in jobs():
        p=EXT/'jobs'/(job['id']+'.json')
        if p.exists():
            d=json.loads(p.read_text());assert d['job']==job
            if d['status']=='complete':rows.append(dict(d['row'],source_result=p.relative_to(ROOT).as_posix()))
            else:
                assert d['status'] in ['not_run_resource_limit','stopped_resource_limit']
                unavailable.append(d)
            hashes[p.relative_to(ROOT).as_posix()]=study.hash_file(p);count+=1
    names=[DISPLAY.get(ds,ds) for ds in DATASETS]
    rows.sort(key=lambda r:(names.index(r['dataset']),r['k'],r['theta_factor'],r['replicate']))
    status='running' if count<len(jobs()) else 'complete_with_resource_limits' if unavailable else 'complete'
    study.write_json(OUT/'results.json',dict(status=status,data_status='measured',
        datasets=names,budgets=[100]+NEW_KS,theta_factors=FACTORS,new_repeats=1,retained_k100_repeats=3,
        processed_new_jobs=count,measured_new_jobs=len(rows)-36,total_new_jobs=len(jobs()),raw_rows=rows,
        unavailable=unavailable,source_sha256=hashes,
        protocol_sha256=study.hash_file(EXT/'protocol.json')))
    print(f'PROGRESS {count}/{len(jobs())} new configurations processed, {len(rows)-36} measured.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--implementation',choices=['bank','stream'])
    p.add_argument('--job');p.add_argument('--summarize-only',action='store_true');args=p.parse_args()
    if args.job:
        protocol=json.loads((EXT/'protocol.json').read_text());worker(next(j for j in jobs() if j['id']==args.job),protocol);return
    assert args.implementation,'Select storage implementation explicitly'
    protocol=setup(args.implementation)
    if args.summarize_only:aggregate(protocol);return
    for job in jobs():
        if not (EXT/'jobs'/(job['id']+'.json')).exists():
            log=EXT/'logs'/(job['id']+'.txt');log.parent.mkdir(exist_ok=True)
            with log.open('a',encoding='utf-8') as f:
                subprocess.run([sys.executable,__file__,'--job',job['id']],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        aggregate(protocol)
    study.write_json(EXT/'active.json',dict(status='finished',processed_new_jobs=len(jobs())))


if __name__=='__main__':main()
