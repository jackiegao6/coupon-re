"""Small end-to-end checks of checkpoint recovery, reuse and exact evaluation."""
from pathlib import Path
import csv
import json
import shutil
import struct
import subprocess
import sys
from run_streaming_sensitivity import ROOT, read, write, sha


def main():
    base=ROOT/'tmp/streaming-sensitivity-runner-check'
    if base.exists():raise RuntimeError('Use a fresh verification directory')
    source=base/'fixture-source';(source/'work/graphs/fixture').mkdir(parents=True)
    graph=ROOT/'z-assets-52-final/verification/streaming/all_consume-19/graph.bin'
    shutil.copy2(graph,source/'work/graphs/fixture/graph.bin')
    n,m=struct.unpack('<II',graph.read_bytes()[:8]);assert n>=2
    write(source/'work/graphs/fixture/preparation.json',dict(n=n,m=m,graph_sha256=sha(graph)))
    shutil.copy2(ROOT/'z-assets-53/work/figure3_kernel.exe',source/'work/figure3_kernel.exe')
    write(source/'protocol.json',dict(behavior_parameters=[1,0,0,0],
        input_sha256={'work/figure3_kernel.exe':sha(source/'work/figure3_kernel.exe')}))
    write(source/'results.json',dict(protocol_sha256=sha(source/'protocol.json'),raw_rows=[],summary=[]))
    runner=ROOT/'experiments/run_streaming_sensitivity.py'
    def call(command,out,*args):
        subprocess.run([sys.executable,str(runner),command,'--output',str(out),*map(str,args)],check=True)
    def initialize(src,out,factors):
        call('init',out,'--source',src,'--datasets','fixture','--budgets',1,2,'--factors',*factors,'--trials',100)
    out=base/'first';initialize(source,out,[1,2]);call('run',out)
    result=read(out/'results.json');assert result['completed_configurations']==4
    for r in result['raw_rows']:
        assert r['spread']==r['consumed_coupons']==r['k']
        assert r['origin']=='new_measured' and r['mc_variance']==0
        assert read(out/r['selection_record'])['stats']['calls']==r['theta']*r['k']
    attempts={p.name:len(read(p)['attempts']) for p in (out/'jobs').glob('*.json')}
    call('run',out)
    assert attempts=={p.name:len(read(p)['attempts']) for p in (out/'jobs').glob('*.json')}
    jobfile=next((out/'jobs').glob('*.json'));job=read(jobfile)
    lost=job['rows'].pop();job['status']='partial';write(jobfile,job)
    call('run',out)
    assert len(read(jobfile)['attempts'])==1
    recovered=next(r for r in read(jobfile)['rows'] if r['k']==lost['k'])
    assert recovered['spread']==lost['spread'] and recovered['seeds']==lost['seeds']
    reuse=base/'reuse';initialize(out,reuse,[1,2,4])
    assert read(reuse/'protocol.json')['retained_configurations']==4
    assert read(reuse/'protocol.json')['new_configurations']==2
    call('run',reuse)
    finished=read(reuse/'results.json');assert finished['completed_configurations']==6
    with (reuse/'figure-6-summary.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==6 and all(float(r['mean_spread_per_coupon'])==1 for r in rows)
    for r in finished['raw_rows']:assert r['spread']==r['k']
    proof=dict(passed=True,deterministic_all_consume=True,initial_points=4,extended_points=6,
        completed_jobs_not_rerun=True,finished_checkpoint_recovered=True,reuse_from_new_format=True,
        independent_evaluation=True,csv_ratios_verified=True)
    write(base/'verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
