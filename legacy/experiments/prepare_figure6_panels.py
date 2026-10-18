"""Verify the first two Figure 6 panels against archived sensitivity jobs.

Creates a JSON intermediate for the CSV exporter. Does not rerun experiments
or use any modified/illustrative result folders.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/results/uniform-ssr-20261012'
OUT=ROOT/'z-assets-4'
DATASETS=['Netscience','DoubanRandom']
SCENARIOS=['balanced','forwarding-heavy']
FACTORS=[1,4,16,64]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    raw_path=SOURCE/'sensitivity_raw.csv'
    summary_path=SOURCE/'sensitivity_summary.csv'
    archived=json.loads((SOURCE/'verification.json').read_text(encoding='utf-8'))
    registered=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    jobs={j['id']:j for j in registered['jobs'] if j['suite']=='sensitivity'}
    with raw_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);raw_fields=reader.fieldnames;raw_csv=list(reader)
    with summary_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);summary_fields=reader.fieldnames;summary_csv=list(reader)
    assert len(raw_csv)==80 and len(summary_csv)==16
    original={}
    hashes={}
    job_parameters={}
    for jobid in sorted({r['job_id'] for r in raw_csv}):
        path=SOURCE/'jobs'/f'{jobid}.json'
        assert sha(path)==archived['result_files_sha256'][str(path.relative_to(SOURCE))],jobid
        job=json.loads(path.read_text(encoding='utf-8'))
        assert job['status']=='complete' and job['job']==jobs[jobid]
        assert len(job['rows'])==4
        hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        job_parameters[jobid]=job['metadata']['scenario_parameters']
        for row in job['rows']:
            original[(jobid,row['theta_factor'])]={**row,'job_id':jobid,
                'n':job['metadata']['n'],'m':job['metadata']['m']}
    raw=[]
    for row in raw_csv:
        source=original[(row['job_id'],int(row['theta_factor']))]
        for field in raw_fields:
            value=source[field]
            if isinstance(value,list):assert json.loads(row[field])==value,(field,row)
            elif isinstance(value,int):assert int(row[field])==value,(field,row)
            elif isinstance(value,float):assert float(row[field])==value,(field,row)
            else:assert row[field]==value,(field,row)
        assert source['dataset'] in DATASETS and source['scenario'] in SCENARIOS
        assert source['k']==10 and source['method']=='Uniform-SSR'
        assert source['forward_simulations']==5000
        assert source['evaluation_seed']!=source['selection_seed']
        assert 0<=source['spread']<=source['consumed_coupons']+1e-9<=10+1e-9
        raw.append(source)
    expected={(d,s,f) for d in DATASETS for s in SCENARIOS for f in FACTORS}
    assert {(r['dataset'],r['scenario'],int(r['theta_factor'])) for r in summary_csv}==expected
    summary=[];ordered_raw=[]
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            for factor in FACTORS:
                group=sorted((r for r in raw if (r['dataset'],r['scenario'],r['theta_factor'])==(dataset,scenario,factor)),key=lambda r:r['replicate'])
                assert [r['replicate'] for r in group]==list(range(5))
                row=next(r for r in summary_csv if (r['dataset'],r['scenario'],int(r['theta_factor']))==(dataset,scenario,factor))
                avg=statistics.mean(r['spread'] for r in group)
                ci=float(t.ppf(.975,4))*statistics.stdev(r['spread'] for r in group)/math.sqrt(5)
                computed=dict(mean_spread=avg,ci95_across_runs_halfwidth=ci,
                    mean_consumed=statistics.mean(r['consumed_coupons'] for r in group),
                    mean_spread_per_coupon=avg/10,
                    mean_training_estimate=statistics.mean(r['training_estimate_not_quality'] for r in group),
                    mean_selection_seconds=statistics.mean(r['selection_seconds'] for r in group))
                for field,value in computed.items():
                    assert math.isclose(float(row[field]),value,rel_tol=1e-12,abs_tol=1e-12),(field,row)
                # Preserve the archived summary values; formulas are independently
                # recalculated by the exporter before writing CSV.
                result={field:(row[field] if field in ['dataset','scenario','method'] else
                               int(row[field]) if field in ['k','theta_factor','runs'] else float(row[field]))
                        for field in summary_fields}
                result.update(n=group[0]['n'],m=group[0]['m'],theta=group[0]['theta'],
                              ci95_per_coupon_halfwidth=float(row['ci95_across_runs_halfwidth'])/10,
                              original_panel='left' if dataset=='Netscience' else 'middle')
                assert result['theta']==result['theta_factor']*result['n']
                summary.append(result);ordered_raw.extend(group)
    sources=[raw_path,summary_path,ROOT/'assets/experiments-20261012/sampling-sensitivity.pdf',
             ROOT/'experiments/build_section6_artifacts.py']
    hashes.update({p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    payload=dict(data_status='measured',source_sha256=hashes,raw_fields=raw_fields,
        raw_rows=ordered_raw,summary_rows=summary,job_parameters=job_parameters,
        verification=dict(archived_job_hashes_match=True,csv_rows_match_all_20_jobs=True,
                          summary_means_and_intervals_match_raw=True,
                          displayed_quantity='mean_spread / k',k=10,runs=5,
                          forward_campaigns_per_run=5000,theta_factors=FACTORS,
                          balanced_setting='Original balanced reference; not the variance-expanded balanced setting.'))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'figure-6-data.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print('Verified 20 archived jobs, 80 evaluations and 16 plot points; no new experiments or scaling.')

if __name__=='__main__':main()
