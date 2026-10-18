"""Registered controlled experiments with exact finite-graph evaluation.

Do not alter historical methods or results. Enumerate all seed subsets on
six predefined 12-node graphs, and evaluate each sampled solution exactly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

import run_real_submission as core
from forward_coverage_baselines import sample_coverage, select_coverage
from uniform_ssr import CouponRR, sample_ssr
from verify_uniform_ssr import exact_matrix, graph_from_rows, spread

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/exact-selection-diagnosis-20261014'
FACTORS = [8, 32, 64]
BUDGETS = [2, 6]
REPEATS = 100
SOURCES = ['experiments/run_exact_selection_diagnosis.py',
           'experiments/run_real_submission.py', 'experiments/uniform_ssr.py',
           'experiments/verify_uniform_ssr.py', 'experiments/forward_coverage_baselines.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def cases():
    """Controls fixed before selection trials; all six are reported."""
    n = 12
    definitions = []

    def add(name, rows, a, d, description):
        definitions.append(dict(name=name, rows=rows, alpha=list(map(float, a)),
                                discard=list(map(float, d)), description=description))

    add('equal-isolates', [[] for _ in range(n)], [.35]*n, [.65]*n,
        'Equal consuming probabilities, no edges: every seed set has identical spread.')
    a = np.linspace(.15, .85, n)
    add('unequal-isolates', [[] for _ in range(n)], a, 1-a,
        'Heterogeneous consuming probabilities without overlap; optimum is top-k consumption.')
    add('uniform-ring', [[(v-1) % n, (v+1) % n] for v in range(n)], [.35]*n, [.15]*n,
        'Bidirectional cycle with homogeneous behavior; seed spacing affects overlap.')
    star = [list(range(1,n))] + [[0] for _ in range(1,n)]
    g = graph_from_rows(star)
    a, d, _ = core.node_probabilities(g, 'balanced')
    add('single-hub', star, a, d,
        'Bidirectional star; balanced degree-dependent behavior from the main experiments.')
    rows = [list(range(1,9))+[9]] + [[0] for _ in range(1,9)] + [[0,10,11],[9],[9]]
    g = graph_from_rows(rows)
    a, d, _ = core.node_probabilities(g, 'balanced')
    add('unequal-hubs', rows, a, d,
        'Two linked hubs with eight and two leaves; same balanced parameter rule.')
    rng = random.Random(20261014)
    rows = []
    for v in range(n):
        targets = { (v+1) % n }
        candidates = [u for u in range(n) if u != v and u not in targets]
        targets.update(rng.sample(candidates, 1+(v % 2)))
        rows.append(sorted(targets))
    add('directed-cycles', rows, [.15,.25,.35,.45,.55,.65]*2, [.10]*n,
        'Directed ring plus seeded random chords; heterogeneous behavior and revisit dropping.')
    return definitions


def specs():
    return [dict(id=f'{case["name"]}-k{k}', case=case['name'], k=k)
            for case in cases() for k in BUDGETS]


def initialize():
    manifest = dict(cases=cases(), jobs=specs(), theta_factors=FACTORS,
        repeats=REPEATS, forward_batches=256,
        source_sha256={rel:sha(ROOT/rel) for rel in SOURCES},
        protocol='Controlled synthetic diagnosis, not new real-network evidence. '
        'All six graphs, both k values, and all three SSR budgets are fixed before results. '
        'Each stochastic configuration uses 100 independent selection streams. '
        'Uniform random spread is the mean over all k-subsets, not sampled. '
        'Exact greedy uses the true single-coupon matrix; OPT enumerates all k-subsets. '
        'Every selected solution is evaluated by exact simple-path probabilities. '
        'Forward-CELF uses the unchanged 256-trajectories-per-source implementation. '
        'Sample budgets are not equal-cost comparisons. Parallel timings are not performance claims. '
        'Report all configurations regardless of method ordering. '
        'Exact refers to finite path/subset enumeration, computed in floating-point arithmetic.')
    path = OUT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest
    else:
        write(path, manifest)
        for rel in SOURCES:
            target = OUT/'source'/Path(rel).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/rel, target)
    return manifest


def exact_dp(g, a, d):
    """Independent subset-state recurrence, using original graph probabilities."""
    @lru_cache(None)
    def visit(v, mask):
        result = np.zeros(g.n)
        result[v] = a[v]
        neighbors = g.out_neighbors(v)
        if len(neighbors):
            weight = (1-a[v]-d[v])/len(neighbors)
            for target in neighbors:
                target = int(target)
                if not (mask & (1 << target)):
                    result += weight * visit(target, mask | (1 << target))
        return result
    return np.asarray([visit(s, 1 << s) for s in range(g.n)])


def exact_greedy(q, k):
    order, details = [], []
    residual = np.ones(q.shape[1])
    for _ in range(k):
        gains = q @ residual
        gains[order] = -1
        best = int(np.argmax(gains))
        details.append(float(gains[best]))
        order.append(best)
        residual *= 1-q[best]
    return order, details


def truth(case, k):
    g = graph_from_rows(case['rows'], case['name'])
    a, d = np.asarray(case['alpha']), np.asarray(case['discard'])
    q = exact_matrix(CouponRR(g,a,d))
    dp = exact_dp(g,a,d)
    np.testing.assert_allclose(q,dp,atol=1e-12,rtol=1e-12)
    assert np.min(q) >= 0 and np.max(q.sum(axis=1)) <= 1+1e-12
    subsets = [list(s) for s in itertools.combinations(range(g.n),k)]
    values = np.asarray([spread(q,s) for s in subsets])
    best = int(np.argmax(values))
    greedy, gains = exact_greedy(q,k)
    if case['name'] == 'equal-isolates':
        np.testing.assert_allclose(values, .35*k, atol=1e-12)
    if case['name'] == 'unequal-isolates':
        assert abs(values[best]-sum(sorted(a,reverse=True)[:k])) < 1e-12
    return g,a,d,q,dict(n=g.n,m=g.m,k=k,exact_matrix_source_consumer=q.tolist(),
        path_vs_subset_dp_max_error=float(np.max(abs(q-dp))),
        all_subsets=subsets,all_subset_spreads=values.tolist(),
        optimum=float(values[best]),optimal_example=subsets[best],
        optimal_subset_count=int(np.sum(np.isclose(values, values[best],rtol=0,atol=1e-12))),
        exact_random_mean=float(values.mean()),
        exact_random_subset_sd=float(values.std(ddof=0)),
        exact_greedy_seeds=greedy,exact_greedy_spread=spread(q,greedy),
        exact_greedy_marginal_gains=gains,expected_rr_size=float(q.sum()/g.n))


def marginal_audit(bank, q, selected):
    covered = bytearray(bank.theta)
    residual = np.ones(bank.n)
    available = np.ones(bank.n,dtype=bool)
    rounds = []
    for j,chosen in enumerate(selected):
        true = q @ residual
        empirical = np.zeros(bank.n)
        for v,indices in bank.index[j].items():
            empirical[v] = bank.n/bank.theta * sum(not covered[t] for t in indices)
        errors = empirical[available]-true[available]
        best_true = float(true[available].max())
        selection_scores = empirical.copy()
        selection_scores[~available] = -1
        assert int(np.argmax(selection_scores)) == chosen
        regret = max(0.,best_true-float(true[chosen]))
        rounds.append(dict(round=j+1,selected=chosen,true_selected_gain=float(true[chosen]),
            true_best_gain=best_true,local_regret=regret,
            selected_gain_error=float(empirical[chosen]-true[chosen]),
            marginal_rmse=float(np.sqrt(np.mean(errors*errors))),
            maximum_absolute_marginal_error=float(np.max(abs(errors))),
            selected_is_true_max=regret < 1e-12))
        for t in bank.index[j].get(chosen,()):
            covered[t] = 1
        available[chosen] = False
        residual *= 1-q[chosen]
    return rounds


def run(job, manifest):
    case = next(c for c in manifest['cases'] if c['name']==job['case'])
    g,a,d,q,reference = truth(case,job['k'])
    rows=[]
    start=time.perf_counter()
    for rep in range(REPEATS):
        for factor in FACTORS:
            seed=core._stable_seed('exact-selection-diagnosis',job['id'],'SSR',factor,rep)
            bank=sample_ssr(CouponRR(g,a,d),job['k'],factor*g.n,seed)
            selected,info=bank.greedy()
            audit=marginal_audit(bank,q,selected)
            rows.append(dict(method='CIM-RIS',theta_factor=factor,replicate=rep,
                selection_seed=seed,seeds=selected,exact_spread=spread(q,selected),
                training_estimate=info['training_estimate_not_quality'],
                memberships=bank.stats['memberships'],rr_positions=factor*g.n*job['k'],
                rounds=audit))
        seed=core._stable_seed('exact-selection-diagnosis',job['id'],'Forward-CELF',rep)
        fbank=sample_coverage(g,a,d,256,seed)
        selected,count,_=select_coverage(fbank,job['k'])
        rows.append(dict(method='Forward-CELF',theta_factor=None,replicate=rep,
            selection_seed=seed,seeds=selected,exact_spread=spread(q,selected),
            training_estimate=count/256,forward_batches=256))
    for row in rows:
        assert len(row['seeds'])==len(set(row['seeds']))==job['k']
        assert 0 <= row['exact_spread'] <= reference['optimum']+1e-12 <= job['k']+1e-12
    assert len(rows)==REPEATS*(len(FACTORS)+1)
    return dict(job=job,status='complete',reference=reference,rows=rows,
                elapsed_seconds_not_benchmark=time.perf_counter()-start)


def launch(job):
    target=OUT/'jobs'/f'{job["id"]}.json'
    if target.exists():
        result=json.loads(target.read_text(encoding='utf-8'))
        assert result['job']==job and result['status']=='complete' and len(result['rows'])==400
        return job['id']+' reused'
    log=OUT/'logs'/f'{job["id"]}.txt'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,__file__,'--job',job['id']],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f'Failed {job["id"]}; see {log}')
    return job['id']+' complete'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--initialize-only',action='store_true')
    args=parser.parse_args()
    if args.job:
        manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        for rel,digest in manifest['source_sha256'].items():
            assert sha(ROOT/rel)==digest,rel
        job=next(j for j in manifest['jobs'] if j['id']==args.job)
        write(OUT/'jobs'/f'{job["id"]}.json',run(job,manifest))
    else:
        manifest=initialize()
        print(f'Registered {len(manifest["jobs"])} configurations; 100 selection runs per method/budget.',flush=True)
        if not args.initialize_only:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(launch,job) for job in manifest['jobs']]
                for future in as_completed(futures):
                    print(future.result(),flush=True)
            print('All exact-diagnosis jobs complete.',flush=True)
