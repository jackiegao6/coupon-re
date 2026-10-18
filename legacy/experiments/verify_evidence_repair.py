"""Check repaired evidence against original records and saved independent runs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results/evidence-20260913'
T4 = 2.7764451051977987

# Historical runs are checked against their exact source, not rewritten hashes.
ARCHIVED_CORE_SHA256 = '240ede44730726a4ed8170e8b7cdf97b33bec6ea94a5fa2b5dd4dc96e9227f20'
_reported_archives = set()


def verify_fingerprint(name, digest):
    current = ROOT / name
    if hashlib.sha256(current.read_bytes()).hexdigest() == digest:
        return
    normalized = name.replace('\\', '/')
    if normalized == 'experiments/run_real_submission.py' and digest == ARCHIVED_CORE_SHA256:
        archived = ROOT / 'experiments/archives/run_real_submission-240ede44730726a4.py'
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest, archived
        if normalized not in _reported_archives:
            print('HISTORICAL SOURCE:', normalized, 'verified against byte-exact archive;',
                  'current working source differs and is NOT certified by this check.')
            _reported_archives.add(normalized)
        return
    raise AssertionError(('Fingerprint mismatch without a registered archive', name, digest))


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def close(a, b, tol=1e-8):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=tol), (a, b)


def verify_ci():
    old = rows(ROOT / 'experiments/results/validated-v2/validated_summary.csv')
    new = rows(RESULTS / 'ci/validated_summary.csv')
    raw = rows(RESULTS / 'ci/validated_raw.csv')
    assert raw == rows(ROOT / 'experiments/results/validated-v2/validated_raw.csv')
    assert len(old) == len(new) == 522
    for a, b in zip(old, new):
        for key in a:
            if key != 'ci95_across_selection_runs':
                if a[key] != b[key]:
                    # Python/NumPy versions may format the last float digit differently.
                    close(a[key], b[key], tol=1e-12)
        assert int(b['selection_repeats']) == 5
        close(b['ci95_across_selection_runs'], T4 * float(b['std_across_selection_runs']) / math.sqrt(5))
    print('CI: 522 summaries; 2610 original rows unchanged; t(df=4) intervals verified.')


def verify_sampler():
    batches = rows(RESULTS / 'sampler/summary_batches.csv')
    summary = rows(RESULTS / 'sampler/summary.csv')
    metadata = json.loads((RESULTS / 'sampler/summary.metadata.json').read_text())
    assert len(batches) == 360 and len(summary) == 12
    grouped = defaultdict(list)
    for row in batches:
        assert row['status'] == 'REAL_EXPERIMENT'
        assert row['protocol_version'] == 'evidence-20260913-sampler'
        assert int(row['batch_size']) == 20000
        assert 0 <= float(row['zero_fraction']) <= 1
        assert 0 < float(row['loop_seconds']) <= float(row['batch_seconds'])
        if row['sampler'] == 'Conditioned root':
            assert float(row['zero_fraction']) == 0
        grouped[(row['dataset'], row['k'], row['sampler'])].append(row)
    expected = {(d, str(k), s) for d in ('Netscience', 'EmailEnron') for k in (10, 50, 200) for s in ('Uniform root', 'Conditioned root')}
    assert set(grouped) == expected
    for row in summary:
        batch = grouped[(row['dataset'], row['k'], row['sampler'])]
        assert len(batch) == 30 and {int(r['repeat']) for r in batch} == set(range(30))
        assert len({r['batch_seed'] for r in batch}) == 30
        estimates = [float(r['spread_estimate']) for r in batch]
        close(row['mean_spread_estimate'], statistics.mean(estimates))
        close(row['std_spread_estimate'], statistics.stdev(estimates))
        close(row['coefficient_of_variation'], statistics.stdev(estimates) / statistics.mean(estimates))
        for raw_key, aggregate_key in [('zero_fraction', 'mean_zero_fraction'), ('loop_seconds', 'mean_loop_seconds'), ('batch_seconds', 'mean_batch_seconds')]:
            close(row[aggregate_key], statistics.mean(float(r[raw_key]) for r in batch))
    for field in ('inputs_sha256', 'code_sha256'):
        for name, digest in metadata[field].items():
            verify_fingerprint(name, digest)
    for name, digest in metadata['outputs_sha256'].items():
        assert hashlib.sha256((RESULTS / 'sampler' / name).read_bytes()).hexdigest() == digest
    old = rows(ROOT / 'experiments/results/validated-v2/validated_sampler_ablation.csv')
    for a, b in zip(old, summary):
        for key in ('dataset', 'k', 'sampler', 'mean_spread_estimate', 'std_spread_estimate', 'mean_zero_fraction'):
            assert a[key] == b[key], key
    ratios = {}
    for dataset in ('Netscience', 'EmailEnron'):
        ratios[dataset] = {}
        for k in (10, 50, 200):
            pair = {r['sampler']: r for r in summary if r['dataset'] == dataset and int(r['k']) == k}
            costs = {method: float(r['std_spread_estimate'])**2 * float(r['mean_batch_seconds']) for method, r in pair.items()}
            ratios[dataset][k] = costs['Uniform root'] / costs['Conditioned root']
    print('SAMPLER: 360 batches and all fingerprints verified; estimates reproduce archived summaries; full-batch efficiency ratios:')
    print(json.dumps(ratios, indent=2))


def verify_large():
    folder = RESULTS / 'large-graph'
    raw = rows(folder / 'raw.csv')
    summaries = rows(folder / 'summary.csv')
    jobs = list((folder / 'jobs').glob('*.json'))
    methods = {'CIM-RIS', 'IC-RIS', '1Hop-Sort', 'Alpha-Sort', 'DegreeTopM', 'PageRank', 'Random'}
    expected = {(k, seed, m) for k in (10, 25, 50, 100, 150, 200) for seed in range(20260715, 20260720) for m in methods}
    assert len(jobs) == 30 and len(raw) == 210 and len(summaries) == 42
    assert {(int(r['k']), int(r['selection_seed']), r['method']) for r in raw} == expected
    indexed = {(r['k'], r['selection_seed'], r['method']): r for r in raw}
    archived_scalability = {(r['k'], r['selection_seed']): r for r in rows(ROOT / 'experiments/results/validated-v2/validated_scalability.csv')}
    for path in jobs:
        job = json.loads(path.read_text())
        assert job['status'] == 'REAL_EXPERIMENT'
        assert len(job['rows']) == 7
        for name, digest in job['key']['fingerprints'].items():
            verify_fingerprint(name, digest)
        evaluation_seeds = {r['evaluation_seed'] for r in job['rows']}
        assert len(evaluation_seeds) == 1
        assert not evaluation_seeds.intersection(job['selection_streams'].values())
        for r in job['rows']:
            row = indexed[(str(r['k']), str(r['selection_seed']), r['method'])]
            for key, value in r.items():
                assert row[key] == str(value)
    grouped = defaultdict(list)
    for row in raw:
        k = int(row['k'])
        seeds = json.loads(row['seeds'])
        assert len(seeds) == len(set(seeds)) == k
        assert all(0 <= node < 154907 for node in seeds)
        assert 0 <= float(row['mean_adopters']) <= float(row['mean_redemptions']) + 1e-9 <= k + 1e-9
        assert int(row['eval_simulations']) == 10000
        if row['method'] == 'CIM-RIS':
            archived = archived_scalability[(row['k'], row['selection_seed'])]
            close(row['training_estimate_not_quality'], archived['estimated_spread'], tol=1e-7)
            assert row['rr_memberships'] == archived['rr_memberships']
        grouped[(row['k'], row['method'])].append(float(row['mean_adopters']))
    for row in summaries:
        values = grouped[(row['k'], row['method'])]
        close(row['mean_adopters'], statistics.mean(values))
        close(row['ci95_across_selection_runs'], T4 * statistics.stdev(values) / math.sqrt(5))
    for line in (folder / 'manifest.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    print('LARGE GRAPH: 30 jobs, 210 evaluations, 42 summaries, physical bounds and all checksums verified; training estimates and RR memberships replay archived scalability records.')
    for k in (10, 25, 50, 100, 150, 200):
        cell = {r['method']: float(r['mean_adopters']) for r in summaries if int(r['k']) == k}
        best = max((m for m in methods if m != 'CIM-RIS'), key=cell.get)
        print(k, 'CIM', round(cell['CIM-RIS'], 4), 'best', best, round(cell[best], 4), 'relative_gain_pct', round(100*(cell['CIM-RIS']/cell[best]-1), 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true', help='Verify CI and sampler while large-graph jobs are running')
    args = parser.parse_args()
    verify_ci()
    verify_sampler()
    if not args.partial:
        verify_large()
