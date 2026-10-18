"""Finalize the first three repetitions after the user's budget amendment.

The original protocol and measurement runner remain unchanged. Optional
--run-missing resumes only the nine jobs authorized by the amendment.
"""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import run_figure6_k100 as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-missing', action='store_true')
    args = parser.parse_args()
    protocol_path = run.EXT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['repeats'] == 5 and protocol['k'] == 100
    for name, digest in protocol['source_sha256'].items():
        assert run.study.hash_file(run.ROOT / name) == digest, name
    jobs = [job for job in protocol['jobs'] if job['replicate'] < 3]
    assert len(jobs) == 9
    amendment_path = run.EXT / 'three-run-amendment.json'
    if not amendment_path.exists():
        run.study.write_json(amendment_path, dict(
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            reason='User requested three repetitions instead of five.',
            original_repeats=5, repeats=3, included_replicates=[0, 1, 2],
            original_protocol_sha256=run.study.hash_file(protocol_path),
            unchanged=['datasets', 'k', 'theta_factors', 'behavior_parameters',
                       'seeds', 'evaluation_trials', 'timing', 'memory'],
            jobs=[job['id'] for job in jobs],
            completed_measurements_discarded=0))
    amendment = json.loads(amendment_path.read_text())
    assert amendment['original_protocol_sha256'] == run.study.hash_file(protocol_path)
    assert amendment['repeats'] == 3 and amendment['jobs'] == [job['id'] for job in jobs]
    rows, hashes = [], {}
    for job in jobs:
        path = run.EXT / 'jobs' / (job['id'] + '.json')
        if not path.exists() and args.run_missing:
            subprocess.run([sys.executable, str(run.ROOT / 'experiments/run_figure6_k100.py'),
                            '--job', job['id']], cwd=run.ROOT, check=True)
        record = json.loads(path.read_text())
        assert record['status'] == 'complete' and record['job'] == job
        assert record['protocol_sha256'] == run.study.hash_file(protocol_path)
        assert record['kernel_sha256'] == protocol['kernel_sha256']
        assert record['dataset_sha256'] == protocol['dataset_sha256'][job['dataset']]
        assert sorted(row['theta_factor'] for row in record['rows']) == run.FACTORS
        for row in record['rows']:
            run.check_row(row)
            assert row['replicate'] == job['replicate'] and row['job_id'] == job['id']
        rows.extend(record['rows'])
        hashes[path.relative_to(run.ROOT).as_posix()] = run.study.hash_file(path)
    datasets = [run.DISPLAY.get(ds, ds) for ds in run.DATASETS]
    rows.sort(key=lambda row: (datasets.index(row['dataset']), row['theta_factor'], row['replicate']))
    assert len(rows) == 36
    run.study.write_json(run.OUT / 'results.json', dict(
        status='complete', data_status='measured', k=100, datasets=datasets,
        scenario='balanced', theta_factors=run.FACTORS, repeats=3,
        evaluation_trials=run.TRIALS, completed_jobs=9, total_jobs=9,
        raw_rows=rows, source_sha256=hashes,
        protocol_sha256=run.study.hash_file(protocol_path),
        amendment_sha256=run.study.hash_file(amendment_path)))
    run.study.write_json(run.EXT / 'active.json', dict(
        status='complete', k=100, repeats=3, completed_jobs=9, observations=36))
    print('Complete: 3 repetitions, 9 jobs, 36 paired quality/time/memory observations.')


if __name__ == '__main__':
    main()
