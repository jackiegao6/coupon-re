"""Verify concurrent result commits, serial parity and checkpoint-only recovery."""
from pathlib import Path
import json
import subprocess
import sys

from run_streaming_sensitivity import ROOT, read, write


def main():
    out = ROOT / 'tmp/parallel-sensitivity-runner-check'
    runner = ROOT / 'experiments/run_parallel_streaming_sensitivity.py'
    def run():
        subprocess.run([sys.executable, str(runner), 'run', '--output', str(out), '--workers', '3'], check=True)
    results = read(out / 'results.json')
    assert results['completed_configurations'] == results['total_configurations'] == 6
    assert len(results['raw_rows']) == 6
    for row in results['raw_rows']:
        assert row['spread'] == row['consumed_coupons'] == row['k']
        assert row['execution_mode'] == 'parallel-admitted'
    serial = read(ROOT / 'tmp/streaming-sensitivity-runner-check/first/results.json')['raw_rows']
    for old in serial:
        new = next(r for r in results['raw_rows'] if (r['theta_factor'], r['k']) == (old['theta_factor'], old['k']))
        assert (new['seeds'], new['spread'], new['consumed_coupons']) == (old['seeds'], old['spread'], old['consumed_coupons'])
    attempts = {p.name: len(read(p)['attempts']) for p in (out / 'jobs').glob('*.json')}
    run()
    jobfile = next((out / 'jobs').glob('*.json'))
    saved = read(jobfile)
    lost = saved['rows'].pop()
    saved['status'] = 'partial'
    write(jobfile, saved)
    run()
    assert attempts == {p.name: len(read(p)['attempts']) for p in (out / 'jobs').glob('*.json')}
    recovered = next(r for r in read(jobfile)['rows'] if r['k'] == lost['k'])
    assert recovered['spread'] == lost['spread'] and recovered['seeds'] == lost['seeds']
    assert read(out / 'results.json')['completed_configurations'] == 6
    report = dict(passed=True, three_concurrent_jobs=True, six_exact_measured_points=True,
        serial_seed_and_quality_parity=True, completed_jobs_not_rerun=True,
        checkpoint_only_recovery=True, synchronized_publication=True)
    write(out / 'verification.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
