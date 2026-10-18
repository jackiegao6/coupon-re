"""Memory-admitted parallel orchestration of a frozen sensitivity experiment.

Keeps the scientific protocol and native executables unchanged. Can adopt live
native selections from the serial coordinator without repeating their work.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import threading
import time

import psutil

GIB = 1024 ** 3


def load_base(out):
    path = out / 'source/run_streaming_sensitivity.py'
    spec = importlib.util.spec_from_file_location('frozen_sensitivity', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def alive(identity):
    try:
        proc = psutil.Process(identity['pid'])
        return abs(proc.create_time() - identity['create_time']) < .01 and proc.is_running()
    except psutil.Error:
        return False


def reservation(job, protocol):
    """Conservative admission estimate, NOT a measured memory result."""
    graph = protocol['graphs'][job['dataset']]
    return 128 * 1024 ** 2 + 14 * graph['n'] * job['theta_factor'] + 32 * graph['n']


def discover(out, protocol):
    found = {}
    exe = (out / 'work/streaming_ssr.exe').resolve()
    for proc in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        try:
            cmd = proc.info['cmdline'] or []
            if len(cmd) < 4 or Path(cmd[0]).resolve() != exe:
                continue
            folder = Path(cmd[3]).resolve().parent
            matches = [j for j in protocol['jobs'] if folder.parent == (out / 'work' / j['id']).resolve()]
            if len(matches) != 1:
                raise RuntimeError(f'Cannot identify live selection {proc.pid}: {cmd}')
            job = matches[0]
            if job['id'] in found:
                raise RuntimeError(f'Duplicate live selections for {job["id"]}')
            found[job['id']] = dict(pid=proc.pid, create_time=proc.info['create_time'],
                                      folder=str(folder.relative_to(out)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


class Runner:
    def __init__(self, out, workers=3, reserve_gib=1.5):
        self.out = out
        self.base = load_base(out)
        self.protocol = self.base.read(out / 'protocol.json')
        self.workers = workers
        self.reserve_bytes = reserve_gib * GIB
        self.mutex = threading.RLock()
        self.local = threading.local()
        self.states = {}
        self.original_write = self.base.write
        self.original_publish = self.base.publish
        self.original_evaluate = self.base.evaluate
        # One publisher and synchronized job commits; separate active state per job.
        self.base.write = self.write
        self.base.publish = self.publish
        self.base.evaluate = self.evaluate

    def write(self, path, value):
        with self.mutex:
            if Path(path) == self.out / 'active.json' and getattr(self.local, 'job_id', None):
                job_id = self.local.job_id
                self.states[job_id] = value
                path = self.out / 'active-jobs' / (job_id + '.json')
            self.original_write(path, value)

    def publish(self, out, protocol=None):
        with self.mutex:
            return self.original_publish(out, protocol or self.protocol)

    def evaluate(self, *args, **kwargs):
        row = self.original_evaluate(*args, **kwargs)
        row['execution_mode'] = 'parallel-admitted'
        row['execution_record'] = 'execution-parallel.json'
        row['timing_comparability'] = 'wall time under concurrent load; not a serial runtime benchmark'
        return row

    def wait_adopted(self, job, identity):
        """Finish the original attempt, then let the frozen runner recover prefixes."""
        proc = psutil.Process(identity['pid']) if alive(identity) else None
        folder = self.out / identity['folder']
        cancelled = False
        code = None
        while proc is not None and alive(identity):
            if (self.out / 'STOP').exists():
                proc.terminate()
                cancelled = True
            self.write(self.out / 'active.json', dict(status='selecting', job_id=job['id'],
                pid=identity['pid'], adopted=True,
                elapsed_seconds=time.time() - identity['create_time'],
                completed_prefixes=[k for k in job['budgets'] if (folder / f'selection.k{k}.json').exists()],
                log=str((folder / 'selection.log').relative_to(self.out)), updated_utc=self.base.now()))
            try:
                code = proc.wait(timeout=2)
                break
            except psutil.TimeoutExpired:
                pass
        jobfile = self.out / 'jobs' / (job['id'] + '.json')
        saved = self.base.read(jobfile)
        attempt = next(a for a in saved['attempts'] if Path(a['folder']) == Path(identity['folder']))
        attempt.update(finished_utc=self.base.now(), returncode=code,
            elapsed_seconds=time.time() - identity['create_time'],
            status='stopped' if cancelled else 'finished', adopted_by_parallel_runner=True)
        self.write(jobfile, saved)

    def worker(self, job, adopted=None):
        self.local.job_id = job['id']
        try:
            if adopted:
                print(f'ADOPT {job["id"]} native PID={adopted["pid"]}', flush=True)
                self.wait_adopted(job, adopted)
            self.base.run_job(self.out, self.protocol, job)
            state = self.base.read(self.out / 'jobs' / (job['id'] + '.json'))
            status = state.get('status', 'pending')
            self.write(self.out / 'active.json', dict(job_id=job['id'], status=status, updated_utc=self.base.now()))
            return status
        except Exception as exc:
            self.write(self.out / 'errors' / (job['id'] + '.json'),
                       dict(error=repr(exc), updated_utc=self.base.now()))
            self.write(self.out / 'active.json', dict(job_id=job['id'], status='failed', error=repr(exc)))
            print(f'FAILED {job["id"]}: {exc!r}', flush=True)
            return 'failed'

    def outstanding_reservation(self, running):
        extra = 0
        with self.mutex:
            for job, _ in running.values():
                state = self.states.get(job['id'], {})
                resident = 0
                if state.get('pid'):
                    try:
                        resident = psutil.Process(state['pid']).memory_info().rss
                    except psutil.Error:
                        pass
                extra += max(0, reservation(job, self.protocol) - resident)
        return extra

    def run(self, takeover_serial=False):
        self.base.verify_inputs(self.out, self.protocol)
        lock = self.out / 'runner.lock'
        old = self.base.read(lock) if lock.exists() else None
        if old and alive(old):
            proc = psutil.Process(old['pid'])
            cmd = proc.cmdline()
            expected = (self.out / 'source/run_streaming_sensitivity.py').resolve()
            if not takeover_serial or not any(Path(s).resolve() == expected for s in cmd[1:] if s.endswith('.py')):
                raise RuntimeError(f'Coordinator already active: {old["pid"]}')
            # Windows TerminateProcess ends only the coordinator, not its native child.
            # Do NOT use STOP or CTRL_C: the old coordinator's finally would kill it.
            if os.name != 'nt':
                raise RuntimeError('Live serial takeover is supported only on Windows')
            before = discover(self.out, self.protocol)
            self.write(self.out / 'serial-takeover.json', dict(previous_coordinator=old,
                live_children=before, requested_utc=self.base.now()))
            proc.terminate()
            proc.wait(timeout=15)
            print(f'Retired serial coordinator {old["pid"]}; native selections retained', flush=True)
        if lock.exists():
            lock.unlink()
        with lock.open('x') as stream:
            json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(), mode='parallel'), stream)
        running = {}
        memory_stall_since = None
        try:
            adopted = discover(self.out, self.protocol)
            self.write(self.out / 'execution-parallel.json', dict(updated_utc=self.base.now(),
                max_workers=self.workers, reserve_free_gib=self.reserve_bytes / GIB,
                admission_estimate='128 MiB + 14*theta bytes + 32*n bytes per selection; not measured data',
                adopted=adopted, protocol_sha256=self.base.sha(self.out / 'protocol.json'),
                supervisor_sha256=self.base.sha(Path(__file__)),
                timing_note='Concurrent wall times must not be treated as serial performance measurements.'))
            pending = []
            for job in self.protocol['jobs']:
                path = self.out / 'jobs' / (job['id'] + '.json')
                rows = self.base.read(path)['rows'] if path.exists() else []
                if job['id'] in adopted or {r['k'] for r in rows} != set(job['budgets']):
                    pending.append(job)
            with ThreadPoolExecutor(max_workers=max(self.workers, len(adopted))) as pool:
                for job in list(pending):
                    if job['id'] in adopted:
                        running[pool.submit(self.worker, job, adopted[job['id']])] = (job, time.time())
                        pending.remove(job)
                while pending or running:
                    for future in list(running):
                        if future.done():
                            future.result()
                            del running[future]
                    stopped = (self.out / 'STOP').exists()
                    if stopped:
                        pending.clear()
                    while pending and len(running) < self.workers:
                        available = psutil.virtual_memory().available
                        capacity = available - self.reserve_bytes - self.outstanding_reservation(running)
                        job = next((j for j in pending if reservation(j, self.protocol) <= capacity), None)
                        if job is None:
                            break
                        pending.remove(job)
                        with self.mutex:
                            self.states[job['id']] = dict(job_id=job['id'], status='starting')
                        running[pool.submit(self.worker, job)] = (job, time.time())
                        print(f'LAUNCH {job["id"]}; active={len(running)}/{self.workers}', flush=True)
                    if pending and not running and not stopped:
                        memory_stall_since = memory_stall_since or time.monotonic()
                    else:
                        memory_stall_since = None
                    waiting_for_memory = memory_stall_since is not None
                    with self.mutex:
                        self.original_write(self.out / 'active.json', dict(status='stopping' if stopped else
                            ('waiting_for_memory' if waiting_for_memory else 'parallel_running'),
                            max_workers=self.workers, running=[self.states.get(j['id'], {'job_id': j['id'], 'status': 'starting'})
                                for j, _ in running.values()], pending=[j['id'] for j in pending],
                            available_memory_gib=psutil.virtual_memory().available/GIB,
                            reserve_free_gib=self.reserve_bytes/GIB, updated_utc=self.base.now()))
                    if waiting_for_memory and time.monotonic() - memory_stall_since >= 60:
                        self.write(self.out / 'memory-deferred.json', dict(updated_utc=self.base.now(),
                            pending=[dict(job_id=j['id'], reserved_gib=reservation(j,self.protocol)/GIB)
                                     for j in pending], available_gib=psutil.virtual_memory().available/GIB,
                            reserve_free_gib=self.reserve_bytes/GIB))
                        print(f'DEFERRED {len(pending)} jobs: insufficient available memory', flush=True)
                        break
                    if running or pending:
                        time.sleep(5)
            result = self.publish(self.out)
            self.write(self.out / 'active.json', dict(status='paused_for_memory' if pending else result['status'],
                updated_utc=self.base.now(),
                completed_configurations=result['completed_configurations'], total_configurations=result['total_configurations']))
        finally:
            lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status', 'stop', 'publish'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--reserve-gib', type=float, default=1.5)
    parser.add_argument('--takeover-serial', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.reserve_gib < 0:
        parser.error('workers must be positive and reserve-gib nonnegative')
    runner = Runner(args.output.resolve(), args.workers, args.reserve_gib)
    if args.command == 'run':
        runner.run(args.takeover_serial)
    elif args.command == 'stop':
        (runner.out / 'STOP').touch()
    elif args.command == 'status':
        data = runner.base.read(runner.out / 'results.json')
        print(f"Measured configurations: {data['completed_configurations']}/{data['total_configurations']}")
        print(json.dumps(runner.base.read(runner.out / 'active.json'), ensure_ascii=False, indent=2))
    elif args.command == 'publish':
        lock = runner.out / 'runner.lock'
        if lock.exists() and alive(runner.base.read(lock)):
            raise RuntimeError('Active coordinator publishes automatically; do not publish concurrently')
        runner.publish(runner.out)


if __name__ == '__main__':
    main()
