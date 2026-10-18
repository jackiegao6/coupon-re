"""Exact full-bank/Python-stream/C++-stream checks using identical random worlds."""
from pathlib import Path
import argparse
import json
import random
import struct
import subprocess

import numpy as np

from uniform_ssr import CouponRR, SSRBank
from streaming_ssr import stream_ssr_greedy
from verify_uniform_ssr import fixtures
from run_figure3_k300_extension import dump_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for graph, alpha, discard, probabilities in fixtures():
        for seed in (19, 47, 20261016):
            folder = args.output / f'{graph.name}-{seed}'
            folder.mkdir(exist_ok=True)
            theta, k = 379, graph.n
            component_seeds = [seed * 1000 + j for j in range(k)]
            rng = random.Random(seed)
            roots = [rng.randrange(graph.n) for _ in range(theta)]
            sampler = CouponRR(graph, alpha, discard, probabilities)
            indices = []
            for cseed in component_seeds:
                index = {}; rng = random.Random(cseed)
                for t, root in enumerate(roots):
                    for v in sampler.sample(root, rng):
                        index.setdefault(v, []).append(t)
                indices.append(index)
            full = SSRBank(graph.n, k, theta, indices, np.bincount(roots, minlength=graph.n),
                           0, sampler.stats(), 0)
            expected, info = full.greedy()
            got, streaming = stream_ssr_greedy(CouponRR(graph, alpha, discard, probabilities),
                                              k, theta, seed, component_seeds)
            assert got == expected
            assert streaming['selected_marginal_counts'] == info['selected_marginal_counts']
            assert streaming['stats'] == sampler.stats()
            assert streaming['training_estimate_not_quality'] == full.estimate(expected)
            graphfile = folder / 'graph.bin'
            with graphfile.open('wb') as f:
                f.write(struct.pack('<II', graph.n, graph.m))
                for data, dtype in [(graph.indptr, '<u8'), (graph.indices, '<i4'), (alpha, '<f8'),
                                    (discard, '<f8'), (sampler.inptr, '<u8'), (sampler.inedges, '<u8'),
                                    (sampler.sources, '<i4'), (sampler.low, '<f8'), (sampler.high, '<f8')]:
                    np.asarray(data, dtype=dtype).tofile(f)
            inp = folder / 'input.bin'; base = folder / 'result'
            with inp.open('wb') as f:
                f.write(struct.pack('<II', theta, k)); dump_state(f, seed)
                for cseed in component_seeds: dump_state(f, cseed)
            prefixes = sorted({1, max(1, k // 2), k})
            for j in prefixes:
                Path(str(base) + f'.k{j}.json').unlink(missing_ok=True)
            subprocess.run([str(args.exe.resolve()), str(graphfile), str(inp), str(base),
                            ','.join(map(str, prefixes))], check=True, capture_output=True)
            last_peak = 0
            for j in prefixes:
                actual = json.loads(Path(str(base) + f'.k{j}.json').read_text())
                assert actual['selected'] == expected[:j]
                assert actual['selected_marginal_counts'] == info['selected_marginal_counts'][:j]
                assert actual['covered_count'] == sum(info['selected_marginal_counts'][:j])
                assert actual['stats']['calls'] == theta * j
                assert actual['retained_component_count'] == 1
                assert actual['peak_private_bytes'] >= last_peak > -1
                last_peak = actual['peak_private_bytes']
            assert actual['stats'] == sampler.stats()
            cases.append(dict(graph=graph.name, seed=seed, k=k, theta=theta,
                              exact_choices=True, exact_gains=True, exact_rr_counters=True,
                              prefixes=prefixes))
    (args.output / 'verification.json').write_text(json.dumps(dict(passed=True, cases=cases,
        comparison='Fully materialized reference, Python streaming, and compiled streaming on identical root/component streams',
        fixtures='weighted edges, branches/merges, cycles, self-loops, sinks, isolates, no consumption, all consumption'), indent=2)+'\n')
    print(f'PASS: {len(cases)} exact full-bank/streaming cases')


if __name__ == '__main__': main()
