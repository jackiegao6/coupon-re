"""Exhaustive world-level verification, separate from empirical quality trials."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import unittest

import numpy as np

from run_real_submission import Graph
from uniform_ssr import CouponRR, sample_ssr, sufficient_budget


def graph_from_rows(rows, name="tiny"):
    degrees = np.array([len(row) for row in rows], dtype=np.int32)
    indices = np.array([v for row in rows for v in row], dtype=np.int32)
    incoming = [[] for _ in rows]
    for s, row in enumerate(rows):
        for v in row:
            incoming[v].append(s)
    return Graph(name, len(rows), len(indices), indices,
                 np.r_[0, np.cumsum(degrees)], incoming, degrees)


def fixtures():
    return [
        (graph_from_rows([[1, 2], [3], [3], [0]], "branch_cycle"),
         np.array([.2, .3, .4, .5]), np.array([.1, .2, .1, .2]),
         np.array([.2, .5, .5, .5, .3])),
        (graph_from_rows([[1, 2], [3], [3], []], "diamond"),
         np.array([0., 0., 0., 1.]), np.zeros(4), None),
        (graph_from_rows([[0, 1], [0, 2], []], "self_loop"),
         np.array([.2, .4, .7]), np.array([.1, .2, .3]), None),
        (graph_from_rows([[], [], []], "isolates"),
         np.array([.9, .6, .2]), np.array([.1, .4, .8]), None),
        (graph_from_rows([[1], [0]], "pure_cycle"),
         np.zeros(2), np.zeros(2), None),
        (graph_from_rows([[1], [0]], "all_consume"),
         np.ones(2), np.zeros(2), None),
    ]


def exact_matrix(sampler):
    """Independent simple-path enumeration; shape [source, consumer]."""
    g = sampler.graph
    q = np.zeros((g.n, g.n))
    for start in range(g.n):
        def visit(v, visited, mass):
            q[start, v] += mass * sampler.alpha[v]
            for edge in range(int(g.indptr[v]), int(g.indptr[v + 1])):
                w = int(g.indices[edge])
                if w not in visited and sampler.probabilities[edge] > 0:
                    visit(w, visited | {w}, mass * sampler.probabilities[edge])
        visit(start, {start}, 1.)
    return q


def spread(q, seeds):
    return float(np.sum(1 - np.prod(1 - q[list(seeds)], axis=0))) if len(seeds) else 0.


def enumerate_worlds(sampler):
    g = sampler.graph
    options = []
    for v in range(g.n):
        row = []
        if sampler.alpha[v] > 0:
            row.append((-2, float(sampler.alpha[v]), sampler.alpha[v]/2))
        if sampler.discard[v] > 0:
            row.append((-1, float(sampler.discard[v]), sampler.alpha[v]+sampler.discard[v]/2))
        for e in range(int(g.indptr[v]), int(g.indptr[v + 1])):
            if sampler.probabilities[e] > 0:
                row.append((int(g.indices[e]), float(sampler.probabilities[e]),
                            float((sampler.low[e]+sampler.high[e])/2)))
        options.append(row)
    for combination in itertools.product(*options):
        yield ([x[0] for x in combination], math.prod(x[1] for x in combination),
               [x[2] for x in combination])


def forward_world(actions, start):
    seen = set()
    v = start
    while v not in seen:
        seen.add(v)
        if actions[v] == -2:
            return v
        if actions[v] == -1:
            return -1
        v = actions[v]
    return -1


def verify_fixture(fixture):
    g, a, d, weights = fixture
    sampler = CouponRR(g, a, d, weights)
    q = np.zeros((g.n, g.n))
    reverse_q = np.zeros_like(q)
    count = comparisons = 0
    probability_sum = 0.
    for actions, probability, draws in enumerate_worlds(sampler):
        count += 1
        probability_sum += probability
        outcomes = [forward_world(actions, s) for s in range(g.n)]
        for s, u in enumerate(outcomes):
            if u >= 0:
                q[s, u] += probability
        for root in range(g.n):
            rr = set(sampler.sample(root, random.Random(0), draws))
            expected = {s for s in range(g.n) if outcomes[s] == root}
            if rr != expected:
                raise AssertionError((g.name, actions, root, rr, expected))
            for s in rr:
                reverse_q[s, root] += probability
            comparisons += g.n
    path_q = exact_matrix(sampler)
    np.testing.assert_allclose(q, reverse_q, atol=1e-13)
    np.testing.assert_allclose(q, path_q, atol=1e-13)
    assert abs(probability_sum-1) < 1e-12
    # Exact expected RR size and edge work, including empty samples.
    qv = q.sum(axis=1)
    indegree = np.diff(sampler.inptr)
    return {"graph": g.name, "n": g.n, "worlds": count,
            "pointwise_comparisons": comparisons, "max_probability_error": float(np.max(abs(q-reverse_q))),
            "path_enumeration_error": float(np.max(abs(q-path_q))),
            "expected_rr_size": float(qv.mean()),
            "expected_edge_scans": float(np.dot(qv, indegree)/g.n),
            "edge_scan_bound": g.m/g.n}


class UniformSSRTests(unittest.TestCase):
    def test_exhaustive_forward_reverse_equivalence(self):
        for fixture in fixtures():
            with self.subTest(graph=fixture[0].name):
                verify_fixture(fixture)

    def test_cached_choice_shared_across_branches(self):
        # Every fixed diamond world reaches root 3 from 0; checking each branch
        # with a fresh draw instead would include 0 with probability only 3/4.
        g, a, d, w = fixtures()[1]
        sampler = CouponRR(g, a, d, w)
        for draw in (.25, .75):
            self.assertEqual(set(sampler.sample(3, random.Random(0), [draw, .5, .5, .5])), set(range(4)))

    def test_empty_samples_retained_and_distinct_seeds(self):
        g = graph_from_rows([[], [], []])
        bank = sample_ssr(CouponRR(g, np.zeros(3), np.ones(3)), 3, 100, 1)
        seeds, info = bank.greedy()
        self.assertEqual(bank.empty_ssr, 100)
        self.assertEqual(bank.stats['calls'], 300)
        self.assertEqual(seeds, [0, 1, 2])
        self.assertEqual(info['training_estimate_not_quality'], 0)

    def test_indexed_greedy_matches_direct_coverage(self):
        g, a, d, w = fixtures()[0]
        bank = sample_ssr(CouponRR(g, a, d, w), 3, 300, 20)
        seeds, info = bank.greedy()
        prefix = []
        for j in range(3):
            expected = max((v for v in range(g.n) if v not in prefix),
                           key=lambda v: (bank.estimate(prefix+[v]), -v))
            self.assertEqual(seeds[j], expected)
            prefix.append(expected)
        self.assertAlmostEqual(info['training_estimate_not_quality'], bank.estimate(seeds))

    def test_budget_covers_bernstein_union_bound(self):
        n, k, L, eps, delta = 6, 2, 1., .2, .05
        theta = sufficient_budget(n, k, L, eps, delta)
        tau = eps*L/(2*k)
        count = sum(math.factorial(n)//math.factorial(n-j) for j in range(1,k+1))
        bound = 2*count*math.exp(-theta*tau*tau/(2*n+2*n*tau/3))
        self.assertLessEqual(bound, delta)

    def test_invalid_inputs(self):
        g = graph_from_rows([[]])
        with self.assertRaises(ValueError):
            CouponRR(g, [.2], [.2])
        with self.assertRaises(ValueError):
            sample_ssr(CouponRR(g, [1.], [0.]), 2, 10, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(UniformSSRTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"tests": result.testsRun, "passed": True,
            "fixtures": [verify_fixture(x) for x in fixtures()]}, indent=2)+'\n', encoding='utf-8')
