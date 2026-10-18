import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from couponim import native
from couponim.forward import sample_coverage, select_coverage
from couponim.rr import CouponRR, SSRBank
from couponim.simulation import evaluate
from couponim.streaming import stream_ssr_greedy
from tests.test_rr import fixtures


def materialized(graph, alpha, discard, weights, k, theta, root_seed, component_seeds):
    rng = random.Random(root_seed)
    roots = [rng.randrange(graph.n) for _ in range(theta)]
    sampler = CouponRR(graph, alpha, discard, weights)
    indices = []
    for seed in component_seeds:
        rng = random.Random(seed)
        index = {}
        for t, root in enumerate(roots):
            for v in sampler.sample(root, rng):
                index.setdefault(v, []).append(t)
        indices.append(index)
    return SSRBank(graph.n, k, theta, indices, np.bincount(roots, minlength=graph.n),
                   0, sampler.stats(), 0)


class StreamingTests(unittest.TestCase):
    def test_identical_worlds_match_full_bank(self):
        for graph, alpha, discard, weights in fixtures():
            for seed in (19, 47):
                with self.subTest(graph=graph.name, seed=seed):
                    components = [seed * 1000 + j for j in range(graph.n)]
                    bank = materialized(graph, alpha, discard, weights, graph.n, 137, seed, components)
                    expected, info = bank.greedy()
                    actual, diagnostics = stream_ssr_greedy(CouponRR(graph, alpha, discard, weights),
                                                            graph.n, 137, seed, components)
                    self.assertEqual(actual, expected)
                    self.assertEqual(diagnostics["selected_marginal_counts"], info["selected_marginal_counts"])
                    self.assertEqual(diagnostics["stats"], bank.stats)
                    self.assertEqual(diagnostics["stats"]["calls"], 137 * graph.n)
                    self.assertEqual(diagnostics["training_estimate_not_quality"], bank.estimate(actual))

    def test_forward_celf_matches_plain_greedy(self):
        rng = random.Random(123)
        for _ in range(20):
            bank = [np.asarray(rng.sample(range(50), rng.randrange(20)), dtype=np.int64) for _ in range(12)]
            chosen, covered = [], set()
            for _ in range(6):
                best = min((v for v in range(12) if v not in chosen),
                           key=lambda v: (-len(set(bank[v])-covered), v))
                chosen.append(best)
                covered.update(bank[best])
            actual, count, _ = select_coverage(bank, 6)
            self.assertEqual(actual, chosen)
            self.assertEqual(count, len(covered))


@unittest.skipUnless(os.environ.get("COUPONIM_TEST_CPP") == "1", "Set COUPONIM_TEST_CPP=1 for native equivalence")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="couponim-native-")
        cls.root = Path(cls.folder.name)
        cls.build_dir = cls.root / "build"
        native.build(cls.build_dir)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_exact_cpp_python_streams(self):
        for graph, alpha, discard, weights in fixtures():
            for seed in (19, 47):
                with self.subTest(graph=graph.name, seed=seed):
                    work = self.root / f"{graph.name}-{seed}"
                    model = native.NativeModel(graph, alpha, discard, self.build_dir, work)
                    # RR supports arbitrary normalized edge intervals; the forward CLI is uniform-neighbor.
                    native.write_graph(model.graphfile, CouponRR(graph, alpha, discard, weights))
                    components = [seed*1000+j for j in range(graph.n)]
                    expected, info = stream_ssr_greedy(CouponRR(graph, alpha, discard, weights),
                                                       graph.n, 137, seed, components)
                    actual, result = model.stream(graph.n, 137, seed, components)
                    self.assertEqual(actual, expected)
                    self.assertEqual(result["selected_marginal_counts"], info["selected_marginal_counts"])
                    self.assertEqual(result["stats"], info["stats"])
                    self.assertEqual(result["covered_count"], info["covered_count"])

    def test_forward_and_evaluation_streams_match(self):
        for graph, alpha, discard, _ in fixtures():
            with self.subTest(graph=graph.name):
                model = native.NativeModel(graph, alpha, discard, self.build_dir, self.root / (graph.name+"-forward"))
                python_bank = sample_coverage(graph, alpha, discard, 151, 43)
                cpp_bank = model.forward_bank(151, 43)
                for expected, actual in zip(python_bank, cpp_bank):
                    np.testing.assert_array_equal(actual, expected)
                seeds = list(range(graph.n))
                expected = evaluate(graph, seeds, alpha, discard, 500, 711)
                actual = model.evaluate(seeds, 500, 711)
                np.testing.assert_allclose(
                    [actual["spread"], actual["within_run_ci95_halfwidth"],
                     actual["evaluation_variance"], actual["consumed_coupons"]], expected, atol=1e-13, rtol=0)
