import random
import unittest

import numpy as np

from forward_coverage_baselines import sample_coverage, select_coverage, forward_coverage_seeds
from test_coupon_core import make_graph, exact_adoption_matrix, exact_spread


class ForwardCoverageTests(unittest.TestCase):
    def test_lazy_matches_exhaustive_greedy_on_fixed_banks(self):
        rng = random.Random(123)
        for _ in range(25):
            bank = [np.asarray(rng.sample(range(50), rng.randrange(20))) for _ in range(12)]
            selected, covered, remaining = [], set(), set(range(len(bank)))
            for _ in range(6):
                best = max(remaining, key=lambda i: (len(set(bank[i]) - covered), -i))
                selected.append(best)
                covered.update(bank[best])
                remaining.remove(best)
            actual, count, _ = select_coverage(bank, 6)
            self.assertEqual(actual, selected)
            self.assertEqual(count, len(covered))

    def test_fixed_allocation_bank_matches_exact_with_repeated_slot(self):
        graph = make_graph([[1], [0]])
        alpha, discard = np.asarray([0.3, 0.6]), np.asarray([0.2, 0.1])
        bank = sample_coverage(graph, alpha, discard, 40000, 42, 2)
        observed = len(set(bank[0]) | set(bank[1])) / 40000
        expected = exact_spread(exact_adoption_matrix(graph, alpha, discard), [0, 0])
        self.assertAlmostEqual(observed, expected, delta=0.02)

    def test_stochastic_full_candidate_limit_matches_greedy(self):
        bank = [np.asarray([0, 1]), np.asarray([1, 2]), np.asarray([3])]
        self.assertEqual(select_coverage(bank, 2)[:2],
                         select_coverage(bank, 2, "stochastic", 1e-6, 42)[:2])

    def test_capacity_is_respected(self):
        graph = make_graph([[], []])
        seeds, stats = forward_coverage_seeds(
            graph, np.asarray([1., 0.]), np.asarray([0., 1.]),
            3, 10, 42, capacity_per_node=2)
        self.assertEqual(len(seeds), 3)
        self.assertLessEqual(max(seeds.count(i) for i in seeds), 2)
        self.assertEqual(stats["training_estimate_not_quality"], 1.0)


if __name__ == "__main__":
    unittest.main()
