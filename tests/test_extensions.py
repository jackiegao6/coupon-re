import csv
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from couponim.adaptive import IncrementalCIMSamplePool, adaptive_select
from couponim.conditioned import cim_ris_seeds
from couponim.convert import convert_graph
from couponim.graph import load_graph
from couponim.diagnostics import estimate_fixed_assignment, graph_statistics, run_diagnostic
from couponim.exact import consumption_matrix, diagnose, greedy, spread
from couponim.graph import Graph
from couponim.parameters import node_probabilities
from couponim.reevaluate import reevaluate
from couponim.runner import prepare, read, run_job, stable, write
from couponim.selection import METHODS, select
from couponim.rr import CouponRR, sample_ssr


class AdditionalAlgorithmsTests(unittest.TestCase):
    def test_cycle_exact_dp_and_enumeration(self):
        # Two possible steps followed by a revisit: no infinite Markov series.
        g = Graph.from_rows([[1], [0]])
        a, d = np.array([.2, .4]), np.array([.1, .1])
        q = consumption_matrix(g, a, d)
        np.testing.assert_allclose(q, [[.2, .28], [.1, .4]])
        seeds, gains = greedy(q, 2)
        self.assertAlmostEqual(sum(gains), spread(q, seeds))
        result = diagnose(g, a, d, 1)
        self.assertAlmostEqual(result["enumeration"]["optimal_spread"], .5)
        with self.assertRaises(ValueError):
            consumption_matrix(g, a, d, max_nodes=1)

    def test_conditioned_pool_is_nested_and_degenerate_cases(self):
        g = Graph.from_rows([[1], [2], [0]])
        a, d, _ = node_probabilities(g)
        partial = IncrementalCIMSamplePool(g, a, d, 2, 123)
        partial.extend(17)
        prefix = [{v: list(ids) for v, ids in col.items()} for col in partial.coverage]
        partial.extend(43)
        full = IncrementalCIMSamplePool(g, a, d, 2, 123)
        full.extend(43)
        self.assertEqual(partial.coverage, full.coverage)
        self.assertEqual(partial.select(), full.select())
        for j, col in enumerate(prefix):
            for v, ids in col.items():
                self.assertEqual([t for t in partial.coverage[j][v] if t < 17], ids)
        zeros = np.zeros(3)
        chosen, _, estimate, _ = cim_ris_seeds(g, zeros, np.ones(3), 2, 50, 8)
        self.assertEqual(chosen, [0, 1])
        self.assertEqual(estimate, 0)
        chosen, info = adaptive_select(g, a, d, 2, 17, initial_samples=8,
                minimum_samples=16, maximum_samples=32, validation_trials=20)
        self.assertEqual(len(set(chosen)), 2)
        self.assertLessEqual(info["final_samples"], 32)

    def test_sampler_estimates_match_fixed_assignment_truth(self):
        g = Graph.from_rows([[1], [2], [0]])
        a, d = np.array([.2, .5, .8]), np.array([.1, .2, .1])
        truth = spread(consumption_matrix(g, a, d), [0, 1])
        for conditioned in (False, True):
            result = estimate_fixed_assignment(g, a, d, [0, 1], 10000, 73, conditioned)
            self.assertLess(abs(result["estimate"]-truth), .05)
        result = estimate_fixed_assignment(g, np.zeros(3), np.ones(3), [0], 3, 1, True)
        self.assertEqual(result["estimate"], 0)

    def test_custom_behavior_variance_and_permutation(self):
        g = Graph.from_rows([[1], [0, 2], [0, 1, 3], [0]])
        a, _, _ = node_probabilities(g, "balanced-reference")
        enlarged, _, t = node_probabilities(g, {"kind": "heterogeneity", "scale": 3})
        shuffled, _, _ = node_probabilities(g, {"kind": "heterogeneity", "scale": 3,
                                                "association": "permuted", "permutation_seed": 5})
        self.assertAlmostEqual(enlarged.mean(), a.mean())
        self.assertAlmostEqual(enlarged.var(), a.var()*9)
        np.testing.assert_allclose(np.sort(shuffled), np.sort(enlarged))
        np.testing.assert_allclose(t, .5)
        _, _, t = node_probabilities(g, {"kind": "transfer-share", "transfer": .85, "share": .4})
        np.testing.assert_allclose(t, .85)
        with self.assertRaises(ValueError):
            node_probabilities(g, {"kind": "constant", "alpha": .9, "discard": .4})

    def test_all_methods_and_matrix_guard(self):
        g = Graph.from_rows([[1, 2], [2], [3], [0]])
        a, d, t = node_probabilities(g)
        opts = {"adaptive_initial_samples": 8, "adaptive_minimum_samples": 16,
                "adaptive_maximum_samples": 32, "adaptive_validation_trials": 10}
        for method in METHODS:
            with self.subTest(method=method):
                seeds, info = select(g, a, d, t, method, 2, 3, 53, 32, opts, stable)
                self.assertEqual(len(set(seeds)), 2)
                self.assertTrue(all(0 <= v < g.n for v in seeds))
        with self.assertRaisesRegex(ValueError, "matrix_max_bytes"):
            select(g, a, d, t, "MC-Greedy", 2, 3, 53, 32, {"matrix_max_bytes": 1}, stable)

    def test_reevaluation_keeps_selection_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root/"config.json"
            write(cfg, {"graphs": [{"format": "synthetic", "n": 8}], "budgets": [2],
                        "methods": ["CIM-Conditioned", "Exact-Greedy"], "evaluation_trials": 10,
                        "scenarios": [{"name": "custom", "kind": "constant", "alpha": .4, "discard": .3}]})
            protocol = prepare(cfg, root/"old", "python", root/"build")
            for job in protocol["jobs"]:
                run_job(root/"old", job["id"], root/"build")
            before = {p.name: p.read_bytes() for p in (root/"old"/"jobs").glob("*.json")}
            self.assertEqual(reevaluate(root/"old", root/"new", trials=25), 2)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root/"old"/"jobs").glob("*.json")})
            for job in protocol["jobs"]:
                record = read(root/"new"/"jobs"/f'{job["id"]}.json')
                self.assertEqual(record["seeds"], read(root/"old"/"jobs"/f'{job["id"]}.json')["seeds"])
                self.assertEqual(record["trials"], 25)
            self.assertEqual(reevaluate(root/"old", root/"new", trials=25), 2)
            with self.assertRaises(ValueError):
                reevaluate(root/"old", root/"new", trials=30)

    def test_diagnostics_write_only_json_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/"config.json"
            write(config, {"graphs": [{"format": "synthetic", "n": 6}], "budgets": [2],
                           "theta_factors": [1, 2], "repeats": 2})
            for mode in ("inspect", "diagnose", "sampler-ablation"):
                run_diagnostic(config, root/mode, mode)
                self.assertTrue((root/mode/"diagnostic.json").exists())
            with (root/"sampler-ablation"/"sampler-ablation.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            stats = graph_statistics(Graph.from_rows([[1], [0], []]))
            self.assertEqual(stats["isolated_nodes"], 1)
            self.assertEqual(stats["reciprocal_arc_fraction"], 1)


    def test_converter_and_new_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            edges, target = root/"edges.txt", root/"graph.npz"
            edges.write_text("0 1\n1 2\n", encoding="utf-8")
            info = convert_graph(edges, target, directed=False, n=4)
            self.assertEqual((info["n"], info["m_directed_arcs"]), (4, 4))
            restored = load_graph({"format": "csr-npz", "path": str(target)}, root)
            self.assertEqual(restored.degrees.tolist(), [1, 2, 1, 0])
            with self.assertRaises(ValueError):
                convert_graph(edges, target)


    def test_all_bundled_generated_graph_configs_validate(self):
        examples = Path(__file__).resolve().parents[1]/"examples"
        with tempfile.TemporaryDirectory() as tmp:
            for config in examples.glob("*.json"):
                if config.name == "external-graph.template.json":
                    continue
                with self.subTest(config=config.name):
                    prepare(config, Path(tmp)/config.stem, "python", Path(tmp)/"build")
