import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from couponim.graph import Graph, load_edge_list, load_graph
from couponim.parameters import node_probabilities
from couponim.runner import read, run, write


class GraphTests(unittest.TestCase):
    def test_edge_list_directed_undirected_and_isolates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "edges.txt"
            path.write_text("# small generated fixture\n0 1\n0 1\n1 1\n", encoding="utf-8")
            directed = load_edge_list(path, directed=True, n=4)
            undirected = load_edge_list(path, directed=False, n=4)
            self.assertEqual(directed.m, 2)
            self.assertEqual(undirected.m, 3)
            self.assertEqual(directed.degrees.tolist(), [1, 1, 0, 0])
            np.savez(Path(folder) / "graph.npz", indptr=directed.indptr, indices=directed.indices)
            restored = load_graph({"format": "csr-npz", "path": "graph.npz"}, folder)
            np.testing.assert_array_equal(restored.indices, directed.indices)
        with self.assertRaises(ValueError):
            Graph.from_csr([0, 2], [0])

    def test_high_variance_balanced_and_sink_handling(self):
        graph = Graph.from_rows([[1], [0, 2], [0, 1, 3], [0]])
        a0, _, t0 = node_probabilities(graph, "balanced-reference")
        a, _, t = node_probabilities(graph, "balanced")
        self.assertAlmostEqual(a.mean(), a0.mean())
        self.assertAlmostEqual(a.var(), 9*a0.var())
        np.testing.assert_allclose(t, t0)
        sink = Graph.from_rows([[1], []])
        for scenario in ("balanced-reference", "balanced", "adoption-heavy", "forwarding-heavy"):
            a, d, t = node_probabilities(sink, scenario)
            self.assertEqual(t[1], 0)
            self.assertEqual(a[1]+d[1], 1)


class WorkflowTests(unittest.TestCase):
    def test_parallel_resume_summary_and_protocol_guard(self):
        with tempfile.TemporaryDirectory(prefix="couponim-workflow-") as folder:
            folder = Path(folder)
            config = {"graphs": [{"format": "synthetic", "n": 8}], "budgets": [2],
                      "theta_factors": [1, 2], "methods": ["CIM-RIS", "Random"],
                      "evaluation_trials": 30, "repeats": 2, "forward_batches": 8}
            path, output = folder / "config.json", folder / "results"
            write(path, config)
            self.assertEqual(run(path, output, workers=2), 0)
            jobs = list((output / "jobs").glob("*.json"))
            self.assertEqual(len(jobs), 6)
            before = {p.name: p.read_bytes() for p in jobs}
            with (output / "raw.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertLessEqual(float(row["spread"]), float(row["consumed_coupons"])+1e-12)
            with (output / "summary.csv").open(newline="") as stream:
                groups = list(csv.DictReader(stream))
            self.assertEqual(len(groups), 3)
            self.assertTrue(all(int(row["runs"]) == 2 for row in groups))
            self.assertEqual(run(path, output, workers=1), 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in jobs})
            # A lost evaluation reuses the stored selection, not resampling seeds.
            missing = jobs[0]
            selection = output / "work" / missing.stem / "selection.json"
            selection_before = selection.read_bytes()
            missing.unlink()
            self.assertEqual(run(path, output, workers=1), 0)
            self.assertEqual(selection.read_bytes(), selection_before)
            config["evaluation_trials"] = 31
            write(path, config)
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                run(path, output)
            self.assertFalse((output / "RUNNING.json").exists())

    def test_stop_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write(folder / "config.json", {"graphs": [{"format": "synthetic", "n": 8}]})
            output = folder / "results"
            output.mkdir()
            (output / "STOP").touch()
            self.assertEqual(run(folder / "config.json", output), 130)
            self.assertFalse((output / "jobs").exists())
            self.assertFalse((output / "RUNNING.json").exists())
