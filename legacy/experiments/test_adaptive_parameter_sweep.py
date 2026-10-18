"""Focused tests for the v2.5 adaptive-sampling runner."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

import run_adaptive_parameter_sweep as adaptive
import run_real_submission as core


def tiny_graph() -> core.Graph:
    return core.Graph(
        name="tiny",
        n=3,
        m=0,
        indices=np.asarray([], dtype=np.int32),
        indptr=np.asarray([0, 0, 0, 0], dtype=np.int64),
        in_neighbors=[[], [], []],
        degrees=np.asarray([0, 0, 0], dtype=np.int32),
    )


class AdaptiveSweepTests(unittest.TestCase):
    def test_expected_grid_has_300_resumable_jobs(self) -> None:
        jobs = adaptive.expected_jobs()
        self.assertEqual(len(jobs), 300)
        self.assertEqual(len(set(jobs)), len(jobs))

    def test_job_path_is_unique_per_budget_and_selection_seed(self) -> None:
        root = Path("/tmp/results")
        first = adaptive.job_path(root, "Netscience", 0.3, 0.6, 25, 20260715)
        second = adaptive.job_path(root, "Netscience", 0.3, 0.6, 100, 20260715)
        third = adaptive.job_path(root, "Netscience", 0.3, 0.6, 25, 20260716)
        self.assertEqual(len({first, second, third}), 3)

    def test_adaptive_stages_use_one_nested_rr_stream(self) -> None:
        calls: list[int] = []
        pool_seeds: list[int] = []

        class FakePool:
            def __init__(self, *args: object) -> None:
                pool_seeds.append(int(args[4]))

            def extend_and_select(
                self, samples: int
            ) -> tuple[list[int], float, float, float, int]:
                calls.append(samples)
                return [samples % 3], 0.01, 0.01 * len(calls), 1.0, samples

        changes = iter([(1.0, 1.1, 10.0), (1.0, 1.0, 0.0)])
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(adaptive, "INITIAL_SAMPLES", 4))
            stack.enter_context(mock.patch.object(adaptive, "HARD_CAP", 20))
            stack.enter_context(
                mock.patch.object(adaptive, "sample_floor", return_value=(10, 9.2))
            )
            stack.enter_context(
                mock.patch.object(adaptive, "IncrementalCIMSamplePool", FakePool)
            )
            stack.enter_context(
                mock.patch.object(adaptive, "validation_means", side_effect=changes)
            )
            stack.enter_context(
                mock.patch.object(adaptive, "VALIDATION_SIMULATIONS", 2)
            )
            _, stages, floor, hard_cap, _ = adaptive.adaptive_cim_selection(
                tiny_graph(),
                np.asarray([0.2, 0.2, 0.2]),
                np.asarray([0.8, 0.8, 0.8]),
                "tiny",
                0.3,
                0.6,
                1,
                7,
            )

        self.assertEqual(floor, 10)
        self.assertFalse(hard_cap)
        self.assertEqual([stage["rr_samples"] for stage in stages], [4, 8, 10])
        self.assertEqual(calls, [4, 8, 10])
        self.assertEqual(len(pool_seeds), 1)
        self.assertEqual(len({stage["rr_stream_id"] for stage in stages}), 1)
        self.assertTrue(stages[-1]["stable"])
        self.assertNotEqual(
            stages[-1]["validation_stream_id"], stages[-1]["rr_stream_id"]
        )

    def test_incremental_pool_preserves_existing_sample_memberships(self) -> None:
        graph = tiny_graph()
        alpha = np.asarray([0.7, 0.4, 0.2])
        discard = 1.0 - alpha
        pool = adaptive.IncrementalCIMSamplePool(graph, alpha, discard, 2, 91)
        pool.extend(20)
        before = [
            {node: list(sample_ids) for node, sample_ids in index.items()}
            for index in pool.coverage
        ]
        pool.extend(40)
        for coupon_index, coverage in enumerate(before):
            for node, sample_ids in coverage.items():
                self.assertEqual(
                    [value for value in pool.coverage[coupon_index][node] if value < 20],
                    sample_ids,
                )

    def test_atomic_json_replaces_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "job.json"
            adaptive.write_json_atomic(path, {"value": 1})
            adaptive.write_json_atomic(path, {"value": 2})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "value": 2\n}')
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
