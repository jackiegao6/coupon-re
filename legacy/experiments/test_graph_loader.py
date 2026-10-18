"""Regress the real NumPy-2 pickle failure without monkey-patching modules."""
import sys
import unittest
import numpy as np
import scipy.sparse as sparse
import run_real_submission as core


class GraphLoaderTests(unittest.TestCase):
    def test_real_graph_load_preserves_numpy_and_scipy_modules(self):
        before = {name: module for name, module in sys.modules.items() if name.startswith(('numpy', 'scipy'))}
        graph = core.load_graph('Netscience')
        self.assertEqual((graph.n, graph.m), (379, 1828))
        self.assertEqual(int(graph.degrees.sum()), graph.m)
        for name, module in before.items():
            self.assertIs(sys.modules[name], module, name)
        np.testing.assert_array_equal(sparse.eye(3).tocsr().toarray(), np.eye(3))


if __name__ == '__main__':
    unittest.main()
