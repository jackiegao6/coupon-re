"""Portable, unweighted directed CSR graphs; no dataset-specific paths."""
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


@dataclass
class Graph:
    name: str
    n: int
    m: int
    indices: np.ndarray
    indptr: np.ndarray
    degrees: np.ndarray

    def out_neighbors(self, node):
        return self.indices[self.indptr[node]:self.indptr[node + 1]]

    @cached_property
    def in_neighbors(self):
        """Construct incoming lists only for algorithms that need them."""
        rows = [[] for _ in range(self.n)]
        for u in range(self.n):
            for v in self.out_neighbors(u):
                rows[v].append(u)
        return rows

    @classmethod
    def from_csr(cls, indptr, indices, name="graph"):
        ptr = np.asarray(indptr)
        idx = np.asarray(indices)
        if (ptr.ndim != 1 or idx.ndim != 1 or len(ptr) < 2
                or ptr.dtype.kind not in "iu" or (len(idx) and idx.dtype.kind not in "iu")):
            raise ValueError("CSR requires one-dimensional integer arrays and at least one node")
        n = len(ptr) - 1
        if n >= 2**31 or len(idx) >= 2**32:
            raise ValueError("Graph exceeds the native format's 32-bit dimensions")
        ptr = ptr.astype(np.int64, copy=True)
        if ptr[0] != 0 or ptr[-1] != len(idx) or np.any(np.diff(ptr) < 0):
            raise ValueError("Invalid CSR offsets")
        if len(idx) and (idx.min() < 0 or idx.max() >= n):
            raise ValueError("Neighbor ID outside [0,n)")
        idx = idx.astype(np.int32, copy=True)
        for u in range(n):
            row = idx[ptr[u]:ptr[u+1]]
            if len(np.unique(row)) != len(row):
                raise ValueError("Duplicate outgoing arcs are not supported")
        return cls(name, n, len(idx), idx, ptr, np.diff(ptr).astype(np.int32))

    @classmethod
    def from_rows(cls, rows, name="graph"):
        rows = [list(row) for row in rows]
        return cls.from_csr(np.r_[0, np.cumsum([len(row) for row in rows])],
                            np.asarray([v for row in rows for v in row], dtype=np.int64), name)


def load_edge_list(path, *, directed, n=None):
    """Read zero-based integer pairs, preserving isolates when n is supplied."""
    path = Path(path)
    edges = set()
    maximum = -1
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path.name}:{line_number}: expected exactly two integer IDs")
            u, v = map(int, parts)
            if min(u, v) < 0:
                raise ValueError("Node IDs must be nonnegative")
            edges.add((u, v))
            if not directed:
                edges.add((v, u))
            maximum = max(maximum, u, v)
    n = maximum + 1 if n is None else n
    if not isinstance(n, int) or not 1 <= n < 2**31 or maximum >= n:
        raise ValueError("Provide n >= 1 covering all node IDs (also needed for empty graphs)")
    rows = [[] for _ in range(n)]
    for u, v in sorted(edges):
        rows[u].append(v)
    return Graph.from_rows(rows, path.stem)


def load_graph(spec, base):
    kind = spec.get("format", "edge-list")
    if kind == "synthetic":
        n = spec.get("n", 24)
        if not isinstance(n, int) or n < 4:
            raise ValueError("Synthetic smoke graph needs n >= 4")
        # Generated at runtime, not a bundled benchmark dataset.
        topology = spec.get("topology", "ring-chords")
        if topology == "ring-chords":
            rows = [sorted({(i+1) % n, (i+3) % n}) for i in range(n)]
        elif topology == "variable-degree":
            rows = [[(i+j+1) % n for j in range(1+i % min(n-1, 5))] for i in range(n)]
        else:
            raise ValueError("Synthetic topology must be ring-chords or variable-degree")
        if not isinstance(spec.get("sink", True), bool):
            raise ValueError("Synthetic sink must be a boolean")
        if spec.get("sink", True):
            rows[-1] = []
        return Graph.from_rows(rows, spec.get("name", "synthetic-smoke"))
    path = Path(base) / spec["path"]
    if kind == "edge-list":
        if not isinstance(spec.get("directed"), bool):
            raise ValueError("edge-list requires an explicit directed: true/false")
        return load_edge_list(path, directed=spec["directed"], n=spec.get("n"))
    if kind == "csr-npz":
        with np.load(path, allow_pickle=False) as data:
            graph = Graph.from_csr(data["indptr"], data["indices"], path.stem)
            if "shape" in data and tuple(data["shape"]) != (graph.n, graph.n):
                raise ValueError("CSR adjacency must be square")
            if "data" in data and not np.all(data["data"] == 1):
                raise ValueError("Only binary adjacency is accepted; edge weights are not inferred")
            return graph
    raise ValueError(f"Unsupported graph format: {kind}")
