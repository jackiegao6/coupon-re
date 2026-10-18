"""Convert user-supplied edge lists or MATLAB adjacency to portable CSR NPZ."""
from pathlib import Path
import numpy as np
from .graph import Graph, load_edge_list


def convert_graph(source, target, input_format="edge-list", directed=True, n=None, matrix_key="network"):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target == source or target.exists():
        raise ValueError("Choose a new output NPZ file")
    if target.suffix.lower() != ".npz":
        raise ValueError("Output must have the .npz extension")
    if input_format == "edge-list":
        graph = load_edge_list(source, directed=directed, n=n)
    elif input_format == "mat":
        try:
            from scipy.io import loadmat
            from scipy.sparse import csr_matrix
        except ImportError:
            raise RuntimeError("MAT conversion requires scipy: pip install '.[data]'") from None
        arrays = loadmat(source)
        if matrix_key not in arrays:
            raise ValueError(f"MAT has no variable {matrix_key!r}")
        adjacency = csr_matrix(arrays[matrix_key])
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("MAT adjacency must be square")
        adjacency.eliminate_zeros()
        if not np.all(adjacency.data == 1):
            raise ValueError("Only unweighted binary adjacency is supported")
        if not directed:
            adjacency = adjacency.maximum(adjacency.T).tocsr()
        adjacency.sort_indices()
        graph = Graph.from_csr(adjacency.indptr, adjacency.indices, source.stem)
        if n is not None and n != graph.n:
            raise ValueError("Explicit n does not match MAT adjacency")
    else:
        raise ValueError("input_format must be edge-list or mat")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, indptr=graph.indptr, indices=graph.indices,
                        shape=np.array([graph.n, graph.n]))
    return {"n": graph.n, "m_directed_arcs": graph.m, "path": str(target)}
