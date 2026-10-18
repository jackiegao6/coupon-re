"""Verify local MAT-to-CSR correspondence without claiming download provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import run_real_submission as core

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"Netscience": "netscience", "NetFacebookEgo": "netfacebookego", "DoubanRandom": "doubanrandom", "EmailEnron": "EmailEnron", "network.douban": "douban"}


def main():
    mat_path = ROOT / "dataset/network/network.mat"
    data = loadmat(mat_path, variable_names=list(KEYS.values()))
    rows = []
    for dataset, key in KEYS.items():
        path = core.DATASETS[dataset]
        state = core._load_raw_csr(path)
        observed = csr_matrix((state.data, state.indices, state.indptr), shape=state._shape)
        reference = data[key].tocsr()
        equal = observed.shape == reference.shape and (observed != reference).nnz == 0
        assert equal, dataset
        graph = core.load_graph(dataset)
        assert np.array_equal(reference.indices, graph.indices) and np.array_equal(reference.indptr, graph.indptr), dataset
        assert reference.nnz == graph.m and not np.any(reference.data == 0)
        rows.append({"dataset": dataset, "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mat_variable": key, "mat_values_equal": equal, "csr_indices_and_indptr_equal": True, "nodes": graph.n, "directed_edges": graph.m, "reciprocal": (reference != reference.T).nnz == 0, "self_loops": int(np.count_nonzero(reference.diagonal())), "nonunit_weights": int(np.count_nonzero(reference.data != 1)), "degree_one_nodes": int(np.count_nonzero(graph.degrees == 1)), "original_download_url": None, "original_download_sha256": None, "preprocessing_before_mat": "UNVERIFIED", "node_mapping": "NOT_FOUND", "provenance_status": "LOCAL_MAT_CORRESPONDENCE_ONLY"})
    output = ROOT / "experiments/results/evidence-20260913/provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source_mat": str(mat_path.relative_to(ROOT)), "source_mat_sha256": hashlib.sha256(mat_path.read_bytes()).hexdigest(), "conversion_script": "gzc-impl/convert_mat_to_pickle.py", "conversion_scope": "loadmat(key), convert to CSR, pickle; historical command execution not independently established", "datasets": rows, "limitation": "Equality to the local MAT bundle does not authenticate original publisher, sampling, cleaning, or node IDs."}, indent=2), encoding="utf-8")
    print('Verified five local MAT/CSR correspondences; upstream provenance remains unverified.')


if __name__ == '__main__':
    main()
