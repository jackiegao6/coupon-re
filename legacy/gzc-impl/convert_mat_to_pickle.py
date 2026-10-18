from scipy.io import loadmat
from scipy import sparse
import pickle
from pathlib import Path

def convert_mat_to_pickle(
    mat_path: str,
    mat_key: str,
    out_path: str = None
):
    mat_path = Path(mat_path)

    if out_path is None:
        out_path = mat_path.with_suffix(f".{mat_key}-adj.pkl")
    else:
        out_path = Path(out_path)

    # 1. 读取 mat
    data = loadmat(mat_path)
    if mat_key not in data:
        raise KeyError(f"{mat_key} 不在 {mat_path.name} 中")

    adj = data[mat_key]

    # 2. 统一成 CSR
    if sparse.issparse(adj):
        adj = adj.tocsr()
    else:
        adj = sparse.csr_matrix(adj)

    # 3. 存 pickle
    with out_path.open("wb") as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ 转换完成")
    print(f"   source: {mat_path}")
    print(f"   key: {mat_key}")
    print(f"   target: {out_path}")
    print(f"   shape: {adj.shape}, nnz: {adj.nnz}")

if __name__ == "__main__":
    convert_mat_to_pickle(
        mat_path="/home/wen/work/coup_spread/dataset/network/network.mat",
        mat_key="EmailEnron"
    )
