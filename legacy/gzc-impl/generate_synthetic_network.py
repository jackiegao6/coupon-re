import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle
import os

def create_your_genius_graph(n=2000, out_dir="/home/wen/work/coup_spread/dataset/"):
    # 按照你的思路划分节点比例
    n_hubs = int(n * 0.05)     # 5% 的超级大V (用来坑 PageRank)
    n_sinks = int(n * 0.05)    # 5% 的孤岛节点 (用来坑 Alpha_sort)
    n_normal = n - n_hubs - n_sinks # 90% 的普通节点 (RIS 的核反应堆)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # 1. 构造普通节点子图 (0 到 n_normal-1)
    # 使用偏好依附(Barabasi-Albert)生成，保证内部有连通性和度数差异
    ba_graph = nx.barabasi_albert_graph(n_normal, 3)
    G.add_edges_from(ba_graph.edges())
    # 变成双向，让券能在内部反复游走
    G.add_edges_from([(v, u) for u, v in ba_graph.edges()])

    # 2. 构造孤岛节点 (n_normal 到 n_normal + n_sinks - 1)
    # 他们是绝对孤立的，没有任何边！(出度=0，入度=0)

    # 3. 构造超级大V (最后 n_hubs 个)
    # 他们具有全网极高的入度（吸引PageRank），也有通向全网的出度
    hub_start = n_normal + n_sinks
    for i in range(hub_start, n):
        # 连向社区的很多人
        targets = np.random.choice(n_normal, int(n_normal * 0.1), replace=False)
        for t in targets:
            G.add_edge(i, t)
            G.add_edge(t, i) # 社区的人也仰望大V，连向大V

    # 转换保存
    adj = nx.to_scipy_sparse_array(G, format='csr')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"2-genius_graph_{n}-adj.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ 天才图纸生成完毕！保存在 {out_path}")

if __name__ == "__main__":
    create_your_genius_graph(2000)