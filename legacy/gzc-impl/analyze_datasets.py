import pickle
import scipy.sparse as sp
import numpy as np
import networkx as nx
import os

def analyze_pkl_datasets(data_dir):
    print(f"{'数据集名称':<25} | {'节点数 (N)':<10} | {'边数 (E)':<10} | {'平均度 (Avg Deg)':<15} | {'最大出度 (Max Out)':<15} | {'图密度 (Density)'}")
    print("-" * 100)
    
    for file in os.listdir(data_dir):
        if file.endswith("-adj.pkl"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'rb') as f:
                adj = pickle.load(f)
                
            if not sp.issparse(adj):
                adj = sp.csr_matrix(adj)
                
            n = adj.shape[0]
            e = adj.nnz
            
            # 计算平均度
            avg_degree = e / n
            
            # 计算最大出度 (如果是 csr_matrix，行和就是出度)
            out_degrees = np.array(adj.sum(axis=1)).flatten()
            max_out_degree = out_degrees.max()
            
            # 计算图密度 Density = E / (N * (N-1))
            density = e / (n * (n - 1)) if n > 1 else 0
            
            dataset_name = file.replace("-adj.pkl", "")
            
            print(f"{dataset_name:<25} | {n:<12} | {e:<12} | {avg_degree:<18.2f} | {max_out_degree:<18.0f} | {density:.6f}")

def analyze_advanced_pkl_datasets(data_dir):
    print(f"{'数据集名称':<25} | {'节点数':<8} | {'边数':<8} | {'平均度':<8} | {'最大出度':<8} | {'稀疏度':<10} | {'平均聚类系数'}")
    print("-" * 105)
    
    for file in os.listdir(data_dir):
        if file.endswith("-adj.pkl"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'rb') as f:
                adj = pickle.load(f)
                
            if not sp.issparse(adj):
                adj = sp.csr_matrix(adj)
                
            n = adj.shape[0]
            e = adj.nnz
            
            avg_degree = e / n
            out_degrees = np.array(adj.sum(axis=1)).flatten()
            max_out_degree = out_degrees.max()
            
            # 计算密度和稀疏度
            density = e / (n * (n - 1)) if n > 1 else 0
            sparsity = (1.0 - density) * 100 # 换算成百分比
            
            # 计算聚类系数 (使用 NetworkX)
            # 将 scipy 稀疏矩阵转为 nx 图
            G = nx.from_scipy_sparse_array(adj)
            # 如果图是无向的，转为无向图计算聚类系数更准
            if nx.is_directed(G):
                G = G.to_undirected()
            
            # 聚类系数计算 (对于几千个节点的图只需一两秒)
            clustering_coeff = nx.average_clustering(G)
            
            dataset_name = file.replace("-adj.pkl", "")
            
            print(f"{dataset_name:<25} | {n:<10} | {e:<10} | {avg_degree:<10.2f} | {max_out_degree:<10.0f} | {sparsity:>6.2f}%   | {clustering_coeff:.4f}")

if __name__ == "__main__":
    # 替换为你存放 pkl 文件的真实路径
    data_dir = "/home/wen/work/coup_spread/dataset/network/" 
    analyze_advanced_pkl_datasets(data_dir)