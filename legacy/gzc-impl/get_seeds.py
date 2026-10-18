import logging
import numpy as np
import networkx as nx
import random
import scipy.sparse as sp
from typing import Dict, List, Any
import time

def deliverers_monteCarlo_greedy_standard(
        n: int, 
        m: int, 
        tranProMatrix: np.ndarray, 
        succ_distribution: np.ndarray, 
        dis_distribution: np.ndarray, 
        simulation_algo_func, # 传入 AgainContinue 那个函数
        simulation_times: int = 100 # 运行一百次
) -> list:
    """
    标准贪心蒙特卡洛策略 (Naive Greedy with Monte Carlo)
    """
    logging.info(f"--- Running: Standard Greedy Monte Carlo (Sims={simulation_times}) ---")
    
    selected_seeds = []
    current_spread = 0.0 # 当前种子集能覆盖的平均人数
    
    # 候选节点集合 (初始为所有节点)
    candidate_nodes = list(range(n))
    
    # --- 第一层循环：我们要选 m 个种子 ---
    for k in range(m):
        start_time = time.time()
        best_node = -1
        max_marginal_gain = -1.0
        
        # --- 第二层循环：遍历所有剩下的节点，看把谁加入集合最好 ---
        # 这一步非常慢！如果是 4000 个节点，第1轮要跑 4000 次评估
        for node in candidate_nodes:
            
            # 构造临时种子集 S_t + {node}
            temp_seeds = selected_seeds + [node]
            
            # --- 第三层循环：运行 100 次蒙特卡洛，求平均覆盖数 ---
            total_activated = 0
            for _ in range(simulation_times):
                # 运行一次模拟
                res_vector = simulation_algo_func(
                    tranProMatrix, 
                    temp_seeds, # 传入合并后的种子集
                    succ_distribution, 
                    dis_distribution, 
                    None # constantFactor 暂时不用
                )
                # 统计激活总数 (注意：这里要根据你的模拟函数返回是 0/1 向量还是列表来调整)
                total_activated += np.sum(res_vector) 
            
            # 计算加入该节点后的期望总覆盖数
            avg_spread = total_activated / simulation_times
            
            # 计算边际增益 (Marginal Gain) = 新的总覆盖 - 旧的总覆盖
            gain = avg_spread - current_spread
            
            if gain > max_marginal_gain:
                max_marginal_gain = gain
                best_node = node
        
        # --- 选定本轮最佳节点 ---
        if best_node != -1:
            selected_seeds.append(best_node)
            candidate_nodes.remove(best_node) # 从候选集中移除
            current_spread += max_marginal_gain # 更新当前的基准覆盖数
            
            # 打印日志，看看进度
            cost_time = time.time() - start_time
            logging.info(f"Step {k+1}/{m}: 选中节点 {best_node}, 增益 {max_marginal_gain:.2f}, 当前总覆盖 {current_spread:.2f} (耗时 {cost_time:.1f}s)")
        else:
            logging.warning("没有节点能带来正增益，提前结束。")
            break
            
    return selected_seeds

def deliverers_monteCarlo_CELF(
        n: int, 
        m: int, 
        tranProMatrix: np.ndarray, 
        succ_distribution: np.ndarray, 
        dis_distribution: np.ndarray, 
        simulation_algo_func, 
        simulation_times: int = 100
) -> list:
    """
    CELF (Lazy Forward) 优化的贪心蒙特卡洛策略。
    """
    logging.info(f"--- Running: CELF Greedy Monte Carlo (Sims={simulation_times}) ---")
    
    # 辅助函数：计算特定种子集的平均影响力
    def compute_spread(seeds):
        total = 0
        for _ in range(simulation_times):
            res, _, _ = simulation_algo_func(
                tranProMatrix, seeds, succ_distribution, dis_distribution, None
            )
            total += np.sum(res)
        return total / simulation_times

    start_time_all = time.time()
    
    # 1. 第一轮：计算所有节点的初始边际增益（相当于 Individual Monte Carlo）
    # 这一步不可避免，需要遍历所有节点
    logging.info("Step 0: 初始化所有节点的边际增益...")
    gains = [] 
    for node in range(n):
        spread = compute_spread([node])
        # (gain, node_id)
        gains.append((spread, node))
    
    # 按增益从大到小排序
    gains.sort(reverse=True, key=lambda x: x[0])
    
    selected_seeds = [gains[0][1]] # 选出第一个最好的
    current_spread = gains[0][0]
    logging.info(f"Seed 1: {selected_seeds[0]}, Spread: {current_spread:.2f}")
    
    # 移除已选的，剩下的作为候选列表
    # 列表结构: [ (marginal_gain, node_id), ... ]
    gains = gains[1:] 
    
    # 记录每个节点是在哪一轮计算的增益 (Lazy Evaluation)
    # 初始都是第 0 轮算的
    last_updated = {node: 0 for _, node in gains} 
    
    # 2. 后续轮次：CELF 加速选择
    num_selected = 1
    while num_selected < m:
        curr_node_idx = 0
        
        while True:
            # 取出当前增益最大的候选节点
            best_guess_gain, best_guess_node = gains[curr_node_idx]
            
            # CELF 核心逻辑：
            # 如果这个节点的增益是基于上一轮 (num_selected) 计算的，
            # 那么因为子模性(Submodularity)，它在当前轮次依然是最大的，直接选中！
            if last_updated[best_guess_node] == num_selected:
                selected_seeds.append(best_guess_node)
                current_spread += best_guess_gain
                
                # 从列表中移除该节点
                gains.pop(curr_node_idx)
                
                num_selected += 1
                logging.info(f"Seed {num_selected}: {best_guess_node}, Gain: {best_guess_gain:.2f}, Total: {current_spread:.2f}")
                break
            
            # 否则，我们需要重新计算它的边际增益
            # 新增益 = Spread(已选 + {u}) - Spread(已选)
            new_spread = compute_spread(selected_seeds + [best_guess_node])
            new_gain = new_spread - current_spread
            
            # 更新该节点的增益和轮次信息
            gains[curr_node_idx] = (new_gain, best_guess_node)
            last_updated[best_guess_node] = num_selected
            
            # 重新排序：只针对这就这一个元素进行重排，保持列表有序
            # 因为只有它变小了，所以只需要把它往后挪
            # 为了简单，这里用 sort，实际上可以用二分插入优化，但 Python sort 针对部分有序列表很快
            gains.sort(reverse=True, key=lambda x: x[0])
            
            # 循环继续，去检查重排后列表头部的那个节点
            
    logging.info(f"CELF 完成，总耗时: {time.time() - start_time_all:.2f}s")
    return selected_seeds


# 随机策略
def deliverers_random(n: int, m: int) -> list:
    logging.info(f"数据集节点总数:  n = {n}, m = {m}")
    selected_nodes = random.sample(range(n), k=m)
    logging.info(f"选择 {m}  {selected_nodes}\n")
    return selected_nodes


# 度中心性策略 认为认识人越多的人影响力越大
def deliverers_degreeTopM(adj, m: int) -> list:
    """
    选择度数最高的 m 个节点。
    """
    logging.info("依据: 选择连接数（度数）最多的节点。")
    
    # .A 属性将稀疏矩阵转为密集 NumPy 数组
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    
    # 使用 argsort 获取排序后的索引
    sorted_indexes = np.argsort(degrees)[::-1] # 从大到小排序
    
    top_m_indexes = sorted_indexes[:m]
    
    logging.info("选择的 Top M 节点及其度数:")
    selected_nodes_with_values = []
    for index in top_m_indexes:
        value = degrees[index]
        selected_nodes_with_values.append((index, value))
    logging.info("") # 打印一个空行
        
    return [node for node, value in selected_nodes_with_values] # 保持原始返回类型

# PageRank 策略
def deliverers_pageRank2(adj, m: int, tranProMatrix) -> list:
    """
    依据实际转移概率 (tranProMatrix) 来选 PageRank Top-m，
    保证选出的节点在 tranProMatrix 中有出概率 (out_prob > 0)。
    """
    logging.info("--- Running: PageRank (on tranProMatrix) Top M ---")


    # 确保 tranProMatrix 是 (n x n)，tranProMatrix[i,j] 表示 j -> i 的概率
    # networkx 需要 A[u,v] 表示 u->v 的权重，因此传入它的是 tranProMatrix.T
    if sp.issparse(tranProMatrix):
        B = tranProMatrix.T.tocsr()
        G = nx.from_scipy_sparse_array(B, create_using=nx.DiGraph)
    else:
        B = np.asarray(tranProMatrix).T
        G = nx.from_numpy_array(B, create_using=nx.DiGraph)

    # 计算 PageRank
    pagerank_scores = nx.pagerank(G, weight='weight')

    # 计算每个节点在 tranProMatrix 中的出概率（列和）
    if sp.issparse(tranProMatrix):
        out_prob = np.array(tranProMatrix.sum(axis=0)).ravel()
    else:
        out_prob = np.asarray(tranProMatrix).sum(axis=0)


    # 在有出概率的节点中按 pagerank 排序并取前 m
    ranked = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    selected = []
    for node, score in ranked:
        if out_prob[node] > 0:
            selected.append(node)
            if len(selected) >= m:
                break

    # 如果不足 m，补上出度最多的节点（作为回退）
    if len(selected) < m:
        logging.warning("PageRank-selected nodes with out_prob>0 fewer than m, fallback to out-prob sorting")
        candidates = np.argsort(-out_prob)  # indices sorted by descending out_prob
        for node in candidates:
            if node not in selected and out_prob[node] > 0:
                selected.append(int(node))
                if len(selected) >= m:
                    break

    logging.info("Selected PageRank seeds: %s", selected)
    return selected

# 忽略图的结构，选择alpha值最高的k个节点作为种子
def deliverers_alpha_sort(
        adj: sp.csr_matrix,
        tranProMatrix: np.ndarray,
        seeds_num: int,
        alpha: Dict[int, float]
) -> list:
    """
    Args:
        adj (sp.csr_matrix): 邻接矩阵 未使用
        tranProMatrix (np.ndarray): 转移概率矩阵 未使用。
        seeds_num (int): 要选择的种子数量 (k)。
        alpha (Dict[int, float]): 包含每个节点领券概率的字典。

    Returns:
        list: 包含k个最优种子节点ID的列表，按alpha值从高到低排序。
    """

    num_nodes = adj.shape[0]
    if seeds_num > num_nodes:
        logging.warning(f"种子数 ({seeds_num}) 大于图中节点数 ({num_nodes})")
        seeds_num = num_nodes

    # 1. 将alpha字典转换为 (节点ID, alpha值) 的元组列表 便于排序
    alpha_items = list(alpha.items())

    # 2. 降序排序 按照元组的第二个元素（即alpha值）进行排序
    alpha_items.sort(key=lambda item: item[1], reverse=True)
    logging.info(f"{len(alpha_items)} 个节点的alpha值降序排序 done")
    logging.info(f"==============>> {alpha_items[:10]}")

    # 3. 提取排序后前k个元组
    selected_items = alpha_items[:seeds_num]
    selected_seeds = [item[0] for item in selected_items]

    logging.info(f"\n选择的种子集 (按alpha排序): {selected_seeds}\n")
    return selected_seeds


import scipy.sparse as sp

def deliverers_teacher_alpha_1hop_sort(
        tranProMatrix: np.ndarray,
        seeds_num: int,
        alpha_distribution: np.ndarray,
        beta_distribution: np.ndarray,
) -> list:
    """
    公式: Score_i = alpha_i + sum_j (p_{ij} * alpha_j)
    """

    # 1. 处理转移矩阵
    # 根据 get_trans_matrix.py，tranProMatrix[j, i] 代表 i -> j 
    # 对 tranProMatrix 进行转置 (Transpose)，
    # 使得 M_T[i, j] 代表 i 指向 j 的概率 p_ij
    if sp.issparse(tranProMatrix):
        M_T = tranProMatrix.T.tocsr()
    else:
        M_T = np.asarray(tranProMatrix).T

    # 2. 矩阵乘法极限加速
    # M_T (N x N) 点乘 alpha_distribution (N x 1)
    # 等价于对每个节点 i 执行：sum_j (p_ij * alpha_j)
    transfer_probability = np.clip(
        1.0 - alpha_distribution - beta_distribution,
        0.0,
        1.0,
    )
    neighbor_influence = transfer_probability * M_T.dot(alpha_distribution)

    # 3. 最终得分计算
    # 得分 = 自身的转化率 + 邻居的期望转化率
    scores = alpha_distribution + neighbor_influence

    # 4. 排序并挑选 Top K
    sorted_indexes = np.argsort(scores)[::-1]
    selected_nodes = sorted_indexes[:seeds_num].tolist()
    return selected_nodes
