import os
import pickle
import numpy as np
from scipy.stats import truncnorm, gamma, expon
import logging
import networkx as nx
import scipy.sparse as sp
from config import ExperimentConfig
from typing import Dict

def _normalize_triplet(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple:
    total = a + b + c
    out_a, out_b, out_c = np.zeros_like(a), np.zeros_like(b), np.zeros_like(c)
    mask = total != 0
    out_a[mask] = a[mask] / total[mask]
    out_b[mask] = b[mask] / total[mask]
    out_c[mask] = c[mask] / total[mask]
    return out_a, out_b, out_c

def _min_max_scale(v: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(v), np.max(v)
    range_val = max_val - min_val
    if range_val == 0:
        return np.clip(v, 0, 1)
    return (v - min_val) / range_val

def _generate_continuous_log_degree_distributions(n: int, degrees: np.ndarray, config) -> dict:
    logging.info(f"===> Generating 'Log-Continuous' distributions (AlphaSlope={config.log_alpha_slope}, BetaSlope={config.log_beta_slope}, h={config.degree_power_h})...")

    log_degrees = np.log1p(degrees) 
    max_log = np.max(log_degrees)
    if max_log == 0: max_log = 1.0
    
    # 1. 先计算基础的归一化值 [0, 1]
    base_norm_deg = log_degrees / max_log 
    
    # 2. 【核心修改】应用指数 h，并做安全截断防止 h<0 时除以 0
    # 保证底数最小为 1e-5
    safe_base_norm = np.clip(base_norm_deg, 1e-5, 1.0)
    norm_deg = np.power(safe_base_norm, config.degree_power_h)
    
    # 再次截断确保概率映射不出界 (特别是 h<0 时，值可能会大于1)
    norm_deg = np.clip(norm_deg, 0.0, 1.0)
    
    expected_beta  = config.log_beta_base + config.log_beta_slope * norm_deg          
    expected_alpha = config.log_alpha_base + config.log_alpha_slope * (1.0 - norm_deg)  
    
    sum_ab = expected_alpha + expected_beta
    if np.any(sum_ab > 1.0):
        raise ValueError("Invalid diffusion parameters: p_a + p_d exceeds 1")

    # Match Algorithm 4 in the manuscript exactly. Earlier code sampled
    # a Dirichlet perturbation, which changed the model across runs and
    # made the reported scenario parameters unreproducible.
    succ_dist = expected_alpha
    dis_dist = expected_beta
    tran_dist = 1.0 - succ_dist - dis_dist
    
    const_factor_dist = np.ones(n, dtype=float)
    
    return {
        'succ_distribution': succ_dist,
        'dis_distribution': dis_dist,
        'tran_distribution': tran_dist,
        'constantFactor_distribution': const_factor_dist
    }

def tmpenerate_continuous_log_degree_distributions(n: int, degrees: np.ndarray, config) -> dict:
    import logging
    import numpy as np
    logging.info(f"===> Generating 'Log-Continuous' distributions (AlphaSlope={config.log_alpha_slope}, BetaSlope={config.log_beta_slope}, h={config.degree_power_h})...")
    
    rng = np.random.default_rng(config.rng)
    
    log_degrees = np.log1p(degrees) 
    max_log = np.max(log_degrees)
    if max_log == 0: max_log = 1.0
    
    # 1. 先计算基础的归一化值 [0, 1]
    base_norm_deg = log_degrees / max_log 
    
    # 2. 【核心修改】应用指数 h，并做安全截断防止 h<0 时除以 0
    # 保证底数最小为 1e-5
    safe_base_norm = np.clip(base_norm_deg, 1e-5, 1.0)
    norm_deg = np.power(safe_base_norm, config.degree_power_h)
    
    # 再次截断确保概率映射不出界 (特别是 h<0 时，值可能会大于1)
    norm_deg = np.clip(norm_deg, 0.0, 1.0)
    
    expected_beta  = config.log_beta_base + config.log_beta_slope * norm_deg          
    expected_alpha = config.log_alpha_base + config.log_alpha_slope * (1.0 - norm_deg)  
    
    sum_ab = expected_alpha + expected_beta
    overflow_mask = sum_ab > 0.95
    expected_alpha[overflow_mask] = (expected_alpha[overflow_mask] / sum_ab[overflow_mask]) * 0.95
    expected_beta[overflow_mask]  = (expected_beta[overflow_mask] / sum_ab[overflow_mask]) * 0.95
    
    expected_tran = 1.0 - expected_alpha - expected_beta
    
    gamma = 20.0 
    
    dirichlet_params = np.vstack([
        expected_alpha * gamma + 1e-3, 
        expected_beta * gamma + 1e-3, 
        expected_tran * gamma + 1e-3
    ]).T 
    
    probs = np.zeros((n, 3))
    for i in range(n):
        probs[i] = rng.dirichlet(dirichlet_params[i])
        
    succ_dist = probs[:, 0]
    dis_dist  = probs[:, 1]
    tran_dist = probs[:, 2]
    
    
    return {
        'succ_distribution': succ_dist,
        'dis_distribution': dis_dist,
        'tran_distribution': tran_dist,
    }



def get_distribution_degree_aware(
        distribution_file: str,
        distribution_type: str,
        adj: nx.Graph,
        config: ExperimentConfig,
) -> tuple:
    if isinstance(adj, np.ndarray):
        n = adj.shape[0]
        degrees = adj.sum(axis=1)
    elif isinstance(adj, nx.Graph):
        n = adj.number_of_nodes()
        degrees = np.array([d for n, d in adj.degree()])
    elif sp.issparse(adj):
        n = adj.shape[0]
        degrees = np.array(adj.sum(axis=1)).flatten()
    else:
        raise TypeError("矩阵类型不支持")

    if os.path.exists(distribution_file):
        logging.info(f"复用概率分布 文件位置: {distribution_file}")
        with open(distribution_file, 'rb') as f:
            dis_dict = pickle.load(f)
        return tuple(dis_dict.values())

    generator_registry = {
        'log_continuous': _generate_continuous_log_degree_distributions, 
    }

    if distribution_type not in generator_registry:
        raise ValueError(f"该分布类型暂不支持: '{distribution_type}'.")

    generator_func = generator_registry[distribution_type]
    dis_dict = generator_func(n, degrees, config)

    os.makedirs(os.path.dirname(distribution_file), exist_ok=True)
    with open(distribution_file, 'wb') as f:
        pickle.dump(dis_dict, f)
    logging.info(f"保存概率分布 文件位置: {distribution_file}")

    return tuple(dis_dict.values())
