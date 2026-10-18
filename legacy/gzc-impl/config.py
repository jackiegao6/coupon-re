from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class ExperimentConfig:
    data_set: str #数据集名字 
    simulation_times: list #控制评估的精度（模拟次数）
    methods: list = None #选种子集的方法
    monte_carlo_L: int = 100 # 蒙特卡洛模拟次数
    distribution_type: str = 'log_continuous'
    personalization: str = 'None'
    data_prefix: str = str(Path(__file__).resolve().parents[1])
    method_type: str = 'None'
    num_samples: int = 50000

    seeds_num: int = 1
    randomness_factor: float = 0.2

    rng: np.random.Generator = np.random.default_rng(1)

    single_sim_func: str = 'AgainReJudge'

    version: str = 'v0'

    single_file_switch: bool = False

    # === 新增：log_continuous 分布的控制参数 ===
    log_alpha_base: float = 0.01
    log_alpha_slope: float = 0.1   # 控制低度数节点的采纳率上限
    log_beta_base: float = 0.01
    log_beta_slope: float = 0.6   # 控制高度数节点的丢弃率上限
    # === 新增：控制度数非线性映射的指数 h ===
    degree_power_h: float = 1.0

    @property
    def param_str(self):
        if self.distribution_type == 'log_continuous':
            return (
                f"_aBase{self.log_alpha_base}_aSlope{self.log_alpha_slope}"
                f"_bBase{self.log_beta_base}_bSlope{self.log_beta_slope}"
                f"_h{self.degree_power_h}"
            )
        return ""
    
    @property
    def adj_file(self):
        return f"{self.data_prefix}/dataset/network/{self.data_set}-adj.pkl"

    @property
    def time_cost_file(self):
        # 专门用于记录选种耗时的 CSV 文件
        return f"{self.data_prefix}/gzc-impl/results/{self.data_set}/{self.version}/TimeCost_{self.distribution_type}_SSRNum-{self.num_samples}.csv"

    def distribution_file(self, m = 0):
        # Node behavior is an instance property and must not change with
        # the coupon budget. Keep m only for backward call compatibility.
        return f"{self.data_prefix}/{self.data_set}/{self.version}/distribution-in-{self.data_set}/{self.distribution_type}_{self.param_str}_node-probabilities.pkl"

    def deliverers_cache_file(self, method, m = 0):
        return f"{self.data_prefix}/{self.data_set}/{self.version}/seeds-with-{self.data_set}/{self.distribution_type}_{self.param_str}_{method}_seedNum-{m}_SSRNum-{self.num_samples}.txt"

    def usage_rate_file(self, m = 0):
        times = ",".join(str(time) for time in self.simulation_times)
        return f"{self.data_prefix}/gzc-impl/results/{self.data_set}/{self.version}/Search_{self.distribution_type}_SSRNum-{self.num_samples}_seedNum-{self.seeds_num}.csv"

    def log_file(self):
        times = ",".join(str(time) for time in self.simulation_times)
        if self.distribution_type == "powerlaw" or self.distribution_type == "gamma" or self.distribution_type == "poisson":
            return f"{self.data_prefix}/gzc-impl/logs/{self.data_set}/{self.version}/{self.distribution_type}_SSRNum-{self.num_samples}-_seedNum-{self.seeds_num}_simuTimes-{times}_t-s-d-baseValue-{self.tran_base_value, self.succ_base_value, self.dis_base_value}_t-s-d-factor-{self.tran_degree_influence_factor, self.succ_degree_influence_factor, self.dis_degree_influence_factor}.log"
        return f"{self.data_prefix}/gzc-impl/logs/{self.data_set}/{self.version}/{self.distribution_type}_SSRNum-{self.num_samples}-_seedNum-{self.seeds_num}_simuTimes-{times}.log"
