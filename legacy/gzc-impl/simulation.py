import numpy as np
import logging
import pandas as pd
import os
import multiprocessing

def evaluate_seed_set(
    seed_list: list,
    simulation_function,  # 传入具体的单次模拟函数
    num_simulations: int,
    tran_matrix: np.ndarray,
    distributions: tuple
) -> tuple: # (mean_E, variance, avg_steps, avg_redemptions)

    # 创建一个列表来存储每一次模拟的结果
    all_simulation_results = []
    all_steps_results = [] # 存储每轮的平均步数
    all_redemption_results = []


    succ_dist, dis_dist, _, const_factor_dist = distributions

    for i in range(num_simulations):
        # print(f"\t\t当前模拟轮次: {i}")
        success_vector, batch_steps, batch_redemptions   = simulation_function(
            tran_matrix,
            seed_list,
            succ_dist,
            dis_dist,
            const_factor_dist
        )
        single_activated_users  = np.sum(success_vector)
        all_simulation_results.append(single_activated_users)

        # 计算这一轮模拟中，平均每张券走了几步
        # batch_steps 是所有种子的总步数，除以种子数量得到平均步数
        avg_steps_this_round = batch_steps / len(seed_list) if len(seed_list) > 0 else 0
        all_steps_results.append(avg_steps_this_round)
        # 记录本轮核销数
        all_redemption_results.append(batch_redemptions) # <--- [新增]

    E_activated_users = np.mean(all_simulation_results)
    Var_activated_users = np.var(all_simulation_results)
    Final_Avg_Steps = np.mean(all_steps_results)
    E_redemptions = np.mean(all_redemption_results) # <--- [新增]

    return E_activated_users, Var_activated_users, Final_Avg_Steps, E_redemptions


def worker_evaluate_method(
        method_name: str,
        seed_set: list,
        simulation_times: list,  # 例如 [5, 10, 50]
        tran_matrix: np.ndarray,
        distribution_list: tuple,
        single_sim_func,
        seed_num: int,
        config_dict: dict  # 建议传 dict 而不是 Config 对象，防止 pickle 问题
):
    """
    单个进程执行的任务：评估某一个方法在不同模拟次数下的表现
    """
    results = []
    logging.info(f"[Process {os.getpid()}] 开始评估方法: {method_name}, 种子数: {len(seed_set)}")

    for num_sims in simulation_times:
        # 执行评估
        E_activated, Var_activated, Avg_Steps, E_redemptions   = evaluate_seed_set(
            seed_list=seed_set,
            simulation_function=single_sim_func,
            num_simulations=num_sims,
            tran_matrix=tran_matrix,
            distributions=distribution_list
        )

        std_dev = np.sqrt(Var_activated)
        # 【核心修改】计算真实的使用率
        # 使用率 = 平均总核销数 / 发放的总券数
        usage_rate = E_redemptions / seed_num if seed_num > 0 else 0.0

        # 1. 计算总曝光步数 (品牌曝光度)
        total_steps = Avg_Steps * seed_num
        
        # 2. 计算你的创新综合得分 (E-ROI 营销综合收益)
        # 权重设定可以自己在论文里圆：1个拉新 = 1.0分，1个老客复购 = 0.5分，1次普通曝光 = 0.1分
        # 老客复购数 = 总核销数 (E_redemptions) - 唯一激活数 (E_activated)
        repeat_purchases = E_redemptions - E_activated
        comprehensive_score = (E_activated * 1.0) + (repeat_purchases * 0.5) + (total_steps * 0.1)

        result_data = {
            "method": method_name,
            "seed_num": seed_num,
            "num_sims": num_sims,
            "E_activated_users": E_activated,
            "variance": Var_activated,
            "std_deviation": std_dev,
            "avg_steps": Avg_Steps, 
            "total_steps": total_steps,
            "log_alpha_slope": config_dict.get('log_alpha_slope'),
            "log_beta_slope": config_dict.get('log_beta_slope'),
            # 【新增】写入 CSV
            "degree_power_h": config_dict.get('degree_power_h'),
            "E_redemptions": E_redemptions,
            "comprehensive_score": comprehensive_score,
        }
        results.append(result_data)
        
        logging.info(f"[{method_name} - 种子:{seed_num}] "
                     f"拉新:{E_activated:.1f}人 | "
                     f"总销量:{E_redemptions:.1f}单 | "
                     f"品牌曝光:{total_steps:.1f}次 | "
                     f"🌟综合得分:{comprehensive_score:.1f}")

    return results

def simulation2(
        methods: list,
        seeds_list: list,
        init_tran_matrix: np.ndarray,
        usage_rate_file: str,
        distribution_list: tuple,
        simulation_times: list,
        single_sim_func,
        seed_num: int,
        config
):
    logging.info(f"多进程评估开始，共有 {len(methods)} 个方法待评估...")

    # 1. 准备多进程参数
    config_dict = {
        'random_dirichlet': getattr(config, 'random_dirichlet', False),
        'succ_degree_influence_factor': getattr(config, 'succ_degree_influence_factor', 0),
        'dis_degree_influence_factor': getattr(config, 'dis_degree_influence_factor', 0),
        'tran_degree_influence_factor': getattr(config, 'tran_degree_influence_factor', 0),
        'log_alpha_slope': getattr(config, 'log_alpha_slope', 0.1),
        'log_beta_slope': getattr(config, 'log_beta_slope', 0.25),
        'degree_power_h': getattr(config, 'degree_power_h', 1.0) 
    }

    tasks = []
    for i in range(len(methods)):
        # 打包每个任务需要的参数
        tasks.append((
            methods[i],
            seeds_list[i],
            simulation_times,
            init_tran_matrix,
            distribution_list,
            single_sim_func,
            seed_num,
            config_dict
        ))

    # 2. 启动进程池
    # processes数量：通常设为 min(任务数, CPU核心数 - 1)
    # 留一个核给系统和主进程写文件
    num_workers = len(methods)

    all_results_flat = []

    logging.info(f"启动进程池，Worker数量: {num_workers}")

    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用 starmap 因为 worker 接收多个参数
        # results_list 是一个列表的列表 [[method1_results], [method2_results], ...]
        results_list = pool.starmap(worker_evaluate_method, tasks)

    # 3. 结果处理与写入 (回到主进程，单线程安全写入)
    logging.info("所有进程执行完毕，开始写入数据...")

    # 展平结果列表
    for res in results_list:
        all_results_flat.extend(res)

    if not all_results_flat:
        logging.warning("没有任何结果生成！")
        return

    # 转换为 DataFrame
    df_new = pd.DataFrame(all_results_flat)

    # 写入文件 (处理表头逻辑)
    file_exists = os.path.exists(usage_rate_file)
    if not file_exists:
        os.makedirs(os.path.dirname(usage_rate_file), exist_ok=True)

    df_new.to_csv(usage_rate_file, mode='a+', header=not file_exists, index=False, encoding='utf-8-sig')

    logging.info(f"成功写入 {len(df_new)} 条记录至 {usage_rate_file}")