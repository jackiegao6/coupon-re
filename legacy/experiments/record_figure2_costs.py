"""Archive recorded Figure 2 costs; does not run new experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "z-assets-21"
DATASETS = ["Netscience", "NetFacebookEgo", "DoubanRandom",
            "EmailEnron", "network.douban", "SNAP-Epinions"]
DISPLAY = {"network.douban": "Douban", "SNAP-Epinions": "Epinions"}


def main():
    runs, summaries, sources = [], [], []
    for directory, k_max in [("z-assets-2", 200), ("z-assets-21", 300)]:
        for dataset in DATASETS:
            for replicate in range(3):
                path = ROOT / directory / "jobs" / f"{dataset}-r{replicate}.json"
                payload = path.read_bytes()
                value = json.loads(payload)
                source = path.relative_to(ROOT).as_posix()
                sources.append(dict(path=source, sha256=hashlib.sha256(payload).hexdigest(),
                                    whole_job_wall_seconds=value["wall_seconds"],
                                    ssr_stats=value["ssr_stats"]))
                for method, field in [("CIM-RIS", "ssr_seconds"),
                                      ("Forward-CELF", "forward_seconds")]:
                    runs.append(dict(dataset=DISPLAY.get(dataset, dataset),
                                     dataset_id=dataset, k_max=k_max,
                                     replicate=replicate, method=method,
                                     n=value["n"], m=value["m"], theta=value["n"],
                                     forward_paths_per_source=64,
                                     selection_block_seconds=value[field],
                                     peak_process_memory_bytes=None,
                                     peak_memory_status="not_measured",
                                     source_json=source, source_field=field))
    for dataset in DATASETS:
        for k_max in [200, 300]:
            for method in ["CIM-RIS", "Forward-CELF"]:
                group = [r for r in runs if r["dataset_id"] == dataset
                         and r["k_max"] == k_max and r["method"] == method]
                assert len(group) == 3
                times = [r["selection_block_seconds"] for r in group]
                summaries.append(dict(dataset=DISPLAY.get(dataset, dataset),
                                      dataset_id=dataset, k_max=k_max, method=method,
                                      repeats=3, median_seconds=median(times),
                                      minimum_seconds=min(times), maximum_seconds=max(times),
                                      peak_process_memory_bytes=None))
    result = dict(
        purpose="Saved for later runtime and space-complexity tables; no new benchmarking",
        timing_scope="Recorded selection blocks include sample generation and greedy "
                     "selection, exclude graph loading and independent quality evaluation. "
                     "Extension blocks also include prefix verification and some cleanup.",
        limitations=[
            "Only CIM-RIS and Forward-CELF were separately timed.",
            "Times are for full k_max=200 or k_max=300 selection, not each plotted k.",
            "The k_max=300 run replays 200 original components and generates 100 new "
            "components; its recorded time is not an incremental cost.",
            "Peak RSS/working set was not systematically measured. Isolated live "
            "process observations are not peak-memory measurements.",
            "Membership and scanned-edge counts are operation counts, not memory in bytes.",
            "These are observed quality-run timings, not a dedicated isolated benchmark."
        ], raw_timing_records=runs, runtime_summary=summaries, source_jobs=sources)
    (OUT / "cost-records.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table = ["| 数据集 | CIM-RIS，k=200 | Fwd-CELF，k=200 | CIM-RIS，k=300 | Fwd-CELF，k=300 |",
             "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        times = [next(r["median_seconds"] for r in summaries
                      if r["dataset_id"] == dataset and r["k_max"] == k
                      and r["method"] == method)
                 for k, method in [(200, "CIM-RIS"), (200, "Forward-CELF"),
                                   (300, "CIM-RIS"), (300, "Forward-CELF")]]
        table.append("| " + DISPLAY.get(dataset, dataset) + " | "
                     + " | ".join(f"{t:.3f}" for t in times) + " |")
    notes = r"""# Figure 2：运行时间与空间复杂度备忘

记录日期：2026-10-15。用户要求保留这些信息，后续制作运行时间和空间相关的两个表格。
本文件从已有 job JSON 提取耗时，不重跑实验。原始逐次耗时、统计范围及文件哈希见 `cost-records.json`。

## 已记录的耗时

下表单位为秒，均为三次运行的中位数。CIM-RIS 对应 `ssr_seconds`，Forward-CELF 对应 `forward_seconds`。
两者均包括样本生成和贪心选择，均不包括加载图、生成全部静态排序和独立正向质量评估。
扩展脚本还在计时区间内进行前缀核对及部分对象释放；不能将这些记录称为严格隔离的性能基准。

__RUNTIME_TABLE__

- `z-assets-2/jobs`：完整选到 k=200 的时间，较小预算共享其选择前缀。
- `z-assets-21/jobs`：完整选到 k=300 的时间，重放前 200 个分量，再生成 100 个分量。
  这不是只增加 100 张券的增量计时，也不是 k=250 的独立计时。
- `wall_seconds` 包含整个任务的图读取、排序、选择和所有质量评估，不可直接充当单方法时间。
- 1Hop、Alpha、Degree、PageRank、Random 没有单独计时，不能用剩余总时间分摊补齐。

## 当前代码的复杂度

以下是依据本地实现的算法分析，不是实际内存测量。令 n、m 为节点和有向边数，
k 为最大选择预算，theta 为 SSR 数量（本批 theta=n），q 为每源前向轨迹数（本批 q=64），
I 为 PageRank 迭代次数（代码最多 50 次）。令 H 为全部 RR 分量的成员总数，
B 为反向生成中扫描的入边总次数，ell 为前向样本轨迹的平均节点处理步数。
不计公共输入图的额外空间列需要再加上 O(n+m)，才能得到包含图存储的总空间。
哈希表操作采用通常的期望常数时间假设。

| 方法 | 当前实现的时间上界 | 不计公共输入图的额外空间 |
|---|---|---|
| CIM-RIS | O(n+m+theta*k+B+H+k*n) | O(n+m+theta+k+H) |
| Forward-CELF | O(n*q*ell+k*n*(q+log n))，CELF 重算次数取最坏上界 | O(n*q+n) |
| 1Hop | O(n+m+n log n) | O(n) |
| Alpha（图例 p^alpha） | O(n log n) | O(n) |
| Degree | O(n log n)，degree 数组已在加载阶段建立 | O(n) |
| PageRank | O(I*(n+m)+n log n) | O(n+m)，本实现分配长度为 m 的边贡献数组 |
| Random | O(n)，本实现生成完整排列后取前 k 个 | O(n) |

CIM-RIS 的最坏情况满足 H<=theta*k*n、B<=theta*k*m，因此最坏额外空间为
O(n+m+theta*k*n)，不是只存一个 RR 集的 O(n)。
在本批的固定动作、重访丢弃、均匀无条件根采样模型中，对任意固定源 s，
它最终在某个根处消费的总概率至多为 1，故单次 RR 样本 E|R|<=1。
同理，对每个被访问节点按入度计数可得单次预期扫描入边数至多 m/n。
所以 E[H]<=theta*k、E[B]<=theta*k*m/n；当前实现的预期空间为
O(n+m+theta*k)，预期时间为 O(n+m+theta*k*(1+m/n)+k*n)。
当 theta=n 时，分别为 O(m+n*k) 和 O(k*(n+m))（k>=1）。
这些是带有模型条件的期望上界，不是由运行时间表验证出的复杂度。

## 内存数据缺口及后续制表口径

本批 job JSON 没有记录每个方法的峰值 RSS/Working Set，也没有五个启发式的独立耗时。
运行时偶尔读取的进程内存快照没有覆盖完整峰值，且进程同时装有其他方法的数据，不能填入算法内存比较表。
`ssr_stats.memberships` 和 `scanned_edges` 已在 cost-records.json 保存，可用于解释采样规模，不能作为峰值内存。

后续若第二张表是理论空间复杂度表，可据上面的公式组织；若是实测内存 MiB 表，则需补测：
每个方法使用独立新进程、相同图和参数，统一记录进程峰值内存；统一说明是否包含图存储。
时间表也应统一样本生成、预处理、选种的计时边界，质量评估单列或排除。
较小预算的耗时和 k=250 耗时不可从当前 k=200/300 的完整时间线性插值。
"""
    (OUT / "cost-notes.md").write_text(
        notes.replace("__RUNTIME_TABLE__", "\n".join(table)) + "\n", encoding="utf-8")
    print("\n".join(table))
    print("Saved cost-records.json and cost-notes.md; peak memory remains unmeasured.")


if __name__ == "__main__":
    main()
