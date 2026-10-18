"""Report the full small pilot; keep pilot data separate from main comparisons."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import run_heterogeneity_pilot as experiment

ROOT=experiment.ROOT
OUT=experiment.OUT


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['jobs']==experiment.specs()
    for rel,digest in manifest['source_sha256'].items():
        assert experiment.study.hash_file(ROOT/rel)==digest,rel
    for name,digest in manifest['dataset_sha256'].items():
        assert experiment.study.hash_file(experiment.study.core.DATASETS[name])==digest
    groups=defaultdict(list)
    parameters={}
    inputs={}
    eval_seeds=set()
    random_sets={}
    for job in manifest['jobs']:
        path=OUT/'jobs'/f'{job["id"]}.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        assert data['job']==job and data['status']=='complete' and len(data['rows'])==3
        meta=data['metadata']
        assert meta['dataset_sha256']==manifest['dataset_sha256'][job['dataset']]
        inputs[path.relative_to(ROOT).as_posix()]=experiment.study.hash_file(path)
        a=np.asarray(meta['alpha_by_node'])
        assert np.min(a)>=0 and np.max(a)<=.5
        assert abs(a.mean()-meta['original_mean'])<1e-12
        assert abs(a.var()-job['scale']**2*meta['original_variance'])<1e-12
        key=(job['dataset'],job['association'],job['scale'])
        if key in parameters:
            assert np.array_equal(parameters[key]['alpha_by_node'],a)
        parameters[key]=meta
        for row in data['rows']:
            assert row['forward_simulations']==5000 and row['k']==25
            assert len(row['seeds'])==len(set(row['seeds']))==25
            assert row['evaluation_seed']!=row['selection_seed'] and row['evaluation_seed'] not in eval_seeds
            assert 0<=row['spread']<=row['consumed_coupons']+1e-10<=25+1e-10
            eval_seeds.add(row['evaluation_seed'])
            groups[key].append(row)
            if row['method']=='Random':
                rk=(job['dataset'],job['replicate'])
                assert rk not in random_sets or random_sets[rk]==row['seeds']
                random_sets[rk]=row['seeds']
    assert len(eval_seeds)==72
    summaries=[]
    for name in experiment.DATASETS:
        for scale in [1,3]:
            a=parameters[(name,'degree',scale)]['alpha_by_node']
            b=parameters[(name,'permuted',scale)]['alpha_by_node']
            np.testing.assert_allclose(np.sort(a),np.sort(b),atol=1e-12,rtol=0)
        for association in ['degree','permuted']:
            for scale in [1,3]:
                key=(name,association,scale)
                rows=groups[key]
                assert len(rows)==9
                info=parameters[key]
                summary=dict(dataset=name,association=association,scale=scale,
                    mean_a=info['consumption_mean'],sd_a=float(np.sqrt(info['consumption_variance'])),
                    min_a=info['consumption_min'],max_a=info['consumption_max'],methods={})
                for method in ['Random','CIM-RIS','Forward-CELF']:
                    values=sorted([r for r in rows if r['method']==method],key=lambda r:r['replicate'])
                    assert [r['replicate'] for r in values]==[0,1,2]
                    summary['methods'][method]=dict(mean_spread=float(np.mean([r['spread'] for r in values])),
                        min_spread=min(r['spread'] for r in values),max_spread=max(r['spread'] for r in values),
                        mean_consumed=float(np.mean([r['consumed_coupons'] for r in values])),
                        mean_seed_a=float(np.mean([r['selected_mean_consumption_probability'] for r in values])))
                means=summary['methods']
                summary['cim_over_random_percent']=100*(means['CIM-RIS']['mean_spread']/means['Random']['mean_spread']-1)
                summary['forward_over_random_percent']=100*(means['Forward-CELF']['mean_spread']/means['Random']['mean_spread']-1)
                summaries.append(summary)
    contrasts=[]
    for name in experiment.DATASETS:
        for association in ['degree','permuted']:
            by_scale={scale:{(r['replicate'],r['method']):r['spread'] for r in groups[(name,association,scale)]}
                      for scale in [1,3]}
            changes=[]
            for rep in range(3):
                gaps={scale:by_scale[scale][rep,'CIM-RIS']-by_scale[scale][rep,'Random'] for scale in [1,3]}
                changes.append(gaps[3]-gaps[1])
            contrasts.append(dict(dataset=name,association=association,
                change_in_cim_minus_random_spread_by_replicate=changes,
                mean_change=float(np.mean(changes)),min_change=min(changes),max_change=max(changes)))
    analysis=dict(configurations=summaries,paired_contrasts=contrasts,
                  jobs_complete=24,solutions=72,forward_campaigns=360000,
                  input_sha256=inputs,analysis_source_sha256=experiment.study.hash_file(Path(__file__)))
    experiment.study.write_json(OUT/'analysis.json',analysis)
    lines=['# 消费概率方差的小规模预实验', '',
        '目的：验证扩大消费概率差异是否拉开 CIM-RIS 与 Random。'
        'Forward-CELF 是强经验参照，不是理论上界，也不是精确最优解。', '',
        '两个真实图实例、k=25、消费概率标准差倍数 lambda=1、3（方差分别为原来的 1 倍、9 倍）。'
        '均值保持不变，总吸收概率固定 0.5，转发概率固定 0.5；不截断概率。'
        '分别保留度相关性和使用一个固定随机置换，置换在两个 lambda 和所有重复中保持一致。'
        '每设置 3 次选种，3 种方法，共 24 个作业、72 个解；每解独立正向评价 5,000 次。'
        'SRR theta=8n；Forward-CELF 每源 256 条轨迹。所有设置均保留。', '',
        '## 传播规模均值', '',
        '| 数据集 | 概率关联 | lambda | 消费概率标准差 | Random | CIM-RIS | Forward-CELF | CIM 相对 Random |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        m=s['methods']
        lines.append(f"| {s['dataset']} | {s['association']} | {s['scale']} | {s['sd_a']:.5f} | "
                     f"{m['Random']['mean_spread']:.4f} | {m['CIM-RIS']['mean_spread']:.4f} | "
                     f"{m['Forward-CELF']['mean_spread']:.4f} | {s['cim_over_random_percent']:+.2f}% |")
    lines+=['', '## 扩大方差后，CIM-RIS 减 Random 的绝对差距变化', '',
            '每一项为 `(CIM_3 - Random_3) - (CIM_1 - Random_1)`，大于零表示差距扩大。'
            '按同一图与重复编号配对；参数设置间 Random 使用相同节点集合，其他方法复用选种随机种子，'
            '但传播评价使用分别独立的随机流。', '',
            '| 数据集 | 概率关联 | 平均变化（消费者数） | 三次变化的最小值 | 最大值 |',
            '|---|---|---:|---:|---:|']
    for c in contrasts:
        lines.append(f"| {c['dataset']} | {c['association']} | {c['mean_change']:+.4f} | "
                     f"{c['min_change']:+.4f} | {c['max_change']:+.4f} |")
    lines+=['', '## 本轮发现', '',
        '四组对照中，三组的 CIM-RIS 相对 Random 的均值优势扩大；NetFacebookEgo 保留度关联时收窄。'
        '三次配对的绝对差距变化在 Netscience 原关联组均为正，在 NetFacebookEgo 原关联组均为负；'
        '两个置换组都存在相反方向的单次结果。不能把三组均值上升写成稳定或显著改善。', '',
        'Netscience 原关联组的 CIM-RIS 均值从 15.6546 降至 15.5540，Random 从 15.4446 降至 14.9561。'
        '相对差距变大主要是 Random 下降，不能据此写“我们的绝对传播质量提高”。'
        '相同组的 Forward-CELF 从 16.0635 升至 16.8767，说明至少这个强参照能利用变化后的参数，'
        '而当前固定 8n 的 SRR 没有充分利用；本轮未将 Forward-CELF 认作 OPT。', '',
        '打乱关联后的 Netscience 上，CIM-RIS 从 15.9740 升至 16.3004，Random 从 15.7363 升至 15.7808，'
        '提供了方差增大可能扩大可利用差异的初步证据。NetFacebookEgo 置换组的均值改善较小。', '',
        '这轮支持“值得研究用户行为异质性”，不支持“增大方差就必然使 CIM-RIS 更占优”。'
        '当前仍是两个真实图上的合成行为参数试验，并非真实用户消费记录。'
        '遵照少量实验的要求，不继续加配置或筛选结果，也暂不改正式论文的 Figure 2。', '',
        '## 解读范围', '',
        '- 只有 3 次重复、两个图和一个置换，是先导验证，不声称统计显著性或普遍优势。',
        '- 方差放大保持了平均自身消费概率，但不保证 Random 的最终传播规模不变。'
        '转发访问节点的分布并不均匀，消费位置及消费者重叠仍可能变化。',
        '- NetFacebookEgo 原概率标准差约 0.00431，扩大后约 0.01294；大量叶节点仍具有相同概率。'
        '提高方差倍数不等于创造了大量高收益且彼此不同的候选节点。',
        '- 本轮没有改变采样算法，也没有改变历史主图或基线结果。新参数结果不能覆盖图二原 balanced 曲线。',
        '- 运行前固定的协议、源码快照、按节点列出的概率及原始解在 '
        '`experiments/results/heterogeneity-pilot-20261014/`；分析可由 '
        '`experiments/analyze_heterogeneity_pilot.py` 重建。并行运行耗时不作效率比较。', '']
    report=ROOT/'plan/review/heterogeneity-pilot-2026-10-14.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    for s in summaries:
        print(s['dataset'],s['association'],s['scale'],
              {m:round(d['mean_spread'],4) for m,d in s['methods'].items()},
              'CIM/Random %',round(s['cim_over_random_percent'],3))
    print('Paired contrasts:',json.dumps(contrasts))


if __name__=='__main__':
    main()
