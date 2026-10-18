"""Replay the existing NetFacebookEgo k=200 evaluations for a category audit.

Use identical seeds and RNG streams and require reproduced spread/usage.
This adds diagnostics to the archived simulations, not new quality trials.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random

import numpy as np

import run_sampling_diagnosis as protocol

ROOT = Path(__file__).resolve().parents[1]


def run(include_sampling=False):
    manifest = json.loads((protocol.REFERENCE/'manifest.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['source_sha256'].items():
        assert protocol.study.hash_file(ROOT/rel) == expected, rel
    g, a, d, tr, meta = protocol.study.prepare({'dataset':'NetFacebookEgo','scenario':'balanced'})
    assert meta['dataset_sha256'] == manifest['dataset_sha256'][g.name]
    leaves = np.flatnonzero(g.degrees == 1)
    parents = {int(v):int(g.out_neighbors(int(v))[0]) for v in leaves}
    classes = Counter(parents.values())
    hubs = set(classes)
    rows, inputs = [], {}
    paths = [protocol.REFERENCE/'jobs'/f'extended-budgets-NetFacebookEgo-balanced-r{rep}.json'
             for rep in range(5)]
    if include_sampling:
        paths += [protocol.OUT/'jobs'/f'sampling-NetFacebookEgo-k200-f{factor}-r{rep}.json'
                  for factor in [32,64] for rep in range(5)]
    for path in paths:
        archived = json.loads(path.read_text(encoding='utf-8'))
        rep = archived['job']['replicate']
        inputs[path.relative_to(ROOT).as_posix()] = protocol.study.hash_file(path)
        for original in archived['rows']:
            if original['k'] != 200 or original['method'] not in ['Uniform-SSR','Random','Forward-CELF']:
                continue
            seeds = original['seeds']
            seed_set = set(seeds)
            counts = Counter(parents[s] for s in seeds if s in parents)
            total_leaf_seeds = sum(counts.values())
            rng = random.Random(original['evaluation_seed'])
            # Categories are disjoint: selected seeds, then non-seed hubs,
            # then all other nodes. Own-start consumption is a coupon metric.
            consumption_counts = Counter()
            activation_counts = Counter()
            own_start = 0
            spread_sum = 0
            hub_consumptions = hub_activations = 0
            for _ in range(original['forward_simulations']):
                consumers = set()
                for source in seeds:
                    target = protocol.study.core.single_coupon_adopter(g,source,a,d,rng)
                    if target < 0:
                        continue
                    own_start += target == source
                    hub_consumptions += target in hubs
                    consumers.add(target)
                    category = 'seed' if target in seed_set else ('nonseed_hub' if target in hubs else 'other')
                    consumption_counts[category] += 1
                spread_sum += len(consumers)
                hub_activations += len(consumers & hubs)
                for target in consumers:
                    category = 'seed' if target in seed_set else ('nonseed_hub' if target in hubs else 'other')
                    activation_counts[category] += 1
            trials = original['forward_simulations']
            spread = spread_sum/trials
            usage = sum(consumption_counts.values())/trials
            assert abs(spread-original['spread']) < 1e-9
            assert abs(usage-original['consumed_coupons']) < 1e-9
            assert sum(activation_counts.values()) == spread_sum
            rows.append(dict(method=original['method'],replicate=rep,
                theta_factor=original.get('theta_factor'),
                replay_evaluation_seed=original['evaluation_seed'],forward_simulations=trials,
                spread=spread,consumed_coupons=usage,
                own_start_consumptions=own_start/trials,
                all_hub_consumptions=hub_consumptions/trials,
                all_hub_activations=hub_activations/trials,
                activation_by_category={cat:activation_counts[cat]/trials for cat in ['seed','nonseed_hub','other']},
                consumption_by_category={cat:consumption_counts[cat]/trials for cat in ['seed','nonseed_hub','other']},
                seed_leaf_count=total_leaf_seeds,seed_leaf_parent_groups=len(counts),
                seed_hub_count=len(seed_set & hubs),
                largest_seed_leaf_group=max(counts.values()),
                seed_leaf_group_concentration=sum((n/total_leaf_seeds)**2 for n in counts.values())))
        print(f'Replayed {path.stem}',flush=True)
    summary=[]
    configurations = [('Uniform-SSR', factor) for factor in ([8,32,64] if include_sampling else [8])]
    configurations += [('Random',None),('Forward-CELF',None)]
    for method,factor in configurations:
        group=[r for r in rows if r['method']==method and r['theta_factor']==factor]
        assert len(group) == 5
        item={'method':method,'theta_factor':factor,'runs':len(group)}
        for key in ['spread','consumed_coupons','own_start_consumptions','seed_leaf_count',
                    'seed_leaf_parent_groups','seed_hub_count','largest_seed_leaf_group',
                    'seed_leaf_group_concentration','all_hub_consumptions','all_hub_activations']:
            item[key]=float(np.mean([r[key] for r in group]))
        for key in ['activation_by_category','consumption_by_category']:
            item[key]={cat:float(np.mean([r[key][cat] for r in group])) for cat in ['seed','nonseed_hub','other']}
        summary.append(item)
    payload=dict(dataset=g.name,k=200,scenario='balanced',leaf_count=len(leaves),
                 leaf_parent_groups=len(classes),leaf_parent_group_sizes=sorted(classes.values(),reverse=True),
                 category_definition='Seed nodes first; then non-seed nodes among the ten leaf parents; then others.',
                 is_replay_not_new_independent_evidence=True,includes_sampling_diagnosis=include_sampling,
                 rows=rows,summary=summary,
                 input_sha256=inputs,source_sha256=protocol.study.hash_file(Path(__file__)))
    protocol.study.write_json(protocol.OUT/'random-gap-profile.json',payload)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-sampling',action='store_true')
    run(parser.parse_args().include_sampling)
