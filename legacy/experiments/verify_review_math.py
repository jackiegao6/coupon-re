"""Exact finite checks of the repaired reduction and one-hop lower bound.

These checks supplement the manuscript's proofs; they are not quality
experiments and do not modify any experimental result.
"""
from fractions import Fraction as F
from itertools import combinations, product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check_reduction():
    graphs = {
        'K4': (4, list(combinations(range(4), 2))),
        'K3,3': (6, [(u,v) for u in range(3) for v in range(3,6)]),
        'triangular-prism': (6, [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(0,3),(1,4),(2,5)]),
        'cube': (8, [(u,v) for u in range(8) for v in range(u+1,8)
                    if (u^v) in [1,2,4]])}
    report = []
    for name, (n, edges) in graphs.items():
        assert all(sum(u in e for e in edges) == 3 for u in range(n))
        size = n+len(edges)
        # M[consumer][source], represented exactly in units of 1/32.
        matrix = [[0]*size for _ in range(size)]
        for u in range(n):
            matrix[u][u] = 16
        for index, (u,v) in enumerate(edges, start=n):
            matrix[index][u] = matrix[index][v] = 1
            matrix[index][index] = 8
        for k in range(1, min(n,4)+1):
            denom = 32**k
            def numerator(seeds):
                total = 0
                for row in matrix:
                    remaining = 1
                    for s in seeds:
                        remaining *= 32-row[s]
                    total += denom-remaining
                return total
            best_all = -1
            best_x = -1
            independent = False
            checked = 0
            for seeds in combinations(range(size), k):
                score = numerator(seeds)
                best_all = max(best_all, score)
                checked += 1
                if all(s < n for s in seeds):
                    internal = sum(u in seeds and v in seeds for u,v in edges)
                    expected = F(19,32)*k-F(internal,1024)
                    assert F(score,denom) == expected
                    best_x = max(best_x, score)
                    independent |= internal == 0
                else:
                    removed = next(s for s in seeds if s >= n)
                    added = next(s for s in range(n) if s not in seeds)
                    changed = tuple(s for s in seeds if s != removed)+(added,)
                    assert F(numerator(changed)-score,denom) >= F(1,4)
            assert best_all == best_x
            threshold = 19*k*32**(k-1)
            assert (best_all >= threshold) == independent
            report.append(dict(graph=name,k=k,seed_sets_checked=checked,
                maximum_spread=str(F(best_all,denom)),
                threshold=str(F(threshold,denom)),has_independent_set=independent))
    return report

def check_lower_bound():
    fixtures = [
        ('isolates', [F(0),F(1,5),F(3,5),F(9,10)], {}),
        ('cycle-and-self-loop', [F(1,10),F(1,5),F(0),F(2,5)],
         {(0,1):F(3,5),(1,2):F(1,2),(2,0):F(3,10),
          (2,3):F(3,5),(3,3):F(3,10)}),
        ('shared-consumer', [F(1,10),F(1,5),F(3,10),F(4,5)],
         {(0,3):F(7,10),(1,3):F(1,2),(2,3):F(2,5)})]
    report = []
    for name, adoption, edges in fixtures:
        n = len(adoption)
        actions = []
        for u in range(n):
            outgoing = [(v,p) for (w,v),p in edges.items() if w == u]
            discard = 1-adoption[u]-sum(p for _,p in outgoing)
            assert discard >= 0
            choices = [('consume',u,adoption[u]),('drop',-1,discard)]
            choices += [('transfer',v,p) for v,p in outgoing]
            actions.append([x for x in choices if x[2] > 0])
        matrix = [[F(0) for _ in range(n)] for _ in range(n)]
        mass, worlds = F(0), 0
        for world in product(*actions):
            weight = F(1)
            for _,_,p in world:
                weight *= p
            mass += weight
            worlds += 1
            for s in range(n):
                visited, u = set(), s
                while u not in visited:
                    visited.add(u)
                    action, target, _ = world[u]
                    if action == 'consume':
                        matrix[u][s] += weight
                        break
                    if action == 'drop':
                        break
                    u = target
        assert mass == 1
        short = [[adoption[s] if u == s else edges.get((s,u),F(0))*adoption[u]
                  for s in range(n)] for u in range(n)]
        assert all(short[u][s] <= matrix[u][s] for u in range(n) for s in range(n))
        def spread(values,seeds):
            result = F(0)
            for row in values:
                remaining = F(1)
                for s in seeds:
                    remaining *= 1-row[s]
                result += 1-remaining
            return result
        count, strict = 0, 0
        for k in range(1,n+1):
            for seeds in combinations(range(n),k):
                direct = sum(adoption[s] for s in seeds)
                onehop = spread(short,seeds)
                exact = spread(matrix,seeds)
                assert direct <= onehop <= exact <= k
                strict += onehop > direct
                count += 1
        report.append(dict(graph=name,enumerated_worlds=worlds,
            checked_seed_sets=count,strict_improvements=strict))
    return report

if __name__ == '__main__':
    report = dict(status='passed',reduction=check_reduction(),
        lower_bound=check_lower_bound(),
        original_reduction_counterexample=dict(graph='K4',k=2,
            independent_set_exists=False,original_threshold='3/2',
            original_two_terminal_seeds_spread='2'))
    out = ROOT/'plan/review/review-math-check-20261014.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],
        reduction_cases=len(report['reduction']),
        reduction_seed_sets=sum(x['seed_sets_checked'] for x in report['reduction']),
        lower_bound_seed_sets=sum(x['checked_seed_sets'] for x in report['lower_bound']))))
