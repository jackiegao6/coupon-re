"""Single RR-set generation extracted for review; no SSR or seed selection.

Source: experiments/run_real_submission.py::reverse_coupon_set.
This is the existing fixed-node-action implementation, NOT the proposed
fresh-resampling-on-revisit model. The core below preserves the source logic.
The optional wrapper adds an ordinary, unconditioned single-coupon root gate;
it does not reproduce the experiment's optimized joint root-gate sampling.
"""

from __future__ import annotations

import random

import numpy as np

from run_real_submission import Graph


def reverse_coupon_set(
    graph: Graph,
    root: int,
    alpha: np.ndarray,
    discard: np.ndarray,
    rng: random.Random,
) -> set[int]:
    """Generate one RR set GIVEN that the root action is consumption.

    alpha[v] = consumption probability; discard[v] = discard probability.
    Conditional on forwarding, the experiment chooses an out-neighbor uniformly.
    choices[v] caches one action for this sample; -1 means no forwarding.
    """
    rr_set = {root}
    queue = [root]
    # 根节点的行为已在外层固定为消费，不再为其抽样转发行为。
    choices: dict[int, int] = {root: -1}
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for predecessor in graph.in_neighbors[current]:
            # 实验现有规则：每个节点在本样本中只生成一次行为。
            if predecessor not in choices:
                draw = rng.random()
                if draw < alpha[predecessor] + discard[predecessor]:
                    choices[predecessor] = -1
                else:
                    neighbors = graph.out_neighbors(predecessor)
                    choices[predecessor] = int(neighbors[rng.randrange(len(neighbors))])
            # 行为确实指向当前节点时才接入；每个节点最多入队一次。
            if choices[predecessor] == current and predecessor not in rr_set:
                rr_set.add(predecessor)
                queue.append(predecessor)
    return rr_set


def generate_single_rr_set(
    graph: Graph,
    root: int,
    alpha: np.ndarray,
    discard: np.ndarray,
    rng: random.Random,
) -> set[int]:
    """Review wrapper: ordinary root-consumption gate followed by one RR set.

    This wrapper is added for presenting a complete single-sample interface;
    the production experiment handles its root gates outside the core function.
    """
    if rng.random() >= alpha[root]:
        return set()
    return reverse_coupon_set(graph, root, alpha, discard, rng)
