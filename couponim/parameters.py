"""Behavior settings, with explicit names for the two balanced variants."""
import numpy as np

SCENARIOS = {
    "balanced-reference": (0.30, 0.10, 0.10, 0.10),
    "adoption-heavy": (0.45, 0.35, 0.15, 0.05),
    "forwarding-heavy": (0.01, 0.05, 0.01, 0.05),
}


def node_probabilities(graph, scenario="balanced-reference"):
    degree = np.log1p(graph.degrees.astype(float))
    z = np.clip(degree / max(float(degree.max()), 1.0), 1e-5, 1.0)
    if isinstance(scenario, dict):
        kind = scenario.get("kind", "degree")
        allowed = {
            "degree": {"name", "kind", "a0", "a1", "d0", "d1"},
            "heterogeneity": {"name", "kind", "scale", "association", "permutation_seed"},
            "transfer-share": {"name", "kind", "transfer", "share", "degree_effect"},
            "constant": {"name", "kind", "alpha", "discard"},
        }
        if kind not in allowed or set(scenario) - allowed[kind]:
            raise ValueError(f"Invalid custom behavior specification: {scenario}")
        if kind == "degree":
            alpha = float(scenario["a0"]) + float(scenario["a1"]) * (1-z)
            discard = float(scenario["d0"]) + float(scenario["d1"]) * z
        elif kind == "heterogeneity":
            original = .30 + .10 * (1-z)
            scale = float(scenario.get("scale", 3))
            if scale < 0:
                raise ValueError("Heterogeneity scale must be nonnegative")
            association = scenario.get("association", "degree")
            if association == "permuted":
                original = np.random.default_rng(scenario.get("permutation_seed", 42)).permutation(original)
            elif association != "degree":
                raise ValueError("association must be degree or permuted")
            alpha = original.mean() + scale * (original-original.mean())
            discard = .50-alpha
        elif kind == "transfer-share":
            t, share = float(scenario["transfer"]), float(scenario["share"])
            if not 0 <= t <= 1 or not 0 <= share <= 1:
                raise ValueError("transfer and share must be in [0,1]")
            # Matches the appendix parameter grid, including its declared clipping.
            z_grid = np.clip(degree / max(float(degree.max()), 1.0), 0, 1)
            shares = np.clip(share+float(scenario.get("degree_effect", .2))*(.5-z_grid), .05, .95)
            alpha, discard = (1-t)*shares, (1-t)*(1-shares)
        else:
            alpha = np.full(graph.n, float(scenario["alpha"]))
            discard = np.full(graph.n, float(scenario["discard"]))
    elif scenario == "balanced":
        original = 0.30 + 0.10 * (1 - z)
        alpha = original.mean() + 3 * (original - original.mean())
        discard = 0.50 - alpha
    else:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        a0, a1, d0, d1 = SCENARIOS[scenario]
        alpha = a0 + a1 * (1-z)
        discard = d0 + d1 * z
    discard[graph.degrees == 0] = 1 - alpha[graph.degrees == 0]
    transfer = 1 - alpha - discard
    if not np.isfinite([alpha, discard, transfer]).all() or min(alpha.min(), discard.min(), transfer.min()) < -1e-12:
        raise ValueError("This graph gives invalid behavior probabilities; no clipping is applied")
    return alpha, discard, np.maximum(0, transfer)
