"""Custom TPE — hyperparameter proposal (DRAFT v0 / placeholder).

Interface contract (stable once agreed; TPE logic implemented later):
    propose(trials, space=None, n_candidates=10, seed=None) -> list[dict]

- trials: list of dicts from trials.json (trial_id, config, objective, uncertainty, summary)
- space:  dict of {param: {"type": "float"|"int"|"choice", ...}} — see DEFAULT_SPACE
- returns n_candidates config dicts (same keys as space)

Placeholder behaviour: uniform random draws within the space, so the loop is
runnable end-to-end before the real TPE lands. To implement later:
  1. fit densities l(x) on good trials / g(x) on bad trials (split by median objective)
  2. sample candidates from l(x), score by l(x)/g(x), return top-n
  3. handle failed trials (objective=None) and the deterministic-vs-noisy objective

NOTE: DEFAULT_SPACE below is a placeholder for discussion — the real search space
is TBD with Anuar (see context.md "Open items").
"""
import numpy as np

DEFAULT_SPACE = {
    "n_runs":        {"type": "int",   "low": 100, "high": 500},
    "n_iter":        {"type": "int",   "low": 100, "high": 400},
    "lr_mu":         {"type": "float", "low": 5.0, "high": 40.0},
    "lr_gamma":      {"type": "float", "low": 0.1, "high": 1.5},
    "sigma_ref":     {"type": "float", "low": 5.0, "high": 25.0},
    "clip":          {"type": "float", "low": 5.0, "high": 20.0},
    "gamma_anneal":  {"type": "float", "low": 0.0, "high": 0.75},
    "h_s_min":       {"type": "float", "low": 0.0, "high": 0.2},
}


def propose(trials, space=None, n_candidates=10, seed=None):
    """Placeholder proposer: uniform random draws within the space."""
    space = space or DEFAULT_SPACE
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(n_candidates):
        c = {}
        for k, s in space.items():
            if s["type"] == "int":
                c[k] = int(rng.integers(s["low"], s["high"] + 1))
            elif s["type"] == "float":
                c[k] = float(rng.uniform(s["low"], s["high"]))
            elif s["type"] == "choice":
                c[k] = rng.choice(s["values"]).item()
            else:
                raise ValueError(f"unknown type {s['type']} for {k}")
        candidates.append(c)
    return candidates
