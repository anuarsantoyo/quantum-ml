"""Custom TPE with EI — hyperparameter proposal (DRAFT v1).

Interface (stable once agreed; real TPE logic implemented later):
    propose(trials, space=None, n_candidates=10, seed=None) -> list[dict]

Each returned candidate:
    {"config": {param: value, ...}, "score": float}

- score = Expected Improvement (EI) estimate from the surrogate.
  The AGENT considers it but decides with physics reasoning on top.
- trials: list of dicts from trials.json (trial_id, config, objective,
  uncertainty, summary). Failed trials have objective=None — handle them.
- space:  dict of {param: {"type": "float"|"int"|"choice", ...}}.

Placeholder behaviour: uniform random draws + mock score, so the loop is
runnable end-to-end. To implement later (TPE + EI):
  1. fit densities l(x) on good trials, g(x) on bad trials (median split);
     use uncertainty for a soft split / noise-aware EI
  2. draw candidates from l(x), score by EI = l(x)/g(x) * expected gain
  3. cold start: with <=2 trials, no surrogate — return spread-out samples
     (latin hypercube over the space) with score = exploration bonus

NOTE (Anuar 2026-08-29):
- structural choices may later enter the space as categorical knobs
  (e.g., anneal on/off, score_type), or a first "structure probe" trial
  runs before the campaign — TBD.
- if n_runs/n_iter stay in the space, enforce a feasibility cap
  (n_runs * n_iter <= budget) so trial cost stays ~1h.
"""
import numpy as np

DEFAULT_SPACE = {
    "n_runs":        {"type": "int",   "low": 100, "high": 500},   # runtime driver!
    "n_iter":        {"type": "int",   "low": 100, "high": 400},   # runtime driver!
    "lr_mu":         {"type": "float", "low": 5.0, "high": 40.0},
    "lr_gamma":      {"type": "float", "low": 0.1, "high": 1.5},
    "sigma_ref":     {"type": "float", "low": 5.0, "high": 25.0},
    "clip":          {"type": "float", "low": 5.0, "high": 20.0},
    "gamma_anneal":  {"type": "float", "low": 0.0, "high": 0.75},
    "h_s_min":       {"type": "float", "low": 0.0, "high": 0.2},
}

# feasibility cap so a trial stays ~1h (n_runs * n_iter ~ 40000 = ~3h at 17f speed)
RUNTIME_CAP = 60000   # n_runs * n_iter; TBD with Anuar


def propose(trials, space=None, n_candidates=10, seed=None):
    """Placeholder proposer: uniform random draws (feasible ones) + mock EI."""
    space = space or DEFAULT_SPACE
    rng = np.random.default_rng(seed)
    candidates = []
    while len(candidates) < n_candidates:
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
        if c.get("n_runs", 0) * c.get("n_iter", 0) > RUNTIME_CAP:
            continue  # infeasible (would blow the ~1h budget)
        candidates.append({"config": c, "score": float(rng.uniform(0.0, 1.0))})
    return candidates
