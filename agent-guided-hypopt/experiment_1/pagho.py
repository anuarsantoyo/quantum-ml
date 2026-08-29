"""PAGHO — Physics-informed Agent-Guided Hyperparameter Optimization.

experiment_1 scripts, ONE module (Anuar's convention 2026-08-29: all scripts in a
single .py file). Contains:

1. The algorithm   — propose(trials, space, n_candidates) -> [{config, score}]
                     (TPE + EI draft; placeholder until implemented)
2. The harness     — run_trial(config) -> results
                     compute_objective(results) -> (objective, uncertainty)
                     (stubs; port machinery from notebooks/17f & 18b)

Interface contract (what makes algorithms swappable):
    propose(trials, space, n_candidates=10, seed=None) -> list[{"config", "score"}]
- trials: list of dicts from trials.json (trial_id, config, objective, uncertainty, summary)
- space:  dict of {param: {"type": "float"|"int"|"choice", ...}} — see DEFAULT_SPACE
- score = Expected Improvement (EI) from the surrogate. The AGENT considers it but
  decides with physics reasoning on top.

A future experiment_2 can ship GPBO / CMA-ES against the same contract, same
benchmark -> clean A/B. Decision TPE vs GPBO: pending.
"""
import os
import sys

# ---- path bootstrap: climb to repo root (dir containing src/), keep importable ----
_REPO = os.getcwd()
while _REPO != os.path.dirname(_REPO) and not os.path.isdir(os.path.join(_REPO, 'src')):
    _REPO = os.path.dirname(_REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

# ============================================================
# SEARCH SPACE (placeholder — TBD with Anuar)
# ============================================================
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

# ============================================================
# ALGORITHM — TPE + EI (draft / placeholder)
# ============================================================
def propose(trials, space=None, n_candidates=10, seed=None):
    """Placeholder proposer: uniform random draws (feasible ones) + mock EI.

    Real TPE + EI to implement later:
      1. fit densities l(x) on good trials, g(x) on bad trials (median split);
         use uncertainty for a soft split / noise-aware EI
      2. draw candidates from l(x), score by EI = l(x)/g(x) * expected gain
      3. cold start: with <=2 trials, no surrogate — return spread-out samples
         (latin hypercube) with score = exploration bonus
    """
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

# ============================================================
# HARNESS — the ~1h trial + the objective (shared, so the metric
# never drifts between trial notebook copies)
# ============================================================
def run_trial(config, experiments=None):
    """Run one trial (~1h). DRAFT — not wired yet.

    Trial definition (Anuar, 2026-08-29):
    - SYNTHETIC benchmark: targets drawn at true params (SYNTH_SEED=12345),
      14 experiments (1nW/3nW x Trans05-100), real n_target counts.
    - Returns per-experiment results (recovered mu, gamma, true values, NLL...).
    TODO(draft): port run_experiment from notebooks/17f-lambda0-scalefree.ipynb
    (cell 6) / 18b into this function (or into src/) so the template cell stays
    ~10 lines.
    """
    raise NotImplementedError("harness not ported yet — see 17f/18b notebooks")


def compute_objective(results):
    """MSE over the 14 exps + sampling uncertainty. DRAFT — not wired yet.

    Returns (objective, uncertainty). objective lower is better.
    uncertainty = SE of the per-experiment squared errors (or bootstrap CI),
    fed to the search algorithm. NO Fisher here (Anuar: optimization only).
    """
    raise NotImplementedError("objective not defined yet — TBD with Anuar")
