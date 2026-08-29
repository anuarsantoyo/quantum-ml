"""eval_harness — trial evaluation (DRAFT v1).

One shared place for the ~1h trial + the objective, so every trial notebook
computes the SAME metric (consistency across copies is critical — a drifted
metric silently poisons trials.json for the surrogate).

    run_trial(config) -> results            # the ~1h evaluation
    compute_objective(results) -> (objective, uncertainty)

Trial definition (Anuar, 2026-08-29):
- SYNTHETIC benchmark: targets drawn at true params (SYNTH_SEED=12345),
  14 experiments (1nW/3nW x Trans05-100), real n_target counts.
- Objective = MSE of recovered (μ, γ) vs true values over the 14 exps.
  Exact combination (MSE_μ + MSE_γ, normalized, ...) = TBD with Anuar.
- Uncertainty = sampling uncertainty of the MSE over the 14 exps
  (SE across per-exp squared errors, or bootstrap) — for the search algorithm.
- NO Fisher here (Anuar: focus on optimization only).

TODO(draft): port run_experiment from notebooks/17f-lambda0-scalefree.ipynb
(cell 6) into this module (or src/) so the template cell stays ~10 lines.
Runtime flag: trial cost ~ n_runs x n_iter (17f: 200x200 = ~3h). If n_runs /
n_iter enter the search space, add a feasibility cap in DEFAULT_SPACE.
"""
import numpy as np


def run_trial(config, experiments=None):
    """Run one trial (~1h). DRAFT — not wired yet.

    Returns a dict with per-experiment results (recovered mu, gamma, true
    values, NLL, ...). See 17f cell 6 / 18b for the machinery.
    """
    raise NotImplementedError("eval harness not ported yet — see 17f/18b notebooks")


def compute_objective(results):
    """MSE over the 14 exps + sampling uncertainty. DRAFT — not wired yet.

    Returns (objective, uncertainty). objective lower is better.
    uncertainty = SE of the per-experiment squared errors (or bootstrap CI).
    """
    raise NotImplementedError("objective not defined yet — TBD with Anuar")
