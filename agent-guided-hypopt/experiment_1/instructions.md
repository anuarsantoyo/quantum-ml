# instructions.md — PAGHO: how to run a trial

> **PAGHO** = Physics-informed Agent-Guided Hyperparameter Optimization.
> This is the short **recipe card** for the agent. The physics lives in `context.md`; the
> step-by-step worksheet lives in `template.ipynb` (every trial notebook is a copy of it and
> tells you exactly what to do at each cell). *DRAFT — reviewed with Anuar (2026-08-29).*

---

## Your role

You are the agent running **one trial** of a PAGHO campaign. One trial ≈ 1 hour — expensive.
Be sample-efficient: every pick must be defensible in writing. The algorithm proposes, you decide
(physics + history), you run, you document.

## The recipe (do this, in order)

1. **Read `context.md`** — the physics and model background. Skim; it is stable.
2. **Read `trial_00.ipynb`** — this experiment's definition and *current state*. Read first among
   the trials.
3. **Read all previous `trial_XXX.ipynb` notebooks** — their **Key insight** lines give you the
   memory index (the template prints them); read fully where depth matters.
4. **Read `trials.json`** — structured history only (config, objective, uncertainty, summary).
5. **Copy `template.ipynb` → `trial_XXX.ipynb`** with the next free number.
6. **Follow the notebook.** It is a worksheet: `▶` cells you run, `✍️` cells you write before
   continuing. It loads history, proposes candidates, and guides your analysis, selection, run,
   reflection, and the `trials.json` update — in that order, cell by cell.
7. **When done**: the notebook has recorded the trial and updated the current best. Report.

## Rules (compressed — the notebook assumes these)

- ⛔ **Never "Run All"** — `✍️` cells must be written by you between execution phases.
- **Execute in place**: `papermill trial_XXX.ipynb trial_XXX.ipynb --log-output` — results land
  directly in the notebook; one file = the record.
- **Fail fast**: if a candidate is obviously broken, abort the run, record `objective: null` +
  the error, and reflect on the cause. Never silently skip a failure.
- **Structural changes** (score, likelihood, bandwidths, model): propose in the notebook, do NOT
  run them without Anuar's explicit OK.
- **Protocol is fixed** per experiment (objective, benchmark, seeds, space — defined in
  `trial_00`). Do not change it per trial; propose protocol changes to Anuar.

## The algorithm contract (one module per experiment)

- ALL scripts live in ONE module: `pagho.py` (algorithm + harness).
- `propose(trials, space, n_candidates=10, seed=None) -> [{"config", "score"}]` — `score` is the
  Expected Improvement (EI) from the surrogate. Consider it, but decide with physics.
- `run_trial(config)` + `compute_objective(results)` define the trial and the metric — identical
  across all trials of an experiment (a drifted metric poisons `trials.json`).
- Swapping `pagho.py` swaps the algorithm (TPE today; GPBO/CMA-ES possible in a future
  experiment). Decision TPE vs GPBO: pending.
