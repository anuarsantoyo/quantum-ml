# instructions.md — AG-HYPOPT: how to run a trial

AG-HYPOPT = Agent-Guided Hyperparameter Optimization.

## Recipe
1. **Find the last trial** — look at the trial notebooks (`trial_00.ipynb` = experiment
   anchor, read it first) and `trials.json`.
2. **Make a new trial** — copy `template.ipynb` to the next number (`trial_XXX.ipynb`).
3. **Follow the notebook** — it tells you everything: run `▶` cells, write `✍️` cells,
   analyze, choose, justify, run, conclude, save.

## Rules
- ⛔ Never "Run All" — `✍️` cells are written by you between execution phases.
- Execute in place (`papermill x.ipynb x.ipynb`) — results land in the notebook.
- Fail fast: abort broken candidates, record `objective: null` + reason.
- Structural changes (score, likelihood, model): propose, but need Anuar's OK.
- Protocol (objective, benchmark, space) is fixed — defined in `trial_00`.

Everything else lives where it belongs:
physics → `context.md` | process → the trial notebook | algorithm contract → `ag_hypopt.py` (AGHyperopt) + `space.json` | trial harness → `template.ipynb`
