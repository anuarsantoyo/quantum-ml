# experiment_template — the base folder for a new AG-HYPOPT experiment

## The idea
<One line: what this experiment tries to do.>

This folder is the seed for a new experiment — a clean snapshot of the **validated
`experiment_3_pukky` campaign (2026-09-11)**, with the trial chain reset. To start one:

```
cp -r experiment_template experiment_N
```

then rewrite this README, `context.md`, and `space.json` to match the new question, and reset
the trial chain: delete executed trial notebooks and ship a fresh unexecuted `trial_01.ipynb`
copied from this folder's `template.ipynb` (with the number `01` stamped in) and an empty
`trials.json` (`{"trials": []}`).

## Benchmark
<Which experiments run. Set `BENCHMARK_SUBSET` in `ag_hypopt.py`: `None` = full 14, or a list
of experiment names. Snapshot default: the 8-experiment subset — 1nW/3nW × {Trans05, Trans20,
Trans60, Trans100}.>

## Search space
<The tunables and their ranges, as declared in `space.json`. Snapshot default:
`sigma_ref [5,25]`, `lr_mu [5,30]`, `lr_gamma [0.2,1.0]`, `gamma_anneal [0,0.75]`.>

## Fixed parameters (`DEFAULT_CONFIG` in `ag_hypopt.py`)
<Snapshot default: `n_runs=100`, `n_iter=10` (a cheap smoke budget — **restore `n_iter=100`**
for the full `100x100` budget), `clip=10.0`, `h_s_min=0.05`.>

## Frozen structural choices
- z-form gamma-score (GAMMA_SCALE=True, H_REF=1.0)
- sigma_ref mu-score
- LAMBDA_MEAN = 0
- mu LR linear decay, no floor; mu in [1,200]; gamma in [0.1,100]
- 4 workers

## Objective
Combined relative MSE of (mu, gamma) over the per-experiment errors, plus the sampling SE as
uncertainty. Lower is better. **No Fisher.**

## Budget
`n_runs x n_iter`; ~6.5 min/trial at `100x10` on the 8-experiment subset.

## Algorithm
- `n_initial = 5`: the first 5 completed trials are uniform random draws.
- Then the trials phase: TPE (Good/Bad split) proposals, with `explore_slots = 2`
  fully-random candidates reserved in every batch.
- The proposal RNG is seeded **per trial** from the trial id (`trial_seed(TRIAL_ID)` in cell 2),
  so every trial draws a fresh batch.
- Campaign cap: `MAX_TRIALS = 20` (cell 2 of the trial notebooks).

## Running a trial (cell by cell, via `nbrun.py`)
A trial is one hyperparameter configuration of the frozen pipeline. Run the trial notebook
**one cell at a time** with `nbrun.py` (persistent kernel, saves in place). Cell 5 is the long
one (~6.5 min); launch it detached and poll. See `instructions.md` for the exact recipe.
Cell 5 renders **two inline figures**: the (μ,γ) phase-space paths (`plot_paths`) and the
parallel-coordinates campaign view (`plot_parallel`).

## Baseline
<Stored baseline, or "none; current best = min objective over recorded trials".>

## Authority
Hyperparameters are the agent's call within this space. Structural changes (score, likelihood,
model) need Anuar's OK.
