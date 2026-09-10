# experiment_template — the base folder for a new AG-HYPOPT experiment

## The idea
<One line: what this experiment tries to do.>

This folder is the seed for a new experiment. To start one, copy it
(`cp -r experiment_template experiment_N`), then rewrite this README, `context.md`, and
`space.json` to match the new question, and reset the trial chain: delete executed trial
notebooks and ship a fresh unexecuted `trial_01.ipynb` copied from the folder's
`template.ipynb`.

## Benchmark
<Which experiments run. Set BENCHMARK_SUBSET in ag_hypopt.py: None = full 14, or a list of
experiment names.>

## Search space
<The tunables and their ranges, as declared in space.json.>

## Fixed parameters
<n_runs, n_iter, and anything else not tuned, from DEFAULT_CONFIG in ag_hypopt.py.>

## Frozen structural choices
- z-form gamma-score (GAMMA_SCALE=True, H_REF=1.0)
- sigma_ref mu-score
- LAMBDA_MEAN = 0
- mu LR linear decay, no floor; mu in [1,200]; gamma in [0.1,100]
- 4 workers

## Objective
<What is minimized and how the uncertainty is defined.>

## Budget
<n_runs x n_iter and the resulting wall-clock per trial.>

## Algorithm
- n_initial: completed trials of uniform random draws before TPE proposals start.
- explore_slots: fully-random candidates reserved in every trials-phase batch.
- Campaign cap: MAX_TRIALS in the trial notebooks (cell 2).

## Baseline
<Stored baseline, or "none; current best = min objective over recorded trials".>

## Authority
Hyperparameters are the agent's call within this space. Structural changes (score,
likelihood, model) need Anuar's OK.
