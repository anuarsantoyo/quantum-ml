# experiment_2 — vanilla test of the AG-HYPOPT algorithm

## The idea
A deliberately small, fast instance of AG-HYPOPT, used to exercise the optimizer
itself (uniform warm-up, then TPE proposals), not to push the physics. The point is
to watch the algorithm propose, learn, and converge over a short campaign.

## Benchmark
8 synthetic experiments: 1nW and 3nW x {Trans05, Trans20, Trans60, Trans100}.
Targets are generated at the true parameters (SYNTH_SEED=12345), identical for every
trial. A subset of the full 14 (the 1st, 3rd, 5th, 7th transmissions) so each trial is
quick.

## Search space (4 tunables, space.json)
- sigma_ref     float  [5.0, 25.0]
- lr_mu         float  [5.0, 30.0]
- lr_gamma      float  [0.2, 1.0]
- gamma_anneal  float  [0.0, 0.75]

## Fixed parameters (DEFAULT_CONFIG in ag_hypopt.py)
- n_runs  = 100
- n_iter  = 100
- clip    = 10.0
- h_s_min = 0.05

## Frozen structural choices
- z-form gamma-score (GAMMA_SCALE=True, H_REF=1.0)
- sigma_ref mu-score
- LAMBDA_MEAN = 0
- mu LR linear decay, no floor; mu in [1,200]; gamma in [0.1,100]
- 4 workers

## Objective
Combined relative MSE of (mu, gamma) over the per-experiment errors, plus the sampling
SE as uncertainty. Lower is better.

## Budget
n_runs x n_iter = 100 x 100 = 10k (~30 min/trial on the 8-experiment benchmark).
Runtime cap 40000, which this stays well under.

## Algorithm
- n_initial = 5: the first 5 completed trials are uniform random draws.
- Then the trials phase: TPE (Good/Bad split) proposals, with explore_slots = 2
  fully-random candidates reserved in every batch.
- Campaign cap: MAX_TRIALS = 20 (cell 2 of the trial notebooks).

## Baseline
No stored baseline. The current best is the minimum objective over the recorded trials,
recomputed every trial.

## Authority
Hyperparameters are the agent's call within this space. Structural changes (score,
likelihood, model) need Anuar's OK.
