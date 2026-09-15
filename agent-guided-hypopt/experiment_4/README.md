# experiment_4 — AG-HYPOPT on REAL data

## The idea
Tune the **frozen** inversion pipeline on the **real measured** PLE line-width data
(not the synthetic benchmark): the optimizer proposes `(sigma_ref, lr_mu, lr_gamma,
gamma_anneal)` and each trial fits the 8-experiment subset against its real FWHM
distribution. Still a test campaign — smoke budget `n_iter=10`, cap 20 trials.

## Data (real, not synthetic)
Set `USE_REAL_DATA = True` in `ag_hypopt.py`. Targets come from
`data/processed/fwhm_linewidths.csv`, loaded with the **15-series convention**:
select `(power_nW, transmission)`, scale raw → MHz (`×1000`), keep valid rows with
`fit_error / fwhm < 10`. Bandwidths: `H_F = std(FWHM)·n^(−1/6)`,
`H_S = max(std(err)·n^(−1/6), h_s_min)`.
Sanity: `median(FWHM @ Trans100)/2` = **8.5 MHz** (1 nW) / **14.1 MHz** (3 nW) = `gamma_true`.

## Benchmark
8 experiments: 1nW and 3nW × {Trans05, Trans20, Trans60, Trans100} (`BENCHMARK_SUBSET`).

## Search space (4 tunables, `space.json`)
- `sigma_ref` float [5, 25]
- `lr_mu` float [5, 30]
- `lr_gamma` float [0.2, 1.0]
- `gamma_anneal` float [0.0, 0.75]

## Fixed parameters (`DEFAULT_CONFIG`)
- `n_runs = 100`, `n_iter = 10` (smoke budget — restore `n_iter=100` for the full budget)
- `clip = 10.0`, `h_s_min = 0.05`

## Frozen structural choices
- z-form gamma-score (GAMMA_SCALE=True, H_REF=1.0) · sigma_ref mu-score · LAMBDA_MEAN=0
- mu LR linear decay, no floor; mu in [1,200]; gamma in [0.1,100]; 4 workers

## Objective
Combined relative MSE of (mu, gamma) over the per-experiment errors, plus the sampling
SE. Reference truth: mu from Gregor's fits, gamma = median FWHM @ Trans100 / 2.
Lower is better. **No Fisher.**

## Budget
`n_runs × n_iter`; ~6.5 min/trial at `100×10` on the 8-experiment subset.

## Algorithm
- `n_initial = 5` uniform warm-up, then TPE proposals (`explore_slots = 2`).
- Proposal RNG seeded per trial from the trial id (`trial_seed(TRIAL_ID)`).
- Campaign cap: `MAX_TRIALS = 20`.

## Figures (inline in cell 5)
- `plot_paths` — (μ, γ) phase-space paths, 2 cols (laser) × 4 rows (transmission),
  with the honest init as a hollow marker at `0.5·true`. No Fisher.
- `plot_parallel` — parallel-coordinates view of all trials vs objective.

## Baseline
None stored; current best = min objective over recorded trials.

## Authority
Hyperparameters are the agent's call within this space. Structural changes (score,
likelihood, model) need Anuar's OK.
