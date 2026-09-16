# experiment_5 — AG-HYPOPT on Anuar's μ reward + the `sigma_prop` score fix (real data)

## The idea
Tune the μ/γ learning-rate **schedules** of the series-21 mechanism, where the μ REINFORCE reward is
**Anuar's per-run Gaussian likelihood** and the μ score denominator is the **experiment's own
`sigma_prop`** (the 21h fix). Every trial = the frozen 21-series model (100×30, real targets, 8-exp
benchmark) with one candidate schedule; the objective is the combined relative MSE of (μ, γ).

This is the first campaign whose μ mechanism is *not* the 21a/17g responsibility reward — the previous
real-data campaign (`experiment_4`) found μ knobs-insensitive at the 0.5× attractor, so tuning that
reward was structurally capped. Here we let the campaign choose the schedules for the *new* mechanism.

## Benchmark
8 of the 14 experiments (`BENCHMARK_SUBSET`): 1nW/3nW × {Trans05, Trans20, Trans60, Trans100}.

## Data source
`USE_REAL_DATA = True` — real measured FWHM targets from `data/processed/fwhm_linewidths.csv`
(15-series load & filter: raw ×1000 → MHz, keep `fit_error/fwhm < 10`).

## Search space (5 tunables, `space.json`)
| tunable | range | note |
|---|---|---|
| `lr_mu` | 0.05 – 10 | μ step; reward scale is large, so low values matter |
| `lr_gamma` | 0.2 – 1.0 | γ step |
| `gamma_anneal` | 0.0 – 0.75 | γ LR anneal coefficient |
| `mu_anneal` | 0.0 – 1.0 | **NEW** — μ LR anneal coefficient (was hard-coded to 1.0) |
| `clip` | 1.0 – 50.0 | gradient clip (interacts with the reward scale) |

`n_runs`/`n_iter` are **fixed by the protocol, not searched** (they are budget/feasibility: a bigger
`n_runs` scores better partly by variance reduction and costs more — mixing them into TPE confounds the
campaign). If the `n_runs` effect is wanted, run it as a separate 1-D scan.

## Fixed parameters (`DEFAULT_CONFIG`)
`n_runs=100`, `n_iter=30` (21-series budget), `h_s_min=0.05`, `sigma_ref=10.0` (retained for reference —
**unused** while `MU_SCORE='sigma_prop'`).

## The μ mechanism (module flags in `ag_hypopt.py`)
- `MU_REWARD = 'loglik_mean'` → reward `r_j = mean_i log W_ij` (Anuar's per-run likelihood, per real
  scan). Other options: `'loglik_sum'` (exact 21g form; scale ∝ n_target) and `'responsibility'` (the
  21a/17g baseline). `loglik_mean` is the default so the reward scale does not vary with `n_target` and
  the schedule hyperparameters are meaningful across experiments.
- `MU_SCORE = 'sigma_prop'` → score denominator `(n_j − μ)/σ_prop²` (the 21h fix; correct REINFORCE
  score, consistent with the Fisher). Other option: `'sigma_ref'` (old fixed 10).
- μ schedule: `lr_mu · (1 − mu_anneal·t/n_iter)`; γ schedule: `lr_gamma · (1 − gamma_anneal·t/n_iter)`.

## Frozen structural choices
- z-form γ-score (`GAMMA_SCALE=True`, `H_REF=1.0`)
- `LAMBDA_MEAN = 0` (mean-matching anchor disabled)
- 2-D `(FWHM, σ_fit)` KDE kernel, Scott bandwidths, Lorentzian fit (2 params)

## Running the campaign
Per-trial worksheet: `trial_01.ipynb` (unexecuted), registry: `trials.json`. Cap: `MAX_TRIALS=40`.
A trial on the 8-exp benchmark at 100×30 is ≈ 9 min. Self-test:
`python3 ag_hypopt.py` (template API + tiny run).
