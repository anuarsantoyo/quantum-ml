# experiment_5 — AG-HYPOPT: schedule tuning for the series-21 μ mechanism (real data, NO clipping)

## The idea
Tune the **learning-rate schedules** of the series-21 mechanism:
- μ reward = **Anuar's per-run likelihood** (`MU_REWARD='loglik_mean'`, i.e. `r_j = mean_i log W_ij`);
- μ score denominator = the experiment's own **`sigma_prop`** (`MU_SCORE='sigma_prop'`, the "21h fix");
- **no gradient clipping and no parameter clamps** (21i showed the clip was masking a *scale* problem:
  unclipped + a rescaled LR took μ from rel-RMSE 550% to 41%).
Everything else is the frozen 17g/series-21 model (real targets, 100×30, 8-exp benchmark).

## Benchmark
8 of the 14 experiments (`BENCHMARK_SUBSET`): 1nW/3nW × {Trans05, Trans20, Trans60, Trans100}.

## Data source
`USE_REAL_DATA = True` — real measured FWHM targets from `data/processed/fwhm_linewidths.csv`
(15-series load & filter: raw ×1000 → MHz, keep `fit_error/fwhm < 10`).

## Search space (4 tunables, `space.json`) — no `clip` dimension
| tunable | range | note |
|---|---|---|
| `lr_mu` | 0.005 – 0.5 | measured raw \|grad_mu\| (mean reward) = 2.0 / 11.2 / 176.6 (min/median/max) |
| `mu_anneal` | 0.0 – 1.0 | μ LR decay coefficient (was hard-coded to 1.0) |
| `lr_gamma` | 0.05 – 1.0 | γ step (raw \|grad_gamma\| ≤ 4.5) |
| `gamma_anneal` | 0.0 – 0.9 | γ LR decay — the lever against the low-T γ divergence |

`n_runs`/`n_iter` are **fixed by the protocol, not searched** (budget/feasibility knobs).

## No clipping + divergence guard
- `CLIP = float('inf')`; the parameter clamps are **removed** (μ, γ unbounded).
- `DIVERGENCE_GUARD = {'mu': (0.2, 1500.0), 'gamma': (0.02, 500.0)}` — if a parameter leaves the band the
  experiment **stops** (recorded `diverged`/`diverged_at`). This is a *stop*, not a clamp: it only prevents
  a runaway photon count from stalling the fits.

## Objective (pre-registered)
```
err_i = min(rel_sq_i, CAP)              # CAP = 1.0  (≈100% relative error)
w(T)  = 0.25 + 0.75 * T/100             # T = transmission (5…100): graded toward high T, 0.25 floor
objective = Σ w_i * err_i / Σ w_i       # T-weighted mean over the 2*n_exps channel errors
uncertainty = std(err_i)/sqrt(n)        # sampling SE across cells (reporting only; NO Fisher)
```
The trial report keeps the **unweighted per-cell table** (with a `!D` flag on diverged cells) so the full
T-behaviour stays visible even though the search is graded toward high T.

## Fixed parameters (`DEFAULT_CONFIG`)
`n_runs=100`, `n_iter=30`, `mu_anneal=0.5`, `gamma_anneal=0.5`, `lr_mu=0.05`, `lr_gamma=0.5`,
`clip=inf`, `h_s_min=0.05`, `sigma_ref=10.0` (retained for reference; **unused**).

## Frozen structural choices
- z-form γ-score (`GAMMA_SCALE=True`, `H_REF=1.0`), `LAMBDA_MEAN=0`
- 2-D `(FWHM, σ_fit)` KDE kernel, Scott bandwidths, Lorentzian fit (2 params)
- init at `0.5 × truth`

## Running
Trial worksheet: `trial_01.ipynb` (unexecuted); registry/protocol: `trials.json`.
Cap: `MAX_TRIALS=40` (~7.5 min/trial at 100×30 on the 8-exp benchmark ⇒ ~5 h).
Self-test: `python3 ag_hypopt.py`.

## Success bar
γ finite on all 8 cells **and** μ rel-RMSE < 40% with μ bias → 0 (beat 21i's 40.6% / +7.3%).
