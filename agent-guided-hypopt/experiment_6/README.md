# experiment_6 — AG-HYPOPT: score-DESIGN campaign for the real-data mu attractor (FM#8)

## The idea
experiment_5 tuned the **learning-rate schedule** and hit a plateau: best objective **0.1074**
(trial_21), honest level ~0.108–0.113, whose residual is (a) the **mu floor** and (b) a **sporadic
gamma divergence** — and neither is schedule-controlled (no knob correlation exceeded |r| = 0.2).
The 19-series autopsy pinned the mu floor to **failure mode #8**: on real data the REINFORCE count
channel is *dead* (KDE responsibility ≈ uniform at 1100–2500 points), so mu can only be informed
through the **sigma_fit channel**, whose **shape** is structurally mismatched (simulated sigma_fit
is 1.04–4.25× too narrow, growing with T). The likelihood therefore develops a spurious **low-mu
mode at mu/mu_true ≈ 0.30–0.50**, and the optimizer descends into it unless the sigma-channel pull
is damped.

experiment_6 freezes the schedule at the experiment_5 winner and searches the **score design**:

- **`sigma_weight`** ∈ [0, 1] — damps the sigma_fit channel of the 2-D KDE kernel
  (`W = exp(-0.5 (dF/H_F)^2 - 0.5·sw·(dS/H_S)^2)`). 1 = the frozen exp5 design, 0 = FWHM-only
  (a control that removes the mu information link entirely). **This is the report's #1 fix
  ("damped sigma-gradient").**
- **`h_f_scale`, `h_s_scale`** ∈ [0.25, 4] — multipliers on the Scott bandwidths H_F, H_S
  (FM#4: one target is pathologically skewed → inflated Scott bandwidth flattens the likelihood).
- **`gamma_rel_cap`** ∈ [0, 0.5] — a per-step **relative** gamma trust region (`|dgamma| ≤ cap·gamma`,
  0 = off). This is the experiment_5 §6.1 recommendation: bound the unbounded low-count gamma
  gradient without touching mu (so it does not revive the 21i clip-masking problem).

Everything else is the frozen series-21 / experiment_5 mechanism.

## Benchmark
8 of the 14 experiments (`BENCHMARK_SUBSET`): 1nW/3nW × {Trans05, Trans20, Trans60, Trans100}.

## Data source
`USE_REAL_DATA = True` — real measured FWHM targets from `data/processed/fwhm_linewidths.csv`
(15-series load & filter: raw ×1000 → MHz, keep `fit_error/fwhm < 10`).

## Search space (4 tunables, `space.json`)
| tunable | range | note |
|---|---|---|
| `sigma_weight` | 0.0 – 1.0 | sigma_fit channel damping (FM#8 fix); 1 = frozen exp5, 0 = FWHM-only |
| `h_f_scale` | 0.25 – 4.0 | multiplier on H_F (Scott, FWHM channel) |
| `h_s_scale` | 0.25 – 4.0 | multiplier on H_S (Scott, sigma channel; floored by `h_s_min=0.05`) |
| `gamma_rel_cap` | 0.0 – 0.5 | per-step relative gamma trust region (0 = off) |

`n_runs`/`n_iter` are fixed by the protocol (100×30), not searched. The schedule
(`lr_mu`, `mu_anneal`, `lr_gamma`, `gamma_anneal`) is **frozen** to the experiment_5 winner.

## Fixed parameters (`DEFAULT_CONFIG`, in `ag_hypopt.py`)
`n_runs=100`, `n_iter=30`, `lr_mu=0.1669`, `mu_anneal=0.3455`, `lr_gamma=0.4716`,
`gamma_anneal=0.4723`, `clip=inf`, `h_s_min=0.05`, `sigma_ref=10.0` (retained; unused under
`MU_SCORE='sigma_prop'`). Defaults of the new knobs (`sigma_weight=1`, scales `1`, cap `0`)
reproduce the experiment_5 mechanism **exactly**.

## Frozen structural choices
- z-form gamma-score (`GAMMA_SCALE=True`, `H_REF=1.0`), `LAMBDA_MEAN=0`
- Anuar's mu reward (`MU_REWARD='loglik_mean'`), mu score denominator `sigma_prop` (`MU_SCORE`)
- 2-D `(FWHM, sigma_fit)` KDE kernel, Scott bandwidths, Lorentzian fit (2 params)
- no gradient clipping, divergence **guard** (stop if mu∉(0.2, 1500) or gamma∉(0.02, 500))
- init at `0.5 × truth`

## Objective (pre-registered)
```
err_i = min(rel_sq_i, CAP)              # CAP = 1.0
w(T)  = 0.25 + 0.75 * T/100             # graded toward high T, 0.25 floor
objective = Σ w_i * err_i / Σ w_i       # T-weighted mean over the 2*8 channel errors
uncertainty = std(err_i)/sqrt(n)        # sampling SE across cells (reporting only; NO Fisher)
```

## Running
Trial worksheet: `trial_01.ipynb` (unexecuted); registry/protocol: `trials.json`.
Cap: `MAX_TRIALS=30` (~8 min/trial at 100×30 on the 8-exp benchmark ⇒ ~4 h + analysis).
Cell-by-cell with `nbrun.py` (see `instructions.md`). Self-test: `python3 ag_hypopt.py`.

## Success bar
γ finite on all 8 cells **and** the T-weighted objective below the experiment_5 honest level
(~0.108–0.113; beating the 0.1074 headline is a bonus), with the μ rel-RMSE / bias improved and
fewer/zero diverged cells.

## Authority
Hyperparameters are the agent's call within this space. The structural change (the configurable
σ-channel + γ trust region) was **pre-authorized by Anuar on 2026-09-19** and is frozen into this
experiment's snapshot; the campaign itself only tunes the four knobs.
