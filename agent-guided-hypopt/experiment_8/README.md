# experiment_8 — AG-HYPOPT: alternative distributional LOSS functions

## The idea
exp5 (schedule), exp6 (score design) and exp7 (bandwidth) all bottom out at a T-weighted objective of
**~0.078–0.082**, and exp7 showed the residual is *not* tuning: the **bandwidth rule** is first-order
(`target` wins), the coefficient is flat, and the error is concentrated in the **low-count (T05/T20)
μ–γ cells** — the sim/target model gap (**FM#8**: sim σ_fit 1.04–4.25× too narrow, growing with T;
**FM#6**: low-T γ deficit). The optimizer's loss is the **2-D KDE negative log-likelihood**, which leans
on the *mis-scaled* σ_fit channel.

experiment_8 asks a different question:

> **Does the CHOICE of distributional loss matter?** Replace the KDE NLL with a family of alternative
> discrepancies — several of them bandwidth-free — and let the campaign find which drives the (μ, γ)
> optimizer best.

A bandwidth-free / robust discrepancy (Wasserstein, energy, MMD) may stop the optimizer from trusting
the σ channel's wrong *scale* — a different attack on the same FM#8 wall.

## Frozen
- **Schedule** (exp5 winner): `lr_mu 0.1669`, `mu_anneal 0.3455`, `lr_gamma 0.4716`, `gamma_anneal 0.4723`.
- **Bandwidth** (exp7 winner): rule `target`, `h_f_scale 4.0`, `h_s_scale 3.6745…`.
- **γ trust region** (exp6 winner): `gamma_rel_cap 0.10736`.
- Everything else in the series-21 mechanism (Anuar's μ reward form, `σ_prop` score, no clipping, guard,
  init `0.5 × truth`, `n_runs=100`, `n_iter=30`).

## Search space (2 tunables, `space.json`)
| tunable | type | values / range | meaning |
|---|---|---|---|
| `loss_family` | choice | `kde` / `w1_2d` / `energy_2d` / `mmd_2d` / `w1_fwhm` / `cvm_fwhm` | the distributional loss |
| `sigma_weight` | float | 0.0 – 1.0 | weight of the σ_fit channel (`loss = loss_FWHM + sw · loss_σ`) |

The families (two 1-D channels combined by `sigma_weight`; all differentiable in γ):

- **`kde`** — the 2-D KDE NLL (exp5/6/7 design). **CONTROL: must reproduce exp6's 0.080328.**
- **`w1_2d`** — Wasserstein-1 (sorted quantile matching) on FWHM **and** σ_fit — bandwidth-free.
- **`energy_2d`** — energy distance on FWHM and σ_fit — bandwidth-free.
- **`mmd_2d`** — RBF-kernel MMD² over the 2-D cloud, **median-heuristic** kernel width (per step).
- **`w1_fwhm`** — Wasserstein-1 on FWHM only (drops the σ link → μ has no channel; control).
- **`cvm_fwhm`** — Cramér–von Mises-style squared quantile distance on FWHM only (control).

### How it plugs into the optimizer
The optimizer has two channels: **μ** via REINFORCE (`grad_mu = −(r − r̄)·score`, `score = (n−μ)/σ_prop²`)
and **γ** via a pathwise gradient. Each loss family supplies, per sim run *j*, a **reward** `r_j = −loss_j`
and a **pathwise derivative** `g_j = ∂loss_j/∂γ`; the γ update is `γ ← γ − lr·mean_j g_j`. This keeps the
existing skeleton and is **exactly** the KDE mechanism when `loss_family='kde'`.

### Scale normalization (`FAMILY_SCALE` + per-step reward rescaling)
Different losses live on different scales (nats vs MHz vs dimensionless), which would confound the comparison
under a **frozen** learning rate. Two normalizations make it a test of the loss **form**:

1. **γ channel:** each family is multiplied by `FAMILY_SCALE[family]`, calibrated **once** (see
   `tools/calibrate_family_scales.py`) so that at the shared frozen start its `|∂loss/∂γ|` matches the KDE
   control's. (`kde` = 1.0 by construction.)
2. **μ channel:** the per-run family reward is **rescaled per step** to the control's reward spread (the KDE
   log-likelihood on the very same draws). Necessary because non-KDE per-run losses span ~3 orders of
   magnitude in spread across cells (T05 vs T60), which the frozen `lr_mu` cannot accommodate — without this,
   `w1_2d` drove μ to ~1120 on 1nW T05 (a reward-scale artifact, removed → μ 4.7 → 6.0 there).

## Benchmark
8 experiments (`BENCHMARK_SUBSET`): 1nW/3nW × {Trans05, Trans20, Trans60, Trans100}. **Held-out** = the 6
others (1nW/3nW × Trans10/40/80), for post-campaign validation only.

## Data source
`USE_REAL_DATA = True` — real measured FWHM targets from `data/processed/fwhm_linewidths.csv`
(15-series load & filter: raw ×1000 → MHz, keep `fit_error/fwhm < 10`).

## Objective (pre-registered, unchanged from exp5/6/7)
```
err_i = min(rel_sq_i, CAP)              # CAP = 1.0
w(T)  = 0.25 + 0.75 * T/100
objective = Σ w_i * err_i / Σ w_i       # T-weighted mean over the 2*8 channel errors
uncertainty = std(err_i)/sqrt(n)
```

## Running
Trial worksheet: `trial_01.ipynb` (stamped); registry/protocol: `trials.json`.
Cap: `MAX_TRIALS=30` (~7.5 min/trial at 100×30 ⇒ ~3.75 h + analysis). Cell-by-cell with `nbrun.py`
(see `instructions.md`). Self-test: `python3 ag_hypopt.py`.

## Success bar
γ finite on all 8 cells **and** the T-weighted objective below the exp6 level (~0.086; beating the
**0.0803** headline is a bonus) — with the winner **holding on the 6 held-out experiments**.

## Authority
Hyperparameters are the agent's call within this space. The structural change (the loss-family knob + the
per-family scale normalization) was **pre-authorized by Anuar on 2026-09-20** and is frozen into this
experiment's snapshot; the campaign itself only tunes the two knobs.
