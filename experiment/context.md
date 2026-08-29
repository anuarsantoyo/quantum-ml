# context.md — qm-ml hyperparameter campaign (DRAFT v0)

> Draft for discussion. Refine with Anuar before trial_002.

## What this project is
ML-based parameter recovery for PLE (pulsed laser excitation) experiments.
A physical forward model maps (μ, γ) → photon counts → FWHM distributions. We invert it by
optimizing (μ, γ) so that simulated FWHM distributions match measured data under a
kernel-density (KDE) likelihood.

## The model (see src/ in the repo root)
- Photon count per shot: n ~ Gaussian(μ, σ_prop) + Poisson(λ) noise (`draw_fixed_noise`)
- FWHM from a Lorentzian fit of the photon stream at scale γ (`fit_profile`, n_iters=80)
- Likelihood: 2D KDE over (FWHM, σ) with Scott bandwidths H_F, H_S (H_S floor = H_S_MIN)
- Gradients: REINFORCE-style on kernel-weighted scores

## Parameters
- **μ** — mean photon count (main recovery target; true range 9.4–175.7 across exps)
- **γ** — FWHM scale in MHz (true: 8.5 for 1nW, 14.1 for 3nW)
- σ_prop, λ — fixed per experiment (Gregor's fits), not optimized

## Current best optimizer config (17f, run 2026-08-28)
- μ score: (n − μ)/σ_ref² with **σ_ref = 10** — scale-invariant, fixes high-T starvation
- LR_MU = 15, linear decay **without floor**, N_ITER = 200, N_RUNS = 200, CLIP = 10
- γ via KDE log-density gradient, LR_GAMMA = 0.5 with anneal ×(1 − 0.5t/N)
- LAMBDA_MEAN = 0 (mean-matching anchor shown harmful — 17d)
- SEED = 42, SYNTH_SEED = 12345 → deterministic per config

## Known physics / failure modes (17–18 series findings)
1. **Starvation**: score /σ_prop² → high-T μ steps ~0.1–0.5 photons, needs ~500 iters (fixed by σ_ref)
2. **Attractor**: REINFORCE drives μ toward the kernel-weighted mean E_B[n] = n̄
3. **Low-count γ noise**: n_target ≲ 500 (T05 exps) → γ gradients noisy; the anneal freezes γ there
4. **KDE skewness**: 3nW T40 target cloud is pathologically skewed → Scott bandwidth inflated → flat NLL
5. **⚠️ 18b (real data, verdict pending 2026-08-29)**: the real-data likelihood optimum appears to sit
   at ~half the reference μ. If confirmed, the *likelihood*, not the hyperparameters, is the problem.
6. λ_mean anchor: pure harm in every metric (17d)

## Data
- 14 real PLE experiments: 1nW/3nW × Trans05–100 → `data/raw_data/fwhm_1nW_240221/`, `fwhm_3nW_210221/`
- Reference "true" values from Gregor's fits (`data_explanation.md`); γ_true = median FWHM @ Trans100 / 2
- Synthetic benchmark: targets drawn at true params (SYNTH_SEED = 12345) with the real n_target counts

## Metrics (objective — definition TBD with Anuar)
- μ/γ RMSE, bias, rel-RMSE over the 14 exps; NLL; Fisher σ_μ, σ_γ (CRB at fitted point, M=500)
- Current reference: 17f synthetic → μ RMSE 1.40 (rel 5.8%), γ RMSE 2.27 (rel 20.2%)

## Constraints & rules
- ~1h per trial, 10–30 trials total, 4 CPUs — every trial must be defensible in writing
- **Structural changes** (score, likelihood, bandwidths, model) are OUT of scope for trials unless approved by Anuar
- **18b verdict may redirect the objective** — check status before trial_002
