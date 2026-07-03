# 2026-07-03

## Morning: Presentation 🎉

Anuar finished and gave the presentation today. It was a big success — they loved it and were impressed with how fast the development progressed. Anuar mentioned it was thanks to Pukky 🔬

He also met the professor afterward. It was a bit awkward — the professor seemed to think Anuar wanted something from him, but Anuar was just being sociable. Overall a great outcome.

## Afternoon: REINFORCE Toy for μ (mean photon count) — Full Summary

### Goal
Build a toy notebook that uses REINFORCE to optimize the **mean photon count** μ through the discrete rounding step (int n), keeping γ fixed. Gamma is already differentiable via implicit differentiation — this tackles the other half of the problem.

### What we built
- **File:** `notebooks/reinforce_N_opt_toy.ipynb`
- Based on `differentiable-gamma-simplified.ipynb` (kept: full MC simulation, pseudo-Voigt fitting, L-BFGS per run)
- Replaced: MMD² → Wasserstein-1 loss, gamma optimization → REINFORCE for μ
- Learnable σ (parameterized as `log_σ` for positivity)

### Mistake #1: Per-run n sampling (correct architecture, wrong gradient)
**Initial implementation:** Each run sampled its own `n_i ~ N(μ, σ)` → 200 different n values → one Wasserstein loss → REINFORCE gradient averaged over all runs.

**Problem:** The score `∇_μ log P(n_i|μ,σ) ≈ (n_i - μ)/σ²` averages to ~0 because `avg_n ≈ μ` by construction. The signal is just random noise `σ/√B`. No convergence.

**Realization:** REINFORCE with a batch-level loss and per-element scores doesn't work when the loss is shared — all runs get the same advantage, and the gradient is dominated by random fluctuations.

### Mistake #2: Learnable sigma with wrong architecture
**Attempt:** Made σ learnable, initialized at 20.

**Problem:** σ gradient always pushed σ to the cap (80). The score `∇_σ log P(n|μ,σ)` wants to match the variance of observed n to σ² — but the observed n are sampled from N(μ,σ), so the expectation is always ~0 regardless of whether μ is right. No useful signal.

### Fix: One nbar per simulation (the right architecture)
**New approach — meta-distribution:** Draw ONE nbar from N(μ, σ) per iteration, use it for ALL 200 runs:

```
Meta:      μ ─────────── σ
               ↘        ↙
           nbar ~ N(μ, σ)   ← ONE REINFORCE score
                │
          200 runs share nbar
           (each with σ_phys per-run noise)
                │
          1 Wasserstein loss
                │
       ∇_μ = (L - b) · ∇_μ log P(nbar | μ, σ)
```

Now the advantage (L - b) is **actually conditioned on the single nbar value**. When nbar=48 gives low loss, gradient pushes μ → 48. When nbar=20 gives high loss, gradient pushes μ away from 20. Real signal.

### Realization: This is like hyperopt
This meta-distribution over nbar sampled from N(μ, σ) and evaluated per iteration is structurally identical to hyperparameter optimization (hyperopt):
- Propose nbar from N(μ, σ) ← like hyperopt's search space
- Evaluate loss ← like hyperopt's objective
- Update belief (μ, σ) ← like hyperopt's surrogate model update

Main difference: REINFORCE only remembers the current loss + decayed baseline (no memory of past evaluations), while hyperopt (GP/TPE) builds a model over all evaluations.

### Key design decisions
- **Wasserstein-1 loss** (instead of MMD² — no kernel bandwidth tuning)
- **SimParams:** γ=20 fixed, σ_phys=6 (physical per-scan noise), λ=2 (background)
- **Target:** synthetic data at nbar_true=50 with σ_phys
- **REINFORCE details:** EMA baseline α=0.05, gradient clipping, μ∈[1,200], σ∈[1,80]
- **Next steps per user:** Integrate dloss/dn, research n+1/n-1 approach

### TODO: Per-run REINFORCE with individual Wasserstein losses

**Idea (discussed, not implemented):** Instead of one shared loss per simulation, compute a **per-run loss** for each of the 200 FWHM values. Each FWHM_i gets its own advantage `(L_i - b)`, so `∇_μ = mean_i (L_i - b) · ∇_μ log P(n_i | μ, σ)`.

**Per-run loss:** The absolute distance from FWHM_i to its nearest point in the experimental FWHM distribution (or rank/percentile in the target CDF).

**Why this might work:** n_i and L_i are correlated — higher n → better fit → lower L_i. This gives a real, non-zero gradient signal even though `avg_n ≈ μ`.

**Refinement suggestion:** Use the **percentile** of FWHM_i in the target CDF instead of raw distance, to be more robust to outliers.

**Next:** Discuss implementability with Anuar. (Parked for later.)
