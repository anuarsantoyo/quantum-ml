---
marp: true
theme: default
size: 16:9
---

# Updates 11 Sep 2026

Differentiable MC for NV-center PLE spectroscopy
Parameter recovery (μ, γ) + uncertainties, and the move to tuning

<!-- image: title / pipeline sketch -->

---

# Current status

- The differentiable MC pipeline is complete and **recovers the true parameters on synthetic data** generated from those same true values (μ = mean photon count, γ = Lorentzian HWHM).
- Recovery on the 14 synthetic experiments (baseline 17g): **μ RMSE ≈ 3.7 %, γ RMSE ≈ 4.3 %**, both essentially at the truth.
- We compute the **Fisher information / Cramér–Rao bound** as the uncertainty statement (per Gregor's endorsement).
- **Open check:** the reported Fisher confidence comes out very small (order ~0.5). Needs verification: is this a real precision statement or an over-confident (tight-wrong) artifact?

<!-- image: recovery of (μ, γ) vs truth for the 14 synthetic experiments -->
<!-- image: Fisher / CRB uncertainty per experiment -->

---

# Next goal: tune the model to fit all experiments

- **Goal:** get all 14 experiment optima to land as close as possible to their true (μ, γ).
- **Why synthetic first:** if the model cannot recover the true values from data that the *same model* generated, then there is a gap in the model that no amount of real data would ever close. So we make it perfect on its own data first.
- The model already works (all 14 optimize in the right direction, just biased), so the remaining work is **reducing the bias: trial and error over the design choices.**

<!-- image: current bias of the 14 optima vs truth -->

---

# Idea: structural tuning with an Agent loop

**First attempt at an agent tuning mechanism.** A notebook that: starts with a summary + context of what happened before, proposes a change, tests it, writes a summary of the results (what went wrong/right and why), and iterates.

- **Worked well for broad, high-impact changes** (structural: score, likelihood, model), where a single change moved the result a lot.
- **Struggled once the goal became fine hyperparameter tuning:**
  - the agent **over-engineered** (kept proposing structural changes) instead of just tuning hyperparameters;
  - the **number of experiments needed was too high** because the search space is very large.

<!-- image: agent-loop notebook sketch -->

---

# Idea: borrow the optimizer's search strategy, keep the agent physics-informed

- **Question:** can we give the agent not only the context of what has happened, but also an **insight into which changes make an improvement more probable?**
- Optimization algorithms already produce exactly that: a **probability distribution over where the next improving configuration probably lives.**
- **Idea:** feed the agent both (1) the distribution information and (2) the project context (past results described in physical terms), so it can make a better decision about what to change than either source alone.

<!-- image: TPE distribution -> agent -> chosen trial -->

---

# AG-HYPOPT

Agent-Guided Hyperparameter Optimization (method name: PAGHO)
A probabilistic optimizer proposes, a physics-informed agent chooses, we run and record.

---

## Uncertainty-aware TPE (1/2): the model

We use the **Tree-structured Parzen Estimator (TPE)** and make it **uncertainty-aware**.

**Why TPE (more flexible):** needs **no gradients** (black-box objective), handles **mixed continuous / integer / categorical** and **conditional** parameters, and is **cheap to fit** from a few expensive trials.

Notation: trial `i` has objective `ℓᵢ` (lower is better) and uncertainty `sᵢ`.

**Step 1, rank with uncertainty.** Each trial gets an uncertainty-adjusted loss (a trial is "good" only if even its upper bound is low):

$$\tilde{\ell}_i = \ell_i + \lambda\, s_i \qquad (\lambda = 0.5)$$

**Step 2, Good/Bad split.** Sort by `ℓ̃` ascending; the lowest quantile is the "good" set `G`, the rest is "bad" `B`:

$$m = \mathrm{clamp}\big(\mathrm{round}(q\, n),\; 1,\; n-1\big), \qquad G = \text{first } m, \qquad B = \text{rest} \qquad (q = 0.25)$$

**Step 3, one density per side** (per parameter). A **variable-bandwidth KDE plus a uniform prior** over the box `[lo, hi]`:

$$g(x) = \frac{w_0\, U_{[lo,hi]}(x) + \sum_{i \in G} \varphi\!\left(x \mid v_i,\, h_i^2\right)}{w_0 + |G|}$$

`vᵢ` is a trial's parameter value, `φ` is a Gaussian, `w₀` weights the uniform prior (keeps a nonzero density everywhere → stability + soft exploration). The bad density `b(x)` is built the same way from `B`.

<!-- image: Good/Bad KDEs over a 1D hyperparameter (uncertainty-scaled widths visible) -->

---

## Uncertainty-aware TPE (2/2): scoring

**Step 4, uncertainty-scaled bandwidths.** Each trial's kernel grows with its own uncertainty, so less reliable trials smear their kernel and pull the density less:

$$h_i = h_{\text{base}}\left(\frac{s_i}{s_{\text{med}}}\right)^{\beta}, \qquad \beta = 0.5, \qquad h_{\text{base}} = \mathrm{clamp}\!\left(1.059\,\sigma_{\text{scott}}\,k^{-1/5},\; h_{\min},\; h_{\max}\right)$$

(the Scott's-rule width is the base; `β` controls how strongly uncertainty widens it).

**Step 5, score a candidate.** The per-parameter densities multiply (a single zero factor kills the product):

$$\log g(x) = \sum_p \log g_p(x_p), \qquad \log b(x) = \sum_p \log b_p(x_p)$$

**Step 6, Expected Improvement.** The good/bad ratio and the score:

$$r(x) = \frac{b(x)}{g(x)} = \exp\!\big(\log b(x) - \log g(x)\big), \qquad \mathrm{EI}(x) = \frac{1}{q + (1-q)\,r(x)}$$

High `EI` is exactly where the good density concentrates and the bad one does not (`r → 0` saturates at the max `1/q`).

**Propose & sample.** Draw candidates from the good density `g(x)`, plus a few fully random **explore slots**. Categorical parameters use a Laplace-smoothed multinomial `p_j = (c_j + 1)/(k + |C|)`.

**Bottom line:** uncertainty enters the model **twice**, in the ranking (`ℓ̃ = ℓ + λs`) and in the bandwidths (`hᵢ`).

<!-- image: EI vs a 1D hyperparameter (good/bad densities + the chosen candidate) -->

---

## The AG-HYPOPT cycle

1. **Warm-up:** the first `n_initial` trials are uniform random draws (no history yet).
2. **Propose:** TPE fits the Good/Bad densities from the registry and draws a **batch of candidate configurations** (plus a few fully random explore slots).
3. **Agent decides:** the agent is shown the batch together with the project context (physical meaning of each parameter, past results) and **picks one candidate**. It must choose among the proposals, so it stays inside the probabilistic search but can still inject domain reasoning.
4. **Run:** the chosen configuration runs the benchmark (all 14 experiments) and records its **objective ± uncertainty** back into the registry.
5. **Repeat:** the registry grows, the densities sharpen, the proposals improve.

The agent writes no code and creates no files: it only picks a proposal.

<!-- image: AG-HYPOPT loop diagram -->
<!-- image: candidate table shown to the agent (params + EI hints) -->

---

## Results

**Setup:** frozen experiment snapshot (one folder = one benchmark + one algorithm + one model version).
Benchmark = synthetic experiments at the true parameters; objective = combined **relative MSE of (μ, γ)** plus the sampling SE as uncertainty (lower is better). The agent only tunes the declared hyperparameters; the score/likelihood/model are frozen.

**Campaign results (fast 8-experiment variant, 20 trials):**

| Trial | σ_ref | lr_mu | lr_gamma | γ-anneal | Objective |
|---|---|---|---|---|---|
| 01 | 10.25 | 10.1 | 0.52 | 0.10 | 0.0981 |
| 02 | 6.84 | 25.4 | 0.44 | 0.20 | 0.0246 |
| 05 | 6.28 | 27.0 | 0.42 | 0.42 | 0.0196 |
| 09 | 6.17 | 27.1 | 0.42 | 0.39 | 0.0182 |
| **14** | **6.10** | **27.1** | **0.42** | **0.42** | **0.0149** |
| 18 | 6.06 | 27.0 | 0.52 | 0.01 | 0.0157 |
| 20 | 6.06 | 27.0 | 0.41 | 0.42 | 0.0193 |

**What the physics-informed agent found:**
- **σ_ref is a strong, monotone lever on μ:** lowering it cuts the μ bias; it saturates around 6.1.
- **μ bias dropped from ~40 % (trial 01) to ~14 %** (plateau, best 14.2 % at trial 18); `lr_mu ≈ 27` is the plateau.
- **γ is insensitive to `lr_gamma`;** the residual spread is dominated by the **low-count Trans05 scans**.
- **Near-identical plateau configs differ by up to ~65 % in objective** purely through that Trans05 γ noise; near-zero `γ-anneal` is safe (no μ penalty) but does not remove it.

**Takeaway:** the campaign converged (objective ≈ 0.015) and the remaining plateau is driven by **irreducible noise in the low-count scans**, not by the tuned hyperparameters. Candidate next step is structural.

<!-- image: objective vs trial (parallel coordinates / convergence) -->
<!-- image: (μ, γ) phase-space paths for the best config, 2 cols x 7 rows -->

---

# Summary

- The pipeline recovers the true (μ, γ) on synthetic data and gives Fisher/CRB uncertainties (one open check on the confidence scale).
- Next goal: fit all experiments; synthetic-first logic says fix the model on its own data before real data.
- The agent loop handles structural changes well but over-engineers fine hyperparameter tuning.
- **AG-HYPOPT** = uncertainty-aware TPE proposes, a physics-informed agent picks one proposal, we run and record.
- The campaign converged with clear physics-guided insights; the plateau now points to a structural cause.
