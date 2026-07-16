# Optimization of mu
## Original idea discreat differenciation

- explain that this has problem as 2 runs must be made(which is expensive) and it explodes with more parameters
- gamma step needed to be averaged and never calculated in N itself

## New idea Reinforce
- Taken from reinfocement learning
- Brief explanation of reinforce

# First Reinforce Experiment: Many samples - One loss
- We dont have only one sample to calulate loss, we have many.
- Idea use the many samples (of each run) and calculate the step using the advantage from the loss
- Result: bad, no optimization.
![alt text](image.png)

- Makes complete sense, as all advantages are the same, as we have many samples no clear direction for improvement

# Second Reinforce Experiment: Many samples - Many losses

- New idea. Each run has its own loss (the distance of their own point for the wasserstein 1d). Each run gets its own loss/advantage and the optimization of mu is average through all the gradients calculated per run. (right?)

![alt text](image-3.png)

![alt text](image-4.png)

This optimizes but when it gets close to the real it is not as good, as wasserstein jump form one place to another (future idea, fix the order from the beginning and dont let the f)

# Joint Optimization: mu and gamma

- mu and gamma optimize together but degenerate: different (mu, gamma) pairs give same FWHM
- Visualization showed the problem
- Solution: add sigma (fit uncertainty) as second matching target

## Joint Optimization with Sigma

**The problem:** different (μ, γ) pairs can produce the same FWHM distribution — the solution is degenerate.

**Example intuition:**
- μ=20, γ=10: few photons on a narrow line → wider FWHM uncertainty
- μ=200, γ=5: many photons on a narrow line → tighter FWHM uncertainty

Both could match the same target FWHM, but the run-to-run uncertainty (σ of the fitted FWHM) tells them apart.

**What we did:** added a second loss term that matches the FWHM uncertainty σ to the target uncertainty:

    L = L_fwhm + λ · L_sigma

**Gradients:**
- μ: REINFORCE with combined reward — per-run advantage combines FWHM and σ matching
- γ: implicit diff for FWHM + CRLB approximation for σ (dσ/dγ ≈ 2/√n)

**Result:** the degeneracy is broken. Both μ and γ converge closer to their true values.

### How sigma optimization works (details)

**Where sigma comes from:** each Lorentzian fit produces the best-fit FWHM and its uncertainty σ, derived from the Hessian of the log-likelihood at the optimum. High curvature → confident fit (low σ). Flat curvature → uncertain fit (high σ).

**Loss function:** both FWHM and σ are matched via quantile-aligned absolute error:

    L = mean(|sort(FWHM_sim) − sort(FWHM_target)|)  +  λ · mean(|sort(σ_sim) − sort(σ_target)|)

The sigma term penalises simulations whose uncertainty distribution differs from the target.

**Gradient for μ (REINFORCE):** σ affects μ through the per-run advantage:

    reward_i = −|FWHM_i − target_i| − λ · |σ_i − target_σ_i|
    ∇μ = mean((reward_i − baseline) · score_i)

Runs with both good FWHM and good σ matching get positive reinforcement.

**Gradient for γ (implicit diff + CRLB):**

    ∇γ = sign(FWHM − target) · dFWHM/dγ  +  λ_γ · sign(σ − target) · dσ/dγ

Where dFWHM/dγ comes from implicit differentiation through the fit, and dσ/dγ ≈ 2/√n is the CRLB approximation (wider lines → higher σ, but more photons → lower σ).

**Why it works:** the two terms (FWHM + σ) anchor different aspects of the solution. μ dominates the σ through photon count (more photons → lower σ), while γ dominates the FWHM through linewidth. Together they break the degeneracy.

- μ update: REINFORCE with combined FWHM + sigma reward
- γ update: implicit diff + sigma gradient via CRLB

# Next: Optimization with real data
