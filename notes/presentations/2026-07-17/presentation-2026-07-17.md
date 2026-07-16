# Optimization of μ

## Original idea: discrete differentiation
- Requires two separate runs → expensive and doesn’t scale with more parameters
- γ gradient was obtained by averaging over shifted‑parameter runs, never inside the N‑sample simulation itself

## New idea: REINFORCE
- From reinforcement learning: a way to get gradients through stochastic computations
- The problem: we want to optimize μ, but the loss involves non‑differentiable sampling
- REINFORCE estimates the gradient of the expected reward:
  $$\nabla_\mu \approx \frac{1}{N}\sum_{i=1}^{N} \text{Loss}_i \cdot \frac{n_i - \mu}{\sigma^2}$$
- Baseline (average loss) reduces variance:
  $$\nabla_\mu \approx \frac{1}{N}\sum_{i=1}^{N} (\text{Loss}_i - \bar{L}) \cdot \frac{n_i - \mu}{\sigma^2}$$
- The score $(n_i - \mu) / \sigma^2$ tells how μ influenced each outcome
- This lets us backpropagate through the whole simulation, including the fitting step

### Baseline
- Without a baseline, even bad runs look "good" relative to nothing
- The baseline centers the signal: above = bad, below = good
- We use an exponential moving average across steps:
  $$\text{bl} = 0.9 \cdot \text{bl}_{\text{prev}} + 0.1 \cdot \bar{L}_{\text{step}}$$
- Each run's advantage = reward − baseline (same baseline for all runs in a step)
- Recent steps matter more than old ones (adapts as μ improves)

# First REINFORCE Experiment: Many samples – One loss
- Many samples per run, but only a single global loss
- Every sample in a run gets the same advantage → no differential signal to guide improvement
- Result: no optimization
![alt text](image.png)

# Second REINFORCE Experiment: Many samples – Many losses
- Each run gets its own loss (Wasserstein‑1D distance of its own point)
- Per‑run advantage → gradient averaged across runs
- Optimizes well until close to the target, where Wasserstein re‑ordering causes sudden jumps
  *(Future idea: fix the event order from the start to avoid those jumps)*
![alt text](image-3.png)
![alt text](image-4.png)

# Joint Optimization: μ and γ
- Optimizing both together is degenerate: different (μ, γ) pairs can produce the same FWHM distribution
- This makes it impossible to identify the true parameters from FWHM alone
![alt text](image-5.png)
![alt text](image-6.png)
![alt text](image-7.png)
![alt text](image-8.png)

# μ and γ influence on FWHM distributions
- Visual inspection to understand the degeneracy
  - Higher μ → more photons → narrower FWHM (thinner distribution)
  - Larger γ → wider line → thicker distribution (plus a shift)
  - The two effects can cancel: many (μ, γ) combinations give identical FWHM
![alt text](image-9.png)

# Joint Optimization with FWHM σ (uncertainty)
**Key insight:** the data contain uncertainties!

- Different (μ, γ) pairs can match the same FWHM but differ in their *FWHM uncertainty* (σ)
- Example:
  - μ=20, γ=10 → few photons, large σ (uncertain FWHM)
  - μ=200, γ=5 → many photons, small σ (precise FWHM)
- Both hit the same FWHM target, but σ tells them apart

**Solution:** add a second term that matches the whole σ distribution

$$L = L_{\mathrm{FWHM}} + \lambda \cdot L_\sigma$$

**Gradients:**
- μ: REINFORCE with combined reward (FWHM + σ matching)
- γ: implicit differentiation for FWHM + CRLB approximation for σ (dσ/dγ ≈ 2/√n)

**Result:** degeneracy broken — μ and γ converge closer to true values.

![alt text](image-11.png)

![alt text](image-12.png)

### Sigma optimisation – how it works
- σ comes from the Hessian of the log‑likelihood at the fit optimum
- Loss: quantile‑aligned absolute error on both FWHM and σ
- μ gradient: per‑run advantage = −|FWHMᵢ − target| − λ·|σᵢ − target_σ|
  ∇μ = mean((advantage − baseline) · score)
- γ gradient: sign(FWHM−target)·(dFWHM/dγ) + λ·sign(σ−target)·(2/√n)
- μ controls σ mainly through photon count; γ controls FWHM through linewidth
  → the two terms anchor different aspects, breaking the degeneracy

# Next: Optimisation with real data