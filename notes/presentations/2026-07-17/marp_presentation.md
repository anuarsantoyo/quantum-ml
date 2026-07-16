---
marp: true
theme: default
size: 16:9
paginate: true
style: |
  section { font-size: 28px; }
  section h1 { font-size: 44px; }
  section h2 { font-size: 36px; }
  section h3 { font-size: 30px; }
  img { max-width: 85%; height: auto; display: block; margin: auto; }
---

# Optimization of μ

<div style="text-align:center">

**From discrete differentiation to REINFORCE**

<br>

Progress update — 17 July 2026

</div>

---

## Original idea: discrete differentiation

- Requires two separate runs → expensive
- Doesn't scale with more parameters
- γ gradient had to be averaged, never calculated inside the N-sample simulation

---

## New idea: REINFORCE

- From reinforcement learning: get gradients through non-differentiable sampling
- REINFORCE estimates gradient of expected reward:
  $$\nabla_\mu \mathbb{E}[R] = \mathbb{E}[\;R \cdot \nabla_\mu \log p(\text{data} \mid \mu)\;]$$
- With baseline to reduce variance:
  $$\nabla_\mu \approx \frac{1}{N}\sum_i (R_i - \bar{R}) \cdot \nabla_\mu \log p(x_i \mid \mu)$$
- Positive reward → increase likelihood; negative → decrease
- Lets us backprop through the full simulation, including fitting

---

## Experiment 1: Many samples – One loss

- Many runs, but only a single global loss
- Every sample gets the same advantage → no differential signal
- **Result:** no optimization

![image.png](image.png)

---

## Experiment 2: Many samples – Many losses

- Each run gets its own loss (Wasserstein distance)
- Per-run advantage → gradient averaged across runs
- Works well, but Wasserstein re-ordering causes jumps near optimum

![image-3.png](image-3.png)

---

![image-4.png](image-4.png)

---

## Joint Optimization: μ and γ

- Optimizing both is **degenerate**: different (μ, γ) pairs produce the same FWHM
- The true parameters cannot be identified from FWHM alone

![image-5.png](image-5.png)

---

![image-6.png](image-6.png)

---

![image-7.png](image-7.png)

---

![image-8.png](image-8.png)

---

## FWHM Distributions: μ vs γ Effects

- **Higher μ** → more photons → narrower FWHM (thinner distribution)
- **Larger γ** → wider line → thicker distribution (plus shift)
- The two effects can cancel → many (μ, γ) pairs give identical FWHM

![image-9.png](image-9.png)

---

## Solution: Match the Uncertainty σ

<div style="font-size:26px">

**Key insight:** different (μ, γ) pairs produce different FWHM *uncertainties* (σ)

| Scenario | μ | γ | Photons | σ (uncertainty) |
|---|---|---|---|---|
| Few photons, narrow line | 20 | 10 | Few | **High** |
| Many photons, narrow line | 200 | 5 | Many | **Low** |

Both match the same FWHM — but σ tells them apart.

</div>

---

## Joint Optimization with σ

$$L = L_{\mathrm{FWHM}} + \lambda \cdot L_\sigma$$

<div style="font-size:26px">

**Gradients:**
- **μ:** REINFORCE with combined reward (FWHM + σ matching)
- **γ:** implicit diff for FWHM + CRLB approximation for σ ($d\sigma/d\gamma \approx 2/\sqrt{n}$)

**Result:** degeneracy broken — μ and γ converge closer to true values.

**How σ is used:**
- σ comes from the Hessian of the log-likelihood at the fit optimum
- Both FWHM and σ matched via quantile-aligned absolute error
- μ controls σ mainly through photon count; γ controls FWHM through linewidth
- The two terms anchor different aspects → breaks the degeneracy

</div>

---

## Summary

| Step | What | Status |
|------|------|--------|
| 1 | γ optimization via implicit diff | ✅ Works |
| 2 | μ optimization via REINFORCE | ✅ Works |
| 3 | Joint μ + γ optimization | ❌ Degenerate |
| 4 | Joint + σ matching | ✅ Degeneracy broken |
| 5 | Real data optimization | 🔶 In progress |

---

## Next: Optimisation with real data
