# qm-ml — Full Project Report

**Differentiable Monte Carlo for NV-center PLE spectroscopy**

*Written 2026-09-10 by Pukky at Anuar's request, as a complete, digestible tour of everything
the project has done so far: the physics, the idea, every notebook, the new math, the results,
the verdicts, and where we stand before the hyperparameter-tuning campaign.*

> **How to read this.** Part I–II give the problem and the data. Part III is the core idea.
> Part IV is the *math*, explained from scratch (this is the part to re-read before tuning).
> Part V walks notebook-by-notebook through the whole story (01 → 19). Part VI is the current
> state and the open failure modes. Part VII is the hyperparameter-tuning starting point —
> what is fixed, what is tunable, and what the objective is. Part VIII is a glossary.
>
> If you only have ten minutes: read Part I §1.4 (the failure that motivates everything),
> Part III, Part VI (the four failure modes that matter now), and Part VII.

---

## Table of contents

- **Part I — The problem**
  - 1.1 The system: NV centers and PLE spectroscopy
  - 1.2 The paper
  - 1.3 The degeneracy that shapes everything
  - 1.4 What breaks, and why simulation-based reconstruction fixes it
  - 1.5 Our goal
- **Part II — The data**
- **Part III — The core idea: differentiable Monte Carlo**
- **Part IV — The math (explained in detail)**
  - 4.1 Cauchy / Lorentzian and the reparameterization trick
  - 4.2 Truncation to the detection window
  - 4.3 The inner fit (Lorentzian MLE)
  - 4.4 Implicit differentiation through the fit
  - 4.5 The per-scan uncertainty σ_fit (CRLB) and dσ/dγ
  - 4.6 REINFORCE for the discrete photon count μ
  - 4.7 The loss: from χ² to a 2D KDE likelihood
  - 4.8 The 2D KDE likelihood and responsibilities
  - 4.9 Fisher information and the Cramér–Rao bound
  - 4.10 Score designs: σ_ref, z-form γ-score, λ_mean, H_S_MIN, annealing
  - 4.11 The μ-attractor mechanism (why μ can converge to the wrong value)
- **Part V — Notebook-by-notebook walk-through**
- **Part VI — Current state and the failure modes that matter**
- **Part VII — Starting point for hyperparameter tuning**
- **Part VIII — Glossary**

---

# Part I — The problem

## 1.1 The system: NV centers and PLE spectroscopy

A **nitrogen-vacancy (NV) center** in diamond is an atom-like defect with a sharp optical
transition. It is a workhorse platform for quantum sensing and photonics. To characterize it
you measure its **optical linewidth** by **photoluminescence excitation (PLE) spectroscopy**:

- a resonant red laser is scanned across the transition frequency;
- at each frequency step the collected fluorescence is recorded;
- one full scan yields one absorption line, from which one **linewidth (FWHM)** is extracted by
  fitting a line shape.

The physical absorption probability of the transition is a **Lorentzian** (= Cauchy) in
frequency — the Fourier transform of the exponential decay (Weisskopf–Wigner). Its scale
parameter **γ is the half-width at half-maximum (HWHM)**, and the full width at half maximum is
**FWHM = 2γ**.

The parameters that shape a measured scan are:

| Symbol | Meaning | Role |
|---|---|---|
| **γ** | Lorentzian HWHM (MHz) | the line width (what we want) |
| **μ** (a.k.a. n̄) | mean photon number per scan | how bright the signal is |
| **σ_prop** | photon-count noise (per experiment, fixed) | how much n fluctuates |
| **λ** | mean of the additive Poisson background | noise photons |

## 1.2 The paper

The foundation is:

> Orphal-Kobin, Pieplow, Gokhale, Unterguggenberger, Schröder (Humboldt-Universität zu Berlin),
> **"Retrieving Lost Atomic Information: Monte Carlo-based Parameter Reconstruction of an
> Optical Quantum System"**, arXiv:2501.07951 (Jan 2025). → `docs/papers/2501.07951.pdf`

**Their problem.** At **low signal**, the standard estimator — the **median of the fitted
linewidths** — becomes *precise but inaccurate*: as you collect more scans its confidence
interval shrinks (it looks trustworthy) while its value is systematically wrong, sometimes
collapsing below the lifetime-limited linewidth. The confidence interval can become narrower
than the bias itself. That is the textbook failure of naive statistics in undersampled regimes.

**Their fix (Monte Carlo reconstruction).** Instead of trusting one number, *simulate the whole
experiment* and match distributions:

1. **Forward model.** One simulated scan: photon detection events are drawn from a **Cauchy
   (Lorentzian)** line $\gamma/\pi(\omega^2+\gamma^2)$ plus **Poisson(mean 2)** noise events;
   the spectrum is fitted with a **Voigt** profile (binned least-squares, 2.2 MHz/step) and its
   FWHM recorded.
2. **Histogram.** Repeat for $k$ scans → a simulated FWHM distribution.
3. **Match.** Compare the simulated histogram to the measured one with a $\chi^2$ statistic,
   minimized by **brute-force grid search** over $(\gamma,\bar n)$:
   $$S(\gamma,\bar n)=\sum_i\frac{(O_i-E_i(\gamma,\bar n))^2}{E_i(\gamma,\bar n)},$$
   with a 99 % confidence region at $S\le S_{\min}+9.21$.

**Their findings.** The median is precise but biased; bias and required photon number both
depend on $(\gamma,\bar n)$. The MCM bias stays below ~2 % for $\bar n\gtrsim30$; the median only
becomes reliable above $\bar n\approx48$ (1 nW) / 56 (3 nW). They also trained a **Bayesian CNN**
that predicts $(\gamma,\bar n,\sigma)$ from raw scans — good on γ, worse on $\bar n$ — and
explicitly **open the door to the ML route** this project takes.

## 1.3 The degeneracy that shapes everything

**The FWHM distribution alone cannot separate μ from γ.** Many (μ, γ) pairs produce nearly the
same FWHM distribution: more photons → better fits → narrower FWHM, which trades against a
larger γ. So matching only the FWHM distribution is **ill-posed**.

The resolution — central to everything below — is to match **two** outputs at once: the FWHM
distribution **and** its per-fit uncertainty σ. The σ channel encodes the photon-count
statistics directly, and it breaks the μ–γ degeneracy. (This is why the model is a *2D*
likelihood over (FWHM, σ), not a 1D one.)

## 1.4 What breaks, and why simulation-based reconstruction fixes it

At low photon numbers per scan the line fit degenerates (near-zero FWHM, exploding fit error,
failed fits). Filtering bad fits removes the informative tail; keeping them adds noise. The
measured FWHM distribution becomes a *biased, heavy-tailed* sample of the true line — exactly
the regime where inverse-simulation beats a naive point estimator, because it propagates the
whole measurement pipeline and reports *honest* intervals.

## 1.5 Our goal

**Recover (μ, γ) from the measured FWHM distribution — the paper's task — but with
gradient-based optimization instead of grid search, *plus honest uncertainties*.**

Concretely, in Anuar's own words (2026-08-30):

> The model must produce **honest uncertainties**: well-identified parameters → tight and
> accurate; poorly-identified parameters → wide but with the truth inside. A point estimate
> that misses is *acceptable* as long as it lies within the reported uncertainty.
> **Tight and wrong is the failure.**

This "honest-wide beats tight-and-wrong" philosophy became the north star of the later series
(§VI).

---

# Part II — The data

Real experimental PLE data from **Dr. Gregor Pieplow** (AG Schröder, HU Berlin), in
`data/raw_data/`. Full write-up: `data/raw_data/data_explanation.md`.

**Organization.** Two sessions by resonant red-laser power — `fwhm_1nW_240221/` and
`fwhm_3nW_210221/` — each with 7 files sweeping the **transmission setting**
`Trans05 … Trans100` (5 % → 100 %, an ND filter in the detection path). **14 experiments total**
(2 powers × 7 transmissions).

**File format.** A 2-column whitespace table, **3200 rows** each (one anomaly: `3nW…Trans40` has
4800). Each row = one fit attempt:

| Column | Meaning |
|---|---|
| 1 | line width (FWHM) |
| 2 | fit error — 1σ uncertainty on that FWHM |
| `nan nan` | fit failed |

**Facts that matter.**

- **Valid rows grow with transmission** (~160 at Trans05 → ~2500 at Trans100): more signal →
  more successful fits. Transmission is the **SNR knob**.
- **Median FWHM grows with power** → power broadening.
- Huge column-2 values pair with near-zero FWHM = degenerate fits; `err/fwhm` is the fit-quality
  metric. Filtering convention: finite values, FWHM > 0, `err/fwhm < 10`.
- The absolute physical unit of the raw linewidth was never recorded (open item with Gregor);
  the pipeline works in **MHz** by convention.

**True parameters** (Gregor's calibration fits). Per experiment we have
`pho_normal_mean` = **μ_true**, `pho_normal_std` = **σ_prop**, `pho_noise_poisson_mean` = **λ**:

| Power | Trans | μ_true | σ_prop | λ |
|---|---|---|---|---|
| 1 nW | 05 | 9.393 | 2.576 | 2.232 |
| 1 nW | 10 | 12.372 | 3.445 | 2.122 |
| 1 nW | 20 | 17.316 | 4.141 | 2.286 |
| 1 nW | 40 | 38.405 | 7.198 | 2.351 |
| 1 nW | 60 | 61.374 | 9.851 | 2.593 |
| 1 nW | 80 | 79.365 | 12.627 | 2.758 |
| 1 nW | 100 | 70.817 | 17.221 | 2.636 |
| 3 nW | 05 | 13.204 | 3.724 | 2.186 |
| 3 nW | 10 | 24.476 | 5.639 | 2.158 |
| 3 nW | 20 | 34.279 | 8.319 | 2.264 |
| 3 nW | 40 | 84.892 | 24.013 | 2.475 |
| 3 nW | 60 | 103.203 | 23.95 | 2.741 |
| 3 nW | 80 | 137.537 | 32.107 | 2.911 |
| 3 nW | 100 | 175.707 | 40.975 | 3.087 |

**The γ reference (there is no independent ground truth for γ).** γ is referenced against the
**median FWHM at Trans100** (full collection, least-broadened):

$$\gamma_{\text{true}} = \text{median FWHM @ Trans100} / 2 \qquad(\text{FWHM}=2\gamma).$$

| Session | median FWHM @ Trans100 | γ_true |
|---|---|---|
| 1 nW | 17.0 MHz | **8.5 MHz** |
| 3 nW | 28.3 MHz | **14.1 MHz** |

> **Subtlety that recurs everywhere:** the paper's 17/29 MHz are **Voigt** FWHMs; our γ is the
> **Cauchy HWHM** of the simulated Lorentzian. They are not directly comparable beyond this
> reference convention. This becomes a real failure mode (FM#6, §VI).

---

# Part III — The core idea: differentiable Monte Carlo

The paper's grid search is expensive and scales badly with the number of parameters. We make the
**entire Monte Carlo pipeline differentiable** so we can use gradient descent.

The pipeline is a loop of **generate → fit → compare → update**:

```
        parameters (μ, γ)
               │
               ▼
   ┌───────────────────────┐
   │ GENERATE  n photons   │  n ~ round(N(μ, σ_prop));  detunings ~ Cauchy(γ)
   │  + background         │  + Poisson(λ) uniform detunings
   └───────────────────────┘
               │  (a set of photon detunings)
               ▼
   ┌───────────────────────┐
   │ FIT  a Lorentzian MLE │  per scan → (FWHM, σ_fit)
   │  (L-BFGS)             │  + gradients via implicit differentiation
   └───────────────────────┘
               │  (M simulated (FWHM, σ) points)
               ▼
   ┌───────────────────────┐
   │ COMPARE to real data  │  2D KDE likelihood over (FWHM, σ)
   │  → loss L(μ, γ)       │
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │ UPDATE (μ, γ)         │  μ: REINFORCE (discrete count)
   │                       │  γ: KDE log-density gradient
   └───────────────────────┘
```

Two obstacles had to be overcome, and each one is a **chapter of the project**:

1. **The photon count is discrete.** $n = \text{round}(\mu + \sigma_{\text{prop}}\varepsilon)$
   hits a rounding step — no ordinary derivative. → **REINFORCE** (policy gradient) for μ.
   (§4.6)
2. **The fit is an iterative optimizer.** Backpropagating through 80 L-BFGS steps is wasteful
   and unstable. → **Implicit differentiation** through the fit via the implicit function
   theorem. (§4.4)

And a third, conceptual, obstacle about **what to compare**: a histogram is
non-differentiable → the loss evolved from χ² to a smooth **2D KDE likelihood** (§4.7–4.8).

---

# Part IV — The math (explained in detail)

> This is the part to understand before touching hyperparameters. Every symbol used in the
> tuning space (Part VII) is defined here.

## 4.1 Cauchy / Lorentzian and the reparameterization trick

The absorption line is a Cauchy (= Lorentzian) with location 0 and scale γ:

$$p(x)=\frac{1}{\pi}\frac{\gamma}{x^2+\gamma^2},\qquad x\in(-\infty,\infty).$$

γ is the **HWHM**: $p(\pm\gamma)=p(0)/2$, and the quartiles sit at $\pm\gamma$. The tails decay
only as $1/x^2$ (**heavy tails** — this is why we must truncate to the detection window, §4.2).

**CDF:**

$$F(x)=\frac12+\frac{1}{\pi}\arctan\!\Big(\frac{x}{\gamma}\Big).$$

**Inverse CDF (quantile function).** Solve $u=F(x)$:

$$u-\tfrac12=\tfrac1\pi\arctan(x/\gamma)\;\Rightarrow\; x=\gamma\tan\!\big(\pi(u-\tfrac12)\big)=F^{-1}(u).$$

**Why this makes sampling differentiable (the reparameterization trick).** By the probability
integral transform, pushing a uniform $u\sim\mathcal U(0,1)$ through $F^{-1}$ gives a
Cauchy(0, γ) sample. The key point for differentiability:

> **All randomness lives in $u$, which is drawn once and frozen. γ enters only as a smooth,
> deterministic multiplier.** There is no randomness between γ and the output, so the derivative
> is exact:
> $$\frac{\partial x}{\partial\gamma}=\tan\!\big(\pi(u-\tfrac12)\big).$$

This is *the* trick that lets γ be recovered by gradient descent. Contrast the old
`np.random.standard_cauchy() * γ`, where the draw and γ are entangled and no clean derivative
exists.

Sanity checks: $u=0.5\to x=0$ (center); $u=0.25,0.75\to x=\mp\gamma$ (quartiles at ±γ);
$u\to0,1\to x\to\mp\infty$ (heavy tails).

## 4.2 Truncation to the detection window

The detector only sees frequencies in a **±75 MHz window** (150 MHz wide, matching the paper's
supplemental). Two ways to handle out-of-window photons:

- **Clipping** (the original, buggy approach): pile everything outside into spikes at ±75 MHz.
  ~17 % of the Cauchy mass becomes edge spikes → the fitted FWHM becomes **non-monotonic in γ**
  (rises then collapses) → γ is *not recoverable*. **This was a real bug, found and fixed.**
- **Truncation** (correct): out-of-window photons simply aren't detected. Sample from the
  truncated Cauchy via its inverse CDF:
  $$x=\gamma\tan\!\Big(\arctan\!\big(\tfrac{L}{\gamma}\big)(2u-1)\Big),\qquad L=75\ \text{MHz}.$$

Truncation is both physically right (undetected photons are lost) and mathematically right
(the FWHM stays monotonic in γ). The number of signal photons stays fixed = `len(u)`; no
rejection loop.

## 4.3 The inner fit (Lorentzian MLE)

Each scan's photon set is fitted with a **Lorentzian by maximum likelihood** (L-BFGS, ≤ 80
iterations, strong-Wolfe line search). Parameters θ = (center, raw_γ), with the width mapped to
keep it positive:

$$\gamma = \varepsilon_w + 150\,\text{sigmoid}(\text{raw}_\gamma),\qquad \varepsilon_w=10^{-2}
\quad(\text{so }\gamma\in(0.01,\,150.01)\ \text{MHz}).$$

Initialization: center = photon median, width = 15 MHz. **No uniform-background term**
(`uniform_bg=False`) — the background lives in the *generator*; the fitter just fits the line.

> **Model-gap note (Finding #3, 28 Jun).** The paper's estimator is a **binned Voigt
> least-squares**; ours is an **unbinned Lorentzian MLE**. This is a deliberate, documented
> difference. The FWHM is an *estimator output*, so if we ever need to reproduce the paper's
> FWHM-bias *identically* this matters — most likely in the low-count regime, which is exactly
> the regime the whole MCM is about. Flagged to revisit if sim-vs-real matching struggles.

Output per scan: **FWHM = 2γ** (exact for a Lorentzian), plus, via implicit differentiation,
σ_fit, ∂FWHM/∂γ, ∂σ/∂γ.

**Fallback:** if fewer than 3 photons or the fit fails, return FWHM = 2γ, σ = 0.5γ,
∂FWHM/∂γ = 2, ∂σ/∂γ = 0.5 (analytic values of the fallback).

## 4.4 Implicit differentiation through the fit

We need $\mathrm d\text{FWHM}/\mathrm d\gamma$ even though the fit is an *iterative optimizer*.
Rather than unrolling L-BFGS, differentiate **around** the optimum with the **implicit function
theorem (IFT)**.

At the optimum $\theta^*(\gamma)$, the stationarity condition is
$\nabla_\theta \text{NLL}(\theta^*(\gamma),\gamma)=0$. Differentiating w.r.t. γ:

$$H_{\theta\theta}\,\frac{\mathrm d\theta^*}{\mathrm d\gamma}+H_{\theta\gamma}=0
\quad\Longrightarrow\quad
\boxed{\frac{\mathrm d\theta^*}{\mathrm d\gamma}=-H_{\theta\theta}^{-1}H_{\theta\gamma}}$$

where $H_{\theta\theta}=\partial^2\text{NLL}/\partial\theta^2$ at θ* (Tikhonov-regularized,
$+\text{reg}\cdot I$) and $H_{\theta\gamma}=\partial^2\text{NLL}/\partial\theta\,\partial\gamma$
computed by central finite difference in γ (step 1e-3). Then

$$\frac{\mathrm d\text{FWHM}}{\mathrm d\gamma}=\frac{\partial F}{\partial\theta}\cdot
\frac{\mathrm d\theta^*}{\mathrm d\gamma}.$$

Implementation details that matter: float64 arithmetic; a scale-aware Tikhonov regularization and
condition/PD guards that **zero out the rare bad run** instead of emitting garbage gradients.

**Validation gates (28 Jun).** Per-run gradient vs central finite difference: median relative
error **0.12 %** (90th pct ~1 %); errors concentrated where the true gradient ≈ 0 (degenerate
fits). Gradient sign points toward γ_true from both sides, ≈ 0 at the truth. End-to-end: from
γ=10, recovered **γ = 20.5 ± 0.15** (true 20.0).

## 4.5 The per-scan uncertainty σ_fit (CRLB) and dσ/dγ

Each fit also yields an uncertainty on its FWHM, from the curvature of the NLL — a
Cramér–Rao-style bound at the fitted point:

$$\sigma_{\text{fit}}=\sqrt{\frac{1}{N}\,\frac{\partial F}{\partial\theta}^\top
H_{\theta\theta}^{-1}\frac{\partial F}{\partial\theta}}.$$

This σ_fit is the **second observable** we match (§1.3). Its **derivative w.r.t. γ** is computed
exactly (a two-channel chain: the movement of θ* plus the direct dependence of H on γ through the
photon positions):

$$\frac{\mathrm d\sigma}{\mathrm d\gamma}=\underbrace{\Big(\frac{\partial\sigma}{\partial\theta}\Big)_{\theta^*}\!\!\cdot\frac{\mathrm d\theta^*}{\mathrm d\gamma}}_{\text{term A: }\theta^*\text{ moves}}\;\;+\;\;\underbrace{\frac{\partial\sigma}{\partial H}\!\cdot\!\frac{\mathrm dH}{\mathrm d\gamma}}_{\text{term B: direct data dependence}}.$$

> **History note (commit `73c85a0`):** term B replaced an earlier **crude approximation**
> $d\sigma/d\gamma\approx 2/\sqrt n$. That fake was the cause of several γ-side pathologies and
> was removed in favor of the exact two-term derivative. If you see old notebooks referencing the
> CRLB approximation, that is the version the current code supersedes.

## 4.6 REINFORCE for the discrete photon count μ

The number of signal photons is drawn as a **rounded Gaussian**:
$$n=\max(\text{round}(\mu+\sigma_{\text{prop}}\varepsilon),\,0),\qquad \varepsilon\sim\mathcal N(0,1).$$
The `round` is non-differentiable, so no analytic $\partial L/\partial\mu$ exists. Use the
**score-function (policy-gradient / REINFORCE)** estimator: for a loss $L$,

$$\nabla_\mu \mathbb E[L]=\mathbb E\big[L\cdot\nabla_\mu\log P(n\mid\mu,\sigma)\big],
\qquad \nabla_\mu\log P(n\mid\mu,\sigma)\approx\frac{n-\mu}{\sigma^2}.$$

**An important architectural lesson (07 Jul, notebook 07 → 08).** The naive version — sample a
*n_i per run*, average the score over 200 runs — **fails**: the scores average to ~0 because
$\text{avg}(n)\approx\mu$ by construction, so the signal is pure noise. The fix is the
**meta-distribution architecture**:

> Draw **one** $\bar n\sim\mathcal N(\mu,\sigma_{\text{ref}})$ per iteration and use it for **all**
> runs. Now the advantage $(L-b)$ is conditioned on a *single* $\bar n$ value, so the gradient
> **actually points**: $\bar n$ low → high loss → push μ away from it; $\bar n$ high → low loss →
> push μ toward it.

This is structurally the same idea as **hyperparameter optimization (hyperopt)** — propose from a
search distribution, evaluate, update the belief. (REINFORCE only remembers the current loss and a
decaying baseline; hyperopt builds a surrogate over all evaluations.)

In the current optimizer, the REINFORCE update uses the **KDE responsibilities** as the
"data mass" per run (§4.8):

$$\mu\leftarrow\mu-\text{LR}_\mu(t)\,\big(\bar B-\bar{\bar B}\big)\cdot s_\mu,
\qquad s_\mu=\frac{n-\mu}{\sigma_{\text{ref}}^2},\qquad
\bar B=\textstyle\sum_i w_{ij},$$

where $w_{ij}$ is the (row-normalized) kernel weight of simulated run $j$ at data point $i$.
Gradient clipped to ±10; μ bounded to [1, 200].

> **The "attractor."** The REINFORCE fixed point is the **kernel-weighted mean photon count** of
> the scans the data actually matches: $E_B[n]=\bar n$. If the likelihood is unbiased this equals
> μ_true; a *biased* likelihood shows up as a **clean, confident convergence to the wrong μ**.
> This is the single most important diagnostic in the whole project (§4.11, §VI).

## 4.7 The loss: from χ² to a 2D KDE likelihood

The loss went through several incarnations — understanding *why* it changed is understanding the
project:

| Loss | Notebook | What it compares | Why it was dropped |
|---|---|---|---|
| histogram χ² | (paper) | binned histograms | non-differentiable |
| **MMD²** (Gaussian kernel) | 05, 06 | two *sample sets* directly | needs a kernel bandwidth; insensitive in tails |
| **Wasserstein-1** | 05, 07, 08 | sorted samples (quantile matching) | no bandwidth, robust — but no likelihood |
| **per-quantile W1 + σ term + mean term** | 12a–12c | FWHM **and** σ | adding σ broke the degeneracy; mean term later shown harmful |
| **2D KDE negative log-likelihood** | 12d, 15–19 | (FWHM, σ) density | *current*: a proper likelihood → Fisher/CRB valid |

The key realization (27 Jun): **the data are individual samples, not a pre-binned histogram** —
each file is a list of FWHM values. So both sides of the loss are just **sets of samples**, and we
can compare them directly (no binning). The FWHM spans orders of magnitude, so losses are applied
in **log space** to make a single kernel scale meaningful.

The transition that matters most for the current work is 12d: replacing the W1/mean losses with a
**2D KDE likelihood over (FWHM, σ)**, turning the optimization into an **MLE**. This is what makes
the **Fisher information / Cramér–Rao bound** a valid uncertainty statement (§4.9).

## 4.8 The 2D KDE likelihood and responsibilities

Each simulated scan produces a 2D point (FWHM, σ_fit). The simulated cloud is turned into a
smooth density with a **2D Gaussian kernel**, bandwidths from **Scott's rule**
$H=\sigma_{\text{data}}\cdot N^{-1/6}$ (with an absolute floor $H_S^{\min}=0.05$ MHz on the
σ-kernel).

Let $d^F_{ij}$ and $d^S_{ij}$ be the FWHM- and σ-distances between data point $i$ and simulated
point $j$. The kernel matrix and the **row-normalized responsibilities** are

$$W_{ij}=\exp\!\Big(-\tfrac12(d^F_{ij}/H_F)^2-\tfrac12(d^S_{ij}/H_S)^2\Big),
\qquad w_{ij}=\frac{W_{ij}}{\sum_j W_{ij}},$$

and the **negative log-likelihood of the measured data** under the simulated density is the
objective:

$$\text{NLL}=-\frac{1}{N_{\text{tgt}}}\sum_i\log\!\Big(\frac{1}{N_{\text{sim}}}
\sum_j W_{ij}\Big).$$

The responsibilities $w_{ij}$ ("how much simulated scan $j$ explains measured point $i$") are the
**engine of both parameter updates and the Fisher matrix**.

**Per-point scores** (the two channels):

- **μ (weighted REINFORCE):** $s_{\mu,i}=\sum_j w_{ij}\cdot\dfrac{n_j-\mu}{\sigma_{\text{ref}}^2}$
- **γ (analytic KDE chain):** $s_{\gamma,i}=\sum_j w_{ij}\Big[\dfrac{(f_i-f_j)\,(\mathrm df_j/\mathrm d\gamma)}{h_F^{2}}
  +\dfrac{(s_i-s_j)\,(\mathrm ds_j/\mathrm d\gamma)}{h_S^{2}}\Big]$

with $\mathrm df/\mathrm d\gamma$ from implicit differentiation (§4.4) and
$\mathrm ds/\mathrm d\gamma$ from §4.5.

## 4.9 Fisher information and the Cramér–Rao bound

At the recovered $(\hat\mu,\hat\gamma)$, the **empirical Fisher information matrix** is the average
outer product of the per-point scores:

$$J=\frac{1}{N_{\text{tgt}}}\sum_i s(x_i)\,s(x_i)^\top,\qquad s=[s_\mu,s_\gamma],$$

computed from **M_FINAL** fresh simulated scans at the fitted point (M = 500, optionally averaged
over FISHER_SEEDS independent seeds). The **Cramér–Rao bound** gives parameter uncertainties:

$$\sigma_\mu=\sqrt{(J^{-1})_{00}},\qquad \sigma_\gamma=\sqrt{(J^{-1})_{11}},\qquad
\text{corr}=\frac{(J^{-1})_{01}}{\sqrt{(J^{-1})_{00}(J^{-1})_{11}}}.$$

These are **lower bounds** on the statistical uncertainty. This is the route **Gregor endorsed**
(his words, from Anuar's journal: he "was very happy to see the results and the covariance
matrix").

**Degeneracy diagnostic.** The 1D (FWHM-only) Fisher should have a **near-zero eigenvalue** — the
flat direction that the σ channel resolves. In 12d: 1D eigenvalue ratio 248 vs 2D ratio 172 — the
σ dimension *does* shrink the flat direction, but a residual degeneracy remains.

**Known pathologies** (both are signatures of a *broken fit*, not trustworthy bounds):
- **near-singular J** → absurdly wide σ (e.g. σ_μ ≈ 1300 seen once);
- **over-confident J** when the optimizer sits in a wrong basin (σ_γ ≈ 0.09 at half-truth γ).

> This is exactly why "honest-wide vs tight-and-wrong" became the acceptance criterion (§VI):
> the Fisher width *tells you* whether the point estimate is trustworthy.

## 4.10 Score designs: σ_ref, z-form γ-score, λ_mean, H_S_MIN, annealing

Four small design choices that the 17–19 series introduced, each fixing a concrete bug:

**(a) σ_ref normalization (17f) — fixes optimizer starvation.** The μ-score divides by
$\sigma_{\text{prop}}^2$. At high transmission σ_prop is large (up to 41), so each step moves μ by
only ~0.1–0.5 photons and the optimizer needs 500+ iterations to climb the remaining distance.
**Fix:** divide by a **fixed reference** $\sigma_{\text{ref}}$ (default 10) instead of the
per-experiment σ_prop, making the step size identical in photon units across experiments.

**(b) z-form γ-score (17g/18c) — fixes γ jumps.** The raw KDE γ-score
$s_\gamma=\dfrac{d^F\partial F/\partial\gamma}{H_F^2}+\dfrac{d^S\partial\sigma/\partial\gamma}{H_S^2}$
magnifies with $1/H^2$ and with mismatch distance → it **saturates the clip** and takes ~5 MHz
steps. **Fix:** one power of bandwidth (unitless distances), scaled by a fixed reference
$H_{\text{REF}}$ (mirrors σ_ref for μ):
$$s_\gamma=\Big[\frac{d^F\,\partial F/\partial\gamma}{H_F}+\frac{d^S\,\partial\sigma/\partial\gamma}{H_S}\Big]\Big/H_{\text{REF}}.$$
This cured the γ disease on synthetic data and is now **frozen** as part of the structural design.

**(c) λ_mean (mean-matching anchor) — TRIED, REJECTED.** A kernel-free term
$|{\langle\text{FWHM}\rangle}_{\text{sim}}-\langle\text{FWHM}\rangle_{\text{tgt}}|$ (12c) with
weight λ_mean. The 17-series λ-sweep showed it is **monotonically harmful**:
γ RMSE **1.20 → 3.32 → 4.48** MHz for λ = 0 → 0.1 → 0.3 (18, §V.17). It is **disabled
(λ_mean = 0)**.

**(d) H_S_MIN (σ-kernel floor).** An absolute floor on the σ-kernel bandwidth, preventing the
σ-kernel from collapsing into a knife-edge at high transmission. Default 0.05 MHz.

**(e) γ annealing.** $\text{LR}_\gamma(t)=0.5\,(1-0.5\,t/N)$ — intended to settle noisy low-count γ.
Known side effect: it can **freeze** γ in the low-count regime ($n_{\text{tgt}}\lesssim500$) before
convergence.

## 4.11 The μ-attractor mechanism (why μ can converge to the wrong value)

The 19-series pinned down **exactly** how a biased μ estimate arises on real data. This is the
deepest piece of "new math" in the project, so it deserves its own subsection.

**The structural fact.** The real data files carry **(FWHM, fit-error) per scan — no photon
counts.** So the μ channel can only receive information through the **σ_fit channel**: σ_fit is a
function of $n(\mu)$ (fewer photons → larger fit errors), so the KDE weight in the σ direction is
the *only* μ-sensitive link to the data.

**Why the count channel dies.** The REINFORCE count score is weighted by the KDE responsibility
$B=w_{\text{mean}}$. With 1100–2500 real data points, the responsibility becomes **≈ uniform**, so
the count score $(n-\mu)/\sigma_{\text{ref}}^2$ averages to ≈ 0. **The count channel carries no μ
information on real data.**

**Why the σ channel points the wrong way.** The simulator's σ_fit is *structurally too narrow* at
μ_true: real fit-errors exceed the simulator's by **1.04–4.25×**, and the gap **grows with
transmission T**. The Lorentzian model can only match the real error scale by **photon starvation**
(lower μ → worse fits → bigger errors). So the likelihood develops a spurious **low-μ mode** at
μ/μ_true ≈ 0.30–0.50. Because the slope of σ w.r.t. μ is
$\mathrm d\sigma/\mathrm d\mu=-\sigma/(2n)$,

- **un-damped** σ-channel gradient → the optimizer *descends into* the low-μ mode (destructive:
  |Δμ|/σ up to **557**);
- **damped** (σ_ref = σ_prop) → honest but weak (σ_μ ≈ 0.3–0.5 μ_true; truth inside).

**This is "failure mode #8": a score-design failure, not a data failure.** The practical rule:
report μ **wide-but-honest** until the σ(μ) *shape* can be matched (not just its scale) or the
σ-gradient is properly damped. Details in §V.19.

---

# Part V — Notebook-by-notebook walk-through

The notebooks are numbered chapters; several share a number when they are variants of one idea.
The numbering reflects the real development order, and each series below ends with its **durable
verdict**.

## 01–03 — Making the pipeline differentiable (May–Jun)

**`01-mc-algorithm.ipynb` — the paper's Monte Carlo, step by step.** Terminology (run,
simulation, MC distribution) and the forward pipeline: noiseless PLE spectrum → photon noise →
Lorentzian fit → one extracted linewidth per run. Establishes the goal: make this a
backpropagatable PyTorch pipeline.

**`02-sampling-toy.ipynb` — the first obstacle.** Sampling $n\sim\mathcal N(\mu,\sigma)$ and
rounding to an integer is non-differentiable (stochastic node + step function). Introduces the
reparameterization trick on a toy and STE + finite-difference surrogate ideas. *(Journal 12 Jun:
the STE idea; 25 May: the whole "differentiable MC" concept, MC-PyT Solution 1.)*

**`03-fitting-toy.ipynb`, `03-fitting-toy-fwmh.ipynb` — the second obstacle.** The per-run fit
(L-BFGS) blocks gradients. Differentiate *around* the fit with the **implicit function theorem**
(§4.4) — no unrolling — pushing the outer loss through the fitted pseudo-Voigt, for both the raw γ
and the fitted FWHM. Demonstrated: a full loop recovers a target linewidth.

## 04 — EDA

**`04-eda.ipynb`.** Exploratory analysis of Gregor's data: failed-fit (NaN) rates vs
transmission/power, FWHM distributions, power-broadening trend, and `err/fwhm` as the fit-quality
metric. *(This is where "the data are samples, not histograms" became clear → loss switched to
MMD²; 27 Jun.)*

## 05 — Loss functions

**`05-loss-mmd.ipynb`, `05-loss-w1.ipynb`.** Distribution-matching losses:
**MMD²** (Gaussian kernel, median-heuristic bandwidth) and the parameter-free **1D
Wasserstein-1** on sorted samples. Both still exist in `src/losses.py`.

## 06 — γ differentiable end-to-end

**`06-gamma.ipynb`, `06-gamma-simple.ipynb`.** First full differentiability for γ:
reparameterized (truncated) Cauchy sampling + implicit differentiation through the fit, so
`loss.backward()` populates `gamma.grad`. **Validated: γ = 20 → recovered ~20.5.**
*(28 Jun — this is the milestone where the "generate → fit" half became gradient-trainable.)*

## 07–08 — REINFORCE for μ

**`07-reinforce-toy.ipynb`.** Optimize μ through the discrete rounding step with REINFORCE and an
EMA baseline, on a Wasserstein-1 loss.

**`08-reinforce-per-run.ipynb`.** Refines to **per-quantile** losses: low-n runs (narrow FWHM) are
paired with low target quantiles, giving a structured, directional gradient per quantile. The
ρ(n, loss) test shows the signal is strong and correctly directed **only when μ is below the
truth**, decaying near it. *(This is the first sighting of the "attractor" behavior.)*

## 09 — (μ, σ) → FWHM map

**`09-mu-sigma-fwhm-map.ipynb`.** Grid sweep over the photon-count proposal distribution (μ, σ) to
build intuition for how it shapes the FWHM distribution, and where the REINFORCE gradient has
signal.

## 10 — Joint optimization

**`10-joint-optimization.ipynb`.** Combines the two estimators into one loop: REINFORCE for μ,
implicit diff for γ, and a CRLB-based dσ/dγ.

## 11 — FWHM pairplot over (μ, γ)

**`11-fwhm-pairplot.ipynb`.** Grid showing how the FWHM distribution changes with **both** μ and
γ — the visual proof of the **μ/γ degeneracy** (§1.3).

## 12 — Joint optimization with σ matching (12a–12d)

This is the heart of the synthetic-data method.

**`12a-joint-opt-with-sigma-noise.ipynb` — add σ matching.** Matching the FWHM distribution alone
is degenerate; matching FWHM **and** its fit uncertainty σ breaks it. μ: REINFORCE with a combined
FWHM+σ per-quantile reward. γ: implicit diff for FWHM + CRLB approximation
$d\sigma/d\gamma\approx2/\sqrt n$ for σ.

**`12b-…-lr-decay.ipynb` — decay the μ learning rate.** Converges fast then settles.

**`12c-…-mean-fwhm.ipynb` — add the mean-matching anchor (12c = best synthetic result).**
Adds a mean-FWHM term with weight λ_mean to give μ a clean signal when per-quantile gets noisy.
**Saved result: μ 8 → 48.45 (true 50), γ 5 → 19.62 (true 20)**, combined loss 38.56 → 1.48, final
FWHM 41.3 ± 10.4 vs target 42.3 ± 10.5.
> ⚠️ The mean anchor (λ_mean) **looks** like the star here, but the 17-series later proved it is
> **harmful** and it was removed (§4.10c, §V.17). On synthetic data *without* it, the model does
> even better. 12c is historically important but **not** the final method.

**`12d-joint-opt-likelihood-fisher.ipynb` — the current likelihood.** Replaces 12c's W1/mean
losses with a **2D KDE negative log-likelihood** over (FWHM, σ), turning the optimization into an
**MLE** and making Fisher/CRB applicable (§4.8–4.9). Weighted-REINFORCE μ (no EMA baseline, no mean
term) + analytic KDE γ-chain. Result example: $(\hat\mu,\hat\gamma)=(46.55,17.39)$,
σ_μ = 26.0, σ_γ = 2.77, corr = +0.70; the 1D-vs-2D eigenvalue ratio exposes the degeneracy.
**This is the ancestor of the current optimizer** and the reason Fisher is our uncertainty method.

## 13 — Real data: first attempt and diagnosis (13a–13f) — ❌ instructive failure

First application of the joint optimization to **real PLE data (3 nW, 40 % transmission)** —
target **26.9 ± 10.2 MHz**, 3725 FWHM values. Then a sequence of experiments to diagnose/tune.
*(All runs: N_RUNS=200, N_ITER=80, μ₀=30, γ₀=15.)*

| Exp | Change vs base | μ_final | γ_final | FWHM final | W₁ | Verdict |
|---|---|---|---|---|---|---|
| **13** (base) | λ_sig=0.3, LR_γ=0.5 | 70.4 | 5.1 | 10.8 ± 2.0 | — | ❌ loss diverged, γ collapsed |
| **13a** | diagnose only | — | — | — | — | found a quantile bug + too-narrow dist |
| **13b** | quantile-matching fix | 100.4 | 12.4 | 25.5 ± 4.0 | 3.18 | ✅ mean 95 %, std 39 % |
| **13c** | λ_sig 0.3→0.1 | 78.2 | 12.3 | 25.6 ± 4.6 | 2.72 | ✅ best W₁ so far, μ still high |
| **13d** | LR_γ 0.5→5.0 (10×) | 63.1 | 18.5 | 38.6 ± 8.1 | 13.4 | ❌ γ overshot, worst |
| **13e** | LR_γ 0.5→1.5 (3×) | 62.9 | 11.8 | **24.8 ± 5.1** | **2.28** | 🏆 best overall |
| **13f** | λ_bg 2.0→5.0 | 85.2 | 11.8 | 25.8 ± 4.3 | 2.91 | ≈ 13e, μ less stable |

*(13c-faster-gamma / 13c-more-iterations were never executed; their conclusions folded into 13c.)*

**Takeaways.** Mean matching works (92–96 %). **The spread is stuck at ~40–50 % of target
regardless of hyperparameters** → a **model gap, not hyperparameters**: real PLE scans are fitted
with **Voigt** profiles (Gaussian broadening from spectral diffusion), our simulator is
**Lorentzian-only**. μ is inconsistent (63 → 100) — weakly identified.
*(This "spread gap" becomes failure mode #6.)*

## 14 — Uncertainties by bootstrap

**`14-uncertainties-dummy.ipynb`.** First uncertainty experiment: bootstrap copies of the target
data (resample the 200 (FWHM, σ) pairs with replacement), re-run the full 12c optimization on each
copy, and take the **spread of recovered (μ, γ)** as the uncertainty. K=30 replicas (cached to
`data/processed/bootstrap_results.json`).

**Honest caveats recorded:** the cloud mixes *data noise* (bootstrap) with *optimizer noise*
(stochastic REINFORCE); bootstrap measures variance, not bias (bias/coverage checkable only on
synthetic data); optimizer is expensive (~4–7 min/run).
*(17 Jul debate: a **seed** bootstrap measures optimizer variability, not statistical uncertainty
— "tight, confident, wrong error bars." Decision: empirical data bootstrap, but ultimately the
**Fisher/CRB** route won because it is mathematically solid and Gregor endorsed it.)*

## 15 — Real-data sweep (15a–15n)

**`15a…15n`** (14 files: 2 powers × 7 transmissions). Applies the **12d model** (2D KDE likelihood
+ Fisher/CRB) to all 14 real experiments. Rerun with `FISHER_SEEDS` 1 → 5 (step 1 documented;
optimization fully seeded → recovered (μ, γ) identical, only the σ estimates shift ±10–30 %).

**Trend.** **σ_γ decreases strongly with transmission** — collapses 1–2 orders of magnitude from
5 % → 20 % transmission, then plateaus at ~1–2 MHz (1 nW: 72.3 → 0.9; 3 nW: 4.2 → 1.5). **σ_μ
stays weakly identified** at all transmissions (~50–60 % relative at 3 nW). **Recovered μ runs
biased low at high transmission** (up to ~2σ). Figures `fig_15_*.png` + `_dist` variants.
*(First appearance of the μ-collapse, then unexplained.)*

## 16 — Closed-loop synthetic diagnostic (16a–16n)

Same pipeline as the 15-series, but the **target is generated by our own simulator at the true
values** (μ_true, γ_true, σ_prop, λ), with N_TARGET mirroring each real experiment. If the model
can't recover *its own* truth → a pure model/optimization gap.

**Result.** Healthy on its own data at low/mid transmission (all within ~2σ, most ≪ 1σ). At high
transmission (60–100 %) it **systematically fails**: γ biased low (16n: 7.6 vs 14.1, 75σ;
16l: 10.3 vs 14.1, 16σ), μ biased low (16n: 91 vs 176). Two pathological Fisher signatures:
16m near-singular (σ_μ = 1317), 16n over-confident (σ_γ = 0.09).

**Diagnosis.** The high-T real-data μ-bias is **at least partly a model/optimization property**,
not only data mismatch. **This is when the focus shifted:** stop trying to fix it with
hyperparameters, go back to synthetic data where the truth is known, and make the model perfect
there first. *(Journal 23 Aug: "if my model is not able to optimize to the data that the same
model generated, then I have some issues with the model and not with the data.")*

## 17 — The improvement playground (17b–17g) — 🎯 the synthetic win

One notebook runs **all 14 synthetic experiments** (16-series setup) with knobs, so we can see the
optimization *paths* and the MSE/STE across experiments. Knobs: `LAMBDA_MEAN`, `H_S_MIN`,
parallelism; config: N_RUNS=200, N_ITER=80→200, LR_MU=15, LR_GAMMA=0.5, CLIP=10, SEED=42,
SYNTH_SEED=12345.

| Notebook | One change | Result |
|---|---|---|
| **17b** | λ_mean = 0.0 baseline | γ RMSE ~1.2; high-T μ loss |
| **17c** | γ_true halved (narrow-line test) | narrow-line regime diagnostic |
| **17d** | λ_mean = 0.1 | γ RMSE 3.32 (worse) |
| **17e** | λ_mean = 0.0, N_ITER 80→200 | fixes "80 too few" |
| **17f** | **σ_ref = 10** (scale-invariant μ step), no LR floor, γ anneal | closes high-T μ loss |
| **17g** | **z-form γ-score** (bandwidth-normalized) | γ RMSE 2.27 → **0.59** |

**The λ-sweep verdict (from 18, §next):** γ RMSE **1.20 (λ=0) → 3.32 (λ=0.1) → 4.48 (λ=0.3)** —
the mean anchor is **monotonically harmful**; λ_mean = 0 wins. (Meanwhile μ was flat-to-better at
λ=0.)

**17f (σ_ref fix).** Diagnosed as **optimizer starvation** (FM#1): the μ-score divided by
σ_prop² makes steps ~0.1 photons at high T; the 80-iter probe on 3nW T100 showed μ still climbing
(+0.12 photons/step) at step 80. Fix: fixed σ_ref → identical step size in photon units.

**17g (z-form γ-score) — the synthetic victory.** Result:

> **μ RMSE 1.131 (3.7 %), γ RMSE 0.588 (4.3 %)** across all 14 synthetic experiments. 13/14
> essentially perfect; the one residual is 3nW T05 γ = 16.24 (truth 14.1, +2.14).

**On self-consistent data the machinery is excellent.** This is the config that became the
**AG-HYPOPT baseline**. *(The z-form fix was later ported to real data in 18c.)*

## 18 — Failure analysis and real-data sanity (18, 18b, 18c)

**`18-failure-analysis.ipynb` — post-mortem of the 17-series.** Three failures: (1) γ degrades as
λ_mean grows; (2) high-T 3nW loses μ (≈60–65 % of truth), **independent of λ**; (3) 1nW T10 γ
outlier. Diagnostics (μ fixed-point scan $E_B[n]$ vs NLL(μ); NLL(γ) scans; full 80-iter re-run
with history) → **verdict: the high-T μ loss is H1, optimizer starvation** (H2 estimator bias and
H3 weighting bias ruled out — the NLL minimum and the REINFORCE fixed point both sit at/above
μ_true). Fix: σ_ref normalization (17f). Also flagged: **KDE skewness/bandwidth inflation** (FM#4)
— 3nW T40 target is pathologically skewed (skew +15.9 F / +36.2 σ), inflating Scott's bandwidth.

**`18b-real-data-sanity.ipynb` — does the 17f fix transfer to real data?** Uses the exact 17f
optimizer on the **real** FWHM data + Fisher/CRB. **Result:** μ lands at **~½ of the reference
everywhere** but within **huge σ_μ** — the μ likelihood is **flat**: *honest but low-information*.
γ: 3nW high-T good; 1nW systematically low. *(This is where "honest-wide" is discovered: the ½
attractor is inside the (huge) interval.)*

**`18c-real-data-gammascale.ipynb` — port the z-form γ-score to real data.** Result:

> **γ RMSE 3.540 (32.0 %) vs 18b's 5.83** — the γ normalization removes the jumps and stabilizes
> 1nW γ. But the **1nW low-γ offset persists** (T20 5.54, T60 7.06 vs 8.5) → **model mismatch,
> not optimizer** (FM#6, optimizer exonerated). New low-T regression: 3nW T05 5.59 (−8.51σ),
> T10 8.31 (−5.79σ). **μ untouched** (RMSE 45.99, 52.4 % — weak identifiability).

After 18c, the picture was clear: **the γ machinery is fixed and understood; μ on real data is the
open problem; and there are two candidate explanations for the low-γ / μ-attractor — the line
shape (Voigt) and the likelihood/score structure.** Series 19 was built to decide between them.

## 19 — Series 19: the honest-uncertainty round (19a–19d) — 🔬 the current frontier

**Goal (Anuar, 2026-08-30):** the model must produce **honest uncertainties** — tight-and-accurate
where identified, wide-but-truth-inside where not. **Tight and wrong is the failure.** Four trials,
each with a written pre-registered design and a post-run verdict.

### 19a — Does Gaussian (Voigt) broadening explain it? → **REJECTED**
Closed-loop sweep: targets generated from a **Voigt** line (Cauchy + Gaussian σ_G, truncated),
fitted with the **same Lorentzian MLE**; σ_G ∈ {0,2,4,6,8} × 6 exps; 18c optimizer untouched.
- Control (σ_G=0) validates the harness (17g-quality recovery).
- **No half-μ attractor**: μ̂/μ_true stays 0.99–1.03 at every σ_G → broadening does **not** pull μ
  down.
- **γ moves the WRONG way**: Voigt targets *inflate* γ̂ by +13–34 % (opposite of real low-γ).
- No spread gap either.
- **Real finding:** **tight-wrong γ basins** at σ_G ≥ 6 (|Δγ|/σ up to 3.9 with small σ).
- **Decision → 19b:** Voigt physics explains neither symptom; attack real-data μ identifiability
  directly. *(src/series19.py Voigt machinery kept for regression.)*

### 19b — Where does the real-data μ attractor come from? → **the σ_fit channel**
Autopsy on 6 real exps (1nW/3nW × T20/T60/T100): NLL(μ) grid at fixed γ_true + channel-score
decomposition + sim-vs-real σ_fit quantiles.
- **The attractor is REAL**: NLL(μ) minimizes at **μ/μ_true = 0.30 in all 6 exps** (NLL 2.6–3.3 —
  a healthy fit, not flatness).
- **Count channel is DEAD**: grad_count ≈ 0 (KDE responsibility uniform at real N).
- **σ_fit channel mismatched**: sim σ_fit vs real error medians = 1.04/2.05/2.36 (1nW),
  2.15/3.44/4.25 (3nW) — gap **grows with T**. Only way to match = photon starvation → low-μ mode.
- Three μ-score designs: **D0** (σ_ref=10) tight-wrong (|Δμ|/σ up to 16.6); **D1**
  (σ_ref=σ_prop) honest-but-useless (σ_μ ≈ 0.5 μ_true); **D2** (raw σ-gradient) collapses μ to the
  floor (|Δμ|/σ up to **557**).
- **Decision → 19c:** scale the simulator's σ_fit by $c_{\exp}=\text{real\_err}/\text{sim}_\sigma@
  \mu_{\text{true}}$ (1.04–4.25, monotone in T).

### 19c — Does σ-scale calibration move the mode? → **NO**
C0 (D1, c=1) control / C1 (D1, c=c_exp) / C2 (D2, c=c_exp).
- **Key negative result:** with c=c_exp the σ scale IS matched at μ_true, yet **NLL still
  minimizes at μ/μ_true = 0.5 in all 6 exps**. **The attractor is the SHAPE of σ(μ), not its
  scale.**
- C1: μ pulled only to 0.54–0.69× (honest-wide); fixes 1nW T100 γ (8.49 vs 8.5, 0.009σ).
- C2: **recovers both T20 exps** (1.04×/0.12σ, 1.13×/0.40σ) but **collapses T60/T100** to the
  floor (|Δμ|/σ 150–323) — the restoring pull ∝ c·σ/n is **un-damped**.
- **Decision (neither branch) →** μ is unidentifiable under any Lorentzian-σ model → 19d = report
  D1 honest-wide on all 14 + document **failure mode #8**.

### 19d — FINAL honest-wide report (all 14 real exps) → **D1 is the answer**

**D1 (σ_ref = σ_prop, c = 1):**

| exp | μ̂/μ_true | σ_μ [MHz] | \|Δμ\|/σ | truth @2σ |
|---|---|---|---|---|
| 1nW T05 | 1.006 | 54.5 | 0.00 | ✅ |
| 1nW T10 | 0.991 | 33.6 | 0.00 | ✅ |
| 1nW T20 | 0.520 | 8.5 | 0.98 | ✅ |
| 1nW T40 | 0.505 | 19.3 | 0.98 | ✅ |
| 1nW T60 | 0.470 | 17.3 | 1.87 | ✅ |
| 1nW T80 | 0.471 | 23.8 | 1.76 | ✅ |
| 1nW T100 | 0.501 | 35.5 | 1.00 | ✅ |
| 3nW T05 | 0.769 | 12.5 | 0.25 | ✅ |
| 3nW T10 | 0.442 | 14.5 | 0.94 | ✅ |
| 3nW T20 | 0.430 | 25.6 | 0.76 | ✅ |
| 3nW T40 | 0.491 | 51.8 | 0.84 | ✅ |
| 3nW T60 | 0.474 | 57.6 | 0.94 | ✅ |
| 3nW T80 | 0.476 | 68.4 | 1.06 | ✅ |
| 3nW T100 | 0.483 | 90.7 | 1.00 | ✅ |

> **14/14 truth inside 2σ; max |Δμ|/σ = 1.87 (1nW T60), mean 0.88; 10/14 inside 1σ (≈ 68 %
> expectation).** σ_μ grows with T (8.5 → 90.7 MHz) — the honest signature of weak identifiability.
> The 0.43–0.52× attractor is real but **fully absorbed by the honest width**.

**C1 cross-check (c(T) law, 8 held-out exps).** $c(T)$ fitted linear-in-T per power from the 19b
autopsy points (1nW: $0.827+0.0165T$; 3nW: $1.705+0.02625T$; clamp [1,5]) and tested on the 8 exps
**not** in the fit: **8/8 inside 2σ, max 1.31σ** — **the c(T) law transfers**. Bonus: calibration
partially un-biases the mode (C1 μ̂ 0.54–0.65× vs D1 0.47–0.52×).

**γ summary.** D1 γ RMSE **3.23 MHz (31.0 %)** vs 18c 3.54 (32.0 %) — modest but real improvement.
**FM#6 persists at 1nW low-T** (γ̂/γ_true = 0.41/0.50/0.47 at T05/T10/T20). One degenerate Fisher
row: 3nW T05 (σ_γ → 5e-12, a tight-wrong artifact).

**Failure mode #8 (final).** *"Tight-wrong point estimates produced by score designs whose only
live gradient comes from a systematically mismatched channel."* Root cause: count channel dead +
σ_fit **shape** mismatch (1.04→4.25×, growing with T). **It is a score-design failure, not a data
failure.**

**Post-series recommendations (not yet run):**
1. **Damped σ-gradient** — the proven mechanism at low T (C2 recovered T20 with ≤0.40σ); damp/clip
   the pull relative to the measured σ mismatch.
2. **σ(μ) SHAPE matching**, not scale — match the *functional form* of sim σ vs real error.
3. **FWHM-spread channel** — model the spread explicitly.
4. **Until then: D1 is the honest answer — wide but true.**

---

# Part VI — Current state and the failure modes that matter

## Where the project stands

- The **differentiable MC pipeline works** and is validated end-to-end.
- **On synthetic data** (17g) it recovers both parameters almost perfectly (μ RMSE 3.7 %,
  γ RMSE 4.3 %). The machinery is *not* the problem.
- **On real data** (18c, 19d):
  - **γ** is well-behaved where the line-shape holds (3nW all T; 1nW T≥40), with z-form fixing the
    optimizer-level jumps. The 1nW low-γ offset is a **line-shape model mismatch (FM#6)**,
    untouched by any optimizer/hyperparameter change (19a proved it is neither the optimizer nor
    Gaussian broadening).
  - **μ** is **unidentifiable under any Lorentzian-σ model** on this dataset (19b/c/d). The
    correct answer is **honest-wide (D1)**: truth inside 2σ on 14/14 (and 8/8 held-out for C1).
- **Uncertainty method:** Fisher information / Cramér–Rao bound, per Gregor's endorsement.
- **Historical closure:** the c(T) σ-mismatch law transfers to held-out experiments → the μ
  problem is *understood*, even though not yet *solved*.

## The four failure modes that matter for the next phase

| # | Name | What it is | Status |
|---|---|---|---|
| **FM#1** | Optimizer starvation | μ-score /σ_prop² → tiny steps at high T | **Fixed** by σ_ref (17f) |
| **FM#4** | KDE skewness / bandwidth inflation | one target (3nW T40) is pathologically skewed → inflated Scott bandwidth | Known, open |
| **FM#6** | Voigt-vs-Lorentzian (γ) | real low-T 1nW γ systematically low; real scans fitted with Voigt | Open; **not** hyperparameters |
| **FM#8** | Un-damped σ-channel gradient | count channel dead + σ(μ) *shape* mismatch → tight-wrong μ | **Root cause found**; fix = damping + shape-matching (future) |

Plus the permanent structural lesson: **the μ–γ degeneracy** (§1.3) — always match FWHM **and** σ.

---

# Part VII — Starting point for hyperparameter tuning

This is the crux for the next phase. The tuning campaign is **AG-HYPOPT** (Agent-Guided
Hyperparameter Optimization) in `agent-guided-hypopt/experiment_1/`; method name **PAGHO**
(Physics-informed Agent-Guided HypOpt).

## 7.1 What is frozen (structural — NOT to be tuned)

Per `trials.json` `fixed_structural` — these are the design decisions the 17–19 series settled:

- **z-form γ-score** (`GAMMA_SCALE=True, H_REF=1.0`) — §4.10b. The 1/H² amplification was the γ
  disease.
- **σ_ref μ-score** — §4.10a. Fixed reference scale for the REINFORCE step.
- **λ_mean = 0** — §4.10c. The mean anchor is harmful (γ RMSE 1.20→4.48 as λ grows).
- **μ LR linear decay, no floor** (17f).
- **μ ∈ [1, 200], γ ∈ [0.1, 100]** clamps.
- **4 workers**; deterministic seeds (SEED=42, SYNTH_SEED=12345).
- Also frozen at the model level: pure-Lorentzian simulator + Lorentzian MLE, truncation (not
  clipping), implicit-diff derivatives, 2D KDE likelihood + Scott bandwidths.

> Structural changes (score, likelihood, model, template, protocol) need **Anuar's OK** — they are
> explicitly *out of scope* for the tuning campaign (per `instructions.md`). This protects against
> "tuning away" residuals that are actually estimator/score artifacts (the trial_01 lesson below).

## 7.2 What is tunable (the declared space)

From `agent-guided-hypopt/experiment_1/space.json` — 8 hyperparameters:

| Parameter | Type | Range | Meaning |
|---|---|---|---|
| `n_runs` | int | 100 – 500 | simulated runs per optimization step |
| `n_iter` | int | 100 – 400 | optimization steps |
| `lr_mu` | float | 5.0 – 40.0 | REINFORCE μ learning rate |
| `lr_gamma` | float | 0.1 – 1.5 | γ learning rate |
| `sigma_ref` | float | 5.0 – 25.0 | μ-score reference scale |
| `clip` | float | 5.0 – 20.0 | gradient clipping bound |
| `gamma_anneal` | float | 0.0 – 0.75 | γ LR anneal factor |
| `h_s_min` | float | 0.0 – 0.2 | σ-kernel bandwidth floor (MHz) |

**Budget constraint:** `n_runs · n_iter ≤ 40000` (~3.5 h for the full 14-experiment benchmark;
`BUDGET_HOURS=3.5`). Dependencies: none declared (flat space).

## 7.3 The benchmark and objective

- **Benchmark:** the 14 **synthetic** experiments (1nW/3nW × Trans05–100), targets at the true
  values with `SYNTH_SEED=12345` (identical across trials); deterministic runs `SEED=42`. **We know
  the ground truth**, so we can score recovery directly — and later calibrate coverage.
- **Objective:** combined **relative MSE of (μ, γ)** over the 28 per-experiment errors + a sampling
  SE term; **lower is better; no Fisher**. This is a clean, scale-free score for tuning.

## 7.4 Results so far

| Trial | Config highlights | Objective (lower = better) | Verdict |
|---|---|---|---|
| **baseline_17g** | the 17g config: n_runs=200, n_iter=200, lr_mu=15, lr_gamma=0.5, σ_ref=10, clip=10, anneal=0.5, h_s_min=0.05 | **0.001578 ± 0.000859** | μ rel-RMSE 3.6 %, γ 4.3 % |
| **trial_01** | lr_mu 30.3, σ_ref 8.2, mild anneal, h_s_min 0.14 | 0.00491 ± 0.00207 | **3.1× worse** — sign-coherent μ overshoot on all 14 exps |
| **trial_02** | lr_mu 36.4, σ_ref 16.1, n_iter 348, clip 6.6 | 0.003112 ± 0.00180 | 2.0× worse than baseline; γ rel-RMSE 2.6 % (best), 3nW T05 γ outlier resolved |

**The tuning campaign's first big insight (trial_02):** gentler-longer μ drive cured the high-T
transient overshoot but left a **drive-independent positive μ bias at low n_target** — the μ
score's **fixed point sits above truth**, so μ recovery needs a **structural score change**
(σ_ref normalization / sample-size correction), **not more hyperparameter tuning**. γ needed no
such fix. **This is precisely the "don't tune away a structural artifact" lesson** that the
authority rule (7.1) is designed to enforce.

## 7.5 What Anuar wants from the tuning phase (his design, journal 23 Aug)

- **Goal:** make all 14 synthetic optima land as close as possible to their true (μ, γ).
- **Method:** TPE (HyperOpt) proposes candidate configurations; an **agent brings physics context**
  to pick among proposals. Concrete idea: sample ~30 candidates from the TPE distribution, let the
  agent analyze them with physical context and **choose one of the 30** — so the agent stays within
  the probabilistic search but can inject domain reasoning.
- **Objective:** MSE of final optima vs truth across the 14 experiments (this is now realized as the
  combined rel-MSE in `trials.json`), with the **std as an uncertainty** that could also feed the
  TPE kernel.
- **Concrete next steps (his):** (1) a compact notebook running all 14 experiments; (2) define all
  hyperparameters; (3) build the agentic optimizer. The first two are **done** (17g / space.json);
  the third is the AG-HYPOPT machinery now in place, with trials 01–02 logged.

> **Conscious starting point — the short version.** The right baseline is the **17g config**
> (objective 0.001578). Everything structurally important is frozen and *not* yours to tune:
> z-form γ-score, σ_ref μ-score, λ_mean=0, decay-no-floor, clamps. What remains to tune is the
> 8-parameter space in 7.2, under the 40 000 budget. And be warned by trial_02: the residual μ bias
> at low n_target is a **score fixed-point artifact**, not a hyperparameter — if it shows up, it is
> a note for Anuar, not something to tune away.

---

# Part VIII — Glossary

| Term | Meaning |
|---|---|
| **μ** (n̄) | mean photon number per scan — a primary recovery target |
| **γ** | Cauchy/Lorentzian scale = HWHM (MHz); FWHM = 2γ |
| **σ_prop** | proportional photon-count noise, fixed per experiment |
| **σ_fit** | per-scan fit uncertainty on the FWHM (the second matched observable) |
| **λ** | mean of the additive Poisson background (fixed per experiment) |
| **n_target** | number of (filtered) measured FWHM samples in an experiment |
| **FWHM / HWHM** | full / half width at half maximum |
| **PLE** | photoluminescence excitation spectroscopy |
| **NV** | nitrogen-vacancy center in diamond |
| **SIL** | solid immersion lens |
| **TransXX** | transmission setting (5 %–100 %), the SNR knob |
| **KDE** | kernel density estimate — our likelihood |
| **REINFORCE** | score-function (policy) gradient estimator used for μ |
| **NLL** | negative log-likelihood |
| **IFT** | implicit function theorem (differentiation through the fit) |
| **CRB / Fisher** | Cramér–Rao bound / Fisher information — our uncertainty method |
| **attractor** | the μ the REINFORCE update converges to: $E_B[n]=\bar n$ |
| **σ_ref** | fixed reference scale in the μ-score (scale-invariant step) |
| **z-form γ-score** | bandwidth-normalized γ-score (fixes saturation/jumps) |
| **λ_mean** | mean-FWHM anchor weight — **disabled (harmful)** |
| **Scott bandwidth** | data-driven KDE width: $\sigma_{\text{data}}\cdot N^{-1/6}$ |
| **honest-wide** | a wide interval that *does* contain the truth (success) vs **tight-wrong** (failure) |
| **FM#k** | failure mode k (§VI) |
| **AG-HYPOPT / PAGHO** | the agent-guided hyperparameter-optimization campaign / one trial of it |

---

## References and key files

- **Paper:** `docs/papers/2501.07951.pdf` (Orphal-Kobin et al., arXiv:2501.07951).
- **Journal (human-readable development log):** `notes/JOURNAL.md` (through 2026-08-23).
- **Model/physics reference:** `agent-guided-hypopt/experiment_1/context.md` (v3, most complete).
- **Source:** `src/samplers.py`, `src/fitting.py`, `src/implicit.py`, `src/losses.py`,
  `src/series19.py`, `src/utils.py`.
- **Data:** `data/raw_data/data_explanation.md` (+ per-experiment true params).
- **Tuning campaign:** `agent-guided-hypopt/experiment_1/{trials.json, space.json, instructions.md}`.
- **Presentations:** `notes/presentations/2026-07-03`, `2026-07-17`.

*End of report.*
