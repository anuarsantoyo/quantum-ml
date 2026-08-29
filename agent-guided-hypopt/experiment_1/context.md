# Context — physics background and model description

> **What this file is.** The complete physics and model background for this project, written so that
> an agent with no prior knowledge can understand *what we are recovering, why it is hard, and how our
> model works* — purely from this file. It is deliberately **generic**: it can be copied into any
> experiment folder of this project without modification.
>
> **What is NOT here.** The concrete experimental protocol (objective, benchmark, data files,
> parameter tables, current results) belongs to the experiment — see that experiment's `trial_00`
> notebook. The optimization loop and agent workflow are described in `instructions.md`.
>
> *DRAFT — being reviewed with Anuar (2026-08-29).*

---

## 1. The paper that started this project

**Orphal-Kobin, Pieplow, Gokhale, Unterguggenberger, Schröder (Humboldt-Universität zu Berlin) —
"Retrieving Lost Atomic Information: Monte Carlo-based Parameter Reconstruction of an Optical
Quantum System"** (arXiv:2501.07951, January 2025).

### 1.1 The problem

Optical characterization of single quantum emitters — here: the nitrogen-vacancy (NV) color center
in diamond — requires measuring the emitter's optical linewidth via photoluminescence excitation
(PLE) spectroscopy. In this technique, a resonant laser scans across the emitter's transition and the
fluorescence is collected as a function of laser frequency. Each scan yields a fitted linewidth.

The trouble starts at **low signal**: when few photons reach the detector (low collection
efficiency, weak emission, high losses), the standard estimator — the **median of the fitted
linewidths** — becomes *precise but inaccurate*: its confidence intervals shrink (it looks
trustworthy) while its value is systematically wrong, e.g. collapsing **below the lifetime-limited
linewidth**. This is the classic failure of naive statistics in undersampled regimes: the fit
degenerates and the few surviving fits are biased.

### 1.2 Their method (Monte Carlo reconstruction)

Instead of trusting a single estimator, they simulate the whole experiment:

1. **Forward model.** A photon detection event is sampled from a **Cauchy (Lorentzian) line**
   $\gamma/\pi(\omega^2 + \gamma^2)$ with scale parameter $\gamma$, plus noise events sampled from a
   Poisson distribution (mean 2 in their setup). The resulting spectrum is fitted with a
   **Voigt profile** and its FWHM recorded.
2. **Histogram.** Repeating this many times produces a *simulated distribution of fitted FWHMs*.
3. **Matching.** The simulated histogram is compared to the *measured* FWHM distribution with a χ²
   test:
   $$S(\gamma, \bar n) = \sum_i \frac{(O_i - E_i(\gamma, \bar n))^2}{E_i(\gamma, \bar n)}$$
   minimized over the linewidth $\gamma$ and the mean photon number $\bar n$. A 99 % confidence
   region is given by $S \le S_{\min} + 9.21$.

### 1.3 Their findings

- The bias of the median depends on **both** $\gamma$ and $\bar n$: **broader lines need more
  photons** to be estimated accurately.
- There is a clear **threshold** in (number of scans, mean photon number) below which standard
  analysis is unreliable; the median can be precise yet biased across a broad parameter range.
- They explicitly identify **machine learning as a promising avenue** for the same parameter
  estimation task.

### 1.4 Why this paper is our foundation

- **Their $\gamma$ = our $\gamma$** — the Cauchy scale parameter. Note the factor-2 convention:
  the FWHM of a Lorentzian is $2\gamma$. Our reference values are defined as
  $\gamma_{\text{true}} = \text{median FWHM @ Trans100} / 2$, exactly to recover the Cauchy $\gamma$.
- **Their $\bar n$ = our $\mu$** — the mean photon number.
- **Their χ² grid search = the ancestor of our likelihood.** We keep their Monte Carlo forward
  model, but replace the brute-force χ² minimization with a **differentiable kernel-density
  likelihood + REINFORCE gradient optimization** (Section 4). That is the "machine learning
  avenue" the paper points to.
- We also refine their noise model: they fixed the Poisson noise mean to 2; we use a
  **per-experiment noise model** $(\sigma_{\text{prop}}, \lambda)$ (Section 3).

---

## 2. The physics of the experiment

### 2.1 The system

- **NV center in diamond**: an atom-like defect with a narrow optical transition; a workhorse
  platform for quantum information and sensing. Its optical properties (linewidth, spectral
  diffusion) are the quantities of interest.
- **PLE spectroscopy**: a resonant red laser scans the transition; fluorescence is collected.
  Because the excitation is resonant, the collected signal is proportional to how well the laser
  hits the (broadened) transition — this is what makes low-signal regimes hard.
- **Solid immersion lens (SIL)**: collection optics that increase the collection efficiency by
  reducing total internal reflection at the diamond-air interface.

### 2.2 The measurement knobs

- **Excitation power** (resonant red laser): higher power → more photons, but also **power
  broadening** of the line (median FWHM rises with power).
- **Transmission setting** (neutral-density filter in the detection path, `Trans05 … Trans100`
  = 5 %–100 %): directly controls how many photons reach the detector → an **effective
  signal-to-noise (SNR) knob**. At low transmission, few fits succeed and the surviving FWHMs are
  biased; at high transmission, thousands of good fits are available.
- **Repump laser** ("Top"): keeps the NV in the correct charge state.

### 2.3 Why low signal breaks naive analysis

At low photon numbers per scan, the Lorentzian fit degenerates (near-zero FWHM, exploding fit
error, failed fits). Filtering out bad fits removes the informative tail; keeping them adds noise.
The measured FWHM distribution therefore becomes a *biased, heavy-tailed* sample of the true line —
which is precisely the regime where simulation-based reconstruction (Section 1.2) wins.

---

## 3. The forward model (what we simulate)

Every simulated "measurement" is generated as follows (see `src/samplers.py`, `src/fitting.py`):

1. **Photon count.** The number of photon events $n$ for one scan is drawn from a Gaussian plus
   Poisson mixture:
   $$n \sim \mathcal{N}(\mu, \sigma_{\text{prop}}) + \text{Poisson}(\lambda)$$
   - $\mu$ — **mean photon number** (the main recovery target; physically the expected signal).
   - $\sigma_{\text{prop}}$ — **proportional noise** (standard deviation of the photon-count
     fluctuations; fixed per experiment from calibration fits).
   - $\lambda$ — **mean of additive Poisson noise** (dark counts / background; fixed per
     experiment). The paper used $\lambda = 2$; we fit it per experiment.
2. **Photon stream.** The photon events are placed on a **Lorentzian line of scale $\gamma$**
   (the emitter linewidth we want to recover).
3. **Fit.** The resulting spectrum is fitted with a **Voigt profile** (Lorentzian with
   `uniform_bg=False`), yielding an FWHM and a fit error. Optionally, implicit differentiation
   gives $\partial(\text{FWHM}, \sigma)/\partial\gamma$.
4. **Repeat.** Thousands of such scans build a *simulated FWHM distribution* (plus the
   distribution of fit errors $\sigma$).

**Key property:** the forward model is stochastic and *cheap to sample* — this is what makes
Monte Carlo inversion feasible. The simulated distribution is a deterministic function of the
parameters $(\mu, \gamma)$ (given fixed noise parameters), which is what the likelihood exploits.

---

## 4. The inverse problem (how we recover parameters)

### 4.1 The likelihood

We compare the *simulated* $(FWHM, \sigma)$ pairs to the *measured* ones with a **2D kernel
density estimate (KDE)** likelihood:

- Each simulated scan contributes a Gaussian kernel of width $H_F$ along FWHM and $H_S$ along the
  fit error $\sigma$.
- **Bandwidths follow Scott's rule** for 2D: $H = \sigma_{\text{data}} \cdot N^{-1/6}$, with an
  absolute floor $H_S^{\min}$ on the $\sigma$-kernel to avoid over-concentration.
- The likelihood of a measured point is the mean kernel density of the simulated cloud at that
  point; the **negative log-likelihood (NLL)** is the optimization target.

Why KDE: no parametric assumption about the FWHM distribution (which is genuinely non-Gaussian,
heavy-tailed, skewed), and it is differentiable end-to-end.

### 4.2 The optimizer (REINFORCE-style)

We jointly optimize $(\mu, \gamma)$ with gradient estimates computed from the kernel weights:

- **Kernel weights** $w_{ij}$: how much simulated scan $j$ "explains" measured point $i$
  (normalized Gaussian similarity). The weight vector $B_j = \langle w \rangle_i$ is the
  probability that scan $j$ belongs to the data.
- **μ gradient (REINFORCE):** the score is the photon count relative to the current mean,
  $(n - \mu)/\sigma_{\text{ref}}^2$, where $\sigma_{\text{ref}}$ is a **fixed reference scale**
  (see 4.3). The gradient is the kernel-weighted covariance between weights and score:
  $\nabla_\mu \propto -(B - \bar B) \cdot \text{score}$.
- **γ gradient:** the derivative of the KDE log-density with respect to γ,
  $\nabla_\gamma \propto -\langle s_\gamma \rangle$, where $s_\gamma$ is built from the
  $\partial FWHM/\partial\gamma$ and $\partial \sigma/\partial\gamma$ terms.
- **Updates:** gradient clipping (CLIP), parameter bounds ($\mu \in [1, 200]$ photons,
  $\gamma \in [0.1, 100]$ MHz), learning rates with scheduled decay.

### 4.3 Why the design choices exist (the reasoning trail)

- **σ_ref normalization (scale invariance).** Dividing the μ-score by the *per-experiment*
  $\sigma_{\text{prop}}^2$ starves high-transmission experiments: at $\mu \approx 176$ photons,
  $\sigma_{\text{prop}}^2 \approx 1680$, so each gradient step moves μ by ~0.1–0.5 photons and the
  optimizer needs ~500+ iterations to climb the remaining distance (the "starvation" failure mode,
  Section 5). Normalizing by a **fixed** $\sigma_{\text{ref}}$ makes the step size identical in
  photon units for every experiment — high-μ experiments travel fast, low-μ experiments don't
  overshoot.
- **The attractor.** The REINFORCE μ-update drives μ toward the *kernel-weighted mean photon
  count* $\bar n$ of the scans that the data actually matches ($E_B[n] = \bar n$). If the model
  and data disagree in shape, the attractor sits at the "best-fit" μ — which is why a biased
  likelihood shows up as a clean, confident convergence to the wrong value.
- **Mean-matching anchor (λ_mean) is harmful.** Adding a term that matches the simulated mean
  FWHM to the data mean *destabilizes* γ recovery (verified experimentally: every metric got
  worse, γ RMSE 1.20 → 3.32 → 4.48 as the anchor weight grew). It is disabled (λ_mean = 0).
- **γ anneal.** The γ learning rate decays as $(1 - 0.5 t/N)$. Intended to settle noisy
  low-count γ estimates; known side effect: it can *freeze* γ in the low-count regime
  ($n_{\text{target}} \lesssim 500$) before convergence (Section 5).
- **Determinism.** All randomness is seeded (per-step noise, target generation), so a given
  configuration is reproducible and trials are comparable.

### 4.4 Synthetic vs real data

- **Synthetic targets:** drawn from the forward model at the *true* parameters — the controlled
  benchmark (model matches data by construction; failure modes are optimizer/architecture failures).
- **Real targets:** measured FWHM distributions from the experiment — the ultimate test. Real data
  has skewed, heavy-tailed clouds and model-data mismatch, which is where the interesting failures
  appear.

---

## 5. Known failure modes and physics insights (durable lessons)

These are *stable* insights about the model — knowledge, not state. They are the physics intuition
the agent should apply when judging candidates.

1. **Optimizer starvation.** μ-score divided by $\sigma_{\text{prop}}^2$ → tiny steps at high μ;
   optimizer stops far from the NLL minimum (hundreds of iterations short). *Fix: fixed σ_ref
   normalization (4.3).* Diagnosis signature: NLL still decreasing at the end of the run; μ stuck
   at ~½–⅔ of the true value while the NLL-minimum is at/above μ_true.
2. **The attractor trap.** μ converges to $E_B[n] = \bar n$, the kernel-weighted mean of the scans
   the data "explains". If the likelihood is biased (model-data mismatch), this is a clean
   convergence to the wrong value — *not* an optimizer bug. Always check the attractor against the
   NLL minimum.
3. **Low-count γ noise.** When $n_{\text{target}} \lesssim 500$ (e.g. the Trans05 experiments), the
   γ-gradient from ~200 runs vs ~few-hundred targets is too noisy for the nominal learning rate:
   γ either explodes upward or collapses toward its initialization. *The γ anneal makes this worse
   (freezes γ); more runs or a per-experiment γ learning rate is the natural counter.*
4. **KDE skewness / bandwidth inflation.** If the target cloud is pathologically skewed (one
   experiment showed skewness +15.9 F / +36.2 σ), Scott's bandwidth inflates, the likelihood
   flattens, and the NLL becomes a poor discriminator. This is a *likelihood-model* issue, not an
   optimizer issue.
5. **Open question — real-data attractor.** On real data, the fitted μ has repeatedly landed at
   roughly *half* the calibration reference μ, even with a correctly converging optimizer. Two
   competing explanations: (a) the real-data likelihood genuinely prefers a lower μ (model-data
   mismatch: the real FWHM cloud is much broader than the model produces at the reference μ), or
   (b) the calibration reference itself is off. Untested discriminator: the Fisher/CRB width at
   the fitted point. *This is an active investigation — the resolution belongs to the experiment's
   trial_00.*

---

## 6. Units and conventions

- **Frequency units:** MHz. (The absolute physical unit of the raw linewidths was never recorded —
  the pipeline works in MHz by convention.)
- **FWHM vs γ:** the fitted quantity is the FWHM of a Lorentzian/Voigt; the model parameter γ is
  the Cauchy scale, **FWHM = 2γ**. Reference values are stored as γ (half-FWHM).
- **Data filtering:** valid rows only (non-NaN, FWHM > 0, fit error ratio `err/fwhm < 10`).
- **Scott exponent:** $N^{-1/6}$ for the 2D KDE bandwidths.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| μ | mean photon number per scan (primary recovery target) |
| γ | Cauchy/Lorentzian scale — half the FWHM (MHz) |
| σ_prop | proportional photon-count noise (fixed per experiment) |
| λ | mean of additive Poisson noise events (fixed per experiment) |
| n_target | number of (filtered) measured FWHM samples in an experiment |
| FWHM | full width at half maximum of the fitted line |
| PLE | photoluminescence excitation spectroscopy |
| NV | nitrogen-vacancy color center in diamond |
| SIL | solid immersion lens |
| TransXX | transmission setting (5 %–100 %), an SNR knob |
| KDE | kernel density estimate (our likelihood) |
| REINFORCE | score-function gradient estimator used for μ |
| NLL | negative log-likelihood |
| attractor | the μ value the REINFORCE update converges to: E_B[n] = n̄ |
| Scott bandwidth | data-driven KDE width: σ_data · N^{-1/6} |
| Voigt | convolution of Lorentzian (line) and Gaussian (instrumental) |
