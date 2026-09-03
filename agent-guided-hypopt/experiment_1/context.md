# Context — physics background and model description

> **What this file is.** The complete physics and model background for this project, written so that
> anyone joining the project (agent or human) can understand *what we are recovering, why it is hard,
> and how our model works* purely from this file. It is deliberately **generic**: it can be copied
> into any experiment folder of this project without modification.
>
> **What is NOT here.** The concrete experimental protocol (objective, benchmark, data files,
> parameter tables, current results) belongs to the experiment — see that experiment's `trial_00`
> notebook. The optimization loop and agent workflow are described in `instructions.md`.
>
> *Version 3 — rewritten 2026-08-29 from a full re-read of the repository, the source code and the paper.*

---

## 1. The paper that started this project

**Orphal-Kobin, Pieplow, Gokhale, Unterguggenberger, Schröder (Humboldt-Universität zu Berlin) —
"Retrieving Lost Atomic Information: Monte Carlo-based Parameter Reconstruction of an Optical
Quantum System"** (arXiv:2501.07951, January 2025).

### 1.1 The problem

Optical characterization of single quantum emitters — here: the nitrogen-vacancy (NV) color center
in diamond — requires measuring the emitter's optical linewidth via photoluminescence excitation
(PLE) spectroscopy. A resonant laser scans across the transition; fluorescence is collected per
frequency step; each scan yields one fitted linewidth.

The trouble starts at **low signal**: when few photons reach the detector, the standard estimator —
the **median of the fitted linewidths** — becomes *precise but inaccurate*: its confidence
intervals shrink (it looks trustworthy) while its value is systematically wrong, e.g. collapsing
**below the lifetime-limited linewidth**. As the number of scans grows, the median's confidence
interval can become *narrower than the bias itself* — the textbook failure of naive statistics in
undersampled regimes.

### 1.2 Their method (Monte Carlo reconstruction)

Instead of trusting a single estimator, they simulate the whole experiment and match distributions:

1. **Forward model.** One simulated scan: photon detection events are sampled from a
   **Cauchy (Lorentzian) line** $\gamma/\pi(\omega^2 + \gamma^2)$ with scale $\gamma$, plus noise
   events from a **Poisson distribution with mean 2**. The spectrum is fitted with a
   **Voigt profile** (least squares on a binned spectrum, 2.2 MHz/step) and its FWHM recorded.
2. **Histogram.** Repeating this for $k$ scans produces a *simulated distribution of FWHMs*.
3. **Matching.** The simulated histogram is compared to the *measured* FWHM distribution with a χ²
   test, minimized over the linewidth $\gamma$ and the mean photon number $\bar n$:
   $$S(\gamma, \bar n) = \sum_i \frac{(O_i - E_i(\gamma, \bar n))^2}{E_i(\gamma, \bar n)},$$
   with the 99 % confidence region given by $S \le S_{\min} + 9.21$.

### 1.3 Their findings

- The median is precise but biased; **both the bias and the needed photon number depend on γ and n̄**:
  broader lines need more photons.
- The MCM is consistent whenever the model represents the experiment; its bias stays below 2 % for
  $\bar n \gtrsim 30$ over a broad linewidth range, and its confidence intervals are honest.
- They define **thresholds**: the median becomes a reliable predictor only above
  $\bar n \approx 48$ (1 nW) / $56$ (3 nW) at $k = 2000$ scans — below that, use the MCM.
- They also tried **deep learning**: a Bayesian CNN (LeNet-inspired, input = 500 unprocessed line
  scans × 75 frequency bins) predicting (γ, n̄, σ) directly. It agrees well on **γ** but deviates
  on **n̄** (especially $\bar n < 20$). They expect an optimized DL routine to eventually surpass
  the MCM — the paper explicitly opens the door for the ML route this project takes.

### 1.4 Why this paper is our foundation

- **Their $\gamma$ = our $\gamma$** — the Cauchy scale = half-width at half-maximum (HWHM). The FWHM
  of a Lorentzian is $2\gamma$; hence our reference convention
  $\gamma_{\text{true}} = \text{median FWHM @ Trans100} / 2$.
- **Their $\bar n$ = our $\mu$** — the mean photon number.
- **Their χ² grid search over (γ, n̄) is the ancestor of our likelihood.** We keep the Monte Carlo
  forward model, but replace the brute-force grid search with a **differentiable pipeline**:
  gradient descent through the simulation (reparameterized sampling + implicit differentiation),
  a smooth **kernel-density likelihood** instead of the histogram χ², and a
  **policy-gradient (REINFORCE)** update for the discrete photon count.
- We also generalize their fixed noise model (Poisson mean 2) to **per-experiment** parameters
  $(\sigma_{\text{prop}}, \lambda)$ from calibration fits.

---

## 2. The physics of the experiment

### 2.1 The system

- **NV center in diamond**: an atom-like defect with a narrow optical transition; a workhorse
  platform for quantum information and sensing. The optical linewidth is the quantity of interest.
- The absorption probability of the NV transition is a **Lorentzian (Cauchy)** in frequency — the
  Fourier transform of the exponential decay (Weisskopf–Wigner); the Cauchy scale γ is the HWHM,
  with $\gamma = 1/(4\pi\tau)$ for a lifetime τ. The quartiles of the distribution sit at ±γ, and
  the tails decay only as $1/x^2$ (heavy tails — physically: rare, far-detuned events).
- **PLE spectroscopy**: a resonant red laser scans the transition; fluorescence is collected. At
  resonance the fluorescence peaks; off-resonance it drops. Each scan = one fitted line.
- **Solid immersion lens (SIL)**: collection optics that increase collection efficiency by reducing
  total internal reflection at the diamond-air interface.

### 2.2 The measurement knobs

- **Excitation power** (resonant red laser, 1 nW / 3 nW): more power → more photons, but also
  **power broadening** (median FWHM rises with power).
- **Transmission setting** (`Trans05 … Trans100` = 5 %–100 %, an ND filter in the detection path,
  equivalent to a collection efficiency $x_{\text{coll}} \in [0.05, 1]$): directly controls the
  photon rate reaching the detector → the **signal-to-noise knob**. At low transmission few fits
  succeed and the survivors are biased; at high transmission thousands of good fits exist.
- **Repump laser** ("Top"): keeps the NV in the correct charge state.

### 2.3 Why low signal breaks naive analysis

At low photon numbers per scan, the line fit degenerates (near-zero FWHM, exploding fit error,
failed fits). Filtering out bad fits removes the informative tail; keeping them adds noise. The
measured FWHM distribution becomes a *biased, heavy-tailed* sample of the true line — precisely the
regime where simulation-based reconstruction wins.

### 2.4 A degeneracy that shapes everything

**The FWHM distribution alone cannot separate μ from γ**: many (μ, γ) pairs produce nearly the same
FWHM distribution (more photons → better fits → narrower FWHM, in a way that trades against γ).
This μ–γ degeneracy is why our model matches **two outputs at once**: the FWHM distribution *and*
its per-fit uncertainty σ (Section 5). Matching only the FWHM makes the problem ill-posed; matching
FWHM + σ pins down a unique solution.

---

## 3. The measurement data

### 3.1 Format

Each real experiment is a 2-column table of **3200 fit attempts** (one anomaly: 4800 rows):

| Column | Meaning |
|---|---|
| 1 | line width (FWHM) of one PLE scan |
| 2 | fit error — 1σ uncertainty on that FWHM (same units) |

`nan nan` rows = failed fits. Filtering convention: keep rows with finite values, FWHM > 0, and
fit-quality ratio `err / fwhm < 10`. Valid-row count rises with transmission (~160 → ~2500):
more signal → more successful fits.

**Units:** the pipeline works in **MHz** by convention; the absolute physical unit of the raw line
widths was never recorded (open item with Gregor).

### 3.2 The 14 experiments and their true parameters

Two sessions (resonant red laser power **1 nW** and **3 nW**) × 7 transmission settings
(Trans05 … Trans100). For each, Gregor's calibration fits provide:

- `pho_normal_mean` = **μ_true** — mean of the Gaussian photon-count distribution,
- `pho_normal_std` = **σ_prop** — its standard deviation,
- `pho_noise_poisson_mean` = **λ** — mean of the additive Poisson noise.

Full table lives with the data (`data/raw_data/data_explanation.md`). Ranges: μ_true 9.4–175.7
photons, σ_prop 2.6–41.0, λ 2.1–3.1.

### 3.3 The γ reference (no independent ground truth for γ)

γ is referenced against the **median FWHM at Trans100** (full collection, least-broadened
measurement of the line): $\gamma_{\text{true}} = \text{median FWHM @ Trans100} / 2$
(FWHM = 2γ). Values: **1 nW → 17.0/2 = 8.5 MHz; 3 nW → 28.3/2 = 14.1 MHz**. These match the
paper's quoted true linewidths (17 MHz and 29 MHz FWHM at 1 nW / 3 nW).

> Important subtlety: the paper's 17/29 MHz are **Voigt** FWHMs; our γ is the **Cauchy HWHM** of the
> simulated Lorentzian. They are not directly comparable beyond the reference convention above.

---

## 4. The project: from the paper to a differentiable Monte Carlo

Our goal: **recover (μ, γ) from the measured FWHM distribution** — the paper's task — but with
gradient-based optimization instead of grid search, plus honest uncertainties. The project arc:

1. **Forward pipeline** (May–Jun): simulate one scan → many scans → FWHM distribution. Loss
   evolved: histogram-χ² (as the paper) → sample-based MMD² → Wasserstein-1.
2. **γ differentiable** (28 Jun): reparameterized Cauchy sampling (frozen noise) + **implicit
   differentiation through the L-BFGS fit** (no optimizer unrolling). γ recovered end-to-end.
3. **μ differentiable via REINFORCE** (Jul): policy gradient through the *discrete* photon count;
   EMA baseline; joint μ + γ (+ σ) optimization. First real-data validation (13-series).
4. **Uncertainty quantification** (Jul–Aug): settled on **Fisher information / Cramér–Rao bound**
   (Gregor endorsed this route) alongside bootstrap. The 2D KDE likelihood (12d) made the Fisher
   approach tractable.
5. **Systematic improvement** (Aug, 15–18 series): real-data sweep (15), closed-loop synthetic
   diagnostic (16), improvement playground (17: λ_mean anchor, σ_ref fix), failure analysis (18).
6. **Now:** the current best optimizer (17f) validated on real data (18b), and AG-HYPOPT (the
   agent-guided hyperparameter campaign) to tune it.

### Where the loss function came from (why it is what it is)

- The paper compares **histogram vs histogram** via χ². A histogram is non-differentiable.
- First differentiable attempt: **MMD²** (Gaussian kernel, median heuristic) — works, but needs a
  kernel bandwidth and is insensitive in the tails.
- Then **Wasserstein-1** (quantile matching) — no bandwidth, robust.
- Then **per-quantile W1 + σ-term + mean-term** (12c) — adding the fit-uncertainty σ breaks the
  μ–γ degeneracy; the mean term anchored the location.
- Finally (12d, current): a **2D kernel-density likelihood over (FWHM, σ)** — a proper likelihood
  (not a divergence), which makes the **Fisher information** machinery valid (Section 6) and is
  the foundation for the current REINFORCE + KDE-gradients optimizer (Section 5).
- Lesson learned along the way: the **mean-matching anchor term (λ_mean) is harmful** — it
  destabilizes γ recovery and was removed (17a–17d; γ RMSE 1.20 → 4.48 as λ_mean grew).

---

## 5. The model in detail (the current pipeline)

All code lives in `src/` (samplers.py, fitting.py, implicit.py, losses.py) and is exercised by the
17f/18b notebooks. The pipeline is: **generate → fit → compare → update**, repeated.

### 5.1 The generator (forward model) — `src/samplers.py`

One simulated "scan" (one FWHM value):

1. **Photon count.** Draw the *number of signal photons* from a Gaussian rounded to an integer:
   $$n = \max(\lfloor \mu + \sigma_{\text{prop}}\, \varepsilon \rceil,\ 0),\qquad
     \varepsilon \sim \mathcal{N}(0,1).$$
   μ = mean photon number; σ_prop = per-experiment proportional noise (fixed).
2. **Signal positions.** $n$ photon detunings from the **Cauchy (Lorentzian) line at scale γ**,
   sampled via the inverse-CDF (quantile) transform with *frozen* uniforms u:
   $$x = \gamma \cdot \tan\!\big(\arctan(75/\gamma)\,(2u - 1)\big),\qquad u \sim \mathcal{U}(0,1),$$
   which is the Cauchy **truncated to the ±75 MHz detection window** (window width 150 MHz, as in
   the paper's supplemental). Truncation matters: clipping the raw Cauchy instead piles ~17 % of
   the mass at the window edges and makes the fitted FWHM *non-monotonic* in γ (a real bug found
   and fixed).
3. **Background.** $m \sim \text{Poisson}(\lambda)$ detunings drawn uniformly in the window
   (flat off-resonance counts, matching the paper).
4. **Photons = signal ∪ background** — typically a few to a few hundred detunings.

### 5.2 The fit (estimator) — `src/fitting.py`

Each scan's photon set is fitted with a **Lorentzian by maximum likelihood** (L-BFGS, up to 80
iterations, strong-Wolfe line search): parameters θ = (center, raw_γ), with the width mapped
$\gamma = \varepsilon_w + 150\,\text{sigmoid}(\text{raw}_\gamma)$ (so γ ∈ (0.01, 150.01) MHz;
$\varepsilon_w = 10^{-2}$), initialized from the photon median (center) and a 15 MHz width.
No uniform-background term (`uniform_bg=False`) — the background is in the *generator*, the fitter
just fits the line (the paper's estimator used a binned Voigt least-squares; ours is an unbinned
Lorentzian MLE — a documented, deliberate difference). Output per scan:

- **FWHM = 2γ** (exact for a Lorentzian),
- plus, via implicit differentiation (below): σ_fit, ∂FWHM/∂γ, ∂σ/∂γ.

**Fallback:** if fewer than 3 photons or the fit fails, return FWHM = 2γ with
σ = 0.5γ, ∂FWHM/∂γ = 2, ∂σ/∂γ = 0.5 (the analytic values of the fallback).

### 5.3 Differentiating through the fit — `src/implicit.py`

We need gradients w.r.t. γ even though the fit is an iterative optimizer. **Implicit function
theorem** (no unrolling): at the optimum θ*(γ),
$$\frac{d\theta^*}{d\gamma} = -H_{\theta\theta}^{-1}\, H_{\theta\gamma},$$
where $H_{\theta\theta}$ is the NLL Hessian at θ* (Tikhonov-regularized) and $H_{\theta\gamma}$ is
the mixed derivative (central finite difference in γ, step 1e-3). Then
$$\frac{d\text{FWHM}}{d\gamma} = \frac{\partial F}{\partial\theta}\cdot\frac{d\theta^*}{d\gamma},
\qquad
\sigma_{\text{fit}} = \sqrt{\frac{1}{N}\, \frac{\partial F}{\partial\theta}^\top
H_{\theta\theta}^{-1} \frac{\partial F}{\partial\theta}}$$
(the CRLB-style per-scan fit uncertainty), and dσ/dγ via a two-channel exact derivative
(θ*-movement + direct data-dependence through the Hessian). This replaced an earlier crude
approximation $d\sigma/d\gamma \approx 2/\sqrt n$.

### 5.4 The likelihood — 2D KDE over (FWHM, σ) — `kde_scores` (12d/17f)

Each simulated scan produces a point (FWHM, σ_fit). The simulated cloud is turned into a smooth
density with a **2D Gaussian kernel** (bandwidths from **Scott's rule**: $H = \sigma_{\text{data}}
\cdot N^{-1/6}$, with an absolute floor $H_S^{\min} = 0.05$ MHz on the σ-kernel), and the
**negative log-likelihood of the measured (FWHM, σ) points** under that density is the objective:
$$W_{ij} = \exp\!\Big(-\tfrac12 (d^F_{ij}/H_F)^2 - \tfrac12 (d^S_{ij}/H_S)^2\Big),\qquad
w_{ij} = W_{ij}/\textstyle\sum_j W_{ij},\qquad
\text{NLL} = -\frac{1}{N_{\text{tgt}}}\sum_i \log\!\Big(\frac{1}{N_{\text{sim}}}\sum_j W_{ij}\Big).$$
The per-point normalized weights $w_{ij}$ ("how much simulated scan $j$ explains measured point
$i$") are the engine of both parameter updates and the Fisher matrix.

### 5.5 The optimizer — joint (μ, γ) with two different gradient mechanisms

- **μ — REINFORCE (policy gradient).** The photon count n is discrete (rounding), so no analytic
  gradient: use the score-function estimator. With the kernel weights, the μ-score is
  $s_\mu = (n - \mu)/\sigma_{\text{ref}}^2$ with a **fixed reference scale**
  $\sigma_{\text{ref}} = 10$ (see 5.6), and the update is
  $$\mu \leftarrow \mu - \text{LR}_\mu(t)\,\big(\bar B - \bar{\bar B}\big)\cdot s_\mu,\qquad
    \bar B = \textstyle\sum_i w_{ij}\ \text{(per-scan "data mass")},$$
  clipped to [−10, 10], with $\text{LR}_\mu(t) = 15\,(1 - t/N)$ (linear decay, **no floor**).
  μ is bounded to [1, 200] photons.
- **γ — KDE log-density gradient.** γ enters everything smoothly (reparameterization + implicit
  diff), so its gradient comes from the derivative of the kernel log-density:
  $$s_\gamma = \frac{d^F\, \partial\text{FWHM}/\partial\gamma}{H_F^2}
              + \frac{d^S\, \partial\sigma/\partial\gamma}{H_S^2}, \qquad
    \gamma \leftarrow \gamma - \text{LR}_\gamma(t)\,\overline{s_\gamma},$$
  with $\text{LR}_\gamma(t) = 0.5\,(1 - 0.5\,t/N)$ (anneal), clipped, γ ∈ [0.1, 100] MHz.

**Determinism:** all randomness is seeded per step (SEED = 42) and for target generation
(SYNTH_SEED = 12345) — a given configuration reproduces exactly, which makes trials comparable.

### 5.6 Why the design choices exist (the reasoning trail)

- **σ_ref normalization (scale invariance).** Dividing the μ-score by the *per-experiment*
  $\sigma_{\text{prop}}^2$ starves high-transmission experiments: at μ ≈ 176, σ_prop² ≈ 1680, so
  each step moves μ by ~0.1–0.5 photons and the optimizer needs ~500+ iterations to climb the
  remaining distance (**starvation**, Section 7.1). A fixed σ_ref makes the step size identical in
  photon units across all experiments — high-μ experiments travel fast, low-μ ones don't overshoot.
- **The attractor.** The REINFORCE μ-update drives μ toward the *kernel-weighted mean photon count*
  of the scans the data actually matches: $E_B[n] = \bar n$. If the likelihood is unbiased this
  equals μ_true; a biased likelihood shows up as a *clean, confident convergence to the wrong μ*.
- **Why the σ-channel exists.** Without matching σ_fit, the μ–γ degeneracy (2.4) makes the problem
  ill-posed; the 2D KDE is what breaks it.
- **λ_mean = 0.** A mean-FWHM matching term was tried (12c) and later shown to be pure harm
  (17a–17d): it destabilizes γ. Disabled.
- **The γ anneal.** Intended to settle noisy low-count γ estimates; known side effect: it can
  *freeze* γ in the low-count regime (n_target ≲ 500) before convergence (Section 7.3).
- **Truncation, not clipping.** Physically, out-of-window photons aren't detected; mathematically,
  clipping breaks monotonicity of FWHM in γ. Truncated-Cauchy quantiles fix both.
- **Lorentzian fit, not pseudo-Voigt.** The generator emits a pure Cauchy, so the Lorentzian is the
  matched estimator (well-conditioned 2-param Hessian, FWHM = 2γ exact). Real lines may carry
  Gaussian broadening (spectral diffusion) — see Section 7.5.

---

## 6. Uncertainty quantification (Fisher / CRB)

At the recovered (μ̂, γ̂), the **Fisher information matrix** is estimated from the score functions:
$$J = \frac{1}{N_{\text{tgt}}} \sum_i s(x_i)\, s(x_i)^\top,\qquad s = [s_\mu, s_\gamma],$$
with $s_\mu$ in the σ_prop²-normalized form (the likelihood's own score), computed from M_FINAL
simulated scans at the fitted point (M = 500, optionally averaged over FISHER_SEEDS). The
**Cramér–Rao bound** gives parameter uncertainties:
$$\sigma_\mu = \sqrt{(J^{-1})_{00}},\quad \sigma_\gamma = \sqrt{(J^{-1})_{11}},\quad
\text{corr} = (J^{-1})_{01}/\sqrt{(J^{-1})_{00}(J^{-1})_{11}}.$$
This is a *lower bound* on the statistical uncertainty (Gregor endorsed this route; it is what the
15-series reported and what 12d introduced). Known pathologies: near-singular J (very wide
likelihood, e.g. σ_μ ≈ 1300 seen once) and over-confident J when the optimizer sits in a wrong
basin (σ_γ ≈ 0.09 at half-truth γ) — both are signatures of a broken fit, not trustworthy bounds.

---

## 7. Known failure modes and physics insights (durable lessons)

Stable knowledge about the model — the physics intuition to apply when judging configurations or
interpreting results.

1. **Optimizer starvation.** μ-score divided by σ_prop² → tiny steps at high μ; the optimizer stops
   hundreds of iterations short of the NLL minimum (μ stuck at ~½–⅔ of truth while the NLL minimum
   sits at/above μ_true). *Fix: fixed σ_ref normalization.* Signature: NLL still decreasing at the
   end of the run.
2. **The attractor trap.** μ converges to the kernel-weighted mean of the scans the data explains
   (E_B[n] = n̄). A biased likelihood → clean convergence to the wrong μ. Always compare the
   attractor against the NLL minimum.
3. **Low-count γ noise.** When n_target ≲ 500 (Trans05 experiments), the γ-gradient is too noisy for
   the nominal learning rate: γ explodes or collapses near its initialization. The anneal worsens
   this (freezes γ); more runs or a per-experiment γ LR is the natural counter.
4. **KDE skewness / bandwidth inflation.** A pathologically skewed target cloud (one experiment:
   skewness +15.9 F / +36.2 σ) inflates Scott's bandwidth, flattens the likelihood, and degrades the
   NLL as a discriminator. A likelihood-model issue, not an optimizer issue.
5. **The high-transmission model gap (16-series).** Even in closed-loop (the model generates its own
   targets at the true values), high transmission (60–100 %) fails: γ → ≈½ γ_true, μ → 50–60 % of
   truth, occasionally with pathological Fisher (near-singular or over-confident). At high T, σ_prop
   is large (12–41), smearing the FWHM distribution and flattening the likelihood along the μ–γ
   degeneracy. This is a *model/optimization* property, not data mismatch — the 17-series fixes
   (σ_ref) addressed the optimizer part.
6. **The Voigt-vs-Lorentzian spread gap (13-series).** On real data, the mean FWHM is matched
   (92–96 %), but the *spread* is stuck at ~40–50 % of the target regardless of hyperparameters.
   Cause: real PLE scans are fitted with **Voigt** profiles (Gaussian broadening from spectral
   diffusion); our simulator emits a pure Lorentzian and cannot produce the observed spread. A
   simulator change (pseudo-Voigt line) is the candidate fix — relevant when real-data fitting is
   the goal.
7. **Open question — real-data attractor (18b, preliminary).** On real data with the fixed
   optimizer, μ repeatedly lands at ~**half** the calibration reference μ. Two explanations:
   (a) the real-data likelihood genuinely prefers a lower μ (model-data mismatch — the real FWHM
   cloud is much broader than the model produces at the reference μ), or (b) the reference itself
   is off. The Fisher width at the fitted point discriminates. *Active investigation — resolution
   belongs to the experiment's trial_00.*
8. **μ is weakly identified.** Across the 15-series, μ recovered at all transmissions with
   σ_μ ~ 20–90 (50–60 % relative at 3 nW) — the data constrain γ far better than μ. σ_γ collapses
   1–2 orders of magnitude from Trans05 → Trans20 and plateaus (~1–2 MHz); σ_μ shows no clean
   trend. Expect the objective's μ-part to be the noisy one.

---

## 8. Units and conventions

- **Frequency units:** MHz. (Absolute physical unit of the raw linewidths never recorded.)
- **FWHM vs γ:** the fitted quantity is the FWHM of a Lorentzian; the model parameter γ is the
  Cauchy scale, **FWHM = 2γ**. References are stored as γ (half-FWHM).
- **Window:** ±75 MHz (150 MHz wide), matching the paper's supplemental.
- **Data filtering:** finite values, FWHM > 0, `err/fwhm < 10`.
- **Scott exponent:** N^{-1/6} for the 2D KDE.
- **Width mapping:** γ = 0.01 + 150·sigmoid(raw) — the fitted width is bounded in (0.01, 150.01) MHz.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| μ | mean photon number per scan (primary recovery target) |
| γ | Cauchy/Lorentzian scale = HWHM (MHz); FWHM = 2γ |
| σ_prop | proportional photon-count noise, fixed per experiment |
| λ | mean of additive Poisson background (fixed per experiment) |
| n_target | number of (filtered) measured FWHM samples in an experiment |
| FWHM | full width at half maximum of the fitted line |
| HWHM | half width at half maximum (= γ) |
| PLE | photoluminescence excitation spectroscopy |
| NV | nitrogen-vacancy color center in diamond |
| SIL | solid immersion lens |
| TransXX | transmission setting (5 %–100 %), the SNR knob |
| KDE | kernel density estimate — our likelihood |
| REINFORCE | score-function (policy) gradient estimator used for μ |
| NLL | negative log-likelihood |
| attractor | the μ value the REINFORCE update converges to: E_B[n] = n̄ |
| Scott bandwidth | data-driven KDE width: σ_data · N^{-1/6} |
| Voigt | convolution of Lorentzian (line) and Gaussian (instrumental/spectral diffusion) |
| CRB | Cramér–Rao bound — parameter uncertainty from the inverse Fisher matrix |
| MCM | Monte Carlo method (the paper's approach) |
