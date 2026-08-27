# Differentiable MC for NV Center Spectroscopy

**Gradient-based parameter estimation for optical quantum systems — replacing grid search with differentiable Monte Carlo.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.07951-b31b1b.svg)](https://arxiv.org/abs/2501.07951)

---

## Description

NV centers in diamond are a central platform for quantum sensing and photonics. In photoluminescence excitation (PLE) spectroscopy, a resonant laser is scanned across the absorption line of a single NV center and the fluorescence is collected. The extracted linewidth (FWHM) distribution is shaped by the physical parameters of the system: the mean photon count μ, the Lorentzian half-width γ, and the noise parameters (σ, λ). Recovering these parameters from noisy measurements is the problem this project solves.

The baseline method — *Monte Carlo-based Parameter Reconstruction of an Optical Quantum System* (Pieplow et al., [arXiv:2501.07951](https://arxiv.org/abs/2501.07951)) — simulates the full measurement with a Monte Carlo pipeline and reconstructs the parameters by brute-force **grid search** over (μ, γ). Grid search is expensive and scales poorly with the number of parameters.

This project replaces grid search with **gradient descent** by making the MC pipeline differentiable end-to-end in PyTorch:

- **μ** (mean photon count) is optimized with the **REINFORCE** gradient estimator through the discrete photon-counting step.
- **γ** (Lorentzian HWHM) is optimized with **implicit differentiation** through the per-run L-BFGS fit.
- Matching both the FWHM distribution *and* its uncertainty σ breaks the μ/γ degeneracy.
- The **Fisher information / Cramér-Rao bound** provides parameter uncertainties.

The approach is validated on synthetic data with known ground truth and applied to real experimental PLE data from Dr. Gregor Pieplow (AG Schröder, Humboldt-Universität zu Berlin).

---

## Structure

```
├── src/                 # Core differentiable modules
│   ├── samplers.py      # Reparameterized & truncated Cauchy, normal, Poisson sampling
│   ├── fitting.py       # Pseudo-Voigt / Lorentzian fitting (differentiable log-pdf, L-BFGS)
│   ├── losses.py        # MMD², L2, Wasserstein-1 distribution losses
│   ├── implicit.py      # Implicit differentiation through the L-BFGS fit
│   └── utils.py         # Data loading helpers
├── notebooks/           # Numbered chapters (01 → 16), described below
├── data/
│   ├── raw_data/        # Experimental PLE measurements (1 nW & 3 nW) + data description
│   └── processed/       # Preprocessed linewidth table + bootstrap results
├── docs/                # Reference papers
├── notes/               # Project journal (JOURNAL.md) and presentations
├── scripts/             # Data preprocessing
├── archive/             # Deprecated notebooks
└── requirements.txt
```

---

## Notebooks

The notebooks are organized as numbered chapters. When several notebooks share a number they are variants or experiments of the same idea, so each number gets a single description.

### 01 — MC algorithm
`notebooks/01-mc-algorithm.ipynb`

Walks through the paper's Monte Carlo simulation step by step: the terminology (run, simulation, MC distribution) and the forward pipeline — noiseless PLE spectrum, photon noise, Lorentzian fit → one extracted linewidth per run. Sets up the goal of converting the MC simulation into a backpropagatable PyTorch pipeline.

### 02 — Making sampling differentiable
`notebooks/02-sampling-toy.ipynb`

The first obstacle: sampling the photon count n ~ N(μ, σ) and rounding it to an integer is non-differentiable (stochastic node + step function). Introduces the reparameterization trick on a simple example, building the tools to make the sampling step gradient-friendly.

### 03 — Implicit differentiation through a fit
`notebooks/03-fitting-toy.ipynb`, `notebooks/03-fitting-toy-fwmh.ipynb`

The per-run fit (L-BFGS) is an iterative optimization that blocks gradients. Differentiates *around* the fit with the implicit function theorem — no unrolling of the optimizer — and pushes the outer loss through the fitted pseudo-Voigt, for both the raw γ and the FWHM of the fitted distribution.

### 04 — EDA
`notebooks/04-eda.ipynb`

Exploratory analysis of the experimental linewidth data from Dr. Pieplow: failed-fit (NaN) rates vs. transmission/power, FWHM distributions, the power-broadening trend, and the fit-error quality metric.

### 05 — Loss functions
`notebooks/05-loss-mmd.ipynb`, `notebooks/05-loss-w1.ipynb`

Distribution-matching losses for comparing simulated and experimental FWHM sets: MMD² with a Gaussian kernel (bandwidth tuning, median heuristic) and the parameter-free 1D Wasserstein-1 distance on sorted samples.

### 06 — γ differentiable end-to-end
`notebooks/06-gamma.ipynb`, `notebooks/06-gamma-simple.ipynb`

First full differentiability of the MC pipeline for the linewidth γ: reparameterized Cauchy sampling (γ·tan(π(u−0.5))) plus implicit differentiation through the per-run fit, so `loss.backward()` populates `gamma.grad`. Validated: γ = 20 → recovers ~20.5.

### 07 — REINFORCE for μ
`notebooks/07-reinforce-toy.ipynb`

Optimizes the mean photon count μ through the discrete rounding step with the REINFORCE gradient estimator (policy gradient) and an exponential-moving-average baseline, on a Wasserstein-1 loss.

### 08 — Per-run REINFORCE
`notebooks/08-reinforce-per-run.ipynb`

Refines REINFORCE to per-quantile losses: sorting pairs low-n runs (narrow FWHM) with low target quantiles, giving a structured, directional gradient per quantile. An empirical ρ(n, loss) test shows the signal is strong and correctly directed when μ is below the truth, and decays near it.

### 09 — (μ, σ) → FWHM map
`notebooks/09-mu-sigma-fwhm-map.ipynb`

Grid sweep over the photon-count proposal distribution (μ, σ) to build intuition for how it shapes the resulting FWHM distribution, and where the REINFORCE gradient has signal.

### 10 — Joint optimization
`notebooks/10-joint-optimization.ipynb`

Combines the two gradient estimators into one joint optimizer: REINFORCE for μ through the discrete step, implicit differentiation for γ through the fit, and a CRLB-based dσ/dγ.

### 11 — FWHM pairplot over (μ, γ)
`notebooks/11-fwhm-pairplot.ipynb`

Pairplot-style grid showing how the FWHM distribution changes as a function of both μ and γ — a visual illustration of the μ/γ degeneracy (many pairs give the same FWHM distribution).

### 12 — Joint optimization with σ matching (12a–12d)
`notebooks/12a-joint-opt-with-sigma-noise.ipynb`, `notebooks/12b-joint-opt-with-sigma-lr-decay.ipynb`, `notebooks/12c-joint-opt-with-sigma-mean-fwhm.ipynb`, `notebooks/12d-joint-opt-likelihood-fisher.ipynb`

Matching the FWHM distribution alone is degenerate: different (μ, γ) pairs produce the same FWHM. Matching FWHM *and* its fit uncertainty σ breaks the degeneracy. 12a adds σ matching; 12b adds learning-rate decay; 12c adds a mean-matching term and is the state of the art on synthetic data (μ 8 → 48.45, true 50; γ 5 → 19.62, true 20); 12d replaces the W₁/mean losses with a 2D KDE negative log-likelihood, turning the optimization into an MLE and making the Fisher information / Cramér-Rao bound applicable as the uncertainty statement.

### 13 — Real data: first attempt and diagnosis (13a–13f)
`notebooks/13-real-data-optimization-executed.ipynb`, `notebooks/13-real-data-optimization-output.ipynb`, `notebooks/13a-diagnose.ipynb`, `notebooks/13b-fix-quantile-bug.ipynb`, `notebooks/13b-quantile-fix.ipynb`, `notebooks/13b-quantile-fix-executed.ipynb`, `notebooks/13c-faster-gamma.ipynb`, `notebooks/13c-more-iterations.ipynb`, `notebooks/13c-reduce-sigma-weight.ipynb`, `notebooks/13c-reduce-sigma-weight-executed.ipynb`, `notebooks/13d-higher-gamma-lr.ipynb`, `notebooks/13d-higher-gamma-lr-executed.ipynb`, `notebooks/13e-moderate-gamma-lr.ipynb`, `notebooks/13e-moderate-gamma-lr-executed.ipynb`, `notebooks/13f-higher-background-noise.ipynb`, `notebooks/13f-higher-background-noise-executed.ipynb`

First application of the joint optimization to real PLE data (3 nW, 40% transmission), then a sequence of experiments to diagnose and tune the optimization: fixing a quantile bug, faster γ, more iterations, reduced σ weight, higher/moderate γ learning rate, higher background noise. Best result: 13e, W₁ = 2.28. Key finding: the FWHM spread is stuck at ~40–50% of the target regardless of hyperparameters → a model gap (real lines are fitted with a Voigt, our simulator is Lorentzian-only).

### 14 — Uncertainties by bootstrap
`notebooks/14-uncertainties-dummy.ipynb`

First uncertainty experiment: bootstrap copies of the target data (resample with replacement), re-run the full 12c optimization on each copy, and use the spread of the recovered (μ, γ) as the statistical uncertainty.

### 15 — Real-data sweep (15a–15n)
`notebooks/15a-real-data-1nW-trans05.ipynb` … `notebooks/15n-real-data-3nW-trans100.ipynb` (14 files: 2 powers × 7 transmissions)

Applies the 12d model (2D KDE likelihood + Fisher information) to all 14 real experiments, reporting Cramér-Rao uncertainties per experiment. σ_γ decreases strongly with transmission (at 1 nW from ~72 to ~0.9 MHz), σ_μ stays weakly identified, and recovered μ runs biased low at high transmission.

### 16 — Closed-loop synthetic diagnostic (16a–16n)
`notebooks/16a-synthetic-1nw-trans05.ipynb` … `notebooks/16n-synthetic-3nw-trans100.ipynb` (14 files: 2 powers × 7 transmissions)

The same pipeline as the 15-series, but the target FWHM distribution is generated by our own simulator at the true values. The model recovers the truth at low/mid transmission (most within ~1σ) but systematically fails at high transmission (γ and μ biased low, pathological Fisher signatures) — showing the high-transmission failure is at least partly a model/optimization property, not only a data mismatch.

---

## Current state and next steps

The differentiable MC pipeline is complete and validated on synthetic data: 12c recovers (μ, γ) = (48.45, 19.62) against a truth of (50, 20), and 12d reformulates the optimization as an MLE whose Fisher information yields Cramér-Rao uncertainties. Applied to all 14 real experiments (15-series), the method recovers parameters with uncertainties that shrink with transmission, though μ remains weakly identified and biased low at high transmission. The closed-loop 16-series confirms the model is healthy at low/mid transmission but fails at high transmission even on its own data. The next steps are to fix the high-transmission failure (12c-style mean-matching anchor, more iterations/restarts, stronger γ gradient), close the Voigt-vs-Lorentzian model gap to match the real FWHM spread, and then fine-tune the optimization on the real data.

---

## Citation & licence

```bibtex
@misc{santoyo2026quantumml,
  author = {Santoyo Alum, Anuar and Pieplow, Gregor and Bänz, Clara},
  title = {Differentiable Monte Carlo for Parameter Estimation in NV Center Spectroscopy},
  year = {2026},
  howpublished = {\url{https://github.com/anuarsantoyo/quantum-ml}}
}
```

This project is released under the [MIT license](LICENSE).
