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
├── notebooks/           # Numbered chapters (01 → 24), grouped in ordered folders — see below
├── agent-guided-hypopt/ # Agent-guided hyperparameter optimization campaigns (experiment_N)
├── data/
│   ├── raw_data/        # Experimental PLE measurements (1 nW & 3 nW) + data description
│   └── processed/       # Preprocessed linewidth table + bootstrap results
├── docs/                # Reference papers
├── notes/               # Project journal (JOURNAL.md) and presentations
├── scripts/             # Data preprocessing
├── archive/             # Deprecated / superseded notebooks (incl. legacy -executed twins)
└── requirements.txt
```

---

## Notebooks

The notebooks are organized as numbered chapters, kept in **ordered folders that follow the numbering** so the directory listing reads in the same order as the sequence: `01-06_foundations/`, `07-11_methods/`, `12a-12d_joint-opt/`, `13-14_real-data-diag/`, `15-16_sweeps/`, `17_playground/`, `18_failure-analysis/`, `19_identifiability/`, `20_realdata_agentic_optimization/`, `21_realdata_supervised_optimization/`, `22_distribution_dynamics/`, `23_fisher_uncertainties/`, `24_realdata_accuracy/`. Superseded copies (legacy `-executed`/`-output` twins) live under `archive/notebooks/`. When several notebooks share a number they are variants or experiments of the same idea, so each number gets a single description.

### 01 — MC algorithm
`notebooks/01-06_foundations/01-mc-algorithm.ipynb`

Walks through the paper's Monte Carlo simulation step by step: the terminology (run, simulation, MC distribution) and the forward pipeline — noiseless PLE spectrum, photon noise, Lorentzian fit → one extracted linewidth per run. Sets up the goal of converting the MC simulation into a backpropagatable PyTorch pipeline.

### 02 — Making sampling differentiable
`notebooks/01-06_foundations/02-sampling-toy.ipynb`

The first obstacle: sampling the photon count n ~ N(μ, σ) and rounding it to an integer is non-differentiable (stochastic node + step function). Introduces the reparameterization trick on a simple example, building the tools to make the sampling step gradient-friendly.

### 03 — Implicit differentiation through a fit
`notebooks/01-06_foundations/03-fitting-toy.ipynb`, `notebooks/01-06_foundations/03-fitting-toy-fwmh.ipynb`

The per-run fit (L-BFGS) is an iterative optimization that blocks gradients. Differentiates *around* the fit with the implicit function theorem — no unrolling of the optimizer — and pushes the outer loss through the fitted pseudo-Voigt, for both the raw γ and the FWHM of the fitted distribution.

### 04 — EDA
`notebooks/01-06_foundations/04-eda.ipynb`

Exploratory analysis of the experimental linewidth data from Dr. Pieplow: failed-fit (NaN) rates vs. transmission/power, FWHM distributions, the power-broadening trend, and the fit-error quality metric.

### 05 — Loss functions
`notebooks/01-06_foundations/05-loss-mmd.ipynb`, `notebooks/01-06_foundations/05-loss-w1.ipynb`

Distribution-matching losses for comparing simulated and experimental FWHM sets: MMD² with a Gaussian kernel (bandwidth tuning, median heuristic) and the parameter-free 1D Wasserstein-1 distance on sorted samples.

### 06 — γ differentiable end-to-end
`notebooks/01-06_foundations/06-gamma.ipynb`, `notebooks/01-06_foundations/06-gamma-simple.ipynb`

First full differentiability of the MC pipeline for the linewidth γ: reparameterized Cauchy sampling (γ·tan(π(u−0.5))) plus implicit differentiation through the per-run fit, so `loss.backward()` populates `gamma.grad`. Validated: γ = 20 → recovers ~20.5.

### 07 — REINFORCE for μ
`notebooks/07-11_methods/07-reinforce-toy.ipynb`

Optimizes the mean photon count μ through the discrete rounding step with the REINFORCE gradient estimator (policy gradient) and an exponential-moving-average baseline, on a Wasserstein-1 loss.

### 08 — Per-run REINFORCE
`notebooks/07-11_methods/08-reinforce-per-run.ipynb`

Refines REINFORCE to per-quantile losses: sorting pairs low-n runs (narrow FWHM) with low target quantiles, giving a structured, directional gradient per quantile. An empirical ρ(n, loss) test shows the signal is strong and correctly directed when μ is below the truth, and decays near it.

### 09 — (μ, σ) → FWHM map
`notebooks/07-11_methods/09-mu-sigma-fwhm-map.ipynb`

Grid sweep over the photon-count proposal distribution (μ, σ) to build intuition for how it shapes the resulting FWHM distribution, and where the REINFORCE gradient has signal.

### 10 — Joint optimization
`notebooks/07-11_methods/10-joint-optimization.ipynb`

Combines the two gradient estimators into one joint optimizer: REINFORCE for μ through the discrete step, implicit differentiation for γ through the fit, and a CRLB-based dσ/dγ.

### 11 — FWHM pairplot over (μ, γ)
`notebooks/07-11_methods/11-fwhm-pairplot.ipynb`

Pairplot-style grid showing how the FWHM distribution changes as a function of both μ and γ — a visual illustration of the μ/γ degeneracy (many pairs give the same FWHM distribution).

### 12 — Joint optimization with σ matching (12a–12d)
`notebooks/12a-12d_joint-opt/12a-joint-opt-with-sigma-noise.ipynb`, `notebooks/12a-12d_joint-opt/12b-joint-opt-with-sigma-lr-decay.ipynb`, `notebooks/12a-12d_joint-opt/12c-joint-opt-with-sigma-mean-fwhm.ipynb`, `notebooks/12a-12d_joint-opt/12d-joint-opt-likelihood-fisher.ipynb`

Matching the FWHM distribution alone is degenerate: different (μ, γ) pairs produce the same FWHM. Matching FWHM *and* its fit uncertainty σ breaks the degeneracy. 12a adds σ matching; 12b adds learning-rate decay; 12c adds a mean-matching term and is the state of the art on synthetic data (μ 8 → 48.45, true 50; γ 5 → 19.62, true 20); 12d replaces the W₁/mean losses with a 2D KDE negative log-likelihood, turning the optimization into an MLE and making the Fisher information / Cramér-Rao bound applicable as the uncertainty statement.

### 13 — Real data: first attempt and diagnosis (13a–13f)
`notebooks/13-14_real-data-diag/13a-diagnose.ipynb`, `notebooks/13-14_real-data-diag/13b-fix-quantile-bug.ipynb`, `notebooks/13-14_real-data-diag/13b-quantile-fix.ipynb`, `notebooks/13-14_real-data-diag/13c-faster-gamma.ipynb`, `notebooks/13-14_real-data-diag/13c-more-iterations.ipynb`, `notebooks/13-14_real-data-diag/13c-reduce-sigma-weight.ipynb`, `notebooks/13-14_real-data-diag/13d-higher-gamma-lr.ipynb`, `notebooks/13-14_real-data-diag/13e-moderate-gamma-lr.ipynb`, `notebooks/13-14_real-data-diag/13f-higher-background-noise.ipynb` — superseded `-executed`/`-output` copies are archived under `archive/notebooks/`.

First application of the joint optimization to real PLE data (3 nW, 40% transmission), then a sequence of experiments to diagnose and tune the optimization: fixing a quantile bug, faster γ, more iterations, reduced σ weight, higher/moderate γ learning rate, higher background noise. Best result: 13e, W₁ = 2.28. Key finding: the FWHM spread is stuck at ~40–50% of the target regardless of hyperparameters → a model gap (real lines are fitted with a Voigt, our simulator is Lorentzian-only).

### 14 — Uncertainties by bootstrap
`notebooks/13-14_real-data-diag/14-uncertainties-dummy.ipynb`

First uncertainty experiment: bootstrap copies of the target data (resample with replacement), re-run the full 12c optimization on each copy, and use the spread of the recovered (μ, γ) as the statistical uncertainty.

### 15 — Real-data sweep (15a–15n)
`notebooks/15-16_sweeps/15a-real-data-1nW-trans05.ipynb` … `notebooks/15-16_sweeps/15n-real-data-3nW-trans100.ipynb` (14 files: 2 powers × 7 transmissions)

Applies the 12d model (2D KDE likelihood + Fisher information) to all 14 real experiments, reporting Cramér-Rao uncertainties per experiment. σ_γ decreases strongly with transmission (at 1 nW from ~72 to ~0.9 MHz), σ_μ stays weakly identified, and recovered μ runs biased low at high transmission.

### 16 — Closed-loop synthetic diagnostic (16a–16n)
`notebooks/15-16_sweeps/16a-synthetic-1nw-trans05.ipynb` … `notebooks/15-16_sweeps/16n-synthetic-3nw-trans100.ipynb` (14 files: 2 powers × 7 transmissions)

The same pipeline as the 15-series, but the target FWHM distribution is generated by our own simulator at the true values. The model recovers the truth at low/mid transmission (most within ~1σ) but systematically fails at high transmission (γ and μ biased low, pathological Fisher signatures) — showing the high-transmission failure is at least partly a model/optimization property, not only a data mismatch.

### 17 — Optimizer playground: λ₀ anchor / scale-free variants (17a–17h)
`notebooks/17_playground/17-improvements-playground.ipynb`, `notebooks/17_playground/17b-lambda0.ipynb`, `notebooks/17_playground/17c-lambda0-gamma-half.ipynb`, `notebooks/17_playground/17d-lambda01.ipynb`, `notebooks/17_playground/17e-lambda0-fixmu.ipynb`, `notebooks/17_playground/17f-lambda0-scalefree.ipynb`, `notebooks/17_playground/17g-lambda0-gammascale.ipynb`, `notebooks/17_playground/17h-lambda0-gammascale-fisher.ipynb`

Exploratory series on the optimizer's anchor/regularization term (λ₀) and scale-free reformulations, ending with 17h, which adds the Fisher-information view later reused in series 19.

### 18 — Failure analysis & real-data sanity (18, 18b, 18c)
`notebooks/18_failure-analysis/18-failure-analysis.ipynb`, `notebooks/18_failure-analysis/18b-real-data-sanity.ipynb`, `notebooks/18_failure-analysis/18c-real-data-gammascale.ipynb`

Diagnosis of the high-transmission failure and sanity checks of the real-data pipeline; 18c applies the γ-scale approach to the real data (γ RMSE ≈ 3.54 MHz, the baseline the 19-series improves on).

### 19 — μ identifiability on real data (19a–19d)
`notebooks/19_identifiability/19a-voigt-target-sweep.ipynb`, `notebooks/19_identifiability/19b-real-mu-identifiability.ipynb`, `notebooks/19_identifiability/19c-sigma-channel-calibration.ipynb`, `notebooks/19_identifiability/19d-honest-wide-report.ipynb`

Series 19 studies μ identifiability on the real Lorentzian-FWHM data: 19a sweeps a Voigt target, 19b probes μ identifiability on the real data, 19c calibrates the σ channel, and **19d is the final report**. Under the honest-wide (D1) configuration the true μ lies within 2σ in 14/14 real experiments (max |Δμ|/σ = 1.87, mean 0.88; 10/14 within 1σ), with a μ attractor at 0.43–0.52× the true value absorbed by the widened posterior.

### 20 — Real-data agentic optimization (20a…, in progress)
`notebooks/20_realdata_agentic_optimization/20a.ipynb` (+ `SERIES_STATE.md`)

Autonomous series: start from the **17g** model (best synthetic performer) and apply it **unchanged** to the real data (**20a** baseline transfer test), then chain **one-change-per-notebook** experiments toward accurate (μ, γ) recovery on the real data, with accuracy graded by transmission (loose at low T, tight at high T). Frozen goal, protocol, metrics and stop rules live in the series' `SERIES_STATE.md`; budget 100 runs × 30 iters per experiment, cap 10 notebooks.

### 21 — Real-data supervised optimization (21a–21k)
`notebooks/21_realdata_supervised_optimization/21a.ipynb` … `21k.ipynb`

A supervised-optimization continuation of the real-data transfer effort: tuning the optimization toward accurate (μ, γ) recovery on the real data, one change per notebook. Exploratory.

### 22 — Distribution dynamics (22a–22g)
`notebooks/22_distribution_dynamics/22a-zscore-adaptive-lr.ipynb` … `22g-mc-at-truth-3-estimators.ipynb`

Studies how (μ, γ) recovery behaves across estimators and the loss landscape. 22g — Monte Carlo at the true values under three estimators — produced the σ calibration table that series 24 later reused.

### 23 — Fisher uncertainties (23a–23e)
`notebooks/23_fisher_uncertainties/23a-fisher-uncertainties-exp6-trial07.ipynb` … `23e-fisher-uncertainties-22a-perexp.ipynb`

Fisher / Cramér-Rao uncertainty analysis of the KDE model (experiment_6 trial_07 and the 22a variant) across the 14 real experiments. Conclusion: the residual is bias, not variance — the raw CRB is too wide and mis-centred, pointing toward the bias-corrected interval.

### 24 — Real-data accuracy campaign (24a–24ad, 30 notebooks)
`notebooks/24_realdata_accuracy/24a.ipynb` … `24ad.ipynb` (+ `SUMMARY.md`, `SERIES_STATE.md`)

The large accuracy campaign: five phases — μ travel control, the σ channel, uncertainty honesty, the forward model, and combination/verification — one named change per notebook. Baseline 24a = 0.0811 → best **24aa** = 0.0651 (confirmed by 24ab). Closed 2026-09-27 with **FM#6** (the Lorentzian line-shape gap) named as the open problem. Full ladder and final summary in `SUMMARY.md`.

### AG-HYPOPT — agent-guided hyperparameter optimization
`agent-guided-hypopt/experiment_N/` (one self-contained folder per experiment)

Agent-guided TPE hyperparameter optimization: the TPE proposes candidate configurations, the agent selects among them using physics context. Produced two canonical baselines — `experiment_3_pukky` (synthetic, objective 0.0149) and `experiment_6 trial_07` (the real-data KDE model that became the series-23/24 reference).

---

## Current status

*Updated 2026-11-05.*

**Best real-data model — 24aa** (E1 role-split): a σ-free FWHM kernel + count prior for μ, a calibrated-σ 2-D KDE for γ, with the bias-corrected report.
`W_obj(all14) = 0.0651 · T≥40 = 0.0341 · T≤20 = 0.1601 · μ 25.7 % · γ 33.3 %`, confirmed by the 3-seed repeat (24ab, `0.0644 ± 0.0026`).

**Open problem — FM#6**: the residual high-transmission μ bias and the low-transmission γ deficit are a line-shape gap — the real scans are fitted with Voigt profiles while the simulator is Lorentzian-only. Series 19 and 24 showed no optimizer change touches it; it needs a forward-model fix.

**Next — series 25** (roadmap in `notebooks/24_realdata_accuracy/SERIES_STATE.md`): fix the D2 σ-surrogate plumbing, attack FM#6 at the simulator level (pseudo-Voigt / per-scan heterogeneity / count model), report bias-corrected intervals by default, and use a random (not transmission-based) held-out split.

### Series index

| Series | What it established | Status | Led by |
|---|---|---|---|
| 01–06 | Pipeline foundations (reparameterization, implicit diff, KDE losses) | canonical | Anuar + agent |
| 07–11 | REINFORCE for μ, joint μ+γ | canonical | Anuar + agent |
| 12 | 2-D KDE likelihood + Fisher (12d = the current MLE) | 12d canonical; 12c superseded | Anuar |
| 13–14 | First real-data run; bootstrap uncertainty (superseded by Fisher) | instructive | Anuar |
| 15–16 | Real-data sweep + closed-loop synthetic | done | Anuar |
| 17 | Optimizer fixes (σ_ref, z-form γ-score) | canonical (17g) | Anuar + agent |
| 18 | Failure analysis; honest-wide μ | solid | agent |
| 19 | μ identifiability; FM#8 diagnosed | solid | agent |
| 20 | Real-data agentic optimization | exploratory | agent (auto) |
| 21 | Real-data supervised optimization | exploratory | agent |
| 22 | Distribution dynamics; 22g σ-calibration | exploratory (22g reused) | agent |
| 23 | Fisher uncertainties (CRB / sandwich / bootstrap) | solid | agent |
| 24 | Real-data accuracy campaign → 24aa best | 24aa = current best | agent (auto) |
| AG-HYPOPT | Agent-guided TPE; baselines exp3_pukky + exp6 trial_07 | canonical baselines | Anuar's idea, mixed |

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
