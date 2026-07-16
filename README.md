# Differentiable MC for NV Center Spectroscopy 🔬⚛️

**Gradient-based parameter estimation for optical quantum systems — replacing grid search with differentiable Monte Carlo.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.07951-b31b1b.svg)](https://arxiv.org/abs/2501.07951)

---

## What

We make the Monte Carlo simulation pipeline for PLE spectroscopy **differentiable end-to-end**, enabling gradient descent on physical parameters (μ, γ) instead of brute-force grid search.

**Application:** Reconstructing optical linewidths of NV centers in diamond from noisy PLE data.

---

## What we achieved

### μ optimization — REINFORCE
- Policy gradient from reinforcement learning estimates ∇μ through non-differentiable sampling
- Per-run advantage with exponential moving average baseline
- Converges reliably on both synthetic and real data

### γ optimization — Implicit Differentiation
- Differentiate through the L-BFGS Lorentzian fit via the implicit function theorem
- No backprop through optimizer iterations — only Hessian + mixed derivative at the optimum
- Provides dFWHM/dγ for each run

### Joint μ + γ optimization
- Naive joint optimization is degenerate: different (μ, γ) pairs produce the same FWHM
- **Solution:** match both the FWHM distribution AND its uncertainty (σ) — breaks the degeneracy
- μ uses REINFORCE, γ uses implicit diff + CRLB approximation for σ gradient

### Real data validation
- Successfully matched FWHM distribution of experimental NV center data (3 nW, 40% transmission)
- FWHM loss converged from 23.9 to **1.6**
- μ converged to 49.6 (synthetic true: 50.0)

---

## Project Structure

```
├── src/                 # Core modules
│   ├── samplers.py      # Reparameterized Cauchy, truncated, masked sampling
│   ├── fitting.py       # Pseudo-Voigt fitting with Z-factor normalization
│   ├── losses.py        # MMD², L2, Wasserstein loss functions
│   ├── implicit.py      # Implicit differentiation through L-BFGS fit
│   └── utils.py         # Data loading helpers
├── notebooks/           # Numbered chapters (01 → 13)
│   ├── 01-mc-algorithm.ipynb        # Paper's MC algorithm
│   ├── 02-sampling-toy.ipynb        # Differentiable sampling intro
│   ├── 03-fitting-toy.ipynb         # Implicit differentiation intro
│   ├── 04-eda.ipynb                 # Experimental data exploration
│   ├── 05-loss-mmd.ipynb            # MMD² loss tests
│   ├── 06-gamma.ipynb               # Gamma optimization
│   ├── 07-reinforce-toy.ipynb       # REINFORCE for μ
│   ├── 10-joint-optimization.ipynb  # Joint μ+γ (FWHM only)
│   ├── 12-joint-opt-with-sigma.ipynb # Joint μ+γ + sigma matching ✅
│   └── 13-real-data-optimization.ipynb # Real data validation ✅
├── data/                # Experimental NV center data
├── notes/               # Journal and presentations
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Start with `01-mc-algorithm.ipynb` for background, then jump to `12-joint-opt-with-sigma.ipynb` for the main results.

---

## Background

Extends [arXiv:2501.07951](https://arxiv.org/abs/2501.07951) — "Monte Carlo-based Parameter Reconstruction of an Optical Quantum System" (Pieplow et al.).

**Collaborator:** Dr. Gregor Pieplow · AG Schröder · Humboldt-Universität zu Berlin  
**Weekly meetings, ongoing since 2026.**

---

## Citation

```bibtex
@misc{santoyo2026quantumml,
  author = {Santoyo Alum, Anuar and Pieplow, Gregor},
  title = {Differentiable Monte Carlo for Parameter Estimation in NV Center Spectroscopy},
  year = {2026},
  howpublished = {\url{https://github.com/anuarsantoyo/quantum-ml}}
}
```

---

## License

MIT
