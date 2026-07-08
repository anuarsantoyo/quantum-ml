# Quantum ML 🔬⚛️

**Differentiable Monte Carlo for Parameter Estimation in Optical Quantum Systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.07951-b31b1b.svg)](https://arxiv.org/abs/2501.07951)

---

## Overview

This project develops  **differentiable Monte Carlo methods** for parameter estimation in low signal-to-noise optical quantum systems. We replace brute-force grid search with gradient-based optimization by making the entire simulation pipeline differentiable.

**Application:** Photoluminescence excitation (PLE) spectroscopy of NV centers in diamond — reconstructing optical linewidths from noisy experimental data.

**Key idea:** Convert the Monte Carlo simulation into a backpropagatable program using PyTorch, enabling gradient descent on physical parameters instead of discrete grid search.

---

## Results

| Approach | Status | Description |
|----------|--------|-------------|
| Differentiable MC framework | ✅ Working | Gradient flows through sampling → fitting → loss |
| MMD² loss | ✅ Working | Continuous, differentiable distribution comparison |
| Gamma optimization | ✅ Working | End-to-end: synthetic data → loss → parameter update |
| Real data validation | 🔶 In progress | Testing on experimental NV center data |

![Optimization convergence](notebooks/differentiable-gamma-simplified.ipynb)

---

## Project Structure

```
├── notebooks/           # Jupyter notebooks (chapters 1-3)
│   ├── chapter-1-*      # MC algorithm explanation
│   ├── chapter-2-*      # Differentiable sampling
│   └── chapter-3-*      # Implicit differentiation through fitting
├── data/                # Experimental data (gitignored)
├── scripts/             # Utility scripts
├── notes/               # Journal, presentations
└── docs/                # Papers, references
```

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Start with `chapter-1-mc-algorithm.ipynb` for background, then `differentiable-gamma-simplified.ipynb` for the main results.

---

## Background

This work extends [arXiv:2501.07951](https://arxiv.org/abs/2501.07951) — "Monte Carlo-based Parameter Reconstruction of an Optical Quantum System" (Pieplow et al.).

**Collaborator:** Dr. Gregor Pieplow, AG Schröder — Humboldt-Universität zu Berlin
**Weekly meetings, ongoing research since 2026.**

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

MIT — see [LICENSE](LICENSE)
