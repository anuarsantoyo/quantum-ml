# Quantum ML Project 

**Continuation of:** [Monte Carlo-based Parameter Reconstruction of an Optical Quantum System](https://arxiv.org/abs/2501.07951)

## Project Description

Extend and improve the Monte Carlo + ML approach for parameter reconstruction from low signal-to-noise optical quantum system data. The original paper applies this to photoluminescence excitation (PLE) spectroscopy of NV centers in diamond — we aim to generalize, optimize, and explore new directions.

**Collaborator:** Dr. Gregor Pieplow (weekly meetings)
**Model:** DeepSeek (via OpenClaw agent "Pukky")

---

> **Note:** This README is maintained by an AI agent (Pukky). It may contain hallucinations, outdated info, or jokes that don't land. For reliable, up-to-date information, consult the [JOURNAL.md](notes/JOURNAL.md). It's written by a human.

## Current Status

After building the forward pipeline (19.06):
- ✅ Full MC simulation built: sampling → fitting → simulation → loss
- ✅ Loss landscape works (L2 on smooth bin counts via `kde_to_bin_counts`)
- 🔴 **Next:** Make the pipeline differentiable — reparameterize sampling + implicit diff
- 📌 Repo shared with the group

---

## Project Structure

```
qm-ml/
├── README.md          # Dashboard — overview, ideas, TODOs
├── JOURNAL.md         # Running log — ideas, decisions, notes
├── docs/papers/       # PDFs of papers we read
├── docs/journal/      # Images, sketches for the journal
├── notebooks/         # Jupyter notebooks (archived prototypes in archive/)
├── requirements.txt
└── .gitignore
```

---

## Ideas

### 1. Differentiable Monte Carlo (main direction)
Convert the MC simulation into a backpropagatable problem (e.g., using PyTorch). Instead of grid search or discrete optimization, treat parameters as learnable and optimize via gradient descent using a differentiable loss.

### 2. Hybrid Physics-Informed Model
Explore hybrid models that combine physical knowledge of the PLE process with learned components — potentially using insights from the Denmark-Covid project (previous work).

---

## TODOs

### Differentiable MC Pipeline
- [x] Sampling (continuous photons)
- [x] Voigt fitting (MLE via L-BFGS)
- [x] Collapse (simulate 2000 runs → FWHM distribution)
- [x] KDE
- [x] Loss function (kde_to_bin_counts + L2)
- [ ] **Make the pipeline differentiable** — reparameterize sampling + implicit diff through fit
- [ ] Test end-to-end optimization on dummy data

### Theory
- [ ] Understand Fisher Information and its role in parameter uncertainty
- [x] Understand the 2-level quantum system with stochastic noise
- [x] Understand PLE NV measurement in detail

### Data
- [ ] Get real data from Gregor
