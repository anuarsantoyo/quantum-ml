# Quantum ML Project 

**Continuation of:** [Monte Carlo-based Parameter Reconstruction of an Optical Quantum System](https://arxiv.org/abs/2501.07951)

## Project Description

Extend and improve the Monte Carlo + ML approach for parameter reconstruction from low signal-to-noise optical quantum system data. The original paper applies this to photoluminescence excitation (PLE) spectroscopy of NV centers in diamond — we aim to generalize, optimize, and explore new directions.

**Collaborator:** Dr. Gregor Pieplow (weekly meetings)
**Model:** DeepSeek (via OpenClaw agent "Pukky")

---

> **Note:** This README is maintained by an AI agent (Pukky). It may contain hallucinations, outdated info, or jokes that don't land. For reliable, up-to-date information, consult the [JOURNAL.md](JOURNAL.md). It's written by a human.

## Current Status

After meeting with Gregor and Carla (27.05):
- ✅ Understanding of the MC simulation was correct
- ✅ Hyperopt approach → interesting but not novel enough, **deprioritized**
- ✅ **Differentiable MC** → the promising direction
- 📌 Repo to be shared with the group

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

### Differentiable MC
- [ ] Research approaches to make MC simulation backpropagatable
- [ ] Consider: reparameterization trick, Gumbel-Softmax, differentiable histogram approximations, score function estimators
- [ ] Prototype the most promising approach
- [ ] Test on dummy data

### Theory
- [ ] Understand Fisher Information and its role in parameter uncertainty
- [ ] Understand the 2-level quantum system with stochastic noise
- [ ] Understand PLE NV measurement in detail

### Data
- [ ] Get access to real experimental data (ask Gregor about Laura's data)
