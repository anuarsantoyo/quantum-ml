# Quantum ML Project 🧪🔬🤖

**Continuation of:** [Monte Carlo-based Parameter Reconstruction of an Optical Quantum System](https://arxiv.org/abs/2501.07951)

## Project Description

Extend and improve the Monte Carlo + ML approach for parameter reconstruction from low signal-to-noise optical quantum system data. The original paper applies this to photoluminescence excitation (PLE) spectroscopy of NV centers in diamond — we aim to generalize, optimize, and explore new directions.

**Collaborator:** Dr. Gregor Pieplow (weekly meetings)
**Model:** DeepSeek (via OpenClaw agent "Pukky")

---

## Project Structure

```
qm-ml/
├── README.md          # Dashboard — overview, ideas, TODOs, questions
├── JOURNAL.md         # Running log — ideas, decisions, notes
├── docs/papers/       # PDFs of papers we read
├── notebooks/         # Jupyter notebooks for exploration & prototyping
├── requirements.txt
└── .gitignore
```

*When coding begins, we'll expand with: `src/`, `tests/`, `data/`.*

---

## Ideas

### 1. Hybrid Model Approach
Explore different ways to model the PLE process. Instead of a pure MC approach, create a hybrid model that leverages physical knowledge of the underlying process — potentially combining a physics-based model with learned components.

### 2. Optimizing More Parameters via Hyperopt
The paper does a grid search for γ and n̄. The idea is to also optimize other fixed parameters like σ (currently fixed at 6) and λ (currently fixed at 2) using hyperopt or similar. This could improve reconstruction accuracy. *Prototyped in `notebooks/Hyperopt-n-sigma-gamma.ipynb`.*

### 3. Deeper Physics Models
Build models from scratch based on more detailed physical formalisms (Weisskopf-Wigner, Master equation / Jaynes-Cummings + Lindblad, Huang-Rhys electron-phonon coupling, charge-state switching). Identify important parameters and optimize them using PyTorch on real data.

### 4. Monte Carlo + PyTorch Batch Optimization
Instead of grid search over many individual MC simulations, treat γ, n̄ (and potentially other parameters) as learnable PyTorch parameters. Run batches of ~2000 MC simulations per step and optimize using a χ²-based loss function. Requires refreshing how loss optimization works for stochastic processes.

---

## TODOs

### Research & Understanding
- [ ] Deepen understanding of the Monte Carlo method used in the paper
- [ ] Understand the ML approach and how it compares to MC
- [ ] Understand the 2-level quantum system with stochastic noise ("spectral shape based on noise process")
- [ ] Understand the description of PLE NV (how the measurement works in detail)
- [ ] Study Fisher Information and parameter uncertainty
- [ ] Refresh how loss-optimization of stochastic processes works (for the PyTorch MC idea)
- [ ] Read the complexer models in detail (Master equation, Huang-Rhys, etc.)

### Implementation
- [ ] **Debug** `Hyperopt-n-sigma-gamma.ipynb` — first test ran without errors but needs line-by-line verification
- [ ] Get access to real experimental data

### Conceptual
- [ ] Understand why Cauchy distribution is used and what physical meaning γ has in relation to it
- [ ] Evaluate: is the hybrid model approach viable given the low signal-to-noise ratio?

---

## Questions for Gregor

1. Can I get access to the real data?
2. Why is σ fixed at 6? Are there more parameters that could be optimized?
3. Is the optimization idea (hyperopt for more parameters) sensible in general?
4. *(Add more as they come up)*

---

*This file auto-syncs from JOURNAL.md — ideas, TODOs, and questions are periodically extracted and updated.*
