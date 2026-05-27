# Quantum ML Project 

**Continuation of:** [Monte Carlo-based Parameter Reconstruction of an Optical Quantum System](https://arxiv.org/abs/2501.07951)

## Project Description

Extend and improve the Monte Carlo + ML approach for parameter reconstruction from low signal-to-noise optical quantum system data. The original paper applies this to photoluminescence excitation (PLE) spectroscopy of NV centers in diamond — we aim to generalize, optimize, and explore new directions.

**Collaborator:** Dr. Gregor Pieplow (weekly meetings)
**Model:** DeepSeek (via OpenClaw agent "Pukky")

---

## Current Status

After meeting with Gregor and Carla (27.05):
- ✅ Understanding of the MC simulation was correct
- ✅ Hyperopt approach → interesting but not novel enough, **deprioritized**
- ✅ **Differentiable MC** → the promising direction, exploit ML for backpropagation through the MC simulation
- 📌 Parameter uncertainty / Fisher Information is a key next step
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

### 2. Parameter Uncertainty via Fisher Information
After achieving differentiable MC, quantify parameter uncertainty. Fisher Information provides a natural framework for this in the ML context.

### 3. Hybrid Physics-Informed Model
Explore hybrid models that combine physical knowledge of the PLE process with learned components — potentially using insights from the Denmark-Covid project (previous work).

### 4. Hyperopt / Grid Search (archived)
The paper uses grid search for γ and n̄. Hyperopt can extend this to more parameters (σ, λ), but this direction has been deprioritized as not sufficiently novel. Prototype in `archive/notebooks/Hyperopt-n-sigma-gamma.ipynb`.

---

## TODOs

### Differentiable MC
- [ ] Research approaches to make MC simulation backpropagatable
- [ ] Consider: reparameterization trick, Gumbel-Softmax, differentiable histogram approximations, score function estimators
- [ ] Prototype the most promising approach
- [ ] Test on dummy data

### Theory
- [ ] Study Fisher Information in the parameter uncertainty context
- [ ] Understand the 2-level quantum system with stochastic noise
- [ ] Understand PLE NV measurement in detail

### Data
- [ ] Get access to real experimental data (ask Gregor about Laura's data)

---

## Questions for Gregor

1. ✅ Can I get access to the real data? → Asked, waiting
2. ✅ Is σ=6 fixed? → Can be optimized, grid search was easiest
3. ✅ Is Hyperopt sensible? → Interesting but not novel enough
4. ❓ How to quantify parameter uncertainty after differentiable MC fitting?
5. ❓ Should I look at the Denmark-Covid project for the differentiable approach?
