# Updates 11 Sep 2026
---

# Last status

- The differentiable MC pipeline is complete and **recovers the true parameters on synthetic data** generated from those same true values (μ = mean photon count, γ = Lorentzian HWHM).
- We compute the **Fisher information / Cramér–Rao bound** as the uncertainty statement (per Gregor's endorsement).

![alt text](image.png)
---

# Next goal: tune the model to fit all experiments

- **Goal:** get all 14 experiment optima to land as close as possible to their true (μ, γ).

## Tried Luck with real data directly (Notebook series: 15)

Wanted to see how my current model performed with the real data

N_ITER=80 × N_RUNS=500, μ LR 15→4.5, γ LR 0.5. True σ_phys and λ fixed.




| 1nW| 3nW |
|:----:|:-----:|
|![alt text](image-1.png)|![alt text](image-8.png)|
|![alt text](image-2.png)|![alt text](image-9.png)|
|![alt text](image-3.png)|![alt text](image-10.png)|
|![alt text](image-4.png)|![alt text](image-11.png)|
|![alt text](image-5.png)|![alt text](image-12.png)|
|![alt text](image-6.png)|![alt text](image-13.png)|
|![alt text](image-7.png)|![alt text](image-14.png)|


## Checked syntetic first (Notebook series: 16)

- **Why synthetic first:** if the model cannot recover the true values from data that the *same model* generated, then there is a gap in the model that no amount of real data would ever close. So we make it perfect on its own data first.
- The model already works (all 14 optimize in the right direction, just biased), so the remaining work is **reducing the bias: trial and error over the design choices.**

| 1nW| 3nW |
|:----:|:-----:|
|![alt text](image-15.png)|![alt text](image-22.png)|
|![alt text](image-16.png)|![alt text](image-23.png)|
|![alt text](image-17.png)|![alt text](image-24.png)|
|![alt text](image-18.png)|![alt text](image-25.png)|
|![alt text](image-19.png)|![alt text](image-26.png)|
|![alt text](image-20.png)|![alt text](image-27.png)|
|![alt text](image-21.png)|![alt text](image-28.png)|



# Idea: structural tuning with an Agent loop

**First attempt at an agent tuning mechanism.** A notebook that: starts with a summary + context of what happened before, proposes a change, tests it, writes a summary of the results (what went wrong/right and why), and iterates.

## 17 series: fixing the optimizer on synthetic data

The 17 notebooks ran the full **14-experiment synthetic benchmark** (closed loop, targets generated at the true values) and fixed the optimizer **one issue at a time**, re-checking the recovered (μ, γ) after each change.

### 17a: improvement playground (14-experiment synthetic sweep, parallel)

Runs the whole 14-experiment sweep with knobs, so we can see the optimization paths and the MSE/STE. First improvements: the exact `dσ/dγ` implicit term, a mean-matching anchor (`LAMBDA_MEAN`, off by default), a σ-kernel bandwidth floor (`H_S_MIN`), and parallel workers. Full sweep in about 75 min.

![alt text](image-29.png)

### 17b: the λ_mean = 0.0 baseline

The baseline for the playground: no mean anchor. γ recovers well (RMSE ~1.2), but there is a clear μ loss at high transmission.

![alt text](image-30.png)

### 17d: the λ_mean = 0.1 sweep

Tried `LAMBDA_MEAN = 0.1`; the anchor made γ worse (RMSE 3.32), so it stays off. This is why `LAMBDA_MEAN = 0` is now frozen.

![alt text](image-31.png)

### 17e: the μ-starvation fix

The μ score was divided by `σ_prop²` (up to 1680 at 3nW T100), so each step was about 0.1 photons and μ could never climb. Switched to `(n−μ)/σ_prop`, removed the LR-decay floor, and raised `N_ITER` to 200. This closed most of the high-transmission μ loss (RMSE 23.4 to 1.75).

![alt text](image-32.png)

### 17f: scale-invariant μ steps + γ anneal

Replaced `σ_prop` with a fixed reference `σ_ref = 10`, so every experiment takes the same photon-scale step (removing the low-μ overshoot); annealed the γ learning rate `LR_γ·(1 − 0.5·t/N)` to settle the noisy low-count γ outliers. This closes the high-transmission μ loss.

![alt text](image-33.png)

### 17g: z-form γ score

The remaining γ jumps came from the KDE γ-score scaling as `1/H²`, which saturated the clip and gave steps of about 5 MHz. Normalizing by one power of bandwidth and a fixed reference `H_REF` (the γ analogue of `σ_ref` for μ) removed the jumps and improved γ RMSE. This is the config that became the AG-HYPOPT baseline.

![alt text](image-34.png)
![alt text](image-35.png)
![alt text](image-36.png)
![alt text](image-37.png)

**Net:** after these fixes the machinery recovers the true (μ, γ) almost exactly on synthetic data (μ RMSE about 3.7 %, γ about 4.3 %).

### Sanity check with real data

Applied to real data it is still a catastrophe, but since the model optimizes its own (synthetic) data well, this tells us the residual error is **not** in the model, it is a data/model mismatch.

![alt text](image-38.png)




---

# AG-HYPOPT

Agent-Guided Hyperparameter Optimization
A probabilistic optimizer proposes, a physics-informed agent chooses, we run and record.

---

## What is uncertainty-aware TPE?

**The idea.** Instead of searching the hyperparameter space blindly, Tree-structured Parzen Estimators TPE **learns from the trials already run**. It fits two models of the space: one to the trials that did well (the **good** set) and one to those that did badly (the **bad** set). New candidates are drawn where the good model is high and the bad model is low, so the search concentrates where an improvement is most likely.

**Why "uncertainty-aware".** Each trial's objective is a noisy measurement (the benchmark runs are stochastic), so we carry its uncertainty along. It then matters in two places: a trial that looks good but was measured loosely is trusted less when we decide good vs bad, and it also gets a wider, flatter kernel so it shapes the density less. Reliable trials are trusted more than lucky ones.

---

## Uncertainty-aware TPE (1/2): the model

We use the **Tree-structured Parzen Estimator (TPE)** and make it **uncertainty-aware**.

**Why TPE (more flexible):** needs **no gradients** (black-box objective), handles **mixed continuous / integer / categorical** and **conditional** parameters, and is **cheap to fit** from a few expensive trials.

Notation: trial `i` has objective `ℓᵢ` (lower is better) and uncertainty `sᵢ`.

**Step 1, rank with uncertainty.** Each trial gets an uncertainty-adjusted loss (a trial is "good" only if even its upper bound is low):

$$\tilde{\ell}_i = \ell_i + \lambda\, s_i \qquad (\lambda = 0.5)$$

**Step 2, Good/Bad split.** Sort trials by `ℓ̃` ascending; the **lowest quantile** (fraction `q = 0.25`) becomes the "good" set `G`, the rest is "bad" `B`.

**Step 3, one density per side** (per parameter). A **variable-bandwidth KDE plus a uniform prior** over the box `[lo, hi]`:

$$g(x) = \frac{w_0\, U_{[lo,hi]}(x) + \sum_{i \in G} \varphi\!\left(x \mid v_i,\, h_i^2\right)}{w_0 + |G|}$$

`vᵢ` is a trial's parameter value, `φ` is a Gaussian, and each kernel width `hᵢ` grows with that trial's uncertainty. `w₀` weights the uniform prior (keeps a nonzero density everywhere → stability + soft exploration).

<!-- image: Good/Bad KDEs over a 1D hyperparameter (uncertainty-scaled widths visible) -->

---

## Uncertainty-aware TPE (2/2): scoring

**Two densities.** Fit one density per side: the **good density** `g(x)` from the good trials `G`, and the **bad density** `b(x)` from the bad trials `B` (same KDE form, per parameter).

**Score.** Each candidate is scored by the **good/bad ratio**:

$$\mathrm{EI}(x) = \frac{g(x)}{b(x)}$$

High where the good trials concentrate and the bad ones do not.

**Propose.** Draw candidates from the good density `g(x)` (plus a few fully random explore slots), and let the agent pick one.

<!-- image: good vs bad density over a 1D hyperparameter, and the ratio g/b -->

---

## AG-HYPOPT knobs (what you can set)

| Knob | Default | What it does |
|---|---|---|
| `n_initial` | 5 | Number of trials the optimizer draws **uniformly at random** before it has enough history to fit models. The first `n_initial` trials are pure exploration; after that, proposals come from TPE. Higher = more initial coverage, lower = switch to model-based proposals sooner. |
| `quantile` | 0.25 | Fraction of trials labelled **good** (`q`) and used to build the good density; the rest form the bad density. Lower = more selective (only the very best count as good), higher = a wider, more tolerant good set. |
| `lcb_lambda` | 0.5 | How strongly a trial's uncertainty counts against it in the **Good/Bad ranking**. Trials are ranked by `ℓ + λ·s`, so a trial is good only if even its pessimistic objective is low. 0 ignores uncertainty; larger values favour reliably-measured trials over lucky noisy ones. |
| `bandwidth_beta` | 0.5 | How strongly a trial's uncertainty **widens its kernel** in the density (each width scales as `(s_i/s_med)^β`). An uncertain trial gets a wider, flatter bump and shapes the density less; 0 means no scaling (all kernels the same width). |
| `prior_weight` | 1.0 | Weight of a **uniform prior** mixed into every density on top of the trial kernels. It keeps the density nonzero everywhere in the box (no dead regions), which stabilises the good/bad ratio and adds soft exploration; 0 means the density comes from the trials alone. |
| `explore_slots` | 2 | How many candidates in each trials-phase batch are drawn **fully at random** instead of from the good density. Guarantees every batch still contains pure exploration, however peaked the model gets (bounded to the batch size). |

---

## How the class is used (sklearn-style)

`AGHyperopt` follows the scikit-learn pattern: build it with the knobs, `fit` on the trial history, then ask for proposals.

```python
from ag_hypopt import AGHyperopt

opt = AGHyperopt(n_initial=5, quantile=0.25, lcb_lambda=0.5,
                 bandwidth_beta=0.5, prior_weight=1.0, explore_slots=2)   # set the knobs

opt.fit(SPACE_PATH, TRIALS_PATH)    # learn the good/bad densities (returns self)
cands = opt.propose_trials(30)      # -> [{'params': {...}, 'ei': 0.9, ...}, ...]
```

- Same shape as `Estimator().fit(X, y).predict(X)`: **constructor = the knobs**, **`fit` = learn**, **`propose_trials` = generate**.
- `fit(space, trials)`: loads the space + the recorded trials, picks the phase, and builds the good/bad densities. Learned attributes carry a trailing underscore (`good_models_`, `history_`, ...).
- `propose_trials(n)`: returns `n` candidate configs (params + EI) in draw order; the agent then picks one.
- Cold start: with fewer than `n_initial` trials, `fit` returns without building densities and `propose_trials` falls back to uniform draws.

---

## The AG-HYPOPT cycle

1. **Warm-up:** the first `n_initial` trials are uniform random draws (no history yet).
2. **Propose:** TPE fits the Good/Bad densities from the registry and draws a **batch of candidate configurations** (plus a few fully random explore slots).
3. **Agent decides:** the agent is shown the batch together with the project context (physical meaning of each parameter, past results) and **picks one candidate**. It must choose among the proposals, so it stays inside the probabilistic search but can still inject domain reasoning.
4. **Run:** the chosen configuration runs the benchmark (all 14 experiments) and records its **objective ± uncertainty** back into the registry.
5. **Repeat:** the registry grows, the densities sharpen, the proposals improve.

The agent writes no code and creates no files: it only picks a proposal.

<!-- image: AG-HYPOPT loop diagram -->
<!-- image: candidate table shown to the agent (params + EI hints) -->

