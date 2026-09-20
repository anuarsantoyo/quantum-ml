# experiment_8 — CONCLUSION

**AG-HYPOPT on the distributional LOSS function · real data · 30 trials · 2026-09-20 18:20 → 22:35 (4 h 14 m)**

---

## 1. What was asked

exp5 (schedule), exp6 (score design) and exp7 (bandwidth) all bottom out at a T-weighted objective of
**~0.078–0.082**. exp7 showed the residual is not tuning: the bandwidth **rule** is first-order (`target`)
and the coefficient is flat. The error is concentrated in the **low-count (T05/T20) μ–γ cells** — the
sim/target model gap (**FM#8**: sim σ_fit 1.04–4.25× too narrow, growing with T; **FM#6**: low-T γ deficit).
The optimizer's loss is the **2-D KDE negative log-likelihood**, which leans on that mis-scaled σ channel.

So the question was:

> **Does the CHOICE of distributional loss matter?** Replace the KDE NLL with alternative discrepancies —
> several bandwidth-free — and see which drives the (μ, γ) optimizer best.

**Frozen:** the schedule (exp5 winner), the bandwidth (exp7 winner: rule `target`, `h_f 4.0`, `h_s 3.6745`)
and the γ trust region (exp6 winner: `gamma_rel_cap 0.10736`).

**Searched (2 knobs):**

| knob | values / range | meaning |
|---|---|---|
| `loss_family` | `kde` / `w1_2d` / `energy_2d` / `mmd_2d` / `w1_fwhm` / `cvm_fwhm` | the distributional loss |
| `sigma_weight` | 0.0 – 1.0 | weight of the σ_fit channel (`loss = loss_FWHM + sw · loss_σ`) |

Families: **`kde`** (2-D KDE NLL, **the control — reproduces exp6's 0.080328 exactly**); **`w1_2d`**
(Wasserstein-1, sorted quantile matching, both channels); **`energy_2d`** (energy distance); **`mmd_2d`**
(RBF MMD², median-heuristic width); **`w1_fwhm`** / **`cvm_fwhm`** (FWHM-only: Wasserstein-1 and a
Cramér–von Mises-style squared quantile distance).

Each family supplies a per-sim-run **reward** `r_j = −loss_j` (μ REINFORCE) and a pathwise
**`g_j = ∂loss_j/∂γ`** (γ update). **Two normalizations** make it a test of the loss *form*:
`FAMILY_SCALE` (γ-gradient magnitude matched to the control at the frozen start,
`tools/calibrate_family_scales.py`) and a **per-step rescaling of the reward to the control's reward
spread** (see §6).

**Objective** (unchanged): T-weighted mean of CAP-capped per-cell rel-sq errors. Benchmark = 8 exps
(1nW/3nW × T05/20/60/100); **held-out** = 6 exps (1nW/3nW × T10/40/80), for validation only. Cap 30.

---

## 2. How the campaign ran

- **30/30 trials**, cell-by-cell (`nbrun.py`), each registered in `trials.json` and committed + pushed
  before the next (~8 min/trial). EI proposals concentrated on the `cvm_fwhm` basin after trial 05.
- **0 diverged cells in all 30 trials** — γ finite on all 8 cells, μ always inside the guard band.

---

## 3. Headline result: a robust FWHM-quantile loss beats the KDE control

**Best trial: `trial_25` — objective 0.0725 ± 0.0201** — `cvm_fwhm`, `sigma_weight 0.1972`.

**This is the only family that beats the KDE control (0.080328)** — by 0.0078 on the benchmark — and it
**reproducibly**: 20 `cvm_fwhm` draws give **min 0.0725, mean 0.0779** (18/20 ≤ the control), with a
near-identical twin at trial_26 (0.0727).

| family | n | benchmark min | benchmark mean |
|---|---|---|---|
| **`cvm_fwhm`** | 20 | **0.0725** | **0.0779** |
| `kde` (control) | 2 | 0.0830 | 0.0848 |
| `w1_2d` | 2 | 0.0849 | 0.0872 |
| `w1_fwhm` | 2 | 0.0908 | 0.0916 |
| `energy_2d` | 2 | 0.0996 | 0.1008 |
| `mmd_2d` | 2 | 0.1755 | 0.1794 |

### Per-cell behaviour at the best trial (trial_25)

```
              1nW T05  T20    T60    T100  | 3nW T05   T20    T60    T100
μ   Δ%         -42.1  +30.0  -31.9   -26.5 |  +8.9    +1.0  -39.2   -44.0
γ   Δ%         -50.4  -30.4  -14.4    -6.2 | -29.0   -10.2   -2.2    -5.5
```
μ rel-RMSE **31.5 %** (control 35.9 %), γ rel-RMSE **24.2 %** (control 22.3 %).

**The gain is in μ, not γ.** The low-count μ errors that dominate the KDE's objective are smaller here
(though still the largest term), while γ is a touch worse. That is the *opposite* of the intuition that
only the σ channel can inform μ — a FWHM-quantile loss **does** carry μ information, apparently because it
does not inherit the KDE count-channel's flat-responsibility (FM#8) failure.

---

## 4. Held-out validation — the direction holds

6 non-benchmark experiments (1nW/3nW × T10/40/80):

| config | benchmark | held-out |
|---|---|---|
| `kde` control (exp6 winner) | 0.0803 | **0.0822 ± 0.0297** |
| `cvm_fwhm` (trial_25, sw 0.197) | **0.0725** | **0.0736 ± 0.0195** |
| `cvm_fwhm` (sw 0.409) | — | 0.0783 ± 0.0209 |
| `w1_2d` (sw 0.209, best w1 draw) | 0.0849 | **0.0691 ± 0.0197** |

- **`cvm_fwhm` keeps its edge out-of-sample** (0.0736 vs control 0.0822) → the win is not a
  benchmark-tuned artifact.
- **`w1_2d` is a surprise:** it looked bad on the benchmark (0.0849, only 2 draws) but is the **best
  held-out config (0.0691)**. With only two benchmark samples its benchmark result was an unlucky draw
  (the objective's chaotic scatter is ~±0.01). **The honest read: both robust quantile losses with a low
  σ-weight (`w1_2d`, `cvm_fwhm`) beat the KDE control out-of-sample; the benchmark under-sampled `w1_2d`.**
- Magnitudes: 0.072–0.074 vs 0.082 ≈ a **~10 % relative improvement**, but each is within ~0.5 SE of the
  other (SE ≈ 0.02–0.03 across cells) — a **consistent direction, not a decisive separation**.

---

## 5. What the family ranking says

1. **The σ_fit channel is the problem — again.** Every *winning* config drives the σ channel weakly or not
   at all (FWHM-only `cvm_fwhm`; `w1_2d` at sw ≈ 0.21). Every *losing* family keeps a strong, structural
   σ channel: `energy_2d` (0.100) and `mmd_2d` (0.176) are 2-D, bandwidth-free losses — but the MMD² RBF
   is dominated by the mis-scaled σ cloud and collapses (0.176).
2. **Robust 1-D quantile distances > kernel density.** Wasserstein-1 and CvM are bandwidth-free, and
   their gradients do not blow up on the sparse, skewed low-count targets — the same robustness the exp7
   bandwidth analysis pointed to, now made concrete.
3. **The KDE at low σ-weight does *not* reproduce the gain** — the campaign ran the `kde` control at
   sw 0.228 (0.0830) and 0.189 (0.0866); down-weighting the KDE's σ channel alone does not help. It is the
   **loss form**, not the σ weight, that moves the needle.
4. **`sigma_weight` is degenerate for the FWHM-only families.** `cvm_fwhm`/`w1_fwhm` do not use `sw` in the
   loss at all; it enters only through the μ-reward reference scale (§6), i.e. it acts purely as a
   **μ-step-size knob** for them. Their `sw ≈ 0.2` optimum is a statement about the μ step, not the loss.

---

## 6. Implementation note (why the campaign needed a fix mid-flight)

The first launch stopped after trial_01 for two self-inflicted setup bugs, both fixed before the relaunch:

1. **Seed-notebook stamping.** Stamping `trial_01.ipynb` with a *global* placeholder replace also clobbered
   the generator's own cell-9 literals (`_stamp('{N}', …)`), so cell 9 could not create trial_02. Fixed by
   stamping only cells 0/2 — **never the generator cell**.
2. **μ-reward scale (the substantive one).** With the raw, γ-calibrated scale, the per-run reward from a
   non-KDE loss has a spread spanning **~3 orders of magnitude across cells** (w1_2d r-std 2.5 → 80;
   cvm_fwhm 2.2 → 1375), which the frozen `lr_mu` cannot accommodate — trial_01 drove μ to **1120** on
   1nW T05 (a reward-unit artifact, not the loss). Fixed by **rescaling each step's reward to the control's
   reward spread** (the KDE log-likelihood on the same draws). After the fix, `w1_2d` on 1nW T05 goes
   μ 4.7 → 6.0, and the **`kde` control reproduces 0.080328 exactly** on both the benchmark and held-out
   (0.0822). This is the only reason the comparison is about the loss *form*.

---

## 7. Where the series goes next

1. **Adopt a robust FWHM-quantile loss as the new default** — `cvm_fwhm` or `w1_2d` at low σ-weight. This
   is the first lever since exp5 that moved the objective and **held out-of-sample**, and it is cheap
   (bandwidth-free).
2. **Re-run a focused campaign on the winning family** to resolve the ~0.01 chaotic scatter: more trials,
   σ-weight re-defined properly for FWHM-only losses (or pinned), and **repeat each config ≥2×**.
3. **The σ_fit channel deserves a direct test:** all evidence (FM#8, exp6 damping, exp8 FWHM-only wins)
   says the simulator's σ_fit model is the bottleneck. Either rebuild it (differentiable forward model,
   Option C) or **drop it** and re-tune μ/γ with the FWHM-quantile loss alone.
4. **Protocol:** keep the held-out split and the pair (benchmark + held-out) reporting — `w1_2d` would have
   been wrongly discarded on the benchmark sample alone.

---

## 8. Provenance

- **Experiment:** `agent-guided-hypopt/experiment_8/` — `ag_hypopt.py` (loss-family knob + reward
  normalization), `space.json`, `trials.json` (protocol + 30 entries), `trial_01.ipynb … trial_30.ipynb`,
  `README.md`, `tools/calibrate_family_scales.py`.
- **Best trial:** `trial_25.ipynb` (0.0725); held-out winner `w1_2d` @ sw 0.209 (0.0691).
- **Repo state:** branch `develop`; campaign closed at `e8f3d69` (trial_30); this conclusion in the next commit.
- **Series context:** series-19 (FM#8) · exp5 schedule (0.108–0.113) · exp6 score design (**0.0803**) ·
  exp7 bandwidth (no gain) · **exp8 loss form: robust FWHM-quantile losses beat the KDE by ~0.007–0.009
  and hold out-of-sample; the σ channel is the culprit.**
