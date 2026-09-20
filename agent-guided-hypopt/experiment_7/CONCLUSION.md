# experiment_7 — CONCLUSION

**AG-HYPOPT on the KDE bandwidth RULE family · real data · 30 trials · 2026-09-20 09:35 → 13:44 (4 h 09 m)**

---

## 1. What was asked

experiment_6 searched the **score design** (schedule frozen) and won: best **0.0803** (trial_07), the μ
floor halved (rel-RMSE 77.9 → 35.9 %) and the γ divergence eliminated (0/30 vs 9/40). Its correlation
table then pointed at one thing:

- `corr(obj, h_f_scale) = −0.948`, `corr(obj, h_s_scale) = −0.809` — **bandwidth is the dominant lever**;
- **17/30 trials sat exactly at the `h_f` upper bound (4.0)** — the answer was *"flatter than the box allows"*.

So the question was:

> **With the schedule (exp5 winner) AND the score design (exp6 winner) frozen, does extending the
> bandwidth coefficients to [1, 12] — or changing the bandwidth RULE itself — beat the exp6 winner?**

The **physics hypothesis** (19 Sep discussion): the KDE kernel lives over the **sim cloud** but its width
has always been computed from the **REAL target** (Scott) — structurally mismatched whenever the sim/target
spreads differ (always, FM#6/#8). A flat kernel makes the responsibilities uniform → reward std → 0 → the
REINFORCE μ channel dies; a knife-edge kernel makes the likelihood spiky. exp6 answered "flatter is better"
but ran out of room.

**Frozen:** schedule (`lr_mu 0.1669`, `mu_anneal 0.3455`, `lr_gamma 0.4716`, `gamma_anneal 0.4723`) and
score design (`sigma_weight 0.86489`, `gamma_rel_cap 0.10736`) — the exp5 and exp6 winners, with the
campaign **reproducing exp6's trial_07 to the last digit (0.080328)** as its starting point.

**Searched (3 knobs)** — the pre-authorized structural change (a configurable bandwidth rule family):

| knob | values / range | meaning |
|---|---|---|
| `bandwidth_rule` | `target` / `sim` / `power` | `target` = Scott from the real target (exp5/6); `sim` = Scott from the sim cloud, **per step** (sim-adaptive); `power` = Scott from the pooled real targets of that laser power (global/per-power) |
| `h_f_scale` | 1.0 – 12.0 | multiplier on the FWHM-channel bandwidth (extended from exp6's 4.0 ceiling) |
| `h_s_scale` | 1.0 – 12.0 | multiplier on the σ-channel bandwidth (floored at `h_s_min = 0.05` MHz) |

**Objective** (unchanged, pre-registered): T-weighted mean of CAP-capped per-cell rel-sq errors
(`err_i = min(rel_sq_i, 1.0)`, `w(T) = 0.25 + 0.75·T/100`). Benchmark = **8 experiments**
(1nW/3nW × Trans05/20/60/100); **held-out** = the 6 others (1nW/3nW × Trans10/40/80), for validation only.
Cap `MAX_TRIALS=30`.

---

## 2. How the campaign ran

- **30/30 trials**, each executed **cell-by-cell** (`nbrun.py`, persistent kernel), each registered in
  `trials.json` and **committed + pushed** before the next (~8 min/trial).
- 28 trials on the `target` rule, **1 on `sim`, 1 on `power`** (the model-informed phase quickly learned
  the two alternative rules were bad; see §4).
- Every trial notebook is self-contained; objective deterministic per config.

---

## 3. Headline result: no improvement — exp6's winner stands

**Best campaign trial: `trial_05` — objective 0.0818 ± 0.0239** (`target`, `h_f_scale 3.580`, `h_s_scale 5.784`).

**The frozen exp6 winner scores 0.0803 — better than every one of the 30 trials.** The campaign's best is
**+0.0015 worse**, i.e. **inside the measurement scatter** (the exp6 chaos finding: <10⁻³ knob changes move
the objective ~0.01). Distribution over the 30 trials: **min 0.0818, median 0.0885, mean 0.0927**, max 0.1541
(the forced `power` probe). Restricted to the `target` rule: min 0.0818, median 0.0881, mean 0.0894 — the
**same plateau exp6 already sat on** (exp6: median 0.0865, mean 0.0907).

> **Verdict: the bandwidth experiment produced no win. exp6's `h_f = 4.0` was not a *binding* optimum —
> it was already deep enough into the flat region. The extended [1, 12] range and the whole `h_s` axis
> buy nothing.**

---

## 4. The bandwidth RULE is first-order; the coefficient is not

**Rule ordering is unambiguous** (benchmark / held-out objective):

| rule | benchmark | held-out (6 exps) | failure mode |
|---|---|---|---|
| **`target`** (Scott from real target) | **0.0818** best / 0.0881 median | **0.0822** (default), 0.0793 (trial_05) | — |
| `power` (pooled per laser power) | 0.1541 | 0.1146 | **μ overshoot** on the low-count cells (up to +123 %) |
| `sim` (sim-adaptive, per step) | 0.1249 | 0.1442 | **γ collapses** to ~31–40 % of truth on the low-count cells |

So the *rule* is a genuine first-order lever — and it says the exp5/exp6 design (bandwidth **from the real
target**) is the right one. The two alternatives both break in the low-count corner: `sim` makes the kernel
track the sim cloud (which is far too tight at the true μ — the FM#8 mismatch made quantitative), and
`power` pools the 7 transmissions into one bandwidth so the sparse low-count targets get a width meant for
the dense high-T ones.

**The coefficients are flat inside the plateau.** On the `target` rule, `corr(obj, h_f_scale) = −0.21`
(was −0.948 in exp6) and `corr(obj, h_s_scale) = +0.09` — i.e. **no usable gradient**. Every trial with
`h_f ≈ 3–5.5` and `h_s ≈ 2.5–11.5` lands in 0.082–0.096, all differences within the SE (0.02–0.04). Only the
**edges degrade**: `h_f = 1.13` → 0.1165 (μ overshoot +173 % on a low-count cell) and `h_f = 9.82` → 0.0926.

**Read:** exp6's steep −0.948 correlation was an artifact of sampling *one side* of the turnover (all of
exp6 lived at `h_f ≤ 4`). Once you go past it, the objective is flat-to-slightly-worse — the "flatter is
better" law saturates almost immediately. The 4.0 bound was **not** a wall.

**Divergence:** **0/30** — γ finite on all 8 cells every trial. The exp6 γ rel-cap keeps working, as expected.

---

## 5. Held-out validation (the campaign's own success bar)

The winner must hold on the 6 non-benchmark experiments. It does — and so does the ranking:

```
                            held-out objective (6 exps, lower better)
exp6 frozen default          0.0822 ± 0.0297
exp7 best trial_05           0.0793 ± 0.0254     (benchmark 0.0818)
power probe                  0.1146 ± 0.0255
sim probe                    0.1442 ± 0.0560
```

- The `target` rule **generalizes** (benchmark 0.080 → held-out 0.079–0.082) — the win is not a
  benchmark-tuned artifact.
- `trial_05` and the frozen default are **indistinguishable** (0.0793 vs 0.0822, both ±~0.03): the choice
  between them is not resolvable. The safe call is to keep the **exp6 winner** (fewer knobs moved).
- The alternative rules fail on held-out too — the rule ordering is robust, not a benchmark quirk.

---

## 6. How well did the AG-HYPOPT idea work?

**The campaign was well-run but the question was already answered.**

- **It found the structure fast**: the `sim` probe (trial_02) and the `power` probe (trial_03) came in the
  uniform phase and both looked structurally bad; from then on the proposals stayed on `target`, correctly.
- **The agent's judgement mattered**: it kept probing the plateau and the `h_f` edges (trial_04 h_f 9.82,
  trial_07 h_f 1.13) rather than re-sampling the middle — which is what pinned down *where* the objective
  turns over.
- **What limited it**: there was **nothing left to find**. The honest reading is that the space was
  **exhausted on the first probe** — the frozen default was already the optimum, and 29 of 30 trials merely
  measured noise. A good stop rule (§7) would have ended this after ~5 trials and saved 3 hours.

**Bugs found and fixed (setup, not campaign):**
1. **`plot_parallel` crashed on the categorical `bandwidth_rule` axis** (it tried to min-max a string). It
   fires *after* `objective()` returns, so **no trial was affected**, but the parallel-coordinates figure
   never rendered in any trial notebook. Fixed in `ag_hypopt.py` (categorical params now map to category
   index; numeric ticks/labels shown per type). Verified.
2. **The shipped `trial_01.ipynb` seed was an un-stamped copy of the template** (`TRIAL_ID = 'trial_XXX'`),
   so cell 9 could not generate trial_02. The runner applied the generator's own stamp (`{N}→01`,
   `trial_XXX→trial_01`) to the seed — the same stamp every later trial gets — and the campaign proceeded.
   **Lesson: always stamp `trial_01` with the generator's `_stamp` when forking a new experiment folder.**

---

## 7. Where the series goes next

1. **Bandwidth is a dead end on this dataset.** The rule is settled (`target`, i.e. Scott from the real
   target); the coefficient is flat over a factor of ~2. Do not spend another campaign on it. Adopt
   exp6's `h_f = 4.0, h_s = 3.675` (keep the frozen default).
2. **The residual is the model gap, not the tuning.** Every trial — exp5, exp6, exp7 — bottoms out around
   **0.078–0.082**, and per-cell the error is concentrated in the **low-count (Trans05/Trans20) μ–γ cells**
   (FM#8: sim σ_fit 1.04–4.25× too narrow; FM#6: low-T γ deficit). This reproduces the 15 Sep series-21d
   finding: *the μ failure is the simulator model gap (E10); the next lever is a **differentiable forward
   model (Option C)**, not a better fitter or a better kernel.*
3. **Protocol:** add an explicit **stop rule** — stop after N consecutive trials with no improvement beyond
   the objective's own SE (here: ~t05, saving ~3 h). And **repeat each config ≥2×** for stochastic objectives.
4. **Keep the γ rel-cap** and the **held-out split** in every future campaign: they are cheap and both did
   their job here (0 divergences; the win was verified out-of-sample).

---

## 8. Provenance

- **Experiment:** `agent-guided-hypopt/experiment_7/` — `ag_hypopt.py` (bandwidth rule family + categorical
  `plot_parallel` fix), `space.json`, `trials.json` (protocol + 30 entries), `trial_01.ipynb … trial_30.ipynb`,
  `README.md`.
- **Best trial:** `trial_05.ipynb` (0.0818); reference/frozen `trial_07`-equivalent = 0.0803.
- **Held-out harness:** `objective(cfg, experiments=[...HELD_OUT])` — 6 exps, run 2026-09-20.
- **Repo state:** branch `develop`, campaign closed at commit `237f932` (trial_30); this conclusion + the
  `plot_parallel` fix in the following commit.
- **Series context:** series-19 autopsy (FM#8) · experiment_5 (schedule plateau 0.108–0.113) ·
  experiment_6 (score design: **0.0803**, μ floor halved, divergence eliminated, h_f pinned at bound) ·
  experiment_7 (**bandwidth rule: `target` wins, coefficients flat, no gain — the plateau is the model gap**).
