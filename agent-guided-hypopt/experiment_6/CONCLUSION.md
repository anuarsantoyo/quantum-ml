# experiment_6 — CONCLUSION

**AG-HYPOPT on the score DESIGN (FM#8 attack) · real data · 30 trials · 2026-09-19 08:48 → 13:17 (4 h 29 m)**

---

## 1. What was asked

experiment_5 tuned the **learning-rate schedule** and hit a plateau: best objective **0.1074** (trial_21),
honest level ~0.108–0.113, with **no knob correlation above |r| = 0.2**. The residual was (a) the **μ floor**
and (b) a **sporadic γ divergence** — neither schedule-controlled.

Series-19's autopsy pinned the μ floor to **failure mode #8 (FM#8)**: on real data the REINFORCE
count channel is *dead* (KDE responsibility ≈ uniform over 1100–2500 points), so μ can only be informed
through the **σ_fit channel**, whose **shape is structurally mismatched** (simulated σ_fit is
**1.04–4.25× too narrow**, growing with T). The likelihood therefore develops a spurious **low-μ mode at
μ/μ_true ≈ 0.30–0.50**, and the optimizer descends into it.

So the question was:

> **With the schedule frozen at the experiment_5 winner, which score DESIGN beats the μ floor — and can
> a γ trust region remove the divergence that schedules could not?**

**Frozen:** the exp5 winning schedule (`lr_mu 0.1669`, `mu_anneal 0.3455`, `lr_gamma 0.4716`,
`gamma_anneal 0.4723`); 17g model; 2-D `(FWHM, σ_fit)` KDE with Scott bandwidths; Lorentzian 2-param fit;
z-form γ-score; Anuar's μ reward (`loglik_mean`) with `σ_prop` denominator; **no clipping, no clamps**;
divergence guard; init `0.5 × truth`; `n_runs=100`, `n_iter=30`.

**Searched (4 knobs)** — the pre-authorized structural change (configurable σ-channel + γ trust region):

| knob | range | meaning |
|---|---|---|
| `sigma_weight` | 0.0 – 1.0 | damping of the σ_fit KDE channel (`W = exp(−½(dF/H_F)² − ½·sw·(dS/H_S)²)`); 1 = frozen exp5, 0 = FWHM-only control. **Report fix #1.** |
| `h_f_scale` | 0.25 – 4.0 | multiplier on the FWHM Scott bandwidth `H_F` |
| `h_s_scale` | 0.25 – 4.0 | multiplier on the σ Scott bandwidth `H_S` (floored at `h_s_min=0.05`) |
| `gamma_rel_cap` | 0.0 – 0.5 | per-step **relative** γ trust region (`|dγ| ≤ cap·γ`, 0 = off). **exp5 §6.1 recommendation.** |

**Objective** (pre-registered): T-weighted mean of CAP-capped per-cell rel-sq errors
(`err_i = min(rel_sq_i, 1.0)`, `w(T) = 0.25 + 0.75·T/100`). Benchmark = **8 experiments**
(1nW/3nW × Trans05/20/60/100). Cap `MAX_TRIALS=30`.

**Success bar:** γ finite on all 8 cells AND objective below the exp5 honest level (~0.108–0.113),
with μ rel-RMSE/bias improved and fewer/zero diverged cells.

---

## 2. How the campaign ran

- **30/30 trials**, each executed **cell-by-cell** (`nbrun.py`, persistent kernel), each registered in
  `trials.json` and **committed + pushed** before the next. Median ~7.6 min/trial.
- First **5 trials uniform exploration**; from trial 06 **model-informed** proposals (TPE good/bad
  densities + reserved explore slots). The agent picked one candidate per trial and wrote the analysis.
- Every trial notebook is self-contained (phase-space path figure + parallel-coordinates view + report table).

---

## 3. Headline result

**Best: `trial_07` — objective 0.0803 ± 0.0239**

```
sigma_weight = 0.865   h_f_scale = 4.0   h_s_scale = 3.675   gamma_rel_cap = 0.107
μ rel-RMSE 35.9 %   γ rel-RMSE 22.3 %   0 diverged cells
```

**This clears the exp5 plateau by ~0.028 (≈26 %).** The whole campaign sits well under the exp5 honest
level: distribution over the 30 trials is **median 0.0865**, mean 0.0907, range 0.0803 – 0.1287;
**22/30 ≤ 0.089**, and every single trial beats exp5's honest level of ~0.108–0.113.

Near-ties at the top: `trial_26` 0.0807 (sw 0.919, h_f 4.0, h_s 3.608), `trial_20` 0.0821
(sw 0.922, h_f 3.90, h_s 3.615), `trial_05` 0.0822. The optimum is a **wide plateau**, not a spike.

### Two effects, together, did the work

**(i) The damped σ-channel fixed the μ floor.** μ rel-RMSE dropped from **77.9 %** (exp5 best) to
**35.9 %** — the first real dent in FM#8. The winning `sigma_weight` is **interior, ~0.85–0.92**:
full-strength (unchanged) and heavily damped both lose (see §4). This confirms the report's #1 fix:
*don't remove the σ channel — down-weight it and flatten its kernel*.

**(ii) The flat kernels set the scale.** `h_f_scale` is **the dominant knob (r = −0.948)** and the optimum
is pinned at the **upper bound 4.0** (17/30 trials sit exactly there); `h_s_scale` follows at **r = −0.809**
with the optimum clustered at **3.6–3.8**. Scott's rule is **too small on real data** — the likelihood
needs a flatter kernel.

**(iii) The γ trust region removed the divergence entirely.** **0/30 diverged**, versus **9/40 in exp5**
(all at 1nW T05 or 3nW T60). The γ rel-cap is the exp5 §6.1 fix working exactly as intended: it bounds the
chaotic low-count γ step **without touching μ**, so it does not revive the 21i clip-masking problem.

### Per-cell behaviour at the best trial (trial_07)

```
              1nW T05  T20    T60    T100  | 3nW T05   T20    T60    T100
μ   Δ%         -35.4  +60.0  -28.5   -27.0 | +21.3    -5.1  -39.0   -43.9
γ   Δ%         -25.4  -36.1  -13.6    -6.7 | -39.0   -17.1   -0.6    -2.0
```

The shape is mild now (no cell runs away), but systematic: **μ overshoots in two cells, undershoots in
five** (the high-T cells are still low), and **γ under-recovers almost everywhere** — γ is still the
"too-small" direction (FM#6 suspicion), though far less extreme than exp5's μ picture.

---

## 4. The structure of the improvement

**Correlation of the objective with each knob (30 trials):**

| knob | corr(obj) | reading |
|---|---|---|
| `h_f_scale` | **−0.948** | dominant; monotone toward the bound → **the bound is the limit** |
| `h_s_scale` | **−0.809** | strong; optimum ~3.6 → also near the bound |
| `sigma_weight` | −0.141 | **non-monotone**: interior optimum ~0.85–0.92 (both extremes lose) |
| `gamma_rel_cap` | −0.019 | value doesn't matter — but any cap > 0 is what keeps γ finite |

- **`sigma_weight` is genuinely interior.** The best low-sw trials (trial_16 sw 0.512 → 0.0850;
  trial_06 sw 0.497 → 0.0861) are good but not best; sw = 0.41 (trial_01, 0.1243) and sw = 0.88 with a
  *bad* kernel (trial_02, 0.1287) are the two worst. The optimum keeps most of the σ channel
  (sw ≈ 0.87) and instead **flattens it** (large `h_s`).
- **The bound is binding.** Since 17/30 trials sit at `h_f = 4.0`, the campaign's answer is literally
  "flatter than the space allows". **This is the single clearest next lever.**
- **The objective is chaotically sensitive.** Near-identical configs scatter by ~0.008–0.010:
  trial_07 (sw 0.865, cap 0.107) → 0.0803 vs trial_23 (sw 0.884, cap 0.114) → 0.0892, and
  trial_17 (sw 0.883, cap 0.096) → 0.0859. The objective is deterministic but **chaotically sensitive**
  (<10⁻³ knob changes can shift it ~0.01 through photon-count rounding), so the plateau's internal
  ordering is partly a draw.

---

## 5. How well did the AG-HYPOPT idea work?

**Better than in exp5 — the problem was better-posed.**

**What it did well**
- **It found the win fast.** Best-so-far: `0.1040` (t03) → **`0.0822` (t05)** — the productive region
  (`sw ≈ 0.85`, `h_f ≈ 4`, `h_s ≈ 3.6`) was identified in **5 trials (~40 min)**, and t07 improved on it.
- **It separated a dominant knob from three secondary ones** with only 30 trials: `h_f` at −0.95 and the
  bound pinned is a clean, actionable read, not a plateau.
- **The agent's judgement mattered.** The uniform phase probed both the low-sw and low-bandwidth corners
  (t01, t02) precisely to bracket the optimum, and from t05 on it held the plateau instead of wandering.
- **The registry transfers**: 30 configs × objective × uncertainty × written analysis, reproducible from
  `trials.json`.

**What limited it**
- **The search box was too small in the one direction that mattered.** `h_f` (and `h_s`) hit the ceiling;
  ~57 % of trials were inside a region the model had already mapped. Extending the bound is the whole story.
- **Chaotic sensitivity caps the resolvable precision.** With a ~0.01 draw-to-draw scatter, the plateau's
  internal ordering (0.0803 vs 0.0807 vs 0.0821) is not meaningful — a stop rule on the objective's own
  uncertainty would have stopped near t10.
- **The γ cap value is unidentifiable** here (corr −0.02): its *existence* is the finding, not its value.

**Verdict:** the loop again behaved exactly as designed, and this time the campaign produced a **real
mechanism win** (μ floor halved, divergence eliminated) plus a **clear next direction** (the bandwidth
bound). The method was not the limit; the box was.

**Recommendations for the next campaign**
1. **Extend the bandwidth bounds** (`h_f`, `h_s` → ~[1, 12]) and search the bandwidth **RULE**, not just a
   multiplier (Scott-from-target vs sim-adaptive vs per-power) — the note is that Scott-from-target is
   itself the misspecification (kernel over the sim cloud, bandwidth from the target).
2. **Keep the γ rel-cap on** (any positive value) — it is free and removes the hazard.
3. **Validate on the 6 held-out experiments**, not just the 8-exp benchmark (guards against tuning the
   bandwidth to the benchmark).
4. **Freeze the exp6 winner** as the new default: `sw 0.865`, `h_f 4.0`, `h_s 3.675`, `cap 0.107`.

---

## 6. Where the series goes next (→ experiment_7)

1. **Bandwidth, properly.** Freeze the exp6 winner and search the **bandwidth RULE family × coefficient**
   with `h_f_scale`/`h_s_scale` extended to ~[1, 12]. The physics reason: bandwidth is a kernel over the
   **sim cloud** but is computed from the **target** — structurally mismatched whenever the sim/target
   spreads differ (always, FM#6/#8). A **flat** kernel makes the responsibilities uniform and **kills the
   REINFORCE μ channel** (a bandwidth mechanism for the dead count channel); a knife-edge kernel makes it
   spiky. exp6 answered "flatter is better" but ran out of room — exp7 finds where it turns over.
2. **Held-out validation** on the 6 non-benchmark experiments (T10/40/80 × 1nW/3nW) for generalization.
3. **One H for both optimizer and UQ** (coherence), chosen by a truth-independent criterion (SBC /
   bootstrap agreement), never tuned against the answer — the open methodological question from 19 Sep.

---

## 7. Provenance

- **Experiment:** `agent-guided-hypopt/experiment_6/` — `ag_hypopt.py`, `space.json`, `trials.json`
  (protocol + 30 entries), `trial_01.ipynb … trial_30.ipynb` (each self-contained with figures), `README.md`.
- **Best trial:** `trial_07.ipynb` (0.0803); runner-up `trial_26.ipynb` (0.0807).
- **Repo state:** branch `develop`, campaign closed at commit `95331a9` (trial_30).
- **Series context:** series-19 autopsy (FM#8) · experiment_5 (schedule plateau, floor 0.108–0.113) ·
  this campaign (score design: **μ floor halved 77.9 → 35.9 %, divergence 9/40 → 0/30**, optimum pinned at
  the bandwidth bound).
