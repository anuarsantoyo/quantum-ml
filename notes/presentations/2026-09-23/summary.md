# What we did since the 11.09.2026 presentation — a chronological summary

_Written 2026-09-22 for the 2026-09-23 presentation. Sources: the project git history
(`develop`) and the daily memory notes (`memory/2026-09-1*.md`, `2026-09-2*.md`)._

**How to read this.** The work since 11.09 ran along **four parallel threads** that sometimes look
scattered but are actually one story:

- **A — the μ channel** (series 20/21 + AG-HYPOPT exp5–exp8): why does μ never land on truth on real
  data, and what can be fixed by score/reward/schedule/loss design?
- **B — understanding the distributions** (series 22a–22e): what does the simulator actually generate,
  and how does it differ from the real data? (This produced the central diagnosis, **FM#8**.)
- **C — re-presenting the results** (22f/22g, this week): the (μ,γ)→distribution map, made explorable.
- **D — uncertainties** (19c/19d, exp5 `fisher_analysis`, and now **series 23**): the second observable,
  σ_fit, and the Fisher/Cramér–Rao machinery — **the current frontier**.

The through-line: **the point estimate is limited by a model/data mismatch (not by tuning), and the
uncertainty is the part we have not yet made honest.**

**The causal spine (one paragraph).** `experiment_4` on real data showed μ is **knob-insensitive** →
*stop tuning, go structural* → **series 20** (one change per notebook) found and fixed **FM#10** but ended
in two no-improvements → the suspicion moved to the **estimator** (21b/21d: lmfit refuted) and then to the
**μ reward** (21e–21h) → **21i** removed the clip and μ finally descended (550 % → 40.6 %), but exposed a
**γ divergence** → guard + schedule question → **exp5** (plateau 0.108–0.113, untunable divergence) →
“the residual is not the schedule” → **two branches**: (a) *understand the distributions* → 22a (reward
refuted) → 22b (sim ≈ real at mid/high T) → 22c (spread = estimator; σ_fit wrong for **both**) → 22d (the
σ channel is μ’s only live gradient) → 22e (the (μ,γ) map); and (b) *attack the σ channel in the score* →
**exp6** (μ floor halved, divergences gone) → its own correlation table pointed at the **bandwidth** →
**exp7** (dead end) → **exp8** (the loss *form*: quantile losses win ~10 %, but they are **not
likelihoods**) → “then which model can carry a Fisher?” → KDE only → project survey → best KDE model =
**exp6 `trial_07`** → Anuar’s scaling question exposes the **`Σ/N` bug** + the **bandwidth confounder**
→ **series 23 (`23a`)**.

---

## 0. Where we stood on 11.09.2026 (the last presentation)

Deck: `notes/presentations/2026-09-11/presentation-2026-09-11.md` (38 figures).

- The differentiable Monte-Carlo pipeline (PyTorch) was working end-to-end: μ via **REINFORCE**, γ via
  **implicit differentiation** through the inner L-BFGS Lorentzian fit; the 2-D `(FWHM, σ_fit)` **KDE**
  likelihood as the loss; **Fisher/CRB** as the uncertainty statement.
- **On synthetic data it recovers the truth almost exactly** (17g: μ 3.7 %, γ 4.3 %) → the machinery is
  not the problem.
- **On real data it is far off**: μ lands at ~0.5× truth (the “μ attractor”), γ OK at high T but low at
  1nW low T.
- Series 19 had just concluded the **honest-uncertainty round** (19a–19d):
  - 19a: Voigt broadening does **not** explain the μ attractor (rejected).
  - 19b: the attractor comes from the **σ_fit channel** (count channel dead; sim σ_fit 1.04–4.25× too
    narrow, growing with T).
  - 19c: matching the σ **scale** does **not** move the mode → the mismatch is the **shape of σ(μ)**.
  - 19d: **D1 (σ_ref = σ_prop, c = 1) is the honest answer** — μ truth inside 2σ on **14/14**, σ_μ grows
    with T (8.5 → 90.7 MHz), γ RMSE 3.23 MHz (31 %). “Wide but true.”
- **AG-HYPOPT** (agent-guided hyperparameter optimisation) was introduced as the tool for the next phase,
  with `experiment_3_pukky` validated as the working template.

**Two declared goals for the phase after 11.09:** (1) get all 14 experiments to land at their true
(μ, γ); (2) **compute uncertainties** (Fisher/CRB). The second is the one we are picking up now.

> **→ The chain starts here.** The AG-HYPOPT tool existed (the `e3_pukky` template) but had only been
> point-estimated on *synthetic* data. So the obvious next step was to point the same machinery at the
> **real** data — `experiment_4`. Everything after that is a reaction to what that run showed.

---

## 1. 11.09 — AG-HYPOPT: template validated (synthetic) + first real-data campaign

- **`experiment_3_pukky` (synthetic, 20/20 trials)** — ran the campaign to the cap via a background
  subagent, cell-by-cell (`nbrun.py`), one commit+push per trial. **Best `trial_14`, objective 0.0149**
  (μ 14.4 %, γ 9.5 %); optimum `sigma_ref ≈ 6.1, lr_mu ≈ 27, lr_gamma ≈ 0.42, gamma_anneal ≈ 0.42`;
  residual spread driven entirely by the low-count Trans05 γ. → **the template works.**
- `experiment_template` rebuilt as a clean snapshot of e3_pukky; added the **honest-init marker**
  (hollow circle at exactly 0.5×true) and **per-trial proposal seeding** (`trial_seed(TRIAL_ID)`) to kill
  the “repeating batch” bug.
- **`experiment_4` — AG-HYPOPT on REAL data** (20/20 trials, ~67 min). **Best `trial_20`, objective
  0.1637.** Scientific read: **μ_fit is stuck at a uniform ~50 % of μ_true in all 20 trials and is
  knob-insensitive** → the real-data μ attractor; all tuning effort went into γ. → real-data HPO is
  **knob-limited**: it needs a **structural** fix, not tuning.
- Also: `notes/ag-hypopt-novelty-assessment.md` (arXiv prior-art map — AG-HYPOPT is “a nice tool, thin
  and incremental novelty”; the interesting part is the LLM-as-selector).

> **→ What that gave us.** `experiment_4` showed **μ is knob-insensitive** — stuck at a uniform 0.5× in
> all 20 trials, so *no* schedule/hyperparameter moved it. That is what led to the conclusion “real-data
> HPO is knob-limited: stop tuning and go **structural**” — i.e. the series 20 / 21 line of work, one
> architectural change per notebook.

## 2. 12–13.09 — quiet days (jobs cron + a side project)

- Job-scan cron only (search backends mostly bot-walled; workaround found: SearXNG mirrors, then later
  INSPIRE-HEP API / academics.de / Euraxess facets).
- 13.09: the **bat-pregnancy** side project was onboarded (separate repo; not part of qm-ml).

## 3. 15.09 — repo hygiene + series 20 (real-data agentic optimisation) + series 21 opens

- **Notebook reorganisation** (`7b57903`): notebooks grouped into order-preserving phase folders
  (`01-06_foundations` … `19_identifiability`), 12 legacy `-executed` twins archived, path bootstraps
  fixed. **Nothing deleted.**
- **Series 20 (`20a`–`20g`)** — the 17g model on real data, one change per notebook:
  | nb | change | result |
  |---|---|---|
  | 20a | 17g model on real data (baseline transfer) | μ **init-locked at 0.5×**, γ OK at high T (μ rel-RMSE 52.3 %, γ 33.8 %) |
  | 20b | count-calibrated μ score (`σ_ref = σ_prop`) | μ escapes the init-lock at low T; a new blow-up (**FM#10**) appears |
  | 20c | implicit-diff regularisation 1e-4 → 1e-3 | FM#10 reduced (1.3e16 → 1.4e11) but not fixed |
  | 20d | **clip the implicit derivatives** (`DERIV_CLIP=20`) | **FM#10 closed**, metric now meaningful |
  | 20e | per-scan linewidth heterogeneity (FM#6) | μ ratios rise, γ does not improve (no improvement) |
  | 20f | γ LR 0.5 → 1.0 | high-T γ converges; no overall improvement |
  | 20g | revert heterogeneity | no improvement → **STOP** + `SERIES_STATE.md` final summary |
- **Series 21 opened**: `21a` = clean copy of 20a; `21b` = pseudo-Voigt **fit** (Lorentzian sampling);
  `21c` = full diagnostics; `21d` = **ours vs Gregor's lmfit** → **lmfit refuted as a forward model**
  (it does not reproduce the real observables; its gradient is not a usable surrogate). Conclusion there:
  the μ failure is the **simulator model gap**, not the fitter.

> **→ What that gave us.** 20g ended with two consecutive “no improvement” → **stop changing the
> architecture**. With the architecture frozen and the fitter soon refuted (21d), the only place left to
> suspect was the **μ reward/gradient** → that is what produced the 21e–21i run the following day.

## 4. 16.09 — the μ-channel diagnosis: **the clip was the bug**

The day’s arc, one notebook per hypothesis (all real data, 14 exps, 100×30):

| nb | hypothesis | verdict |
|---|---|---|
| 21e | redesign the μ reward: `responsibility` / `loglik` / `corr` / `loglik_z` | **no fix** — raw loglik runs away (saturates CLIP); `corr` = control; `loglik_z` doesn’t land |
| 21f | Anuar’s reward only (`r_j = mean_i log W_ij`) | **μ runaway** to the 200 clamp (rel-RMSE 529.5 %, bias +372.6 %); **γ improves** (18.6 %) |
| 21g | same but **SUM** log-likelihood (no mean) | same runaway, worse (770.9 %) — it is only a scale factor |
| 21h | μ score denominator `SIGMA_REF=10 → σ_prop` | mild (550.6 %) — the clip still dominates |
| **21i** | **NO clip + NO clamps + LR_MU = 1e-4** (rescaled to the measured raw gradient) | **BREAKTHROUGH: μ rel-RMSE 550.6 % → 40.6 %, bias +366 % → +7.3 %**; 1/14 diverged (γ at 1nW T05) |
| 21j/21k | N_RUNS sweep + estimator-noise study | n_runs is not the lever (μ residual is **bias**, not variance); estimator noise ∝ 1/√M |

**The key insight:** the gradient **clip** was masking a **scale** problem — when `|grad| > CLIP` the step
is `LR·CLIP`, so the gradient *signal* is discarded. Removing the clip (and rescaling the LR) gave the
first correct-direction μ descent on real data. It also revealed that the clip had been protecting **γ**
at low T.

Also this day: the **exp5 protocol** was frozen (4-dim schedule space, `MU_REWARD=loglik_mean`,
`MU_SCORE=σ_prop`, `CLIP=inf`, divergence **guard**, T-weighted **capped** objective, 8-exp benchmark).

> **→ What that gave us.** With the clip gone μ finally descended in the right direction (40.6 %) — but a
> **1/14 γ divergence at 1nW T05** appeared (the clip had also been protecting γ). Two consequences: the
> **divergence guard** was written into the protocol, and the next question became “can a better
> *schedule* build on 21i?” → **experiment_5, launched that same night.**

## 5. 17.09 — exp5 campaign (schedule) + the first Fisher analysis

- **`experiment_5` — 40/40 trials, 6 h.** **Best `trial_21` = 0.1074**; honest level ≈ **0.108–0.113**.
  - The schedule optimum is a **narrow ridge** (`lr_gamma ≈ 0.41–0.48`, `gamma_anneal ≈ 0.47–0.53`).
  - **Dominant residual: a sporadic 1nW T05 γ divergence** (~1 in 3 trials near the optimum; costs
    0.02–0.07) — **not tunable** by the 4 knobs (a near-copy of trial_21 diverged identically).
  - CONCLUSION written (`aa1cc29`).
- **`fisher_analysis.ipynb`** at the best trial_21 config, full 14 exps → **CRB verdict:
  μ honest-but-uninformative (14/14 inside 2σ, but huge σ), γ tight-and-wrong (6/14)**; the degeneracy
  peaks at low T. This is the first time the uncertainty was measured at a campaign optimum.

> **→ What that gave us.** exp5 hit a **plateau** (0.108–0.113) plus a **sporadic, untunable 1nW T05 γ
> divergence** → the residual is *not* the schedule. That produced two independent follow-ups: (i)
> **understand what the simulator actually generates** (→ series 22 the next day), and (ii) **attack the
> σ_fit channel at the score level** (→ experiment_6). And the `fisher_analysis` result (μ
> honest-but-uninformative 14/14, γ tight-and-wrong 6/14) is what eventually motivated today’s series 23.

## 6. 18.09 — series 22 opens: what does the simulator actually generate?

- **`22a`** — z-scored (adaptive-LR) reward → **REFUTED**: μ rel-RMSE 132.5 % (worse than 21i’s 40.6 %).
  γ 18.6 %, 0 divergences. → the residual is **structural**, not reward scale. *Value = the machinery.*
- **`22b`** — Monte-Carlo **at truth** (N_MC = 1000, 14 exps): MC precision is ample (mean SEM 0.1–0.6 MHz);
  **sim ≈ real at mid/high T** (both ≈ 2γ_true); the gap is at **low T** (heavy-tailed σ_fit).
- **`22c`** — **Gregor’s lmfit estimator at truth**: it **reproduces the real FWHM *spread*** (within
  3–10 % at mid/high T, where our unbinned MLE is ~2× too narrow) but it **overshoots σ_fit 3–5×**;
  its FWHM centre is low and pinned at the 6 MHz bound at low T.
- Established the **interactive-Plotly conventions** (renderer `vscode`, dropdown = experiment via
  `method='animate'`, ▶/❚❚ + slider, frames update a prefix of the traces) — used by all of series 22.

> **→ What that gave us.** 22c was the pivot: it showed the FWHM **spread** is an **estimator** effect
> (our unbinned MLE is too precise, lmfit is about right) while **σ_fit is wrong for both** estimators.
> That is what led to (i) “then let us remove the σ channel and see” → **22d**, and (ii) “stop arguing
> about it, map (μ,γ)→distribution directly” → **22e**.

## 7. 19.09 — 22d/22e + the bandwidth–Fisher discussion + exp6

- **`22d`** — 1-D **FWHM-only** loss → **REFUTED** (μ rel-RMSE 342 %): the σ_fit channel is μ’s **only
  live gradient** (FM#8); dropping it hands μ a spurious low-count runaway.
- **`22e`** — the **(μ, γ) → distribution grid** explorer: 13 μ × 10 γ nodes, 3 estimators, N_MC = 2000,
  pinned noise condition (1nW T60); ~8.5 h precompute; companion data `data/processed/22e_clouds.npz`.
- **Late-night discussion (bandwidth & Fisher):** Scott bandwidths computed **from the target** but used
  as a **kernel over the sim cloud** are structurally mismatched; a flat kernel kills the REINFORCE μ
  channel; for a **misspecified** surrogate the honest interval is the **sandwich** `J⁻¹KJ⁻¹`, not the
  plain inverse Fisher; the same H must serve optimizer and UQ (coherence), and H must **never** be tuned
  against the answer.
- **`experiment_6` designed & launched** (score design; schedule frozen).

> **→ What that gave us.** The discussion’s conclusion — “bandwidth is the biggest lever, but tune the
> **rule** not the value, and never against the answer” — is what shaped **experiment_6** the next day.
> It also flagged that the **same H** has to serve the optimizer *and* the uncertainty; that suspicion is
> what later turned into the bandwidth confounder in series 23.

## 8. 20.09 — exp6 / exp7 / exp8: three campaigns, one conclusion

- **exp6 (score design) — CONCLUSION (`a50b2e1`)**: **best `trial_07` = 0.0803** (`sigma_weight 0.865`,
  `h_f_scale 4.0`, `h_s_scale 3.675`, `gamma_rel_cap 0.107`); **μ floor halved 77.9 → 35.9 %**;
  **0/30 divergences** (exp5: 9/40). Three effects: (i) **damped σ-channel**; (ii) **flat kernels**
  (`corr(obj, h_f) = −0.948`, 17/30 pinned at the `h_f = 4.0` bound → Scott is too small on real data);
  (iii) the **γ trust region** removed the divergence (its *value* is irrelevant). Optimum is a **wide
  plateau**; the objective is **chaotically sensitive** (~±0.01 for <1e-3 knob changes).
  > **→ Next:** exp6’s correlation table said `h_f` is dominant and **pinned at its 4.0 bound** → “the
  > answer is flatter than the box allows” is what gave us **exp7** (extend the box + change the rule).
- **exp7 (bandwidth RULE family) — CONCLUSION (`d23ff32`)**: **no win.** Best campaign trial 0.0818 vs the
  frozen exp6 default 0.0803 (Δ inside the chaos band). The **rule** is first-order (`target` ≈ 0.082 ≫
  `power` 0.154 > `sim` 0.125, confirmed held-out) but the **coefficient is flat** (`corr(obj, h_f) = −0.21`,
  was −0.948) → **exp6’s h_f = 4.0 bound was NOT binding. Bandwidth is a dead end here.**
  > **→ Next:** schedule (exp5) and bandwidth (exp7) both exhausted → the only knob left was the
  > **loss form itself** → **exp8**.
- **exp8 (loss FUNCTION family) — CONCLUSION (`95ea64a`)**: **a win.** **Best `trial_25` = 0.0725**
  (`cvm_fwhm`, `sigma_weight 0.197`) — the only family under the KDE control (0.0803), reproducible
  (20 draws: mean 0.0779, 18/20 ≤ control) and it **holds out-of-sample** (held-out 0.0736 vs control
  0.0822; `w1_2d` @ sw 0.209 even better at 0.0691). Gain is in **μ** (31.5 % vs 35.9 %); the σ channel is
  again the culprit. Two implementation fixes were needed: **seed-notebook stamping** (never stamp the
  generator cell 9) and a **per-step μ-reward rescale** to the control’s spread (non-KDE per-run losses
  span ~3 decades).
- Also: a **cron-store wipe** was discovered and the daily jobs cron restored; a 15-series-style sweep of
  exp8’s settings.
- ⚠️ **Caveat that matters for series 23:** `cvm_fwhm`/`w1_2d` are **discrepancies, not likelihoods** →
  **no Fisher/CRB is defined for them.** The KDE is the only Fisher-compatible model.

> **→ What that gave us.** The realisation that the winning loss **has no likelihood** is what forced the
> question “*which* model can we even compute a Fisher on?” → restrict to **KDE** → the whole-project
> survey → and, with Anuar’s scaling question (below), **series 23**. So exp8’s win and the uncertainty
> thread are directly linked: we had to give up the best point-estimate loss to do UQ at all.

## 9. 21.09 — (outside qm-ml) daily job scan only.

## 10. 22.09 (today) — from the (μ,γ) map to the uncertainty frontier

**Thread C — re-presenting 22e’s map as usable figures (22f), then a corrected re-derivation (22g):**

- **22f** (`22f-mu-gamma-grid-contours.ipynb`, + 2 HTMLs): one big **13×10 contour figure** rendered purely
  from 22e’s npz (no re-run), with **legend-group** layer switching (`legend.groupclick='togglegroup'`
  toggles a layer in **all 130 panels** at once — fixes the “only one panel changes” bug), plus HTML export
  for browser zoom. Iterated live with Anuar: clearer **μ/γ labelling**; real data drawn **only at its own
  (μ, γ) node**; a **second diagram** (2 columns power × 7 rows transparency) for the real-data nodes;
  **4 density bands**; the **real cloud’s own contours** (dashed) for a fair band-vs-band comparison;
  and several stacking iterations.
- **Key plotly lesson:** plotly.js draws traces in **fixed layers**
  (`… heatmaplayer, contourlayer, …, scatterlayer`), so a `go.Contour` is **always below** scatter points.
  To draw contours **over** the data you must render them as **scatter polygons** (marching-squares loops
  with `fill='toself'`). Also: no widgets in exported HTML — plotly-native legend groups / frames only.

> **→ What that gave us.** Building the map this way is what let Anuar *look* at it — and looking at it
> produced his question “**why does the Lorentzian fit look best, when Gregor’s estimator is the same
> process as the data?**” Answering that is exactly **22g**, and 22g is what surfaced the two confounds
> (the pinned noise condition in 22e, and the three different meanings of σ_fit) that led into the
> uncertainty work.
- **22g** (`22g-mc-at-truth-3-estimators.ipynb` + HTML + `data/processed/22g_clouds.npz`): Monte-Carlo at
  **each experiment’s own true** `(μ, γ, σ_prop, λ)` for all three estimators (ours-Lorentzian,
  ours-pseudo-Voigt, Gregor-lmfit) on the **same** photons. **Answered Anuar’s “why does Lorentzian look
  best?” question:** (a) 22e had **pinned** the noise condition to 1nW T60, so 13/14 experiments were
  compared against the wrong count noise (σ_prop/μ off by 0.22–4.5×); (b) the σ_fit axis mixes **three
  different quantities** (our CRLB, lmfit’s covariance stderr, the real pipeline’s fit error). Result at
  the true values (clean high-T subset): **FWHM centre → Lorentzian best (7.3 %)**; **FWHM spread →
  pseudo-Voigt 13.8 % < Gregor 20.4 % ≪ Lorentzian 45 %**; **σ_fit → pseudo-Voigt 28 % < Lorentzian 62 %
  ≪ Gregor 250 %**. → **the matched estimator does not reproduce the real cloud, and is worst on σ_fit.**
  New lead: our lmfit reproduction gives per-fit errors ~3.5× larger than the real recorded values.

**Thread D — the whole-project survey and the uncertainty frontier:**

- Full survey of the project (report, journal, series 12→22, AG-HYPOPT) to identify **the best model we
  have trained on the real data** → **`experiment_6 trial_07`** (KDE; μ 35.9 %, γ 22.3 %, 0/30 divergences,
  held-out 0.0822) — the best **KDE-based** model, which is the only kind a Fisher analysis can use.
  Runners-up: exp8 `trial_25` (best overall but not a likelihood), 19d D1 (the only **calibrated**
  uncertainty: 14/14 inside 2σ).
- **The Fisher normalisation finding (today):** every Fisher block in the project
  (`12d`, `15/16-series`, `17h`, `19b/c/d`, `20a–g`, `21a–j`, `22a/d`, exp5 `fisher_analysis`) computes
  `J = Σ_i s_i s_iᵀ / N` — i.e. the **per-data-point average**, whose inverse is the **single-scan CRB**,
  which is **independent of the number of points**. The **dataset** CRB is `σ/√N`. The scores themselves
  are legitimate (the μ-score is the exact REINFORCE score of the KDE likelihood; the γ-score uses the raw
  likelihood chain), so the fix is one line — but it will **over-shrink** σ, and two structural
  confounders remain: the **inflated bandwidth** of the winner (`h_f × 4`, `h_s × 3.7` — σ grows with H)
  and **FM#8**.
- **Series 23 opened**: `notebooks/23_fisher_uncertainties/23a-fisher-uncertainties-exp6-trial07.ipynb` —
  the frozen exp6 `trial_07` model on **all 14** real experiments + the CRB in **three variants**:
  (A) the existing per-scan convention, (B) the dataset CRB `σ/√N`, (C) the dataset CRB at the **Scott
  baseline** bandwidth. Plus 22a-style visuals with the uncertainty report **split by power (1nW | 3nW)**.
  Smoke preview (1nW T60): σ_μ 89.4 → 1.82 → 0.37 for A/B/C, with ‖Δμ‖/σ 0.34 → 16.9 → 83 — i.e. the
  correction **breaks coverage**, and the Scott baseline shows the winner’s σ is bandwidth-dominated.
  _Status at the time of writing: full 14-experiment run in progress._

> **→ What that gave us (and where we are).** The 22f/22g detour is not a side quest: it exposed that the
> **uncertainty axis (σ_fit) is the weak link**, which is the same conclusion FM#8 had reached from the
> other side. That, plus Anuar’s question “**shouldn’t more data points increase the Fisher information?**”,
> is what produced the discovery of the `Σ/N` normalisation bug and — directly from that question — the
> **three-variant design of 23a (per-scan / `σ·N^(−1/2)` / Scott-baseline)**. So the sequence of the last
> ten days closes into a loop: **the distributions (22) told us the σ channel is broken, and the Fisher
> study (23) is where we now measure that break honestly.**

---

## 11. The four threads, in one picture

| thread | series | question | where it stands |
|---|---|---|---|
| **A — μ channel** | 20, 21, exp5–8 | why is μ stuck, and what fixes it? | **answered for the mechanism**: the clip was the bug (21i); schedule plateau (exp5); score design halves the μ floor (exp6); bandwidth is a dead end (exp7); robust quantile losses win ~10 % (exp8) — but the residual is the **model/data gap (FM#8/FM#6)** |
| **B — distributions** | 22a–e | what does the simulator generate vs the real data? | **FM#8 diagnosed**: sim σ_fit 1.04–4.25× too narrow; the *shape* of σ(μ) is the problem, not its scale (19c); 22g quantified the 3-estimator mismatch |
| **C — re-presentation** | 22f/22g | make the (μ,γ)→distribution map explorable and honest | done: 2 diagrams + 3 HTMLs, legend-group switching, real-cloud contours; plus the pinned-noise confound removed in 22g |
| **D — uncertainties** | 19c/19d, exp5 fisher, **23** | are our error bars honest? | 19d D1 is calibrated-but-wide (14/14 in 2σ); exp5 fisher: γ tight-and-wrong; **the Fisher has an N-normalisation bug + a bandwidth confounder → series 23 is the fix** |

## 12. What is settled vs open (so we do not walk in circles)

**Settled (do not re-litigate):**
- The **clip** was masking a scale problem → removed since 21i. (Any future clip is a red flag.)
- **Schedule tuning** (exp5) and **bandwidth tuning** (exp7) are exhausted; **reward redesigns** (21e–21h,
  22a, 22d) do not move μ.
- **Gregor’s lmfit** is not a better forward model (21d, 22c, 22g — it overshoots σ_fit).
- The residual is a **model/data mismatch**: sim σ_fit too narrow (FM#8) + the 1nW low-T γ deficit (FM#6).
- **KDE is the only Fisher-compatible loss**; exp8’s winners cannot carry a CRB.
- **D1 (19d)** is our only calibrated uncertainty statement; **exp6 trial_07** is our best KDE model.

**Open / next:**
1. **Series 23**: make the Fisher honest — fix the `1/√N` normalisation, separate the **bandwidth**
   contribution (Scott baseline), and referee everything against the **empirical data bootstrap**
   (resample the real scans). Add the **sandwich** interval for the misspecified surrogate.
2. **σ(μ) shape matching** (19d rec #2) — the one lever that has not been tried on the loss side.
3. **1nW low-T γ** (FM#6) — a γ-specific mechanism (γ-only guard/damping), not a global clip.
4. **A differentiable forward model** (“Option C”) if we want the point estimate to land at truth.
5. Adopt **cvm_fwhm / w1_2d (low sw)** as the default *point-estimate* design, and keep the **KDE** for the
   *uncertainty* statement — two roles, two losses, stated explicitly.

## 13. Process lessons (why it felt spread out)

- Campaigns ran as **isolated subagents** (cell-by-cell `nbrun.py`, commit+push per trial). Fast, but the
  results land as 30–40 commits; the **CONCLUSION.md** per campaign is what makes them readable — write it
  always.
- **Finalize crons are unreliable** for unattended work (main-session `systemEvent` wakes get lost); prefer
  isolated `agentTurn` crons or stay in-session and poll.
- The objective is **chaotically sensitive** (<1e-3 knob changes → ~0.01 objective shift, from integer
  photon rounding). Plateau *ordering* is not meaningful; only coarse trends are.
- Two recurring setup bugs to avoid: **stamping `trial_01` must skip the generator cell (cell 9)**, and any
  new loss family needs **both** a γ-scale match (`FAMILY_SCALE`) **and** a per-step μ-reward rescale.
- Figures: **no PNGs outside notebooks**, notebooks executed **in place** (papermill), companion run data in
  `data/processed/`, and for interactive figures an **HTML export** (a widget never survives HTML).

---

## 14. Index of the artefacts

**Series 20** `notebooks/20_realdata_agentic_optimization/20a…20g.ipynb` (+ `SERIES_STATE.md`).
**Series 21** `notebooks/21_realdata_supervised_optimization/21a…21k.ipynb` (21i = the μ breakthrough).
**Series 22** `notebooks/22_distribution_dynamics/22a…22g.ipynb` (+ `22e_clouds.npz`, `22g_clouds.npz`,
`22f-*.html`).
**Series 23** `notebooks/23_fisher_uncertainties/23a-fisher-uncertainties-exp6-trial07.ipynb`.
**AG-HYPOPT** `agent-guided-hypopt/experiment_{3_pukky,4,5,6,7,8}/` — each with `ag_hypopt.py`, `space.json`,
`trials.json`, `trial_*.ipynb`, `CONCLUSION.md`.
**Reference** `docs/PROJECT_REPORT.md`, `notes/JOURNAL.md`, `notes/ag-hypopt-novelty-assessment.md`,
`data/raw_data/data_explanation.md`.
**Comparators** 17g / experiment_1 baseline (best on synthetic), 19d D1 (calibrated uncertainty),
exp6 trial_07 (best KDE model on real data), exp8 trial_25 (best overall point estimate, Fisher-free).

**Key commits**: `ae...` series-19 · `7b57903` notebooks reorg · `38e9dcb` exp5 protocol · `5b9c3de` exp5
end · `c7bec5e`/`7511a53`/`e2d8f0e` 22a/b/c · `95331a9`/`a50b2e1` exp6 · `237f932`/`d23ff32` exp7 ·
`e8f3d69`/`95ea64a` exp8 · `7a75020`…`55d66ee` 22f/22g · 23a (in progress).
