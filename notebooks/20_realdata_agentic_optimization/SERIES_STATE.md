# SERIES 20 — SERIES_STATE.md

**Real-data agentic optimization. Frozen goal + protocol + log.**
This file is the durable state of the series. Every notebook appends a row to the log.
**Do not change the goal or the metric definitions.** (Adjustable only by Anuar.)

---

## 1. Objective
Start from **17g** — the best model so far (all 14 *synthetic* exps recovered: mu RMSE 3.7%, gamma
RMSE 4.3%) — and, through a chain of **one-change-per-notebook** experiments on the **real** 14-exp
dataset, reach a model that recovers the optimal (mu, gamma) **within a reasonable uncertainty**,
with accuracy **graded by transmission**: loose at low T, tight at high T.

20a = 17g EXACTLY on real targets (baseline transfer test). Each later notebook learns from the
previous one's results.

## 2. Frozen protocol
- **Data:** all 14 real experiments (1nW/3nW x T05..T100); real-target load+filter from 18b/19d
  (`filt = ok & (err/fwhm < 10)`); per-exp table (mu_true, sigma_prop, lam, gamma_true, data_file)
  is in 20a cell 3.
- **Budget per notebook:** `N_RUNS=100`, `N_ITER=30`; Fisher `M_FINAL=500, FISHER_SEEDS=1`.
- **Model freedom:** anything may change (simulator / sigma-model / score / loss / optimizer).
  `src/` edits must be **additive** (new option/flag) so series 17/18/19 stay reproducible.
- **Figures:** fixed panel FIG1..FIG10 (see 20a). Add diagnostics freely; keep the panel.
- **Conventions:** execute in place (`papermill 20x.ipynb 20x.ipynb`); figures inline only
  (no savefig outside the notebook); run history -> `data/processed/20x_history.json`;
  commit+push after every notebook. **No `-executed` copies.**
- **Never edit** `notes/JOURNAL.md`, or any notebook outside `notebooks/20_realdata_agentic_optimization/`.

## 3. Frozen success metric (uncertainty = Fisher/CRB at the fitted point)
Ratio r = rec/true. Graded by transmission:
- **high T (60/80/100):** `|dmu|/sigma_mu <= 2` AND `|dgamma|/sigma_gamma <= 2`; mu ratio in [0.8,1.2];
  gamma rel-RMSE <= 5%.
- **mid T (20/40):** `|d|/sigma <= 3`; ratios >= 0.6.
- **low T (05/10):** coverage only (`|d|/sigma <= 3`); ratios may be loose.
- **overall:** mu rel-RMSE <= 15%, gamma rel-RMSE <= 10%; and accuracy must **improve with T**.

**Primary progress score** (lower = better, computed each notebook):
`S = mean_exps( |dmu|/sigma_mu + |dgamma|/sigma_gamma )`.

## 4. Stop conditions
1. Goal (section 3) met on all 14. 2. **10 notebooks** reached (cap).
3. **2 consecutive notebooks with no improvement in S** -> stop, leave final summary here.

## 5. Anti-overfit
Report tuned-subset (T20/60/100) vs held-out (T05/10/40/80) split of the metric. Wins must hold on held-out.

## 6. Failure-mode tags
FM#6 = Lorentzian line-shape mismatch (1nW low-T gamma deficit). FM#8 = un-damped sigma-channel
gradient collapsing mu. FM#9 = mu init-lock / step too small (count channel dead, REINFORCE step
≈0 → mu stays at 0.5·mu_true). FM#10 = low-count Fisher degeneracy (n~5 photons → dFWHM/dgamma
derivatives blow up → J_gg ~1e28 → sigma_gamma ~1e-14, S blows up).
Tag each notebook: which FM it attacks / fixed / created.

**S-bookkeeping note:** S is the frozen mean of |dmu|/sigma_mu + |dgamma|/sigma_gamma. When a
single experiment's Fisher is degenerate (FM#10) it dominates the mean (e.g. 20a: 6.1e14 of the
4.4e13 mean). Report the official S *plus* a robust companion **S_rob = mean over experiments with
finite, non-pathological Fisher** (drop entries with sigma_gamma < 1e-6 or J_gg > 1e20), for
tracking progress. Official S is what the stop rule watches.

## 7. Log
| # | notebook | one change | S (primary) | gamma rel-RMSE | mu rel-RMSE | S_rob | status |
|---|----------|-----------|-------------|----------------|-------------|-------|--------|
| a | 20a | 17g exactly, real targets (baseline) | 4.37e13 | 33.8% | 52.3% | 19.9 | DONE |
| b | 20b | mu score σ_ref = σ_prop (count-calibrated) | 1.26e16 | 36.8% | 47.7% | 6.9 | DONE |
| c | 20c | implicit-diff reg 1e-4 -> 1e-3 (attack FM#10) | 1.38e11 | 35.0% | 47.3% | 4.9 | DONE |
| d | 20d | clip implicit derivatives at DERIV_CLIP=20 | **4.94** | 34.8% | 47.4% | **4.94** | DONE |
| e | 20e | per-scan FWHM heterogeneity (HET_FRAC=1) | 4.74 | 36.0% | 45.4% | 4.74 | DONE |
| f | 20f | LR_GAMMA 0.5 -> 1.0 (converge high-T γ) | 4.93 | 34.6% | 44.7% | 4.93 | DONE |
| g | 20g | HET_FRAC 1.0 -> 0.0 (revert heterogeneity) | 6.50 | 35.1% | 47.2% | 6.50 | DONE |

### Log notes
- **20a (2026-09-15, 18 min):** μ lands at **0.41–0.55 × μ_true in ALL 14** exps (mu rel-RMSE 52.3%)
  and barely moves off its init (0.5·μ_true) — μ paths are essentially flat (e.g. 1nW T10: 6.19→6.78
  over 30 iters). Diagnosis: **μ init-lock (FM#9)** — the count channel is dead (uniform KDE
  responsibility at N=61–3742) so the REINFORCE μ-gradient ≈ 0; μ never escapes 0.5·truth.
  γ is the healthy channel: within 2σ on all 6 high-T exps (γ rel-RMSE 9.0% there, 5.3% at T100),
  but 3nW T20/T40 are −0.9σ/−3.2σ and 1nW T05/T10 are pathological.
- **FM#10 observed (20a):** 1nW T05 has H_S=212 MHz (huge fit-error scatter) and μ stuck at ~5
  photons → dσ/dγ derivatives blow up → J_gg ≈ 1e28 → σ_γ ≈ 9.4e-15 → the single |dγ|/σ = 6.1e14
  term dominates S. (20a's other terms sum to ≈259 → S_rob 19.9; the "other 12 are small" intuition
  is false — 1nW T10 alone contributes 179.)
- **20b (2026-09-15, 19 min):** ONE change = per-exp `σ_ref = σ_prop` in the μ step (D1 score).
  **Prediction largely HIT:** μ *escaped init* at low T (1nW T05 0.53→0.76, T10 0.55→0.79,
  3nW T05 0.53→0.64) while high-T μ stayed ≈0.47–0.49 — the μ-ratio-vs-T curve **inverts** as
  predicted. So part of 20a's μ≈0.5 is genuine init-lock (FM#9); the rest is a real high-T model bias.
  μ rel-RMSE 52.3%→47.7%.
  **But official S got WORSE (4.4e13→1.3e16):** μ at 1nW T05 only reached 7.1 (n≈7) so FM#10
  *persisted* (σ_γ = 3e-17!). Prediction "FM#10 disappears" was **falsified**. γ also regressed at
  1nW low/mid T (T05/T20 ratio 0.34; a basin jump to 0.1/0.8 stayed pinned low) — the μ change
  perturbed the γ trajectories through the shared sim cloud. S_rob improved 19.9→6.9 (the 1nW T10
  term dropped 179→5.9). **Status: 1st no-official-improvement.**
- **Root cause of FM#10 found (offline autopsy, 20b):** `compute_fwhm_and_dgamma` returns a
  dσ_fwhm/dγ that **blows up to 1.1e17** for 1–2 degenerate low-count draws near the 1nW T05 fitted
  point, because the L-BFGS Hessian inversion uses `reg=1e-4` and is ill-conditioned there. One such
  draw makes J_gg≈1e28 → σ_γ≈9e-15 → the whole of S. Fix tested offline: `reg=1e-3` removed that
  blow-up (max|dσ/dγ| 1.1e17→43, dFWHM/dγ 20.4→18.7).
- **20c (2026-09-15, 20 min):** ONE change = `IMPLICIT_REG` 1e-4→1e-3 in `_run_one` (affects optimizer
  γ-score + Fisher). **Prediction PARTLY falsified:** official S fell 5 orders (1.26e16→1.38e11) and
  S_rob improved 6.9→4.9, but it did **NOT** collapse to O(10): the 1nW T05 γ-Fisher is *still*
  degenerate — σ_γ ≈ 3.7e-12 (|dγ|/σ = 1.9e12). Autopsy at the new fitted point: reg=1e-3 leaves a
  **1.6e15** dσ/dγ blow-up from a single **n_sig=1** draw; reg=3e-3 removes it (max 9.5). So the
  pathology is a degenerate 1-signal-photon fit, and raising reg is fragile whack-a-mole (the fitted
  point moves → a new blow-up appears). μ rel-RMSE 47.7→47.3%, γ 36.8→35.0% (γ at 1nW T10/T20 pinned
  low ~0.42). **Status: improvement in S (streak reset).**
- **20e (2026-09-15, 21 min):** ONE change = per-scan FWHM excess scatter
  `ft += N(0, (HET_FRAC·IQR(target_f)/1.349)²)`, HET_FRAC=1, applied in optimizer + Fisher + FIG8.
  **Mixed:** S improved slightly 4.94→4.74 (μ rel-RMSE 47.4→45.4%), and μ ratios at low T rose
  further (1nW T10 0.78→0.88, T20 0.52→0.60). **But prediction (γ rel-RMSE drops sharply) FAILED:**
  γ rel-RMSE *rose* 34.8→36.0%; high-T γ rel err rose 6.4→11.0% (γ ratios dropped to 0.79–0.94).
  The jitter-matched cloud did not turn the γ channel into a clean location match — the γ estimate
  drifts with the KDE score history and brings the bias along. **Status: improvement in S (streak 0).**
- **Diagnosis → 20f:** γ is the S-dominant failure (mean |dγ|/σ = 3.95, driven by 3nW T05 10.0, 3nW T10
  15.8, 1nW T05 7.7, 1nW T20 5.5). The *data* says γ_true = median(FWHM)/2 (1nW T100: 17.01/2 = 8.50;
  3nW T80: 28.88/2 = 14.44 ≈ 14.1). A robust **median anchor** should pull γ there where the fragile
  KDE γ-score pins it at ~0.4×.
- **20f (2026-09-15, 21 min):** ONE change = `LR_GAMMA` 0.5→1.0 (anneal unchanged), on the evidence
  that high-T γ paths were *still climbing at iter 30* (1nW T60 ended 6.70/8.5) while low-T γ sat at
  a smooth biased optimum. **Prediction 1 HIT:** high-T γ improved a lot — **3nW T60/T80/T100 now
  ratio 1.00** (γ 14.13/14.13/14.06 vs 14.1), high-T γ rel err 11.0→6.2%, overall γ rel-RMSE
  36.0→34.6%. **Prediction 3 FAILED:** S rose 4.74→4.93 (low/mid-T γ |dγ|/σ grew: 3nW T05 10.0→17.1,
  1nW T20 5.5→8.9). So 20f *improved the goal* (γ) but worsened the *primary metric* — S and the goal
  diverge because S penalises the low-T γ mismatch hard. **Status: 1st no-improvement in S.**
- **Note (S vs goal divergence):** from 20e on, S has plateaued at ≈4.7–4.9 (differences ~4%). Its
  value is dominated by low/mid-T γ terms whose |dγ|/σ is large because σ_γ is small (
  systematic model mismatch, not statistical). The high-T γ goal is now essentially met (3nW exact,
  1nW ≈0.9); the μ-ratio goal is **not** met and — per the offline NLL(μ) autopsies (diag/mode_diag2:
  min at μ/μ_true≈0.3 for every model variant tried, incl. heterogeneity + σ-floor) — is a genuine
  model bias, not an optimiser artefact.
- **20g (2026-09-15, 23 min):** ONE change = `HET_FRAC` 1.0→0.0 (revert the 20e heterogeneity).
  **Prediction falsified — and decisively so:** S *worsened* to **6.50** (20f 4.93). The low-T γ
  regressed hard (3nW T05 |dγ|/σ 17.1→**31.3**; 1nW T20 8.9→… still 7.3). So the heterogeneity jitter
  is **net-positive for S** (it widens the low-T sim cloud so the γ estimate is not pinned so far
  low) — 20e (HET=1) remains the best-S config. **Status: 2nd consecutive no-improvement in S →
  STOP (SERIES_STATE §4.3).**

---

## 8. FINAL SUMMARY (series 20 — STOPPED on §4.3: 2 consecutive no-improvement, notebooks 20f+20g)

**Notebooks run: 7 (20a–20g), ~150 min compute. All one-change, all committed + pushed on `develop`.**
All changes were **notebook-level only** — no `src/` edits, so series 17/18/19 are untouched. Figures
render inline (11 panels per notebook incl. the fixed FIG1–FIG10); run histories in
`data/processed/20[a-g]_history.json`.

### Ladder of one-change experiments (S = primary metric, lower better)
| nb | one change | S | S_rob | μ rel-RMSE | γ rel-RMSE |
|----|-----------|-----|-------|-----------|-----------|
| 20a | 17g exactly on real targets (baseline) | 4.37e13 | 19.9 | 52.3% | 33.8% |
| 20b | μ score σ_ref = σ_prop (count-calibrated) | 1.26e16 | 6.9 | 47.7% | 36.8% |
| 20c | implicit-diff reg 1e-4→1e-3 | 1.38e11 | 4.9 | 47.3% | 35.0% |
| 20d | clip implicit derivatives (DERIV_CLIP=20) | **4.94** | 4.94 | 47.4% | 34.8% |
| 20e | per-scan FWHM heterogeneity (HET_FRAC=1) | **4.74 (best)** | 4.74 | 45.4% | 36.0% |
| 20f | LR_GAMMA 0.5→1.0 | 4.93 | 4.93 | 44.7% | 34.6% |
| 20g | HET_FRAC 1.0→0.0 (revert) | 6.50 | 6.50 | 47.2% | 35.1% |

### Final metrics (best-S config 20e; best-γ config 20f)
- **μ: ratio 0.45–0.88, rel-RMSE 44.7–47.4%; coverage 14/14 inside 2σ.** μ is *honest but biased
  low*: the point estimate lands at ≈0.5× truth at mid/high T. **μ-goal (ratio ∈[0.8,1.2] at high T)
  NOT met.**
- **γ: high-T essentially solved.** 20f gives γ_true at the 3nW high-T points (ratios 1.00 at T60/80/100)
  and high-T γ rel err 6.2%; overall γ rel-RMSE 34.6%. Low/mid-T γ is still biased low (ratios 0.38–0.56
  at T05/T10/T20) → low-T γ coverage fails. **γ-goal (≤5% at high T) nearly met; overall γ ≤10% NOT met.**
- **Transmission grading:** γ accuracy does climb with T (rel err 59%→4% from T05→T100); μ does *not*
  (it is pinned at ≈0.5× at all T).
- **Anti-overfit:** tuned (T20/60/100) vs held-out (T05/10/40/80) split always reported in-notebook;
  the S improvements (20d: 4.94) held on held-out.

### Failure modes
- **FM#10 (low-count Fisher degeneracy) — FIXED.** Root cause found by autopsy: `compute_fwhm_and_dgamma`
  returned dσ/dγ ≈ 1e17 for a single **n_sig=1** draw (ill-conditioned L-BFGS Hessian). 20c (reg 1e-4→1e-3)
  cut S by 5 orders; 20d (clip |dσ/dγ|,|dFWHM/dγ| at 20) closed it: S 1.4e11 → **4.94**, all 14 exps
  finite. This is the series' cleanest, most transferable result.
- **FM#9 (μ init-lock) — PARTLY fixed.** With σ_ref=σ_prop (20b) μ escapes its 0.5× init at low T
  (ratio 0.53→0.76 at 1nW T05). But at mid/high T μ stays 0.47–0.50.
- **FM#6 / FM#8 (Lorentzian + scatter-model mismatch) — attacked, not cured.** 20e's per-scan FWHM
  heterogeneity is net-positive for S (best S 4.74) but does not fix the γ bias.
- **Persisting core failure (the real blocker): the high-T μ bias.** Offline NLL(μ) autopsies at fixed
  γ_true (1nW T100, 3nW T80, 1nW T20) show the likelihood minimum sits at μ/μ_true ≈ 0.3 for the
  baseline and for **every** variant tried (additive σ_fit floor, multiplicative σ_fit scale, per-scan
  γ-heterogeneity, 1/√n extra jitter). The real FWHM clouds are far broader/right-skewed than any
  single-γ Lorentzian+photon-noise simulator produces, so the likelihood systematically prefers fewer
  photons. **μ is weakly identifiable / biased on this dataset — confirming series 19** — and the
  high-T μ-ratio goal is unreachable within this model class.

### Where the series ended
- **Stopped by §4.3** (2 consecutive no-improvement in S: 20f 4.93, 20g 6.50, vs best 20e 4.74).
- **Goal not met** on all 14 (μ ratio at high T, and overall μ/γ rel-RMSE thresholds).
- **Best config: 20e** (σ_ref=σ_prop, reg=1e-3, DERIV_CLIP=20, HET_FRAC=1.0, LR_GAMMA=0.5), S=4.74,
  μ rel-RMSE 45.4%, γ rel-RMSE 36.0%, high-T γ within 2σ on 6/6.
- **Not reached (stopped just before):** the obvious combination HET_FRAC=1.0 **+** LR_GAMMA=1.0
  (20e's S-best with 20f's γ-best) was never run — a natural first experiment if the series is resumed.
- Files: `notebooks/20_realdata_agentic_optimization/20[a-g].ipynb`, `data/processed/20[a-g]_history.json`,
  `SERIES_STATE.md`. **No `src/` changes; `notes/JOURNAL.md` untouched.**
- **Lesson → 20d:** attack FM#10 *robustly* instead of tuning reg — clip the per-draw implicit
  derivatives at a physical bound (every sane draw has |dσ/dγ| ≲ 10, |dFWHM/dγ| ≲ 25; the n_sig=1
  draw has |dσ/dγ|→∞).
- **20d (2026-09-15, 22 min):** ONE change = clip |dFWHM/dγ|,|dσ/dγ| at `DERIV_CLIP=20` in `_run_one`.
  **Prediction HIT:** official **S collapses 1.38e11 → 4.94** and now equals S_rob (14/14 exps finite,
  σ_γ = 0.5–1.2 for all); μ/γ point estimates moved < 3% (μ rel-RMSE 47.4%, γ 34.8%). **FM#10 CLOSED**
  — the metric is now meaningful. FIG12 (max unclipped |dσ/dγ| per fitted point) shows no blow-ups in
  100 draws; the blow-up is rare (a single n_sig=1 draw in 500) and the clip bounds it deterministically.
  **Goal status after 20d:** high-T γ 6/6 within 2σ (rel err mean 6.4%, ~5% at T100); high-T μ ratio
  still 0.47–0.49 (0/6) → μ-goal unmet; low-T γ coverage poor (3nW T05 |dγ|/σ=16). S improved again
  (streak 0). **20e must now attack the goal (γ low-T / μ bias).**

---
### Agent instructions (per tick / per notebook)
1. Read this file. If §7 status is `DONE`, stop.
2. If a notebook is unrun: `cd notebooks/20_realdata_agentic_optimization && NB_SMOKE=1 papermill 20x.ipynb /tmp/_smoke.ipynb` (fast validate), then full: `papermill 20x.ipynb 20x.ipynb` (in place).
3. Save history JSON -> `data/processed/20x_history.json`; append the §7 row.
4. Design notebook 20(x+1): change **exactly ONE thing**, state hypothesis + falsifiable
   prediction in markdown BEFORE the config; justify from the previous notebook's figures.
5. `git add -A && git commit && git push` (branch `develop`) after every notebook.
6. Check stop conditions after each notebook.
