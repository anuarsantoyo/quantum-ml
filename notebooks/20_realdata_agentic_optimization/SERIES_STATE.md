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
| # | notebook | one change | S (primary) | gamma rel-RMSE | mu rel-RMSE | status |
|---|----------|-----------|-------------|----------------|-------------|--------|
| a | 20a | 17g exactly, real targets (baseline) | 4.37e13 (robust 19.9) | 33.8% | 52.3% | DONE |

### Log notes
- **20a (2026-09-15, 18 min):** μ lands at **0.41–0.55 × μ_true in ALL 14** exps (mu rel-RMSE 52.3%)
  and barely moves off its init (0.5·μ_true) — μ paths are essentially flat (e.g. 1nW T10: 6.19→6.78
  over 30 iters). Diagnosis: **μ init-lock (FM#9)** — the count channel is dead (uniform KDE
  responsibility at N=61–3742) so the REINFORCE μ-gradient ≈ 0; μ never escapes 0.5·truth.
  γ is the healthy channel: within 2σ on all 6 high-T exps (γ rel-RMSE 9.0% there, 5.3% at T100),
  but 3nW T20/T40 are −0.9σ/−3.2σ and 1nW T05/T10 are pathological.
- **FM#10 created/observed:** 1nW T05 has H_S=212 MHz (huge fit-error scatter) and μ stuck at
  ~5 photons → dFWHM/dγ derivatives blow up → J_gg ≈ 1e28 → σ_γ ≈ 9.4e-15 → the single
  |dγ|/σ = 6.1e14 term dominates S. 1nW T10 also small (σ_γ=0.021 → 179). These two are the
  whole of S; the other 12 sum to ≈4.8. Fixing μ init-lock should dissolve FM#10 (higher n ⇒
  stable fits).
- Robust companion S_rob (non-degenerate exps: drop σ_γ<1e-6 or J_gg>1e20) = **19.9** → the
  baseline to beat on a meaning-comparable scale. (Only 1nW T05 is dropped; 1nW T10, σ_γ=0.021,
  stays and contributes 179, i.e. it is *small*, not degenerate.)

| b | 20b | μ score σ_ref = σ_prop (count-calibrated) | 1.26e16 (robust 6.9) | 36.8% | 47.7% | DONE |

- **20b (2026-09-15, 19 min):** ONE change = per-exp `σ_ref = σ_prop` in the μ step (D1 score).
  **Prediction largely HIT:** μ *escaped init* at low T (1nW T05 0.53→0.76, T10 0.55→0.79,
  3nW T05 0.53→0.64) while high-T μ stayed ≈0.47–0.49 — the μ-ratio-vs-T curve **inverts** as
  predicted. So part of 20a's μ≈0.5 is genuine init-lock (FM#9), part is a real high-T model bias.
  μ rel-RMSE 52.3%→47.7%.
  **But official S got WORSE (4.4e13→1.3e16):** μ at 1nW T05 only reached 7.1 (n≈7) so FM#10
  *persisted* (σ_γ = 3e-17!). Prediction "FM#10 disappears" was **falsified**. γ also regressed at
  1nW low/mid T (T05 ratio 0.34, T20 0.34; basin jumps to 0.1 and 0.8 stayed pinned low) — the μ
  change perturbed the γ trajectories through the shared sim cloud. Robust S_rob improved 19.9→6.9
  (the 1nW T10 term dropped 179→5.9). **Status: 1st no-official-improvement.**
- **Root cause of FM#10 found (offline autopsy, 2026-09-15):** the implicit derivative dσ_fwhm/dγ
  in `compute_fwhm_and_dgamma` **blows up to 1.1e17** for 1–2 degenerate low-count draws (n≈6–7) at
  the fitted point of 1nW T05, because the L-BFGS Hessian inversion uses `reg=1e-4` and is
  ill-conditioned there. That single draw makes J_gg≈1e28 → σ_γ≈9e-15 → the whole of S. Fix tested:
  `reg=1e-3` removes the blow-up entirely (|dσ/dγ|max 1.1e17→43, dFWHM/dγ barely moves 20.4→18.7).
  → **20c change.**

---
### Agent instructions (per tick / per notebook)
1. Read this file. If §7 status is `DONE`, stop.
2. If a notebook is unrun: `cd notebooks/20_realdata_agentic_optimization && NB_SMOKE=1 papermill 20x.ipynb /tmp/_smoke.ipynb` (fast validate), then full: `papermill 20x.ipynb 20x.ipynb` (in place).
3. Save history JSON -> `data/processed/20x_history.json`; append the §7 row.
4. Design notebook 20(x+1): change **exactly ONE thing**, state hypothesis + falsifiable
   prediction in markdown BEFORE the config; justify from the previous notebook's figures.
5. `git add -A && git commit && git push` (branch `develop`) after every notebook.
6. Check stop conditions after each notebook.
