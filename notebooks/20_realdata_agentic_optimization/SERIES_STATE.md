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
| a | 20a | 17g exactly, real targets (baseline) | 4.37e13 (robust 5.6) | 33.8% | 52.3% | DONE |

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
- Robust companion S_rob (non-degenerate exps only) = **5.6** → the baseline to beat on a
  meaning-comparable scale.

---
### Agent instructions (per tick / per notebook)
1. Read this file. If §7 status is `DONE`, stop.
2. If a notebook is unrun: `cd notebooks/20_realdata_agentic_optimization && NB_SMOKE=1 papermill 20x.ipynb /tmp/_smoke.ipynb` (fast validate), then full: `papermill 20x.ipynb 20x.ipynb` (in place).
3. Save history JSON -> `data/processed/20x_history.json`; append the §7 row.
4. Design notebook 20(x+1): change **exactly ONE thing**, state hypothesis + falsifiable
   prediction in markdown BEFORE the config; justify from the previous notebook's figures.
5. `git add -A && git commit && git push` (branch `develop`) after every notebook.
6. Check stop conditions after each notebook.
