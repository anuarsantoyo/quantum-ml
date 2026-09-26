# SERIES 24 — SERIES_STATE.md

**Real-data accuracy: close the model/data gap, one change per notebook.**
Opened 2026-09-26 (Anuar voice brief: *"start a new series … analyze the whole project … you decide the
best starting point … hypothesis → test → conclusion → the next notebook follows the last one … run at
least 30 experiments … the model must optimise correctly to the real data; higher transitions matter
more, but it would be good if all of the real data are approximated somehow"*).

This file is the durable state. Every notebook appends a row to §7 and writes its verdict inline.
**Do not change the goal or the metric definitions.**

---

## 1. Starting point (chosen 2026-09-26, from the whole-project survey)

The project's real-data **best KDE model** is **AG-HYPOPT `experiment_6` `trial_07`** — the only loss
family that can also carry a Fisher/CRB (exp8's `cvm_fwhm`/`w1_2d` win ~10 % on the point estimate but
are *discrepancies*, not likelihoods). Series 23 (`23a`) re-ran exactly this model on all 14 real
experiments: **T-weighted objective 0.0811 (all 14), 0.0675 (T ≥ 40), 0.1228 (T ≤ 20), 0 divergences**
(re-confirmed by **24a**, 2026-09-26; μ/γ rel-RMSE 33.2 % / 26.1 % over all 14 — the 35.9 %/22.3 % figures
are the 8-exp benchmark subset). That is **24a** and the frozen baseline for this series.

**Where the remaining error lives (carried in from series 19–23):**
- **μ** is still the dominant channel (35.9 % vs γ 22.3 %). The two mechanisms on record:
  (i) a **model/data gap (FM#8)**: the simulated σ_fit (analytic CRLB) is 1.04–4.25× too narrow vs the
  real fit-error and grows with T → the KDE's σ channel favours fewer photons → the μ attractor;
  (ii) an **optimiser-travel** effect: `23b/23c/23e` showed the high-T μ landing is **monotone in the step
  budget** (0.43× → 0.409 err, 1× → 0.335, 4.4× → 0.217), and that a *uniform photon target* rebalances
  (fixes starved cells, breaks the already-good ones) instead of improving.
- **γ** fails at **1nW low T (FM#6**, the Lorentzian line-shape deficit); high-T γ is essentially solved.
- **Low-T μ overshoots** (22a arm reached ×3.6) when the travel is not cell-adapted.
- **Uncertainty**: no honest interval exists yet — the CRB is bias-blind (`23a–e`), the observed Fisher
  is outlier-dominated/fragile (`23d`) → needs the empirical data bootstrap + the sandwich `J⁻¹KJ⁻¹`.

**Declared stop:** none hard; keep going while the roadmap has productive entries (Anuar wants ≥ 30).
Two consecutive no-improvement → re-read §6 and change phase rather than stop.

---

## 2. Frozen protocol (identical in every series-24 notebook)

- **Data:** all 14 real experiments (1nW/3nW × T05…T100); real-target load + filter from 18b/19d/23a
  (`filt = ok & (err/fwhm < 10)`); the per-exp true table (`mu_true, sigma_prop, lam, gamma_true,
  n_target`) is the 23a cell-3 table — do not edit it.
- **Budget:** `N_RUNS = 100`, `N_ITER = 30`, `SEED = 42` (deterministic per config). Inner fitter =
  Lorentzian, `n_iters=80`. `N_WORKERS = 4` (`fork` pool).
- **Model freedom:** anything may change (simulator / σ-model / score / loss / schedule / init).
  `src/` edits must be **additive** (new flag/option) so series 17–23 stay reproducible. Prefer
  notebook-level changes; if a `src/` change is unavoidable, add a new flag with a default that
  reproduces the old behaviour.
- **Init:** `μ_init = 0.5·μ_true`, `γ_init = 0.5·γ_true` (the protocol's honest init) unless the
  notebook's one change is explicitly about the init.
- **One change per notebook.** State the hypothesis + a falsifiable prediction in a markdown cell
  BEFORE the config cell. The change must be a single named knob/mechanism vs the previous notebook.
- **Figures:** fixed panel FIG1..FIG3 as in 23a; **inline only** (no `savefig`, no PNG outside the
  notebook); interactive FIG1 exported to HTML (needed for the browser). Add diagnostics freely.
- **Execution:** run **in place** — `papermill 24x.ipynb 24x.ipynb`. Never create `-executed` copies.
  Smoke first: `NB_SMOKE=1 papermill 24x.ipynb /tmp/_smoke_24x.ipynb`.
- **Run data:** `data/processed/24x_history.json` (paths + per-step clouds + per-exp true/fit).
  Companion run data must live in the repo, never /tmp.
- **Commit + push after every notebook** (branch `develop`): `git add -A && git commit -m "series 24: 24x — <one change> (<verdict one-liner>)" && git push`.
- **Never edit** `notes/JOURNAL.md`, or any notebook outside `notebooks/24_realdata_accuracy/`.

---

## 3. Frozen success metric (identical across series-20/21 + exp5–8)

Per experiment: `rel_sq_mu = min(((μ_fit−μ_true)/μ_true)², CAP)`, `rel_sq_gamma = min(((γ_fit−γ_true)/γ_true)², CAP)`,
with `CAP = 1.0`. Weight `w(T) = W_FLOOR + (1−W_FLOOR)·T/100`, `W_FLOOR = 0.25`.

**Primary score** (lower = better, computed on all 14 by `tw_objective`):
`W_obj = Σ w·err / Σ w` over the 28 (μ,γ) cells.
Report always: **W_obj(all14)**, **W_obj(T ≥ 40)**, **W_obj(T ≤ 20)**, μ rel-RMSE, γ rel-RMSE,
n_diverged. Goal bands (from series 20 §3): high T `|Δ|/σ ≤ 2` and μ ratio ∈ [0.8, 1.2] and γ rel-RMSE
≤ 5 %; overall μ rel-RMSE ≤ 15 %, γ ≤ 10 %; accuracy **improving with T**.

**Anti-overfit:** the metric is computed on **all 14**; when a config wins, verify on the **high-T /
low-T split** and (for any knob that was picked from the data) on a held-out subset. Bandwidths and all
kernel/loss parameters must be **truth-free** (derived from the target cloud or the sim cloud only —
never from `mu_true`/`gamma_true`).

**Chaos caveat:** the objective is chaotically sensitive (~±0.01 for <1e-3 knob changes, integer photon
rounding). Only coarse trends are meaningful; a single ≤0.005 gap is not a result. Repeat configs ≥2×.

---

## 4. Failure-mode tags (carried)

`FM#6` Lorentzian line-shape mismatch / 1nW low-T γ deficit · `FM#8` un-damped/ill-scaled σ_fit channel
collapsing μ (sim σ 1.04–4.25× too narrow, growing with T) · `FM#9` μ init-lock / step starvation ·
`FM#10` low-count Fisher degeneracy (fixed by DERIV_CLIP in 20d) · `FM#11` (new) **observed-Fisher
fragility / outlier domination** (23d) · `FM#12` (new) **CRB bias-blindness** (23a–c). Tag each notebook.

---

## 5. Roadmap — one change per notebook (prioritised; a verdict may reprioritise)

**Phase A — μ travel / step control (the proven lever, series-23 handoff).**
- **A1 (24a)** — baseline: `experiment_6 trial_07` reproduced exactly (no change; reference 0.0811).
- **A2 (24b)** — **truth-free travel normalisation** (23f proposal): replace the fixed `lr_mu` anneal by a
  **normalised-gradient step** `Δμ = (μ_init/N_ITER)·grad/⟨|grad|⟩`, so each cell covers its own
  `μ_init` distance over the run. Prediction: high-T μ ratio → 1, low-T overshoot shrinks; no change to γ.
- **A3 (24c)** — **total-travel budget = k·μ_init** with a *cosine* decay (vs the linear anneal), at the
  same total travel as A2 vs a larger k. Prediction: the landing is set by total travel, not the shape.
- **A4 (24d)** — **per-cell LR ∝ σ_prop²** (= 23e's finding) but with a *cap/floor* so high-σ_prop cells
  are not over-driven when they are already near truth. Prediction: keeps 23e's low-T gain without its
  high-T regression.
- **A5 (24e)** — **residual-aware stop/step**: freeze μ when `⟨|grad|⟩` drops below a threshold (truth-free
  convergence test). Prediction: kills the overshoot cells, costs nothing at high T.
- **A6 (24f)** — `N_ITER 30 → 60` at fixed *total* travel (finer integration). Prediction: no change if
  travel is the only lever (a negative control for A2/A3).
- **A7 (24g)** — **two-phase schedule**: explore wide for 15 steps, then damp to a small step for 15.
  Prediction: better than both fixed arms on the low-T overshoot cells.
- **A8 (24h)** — **γ travel normalisation** (γ_init = 0.5·γ_true, travel = γ_init), attacking FM#6.
- **A9 (24i)** — **coupled μ/γ travel**: hold `Δγ/Δμ` fixed to the valley direction (from the local Hessian
  of the reward). Prediction: path stays in the valley → both channels improve.
- **A10 (24j)** — **truth-free μ-init from the data** (count-based μ̂ from `n_target`, `σ_prop`, `λ`) so the
  protocol no longer needs to know `μ_true`; then re-run A2 with it. Prediction: near-identical landing.

**Phase B — the σ_fit channel / loss form (FM#8).**
- **B1 (24k)** — **σ(μ) shape matching** (19d rec#2, never tried on the loss side): per experiment, map the
  simulated σ_fit through a **monotone quantile map** fitted sim→target (truth-free). Prediction: removes
  the μ attractor → μ ratio ↑ sharply.
- **B2 (24l)** — same map but **scale-only** (19c's rejected scale fix) as the control for B1.
- **B3 (24m)** — **σ-weight anneal**: `sw` high early (μ fuel) → low late (kill the bias). Prediction:
  captures B1's μ gain without a new estimator.
- **B4 (24n)** — **Student-t / robust kernel** on the σ channel (heavy-tailed; 23d showed outlier
  domination). Prediction: σ channel robust, μ bias smaller.
- **B5 (24o)** — **estimator-matched σ_fit**: use the 22g pseudo-Voigt calibration to replace the analytic
  CRLB σ with the *recorded*-style fit error. Prediction: the channel becomes meaningful → μ ↑.
- **B6 (24p)** — **drop σ, add a μ-sensitive robust statistic** (exp8's `cvm_fwhm` for μ + KDE for γ —
  two-role hybrid). Prediction: best of both (point estimate like exp8, γ/figure from KDE).
- **B7 (24q)** — **per-scan γ heterogeneity** (20e `HET_FRAC`) re-tested on the A2 travel fix (FM#6).
  Prediction: helps low-T γ, harmless elsewhere.
- **B8 (24r)** — **1-D FWHM-only KDE** but with the **count channel re-added as a prior** (no σ channel).
  Prediction: separates "σ is the culprit" from "2-D is the culprit" (22d said dropping σ alone fails).

**Phase C — uncertainty honesty (side-car; cheap, no re-optimisation or reuses the paths).**
- **C1 (24s)** — **empirical data bootstrap** of the 14 real experiments (resample retained rows, re-run
  the *frozen* optimiser) → honest intervals; compare to the CRB. Prediction: bootstrap σ ≫ dataset CRB
  but ≈ per-scan/√N_eff with the bias included.
- **C2 (24t)** — **sandwich** `J⁻¹KJ⁻¹` (K = empirical score covariance) vs `J⁻¹`. Prediction: sandwich
  wider, coverage restored at high T.
- **C3 (24u)** — **profile-likelihood interval** for μ (bias-aware, no CRB). Prediction: covers the biased
  high-T cells where the CRB cannot.
- **C4 (24v)** — **bias-corrected CRB**: subtract the sim-at-truth NLL-minimum bias (from 22b/22g) from the
  point estimate, then re-check coverage against the bootstrap.
- **C5 (24w)** — **coverage referee**: one table comparing CRB / sandwich / bootstrap / profile across the
  14 → the honest-interval verdict.

**Phase D — estimator / forward model (larger, only if B fails).**
- **D1 (24x)** — **matched estimator**: simulate with the pseudo-Voigt estimator (22g's best-matched) so sim
  and real share the estimator; re-derive the KDE. Prediction: sim cloud shape → real (fixes FM#8 at root).
- **D2 (24y)** — **differentiable forward model (Option C)**: replace the analytic σ_fit with a
  differentiable surrogate of the fit-error map. Prediction: μ lands at truth.
- **D3 (24z)** — **per-(T,power) σ-bias correction** table from 22g applied to the sim σ (cheapest D form).

**Phase E — protocol / combination / verification.**
- **E1 (24aa)** — **combination**: best config from each phase (A + B + C) in one model.
- **E2 (24ab)** — **repeat the winner 3×** (different `SEED`) to separate signal from the chaos band.
- **E3 (24ac)** — **independent referee**: run the winner through exp8's `cvm_fwhm`/`w1_2d` objective
  (frozen, Fisher-free) — a second metric must agree.
- **E4 (24ad)** — **held-out protocol**: tune on T20/60/100, report on T05/10/40/80; final honest verdict.

> The runner **may insert a notebook** at any point when a verdict opens an obvious follow-up; it must
> keep the one-change rule and add it to this roadmap (mark it `A/B/C/D/E` + next letter). Cap: none, but
> after 24ad either declare the series closed or open `25_*`.

---

## 6. Agent instructions (the loop)

1. Read this file. Find the highest-priority unrun roadmap entry (or the follow-up the last verdict named).
2. `cp <prev>.ipynb <new>.ipynb` (fork the **previous** notebook; 24a forks `../23_fisher_uncertainties/23a-*.ipynb`).
3. Edit: the title markdown cell (state the ONE change + hypothesis + falsifiable prediction), the CONFIG
   cell (ONE named change), `OUT_JSON`, and the HTML export paths. Keep everything else byte-identical.
4. Smoke: `NB_SMOKE=1 papermill <new>.ipynb /tmp/_smoke_<new>.ipynb` → fix any error, confirm inline
   image outputs exist. Then full: `papermill <new>.ipynb <new>.ipynb` (in place, ~10–30 min).
5. Replace the placeholder verdict markdown cell with the **verdict**: the frozen metrics table, the
   one-line scientific read, and the FM tag. State whether the prediction was HIT / PARTLY / FALSIFIED.
6. Append the §7 log row. Save `data/processed/<new>_history.json` from the run cell.
7. `git add -A && git commit && git push`. Only then start the next notebook.
8. Keep the metric truth-free; never tune a knob against `mu_true`/`gamma_true`.

**Workspace hygiene:** disk is small (7 GB RAM / check `df`); delete smoke artifacts; never commit figures.

---

## 7. Log

| # | notebook | one change | W_obj(all14) | W_obj(T≥40) | W_obj(T≤20) | μ rel-RMSE | γ rel-RMSE | div | status |
|---|----------|-----------|--------------|-------------|-------------|-----------|-----------|-----|--------|
| a | 24a | baseline: exp6 `trial_07` exactly (reference) | 0.0811 | 0.0675 | 0.1228 | 33.2% | 26.1% | 0/14 | DONE (baseline reproduced exactly, 19 min) |

| b | 24b | A2: truth-free μ-travel budget (Δμ clipped to μ_init/n_iter) | 0.0730 | 0.0435 | 0.1633 | 28.0% | 33.6% | 0/14 | DONE — μ ↑ in all 14 cells; γ collateral ↓ at low T |

**Current best = 24b** (all14 0.0730, T≥40 0.0435) — but its γ regressed; 24a keeps the best γ (22.3%/26.1%).

**Baseline per-cell truth (μ/true, γ/true)** — the target every later notebook must improve:

| T | 5 | 10 | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|
| μ 1nW | 0.646 | 1.212 | 1.600 | 0.817 | 0.715 | 0.596 | 0.730 |
| μ 3nW | 1.213 | 1.090 | 0.949 | 0.715 | 0.610 | 0.578 | 0.561 |
| γ 1nW | 0.746 | 0.411 | 0.639 | 0.824 | 0.864 | 0.901 | 0.933 |
| γ 3nW | 0.610 | 0.620 | 0.829 | 0.857 | 0.994 | 0.983 | 0.980 |

**Read:** low-T μ **overshoots** (1nW T20 ×1.60), high-T μ **undershoots** (3nW T100 ×0.56) — the one global
`lr_mu` anneal cannot serve both regimes → Phase A. γ climbs monotonically with T toward truth; the 1nW low-T
deficit (FM#6) is the γ residual.
