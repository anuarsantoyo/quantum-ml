# Series 24 — Real-data accuracy: executive summary

**Opened** 2026-09-26 (Anuar's brief: *“analyse the whole project … the model must optimise correctly to the
real data; higher transitions matter more, but it would be good if all of the real data are approximated
somehow”*). **Closed** 2026-09-27 after **30 notebooks** (`24a`…`24ad`) on branch `develop`.
State file: `SERIES_STATE.md` (protocol §2, metric §3, failure modes §4, roadmap §5, log §7).

---

## 1. Goal

Take the project's real-data KDE model from the series-23 baseline (`experiment_6 trial_07`, `W_obj` 0.0811)
to the best attainable fit of **all 14 real experiments** (1nW / 3nW × T05…T100), **weighted toward high
transmission**, with an **honest uncertainty statement** — using exactly one named change per notebook,
hypothesis → test → conclusion.

## 2. Frozen protocol & metric

- **Data**: all 14 real experiments; real-target load + `filt = ok & (err/fwhm < 10)` (18b/19d/23a).
- **Budget**: `N_RUNS=100`, `N_ITER=30`, `SEED=42`, init `0.5×truth`, inner Lorentzian fit (`n_iters=80`).
- **Rules**: one change per notebook, truth-free knobs, notebook-level changes preferred, figures inline.
- **Metric** (series-20/21 + exp5–8, frozen): per cell `rel_sq = min(((x̂−x)/x)², 1)`, weight
  `w(T)=0.25+0.75·T/100`; **`W_obj = Σw·err / Σw`** over the 28 (μ,γ) cells.
  Always reported: `W_obj(all14)`, `W_obj(T≥40)`, `W_obj(T≤20)`, μ rel-RMSE, γ rel-RMSE, divergences.
- **Measured chaos band** (new in E2): **≈ 0.0026** on all14 (replicate sd), **4× tighter** than the
  ±0.01 caveat the series started from.

## 3. The notebook ladder (`24a` → `24ad`)

| notebook | one change | all14 | T≥40 | T≤20 | verdict |
|---|---|---|---|---|---|
| `24a` | baseline: exp6 `trial_07` exactly (reference) | 0.0811 | 0.0675 | 0.1228 |  |
| `24b` | A2: truth-free μ-travel budget (Δμ clipped to μ_init/n_iter) | 0.0730 | 0.0435 | 0.1633 | μ ↑ in all 14 cells; γ collateral ↓ at low T |
| `24c` | A3: cosine travel schedule at FIXED total travel (μ_init) | 0.0787 | 0.0505 | 0.1650 | shape is 2nd-order (+0.006); new FM#13: clip mostly idle, init-LR under-travels (net 0.42×μ_init |
| `24d` | A4: per-cell μ LR × clip(σ_prop²/median, 1, 2) | 0.0744 | 0.0448 | 0.1650 | floor cells bit-identical to 24c; scaled cells +0.03–0.05; new FM#14: net travel saturates ~0.43×μ_init (clip bursts + gradient sign flips |
| `24e` | A5: residual-aware μ freeze (\|∇μ\|< 0.15·\|∇μ\|_init) | 0.0771 | 0.0452 | 0.1747 | freezes 3–22/30 steps; neutral on all14 (+0.0027) but hurts low T (0.1650→0.1747) ⇒ late steps carry REAL travel, not flip-waste |
| `24f` | A6 control: n_iter 30→60 at FIXED total travel | 0.0781 | 0.0435 | 0.1839 | aggregate unchanged (+0.0010 ⇒ travel-time, not step-count, is the lever); per-cell spread ±0.08–0.15 = the measured chaos width; control branch — 24g re-forks 24e |
| `24g` | A7: two-phase μ schedule (2× first half, 1/3 second half) | 0.0747 | 0.0463 | **0.1614** | best T≤20 + best γ of Phase A (1nW T05 γ 0.617→0.848); misses 24d by +0.0003 (chaos); Phase A exhausted as a global lever |
| `24h` | A8: γ travel normalisation (A2 idea on γ) | 0.1071 | 0.0884 | 0.1645 | breaks the already-good high-T γ (3nW T60–100 0.99→0.82) and does not lift low-T γ ⇒ FM#13 applies to BOTH channels; neither is travel-starved ⇒ residual is FM#8; branch dropped, 24i re-forks 24g |
| `24i` | A9: coupled μ/γ travel along the init valley (fixed ρ) | 0.1870 | 0.2135 | **0.1057** | fixed init-ρ is not transferable (ρ spans −0.36…+1.23, cap hits); coupling wrecks the high-T γ (1nW T100 0.94→0.27 |
| `24j` | A10: truth-free μ init (bisect sim median σ_fit ↔ target median σ_fit) | 0.2467 | 0.2379 | 0.2737 | μ̂/μ_true = 1.02/0.98 at 1nW T10/T20 but 0.22 at 3nW T100; the σ-matching estimator inherits the growing sim/target σ mismatch ⇒ keep `0.5×truth` until the σ model is fixed (Phase B/D |
| `24k` | B1: sigma shape matching (monotone quantile map sim->target inside the KDE, chain-ruled to sim_ds) | 0.0799 | 0.0514 | 0.1674 | mu table moves as a *wash* (8 cells toward 1, 6 away; 3nW T80 0.83->0.80 but T100 0.70->0.66); W_obj 0.0747->0.0799 (+0.005). FM#8 re-scoped: scale (19c) AND marginal shape (24k) both fail => the mu mode is set by the sigma(mu) *conditional*, not the… |
| `24l` | B2: scale-only control for B1 (single scalar sigma remap, f'=a) | 0.0751 | 0.0453 | 0.1664 | W_obj within 0.0004 of 24g; no systematic mu motion. Shape map (24k +0.0052 worse) vs scale map (neutral) => B1's failure is not plumbing; FM#8 is the sigma(mu) *conditional*, not the sigma marginal/scale |
| `24m` | B3: sigma-weight anneal (sw_hi=2x early -> sw_lo=0.5x late) | 0.0820 | 0.0520 | 0.1738 | amplifying the sigma channel early does not buy mu (mu RMSE 29.4->30.6%) and damages gamma (32.5->34.3%, 1nW T05 gamma 0.85->0.69) because sw also drives the gamma-chain sigma term => no weighting of the *current* sigma channel helps; it must be corr… |
| `24n` | B4: robust Student-t (nu=3) kernel on the sigma channel | 0.0771 | **0.0402** | 0.1901 | best T>=40 of the series (0.0463->0.0402): down-weighting the sigma outliers removes part of the mu pull (1nW T40 mu-err 0.093->0.046); but low-T gamma pays (1nW T05 gamma 0.85->0.56) => the gamma-chain sigma term is load-bearing at low T. Parked as … |
| `24o` | B5: estimator-matched sigma_fit (pseudo-Voigt sigma channel, 22g calibration) | 0.0796 | **0.0353** | 0.2155 | best T>=40 yet (0.0463->0.0353) and best B-phase mu (28.6%): the raw sim/target sigma ratio moves 0.32-0.87 -> 0.60-1.38 (22g reproduced in-optimiser) => FM#8 is an ESTIMATOR problem; but low-T gamma collapses (1nW T05 0.85->0.39) because the gamma c… |
| `24p` | B6: hybrid two-role loss (exp8 cvm_fwhm for the mu reward + KDE for gamma) | **0.0684** | 0.0354 | 0.1695 | the bandwidth-free, sigma-free cvm_fwhm mu reward gives the best all14 (0.0730->0.0684) and best mu (27.7%); gamma pays only +0.9pp (the KDE keeps sigma, but no longer for mu) and the run is back to 19.6 min |
| `24q` | B7: per-scan FWHM heterogeneity on the A2 travel (B6 off) | 0.0809 | 0.0536 | 0.1645 | FALSIFIED; 20e's heterogeneity does NOT transfer to the travel optimiser |
| `24r` | B8: 1-D FWHM-only KDE + count channel re-added as a prior (σ channel off) | **0.0672** | **0.0326** | 0.1735 | new series best on all14/T≥40/μ: σ *is* the μ culprit (22d resolved: the count prior `N(μ_init,μ_init²)` tames the runaway, net travel 0.32→0.54×μ_init), but the 2-D σ-bearing γ chain is load-bearing for low-T γ (γ 33.4→35.1%; 1nW T05 0.75→0.61) ⇒ "σ… |
| `24s` | C1: empirical data bootstrap (8 exps × B=10, M_FINAL 150) | 0.0672 | 0.0326 | 0.1735 | bootstrap sd is 0.24× the dataset CRB (4× *narrower*, not wider) and 95% μ coverage 0/8: the μ landing is nearly data-independent (set by the A2 travel rule) ⇒ the residual is BIAS (FM#8) and the CRB is *bias-blind*, not too narrow (FM#12 re-scoped |
| `24t` | C2: sandwich `J⁻¹KJ⁻¹` (K = score cov, H = FD-Hessian observed info), DO_FISHER on (M_FINAL 150) | 0.0672 | 0.0326 | 0.1735 | 2σ μ coverage 5/14 (CRB) → 10/14 (sandwich), 1σ 2→7/14: the CRB misses are model-misspecification (K≠H); but H is near-singular at low N (sand/CRB up to 2655×, 1nW T10) ⇒ FM#11 confirmed + quantified — the sandwich is honest but *uninformative at low… |
| `24u` | C3: profile-likelihood interval for μ (grid ±5 dataset-CRB σ, 13 pts × 2 seeds) | 0.0672 | 0.0326 | 0.1735 | profile σ is 0.64× the CRB (sharper, not ≥1.5× wider) and covers 3/14; the Δ=1 crossing resolves in only 6/14 cells and is noise ⇒ agrees with C1: the CRB is too *wide* AND mis-centred; only a bias-corrected interval can work (⇒ C4 |
| `24v` | C4: bias-corrected CRB via the sim-at-truth bias (22b/22g) | 0.0672 | 0.0326 | 0.1735 | the sim-at-truth bias is large and negative (median −9.4 μ, −1.9 … −42.8); subtracting it cuts the μ error 13.74 → 2.04 (6.7×) and restores 2σ CRB coverage 5/14 → 12/14 ⇒ the residual is a measurable, correctable BIAS (FM#8 resolved on the loss side)… |
| `24w` | C5: coverage referee (CRB/sandwich/bootstrap/profile/bias-corrected over the 14) | 0.0672 | 0.0326 | 0.1735 | bias-corrected CRB 12/14 & corrected sandwich 13/14 at 2-sigma vs raw CRB 5/14, sandwich 10/14, profile 3/14, bootstrap 0/8; raw CRB width adequate (1.00x) but mis-centred ⇒ the honest interval is the bias-corrected one (sandwich 2.51x wider |
| `24x` | D1: estimator-matched pseudo-Voigt forward model (pv FWHM + sigma, 2-D KDE) | 0.1953 | 0.1597 | 0.3044 | matching the estimator at the FWHM channel does NOT fix mu (27.1% vs 24r 25.5%; the table moves *down*) and wrecks gamma (61.4% vs 35.1%) ⇒ the mu residual is a line-shape/model gap (FM#6), not an estimator mismatch; 92.7 min (pv fit |
| `24y` | D2: differentiable surrogate sigma = kappa(FWHM,n,gamma)*sigma_CRLB (22g-calibrated, R2 0.66) | 0.0740 | 0.0476 | **0.1551** | the calibrated sigma repairs gamma (33.0%, near the 24g best 32.5%; 3nW T60-100 gamma ~1.00) and the low-T split (0.1551 vs 0.1735) but mu worsens (28.6% vs 24r 25.5%) ⇒ the sigma SCALE does not set the mu mode; mu wants the sigma-free FWHM kernel (2… |
| `24z` | D3: per-(T,power) sigma-bias table c(exp)=real/sim on top of the D2 surrogate | 0.0764 | 0.0488 | 0.1609 | the exact per-cell sigma table (c3 0.58–1.56, median 1.00) leaves mu unchanged (28.7% vs 24y 28.6%) ⇒ the mu landing is insensitive to the sigma scale at every level (D1/D2/D3 all fail) ⇒ the residual high-T mu error is the line-shape/photon model (F… |
| `24aa` | E1: mu = 1-D FWHM kernel + count prior (24r), gamma = 2-D calibrated-sigma KDE (D2/D3) + 24v bias-corrected report | **0.0651** | 0.0341 | **0.1601** | the role-split collects 24r's mu (25.7% ~ 25.5%) with the calibrated sigma's gamma (33.3% vs 35.1%) and the best global low-T (0.1601); all14 0.0672→0.0651 (−0.0021, inside the chaos band ⇒ E2 must confirm); the bias-corrected mu table restores 12/14… |
| `24ab` | E2: repeat the E1 winner 3x at SEED bases 42/43/44 (single knob: the noise seed) | **0.0644** | 0.0339 | 0.1576 | all 3 replicates below 24r (0.0615–0.0666 vs 0.0672; mean margin +0.0028 = 1.1x the seed sd 0.0026) ⇒ the E1 all14 win is real and the *measured* chaos band is only ~0.0026 (4x tighter than the ±0.01 caveat); seed 42 is bit-identical to 24aa; but T>=… |
| `24ac` | E3: independent referee — exp8 `cvm_fwhm`/`w1_2d` on the winner + 8 candidates (fresh sims at each final point) | 0.0651 | 0.0341 | 0.1601 | the truth-weighted W_obj and the data-discrepancy referee agree only weakly (Spearman +0.36/+0.43, p≈0.3), the winner ranks #3 cvm / #5 w1_2d; BUT the winner's data fit is at the model's own FLOOR (cvm x0.98 vs the truth point) and the per-exp agreem… |
| `24ad` | E4: held-out protocol — select on CORE T∈{20,60,100}, report on HELD-OUT T∈{5,10,40,80} (closing notebook) | 0.0651 | 0.0341 | 0.1601 | the winner GENERALISES (held-out 0.0626 ≤ core 0.0676, rank #1/8 on held-out, no penalty) but a T-based core split is a poor SELECTOR (picks 24p, held-out #4/8) because the metric weights high T; seed-42 re-run bit-identical to 24aa. FINAL: vs 24a ba… |

## 4. The best model — `24aa` (E1 role-split), and its exact config

**`W_obj(all14) = 0.0651` · `T≥40 = 0.0341` · `T≤20 = 0.1601` · μ 25.7 % · γ 33.3 % · 0/14 divergences.**
**E2 (3-seed repeat, `24ab`)**: `all14 = 0.0644 ± 0.0026` (min 0.0615, max 0.0666);
seed 42 reproduces `24aa` bit-identically.

Idea: **give μ the σ-free FWHM-only kernel + count prior (24r), give γ the 2-D σ-bearing KDE (24p/24y)**.
Exact frozen config (`two_phase` + B8 + E1):

```json
{
  "mu_gamma_split": true,
  "mu_sched_shape": "two_phase",
  "mu_tp_hi": 2.0,
  "mu_tp_lo": 0.3333333333333333,
  "mu_lr_sign2": true,
  "mu_lr_cap": 2.0,
  "mu_lr_floor": 1.0,
  "mu_freeze": true,
  "mu_freeze_frac": 0.15,
  "sigma_channel": "on",
  "cnt_prior_w": 0.25,
  "sigma_estimator": "lorentzian",
  "sigma_surrogate": true,
  "sigma_calib": "table_22g",
  "sigma_kernel": "gauss",
  "sigma_map": "none",
  "sw_sched": "none",
  "mu_reward": "kde",
  "het_frac": 0.0
}
```
Plus the frozen schedule (`lr_gamma=0.4716, gamma_anneal=0.4723, sigma_weight=0.8649, gamma_rel_cap=0.1074,
h_f_scale=4.0, h_s_scale=3.6745, h_s_min=0.05`), and the **bias-corrected μ report** (24v, 12/14 2σ coverage).

**Runner-ups**: `24r` all14 0.0672 (**best T≥40 0.0326**, μ 25.5 %) ·
`24p` (B6 hybrid) 0.0684, μ 27.7 % ·
`24g` (γ best) γ 32.5 % · `24i` (low-T sub-metric) T≤20 0.1057.

## 5. Four headline findings

**(1) μ travel is a real but *saturated* lever (Phase A, `24b`–`24j`).** Replacing the fixed `lr_mu` anneal by a
**truth-free travel budget** (`24b`, all14 0.0811→0.0730, μ 33.2→28.0 %) was the single biggest Phase-A gain, but the
schedule *shape* is second-order (`24c`), the clip is mostly idle and the net travel saturates at ≈0.43·μ_init because
the REINFORCE gradient flips sign (**FM#13/FM#14**). A hard freeze hurts low T (`24e`); the two-phase schedule (`24g`)
gives the best Phase-A low-T + γ (T≤20 0.1614, γ 32.5 %). Phase A ends exhausted as a *global* lever.

**(2) The σ channel is the μ culprit — resolved by dropping it (`24p`–`24r`, Phase B).** σ-side fixes D1/D2/D3
(`24x`–`24z`) all fail to move μ, but the **1-D FWHM-only KDE + a count prior** (`24r`) gives the series' best μ
(25.5 %) and T≥40 (0.0326). "**σ or 2-D?**" = **both**: μ wants the σ-free kernel, γ needs a real σ channel
(the 2-D γ chain is load-bearing at low T). **E1 (`24aa`) collects both in one model** — the series best.

**(3) The honest interval is the *bias-corrected* one (Phase C, `24s`–`24w`).** Every *raw* interval is dishonest:
CRB 5/14, sandwich 10/14, profile 3/14, bootstrap 0/8 (2σ μ coverage of μ_true). The **sim-at-truth bias** (`24v`) is
large and negative (median −9.4), and correcting it cuts the μ error 13.7 → 2.0 and restores **12/14** (CRB) /
**13/14** (sandwich) coverage. ⇒ the residual is a measurable, correctable **bias**, not variance, and the CRB is
*bias-blind* (**FM#12**), not too narrow.

**(4) The last residual is the line-shape / photon-count model (FM#6), and the two objectives disagree (Phase D/E).**
No σ-forward-model fix moves μ (D1/D2/D3), so the high-T μ error is the **Lorentzian line-shape + count model**.
E3 (`24ac`) makes this quantitative: the winner reproduces the real FWHM distribution **as well as the truth point does**
(`cvm_fwhm` ×0.98 of the truth-point floor), so a data-discrepancy metric and a
truth-error metric **cannot** track each other (Spearman +0.36/+0.43) — the data do **not** prefer the true μ.

## 6. Parked candidates (kept for a series 25)

- **`24i`** — coupled μ/γ travel: best low-T sub-metric of the series (T≤20 **0.1057**), at the cost of high T.
- **`24g`** — two-phase schedule: **best γ** of the series (**32.5 %**).
- **`24n`** — Student-t σ kernel: a different high-T variant (T≥40 0.0402) with a different low-T trade-off.
- **`24y`** — the calibrated-σ γ arm (γ 33.0 %, the γ source E1 reuses).
- **`24r`** — the σ-free μ model: **best T≥40 (0.0326)** and the μ tie; keep as the high-T-optimal variant.

## 7. Next steps for a series 25

1. **Fix the dead D2 plumbing.** `sigma_estimator='lorentzian'` means the `surrogate` branch in `_sims` is *never*
   entered; the D3 table `c3 = real/(κ·σ_lor)` was meant to sit *on top of* `κ·σ_CRLB`, so applied to `σ_lor` it
   returns `real/κ ≈ σ_lor` (median c3 = 1.00) — a **numerical no-op**.  This explains why `24y ≈ 24z ≈ 24aa`
   on γ (γ 33.0/33.9/33.3 %).  No verdict changes, but series 25 must wire it correctly before re-testing D.
2. **Attack FM#6 at the simulation level**: the residual high-T μ error is the line shape/photon-count model, not the
   σ channel.  Try a pseudo-Voigt **simulator**, per-scan linewidth heterogeneity, and a count model that lets the
   FWHM distribution prefer the true μ.
3. **Keep the honest interval as a first-class deliverable**: report the **bias-corrected** CRB (12/14) as the default,
   with the corrected sandwich (13/14, 2.5× wider) as the conservative variant; drop the raw CRB/profile/bootstrap.
4. **Separate the two objectives.**  Use the exp8 discrepancy (`cvm_fwhm`/`w1_2d`) for **model checking** and the
   truth-weighted `W_obj` for **parameter recovery**; do not assume they agree (E3).
5. **Use a random (not T-based) held-out split** — E4 (`24ad`) shows a core-T split selects a different model than the
   all-14 objective, so the split itself must be representative.
6. **Carry the measured chaos band (≈0.0026)** instead of the ±0.01 caveat when judging future wins.

---
*Generated from `SERIES_STATE.md` §7 + `data/processed/24*_history.json` / `24ab`/`24ac`/`24ad` side-cars.*
