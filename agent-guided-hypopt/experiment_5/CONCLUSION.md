# experiment_5 — CONCLUSION

**AG-HYPOPT on the series-21 μ mechanism · real data · 40 trials · 2026-09-16 23:03 → 2026-09-17 05:05 (6 h 02 m)**

---

## 1. What was asked

After series 21 established that the μ channel was dead/biased *because the gradient clip was masking a
scale problem* (21i: removing the clip and rescaling `LR_MU` to 1e-4 moved μ from rel-RMSE 550.6 % → 40.6 %,
bias +366 % → +7.3 %), the question for this campaign was:

> **Given Anuar's μ reward and the `σ_prop` score, which learning-rate schedules work best?**

**Frozen mechanism** (all of series 21): 17g model, 2-D `(FWHM, σ_fit)` KDE kernel with Scott bandwidths,
Lorentzian 2-parameter fit, z-form γ-score, real measured targets, init at `0.5 × truth`, `n_runs=100`,
`n_iter=30`, μ reward = **Anuar's per-run likelihood** (`r_j = mean_i log W_ij`), μ score denominator =
the experiment's own **`σ_prop`**, **no gradient clip and no parameter clamps**, divergence **guard**
(stop if μ∉(0.2, 1500) or γ∉(0.02, 500)).

**Searched (4 knobs)** — no `clip`, no `n_runs` (both shown to be dead ends in 21j):

| knob | range | meaning |
|---|---|---|
| `lr_mu` | 0.005 – 0.5 | μ step |
| `mu_anneal` | 0 – 1 | μ LR decay (`lr_mu·(1 − mu_anneal·t/n)`) |
| `lr_gamma` | 0.05 – 1.0 | γ step |
| `gamma_anneal` | 0 – 0.9 | γ LR decay |

**Objective** (pre-registered in `trials.json`): per-cell relative squared error, **capped at 1.0**,
averaged with a **transmission weight** `w(T) = 0.25 + 0.75·T/100` — i.e. graded toward high transmission,
the series-21 goal. Benchmark = **8 experiments** (1nW/3nW × Trans05/20/60/100). Cap `MAX_TRIALS=40`.

---

## 2. How the campaign ran

- **40/40 trials**, every one executed **cell-by-cell** (`nbrun.py`, persistent kernel), each registered in
  `trials.json` and **committed + pushed** before the next. ~8–12 min per trial (median 8.3 min).
- **Two phases:** the first **5 trials were uniform exploration**; from trial 06 the proposals were
  **model-informed** with expected-improvement (EI) values (TPE good/bad densities, plus reserved
  fully-random explore slots). The agent picked the trial from the candidate table and wrote the analysis —
  that judgement mattered (§5).
- Every trial notebook is self-contained (phase-space path figure + parallel-coordinates view + report table).

---

## 3. Headline result

**Best: `trial_21` — objective 0.1074**

```
lr_mu = 0.1669   mu_anneal = 0.3455   lr_gamma = 0.4716   gamma_anneal = 0.4723
μ travel = 0.138;  0 diverged cells
μ rel-RMSE 77.9 %   γ rel-RMSE 21.7 %
```

**But the honest level is ≈ 0.108 – 0.113, not 0.1074.** `trial_21` is partly a lucky *divergence-free draw*:
`trial_26` is a near-exact copy (every knob within ~0.007) and it **diverged** at the same cell, scoring 0.1368.
The runner-up family — `trial_30` 0.1083, `trial_22` 0.1095, `trial_32` 0.1097, `trial_08` 0.1099,
`trial_15` 0.1103 — sits at the same place.

Distribution over the 40 trials: **27/40 ≤ 0.114**, 9/40 > 0.12 (all divergence-affected), range 0.1074 – 0.1975.

### The optimum is a narrow μ-travel ridge
Define **μ travel = `lr_mu · (1 − mu_anneal/2)`** (the campaign's own diagnostic — the effective μ step
integrated over the linear decay).

| travel bin | n | min obj | mean obj |
|---|---|---|---|
| 0.060–0.085 | 3 | 0.1188 | 0.1322 |
| 0.085–0.105 | 4 | 0.1135 | 0.1395 |
| 0.105–0.125 | 3 | 0.1126 | 0.1140 |
| **0.125–0.145** | **27** | **0.1074** | 0.1151 |
| 0.145–0.175 | 1 | 0.1188 | 0.1188 |
| 0.175–0.220 | 1 | 0.1149 | 0.1149 |
| 0.220–0.300 | 1 | 0.1449 | 0.1449 |

A quadratic fit on the 31 non-diverged trials puts the **vertex at travel 0.1385** — matching the best
configuration (0.1381) and the campaign's repeated "optimum travel ≈ 0.137–0.138" reads. The axis falls off
**monotonically on both sides**: too little travel leaves the heavily-weighted 1nW high-T cells undershooting;
too much travel blows up the mid/low-T μ overshoot.

The γ knobs matter only weakly — `lr_gamma` ≈ 0.41–0.48 with `gamma_anneal` ≈ 0.42–0.53 is marginally best
(`trial_21`'s balanced `lr_gamma ≈ gamma_anneal ≈ 0.47` edged `trial_08` by 0.0025), while `lr_gamma = 0.141`
or 0.344 or 0.80 all land mid-plateau. **No knob correlation exceeds |r| = 0.2** — because the plateau, not
the knobs, owns the residual.

### Per-cell behaviour at the best trial (trial_21)

```
              1nW T05  T20    T60    T100  | 3nW T05   T20    T60    T100
μ   Δ%         +17.2  +155.5  -2.9   -4.1  | +146.0   +30.1  -21.8  -36.8
γ   Δ%         -44.3  -26.5  -10.0   -1.3  |  -30.1    -8.3   +4.6   +2.0
```

The shape is systematic and unchanged from 21i/21j: **μ overshoots at low/mid T and undershoots at high T**
(the 3nW high-T cells never reach truth), while **γ is excellent at high T (1nW T100 −1.3 %, 3nW T60/100
+4.6/+2.0 %) and under-recovers at low T** — exactly the graded picture the series has been chasing.

---

## 4. The dominant finding: the sporadic γ divergence

**9 of 40 trials** hit the divergence guard on a γ blow-up: `trial_02, 03, 04, 10, 17, 26, 28, 29, 34`.
All at **1nW T05** except `trial_10`, which blew up at **3nW T60**. Each costs **+0.02 – 0.07** on the
objective (the capped 1.0 penalty on that cell), and each produced a huge nonsense γ (10¹⁰ – 10¹⁶).

**It is not tunable by the four knobs.** The evidence is decisive: at the optimum, four near-identical
schedules survived (`21, 22, 25`, and `30` nearby) and four diverged (`26, 28, 29, 34`) — `trial_26` matching
`trial_21` to within 0.007 on every parameter. Cutting `lr_gamma` to 0.60 and raising `gamma_anneal` to 0.85
(`trial_04`) did **not** remove it; changing the travel regime did not move it (`trial_17`).

⇒ At the optimum the divergence is roughly a **coin flip** — a chaotic instability of the unbounded low-count
γ gradient (the same object 21i/21j identified). **The honest expectation at the best schedule is the
surviving level ≈ 0.108 – 0.113**, and `trial_21`'s headline 0.1074 should not be quoted without that caveat.

---

## 5. How well did the AG-HYPOPT idea work?

**It worked as designed — but its value was front-loaded.**

**What it did well**
- **It found the optimum fast.** Best-so-far: `0.1127` (t01) → `0.1135` (t05) → **`0.1099` (t08)** — the
  productive region was identified in ~8 trials (~1 h of the 6 h campaign).
- **It pinned a narrow optimum in a 4-D box.** After t08 essentially every proposal sat inside
  `lr_mu 0.15–0.18`, `mu_anneal 0.34–0.36`, `lr_gamma 0.41–0.48`, `gamma_anneal 0.42–0.53`, and *all*
  sub-0.111 runs sit in it. Finding a ridge that tight is the thing HPO is for.
- **The agent's judgement added signal beyond pure EI.** In the uniform phase it deliberately probed the
  *opposite* side of the travel axis (t03: travel 0.074) instead of re-sampling; and at t21 it **overrode a
  tied top-EI candidate** (`lr_gamma = 0.141`, far outside the proven-safe band) in favour of the balanced
  one — which became the best trial. A pure-EI policy would likely have taken the risky candidate.
- **The registry is a clean, transferable artefact**: 40 configs × objective × uncertainty × written
  analysis, each with its own figure, reproducible from `trials.json`.

**What limited it**
- **Diminishing returns.** The plateau was effectively reached by ~t10; **30 of the 40 trials (75 %)
  re-measured the same region**, and only t21 (13 trials later) beat t08 — by 0.0025, i.e. inside the hazard
  noise. The last 30 trials bought very little.
- **The objective is stochastic, and the signal is smaller than the noise.** The divergence injects a
  +0.02–0.06 jump ~1 time in 3, while the interesting spread of the plateau is ~0.006 wide. TPE cannot model
  that — a hazard that doesn't depend on the knobs simply looks like noise, so the surrogate flattens.
- **The search space was small and pre-narrowed** (we set `lr_mu`'s range from the measured raw gradient),
  so the campaign was as much a *confirmation* as a search.

**Verdict:** the algorithm + agent loop behaved exactly as intended (informed proposals, sensible decisions,
reproducible bookkeeping, no wasted structural drift). The limit was the **problem**, not the method: a
4-D plateau plus an unmodellable stochastic failure mode. The practical lesson is about **stopping and
protocol**, not about the optimiser.

**Recommendations for future campaigns**
1. **Add an explicit stop rule** (e.g. stop after *N* consecutive trials with no improvement beyond the
   objective's own uncertainty) instead of always running to the cap — here it would have stopped near t10–t15.
2. **Treat the divergence as a separate mechanism problem**, not something schedules should fix: it is
   the only remaining lever on the objective (see §6).
3. **For stochastic objectives, repeat**: ≥2 runs per configuration (or objective averaging) so the
   surrogate sees the mean rather than one draw of the hazard.
4. Keep the good parts: cell-by-cell execution with commit-per-trial, the pre-registered objective, and the
   ✍️ analysis cells — including the agent's freedom to reject a high-EI candidate.

---

## 6. Where the series goes next

1. **γ-specific safeguard for the low-count cells** — the single highest-value item. Options: a γ-only clip,
   a γ LR annealed to ~0, or a robustified γ gradient/KDE treatment. This is what still separates ~0.110
   from anything better.
2. **Adopt the winning schedule as the series-21 default**: `lr_mu 0.167`, `mu_anneal 0.346`,
   `lr_gamma 0.472`, `gamma_anneal 0.472` (μ travel 0.138).
3. **Mechanism cleanup**: make the μ gradient a *mean* over draws instead of a sum — the asymmetry with γ
   (whose gradient is M-invariant via the row-normalised `w`) that forces the awkward `LR ∝ 1/n_runs` scaling.
4. **Reporting hygiene**: quote γ rel-RMSE *excluding* diverged cells, and have the guard store the **last
   in-band value** rather than the exploded one.

---

## 7. Provenance

- **Experiment:** `agent-guided-hypopt/experiment_5/` — `ag_hypopt.py`, `space.json`, `trials.json`
  (protocol + 40 entries), `trial_01.ipynb … trial_40.ipynb` (each self-contained with figures), `README.md`.
- **Best trial:** `trial_21.ipynb` (objective 0.1074); runner-up `trial_30.ipynb` (0.1083).
- **Repo state:** branch `develop`, campaign closed at commit `5b9c3de` (`f7e5c5f..5b9c3de`), working tree clean.
- **Series context:** 21i (clip removal + LR rescale, μ 550 %→41 %) · 21j (N_RUNS sweep — no benefit) ·
  this campaign (schedule tuning — ridge at travel 0.138, floor ~0.108–0.113, γ divergence the residual).
