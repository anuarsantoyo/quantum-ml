# Series 21 → 25 — the story of the differentiable Monte-Carlo optimizer

Presentation prep. A narrative recap of the notebooks `21a` → `25d`, one change at a time.
Convention: **"(Anuar)"** marks the notebooks and directions the user drove by hand; **"(agent)"**
marks the systematic one-change-per-notebook execution. The split is real: across the repo Anuar
authored ~100 commits and the agent ~544, and the pattern is that Anuar sets the direction and the
pivotal questions while the agent runs the ladders and tags the failure modes (FM#6–FM#14).

---

## Series 21 — "the model that aced synthetic data fails on real data"

Opening state: `17g` had recovered all 14 *synthetic* experiments near-perfectly (μ RMSE 3.7 %,
γ 4.3 %). Series 21 asks the obvious next question: does it work on the *real* measured data?

- **21a (Anuar) — "run 17g, unchanged, on the real targets."** One change: swap the simulator-generated
  targets for the real per-experiment FWHM + fit-error columns. This is the transfer baseline.
  **Realized:** the model that was near-perfect on synthetic data does *not* transfer — the fit is off
  on the real targets. Everything downstream is about why.

- **21b (Anuar) — "maybe the estimator is wrong."** Gregor's real pipeline fitted a Voigt; ours always
  fitted a Lorentzian. Change: fit a pseudo-Voigt (sampling stays Lorentzian). **Realized:** not the
  whole story.

- **21c (agent) — "autopsy the baseline."** No change, just diagnostics on 21a. **Realized:** the σ_fit
  (fit-error) channel is mis-scaled — our simulated fit errors are up to ~4× *smaller* than the real
  ones, and the gap grows with transmission. This is the first sighting of the failure mode later named
  **FM#8**: μ is identified *through* σ ≈ f(1/√n), so a mis-scaled σ collapses μ.

- **21d (agent) — "our estimator vs Gregor's lmfit."** **Realized:** lmfit is ~4× cheaper but does NOT
  fix the observable gap, and its gradient isn't a usable surrogate.

- **21e (Anuar) — "redesign the μ REINFORCE reward."** Three reward formulations (responsibility vs
  per-run likelihood vs count-corrected credit). **Realized:** none of them fixes μ.

- **21f (agent) — "Anuar's per-run likelihood reward, alone."** μ runs away to the clip (×1.14–12.75);
  γ improves. **21g (agent)** — same reward as a *sum*: same runaway, larger. **21h (agent)** — fix the
  score denominator to the experiment's own σ_prop: mild improvement, runaway persists.

- **21i (agent) — "remove the clipping entirely."** The clip had been masking the problem. One change:
  CLIP=∞, no clamps, LR rescaled to the measured raw gradient. **Realized:** removing the clip turns
  the μ runaway into a real descent — the pivotal positive finding of the series.

- **21j (Anuar) / 21k (agent) — "does the number of runs matter?"** Runs shrink the estimator's *noise*
  (∝1/√M), not its answer.

**Series 21 takeaway:** the gap is not the estimator (21b/d), not the reward (21e–g), not the σ
denominator (21h) — it's the clipping (21i) and, underneath it, the mis-scaled σ_fit channel (21c, FM#8).

---

## Series 22 — "let's actually understand the distributions"

- **22a (agent) — "μ won't converge. Is it just a bad learning rate?"** The μ gradient swings over
  orders of magnitude, so you z-scored the μ reward (an adaptive-LR diagnostic) and rebuilt an
  interactive view: the (μ,γ) path next to the generated (FWHM, σ_fit) cloud. **Realized:** it didn't
  fix μ — the residual is *structural*, not reward-scale. The learning-rate story was a red herring.

- **22b (agent) — "then the simulation itself is off. Freeze it at the truth."** Generate a big cloud
  at the true (μ,γ) and ask: how precise is my Monte Carlo, and does its distribution match the data?
  **Realized:** the MC is precise (not the limiting factor) and matches at mid/high T, but there's a
  real gap at *low* T.

- **22c (agent, Anuar's question) — "is the gap just my estimator? Use Gregor's."** Run Gregor's
  verbatim bin+Voigt least-squares. **Realized:** the estimator explains the FWHM *spread*, but NOT
  σ_fit (lmfit overshoots it).

- **22d (agent) — "σ_fit is the bad channel, so cut it from the loss."** 1-D FWHM-only loss.
  **Realized:** it doesn't help μ — the σ_fit channel is *load-bearing*.

- **22e (Anuar) / 22f (agent) — "map out how (μ,γ) actually shapes the distribution."** A full (μ,γ)
  grid of clouds. **Realized:** γ moves the cloud along FWHM ≈ 2γ; μ mostly *compresses* it (width set
  by σ_prop/μ) — which is exactly why μ is weakly identified from FWHM alone.

- **22g (agent) — "the clean test: same estimator as Gregor, at each experiment's own truth."**
  **Realized:** even the matched estimator does NOT reproduce the real cloud, worst on σ_fit. The
  confound is the count-noise calibration (σ_prop, λ), not the estimator, optimizer, or MC precision.

**Series 22 takeaway:** μ was never a learning-rate or estimator problem — it's a σ_fit + count-noise
problem, precisely the FM#8 territory the next two series unpack.

---

## Series 23 — "the uncertainty is bias, not variance"

- **23a (agent) — "what's the Fisher/CRB for the exp6 trial_07 model, on all 14?"** **Realized:** the
  CRB as we'd been quoting it is honest *only* because it's ~√N too wide; corrected, it's still
  dishonest, because the problem is **bias, not variance**.

- **23b (agent) — "same, for the 22a model."** **Realized:** even less honest — the sharper Scott
  bandwidth *shrinks* σ while the point estimate gets worse. Tighter error bar, wronger answer.

- **23c (agent) — "is the reward change alone enough?"** No — 23c is worse at high T (23b's gain came
  from the step budget, not the reward).

- **23d (agent) — "does the target filter explain the σ collapse?"** Only modestly, and sign-flipping;
  its real payload is exposing how *fragile* the observed Fisher is.

- **23e (Anuar) — "the step budget IS the lever — calibrate μ's LR per experiment."** **Realized:** a
  uniform photon target is the wrong instrument (fixes low T, breaks high T). This hands straight into
  series 24's Phase A.

**Series 23 takeaway:** the CRB is **bias-blind (FM#12)**; the residual is a measurable bias, and the
μ step budget is the one real lever — which is exactly where series 24 starts.

---

## Series 24 — the systematic ladder (30 notebooks, one change each)

**Anuar's brief** (the whole series): *"analyse the whole project … the model must optimise correctly
to the real data; higher transitions matter more."* Anuar set the goal and the frozen metric; the
**agent** then ran a 30-notebook ladder (`24a`→`24ad`), one named change per notebook, across five
phases — A (travel), B (σ channel), C (uncertainty), D (forward model), E (combination/verification).

Baseline `24a` (agent): exp6 `trial_07` reproduced exactly — **W_obj 0.0811** (μ 33.2 %, γ 26.1 %).
The error was asymmetric: low-T μ *overshoots* (1nW T20 ×1.60), high-T μ *undershoots* (3nW T100 ×0.56).

**Phase A — μ travel / step control (`24b`–`24j`, the 23e handoff).** The idea: one global learning
rate can't serve both overshooting and undershooting cells, so give each cell a *travel budget*.
- `24b` — truth-free μ-travel budget → **0.0730**, μ 33.2→28.0 % (the biggest Phase-A gain).
- `24c`–`24f` — schedule *shape* (cosine), σ_prop²-scaled LR, freeze, and an n_iter control — all
  second-order or worse. Two new failure modes are tagged here: **FM#13** (the clip is mostly idle, net
  travel only ~0.4× μ_init) and **FM#14** (the REINFORCE gradient flips sign up to 7×/run, so net travel
  *saturates* — damping, not re-scaling, is needed).
- `24g` — two-phase schedule: best Phase-A low-T + γ, but Phase A is **exhausted as a global lever**.
- `24h`/`24i` — γ travel normalisation and coupled μ/γ travel both *falsified* (break the good high-T γ);
  `24i` is parked as the best low-T sub-metric (T≤20 0.1057).
- `24j` — truth-free μ init *falsified*, but it *measures* FM#8: μ̂/μ_true falls 1.0 → 0.22 with T,
  tracking the growing σ mismatch.

**Phase B — the σ_fit channel (`24k`–`24r`).** FM#8 is the target.
- `24k`/`24l` — σ *shape* matching and its scale-only control: both fail ⇒ FM#8 is the σ(μ)
  *conditional*, not the σ marginal/scale.
- `24m` — σ-weight anneal: falsified (the σ chain is load-bearing for γ too).
- `24n`/`24o` — Student-t σ kernel and estimator-matched σ: each buys the best-yet T≥40 (0.0402, then
  0.0353) but breaks low-T γ — the σ term can't just be down-weighted.
- `24p` — **hybrid two-role loss** (bandwidth-free cvm_fwhm for μ + KDE for γ): **0.0684**, μ 27.7 % —
  the first real sign that μ wants a σ-*free* objective.
- `24r` — **1-D FWHM-only KDE + a count prior** (σ channel off): **0.0672, T≥40 0.0326, μ 25.5 %** —
  new series best, resolving 22d's puzzle: "σ or 2-D?" = *both* — μ wants σ-free, γ needs σ.

**Phase C — uncertainty honesty (`24s`–`24w`).** Cheap side-car on the frozen point estimate.
- `24s` — data bootstrap: σ is *0.24×* the CRB (4× narrower) with 0/8 coverage ⇒ the μ landing is set by
  the travel rule, not the data ⇒ the residual is **bias**.
- `24t` — sandwich `J⁻¹KJ⁻¹`: coverage 5/14→10/14, but near-singular at low count (up to 2655× wider).
- `24u` — profile likelihood: 3/14, falsified.
- `24v` — **bias-corrected CRB**: the sim-at-truth bias (median −9.4) cuts μ error 13.7→2.0 and restores
  coverage 5/14→12/14.
- `24w` — coverage referee: the **bias-corrected** interval is the honest one (12/14 CRB, 13/14 sandwich).

**Phase D — forward model (`24x`–`24z`).** Attack the residual at the simulator level.
- `24x` — estimator-matched pseudo-Voigt forward model: falsified (μ unchanged, γ wrecked) ⇒ the μ
  residual is a **line-shape gap (FM#6)**, not an estimator mismatch.
- `24y`/`24z` — differentiable σ surrogate + exact per-cell σ table: repair γ (33.0 %) but leave μ
  unchanged ⇒ σ *scale* does not set the μ mode at any level.

**Phase E — combination / verification (`24aa`–`24ad`).**
- `24aa` — **the role-split**: μ gets the σ-free FWHM kernel + count prior (24r), γ gets the
  calibrated-σ KDE (24y). **W_obj 0.0651** (new series best all14), with the 24v bias-corrected report.
- `24ab` — repeat the winner 3× at different seeds: **0.0644 ± 0.0026** — the win is real, and the
  *measured* chaos band is ~0.0026 (4× tighter than the ±0.01 caveat).
- `24ac` — independent referee: the truth-weighted W_obj and the data-discrepancy metric agree only
  weakly (Spearman +0.36/+0.43) — the data do *not* prefer the true μ.
- `24ad` — held-out protocol: the winner *generalises*, but a T-based split is a poor selector.

**Series 24 verdict (agent-executed, Anuar-briefed):** all14 **0.0811 → 0.0651 (−20 %)**, T≥40
**0.0675 → 0.0341 (−49 %)**. Four headline findings: (1) μ travel is a real but *saturated* lever;
(2) the σ channel is the μ culprit, resolved by dropping it for μ (24r) while keeping it for γ (E1);
(3) the honest interval is the *bias-corrected* one; (4) the last residual is the **line-shape/photon
model (FM#6)**, and truth-recovery ≠ data-discrepancy.

---

## Series 25 — bandwidth-free Fisher, and the KDE-vs-Gaussian trade-off

The point-estimate ladder had stalled on FM#6, so series 25 pivots to the *uncertainty* itself — the
thing series 23/24 showed is bias-dominated and bandwidth-sensitive.

- **25a (agent) — "how sensitive is the KDE Fisher to the bandwidth?"** Sweep the bandwidth over all 14
  experiments at the truth. **Realized:** σ_γ swings ~1000× with the bandwidth (the 1/h² score), μ only
  ~6×; Scott ×1 sits at the boundary of sane. The KDE uncertainty is honest but *bandwidth-fragile*.

- **25b (agent) — "bypass the KDE: assume the summary is Gaussian."** A bandwidth-free one-Gaussian
  estimator (σ_γ = σ_FWHM/(∂FWHM/∂γ·√N)). **Realized:** at T≥40 it matches the KDE (both ~0.1–0.5 %),
  but at low T the summary is non-Gaussian (skew up to +6.6), so the Gaussian assumption breaks exactly
  where the KDE's tail mattered.

- **25c (agent) — "then use the Gaussian summary as the *optimizer* loss."** Match mean+spread of the
  2-D summary instead of the full KDE. **Realized:** the optimizer *overshoots* (μ ×1.9–4.2, γ ×2.2–3.8)
  — matching the heavy-tailed summary's two moments is a *biased* objective.

- **25d (agent) — "make μ pathwise on that same loss" (init at ½ truth, like 22a).** **Realized:** the
  optimizer now *undershoots* (μ ×0.2–0.5, γ ×0.5–0.7) — the opposite bias of 25c. So neither μ
  formulation recovers the truth, and they bias in opposite directions.

**Series 25 verdict so far:** the KDE is good at the point estimate (full distribution, respects the
tail) but carries a fragile bandwidth; the Gaussian summary sheds the bandwidth but then can't
represent the tail, so it's a biased optimizer target. 25c overshoots and 25d undershoots — the bias is
in the loss, not in the μ-gradient choice.

---

## One-line arc

21 found the model gap (FM#8: mis-scaled σ). 22 showed it's structural (σ_fit + count noise), not the
estimator or the learning rate. 23 showed the uncertainty is bias, not variance (FM#12). 24 turned the
point estimate into a 30-step ladder and won ~20 % with a role-split, ending on the line-shape gap
(FM#6). 25 turned to the uncertainty and exposed the central trade-off: full-distribution (KDE) gives
good points but fragile errors; two-moment (Gaussian) gives clean errors but biased points.
