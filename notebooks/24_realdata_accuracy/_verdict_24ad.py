#!/usr/bin/env python3
"""Fill the 24ad verdict cell (E4 held-out protocol + FINAL series verdict)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
P = os.path.join(ROOT, 'data', 'processed')
nb = json.load(open(os.path.join(HERE, '24ad.ipynb')))
O = json.load(open(os.path.join(P, '24ad_history.json')))['objective']
HO = json.load(open(os.path.join(P, '24ad_holdout.json')))
E2 = {int(k): v for k, v in (json.load(open(os.path.join(P, '24ab_history.json'))).get('e2_by_seed') or {}).items()}
REF = json.load(open(os.path.join(P, '24ac_referee.json')))
R3, F3 = REF['referee'], REF['floor']
A = json.load(open(os.path.join(P, '24a_history.json')))['objective']

T = HO['table']
best_core, best_hold = HO['best_core'], HO['best_hold']
# rank on the held-out split, ignoring the duplicate re-run of the same model
_hold_only = sorted([k for k in T if k != '24ad (winner, re-run)'], key=lambda k: T[k]['hold'])
_NH = len(_hold_only)
hold_rank = _hold_only.index('24aa') + 1
w_gap = T['24aa']['hold'] - T['24aa']['core']
p_hold_rank = _hold_only.index('24p') + 1

rows = '\n'.join(
    f"| {k} | {T[k]['core']:.4f} | {T[k]['hold']:.4f} | {T[k]['all14']:.4f} | {T[k]['hold'] - T[k]['core']:+.4f} |"
    for k in sorted(T, key=lambda k: T[k]['core']))

d_all = O['all14'] - A['all14']
d_hi = O['hiT'] - A['hiT']
d_lo = O['loT'] - A['loT']
d_mu = O['mu_rel_rmse'] - A['mu_rel_rmse']
d_g = O['gamma_rel_rmse'] - A['gamma_rel_rmse']
rcv = R3['24aa']['cvm_fwhm'] / max(F3['cvm_fwhm'], 1e-12)

VERDICT = f"""## Verdict (FINAL, E4) \u2014 the series' winner **generalises across transmission** (held-out `W_obj` {T['24aa']['hold']:.4f} \u2264 its core {T['24aa']['core']:.4f}, **rank #{hold_rank}/9 on the held-out T's**) \u2014 but a **T-based core split is a poor SELECTION design** (it picks `{best_core}`, held-out rank #{p_hold_rank}/9)

`W_obj(all14) = {O['all14']:.4f} | T>=40 {O['hiT']:.4f} | T<=20 {O['loT']:.4f} | mu rel-RMSE {O['mu_rel_rmse']:.1f} % | gamma rel-RMSE {O['gamma_rel_rmse']:.1f} % | diverged {O['n_diverged']}/14`
(this notebook re-runs the E1 winner; seed-42 is **bit-identical to 24aa**, the 4th independent reproducibility check.)

**Held-out protocol** \u2014 SELECT on CORE `T \u2208 {{20,60,100}}`, REPORT on HELD-OUT `T \u2208 {{5,10,40,80}}`
(no knob anywhere in series 24 was ever selected on the held-out split):

| candidate | core | held-out | all14 | gap |
|---|---|---|---|---|
{rows}

**(a) The winner generalises: yes.**  `24aa`'s held-out `W_obj` is **{T['24aa']['hold']:.4f}** \u2014 **#{hold_rank}/{_NH}** on the held-out transmissions, better than its own core value
({w_gap:+.4f} gap) and better than `{best_core}`'s held-out {T[best_core]['hold']:.4f}.  There is **no held-out penalty**: the model is not tuned to the specific
transmissions it was selected on.  Combined with E2 ({len(E2)}-seed repeat, all14 {sum(v['all14'] for v in E2.values()) / len(E2):.4f} \u00b1 0.0026) this is the strongest
generalisation evidence in the series.

**(b) But the core split selects the wrong model.**  Selecting on the CORE T's alone picks **`{best_core}`** (core {T[best_core]['core']:.4f}), which then ranks
**#{p_hold_rank}/{_NH}** on the held-out T's ({T[best_core]['hold']:.4f}).  The cause is structural: the frozen metric weights high T
(`w(T)=0.25+0.75\u00b7T/100`), so a core split amputates two of the three high-weight transmissions and re-weights the
remainder \u2014 and the candidate models specialise differently across T (`24p`'s \u03c3-adjacent \u03bc reward vs `24aa`'s \u03c3-free one).
\u21d2 **a T-based held-out split is not a valid selection set for this metric**; an all-14 or random split is required.

**Prediction: PARTLY FALSIFIED.**  Clause 1 (`best_core == best_hold`) \u2717 (`{best_core}` \u2260 `24aa`); clause 2 (winner's held-out within +0.02 of core) \u2713
({w_gap:+.4f}).  The series' model **does** generalise; the *selection procedure* does not.

---

## FINAL SERIES-24 VERDICT \u2014 did the series meet Anuar's goal?

*vs the 24a baseline (`experiment_6 trial_07`)*: all14 **{A['all14']:.4f} \u2192 {O['all14']:.4f}** ({d_all:+.4f}, {100 * d_all / A['all14']:+.0f} %) \u00b7
T\u226540 **{A['hiT']:.4f} \u2192 {O['hiT']:.4f}** ({d_hi:+.4f}, {100 * d_hi / A['hiT']:+.0f} %) \u00b7 T\u226420 **{A['loT']:.4f} \u2192 {O['loT']:.4f}** ({d_lo:+.4f}) \u00b7
\u03bc rel-RMSE **{A['mu_rel_rmse']:.1f} % \u2192 {O['mu_rel_rmse']:.1f} %** ({d_mu:+.1f} pp) \u00b7 \u03b3 rel-RMSE **{A['gamma_rel_rmse']:.1f} % \u2192 {O['gamma_rel_rmse']:.1f} %** ({d_g:+.1f} pp).

- **The primary objective (all 14, weighted toward high transmission) is met and improved a lot.**  `W_obj(all14)`
  is down **{100 * d_all / A['all14']:.0f} %** and the high-transmission split \u2014 the part Anuar said matters most \u2014 is down
  **half** ({100 * d_hi / A['hiT']:.0f} %), with the dominant \u03bc channel improving **{abs(d_mu):.1f} pp** (33.2 \u2192 25.7 %).  All 14 experiments are
  approximated, none diverges, and the winner's every-T \u03bc ratio sits in 0.60\u20130.95 (vs the baseline's 0.56\u20131.60 overshoot).
- **But the improvement is NOT Pareto.**  \u03b3 rel-RMSE **regressed** ({A['gamma_rel_rmse']:.1f} \u2192 {O['gamma_rel_rmse']:.1f} %; the *baseline* still holds the series' best \u03b3) and the
  lowest-T split is slightly worse ({A['loT']:.4f} \u2192 {O['loT']:.4f}): every \u03bc-focused mechanism in the series traded \u03b3/low-T for \u03bc/high-T.
  The goal bands (\u03bc \u2264 15 %, \u03b3 \u2264 10 %) are **not** reached \u2014 the residual is the FM#6 line-shape/photon-count model,
  which no \u03c3-channel fix (D1/D2/D3) or travel fix (A) could remove, and which E3 shows makes the *data* prefer a
  non-true \u03bc (`cvm_fwhm` \u00d7{rcv:.2f} of the truth-point floor).
- **Honest interval: delivered.**  The bias-corrected CRB (12/14 at 2\u03c3) is the series' uncertainty product; every raw
  interval is dishonest.
- **Bottom line:** the series **met the weighted/high-T goal** (big, reproducible, generalising gains in the channel that
  matters and in the high-T split) but **did not solve the real-data gap**: the remaining error is the line-shape/photon
  model (FM#6), and the winner buys high-T accuracy with low-T \u03b3.  A series 25 should attack the simulator, not the loss.

**FM tag:** FM#6 (line shape / photon count \u2014 the last residual), FM#8 (estimator bias, correctable, quantified in E3), FM#12 (CRB bias-blind).
**Series 24 CLOSED.**  Executive summary: `SUMMARY.md`.  Handoff to a series 25 in `SERIES_STATE.md` \u00a79.
"""

for c in nb['cells']:
    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## Verdict'):
        c['source'] = VERDICT.splitlines(keepends=True); break
json.dump(nb, open(os.path.join(HERE, '24ad.ipynb'), 'w'), indent=1, ensure_ascii=False)
print('24ad verdict written')
ROW = ("| dd | 24ad | E4: held-out protocol \u2014 select on CORE T\u2208{20,60,100}, report on HELD-OUT T\u2208{5,10,40,80} "
       "(closing notebook) | "
       f"{O['all14']:.4f} | {O['hiT']:.4f} | {O['loT']:.4f} | {O['mu_rel_rmse']:.1f}% | {O['gamma_rel_rmse']:.1f}% | 0/14 | "
       f"DONE (**PARTLY FALSIFIED**) \u2014 the winner GENERALISES (held-out {T['24aa']['hold']:.4f} \u2264 core {T['24aa']['core']:.4f}, "
       f"rank #{hold_rank}/{_NH} on held-out, no penalty) but a T-based core split is a poor SELECTOR (picks {best_core}, "
       f"held-out #{p_hold_rank}/{_NH}) because the metric weights high T; seed-42 re-run bit-identical to 24aa. "
       f"FINAL: vs 24a baseline all14 {A['all14']:.4f}\u2192{O['all14']:.4f} ({100 * d_all / A['all14']:+.0f}%), T\u226540 {A['hiT']:.4f}\u2192{O['hiT']:.4f} "
       f"({100 * d_hi / A['hiT']:+.0f}%), \u03bc {A['mu_rel_rmse']:.1f}\u2192{O['mu_rel_rmse']:.1f}% but \u03b3 {A['gamma_rel_rmse']:.1f}\u2192{O['gamma_rel_rmse']:.1f}% "
       f"(NOT Pareto) \u21d2 the weighted/high-T goal is met, the FM#6 line-shape gap is not \u2014 series 24 CLOSED |")
print('ROW:', ROW)
