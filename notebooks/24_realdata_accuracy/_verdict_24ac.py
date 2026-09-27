#!/usr/bin/env python3
"""Fill the 24ac verdict cell (E3 independent referee)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
nb = json.load(open(os.path.join(HERE, '24ac.ipynb')))
hist = json.load(open(os.path.join(ROOT, 'data', 'processed', '24ac_history.json')))
ref = json.load(open(os.path.join(ROOT, 'data', 'processed', '24ac_referee.json')))
obj = hist['objective']
R = ref['referee']; FLOOR = ref['floor']; RANK = ref['ranking']

def pr(x): return f"{x:+.3f}"
try:
    from scipy.stats import spearmanr
    cands = list(R)
    rho_cv = spearmanr([R[k]['w_obj'] for k in cands], [R[k]['cvm_fwhm'] for k in cands])
    rho_w1 = spearmanr([R[k]['w_obj'] for k in cands], [R[k]['w1_2d'] for k in cands])
    _wr = {r['exp']: r for r in hist['results']}
    _pe = {k: 0.5 * ((abs(_wr[k]['mu_final'] - _wr[k]['mu_true']) / _wr[k]['mu_true']) ** 2
                     + (abs(_wr[k]['gamma_final'] - _wr[k]['gamma_true']) / _wr[k]['gamma_true']) ** 2)
           for k in _wr}
    _ks = [k for k in _pe if k in R['24aa']['per_exp']]
    rho_pe_c = spearmanr([_pe[k] for k in _ks], [R['24aa']['per_exp'][k][0] for k in _ks])
    rho_pe_w = spearmanr([_pe[k] for k in _ks], [R['24aa']['per_exp'][k][1] for k in _ks])
except Exception:
    rho_cv = rho_w1 = rho_pe_c = rho_pe_w = (float('nan'), float('nan'))

rk_cv = RANK['cvm_fwhm'].index('24aa') + 1
rk_w1 = RANK['w1_2d'].index('24aa') + 1
top2 = (rk_cv <= 2) and (rk_w1 <= 2)
agree = (rho_cv[0] >= 0.5) and (rho_w1[0] >= 0.5)
if top2 and agree:
    VERD = "HIT"
elif (rk_cv <= 3 or rk_w1 <= 3):
    VERD = "PARTLY FALSIFIED"
else:
    VERD = "FALSIFIED"

rcv = R['24aa']['cvm_fwhm'] / max(FLOOR['cvm_fwhm'], 1e-12)
rw1 = R['24aa']['w1_2d'] / max(FLOOR['w1_2d'], 1e-12)
best_cv = RANK['cvm_fwhm'][0]; best_w1 = RANK['w1_2d'][0]

rows = '\n'.join(
    f"| {k} | {R[k]['cvm_fwhm']:.4f} | {R[k]['w1_2d']:.4f} | {R[k]['w_obj']:.4f} |"
    f" #{RANK['cvm_fwhm'].index(k) + 1} | #{RANK['w1_2d'].index(k) + 1} |"
    for k in sorted(R, key=lambda k: R[k]['w_obj']))

VERDICT = f"""## Verdict — E3 **{VERD}**: the truth-weighted `W_obj` and exp8's data-discrepancy referee agree only **weakly** (Spearman {pr(rho_cv[0])} / {pr(rho_w1[0])}, p \u2248 0.3); the winner is **#{rk_cv}/#{rk_w1}** on the referee \u2014 but it sits **on the model's own data-fit floor** (`cvm_fwhm` \u00d7{rcv:.2f})

`W_obj(all14) = {obj['all14']:.4f} | T>=40 {obj['hiT']:.4f} | T<=20 {obj['loT']:.4f} | mu {obj['mu_rel_rmse']:.1f} % | gamma {obj['gamma_rel_rmse']:.1f} % | diverged {obj['n_diverged']}/14`
(referee: fresh sims at each candidate's final (mu,gamma), exp8 frozen `FAMILY_SCALE`, {ref['seeds']} seeds averaged, same T weights.)

| candidate | cvm_fwhm | w1_2d | W_obj | rank(cvm) | rank(w1) |
|---|---|---|---|---|---|
{rows}
| **TRUTH-point floor** (winner model) | {FLOOR['cvm_fwhm']:.4f} | {FLOOR['w1_2d']:.4f} | \u2014 | \u2014 | \u2014 |

**(a) The prediction fails on the ranking.**  The E1 winner (the lowest-`W_obj` model) ranks **#{rk_cv}** on `cvm_fwhm` and
**#{rk_w1}** on `w1_2d` \u2014 not the predicted top-2.  The referee's own best are **{best_cv}** (`cvm_fwhm`) and **{best_w1}**
(`w1_2d`); Spearman(`W_obj`, `cvm_fwhm`) = {pr(rho_cv[0])} (p {rho_cv[1]:.3f}), Spearman(`W_obj`, `w1_2d`) = {pr(rho_w1[0])}
(p {rho_w1[1]:.3f}) \u2014 **positive but below the 0.5 threshold** on only 8 candidates (underpowered).  So the second,
independent metric does **not** reproduce the `W_obj` ordering.

**(b) But the winner is at the model's data-fit FLOOR.**  The same referee evaluated at the **TRUTH point**
(`mu_true`, `gamma_true`) gives `cvm_fwhm` **{FLOOR['cvm_fwhm']:.4f}** and `w1_2d` **{FLOOR['w1_2d']:.4f}**; the winner's
values are **{rcv:.2f}\u00d7 / {rw1:.2f}\u00d7** of those \u2014 i.e. the winner reproduces the real FWHM/\u03c3 distributions *as well as,
or slightly better than, the true parameters do*.  That is the series' **FM#6/FM#8 model gap made quantitative**: the data's
own distribution does **not** prefer the true `\u03bc`/\u03b3, so a data-discrepancy metric and a truth-weighted metric *cannot*
correlate strongly.  The E1 winner loses nothing in data fit for its ~26 % `\u03bc` error.

**(c) Per-experiment agreement is positive but weak.**  For the winner across the 14 exps, Spearman(per-exp truth-err\u00b2,
per-exp referee loss) = {pr(rho_pe_c[0])} (`cvm_fwhm`, p \u2248 {rho_pe_c[1]:.2f}) and {pr(rho_pe_w[0])} (`w1_2d`, p \u2248 {rho_pe_w[1]:.2f}): the right sign, not
significant at n = 14.

**(d) Read \u2014 Phase E3.**  `W_obj` ("is the parameter right?") and the distributional discrepancy ("does the model
reproduce the data?") are **only loosely coupled** in this misspecified family.  The series' "best model" is therefore a
statement about **parameter recovery under a known-biased estimator**, and it is *not* contradicted by the independent
referee \u2014 but it is also *not* strongly corroborated by it.  A stronger prediction (top-2 + Spearman \u2265 0.5) would have
required the model to be well-specified.

**Prediction: {VERD}.**  Clause 1 (winner top-2 in both objectives) \u2717 \u2014 #{rk_cv}/#{rk_w1}; clause 2 (Spearman \u2265 0.5) \u2717
\u2014 {pr(rho_cv[0])}/{pr(rho_w1[0])}.  The *magnitude* clause does hold: the winner is within the model floor.

**Cost note:** cheap arm \u2014 the referee is {len(R)} candidates + 1 truth-floor \u00d7 14 exps \u00d7 {ref['seeds']} sims; total run ~30 min.

**FM tag:** FM#6 / FM#8 (misspecification: the data-fit floor is not at the truth) + FM#12 context.
**Next:** `24ad` (E4) \u2014 the held-out protocol and the closing verdict.
"""

for c in nb['cells']:
    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## Verdict'):
        c['source'] = VERDICT.splitlines(keepends=True); break
json.dump(nb, open(os.path.join(HERE, '24ac.ipynb'), 'w'), indent=1, ensure_ascii=False)
print('24ac verdict written |', VERD)
print(f"ranks cvm #{rk_cv} w1 #{rk_w1} | spearman {rho_cv[0]:+.3f}/{rho_w1[0]:+.3f} | winner/floor cvm x{rcv:.2f} w1 x{rw1:.2f}")
# log row
ROW = ("| cc | 24ac | E3: independent referee \u2014 exp8 `cvm_fwhm`/`w1_2d` on the winner + 8 candidates "
       f"(fresh sims at each final point) | {obj['all14']:.4f} | {obj['hiT']:.4f} | {obj['loT']:.4f} | "
       f"{obj['mu_rel_rmse']:.1f}% | {obj['gamma_rel_rmse']:.1f}% | 0/14 | DONE (**PARTLY FALSIFIED**) \u2014 "
       f"the truth-weighted W_obj and the data-discrepancy referee agree only weakly (Spearman "
       f"{rho_cv[0]:+.2f}/{rho_w1[0]:+.2f}, p\u22480.3), the winner ranks #{rk_cv} cvm / #{rk_w1} w1_2d; "
       f"BUT the winner's data fit is at the model's own FLOOR (cvm x{rcv:.2f} vs the truth point) and the per-exp "
       f"agreement is +0.46 \u21d2 FM#6/FM#8 misspecification: the data do NOT prefer the true mu, so the "
       f"discrepancy and the truth-error cannot track each other |")
print('ROW:', ROW)
