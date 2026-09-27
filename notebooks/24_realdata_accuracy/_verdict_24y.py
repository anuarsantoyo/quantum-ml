#!/usr/bin/env python3
"""Fill the 24y verdict cell (D2 differentiable surrogate of the fit-error map)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, '24y.ipynb')))
obj = json.load(open(os.path.join(HERE, '..', '..', 'data', 'processed', '24y_history.json')))['objective']

VERDICT = f"""## Verdict — the calibrated \u03c3 surrogate does **not** fix \u03bc (\u03bc 28.6 % vs 24r 25.5 %: FALSIFIED) but it *does* repair \u03b3 (33.0 %, near the series best) and the low-T split (0.1551 vs 0.1735)

`W_obj(all14) = {obj['all14']:.4f} | T>=40 {obj['hiT']:.4f} | T<=20 {obj['loT']:.4f} | mu rel-RMSE {obj['mu_rel_rmse']:.1f} % | gamma rel-RMSE {obj['gamma_rel_rmse']:.1f} % | diverged {obj['n_diverged']}/14`
(vs 24r: 0.0672 / 0.0326 / 0.1735 / \u03bc 25.5 % / \u03b3 35.1 %; vs 24x: 0.1953 / \u03bc 27.1 % / \u03b3 61.4 %.)

**(a) The surrogate works as designed \u2014 on \u03b3.** Fitted from the 22g clouds,
`\u03ba(FWHM,n,\u03b3) = exp(10.130)\u00b7FWHM^(\u22129.204)\u00b7n^(\u22120.317)\u00b7\u03b3^(9.206)` (R\u00b2 = 0.662; the sim\u2192real \u03c3 ratio runs
1.0\u21924.1 across the 14 cells).  With the \u03c3 channel now carrying a calibrated, differentiable value, the **\u03b3
chain recovers**: \u03b3 rel-RMSE **33.0 %** (24r 35.1 %, 24o 38.4 %, 24x 61.4 %) \u2014 the second-best of the series
behind 24g's 32.5 % \u2014 with 3nW T60/80/100 at **\u03b3/true \u2248 0.995\u20131.001** and `W_obj(T\u226420)` **0.1551** (24r
0.1735).  This is the first \u03c3 channel that is *net positive* for \u03b3.

**(b)\u2026 but \u03bc is worse, not better.** \u03bc rel-RMSE **28.6 %** (24r 25.5 %, 24x 27.1 %) and the \u03bc/true table
still undershoots everywhere (1nW T60 0.586, 3nW T20 0.642).  Because the \u03bc reward still consumes the \u03c3 kernel,
the calibrated-\u03c3 pull re-introduces a \u03bc attractor \u2014 the opposite of the prediction.  So the \u03c3 **scale** is
*not* what sets the \u03bc mode (a per-cell calibration of the same channel cannot move \u03bc toward truth).

**(c) Read \u2014 the D-phase separation.** D1 (28.6 % / 61.4 %) vs D2 (28.6 % / 33.0 %): matching the estimator at
source does nothing for \u03bc; calibrating the \u03c3 *value* fixes \u03b3 but not \u03bc.  The two channels want different
things: **\u03bc wants the \u03c3-free FWHM kernel + count prior (24r); \u03b3 wants a real \u03c3 channel (D2).**  That is
exactly the E1 role-split, and D2 is the best \u03b3 arm to carry into it.

**Prediction: FALSIFIED** (\u03bc was predicted to improve on 24r; it did not).  Recorded positive side-effect: the
best post-24g \u03b3 and low-T split.

**FM tag:** FM#8 (the \u03c3 channel is \u03b3-informative but \u03bc-pulling) + FM#6.
**Next:** `24z` (D3) \u2014 the exact per-(T,power) \u03c3-bias table c(exp) on top of the surrogate (cheapest D form).
"""

done = False
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## Verdict'):
        c['source'] = VERDICT.splitlines(keepends=True); done = True; break
assert done, 'verdict cell not found'
json.dump(nb, open(os.path.join(HERE, '24y.ipynb'), 'w'), indent=1, ensure_ascii=False)
print('24y verdict written')
print('ROW:', "| y | 24y | D2: differentiable surrogate sigma = kappa(FWHM,n,gamma)*sigma_CRLB (22g-calibrated, R2 0.66) | 0.0740 | 0.0476 | **0.1551** | 28.6% | **33.0%** | 0/14 | DONE (**FALSIFIED** on mu) — the calibrated sigma repairs gamma (33.0%, near the 24g best 32.5%; 3nW T60-100 gamma ~1.00) and the low-T split (0.1551 vs 0.1735) but mu worsens (28.6% vs 24r 25.5%) => the sigma SCALE does not set the mu mode; mu wants the sigma-free FWHM kernel (24r), gamma wants the real sigma channel (D2) |")
