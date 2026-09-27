#!/usr/bin/env python3
"""Fill the 24z verdict cell (D3 per-cell sigma-bias table)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, '24z.ipynb')))
obj = json.load(open(os.path.join(HERE, '..', '..', 'data', 'processed', '24z_history.json')))['objective']

VERDICT = f"""## Verdict — the exact per-(T,power) \u03c3 table leaves \u03bc unchanged (28.7 % vs 24y 28.6 %): FALSIFIED \u2014 the \u03c3 **scale** does not set the \u03bc mode at any level

`W_obj(all14) = {obj['all14']:.4f} | T>=40 {obj['hiT']:.4f} | T<=20 {obj['loT']:.4f} | mu rel-RMSE {obj['mu_rel_rmse']:.1f} % | gamma rel-RMSE {obj['gamma_rel_rmse']:.1f} % | diverged {obj['n_diverged']}/14`
(vs 24y: 0.0740 / 0.0476 / 0.1551 / \u03bc 28.6 % / \u03b3 33.0 %; vs 24r: 0.0672 / 0.0326 / \u03bc 25.5 % / \u03b3 35.1 %.)

**(a) FALSIFIED \u2014 a clean null.** c3(exp) (the exact per-cell residual of the D2 surrogate against the recorded
real \u03c3) spans **0.58\u20131.56** with median **1.002**; forcing the simulated \u03c3 onto the real \u03c3 *exactly* moves
\u03bc by essentially nothing (28.6 \u2192 28.7 %; every \u03bc/true cell within \u00b10.03 of 24y) and \u03b3 by +0.9 pp
(33.0 \u2192 33.9 %).  The result is inside the chaos band, but it is *exactly* the prediction's falsifier: the \u03bc
landing is insensitive to the \u03c3 scale.

**(b) Read \u2014 D-phase closes negatively and consistently.** D1 (estimator match: \u03bc 27.1 %, \u03b3 61.4 %), D2
(calibrated smooth surrogate: \u03bc 28.6 %, \u03b3 33.0 %), D3 (exact per-cell table: \u03bc 28.7 %, \u03b3 33.9 %) all fail to
move \u03bc toward truth.  Three independent \u03c3-forward-model fixes (\u03b1: the estimator; \u03b2: a smooth scale map; \u03b3:
an exact per-cell map) leave the \u03bc mode untouched \u21d2 **the residual high-T \u03bc error is not the \u03c3 channel's
scale/shape at all**; it is the line-shape/photon model (FM#6) interacting with the travel optimiser.  Combined
with 24r (dropping \u03c3 for \u03bc + a count prior) giving the series-best \u03bc, this says: use the \u03c3-free \u03bc treatment
and use \u03c3 only for \u03b3.

**(c) Positive residue for E1:** D2/D3 give the best post-24g \u03b3 (33.0\u201333.9 %, with 3nW T60\u201380 \u03b3/true \u2248
1.00\u20131.01) \u2014 the calibrated \u03c3 channel is the \u03b3 arm for the combination.

**Prediction: FALSIFIED** (\u03bc was predicted to move toward 1 and beat 24y; it did not).

**FM tag:** FM#8 closed on the \u03bc side (the \u03c3 channel cannot fix \u03bc at any calibration level) + FM#6.
**Next:** `24aa` (E1) \u2014 combine 24r's \u03bc treatment (1-D FWHM kernel + count prior) with the D2/D3 \u03c3-bearing
2-D \u03b3 chain, and report the 24v bias-corrected \u03bc.
"""

done = False
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## Verdict'):
        c['source'] = VERDICT.splitlines(keepends=True); done = True; break
assert done, 'verdict cell not found'
json.dump(nb, open(os.path.join(HERE, '24z.ipynb'), 'w'), indent=1, ensure_ascii=False)
print('24z verdict written')
print('ROW:', "| z | 24z | D3: per-(T,power) sigma-bias table c(exp)=real/sim on top of the D2 surrogate | 0.0764 | 0.0488 | 0.1609 | 28.7% | 33.9% | 0/14 | DONE (**FALSIFIED**) — the exact per-cell sigma table (c3 0.58-1.56, median 1.00) leaves mu unchanged (28.7% vs 24y 28.6%) => the mu landing is insensitive to the sigma SCALE at every level (D1/D2/D3 all fail) ⇒ the residual high-T mu error is the line-shape/photon model (FM#6), not the sigma channel |")
