#!/usr/bin/env python3
"""Fill the 24x verdict cell (D1 estimator-matched pv forward model)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, '24x.ipynb')))
obj = json.load(open(os.path.join(HERE, '..', '..', 'data', 'processed', '24x_history.json')))['objective']

VERDICT = f"""## Verdict — the estimator-matched (pseudo-Voigt) forward model does **not** fix \u03bc and wrecks \u03b3 (\u03bc 27.1 % vs 24r 25.5 %; \u03b3 61.4 % vs 35.1 %) \u2014 prediction FALSIFIED

`W_obj(all14) = {obj['all14']:.4f} | T>=40 {obj['hiT']:.4f} | T<=20 {obj['loT']:.4f} | mu rel-RMSE {obj['mu_rel_rmse']:.1f} % | gamma rel-RMSE {obj['gamma_rel_rmse']:.1f} % | diverged {obj['n_diverged']}/14`
(vs 24r: 0.0672 / 0.0326 / 0.1735 / \u03bc 25.5 % / \u03b3 35.1 %; vs 24o's \u03c3-only pv match: 0.0796 / 0.0353 / \u03bc 28.6 % / \u03b3 38.4 %.)

**(a) \u03bc did not move toward truth \u2014 it moved *down*.** The per-cell \u03bc/true table *deepens* the undershoot
(e.g. 1nW T60 0.596, 3nW T100 0.637, 1nW T40 0.670); \u03bc rel-RMSE **27.1 %** (24r 25.5 %, 24o 28.6 %).  The best
cell is 1nW T05 (0.927).  So replacing the analytic Lorentzian CRLB with the pseudo-Voigt estimator at the
FWHM channel too does **not** fix the \u03c3(\u03bc) conditional at source.

**(b) \u03b3 collapses.** \u03b3 rel-RMSE **61.4 %** (worse than *every* A/B/C notebook); 1nW T10 \u03b3/true 0.131, 3nW T05
0.137.  The cause is now doubly clear: the pv estimator's FWHM (Voigt FWHM via Olivero-Longbothum, inflated
~84 % at truth per 22g) *and* the pv \u03c3 channel (24o) both feed the \u03b3 score, and a coherent-but-wrong line model
pulls \u03b3 hard down.  24o's "the \u03b3 chain should not consume \u03c3" is confirmed and extends to the FWHM channel: an
estimator-weighted \u03b3 chain is fragile.

**(c) Read.** Matching the estimator at the FWHM level is the wrong lever for \u03bc (it is a *line-shape* mismatch,
FM#6, not an estimator mismatch) and it is actively harmful for \u03b3.  The \u03c3 channel remains the only \u03bc-side
lever, and even a "matched" \u03c3 channel (24o) only reaches \u03bc 28.6 %.  This closes D1 negatively and motivates the
**calibrated** forms D2/D3, which keep the \u03c3 channel but correct its *value*, and the E1 role-split.

**Prediction: FALSIFIED.** (\u03bc was predicted to improve; it did not.  \u03b3 was predicted to possibly pay; it paid
catastrophically.)

**Cost note (engineering):** one pv fit per run (3-parameter, n_iters=80) at the full protocol budget
(N_RUNS=100, N_ITER=30) \u2014 **92.7 min wall** (\u2248 5.3\u201310 min/experiment, growing with the photon count).

**FM tag:** FM#8 (estimator part \u2014 does **not** fix \u03bc) + FM#6 (line-shape weight in the \u03b3 chain).
**Next:** `24y` (D2) \u2014 a **smooth differentiable surrogate** \u03c3 = \u03ba(FWHM,n,\u03b3)\u00b7\u03c3_CRLB calibrated to the
22g (real/estimator) \u03c3 map, keeping the cheap Lorentzian FWHM channel.
"""

done = False
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## Verdict'):
        c['source'] = VERDICT.splitlines(keepends=True); done = True; break
assert done, 'verdict cell not found'
json.dump(nb, open(os.path.join(HERE, '24x.ipynb'), 'w'), indent=1, ensure_ascii=False)
print('24x verdict written')
print('ROW:', "| x | 24x | D1: estimator-matched pseudo-Voigt forward model (pv FWHM + sigma, 2-D KDE) | 0.1953 | 0.1597 | 0.3044 | 27.1% | 61.4% | 0/14 | DONE (**FALSIFIED**) — matching the estimator at the FWHM channel does NOT fix mu (27.1% vs 24r 25.5%; table moves *down*) and wrecks gamma (61.4% vs 35.1%) => the mu residual is a line-shape/model gap (FM#6), not an estimator mismatch; 92.7 min |")
