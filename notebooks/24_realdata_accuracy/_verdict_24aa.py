#!/usr/bin/env python3
"""Fill the 24aa verdict cell (E1 combination)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, '24aa.ipynb')))
obj = json.load(open(os.path.join(HERE, '..', '..', 'data', 'processed', '24aa_history.json')))['objective']

VERDICT = f"""## Verdict — E1 combination is the **new series best**: `W_obj(all14)` **0.0651** (24r 0.0672) with the 2nd-best \u03b3 (**33.3 %**) and the best low-T split of any global config (**0.1601**); \u03bc essentially tied (25.7 % vs 25.5 %)

`W_obj(all14) = {obj['all14']:.4f} | T>=40 {obj['hiT']:.4f} | T<=20 {obj['loT']:.4f} | mu rel-RMSE {obj['mu_rel_rmse']:.1f} % | gamma rel-RMSE {obj['gamma_rel_rmse']:.1f} % | diverged {obj['n_diverged']}/14`
(vs 24r: 0.0672 / **0.0326** / 0.1735 / **25.5 %** / 35.1 %; vs 24y: 0.0740 / 0.0476 / 0.1551 / \u03bc 28.6 % / \u03b3 33.0 %.)

| metric | 24r (prev best) | **24aa (E1)** | \u0394 |
|---|---|---|---|
| W_obj all14 | 0.0672 | **0.0651** | \u22120.0021 |
| W_obj T\u226540 | **0.0326** | 0.0341 | +0.0015 |
| W_obj T\u226420 | 0.1735 | **0.1601** | \u22120.0134 |
| \u03bc rel-RMSE | **25.5 %** | 25.7 % | +0.2 pp |
| \u03b3 rel-RMSE | 35.1 % | **33.3 %** | \u22121.8 pp |

**(a) HIT (qualified) \u2014 the role-split composes.** Giving \u03bc the \u03c3-free 1-D FWHM kernel + count prior (24r) while
giving \u03b3 the full calibrated 2-D \u03c3-bearing KDE (D2/D3) collects both: \u03b3 falls **35.1 \u2192 33.3 %** (the
second-best of the series behind 24g's 32.5 %, with 3nW T60\u2013100 \u03b3/true = 0.997\u20131.002) and `T\u226420` falls
**0.1735 \u2192 0.1601**, while \u03bc stays at the series best (25.7 %).  Mid-T \u03bc is much improved (1nW/3nW T40 =
0.945 / 0.925 vs 24r 0.72 / 0.69).

**(b) The bias correction composes too.** The 24v bias-corrected \u03bc table moves the \u03bc-channel contribution to
`W_obj` from **0.0648 \u2192 0.0021** and restores **12/14** 2\u03c3 CRB coverage (identical to 24v, since the raw \u03bc
table is nearly the same).  So E1's honest report = the raw table above + the bias-corrected one.

**(c) Read \u2014 Phase E1.** The consistent picture across D1\u2013D3 (the \u03c3 channel cannot fix \u03bc at any calibration)
and 24r/E1 (\u03bc wants \u03c3-free + count prior; \u03b3 wants a real \u03c3 channel) is now realised in one model.  **E1 is the
new series best on `all14` (0.0651), on the low-T split (0.1601) and on \u03b3 (33.3 %)**, at a \u03bc cost of 0.2 pp and
a high-T cost of 0.0015.  Caveat: the `all14` margin over 24r is **\u22120.0021, inside the \u00b10.01 chaos band** \u2014
E2 (repeat 3\u00d7 at different seeds) must confirm it before it is called a genuine win.

**Prediction: HIT (qualified).** `all14` < 0.0672 \u2713 (0.0651), \u03b3 < 35.1 % \u2713 (33.3 %); the "\u03bc \u2264 25.5 %"
clause is missed by 0.2 pp (a tie inside the chaos band).

**Cost note:** cheap arm (Lorentzian fit, \u03c3 from the surrogate+table) \u2014 **~20 min**.

**FM tag:** FM#8 (resolved: \u03bc = \u03c3-free + count prior; \u03b3 = calibrated \u03c3 channel) + FM#6 (remaining low-T \u03b3 deficit).
**Next:** `24ab` (E2) \u2014 repeat the E1 winner 3\u00d7 at different `SEED` to separate signal from the chaos band, then
`24ac` (E3, exp8\u2013`cvm_fwhm`/`w1_2d` referee) and `24ad` (E4, held-out tune/report).
"""

done = False
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## Verdict'):
        c['source'] = VERDICT.splitlines(keepends=True); done = True; break
assert done, 'verdict cell not found'
json.dump(nb, open(os.path.join(HERE, '24aa.ipynb'), 'w'), indent=1, ensure_ascii=False)
print('24aa verdict written')
print('ROW:', "| aa | 24aa | E1: mu = 1-D FWHM kernel + count prior (24r), gamma = 2-D calibrated-sigma KDE (D2/D3) + 24v bias-corrected report | **0.0651** | 0.0341 | **0.1601** | 25.7% | **33.3%** | 0/14 | DONE (**HIT, qualified — NEW SERIES BEST all14**) — the role-split collects 24r's mu (25.7% ~ 25.5%) with the calibrated sigma's gamma (33.3% vs 35.1%) and the best global low-T (0.1601); all14 0.0672->0.0651 (−0.0021, inside the chaos band ⇒ E2 must confirm); bias-corrected mu table restores 12/14 2-sigma coverage |")
