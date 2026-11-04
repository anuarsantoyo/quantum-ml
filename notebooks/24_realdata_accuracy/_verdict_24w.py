#!/usr/bin/env python3
"""Fill the 24w verdict cell (coverage referee) and print the §7 row."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, '24w.ipynb')))
ref = json.load(open(os.path.join(HERE, '..', '..', 'data', 'processed', '24w_referee.json')))['summary']

def line(lbl):
    s = ref[lbl]
    return f"| {lbl} | {s['n']} | {s['h1']}/{s['n']} | {s['h2']}/{s['n']} | {s['med_ratio']:.2f} |"

VERDICT = f"""## Verdict — the honest interval is the **bias-corrected** one: corrected CRB **12/14** and corrected sandwich **13/14** at 2σ, vs raw CRB **5/14**, raw sandwich **10/14**, profile **3/14**, bootstrap **0/8** (prediction HIT)

`W_obj(all14) = 0.0672 | T>=40 0.0326 | T<=20 0.1735 | mu rel-RMSE 25.5 % | gamma rel-RMSE 35.1 % | diverged 0/14`
(point estimate unchanged — C5 is a read-only referee over the C-phase artefacts; no re-optimisation.)

**The referee table (μ coverage of μ_true across the 14 real experiments):**

| interval | n | 1σ | 2σ | median σ / dataset-CRB |
|---|---|---|---|---|
{line('CRB (raw)')}
{line('sandwich (raw)')}
{line('bootstrap (raw)')}
{line('profile quad (raw)')}
{line('CRB (bias-corrected)')}
{line('sandwich (bias-corrected)')}

**(a) HIT.** The ordering is exactly the one the C-phase predicted: **bias-corrected ≥ sandwich > CRB > profile >
bootstrap**.  Every *raw* interval is dishonest (CRB 5/14, sandwich 10/14, profile 3/14, bootstrap 0/8); every
*bias-corrected* interval restores coverage (CRB 12/14, sandwich 13/14 at 2σ; 11/14 and 12/14 at 1σ).  The
**bias-corrected sandwich is the best single interval** (13/14 at 2σ) but it is **2.51× wider** than the CRB;
the **bias-corrected CRB** is the efficient choice (12/14 at 2σ at 1.00× width).

**(b) Why each raw interval fails — three different reasons, all consistent with "the residual is a bias":**
- **CRB (5/14):** width adequate (1.00×), *centring* wrong — the estimator undershoots high-T μ (e.g. 3nW T100
  |d| = 43.4 = 5.2 CRB σ; 3nW T60 34.2 = 8.1 σ).  Bias-blind (FM#12).
- **sandwich (10/14):** the observed-information sandwich *widens* where H is ill-conditioned (median 2.51×
  CRB; 2656× at 1nW T10, 3014× at 3nW T05) — it hedges the bias by inflating the width (FM#11).
- **bootstrap (0/8):** σ is **0.24×** the CRB (4× *narrower*) — the μ landing is nearly data-independent
  (the A2 travel rule sets it), so resampling cannot see the bias (FM#12 re-scoped).
- **profile (3/14):** σ 0.64× CRB and mis-centred; the Δ(−2logL)=1 crossing resolves in only 6/14 cells (24u).

**(c) Read — Phase C closes.** The width of the CRB was never the problem; its **centring** was.  The measured
sim-at-truth bias (24v, median −9.4 μ) is the whole μ residual, and subtracting it is what turns a 5/14
interval into a 12/14 one.  The honest interval to report for this pipeline is therefore the
**bias-corrected CRB** (or the bias-corrected sandwich if 2.5× width is acceptable).  This is a *statement about
the estimator*, not a tuning knob: the same correction must be re-measured for any new forward model.

**(d) Caveat.** The bias correction uses a **single** sim-at-truth draw per experiment (one 22b/22g optimiser
run, not a distribution), so its per-cell value is noisy; and coverage is scored on 14 cells (binomial s.e.
≈ 0.13), so 12/14 vs 13/14 is not a resolved difference.  The robust statement is "corrected ⇒ ≈12–13/14;
raw ⇒ ≤10/14".

**Prediction: HIT.** The bias-corrected intervals dominate; the raw ordering (sandwich > CRB > profile >
bootstrap) is as predicted.

**FM tag:** FM#8 (bias, correctable) + FM#12 (CRB bias-blindness) + FM#11 (sandwich ill-conditioning).
**Next:** `24x` (D1, Phase D) — the bias is the whole μ residual, so fix it at the estimator/forward-model
source: an **estimator-matched** forward model (pseudo-Voigt) so sim and real share the estimator.
"""

done = False
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## Verdict'):
        c['source'] = VERDICT.splitlines(keepends=True)
        done = True
        break
assert done, 'verdict cell not found'
json.dump(nb, open(os.path.join(HERE, '24w.ipynb'), 'w'), indent=1, ensure_ascii=False)
print('24w verdict written')
print('ROW:', "| w | 24w | C5: coverage referee (CRB/sandwich/bootstrap/profile/bias-corrected over the 14) | 0.0672 | 0.0326 | 0.1735 | 25.5% | 35.1% | 0/14 | DONE (**HIT**) — bias-corrected CRB 12/14 & corrected sandwich 13/14 at 2-sigma vs raw CRB 5/14, sandwich 10/14, profile 3/14, bootstrap 0/8; raw CRB width adequate (1.00x) but mis-centred; the honest interval is the bias-corrected one, sandwich is 2.51x wider |")
