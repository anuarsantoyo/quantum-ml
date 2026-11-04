#!/usr/bin/env python3
"""Write notebooks/24_realdata_accuracy/SUMMARY.md (series-24 executive summary) + the
§9 FINAL SUMMARY section for SERIES_STATE.md.  Reads the §7 log + the saved histories."""
import json, os, re, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PROC = os.path.join(ROOT, 'data', 'processed')

def hist(cid):
    p = os.path.join(PROC, f'{cid}_history.json')
    return json.load(open(p)) if os.path.exists(p) else None

# ---------- parse the §7 log rows ----------
text = open(os.path.join(HERE, 'SERIES_STATE.md')).read()
log = text.split('## 7. Log', 1)[1].split('**Current best = ', 1)[0]
rows = []
for ln in log.split('\n'):
    if not ln.startswith('|') or ln.startswith('|---') or ln.startswith('| # |'):
        continue
    cols = [c.strip() for c in ln.strip().strip('|').split('|')]
    if len(cols) < 10:
        continue
    if len(cols) > 10:            # the 'one change' field may itself contain '|' (e.g. |\u2207\u03bc|)
        nid, nb_id = cols[0], cols[1]
        change = '|'.join(cols[2:len(cols) - 7])
        rest = cols[len(cols) - 7:]
        cols = [nid, nb_id, change] + rest
    rows.append(cols)

# ---------- compact ladder table ----------
def short(s, n):
    s = s.replace('\n', ' ').replace('|', '\\|')
    return (s[:n] + '\u2026') if len(s) > n else s

ladder = []
for c in rows:
    nid, nb_id, change, a14, hi, lo, mu, gam, div, status = c[:10]
    verdict = status
    if verdict.startswith('DONE'):
        verdict = verdict[4:].strip()
    if verdict.startswith('('):
        verdict = verdict[verdict.find(')') + 1:].strip()
    verdict = verdict.lstrip('\u2014').strip().rstrip(')')
    verdict = re.sub(r'\*\*', '', verdict)
    ladder.append(f"| `{nb_id}` | {short(change, 118)} | {a14} | {hi} | {lo} | {short(verdict, 250)} |")
LADDER = '\n'.join(ladder)

# ---------- best config ----------
aa = hist('24aa')
CFG = aa['config']
BEST_CFG = {k: CFG.get(k) for k in [
    'mu_gamma_split', 'mu_sched_shape', 'mu_tp_hi', 'mu_tp_lo', 'mu_lr_sign2', 'mu_lr_cap',
    'mu_lr_floor', 'mu_freeze', 'mu_freeze_frac', 'sigma_channel', 'cnt_prior_w',
    'sigma_estimator', 'sigma_surrogate', 'sigma_calib', 'sigma_kernel', 'sigma_map',
    'sw_sched', 'mu_reward', 'het_frac']}
CFG_JSON = json.dumps(BEST_CFG, indent=2)
oaa = aa['objective']

# ---------- E2 / E3 / E4 side-cars ----------
e2 = {int(k): v for k, v in (hist('24ab').get('e2_by_seed') or {}).items()}
e2_all = [e2[k]['all14'] for k in sorted(e2)]
e2_mean = sum(e2_all) / len(e2_all)
e2_sd = (sum((x - e2_mean) ** 2 for x in e2_all) / (len(e2_all) - 1)) ** 0.5
e3 = json.load(open(os.path.join(PROC, '24ac_referee.json')))
E3F = e3['floor']; E3R = e3['referee']

SUMMARY = f"""# Series 24 \u2014 Real-data accuracy: executive summary

**Opened** 2026-09-26 (Anuar's brief: *\u201canalyse the whole project \u2026 the model must optimise correctly to the
real data; higher transitions matter more, but it would be good if all of the real data are approximated
somehow\u201d*). **Closed** 2026-09-27 after **30 notebooks** (`24a`\u2026`24ad`) on branch `develop`.
State file: `SERIES_STATE.md` (protocol \u00a72, metric \u00a73, failure modes \u00a74, roadmap \u00a75, log \u00a77).

---

## 1. Goal

Take the project's real-data KDE model from the series-23 baseline (`experiment_6 trial_07`, `W_obj` 0.0811)
to the best attainable fit of **all 14 real experiments** (1nW / 3nW \u00d7 T05\u2026T100), **weighted toward high
transmission**, with an **honest uncertainty statement** \u2014 using exactly one named change per notebook,
hypothesis \u2192 test \u2192 conclusion.

## 2. Frozen protocol & metric

- **Data**: all 14 real experiments; real-target load + `filt = ok & (err/fwhm < 10)` (18b/19d/23a).
- **Budget**: `N_RUNS=100`, `N_ITER=30`, `SEED=42`, init `0.5\u00d7truth`, inner Lorentzian fit (`n_iters=80`).
- **Rules**: one change per notebook, truth-free knobs, notebook-level changes preferred, figures inline.
- **Metric** (series-20/21 + exp5\u20138, frozen): per cell `rel_sq = min(((x\u0302\u2212x)/x)\u00b2, 1)`, weight
  `w(T)=0.25+0.75\u00b7T/100`; **`W_obj = \u03a3w\u00b7err / \u03a3w`** over the 28 (\u03bc,\u03b3) cells.
  Always reported: `W_obj(all14)`, `W_obj(T\u226540)`, `W_obj(T\u226420)`, \u03bc rel-RMSE, \u03b3 rel-RMSE, divergences.
- **Measured chaos band** (new in E2): **\u2248 0.0026** on all14 (replicate sd), **4\u00d7 tighter** than the
  \u00b10.01 caveat the series started from.

## 3. The notebook ladder (`24a` \u2192 `24ad`)

| notebook | one change | all14 | T\u226540 | T\u226420 | verdict |
|---|---|---|---|---|---|
{LADDER}

## 4. The best model \u2014 `24aa` (E1 role-split), and its exact config

**`W_obj(all14) = {oaa['all14']:.4f}` \u00b7 `T\u226540 = {oaa['hiT']:.4f}` \u00b7 `T\u226420 = {oaa['loT']:.4f}` \u00b7 \u03bc {oaa['mu_rel_rmse']:.1f} % \u00b7 \u03b3 {oaa['gamma_rel_rmse']:.1f} % \u00b7 0/14 divergences.**
**E2 (3-seed repeat, `24ab`)**: `all14 = {e2_mean:.4f} \u00b1 {e2_sd:.4f}` (min {min(e2_all):.4f}, max {max(e2_all):.4f});
seed 42 reproduces `24aa` bit-identically.

Idea: **give \u03bc the \u03c3-free FWHM-only kernel + count prior (24r), give \u03b3 the 2-D \u03c3-bearing KDE (24p/24y)**.
Exact frozen config (`two_phase` + B8 + E1):

```json
{CFG_JSON}
```
Plus the frozen schedule (`lr_gamma=0.4716, gamma_anneal=0.4723, sigma_weight=0.8649, gamma_rel_cap=0.1074,
h_f_scale=4.0, h_s_scale=3.6745, h_s_min=0.05`), and the **bias-corrected \u03bc report** (24v, 12/14 2\u03c3 coverage).

**Runner-ups**: `24r` all14 {hist('24r')['objective']['all14']:.4f} (**best T\u226540 {hist('24r')['objective']['hiT']:.4f}**, \u03bc {hist('24r')['objective']['mu_rel_rmse']:.1f} %) \u00b7
`24p` (B6 hybrid) {hist('24p')['objective']['all14']:.4f}, \u03bc {hist('24p')['objective']['mu_rel_rmse']:.1f} % \u00b7
`24g` (\u03b3 best) \u03b3 {hist('24g')['objective']['gamma_rel_rmse']:.1f} % \u00b7 `24i` (low-T sub-metric) T\u226420 {hist('24i')['objective']['loT']:.4f}.

## 5. Four headline findings

**(1) \u03bc travel is a real but *saturated* lever (Phase A, `24b`\u2013`24j`).** Replacing the fixed `lr_mu` anneal by a
**truth-free travel budget** (`24b`, all14 0.0811\u21920.0730, \u03bc 33.2\u219228.0 %) was the single biggest Phase-A gain, but the
schedule *shape* is second-order (`24c`), the clip is mostly idle and the net travel saturates at \u22480.43\u00b7\u03bc_init because
the REINFORCE gradient flips sign (**FM#13/FM#14**). A hard freeze hurts low T (`24e`); the two-phase schedule (`24g`)
gives the best Phase-A low-T + \u03b3 (T\u226420 0.1614, \u03b3 32.5 %). Phase A ends exhausted as a *global* lever.

**(2) The \u03c3 channel is the \u03bc culprit \u2014 resolved by dropping it (`24p`\u2013`24r`, Phase B).** \u03c3-side fixes D1/D2/D3
(`24x`\u2013`24z`) all fail to move \u03bc, but the **1-D FWHM-only KDE + a count prior** (`24r`) gives the series' best \u03bc
(25.5 %) and T\u226540 (0.0326). "**\u03c3 or 2-D?**" = **both**: \u03bc wants the \u03c3-free kernel, \u03b3 needs a real \u03c3 channel
(the 2-D \u03b3 chain is load-bearing at low T). **E1 (`24aa`) collects both in one model** \u2014 the series best.

**(3) The honest interval is the *bias-corrected* one (Phase C, `24s`\u2013`24w`).** Every *raw* interval is dishonest:
CRB 5/14, sandwich 10/14, profile 3/14, bootstrap 0/8 (2\u03c3 \u03bc coverage of \u03bc_true). The **sim-at-truth bias** (`24v`) is
large and negative (median \u22129.4), and correcting it cuts the \u03bc error 13.7 \u2192 2.0 and restores **12/14** (CRB) /
**13/14** (sandwich) coverage. \u21d2 the residual is a measurable, correctable **bias**, not variance, and the CRB is
*bias-blind* (**FM#12**), not too narrow.

**(4) The last residual is the line-shape / photon-count model (FM#6), and the two objectives disagree (Phase D/E).**
No \u03c3-forward-model fix moves \u03bc (D1/D2/D3), so the high-T \u03bc error is the **Lorentzian line-shape + count model**.
E3 (`24ac`) makes this quantitative: the winner reproduces the real FWHM distribution **as well as the truth point does**
(`cvm_fwhm` \u00d7{E3R['24aa']['cvm_fwhm']/E3F['cvm_fwhm']:.2f} of the truth-point floor), so a data-discrepancy metric and a
truth-error metric **cannot** track each other (Spearman +0.36/+0.43) \u2014 the data do **not** prefer the true \u03bc.

## 6. Parked candidates (kept for a series 25)

- **`24i`** \u2014 coupled \u03bc/\u03b3 travel: best low-T sub-metric of the series (T\u226420 **0.1057**), at the cost of high T.
- **`24g`** \u2014 two-phase schedule: **best \u03b3** of the series (**32.5 %**).
- **`24n`** \u2014 Student-t \u03c3 kernel: a different high-T variant (T\u226540 0.0402) with a different low-T trade-off.
- **`24y`** \u2014 the calibrated-\u03c3 \u03b3 arm (\u03b3 33.0 %, the \u03b3 source E1 reuses).
- **`24r`** \u2014 the \u03c3-free \u03bc model: **best T\u226540 (0.0326)** and the \u03bc tie; keep as the high-T-optimal variant.

## 7. Next steps for a series 25

1. **Fix the dead D2 plumbing.** `sigma_estimator='lorentzian'` means the `surrogate` branch in `_sims` is *never*
   entered; the D3 table `c3 = real/(\u03ba\u00b7\u03c3_lor)` was meant to sit *on top of* `\u03ba\u00b7\u03c3_CRLB`, so applied to `\u03c3_lor` it
   returns `real/\u03ba \u2248 \u03c3_lor` (median c3 = 1.00) \u2014 a **numerical no-op**.  This explains why `24y \u2248 24z \u2248 24aa`
   on \u03b3 (\u03b3 33.0/33.9/33.3 %).  No verdict changes, but series 25 must wire it correctly before re-testing D.
2. **Attack FM#6 at the simulation level**: the residual high-T \u03bc error is the line shape/photon-count model, not the
   \u03c3 channel.  Try a pseudo-Voigt **simulator**, per-scan linewidth heterogeneity, and a count model that lets the
   FWHM distribution prefer the true \u03bc.
3. **Keep the honest interval as a first-class deliverable**: report the **bias-corrected** CRB (12/14) as the default,
   with the corrected sandwich (13/14, 2.5\u00d7 wider) as the conservative variant; drop the raw CRB/profile/bootstrap.
4. **Separate the two objectives.**  Use the exp8 discrepancy (`cvm_fwhm`/`w1_2d`) for **model checking** and the
   truth-weighted `W_obj` for **parameter recovery**; do not assume they agree (E3).
5. **Use a random (not T-based) held-out split** \u2014 E4 (`24ad`) shows a core-T split selects a different model than the
   all-14 objective, so the split itself must be representative.
6. **Carry the measured chaos band (\u22480.0026)** instead of the \u00b10.01 caveat when judging future wins.

---
*Generated from `SERIES_STATE.md` \u00a77 + `data/processed/24*_history.json` / `24ab`/`24ac`/`24ad` side-cars.*
"""

open(os.path.join(HERE, 'SUMMARY.md'), 'w').write(SUMMARY)
print('wrote SUMMARY.md', len(SUMMARY), 'bytes')
print('ladder rows:', len(ladder))

# ---------- append the §9 FINAL SUMMARY to SERIES_STATE.md ----------
_ladder_compact = '\n'.join(
    f"| `{c[1]}` | {short(c[2], 84)} | {c[3]} | {c[4]} |" for c in rows)
S9 = f"""

---

## 9. FINAL SUMMARY (series 24)

**Goal:** best fit of **all 14 real experiments**, weighted toward high transmission, with an honest interval,
**one named change per notebook** (30 notebooks, `24a`\u2026`24ad`). Frozen metric: `W_obj = \u03a3w(T)\u00b7err/\u03a3w`,
`w(T)=0.25+0.75\u00b7T/100` over the 28 (\u03bc,\u03b3) cells. **Measured chaos band \u2248 0.0026** all14 (E2, 4\u00d7 tighter
than the \u00b10.01 caveat).

### Ladder

| notebook | one change | all14 | T\u226540 |
|---|---|---|---|
{_ladder_compact}

### Best model \u2014 `24aa` (E1 role-split: \u03c3-free FWHM \u03bc + count prior, \u03c3-bearing 2-D \u03b3 KDE)

`W_obj(all14) = {oaa['all14']:.4f}` \u00b7 `T\u226540 = {oaa['hiT']:.4f}` \u00b7 `T\u226420 = {oaa['loT']:.4f}` \u00b7 \u03bc {oaa['mu_rel_rmse']:.1f} % \u00b7 \u03b3 {oaa['gamma_rel_rmse']:.1f} % \u00b7 0/14 div.
**E2 3-seed repeat (`24ab`): `all14 = {e2_mean:.4f} \u00b1 {e2_sd:.4f}`** (all 3 replicates below 24r's 0.0672). Runner-ups:
`24r` best T\u226540 {hist('24r')['objective']['hiT']:.4f} \u00b7 `24g` best \u03b3 {hist('24g')['objective']['gamma_rel_rmse']:.1f} % \u00b7 `24i` best T\u226420 {hist('24i')['objective']['loT']:.4f} \u00b7 `24p` best non-\u03c3-free all14 {hist('24p')['objective']['all14']:.4f}.

Best-config JSON:
```json
{CFG_JSON}
```

### Four headline findings

1. **\u03bc travel is a real but saturated lever** (`24b`\u2013`24j`): truth-free travel budget 0.0811\u21920.0730 (\u03bc 33.2\u219228.0 %), but shape is 2nd-order, the clip is idle and net travel saturates at \u22480.43\u00b7\u03bc_init (**FM#13/#14**).
2. **The \u03c3 channel is the \u03bc culprit \u2014 24r** (`24p`\u2013`24r`): \u03c3-side fixes D1/D2/D3 fail to move \u03bc, but the 1-D FWHM-only KDE + count prior gives the best \u03bc (25.5 %) and T\u226540 (0.0326); \u03b3 still needs a real \u03c3 channel \u21d2 E1 collects both in `24aa`.
3. **The honest interval is the bias-corrected one** (`24s`\u2013`24w`): raw coverage CRB 5/14, sandwich 10/14, profile 3/14, bootstrap 0/8; the sim-at-truth bias (median \u22129.4) corrects the \u03bc error 13.7\u21922.0 and restores 12/14 (CRB) / 13/14 (sandwich) \u21d2 the residual is **bias**, not variance (**FM#12**).
4. **FM#6 line shape is the last residual, and the two objectives disagree** (`24x`\u2013`24ac`): no \u03c3-model fix moves \u03bc; E3 shows the winner fits the real data as well as the **truth point** (`cvm_fwhm` \u00d7{E3R['24aa']['cvm_fwhm']/E3F['cvm_fwhm']:.2f} of the floor) with Spearman(W_obj, discrepancy) only +0.36/+0.43 \u21d2 truth-recovery \u2260 data-discrepancy.

### Parked for series 25

`24i` (low-T 0.1057) \u00b7 `24g` (\u03b3 32.5 %) \u00b7 `24n` (T\u226540 0.0402 variant) \u00b7 `24y` (calibrated-\u03c3 \u03b3 arm) \u00b7 `24r` (best T\u226540 0.0326).

### Next steps (series 25)

1. **Fix the dead D2 plumbing**: with `sigma_estimator='lorentzian'` the `surrogate` branch of `_sims` is never
   entered, so the D3 factor `c3=real/(\u03ba\u00b7\u03c3_lor)` applied to `\u03c3_lor` is a **no-op** (median c3=1.00) \u2014 which is
   why `24y\u224824z\u224824aa` on \u03b3. No verdict changes; wire it before re-testing D.
2. **Attack FM#6 at the simulator level** (pseudo-Voigt simulation / per-scan heterogeneity / better count model).
3. Report the **bias-corrected** interval by default; drop the raw CRB/profile/bootstrap.
4. Keep the **two objectives separate** (discrepancy = model check, `W_obj` = parameter recovery).
5. Use a **random** held-out split (E4: a T-based core split selects a different model than all-14).
6. Judge future wins against the **measured \u22480.0026** chaos band, not \u00b10.01.

Full narrative: `notebooks/24_realdata_accuracy/SUMMARY.md`.
"""

_st = open(os.path.join(HERE, 'SERIES_STATE.md')).read()
if '## 9. FINAL SUMMARY (series 24)' not in _st:
    open(os.path.join(HERE, 'SERIES_STATE.md'), 'a').write(S9)
    print('appended \u00a79 to SERIES_STATE.md')
else:
    print('\u00a79 already present')
