#!/usr/bin/env python3
"""Build 24x from 24w — Phase D1: estimator-matched (pseudo-Voigt) forward model.

ONE mechanism: the simulator's FWHM *and* sigma_fit channels both come from the pseudo-Voigt
estimator (22g's best-matched) instead of the analytic Lorentzian CRLB; the sigma channel is
re-enabled (2-D KDE re-derived) and the B8 count-prior mu stabiliser is kept. Single pv fit per
run (cheaper than 24o's two-fit sigma-only match).  DO_FISHER off (C-phase frozen).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24x.ipynb')
if os.path.exists(p):
    sys.exit('24x.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24x', '24w',
                'estimator-matched pseudo-Voigt forward model (Phase D1)',
                'D1: the simulator\'s FWHM AND sigma_fit channels both come from the pseudo-Voigt '
                'estimator (22g best-matched; one pv fit per run) instead of the analytic Lorentzian '
                'CRLB, and the sigma channel is re-enabled (2-D KDE re-derived) with the B8 count-prior '
                'mu stabiliser kept. DO_FISHER off (C-phase frozen).', ],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: pv-full estimator, sigma channel ON, Fisher off, add the pv-full mode note ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_FISHER' in s and 'CFG = dict(' in s:
        s = s.replace("DO_FISHER  = True      # C2 (24t): Fisher + sandwich over all 14 cells",
                      "DO_FISHER  = False     # C2 (24t): Fisher frozen in the C-phase; D1 is a point-estimate arm")
        s = s.replace("DO_REFEREE = True      # C5 (24w): one coverage table over CRB/sandwich/boot/profile/bias-corrected",
                      "DO_REFEREE = False     # C5 off (frozen in data/processed/24w_referee.json)")
        s = s.replace("    sigma_channel='off', cnt_prior_w=0.25,       # B8: 1-D FWHM KDE + count channel as a prior",
                      "    sigma_channel='on', cnt_prior_w=0.25,        # D1: 2-D KDE re-derived; B8 count prior kept\n"
                      "    sigma_estimator='pseudo_voigt_full',          # D1: FWHM + sigma_fit both from the pv estimator")
        s = s.replace("    sigma_estimator='lorentzian',                # B5 OFF (branch measured)",
                      "    # (B5 sigma_estimator line superseded by sigma_estimator='pseudo_voigt_full' above)")
        set_txt(c, s); done = True; break
if not done: sys.exit('config cell not found')

# ---- cell 8: DO_REFEREE is off now, so restore the plain cached/recompute logic ----
done = False
for c in cells:
    s = txt(c)
    if 'RUN: all 14 real experiments' in s and 'REF_SRC' in s:
        old = ("REF_SRC = os.path.join(REPO_ROOT, 'data', 'processed', '24v_history.json')\n"
               "CACHED = (not SMOKE) and os.path.exists(OUT_JSON) and os.environ.get('NB_RECOMPUTE') != '1'\n"
               "RESULTS = []\n"
               "if DO_REFEREE and not SMOKE:\n"
               "    # C5 (24w): inherit the 24v point estimate + Fisher (the C-phase is a pure uncertainty side-car;\n"
               "    # no re-optimisation, so the referee is cheap and consistent with the 24s/24u/24v JSONs).\n"
               "    RESULTS = json.load(open(REF_SRC))['results']\n"
               "    CACHED = True\n"
               "    print(f'C5 referee: inherited {len(RESULTS)} experiments from {REF_SRC} (no re-optimisation)')\n"
               "elif CACHED:\n")
        new = ("CACHED = (not SMOKE) and os.path.exists(OUT_JSON) and os.environ.get('NB_RECOMPUTE') != '1'\n"
               "RESULTS = []\n"
               "if CACHED:\n")
        assert old in s, 'cell-8 anchor not found'
        set_txt(c, s.replace(old, new)); done = True; break
if not done: sys.exit('run cell anchor not found')

# ---- cell 16: restore the plain save condition ----
done = False
for c in cells:
    s = txt(c)
    if 'save companion run data' in s:
        s = s.replace("if SMOKE or (CACHED and not DO_REFEREE):", "if SMOKE or CACHED:")
        set_txt(c, s); done = True; break
if not done: sys.exit('save cell not found')

# ---- cell 6: add the pv-full mode to _sims ----
done = False
for c in cells:
    s = txt(c)
    if 'def _sims(' in s:
        old = ("    tasks.append((gamma, u.numpy(), b.numpy())); ns.append(n)\n"
               "    res = _parallel_map(pool, tasks)\n")
        new = ("    tasks.append((gamma, u.numpy(), b.numpy())); ns.append(n)\n"
               "    # --- D1 (24x): FULL estimator match.  'pseudo_voigt_full' runs the pv fit ONCE per run and\n"
               "    #     takes BOTH FWHM and sigma_fit (and their gamma derivatives) from it (one fit, not two).\n"
               "    if str(sigma_est) == 'pseudo_voigt_full':\n"
               "        res = _parallel_map(pool, tasks, fn=_run_one_pv)\n"
               "    else:\n"
               "        res = _parallel_map(pool, tasks)\n")
        assert old in s, 'sims anchor not found'
        set_txt(c, s.replace(old, new)); done = True; break
if not done: sys.exit('sims cell not found')

# ---- cell 7: generalise the count-prior trigger so it stays active with the sigma channel ON ----
done = False
for c in cells:
    s = txt(c)
    if 'def run_experiment(' in s:
        old = ("        if str(cfg.get('sigma_channel', 'on')) == 'off':\n"
               "            # --- B8 (24r): the COUNT CHANNEL re-added as a PRIOR ----------------------------\n")
        new = ("        if float(cfg.get('cnt_prior_w', 0.0)) > 0.0:\n"
               "            # --- B8 (24r): the COUNT CHANNEL re-added as a PRIOR ----------------------------\n"
               "            #   (D1: trigger generalised from 'sigma channel off' to cnt_prior_w>0 so the\n"
               "            #    stabiliser also acts when the estimator-matched sigma channel is ON.)\n")
        assert old in s, 'count-prior anchor not found'
        set_txt(c, s.replace(old, new)); done = True; break
if not done: sys.exit('run_experiment cell not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24x (Phase D1)

- **Change vs previous notebook (`24w`):** the forward model is now **estimator-matched** \u2014 the simulator's
  FWHM **and** \u03c3_fit channels both come from the **pseudo-Voigt** estimator (22g's best-matched: 13.8 % FWHM-IQR
  and 28.1 % \u03c3_fit error vs real, against 45.2 % / 61.5 % for the unbinned Lorentzian MLE), produced by a
  **single** pv fit per run.  The \u03c3 channel is re-enabled (the 2-D KDE is re-derived) **and** the B8
  count-prior stabiliser is kept (new: it acts whenever `cnt_prior_w>0`, not only when \u03c3 is off).
  `DO_FISHER=False` (the C-phase uncertainty tools are frozen in `data/processed/24w_referee.json`).
- **Hypothesis (FM#8 / D1):** the whole \u03bc residual is the \u03c3(\u03bc)-conditional mis-calibration of the *analytic
  Lorentzian CRLB*; matching the estimator at source (pv) removes it, so \u03bc should land near truth, while the
  \u03b3 channel \u2014 which still consumes \u03c3 \u2014 may pay (24o's finding).
- **Falsifiable prediction:** \u03bc rel-RMSE improves on 24r's 25.5 % and the per-cell \u03bc/true table moves toward 1
  (especially high T, where the CRLB mismatch grows); **falsified if** \u03bc is unchanged or worse \u2014 which would
  mean the estimator choice does not set the \u03bc mode and the residual is a genuine photon/model gap (line
  shape, FM#6), not an estimator artefact.
- **If falsified:** the D1 lever is closed; escalate to a *calibrated* \u03c3 map (D2 surrogate / D3 per-cell table)
  and keep the 24r \u03bc treatment.
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
for c in cells:
    for ln in txt(c).splitlines():
        if any(k in ln for k in ('DO_FISHER', 'DO_REFEREE', 'sigma_channel', 'sigma_estimator', 'pseudo_voigt_full')):
            print('   ', ln.strip())
