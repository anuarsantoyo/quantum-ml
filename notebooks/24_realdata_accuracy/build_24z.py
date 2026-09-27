#!/usr/bin/env python3
"""Build 24z from 24y — Phase D3: per-(T,power) sigma-bias correction table from 22g.

ONE change: sigma_calib='table_22g' -- multiply the simulated sigma_fit by the per-cell factor
c(exp) = median(REAL sigma_fit) / median(22g pseudo_voigt sigma_fit at truth)  (the cheapest D
form).  Applied on top of the D2 surrogate, so sigma = kappa(FWHM,n,gamma) * c(exp) * sigma_CRLB.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24z.ipynb')
if os.path.exists(p):
    sys.exit('24z.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24z', '24y',
                'per-(T,power) sigma-bias correction table from 22g (Phase D3)',
                "D3: sigma_calib='table_22g' -- multiply the simulated sigma_fit by the per-cell factor "
                'c(exp) = median(real sigma_fit) / median(22g pseudo_voigt sigma_fit at truth); the '
                'cheapest D form, applied on top of the D2 surrogate. DO_FISHER off.', ],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: add the table switch ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_FISHER' in s and 'CFG = dict(' in s:
        s = s.replace("    sigma_surrogate=True,                         # D2: sigma_fit = kappa(FWHM,n,gamma) * sigma_CRLB",
                      "    sigma_surrogate=True,                         # D2: sigma_fit = kappa(FWHM,n,gamma) * sigma_CRLB\n"
                      "    sigma_calib='table_22g',                      # D3: * per-cell 22g sigma-bias factor c(exp)")
        set_txt(c, s); done = True; break
if not done: sys.exit('config cell not found')

# ---- run_experiment: apply the per-cell sigma table after the two _sims calls ----
done = False
for c in cells:
    s = txt(c)
    if 'def run_experiment(' in s:
        old1 = ("        _ft0 = _ft0 + _jit * torch.tensor(np.random.default_rng(SEED + 991).standard_normal(n_runs), dtype=torch.float32)\n")
        new1 = old1 + ("    if str(cfg.get('sigma_calib', 'none')) != 'none':   # D3 (24z): per-(T,power) sigma-bias table\n"
                       "        _sc_ = float(SIGMA_CALIB.get(exp['name'], 1.0))\n"
                       "        _si0 = _si0 * _sc_; _ds0 = _ds0 * _sc_\n")
        assert old1 in s, 'init _jit anchor not found'
        s = s.replace(old1, new1, 1)
        old2 = ("            ft = ft + _jit * torch.tensor(np.random.default_rng(SEED + 991 + step).standard_normal(n_runs), dtype=torch.float32)\n")
        new2 = old2 + ("        if str(cfg.get('sigma_calib', 'none')) != 'none':   # D3 (24z)\n"
                       "            _sc_ = float(SIGMA_CALIB.get(exp['name'], 1.0))\n"
                       "            si_t = si_t * _sc_; ds_t = ds_t * _sc_\n")
        assert old2 in s, 'step _jit anchor not found'
        s = s.replace(old2, new2, 1)
        set_txt(c, s); done = True; break
if not done: sys.exit('run_experiment cell not found')

# ---- insert the calibration cell BEFORE the main run cell ----
run_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'RUN: all 14 real experiments' in txt(c):
        run_idx = i; break
if run_idx is None: sys.exit('run cell not found')

cal_src = '''# ============================================================
# SERIES 24 / D3 (24z) \u2014 per-(T,power) sigma-bias correction table from 22g
#   c(exp) = median(REAL sigma_fit) / median(22g pseudo_voigt sigma_fit at truth): the ratio that maps
#   the estimator's simulated sigma_fit onto the RECORDED real sigma_fit for that cell.  Truth-free
#   (uses the real target cloud + the 22g estimator cloud; no mu_true/gamma_true knob).  The simulated
#   sigma is multiplied by c(exp) (and dsigma/dgamma with it).
# ============================================================
_S22c = np.load(os.path.join(REPO_ROOT, 'data', 'processed', '22g_clouds.npz'), allow_pickle=True)
SIGMA_CALIB = {}
for e in EXPERIMENTS:
    nm = e['name']
    sr = float(np.median(_S22c[f'{nm}|real|s']))
    sp = float(np.median(_S22c[f'{nm}|pseudo_voigt|s']))
    SIGMA_CALIB[nm] = sr / max(sp, 1e-9)
print(f"{'exp':14s}{'real sig_med':>13s}{'pv sig_med':>11s}{'c(exp)':>9s}")
for nm, cc in SIGMA_CALIB.items():
    print(f'{nm:14s}{float(np.median(_S22c[f"{nm}|real|s"])):13.3f}'
          f'{float(np.median(_S22c[f"{nm}|pseudo_voigt|s"])):11.3f}{cc:9.3f}')
print(f'median c(exp) = {float(np.median(list(SIGMA_CALIB.values()))):.3f}')
'''
cells.insert(run_idx, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                           source=cal_src.splitlines(keepends=True)))

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24z (Phase D3)

- **Change vs previous notebook (`24y`):** `sigma_calib='table_22g'` \u2014 multiply the simulated \u03c3_fit (already
  surrogate-corrected) by the **per-(T,power) factor** `c(exp) = median(real \u03c3_fit) / median(22g pseudo_voigt
  \u03c3_fit at truth)`, i.e. the cheapest D form: move the simulated \u03c3 onto the *recorded* real \u03c3, cell by cell.
  `dsigma/dgamma` is scaled with it.  Truth-free (real target cloud + the 22g estimator cloud).
- **Hypothesis (FM#8 / D3):** the *residual* of the \u03c3 channel after D1/D2 is a pure per-cell **scale** bias; a
  per-(T,power) correction removes it, so the \u03c3 channel becomes honest and \u03bc \u2192 truth.
- **Falsifiable prediction:** the \u03bc table moves toward 1 and \u03bc rel-RMSE **beats 24y** (and 24r's 25.5 %).
  **Falsified if** \u03bc is unchanged \u2014 which would show the \u03bc mode is not set by the \u03c3 *scale* at all (a repeat
  of 24l's global-scale null, now per cell), and the residual is the line-shape/photon model (FM#6).
- **If falsified:** the \u03c3 channel is genuinely un-informative for \u03bc at every level of calibration \u21d2 E1 keeps
  the 24r \u03bc treatment and uses \u03c3 only for \u03b3.
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
