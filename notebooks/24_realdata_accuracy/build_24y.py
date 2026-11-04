#!/usr/bin/env python3
"""Build 24y from 24x — Phase D2: differentiable surrogate of the fit-error map.

ONE mechanism: the per-run analytic Lorentzian CRLB sigma_fit is replaced by a smooth parametric
surrogate sigma = kappa(FWHM, n, gamma) * sigma_CRLB, where kappa is fitted to the estimator's
measured behaviour (the 22g pv/lorentzian sigma clouds at truth).  Keeps the per-run variability
(so the sigma channel stays informative) while fixing its scale/shape at source.
FWHM channel reverts to the cheap Lorentzian estimator (the pv fit was only needed while sigma
came from the estimator; 22g showed the Lorentzian FWHM centre is in fact the best match).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24y.ipynb')
if os.path.exists(p):
    sys.exit('24y.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24y', '24x',
                'differentiable surrogate of the fit-error map (Phase D2)',
                'D2: replace the per-run analytic CRLB sigma_fit with a smooth parametric surrogate '
                'sigma = kappa(FWHM, n, gamma) * sigma_CRLB, where kappa(exp) = median(real sigma_fit) / '
                'median(analytic CRLB sigma_fit) is measured on the 22g MC-at-truth clouds and then '
                'smoothed into a global power law; the FWHM channel uses the '
                'cheap Lorentzian estimator (the pv fit was only needed while sigma came from the estimator). '
                'sigma channel stays ON; DO_FISHER off.', ],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: switch the sigma estimator to the surrogate (FWHM back to the cheap Lorentzian) ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_FISHER' in s and 'CFG = dict(' in s:
        s = s.replace("    sigma_estimator='pseudo_voigt_full',          # D1: FWHM + sigma_fit both from the pv estimator",
                      "    sigma_estimator='lorentzian',                 # D2: cheap FWHM fit (sigma now comes from the surrogate)\n"
                      "    sigma_surrogate=True,                         # D2: sigma_fit = kappa(FWHM,n,gamma) * sigma_CRLB")
        set_txt(c, s); done = True; break
if not done: sys.exit('config cell not found')

# ---- cell 6: define the surrogate kappa + the 'surrogate' branch in _sims ----
done = False
for c in cells:
    s = txt(c)
    if 'def _sims(' in s:
        helper = (
            "# --- D2 (24y): differentiable surrogate of the fit-error map -------------------------\n"
            "#   kappa(FWHM, n, gamma) = exp(p0) * FWHM^p1 * n^p2 * gamma^p3, fitted to the estimator's\n"
            "#   measured behaviour (22g pv/lorentzian sigma clouds at truth).  It is a smooth correction\n"
            "#   of the analytic CRLB (truth-free: an estimator property, no mu_true).\n"
            "SURR_P = None            # [p0,p1,p2,p3] set by the calibration cell\n"
            "def _surrogate_kappa(fwhm, n_ph, gamma, p):\n"
            "    if p is None: return torch.ones_like(fwhm)\n"
            "    kap = math.exp(p[0]) * torch.pow(fwhm.clamp_min(1e-6), p[1])\n"
            "    kap = kap * (max(float(n_ph), 1.0) ** p[2]) * (max(float(gamma), 1e-6) ** p[3])\n"
            "    return kap\n\n")
        old = ("def _sims(pool, mu, sigma_prop, lam, gamma, n_runs, seed, sigma_est='lorentzian'):\n")
        s = s.replace(old, helper + old)
        # the surrogate branch inside _sims (after the existing pv branch)
        old2 = ("    if str(sigma_est) == 'pseudo_voigt':\n"
                "        # B5 (24o): take sigma_fit and dsigma/dgamma from the pseudo-Voigt fit on the SAME\n"
                "        # photons (estimator-matched); FWHM and dFWHM/dgamma stay with the Lorentzian fit.\n"
                "        rpv = _parallel_map(pool, tasks, fn=_run_one_pv)\n"
                "        st = torch.tensor([r[1] for r in rpv], dtype=torch.float32)\n"
                "        dst = torch.tensor([r[3] for r in rpv], dtype=torch.float32)\n")
        new2 = old2 + ("    if str(sigma_est) == 'surrogate':\n"
                       "        # D2 (24y): sigma_fit = kappa(FWHM,n,gamma) * sigma_CRLB (per-run variability kept).\n"
                       "        _nbar = float(np.mean(ns)) if len(ns) else 1.0\n"
                       "        _kap = _surrogate_kappa(ft, _nbar, gamma, SURR_P)\n"
                       "        dst = _kap * dst + st * _kap * (SURR_P[1] / ft.clamp_min(1e-6) * dgt\n"
                       "                                        + SURR_P[3] / max(float(gamma), 1e-6))\n"
                       "        st = _kap * st\n")
        assert old2 in s, 'pv branch anchor not found'
        s = s.replace(old2, new2)
        set_txt(c, s); done = True; break
if not done: sys.exit('sims cell not found')

# ---- insert the calibration cell BEFORE the main run cell ----
run_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'RUN: all 14 real experiments' in txt(c):
        run_idx = i; break
if run_idx is None: sys.exit('run cell not found')

cal_src = '''# ============================================================
# SERIES 24 / D2 (24y) \u2014 CALIBRATE the differentiable fit-error surrogate kappa(FWHM, n, gamma)
#   Reference behaviour = the 22g MC-at-truth clouds (1000 fits/experiment) + the RECORDED real
#   sigma_fit.  kappa(exp) = median(real sigma_fit) / median(analytic CRLB sigma_fit at truth) is the
#   per-cell FM#8 correction; it is then smoothed into a global power law over (FWHM, n, gamma) so the
#   optimiser sees a smooth, differentiable map instead of a table.  Truth-free: kappa is a property of
#   the ESTIMATOR+DATA, not of the metric (no mu_true is used; gamma_true enters only as a catalogue
#   variable of the estimator regime, exactly as in 22b/22g).  No new simulation is needed.
# ============================================================
_S22 = np.load(os.path.join(REPO_ROOT, 'data', 'processed', '22g_clouds.npz'), allow_pickle=True)
_K = []          # (name, FWHM_lor, n, gamma_true, kappa)
for e in EXPERIMENTS:
    nm = e['name']
    fl = float(np.median(_S22[f'{nm}|lorentzian|f']));  sl = float(np.median(_S22[f'{nm}|lorentzian|s']))
    sp = float(np.median(_S22[f'{nm}|real|s']))
    _K.append((nm, fl, float(e['n_target']), float(e['gamma_true']), sp / max(sl, 1e-9)))
_kap = np.array([r[4] for r in _K])
_A = np.array([[1.0, math.log(r[1]), math.log(r[2]), math.log(r[3])] for r in _K])
_b = np.log(_kap)
SURR_P, *_ = np.linalg.lstsq(_A, _b, rcond=None)
_pred = np.exp(_A @ SURR_P)
_r2 = 1.0 - float(np.sum((_kap - _pred) ** 2) / np.sum((_kap - _kap.mean()) ** 2))
print('D2 surrogate kappa(FWHM,n,gamma) = exp(%.3f) * FWHM^%.3f * n^%.3f * gamma^%.3f  (R^2 %.3f)' %
      (SURR_P[0], SURR_P[1], SURR_P[2], SURR_P[3], _r2))
print(f"{'exp':14s}{'FWHM_lor':>10s}{'n':>7s}{'gamma':>7s}{'kappa(meas)':>12s}{'kappa(fit)':>11s}")
for (nm, fl, n, gt, k), kp in zip(_K, _pred):
    print(f'{nm:14s}{fl:10.2f}{int(n):7d}{gt:7.2f}{k:12.2f}{kp:11.2f}')
print('=> the surrogate replaces the analytic CRLB with a smooth, differentiable version of the same map.')
'''
cells.insert(run_idx, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                           source=cal_src.splitlines(keepends=True)))

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24y (Phase D2)

- **Change vs previous notebook (`24x`):** the per-run **analytic Lorentzian CRLB** \u03c3_fit is replaced by a
  **smooth parametric surrogate** `\u03c3 = \u03ba(FWHM, n, \u03b3)\u00b7\u03c3_CRLB`, with `\u03ba = exp(p0)\u00b7FWHM^p1\u00b7n^p2\u00b7\u03b3^p3` fitted to the
  estimator's **measured** behaviour (the 22g MC-at-truth pv/lorentzian \u03c3 clouds).  This is the tractable
  (parametric, not full differentiable-simulator) form of Option C; it **keeps the per-run variability** so the
  \u03c3 channel stays informative.  The FWHM channel reverts to the cheap Lorentzian estimator (the pv fit was
  only needed while \u03c3 came from the estimator; 22g found the Lorentzian FWHM *centre* is the best match).
  The \u03c3 channel stays ON, the B8 count prior stays on.
- **Hypothesis (FM#8 / D2):** the \u03c3(\u03bc) conditional is set by the *scale* of the analytic CRLB, and the
  calibrated surrogate fixes exactly that scale \u2014 so \u03bc should land near truth.
- **Falsifiable prediction:** \u03bc rel-RMSE improves on 24x and on 24r's 25.5 %, and the per-cell \u03bc/true table
  moves toward 1 at high T.  **Falsified if** \u03bc is unchanged/worse \u2014 which would mean the residual is not the
  \u03c3 scale but the model/photon gap (line shape, FM#6).
- **Degradation clause:** if the power-law fit is poor (low R\u00b2) the surrogate is reported as a *diagnostic*
  only and the verdict says so.
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
