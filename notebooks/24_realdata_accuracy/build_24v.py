#!/usr/bin/env python3
"""Build 24v from 24u — Phase C4: bias-corrected CRB (sim-at-truth bias).

Adds DO_BIAS=True -- generate a pseudo-data cloud at (mu_true,gamma_true) (the 22b/22g
sim-at-truth), run the SAME frozen optimiser on it, take bias = mu_hat_sim - mu_true, and
apply mu_corr = mu_hat_real - bias; then re-check coverage vs the dataset CRB and the 24s
bootstrap.  DO_PROFILE off (C3 read from 24u); DO_FISHER stays on (needs the CRB).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24v.ipynb')
if os.path.exists(p):
    sys.exit('24v.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24v', '24u',
                'bias-corrected CRB via the sim-at-truth bias (Phase C4)',
                'C4: DO_BIAS=True -- generate a sim-at-truth pseudo-data cloud (22b/22g), run the SAME '
                'frozen optimiser on it to measure the estimator bias, subtract it from the real point '
                'estimate, then re-check mu coverage against the dataset CRB and the 24s bootstrap. '
                'DO_PROFILE off (C3 read from 24u).', ],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: DO_PROFILE off, DO_BIAS on ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_PROFILE' in s and 'CFG = dict(' in s:
        s = s.replace("DO_PROFILE = True      # C3 (24u): profile-likelihood 1-sigma interval for mu",
                      "DO_PROFILE = False     # C3 off (profile read from data/processed/24u_profile.json)\n"
                      "DO_BIAS = True         # C4 (24v): bias-corrected CRB via the sim-at-truth bias")
        set_txt(c, s); done = True; break
if not done: sys.exit('config cell not found')

# ---- insert the bias cell AFTER the profile cell ----
prof_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'PROFILE-LIKELIHOOD INTERVAL FOR mu' in txt(c):
        prof_idx = i; break
if prof_idx is None: sys.exit('profile cell not found')

bias_src = '''# ============================================================
# SERIES 24 / C4 (24v) \u2014 BIAS-CORRECTED CRB (sim-at-truth bias, 22b/22g)
#   Measure the estimator's bias on SELF-CONSISTENT data: draw a pseudo-data cloud at
#   (mu_true, gamma_true), run the SAME frozen optimiser on it, bias = mu_hat_sim - mu_true.
#   Then mu_corr = mu_hat_real - bias and re-check coverage vs the dataset CRB and the 24s
#   bootstrap.  NOTE: this is the only cell that reads mu_true/gamma_true -- it is the METHOD's
#   ingredient (the 22b/22g sim-at-truth correction), not a knob tuned against the truth.
# ============================================================
BIAS = {}
if DO_BIAS:
    with _PPE(max_workers=N_WORKERS, mp_context=_mp.get_context('fork'), initializer=_init_worker) as _bp:
        for r in RESULTS:
            exp = next(e for e in EXPERIMENTS if e['name'] == r['exp'])
            Nb = int(r.get('n_target', 100))
            _fb, _sb, _nb, _dgb, _dsb = _sims(_bp, r['mu_true'], r['sigma_prop'], r['lam'],
                                               r['gamma_true'], Nb, SEED + 4242)
            _c2 = dict(CFG); _c2['_target_override'] = (_fb, _sb)
            rb = run_experiment(exp, _c2, _bp)
            BIAS[r['exp']] = dict(mu_hat_sim=rb['mu_final'], g_hat_sim=rb['gamma_final'],
                                  bias_mu=rb['mu_final'] - r['mu_true'],
                                  bias_gamma=rb['gamma_final'] - r['gamma_true'],
                                  mu_hat=r['mu_final'], gamma_hat=r['gamma_final'],
                                  mu_corr=r['mu_final'] - (rb['mu_final'] - r['mu_true']),
                                  g_corr=r['gamma_final'] - (rb['gamma_final'] - r['gamma_true']),
                                  mu_true=r['mu_true'], gamma_true=r['gamma_true'], N=Nb)
            print(f"  {r['exp']:13s} mu_true {r['mu_true']:7.2f} | sim_hat {rb['mu_final']:7.2f} "
                  f"| bias {rb['mu_final']-r['mu_true']:+7.2f} | real_hat {r['mu_final']:7.2f} "
                  f"-> corr {BIAS[r['exp']]['mu_corr']:7.2f}", flush=True)
    _bf = np.array([v['bias_mu'] for v in BIAS.values()]); _bg = np.array([v['bias_gamma'] for v in BIAS.values()])
    print(f"sim-at-truth bias: median mu {np.median(_bf):+.3f} (std {_bf.std():.3f}) | "
          f"median gamma {np.median(_bg):+.3f} (std {_bg.std():.3f})")
    _unc = np.array([abs(v['mu_hat'] - v['mu_true']) for v in BIAS.values()])
    _cor = np.array([abs(v['mu_corr'] - v['mu_true']) for v in BIAS.values()])
    print(f"|mu_true error|: raw mean {_unc.mean():.3f} -> corr mean {_cor.mean():.3f} "
          f"({'IMPROVED' if _cor.mean() < _unc.mean() else 'NOT improved'})")
    # coverage versus the dataset CRB (2 sigma): raw vs bias-corrected
    _crb = {r['exp']: float((r.get('fisher_frozen') or {}).get('sig_mu_data') or float('nan')) for r in RESULTS}
    _raw2 = sum(1 for k, v in BIAS.items() if abs(v['mu_hat'] - v['mu_true']) <= 2.0 * _crb.get(k, float('nan')))
    _cor2 = sum(1 for k, v in BIAS.items() if abs(v['mu_corr'] - v['mu_true']) <= 2.0 * _crb.get(k, float('nan')))
    print(f'2-sigma CRB coverage of mu: raw {_raw2}/{len(BIAS)} -> bias-corrected {_cor2}/{len(BIAS)}')
    try:
        _bs = json.load(open(os.path.join(REPO_ROOT, 'data', 'processed', '24s_bootstrap.json')))['boot']
        _nb = sum(1 for k in BIAS if k in _bs)
        _b2r = sum(1 for k, v in BIAS.items() if k in _bs and abs(v['mu_hat'] - v['mu_true']) <= 2.0 * _bs[k]['mu_sd'])
        _b2c = sum(1 for k, v in BIAS.items() if k in _bs and abs(v['mu_corr'] - v['mu_true']) <= 2.0 * _bs[k]['mu_sd'])
        print(f'2-sigma 24s-bootstrap coverage (subset {_nb}): raw {_b2r}/{_nb} -> corrected {_b2c}/{_nb}')
    except Exception as _e:
        print('bootstrap compare skipped:', _e)
    if not SMOKE:
        _bo = os.path.join(REPO_ROOT, 'data', 'processed', '24v_bias.json')
        json.dump(dict(notebook=NB_ID, method='sim_at_truth_bias', bias=BIAS,
                       median_bias_mu=float(np.median(_bf)), median_bias_gamma=float(np.median(_bg)),
                       raw_abs_err_mean=float(_unc.mean()), corr_abs_err_mean=float(_cor.mean())),
                  open(_bo, 'w'))
        print('saved', _bo)
else:
    print('DO_BIAS=False - bias block skipped')
'''
cells.insert(prof_idx + 1, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                                source=bias_src.splitlines(keepends=True)))

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24v (Phase C4)

- **Change vs previous notebook (`24u`):** `DO_BIAS=True` (the profile is off; C3 is read from
  `data/processed/24u_profile.json`).  For each experiment we draw a **pseudo-data cloud at
  `(mu_true, gamma_true)`** (the 22b/22g sim-at-truth), run the **same frozen optimiser** on it, and take
  `bias = mu_hat_sim \u2212 mu_true`.  The **bias-corrected point estimate** is `mu_corr = mu_hat_real \u2212 bias`;
  we then compare `|mu_corr \u2212 mu_true|` and its coverage against the raw estimate, the dataset CRB and the
  24s bootstrap.  (This cell is the *only* place `mu_true/gamma_true` enters \u2014 it is the method's own
  ingredient, exactly as in 22b/22g, not a knob tuned against the truth.)
- **Hypothesis (C4 / FM#8):** the residual high-T \u03bc error is largely a **bias** of the estimator on
  self-consistent data (the same \u03c3(\\u03bc) conditional mis-calibration that 24j measured), not a variance.
  A sim-at-truth bias subtraction should **shift the high-T \u03bc estimates toward truth** and restore the
  CRB's coverage there, without touching the (already good) 3nW high-T \u03b3.
- **Falsifiable prediction:** median `bias_mu > 0` at high T (the optimiser undershoots: \u03bc_hat < \u03bc_true,
  so bias < 0 and the correction *raises* \u03bc), the mean `|mu_corr \u2212 mu_true|` **improves** on the raw
  mean, and the 2\u03c3 CRB coverage of the *corrected* estimate exceeds the raw coverage.  **Falsified if**
  the sim-at-truth binary recovers truth (median bias \u2248 0 \u2192 the estimator is unbiased on self-consistent
  data and the real-data error is a **model/data gap**, not an estimator bias) or the correction makes
  `|mu_corr \u2212 mu_true|` worse.
- **If falsified:** the residual is a genuine forward-model gap (sim \u2260 real even at truth) \u2192 escalate to
  Phase D (matched forward model / differentiable surrogate).
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
for c in cells:
    for ln in txt(c).splitlines():
        if 'DO_BIAS' in ln or 'DO_PROFILE' in ln:
            print('   ', ln.strip())
