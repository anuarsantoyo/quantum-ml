#!/usr/bin/env python3
"""Build 24aa from 24z — Phase E1: combination (24r mu treatment + 24p sigma-bearing gamma chain).

ONE change: mu_gamma_split=True -- the MU reward is the 1-D FWHM-only kernel + the B8 count prior
(24r's best-mu treatment) while the GAMMA chain keeps the full 2-D sigma-bearing KDE (24p's design).
Reports the 24v bias-corrected mu as a side-car (read from data/processed/24v_bias.json).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24aa.ipynb')
if os.path.exists(p):
    sys.exit('24aa.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24aa', '24z',
                'combination: 24r mu treatment + 24p sigma-bearing gamma chain (Phase E1)',
                'E1: mu_gamma_split=True -- the MU reward is the 1-D FWHM-only kernel + the B8 count '
                'prior (24r best-mu treatment), while the GAMMA chain keeps the full 2-D sigma-bearing '
                'KDE (24p design) with the D2/D3 calibrated sigma; the 24v bias-corrected mu is reported '
                'as a side-car. DO_FISHER off.', ],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: add the mu/gamma split switch ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_FISHER' in s and 'CFG = dict(' in s:
        s = s.replace("    sigma_calib='table_22g',                      # D3: * per-cell 22g sigma-bias factor c(exp)",
                      "    sigma_calib='table_22g',                      # D3: * per-cell 22g sigma-bias factor c(exp)\n"
                      "    mu_gamma_split=True,                          # E1: mu = 1-D FWHM kernel (24r), gamma = 2-D sigma-bearing (24p)")
        set_txt(c, s); done = True; break
if not done: sys.exit('config cell not found')

# ---- run_experiment: split the mu reward from the gamma chain ----
done = False
for c in cells:
    s = txt(c)
    if 'def run_experiment(' in s:
        old1 = ("    else:\n"
                "        _r0 = torch.log(_W0.clamp_min(1e-30)).mean(dim=0)\n")
        new1 = ("    else:\n"
                "        if cfg.get('mu_gamma_split', False):\n"
                "            # E1 (24aa): the MU reward is the 1-D FWHM-only kernel (24r), while s_gamma\n"
                "            # (from the full 2-D KDE above) keeps its sigma term (24p).\n"
                "            _Wf0 = torch.exp(-0.5 * ((target_f[:, None] - _ft0[None, :]) / H_F) ** 2)\n"
                "            _r0 = torch.log(_Wf0.clamp_min(1e-30)).mean(dim=0)\n"
                "        else:\n"
                "            _r0 = torch.log(_W0.clamp_min(1e-30)).mean(dim=0)\n")
        assert old1 in s, 'init reward anchor not found'
        s = s.replace(old1, new1, 1)
        old2 = ("        else:\n"
                "            r = torch.log(W.clamp_min(1e-30)).mean(dim=0)             # Anuar's reward\n")
        new2 = ("        else:\n"
                "            if cfg.get('mu_gamma_split', False):\n"
                "                _Wf = torch.exp(-0.5 * ((target_f[:, None] - ft[None, :]) / H_F) ** 2)\n"
                "                r = torch.log(_Wf.clamp_min(1e-30)).mean(dim=0)   # E1: 1-D FWHM-only mu reward\n"
                "            else:\n"
                "                r = torch.log(W.clamp_min(1e-30)).mean(dim=0)     # Anuar's reward\n")
        assert old2 in s, 'step reward anchor not found'
        s = s.replace(old2, new2, 1)
        set_txt(c, s); done = True; break
if not done: sys.exit('run_experiment cell not found')

# ---- insert the E1 composition cell AFTER the frozen-metric cell ----
met_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'FROZEN PROTOCOL METRIC' in txt(c):
        met_idx = i; break
if met_idx is None: sys.exit('metric cell not found')

comp_src = '''# ============================================================
# SERIES 24 / E1 (24aa) \u2014 COMPOSITION REPORT: raw vs 24v bias-corrected mu, and the objective
#   on the corrected table.  Read-only side-car (data/processed/24v_bias.json + 24v_history.json):
#   no new simulation.
# ============================================================
_biasJ = json.load(open(os.path.join(REPO_ROOT, 'data', 'processed', '24v_bias.json')))['bias']
_histJ = json.load(open(os.path.join(REPO_ROOT, 'data', 'processed', '24v_history.json')))
_crb = {r['exp']: float((r.get('fisher_frozen') or {}).get('sig_mu_data') or float('nan'))
        for r in _histJ['results']}
_raww = _corw = 0.0; _wsum = 0.0; _cov2 = 0; _n = 0
print(f"{'exp':14s}{'mu_true':>8s}{'mu_hat':>8s}{'mu_corr':>8s} | {'|d|':>7s}{'|dc|':>7s} | {'CRB':>7s}{'|dc|/C':>8s}")
for r in RESULTS:
    k = r['exp']; b = _biasJ.get(k, {})
    if not b: continue
    T = int(str(k).split('Trans')[-1]); w = W_FLOOR + (1.0 - W_FLOOR) * T / 100.0
    mt, mh, mc = float(r['mu_true']), float(r['mu_final']), float(b.get('mu_corr', float('nan')))
    d, dc = abs(mh - mt), abs(mc - mt)
    _raww += w * min((d / mt) ** 2, CAP); _corw += w * min((dc / mt) ** 2, CAP); _wsum += w
    _n += 1; _cov2 += int(dc <= 2.0 * _crb.get(k, float('nan')))
    print(f"{k:14s}{mt:8.2f}{mh:8.2f}{mc:8.2f} | {d:7.2f}{dc:7.2f} | {_crb.get(k,float('nan')):7.2f}{dc/max(_crb.get(k,float('nan')),1e-9):8.1f}")
print(f'\\nmu-channel contribution to W_obj: RAW {_raww/_wsum:.4f} -> bias-corrected {_corw/_wsum:.4f} '
      f'(E1 mu arm; the full W_obj above includes gamma)')
print(f'bias-corrected 2-sigma CRB coverage (E1 mu table): {_cov2}/{_n}')
print('=> E1 composition: the 24r mu arm + (optionally) the 24v bias correction on the report.')
'''
cells.insert(met_idx + 1, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                               source=comp_src.splitlines(keepends=True)))

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24aa (Phase E1)

- **Change vs previous notebook (`24z`):** `mu_gamma_split=True` \u2014 the **\u03bc reward** becomes the **1-D
  FWHM-only kernel + the B8 count prior** (24r's best-\u03bc treatment), while the **\u03b3 chain keeps the full 2-D
  \u03c3-bearing KDE** (24p's design) with the D2/D3 calibrated \u03c3.  The 24v bias-corrected \u03bc is reported as a
  side-car (read from `data/processed/24v_bias.json`; no new simulation).  This is the "best of each phase"
  combination: A (travel, two-phase) + B (24r \u03bc role-split + count prior) + C (\u03c3 calibration) + D (\u03c3 model).
- **Hypothesis (E1):** the series best \u03bc (24r) and the best low-T \u03b3 (24p) are *compatible* \u2014 using the FWHM-only
  kernel for \u03bc but keeping \u03c3 in the \u03b3 chain collects both, so `W_obj(all14)` beats **24r's 0.0672**.
- **Falsifiable prediction:** `W_obj(all14) < 0.0672` and \u03b3 rel-RMSE < 24r's 35.1 %, with \u03bc rel-RMSE \u2264 25.5 %.
  **Falsified if** the split does not beat 24r \u2014 i.e. the \u03bc and \u03b3 role-splits cannot be combined without one
  paying for the other, and the E-phase must fall back to 24r as the final model.
- **If falsified:** declare 24r the series winner and report the E2/E3 verification as the remaining work.
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
