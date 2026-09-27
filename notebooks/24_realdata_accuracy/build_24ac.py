#!/usr/bin/env python3
"""Build 24ac from 24ab — Phase E3: independent referee (exp8 cvm_fwhm / w1_2d).

ONE change: an independent, Fisher-free referee (exp8's bandwidth-free distributional
objectives cvm_fwhm and w1_2d) evaluated on the E1 winner and the series' key candidates,
computed from FRESH sims at each candidate's final (mu,gamma).  The E2 3-seed repeat is
turned back into a single campaign (E2_SEEDS=[42]) and read from 24ab's side-car.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24ac.ipynb')
if os.path.exists(p):
    sys.exit('24ac.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24ac', '24ab',
                'E3: independent referee (exp8 cvm_fwhm / w1_2d) on the E1 winner + candidates',
                'E3: run the E1 winner (and the series key candidates) through exp8\'s frozen, '
                'Fisher-free distributional objectives cvm_fwhm (squared-quantile, FWHM only) and '
                'w1_2d (Wasserstein-1 on FWHM + sigma), computed from fresh sims at each candidate\'s '
                'final (mu,gamma); a second, independent metric must agree with the T-weighted W_obj.'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: E3 is a single campaign again (the E2 repeat is read from 24ab's side-car) ----
done = False
for c in cells:
    s = txt(c)
    if 'E2_SEEDS = [42, 43, 44]' in s:
        s = s.replace("E2_SEEDS = [42, 43, 44]  # E2 (24ab): repeat the frozen campaign at these seed bases\n"
                      "                         #   (only the per-step noise seed changes between repeats)\n",
                      "E2_SEEDS = [42]        # E3 (24ac): single campaign; the 24ab 3-seed repeat is read below\n")
        set_txt(c, s); done = True; break
if not done: sys.exit('config E2_SEEDS not found')

# ---- replace the inherited E2 compute cell with a read-only side-car cell ----
e2_old_head = 'E2 \u2014 repeat of the E1 winner at different SEED bases:'
e2_src = '''# ============================================================
# SERIES 24 / E2 (24ab) \u2014 SIGNAL vs CHAOS BAND (read-only side-car)
#   The E1 winner was repeated at three SEED bases in 24ab; this notebook READS that result
#   (data/processed/24ab_history.json -> e2_by_seed) instead of re-running the triple campaign.
# ============================================================
E2_OBJ = {}
_e2p = os.path.join(REPO_ROOT, 'data', 'processed', '24ab_history.json')
if os.path.exists(_e2p):
    E2_OBJ = {int(k): v for k, v in (json.load(open(_e2p)).get('e2_by_seed') or {}).items()}
if E2_OBJ:
    print('E2 (24ab) \u2014 E1 winner repeated at SEED bases:')
    print(f"{'seed':>6}{'all14':>10}{'T>=40':>10}{'T<=20':>10}{'mu%':>8}{'gam%':>8}")
    for _s, _o in E2_OBJ.items():
        print(f"{_s:>6}{_o['all14']:>10.4f}{_o['hiT']:>10.4f}{_o['loT']:>10.4f}"
              f"{_o['mu_rel_rmse']:>8.2f}{_o['gamma_rel_rmse']:>8.2f}")
    _a14 = np.array([_o['all14'] for _o in E2_OBJ.values()])
    _sd = float(_a14.std(ddof=1)) if len(_a14) > 1 else 0.0
    print(f"all14: mean {_a14.mean():.4f}  sd {_sd:.4f}  range [{_a14.min():.4f}, {_a14.max():.4f}]")
    _m = 0.0672 - _a14.mean()
    print(f"vs 24r 0.0672: margin {_m:+.4f}  ({'SEPARATED' if abs(_m) > _sd else 'inside the chaos band'})")
else:
    print('E2 side-car (24ab_history.json -> e2_by_seed) not found \u2014 E2 evidence unavailable here.')
'''
done = False
for c in cells:
    if c['cell_type'] == 'code' and e2_old_head in txt(c):
        set_txt(c, e2_src); done = True; break
if not done: sys.exit('E2 cell anchor not found')

# ---- insert the E3 referee cell right AFTER the E2 cell ----
e2_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'E2 (24ab) \u2014 SIGNAL vs CHAOS' in txt(c):
        e2_idx = i; break
if e2_idx is None: sys.exit('E2 cell not located for insertion')

ref_src = '''# ============================================================
# SERIES 24 / E3 (24ac) \u2014 INDEPENDENT REFEREE (exp8, frozen, Fisher-free)
#   Evaluate the E1 winner (and the series' key candidates) with exp8's bandwidth-free
#   distributional objectives:
#     cvm_fwhm = Cramer-von-Mises-style squared quantile distance on FWHM only   (power=2)
#     w1_2d    = Wasserstein-1 on FWHM  +  sw * Wasserstein-1 on sigma           (power=1)
#   Both are DISCREPANCY-to-the-real-data scores (lower = better fit), computed from FRESH
#   sims at each candidate's final (mu,gamma) and aggregated with the SAME T weights as W_obj.
#   Loss scales are exp8's frozen FAMILY_SCALE.  TRUTH-FREE: mu_true/gamma_true are never used.
#   This is the second, independent metric: it must AGREE with the T-weighted W_obj ranking.
# ============================================================
_FS = {'cvm_fwhm': 0.028428, 'w1_2d': 0.733844}      # exp8 FAMILY_SCALE (frozen)
REF_CANDS = ['24a', '24g', '24i', '24n', '24p', '24r', '24y', '24aa']
REF_SEEDS = 2

def _ref_losses(pool, exp, mu, gamma, ccfg):
    tf, ts = _load_real_target(exp)
    tgt_f = _sorted_target(tf); tgt_s = _sorted_target(ts)
    sw = float(ccfg.get('sigma_weight', 1.0))
    cv = w1 = 0.0
    _nrun = 20 if SMOKE else int(ccfg.get('n_runs', 100))
    for si in range(REF_SEEDS):
        ft, st, nt, dgt, dst = _sims(pool, float(mu), exp['sigma_prop'], exp['lam'], float(gamma),
                                     _nrun, 20240 + 17 * si,
                                     sigma_est=ccfg.get('sigma_estimator', 'lorentzian'))
        if str(ccfg.get('sigma_calib', 'none')) != 'none':
            _c = float(SIGMA_CALIB.get(exp['name'], 1.0)); st = st * _c; dst = dst * _c
        cv += float(_quantile_loss(ft, dgt, tgt_f, power=2)[0].mean()) * _FS['cvm_fwhm']
        _lf = _quantile_loss(ft, dgt, tgt_f, power=1)[0]
        _ls = _quantile_loss(st, dst, tgt_s, power=1)[0]
        w1 += float((_lf + sw * _ls).mean()) * _FS['w1_2d']
    return cv / REF_SEEDS, w1 / REF_SEEDS

REF_OBJ = {}
print('E3 (24ac) \u2014 independent referee (exp8 cvm_fwhm / w1_2d), T-weighted over the 14 real exps')
print(f"{'cand':>6}{'cvm_fwhm':>11}{'w1_2d':>11}{'W_obj(all14)':>14}")
with _PPE(max_workers=N_WORKERS, mp_context=_mp.get_context('fork'), initializer=_init_worker) as _rfp:
    for cid in REF_CANDS:
        _hp = os.path.join(REPO_ROOT, 'data', 'processed', f'{cid}_history.json')
        if not os.path.exists(_hp):
            print(f'  {cid}: history missing -> skipped'); continue
        _hj = json.load(open(_hp))
        # faithful forward model per candidate: only the sigma-model keys are taken from its
        # OWN saved config (None -> the frozen pre-flag default); everything else is the E1 CFG.
        _ccfg = dict(CFG); _hc = _hj.get('config') or {}
        _ccfg['sigma_channel'] = _hc.get('sigma_channel') or 'on'
        _ccfg['sigma_estimator'] = _hc.get('sigma_estimator') or 'lorentzian'
        _ccfg['sigma_calib'] = _hc.get('sigma_calib') or 'none'
        _ccfg['sigma_weight'] = float(_hc.get('sigma_weight', CFG['sigma_weight']))
        _ccfg['n_runs'] = int(_hc.get('n_runs', CFG['n_runs']))
        _fin = {r['exp']: r for r in _hj['results']}
        _cw = _ww = _psum = 0.0; _per = {}
        for exp in EXPERIMENTS:
            r = _fin.get(exp['name'])
            if r is None: continue
            T = int(str(exp['name']).split('Trans')[-1]); w = W_FLOOR + (1.0 - W_FLOOR) * T / 100.0
            c_, w_ = _ref_losses(_rfp, exp, r['mu_final'], r['gamma_final'], _ccfg)
            _per[exp['name']] = [round(c_, 6), round(w_, 6)]
            _cw += w * c_; _ww += w * w_; _psum += w
        REF_OBJ[cid] = dict(cvm_fwhm=_cw / _psum, w1_2d=_ww / _psum, per_exp=_per,
                            w_obj=float((_hj.get('objective') or {}).get('all14', float('nan'))))
        print(f"{cid:>6}{REF_OBJ[cid]['cvm_fwhm']:>11.5f}{REF_OBJ[cid]['w1_2d']:>11.5f}"
              f"{REF_OBJ[cid]['w_obj']:>14.4f}", flush=True)

def _rank(key, rev=False):
    return sorted(REF_OBJ, key=lambda k: REF_OBJ[k][key], reverse=rev)

_w, _c, _s = _rank('w_obj'), _rank('cvm_fwhm'), _rank('w1_2d')
print()
print('rank by W_obj    :', ' > '.join(_w))
print('rank by cvm_fwhm :', ' > '.join(_c))
print('rank by w1_2d    :', ' > '.join(_s))
try:
    from scipy.stats import spearmanr
    _kw = [_w.index(k) for k in _w]
    for _nm, _rk in [('cvm_fwhm', _c), ('w1_2d', _s)]:
        _x = [REF_OBJ[k]['w_obj'] for k in REF_OBJ]
        _y = [REF_OBJ[k][_nm] for k in REF_OBJ]
        _rho, _pv = spearmanr(_x, _y)
        print(f"Spearman(W_obj, {_nm}) = {_rho:+.3f} (p {_pv:.3f})")
except Exception as _e:
    print('spearman skipped:', _e)
_pc = _w.index('24aa') if '24aa' in _w else -1
print(f"winner 24aa rank: W_obj #{_pc + 1} | cvm_fwhm #{_c.index('24aa') + 1 if '24aa' in _c else -1} "
      f"| w1_2d #{_s.index('24aa') + 1 if '24aa' in _s else -1}")
print('=> E3: the independent distributional referee should reproduce the W_obj ordering of the candidates.')

if not SMOKE:
    _ro = os.path.join(REPO_ROOT, 'data', 'processed', '24ac_referee.json')
    json.dump(dict(notebook=NB_ID, method='exp8_referee', family_scale=_FS, seeds=REF_SEEDS,
                   ranking=dict(w_obj=_w, cvm_fwhm=_c, w1_2d=_s), referee=REF_OBJ),
              open(_ro, 'w'))
    print('saved', _ro)
else:
    print('smoke: referee JSON not saved')
'''
cells.insert(e2_idx + 1, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                              source=ref_src.splitlines(keepends=True)))

# ---- save cell: add the referee table ----
done = False
for c in cells:
    s = txt(c)
    if 'e2_seeds=list(E2_SEEDS), e2_by_seed=E2_OBJ,' in s:
        s = s.replace("               e2_seeds=list(E2_SEEDS), e2_by_seed=E2_OBJ,",
                      "               e2_seeds=list(E2_SEEDS), e2_by_seed=E2_OBJ,\n"
                      "               referee=REF_OBJ,")
        set_txt(c, s); done = True; break
if not done: sys.exit('save cell e2 anchor not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24ac (Phase E3)

- **Change vs previous notebook (`24ab`):** add an **independent referee**.  The E1 winner and the
  series' key candidates (`24a`, `24g`, `24i`, `24n`, `24p`, `24r`, `24y`, `24aa`) are each evaluated
  with **exp8's frozen, Fisher-free distributional objectives** \u2014 `cvm_fwhm` (squared-quantile,
  FWHM only) and `w1_2d` (Wasserstein-1 on FWHM + `sw`\u00b7\u03c3) \u2014 from **fresh sims at each candidate's
  final (mu, gamma)**, aggregated with the same T weights as `W_obj`.  `E2_SEEDS=[42]`: the triple
  campaign is not repeated (it is read from `24ab`'s side-car).
- **Hypothesis (E3):** `W_obj` (truth-weighted) and the exp8 referee (data-discrepancy, truth-free) are
  two independent ways to score the same models; if the series' finding is real, the model with the
  lowest `W_obj` (the E1 winner) must also rank at/near the top of the referee, and the candidate
  orderings must correlate.
- **Falsifiable prediction:** the E1 winner is in the **top-2** of both `cvm_fwhm` and `w1_2d`, and the
  Spearman rank correlation between `W_obj` and each referee objective is **\u2265 0.5**.  **Falsified if**
  the winner ranks outside the top-3, or the correlation is \u2264 0 \u2014 then the truth-weighted objective and
  the data-discrepancy metric disagree, and the series' "best model" is an artefact of the metric.
- **If falsified:** report the disagreement explicitly (it is a first-class negative result about the
  metric) and prefer the candidate that wins both.
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
