#!/usr/bin/env python3
"""Build 24u from 24t — Phase C3: profile-likelihood interval for mu.

Adds DO_PROFILE=True -- scan mu at fixed gamma_hat on a truth-free grid around the optimum,
evaluate the SAME frozen KDE NLL, and find the Delta(-2 logL) = 2*N*(NLL(mu)-NLL_min) <= 1
interval (= 1 sigma, 1 dof).  Budget reduction: 13 grid points at +/-5 dataset-CRB sigmas,
2 sim seeds averaged per point.  DO_FISHER stays on (needs the CRB scale); bootstrap off.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24u.ipynb')
if os.path.exists(p):
    sys.exit('24u.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24u', '24t',
                'profile-likelihood interval for mu (Phase C3)',
                'C3: DO_PROFILE=True -- scan mu at fixed gamma_hat on a truth-free grid around the optimum '
                'and find the Delta(-2 logL)=2N*(NLL(mu)-NLL_min)<=1 profile interval, compared with the '
                'CRB and the 24s bootstrap. Budget reduction: 13 grid points at +/-5 dataset-CRB sigmas, '
                '2 sim seeds averaged per point.', ],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: add DO_PROFILE ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_BOOTSTRAP' in s and 'CFG = dict(' in s:
        s = s.replace("DO_BOOTSTRAP = False   # C2 keeps C1 off (bootstrap read from data/processed/24s_bootstrap.json)",
                      "DO_BOOTSTRAP = False   # C2 keeps C1 off (bootstrap read from data/processed/24s_bootstrap.json)\n"
                      "DO_PROFILE = True      # C3 (24u): profile-likelihood 1-sigma interval for mu")
        set_txt(c, s); done = True; break
if not done: sys.exit('config cell not found')

# ---- insert the profile cell AFTER the Fisher cell ----
fish_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'FISHER at each optimum' in txt(c):
        fish_idx = i; break
if fish_idx is None: sys.exit('Fisher cell not found')

prof_src = '''# ============================================================
# SERIES 24 / C3 (24u) \u2014 PROFILE-LIKELIHOOD INTERVAL FOR mu
#   Scan mu at fixed gamma=gamma_hat on a truth-free grid around the optimum, evaluate the
#   SAME frozen KDE NLL (sigma-channel state inherited), and find the Delta(-2 logL) =
#   2*N*(NLL(mu) - NLL_min) <= 1 interval (= 1 sigma, 1 d.o.f.).  Budget reduction (recorded):
#   13 grid points at +/-5 dataset-CRB sigmas, 2 sim seeds averaged per point.
# ============================================================
PROFILE_PTS, PROFILE_SEEDS, PROFILE_HALFWIDTH = 13, 2, 5.0
PROF = {}
if DO_PROFILE:
    with _PPE(max_workers=N_WORKERS, mp_context=_mp.get_context('fork'), initializer=_init_worker) as _ppool:
        for r in RESULTS:
            exp = next(e for e in EXPERIMENTS if e['name'] == r['exp'])
            tf_pp = torch.tensor(r['target_f'], dtype=torch.float32)
            ts_pp = torch.tensor(r['target_s'], dtype=torch.float32)
            H_Fp, H_Sp = r['H_F'], r['H_S']
            mu_hat, g_hat = float(r['mu_final']), float(r['gamma_final'])
            fr = r.get('fisher_frozen') or {}
            _sp = float(fr.get('sig_mu_data') or (0.15 * abs(mu_hat)))
            _sp = max(_sp, 1e-6)
            grid = np.clip(mu_hat + np.linspace(-PROFILE_HALFWIDTH, PROFILE_HALFWIDTH, PROFILE_PTS) * _sp, 0.05, 1500.0)
            Npp = int(len(tf_pp))
            nll_g = np.zeros(PROFILE_PTS)
            for gi, mg in enumerate(grid):
                _acc = 0.0
                for si in range(PROFILE_SEEDS):
                    _ft, _si, _nt, _dg, _ds = _sims(_ppool, float(mg), r['sigma_prop'], r['lam'], g_hat,
                                                     CFG['n_runs'], SEED + 777 + si)
                    _acc += float(_kde_scores(_ft, _si, _nt, _dg, _ds, tf_pp, ts_pp, H_Fp, H_Sp,
                                              float(mg), r['sigma_prop'], CFG)[2])
                nll_g[gi] = _acc / PROFILE_SEEDS
            d2 = 2.0 * Npp * (nll_g - nll_g.min())          # Delta(-2 logL), total-likelihood scale
            _amin = int(np.argmin(d2))
            lo = hi = float('nan')
            for gi in range(PROFILE_PTS - 1):
                d0, d1 = d2[gi], d2[gi + 1]
                if (d0 - 1.0) * (d1 - 1.0) < 0.0:
                    _f = (1.0 - d0) / (d1 - d0)
                    _x = float(grid[gi] + _f * (grid[gi + 1] - grid[gi]))
                    if gi < _amin:
                        lo = _x
                    else:
                        hi = _x
            PROF[r['exp']] = dict(mu_hat=mu_hat, gamma_hat=g_hat, N=Npp,
                                  grid=[round(float(x), 4) for x in grid],
                                  d2=[round(float(x), 4) for x in d2], lo=lo, hi=hi,
                                  sig_mu_data=fr.get('sig_mu_data'), sig_mu_single=fr.get('sig_mu_single'),
                                  mu_true=exp['mu_true'])
            _w = (hi - lo) if not (np.isnan(hi) or np.isnan(lo)) else float('nan')
            print(f"  {r['exp']:13s} mu_hat {mu_hat:7.2f} | profile [{lo:8.3f},{hi:8.3f}] (w {_w:6.3f}) "
                  f"| CRB_data {_sp:6.3f} | mu_true {exp['mu_true']:7.2f}", flush=True)
    _pcov = sum(1 for v in PROF.values()
                if not (np.isnan(v['lo']) or np.isnan(v['hi'])) and v['lo'] <= v['mu_true'] <= v['hi'])
    _pn = sum(1 for v in PROF.values() if not (np.isnan(v['lo']) or np.isnan(v['hi'])))
    _pw = [v['hi'] - v['lo'] for v in PROF.values() if not np.isnan(v['hi'])]
    print(f"profile 1-sigma (Delta=1) coverage of mu_true: {_pcov}/{_pn} resolved ({len(PROF)} cells total)")
    if _pw:
        print(f"median profile width {float(np.median(_pw)):.3f} vs median dataset CRB "
              f"{float(np.median([v['sig_mu_data'] for v in PROF.values() if v['sig_mu_data']])):.3f}")
    else:
        print('NO profile interval resolved on the grid (Delta(-2logL) > 1 everywhere) -- see verdict')
    if not SMOKE:
        _out = os.path.join(REPO_ROOT, 'data', 'processed', '24u_profile.json')
        json.dump(dict(notebook=NB_ID, method='profile', pts=PROFILE_PTS, seeds=PROFILE_SEEDS,
                       halfwidth=PROFILE_HALFWIDTH, coverage_mu=_pcov, resolved=_pn, profile=PROF),
                  open(_out, 'w'))
        print('saved', _out)
else:
    print('DO_PROFILE=False - profile block skipped')
'''
cells.insert(fish_idx + 1, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                                source=prof_src.splitlines(keepends=True)))

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24u (Phase C3)

- **Change vs previous notebook (`24t`):** `DO_PROFILE=True` \u2014 a **profile-likelihood interval for \u03bc**.
  At the optimum we scan `\u03bc` on a truth-free grid (`\u03bc\u0302 \u00b1 5 \u00d7` the *dataset* CRB \u03c3, 13 points) with
  `\u03b3 = \u03b3\u0302` fixed, evaluate the **same frozen KDE NLL** (`\u03c3`-channel state inherited, 2 sim seeds
  averaged per grid point) and find the interval where `\u0394(-2 logL) = 2N(NLL(\u03bc)\u2212NLL_min) \u2264 1` (1\u03c3,
  1 d.o.f.).  `DO_FISHER` stays on (the CRB supplies the grid scale); the bootstrap is off (read from 24s).
- **Hypothesis (C3):** the CRB is a **local-curvature** claim and is blind to the estimator's **bias**
  (FM#8/#12).  A profile interval is built from the *actual* likelihood shape, so it should be
  **asymmetric and/or wider** exactly where the point estimate is biased (the high-T \u03bc undershoot cells
  and the low-T \u03bc-overshoot-free cells of 24r), and should **contain `\u03bc_true` more often** than the CRB.
- **Falsifiable prediction:** the profile interval is **resolvable** (\u0394(-2logL) dips below 1 inside the grid)
  for at least ~12/14 cells, its **median width \u2265 the dataset CRB** (expect \u2265 1.5\u00d7), and its coverage of
  `\u03bc_true` is **\u2265 8/14** \u2014 better than the CRB's (\u2264 8/14 in the 23-series) and comparable to the 24s
  bootstrap.  **Falsified if** the interval is unresolvable (the 100-sim KDE NLL is *too noisy* at the
  \u03c3_data scale \u2192 FM#11: no honest interval from the likelihood at this budget) or its coverage is no
  better than the CRB's.
- **If falsified:** the KDE likelihood at this sim budget carries **less information than its own CRB
  claims** \u2192 both the CRB and the profile are unusable \u2192 the only honest interval is the **data bootstrap**
  (24s) + the bias correction (C4).
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
for c in cells:
    for ln in txt(c).splitlines():
        if 'DO_PROFILE' in ln or 'PROFILE_PTS' in ln:
            print('   ', ln.strip())
