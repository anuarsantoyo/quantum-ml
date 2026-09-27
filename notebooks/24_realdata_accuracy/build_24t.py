#!/usr/bin/env python3
"""Build 24t from 24s — Phase C2: the sandwich J^-1 K J^-1 vs the plain J^-1 CRB.

Turns DO_FISHER=True (Fisher over all 14, M_FINAL=150, 1 seed) and extends `_fisher_at` to
return K (empirical centred score covariance) and H (observed information, central-difference
Hessian of the frozen mean NLL), then the sandwich S = H^-1 K H^-1 (per-scan; dataset = /N).
DO_BOOTSTRAP stays off (C1 numbers are read from data/processed/24s_bootstrap.json).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24t.ipynb')
if os.path.exists(p):
    sys.exit('24t.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24t', '24s',
                'sandwich J^-1 K J^-1 vs the plain J^-1 CRB (Phase C2)',
                'C2: DO_FISHER=True -- compute the heteroskedasticity-robust (sandwich) covariance '
                'S = H^-1 K H^-1 at each optimum, with K the empirical centred score covariance and H '
                'the observed information (central-difference Hessian of the frozen mean NLL); report '
                'CRB vs sandwich widths and coverage over the 14 cells. Bootstrap OFF (C1 read from 24s).',
                check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: DO_FISHER ON, bootstrap OFF ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_FISHER' in s and 'CFG = dict(' in s:
        s = s.replace("DO_FISHER  = False     # C1 is a point-estimate + bootstrap notebook (Fisher only inside the bootstrap cell)",
                      "DO_FISHER  = True      # C2 (24t): Fisher + sandwich over all 14 cells")
        s = s.replace("DO_BOOTSTRAP = True    # C1 (24s): empirical data bootstrap of the retained real scans",
                      "DO_BOOTSTRAP = False   # C2 keeps C1 off (bootstrap read from data/processed/24s_bootstrap.json)")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config cell not found')

# ---- _fisher_at: add K + observed-information Hessian + sandwich ----
done = False
for c in cells:
    s = txt(c)
    a = """    Js = []
    try:
        for s_i in range(FISHER_SEEDS):
            ft, si_t, nt, dg_t, ds_t = _sims(pool, mu, sigma_prop, lam, gamma, M_FINAL, FISHER_BASE_SEED + s_i)
            s_mu, s_gamma, _, _, _ = _kde_scores(ft, si_t, nt, dg_t, ds_t,
                                                 target_f, target_s, H_F, H_S, mu, sigma_prop, cfg)
            s = torch.stack([s_mu, s_gamma], dim=1)
            Js.append(s.T @ s / N)                 # the EXISTING convention (per-scan)
    finally:
        GAMMA_SCALE = gs
    J_single = torch.stack(Js).mean(dim=0)
    inv_single = torch.linalg.inv(J_single + 1e-12 * torch.eye(2))
    inv_data = inv_single / N                      # dataset CRB = (N*J)^-1
    out = dict(N=N, J=J_single.tolist(),
               sig_mu_single=float(math.sqrt(inv_single[0, 0])),
               sig_g_single=float(math.sqrt(inv_single[1, 1])),
               corr_single=float(inv_single[0, 1] / math.sqrt(inv_single[0, 0] * inv_single[1, 1])),
               sig_mu_data=float(math.sqrt(inv_data[0, 0])),
               sig_g_data=float(math.sqrt(inv_data[1, 1])),
               corr_data=float(inv_data[0, 1] / math.sqrt(inv_data[0, 0] * inv_data[1, 1])))
    return out
"""
    b = """    Js = []; Ks = []
    try:
        for s_i in range(FISHER_SEEDS):
            ft, si_t, nt, dg_t, ds_t = _sims(pool, mu, sigma_prop, lam, gamma, M_FINAL, FISHER_BASE_SEED + s_i)
            s_mu, s_gamma, _, _, _ = _kde_scores(ft, si_t, nt, dg_t, ds_t,
                                                 target_f, target_s, H_F, H_S, mu, sigma_prop, cfg)
            s = torch.stack([s_mu, s_gamma], dim=1)
            Js.append(s.T @ s / N)                 # the EXISTING convention (per-scan)
            # --- C2 (24t): K = empirical CENTRED score covariance (per-scan normalisation) ---
            _sc = s - s.mean(dim=0, keepdim=True)
            Ks.append(_sc.T @ _sc / N)
    finally:
        GAMMA_SCALE = gs
    J_single = torch.stack(Js).mean(dim=0)
    K_single = torch.stack(Ks).mean(dim=0)
    inv_single = torch.linalg.inv(J_single + 1e-12 * torch.eye(2))
    inv_data = inv_single / N                      # dataset CRB = (N*J)^-1
    # --- C2 (24t): the SANDWICH  S = H^-1 K H^-1  vs the plain CRB J^-1 -------------------
    #   H = OBSERVED information = Hessian of the mean NLL of the SAME frozen likelihood,
    #   central differences with FRESH sim clouds.  The FD step is 0.7x the per-scan CRB sigma
    #   (truth-free: taken from the Fisher at this optimum, never from mu_true/gamma_true).
    HM = max(1e-3, 0.7 * float(math.sqrt(inv_single[0, 0])))
    HG = max(1e-3, 0.7 * float(math.sqrt(inv_single[1, 1])))
    gs2 = GAMMA_SCALE; GAMMA_SCALE = False
    def _nll_at(mu_, gm_):
        _ft, _si, _nt, _dg, _ds = _sims(pool, mu_, sigma_prop, lam, gm_, M_FINAL, FISHER_BASE_SEED + 500)
        return float(_kde_scores(_ft, _si, _nt, _dg, _ds, target_f, target_s, H_F, H_S, mu_, sigma_prop, cfg)[2])
    try:
        f0 = _nll_at(mu, gamma)
        Hmm = (_nll_at(mu - HM, gamma) - 2.0 * f0 + _nll_at(mu + HM, gamma)) / HM ** 2
        Hgg = (_nll_at(mu, gamma - HG) - 2.0 * f0 + _nll_at(mu, gamma + HG)) / HG ** 2
        Hmg = (_nll_at(mu + HM, gamma + HG) - _nll_at(mu + HM, gamma - HG)
               - _nll_at(mu - HM, gamma + HG) + _nll_at(mu - HM, gamma - HG)) / (4.0 * HM * HG)
    finally:
        GAMMA_SCALE = gs2
    H_obs = torch.tensor([[Hmm, Hmg], [Hmg, Hgg]], dtype=torch.float32)
    H_obs = 0.5 * (H_obs + H_obs.T)
    _ev = torch.linalg.eigvalsh(H_obs)
    if float(_ev.min()) <= 1e-9:
        H_obs = H_obs + (1e-6 - float(_ev.min())) * torch.eye(2)   # PSD floor (fragile observed info, FM#11)
    _invH = torch.linalg.inv(H_obs)
    sand_single = _invH @ K_single @ _invH
    sand_data = sand_single / N
    out = dict(N=N, J=J_single.tolist(), K=K_single.tolist(), H=H_obs.tolist(),
               sig_mu_single=float(math.sqrt(inv_single[0, 0])),
               sig_g_single=float(math.sqrt(inv_single[1, 1])),
               corr_single=float(inv_single[0, 1] / math.sqrt(inv_single[0, 0] * inv_single[1, 1])),
               sig_mu_data=float(math.sqrt(inv_data[0, 0])),
               sig_g_data=float(math.sqrt(inv_data[1, 1])),
               corr_data=float(inv_data[0, 1] / math.sqrt(inv_data[0, 0] * inv_data[1, 1])),
               sig_mu_sand_single=float(math.sqrt(max(sand_single[0, 0], 0.0))),
               sig_g_sand_single=float(math.sqrt(max(sand_single[1, 1], 0.0))),
               sig_mu_sand=float(math.sqrt(max(sand_data[0, 0], 0.0))),
               sig_g_sand=float(math.sqrt(max(sand_data[1, 1], 0.0))))
    return out
"""
    if a in s:
        set_txt(c, s.replace(a, b)); done = True; break
if not done: sys.exit('_fisher_at body not found')

# ---- cell 12: add sandwich columns + coverage ----
done = False
for c in cells:
    s = txt(c)
    a = """                   dg_sigma_single=dg / fr['sig_g_single'], dg_sigma_data=dg / fr['sig_g_data'])"""
    if a in s:
        s = s.replace(a,
            """                   dg_sigma_single=dg / fr['sig_g_single'], dg_sigma_data=dg / fr['sig_g_data'],
                   sig_mu_sand=fr.get('sig_mu_sand'), sig_g_sand=fr.get('sig_g_sand'),
                   dmu_sigma_sand=(dmu / fr['sig_mu_sand']) if fr.get('sig_mu_sand') else None,
                   dg_sigma_sand=(dg / fr['sig_g_sand']) if fr.get('sig_g_sand') else None)""")
        done = True; break
if not done: sys.exit('row.update not found')

done = False
for c in cells:
    s = txt(c)
    a = """    print('coverage within 2 sigma (mu):  per-scan %d/14 | dataset %d/14'
          % (cov('dmu_sigma_single'), cov('dmu_sigma_data')))
    print('coverage within 1 sigma (mu):  per-scan %d/14 | dataset %d/14'
          % (cov('dmu_sigma_single', 1), cov('dmu_sigma_data', 1)))"""
    if a in s:
        s = s.replace(a,
            """    print('coverage within 2 sigma (mu):  per-scan %d/14 | dataset %d/14 | SANDWICH %d/14'
          % (cov('dmu_sigma_single'), cov('dmu_sigma_data'), cov('dmu_sigma_sand')))
    print('coverage within 1 sigma (mu):  per-scan %d/14 | dataset %d/14 | SANDWICH %d/14'
          % (cov('dmu_sigma_single', 1), cov('dmu_sigma_data', 1), cov('dmu_sigma_sand', 1)))
    _sm = [t['dmu_sigma_sand'] / t['dmu_sigma_data'] for t in TABLE if t.get('dmu_sigma_sand')]
    if _sm:
        print('median sandwich/CRB sigma_mu ratio (dataset) = %.2f' % float(np.median(_sm)))"""
        )
        done = True; break
if not done: sys.exit('coverage print not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24t (Phase C2)

- **Change vs previous notebook (`24s`):** `DO_FISHER=True` (the bootstrap is switched off; C1 numbers are
  read from `data/processed/24s_bootstrap.json`).  `_fisher_at` now also returns the **empirical centred
  score covariance** `K` and the **observed information** `H` (central-difference Hessian of the *frozen
  mean NLL* with fresh sim clouds, FD step = 0.7\u00d7 the per-scan CRB \u03c3 \u2014 truth-free), and the
  **sandwich** `S = H\u207b\u00b9 K H\u207b\u00b9` (per-scan; dataset = `/N`).  Everything else byte-identical.
- **Hypothesis (C2 / FM#11 #12):** the plain CRB `J\u207b\u00b9` is the *expected*-information guess and is
  bias- and outlier-blind; the real likelihood is **misspecified** (FM#8, heavy-tailed fit-error), so the
  curvature `H` and the score covariance `K` disagree (`K \u226b H`).  The **sandwich** is the
  heteroskedasticity-robust variance and should (i) be **wider** than the CRB and (ii) **restore coverage**
  at high T where the CRB misses the truth.
- **Falsifiable prediction:** median `sandwich/CRB \u2265 1.5`, and the **2\u03c3 sandwich coverage on \u03bc beats
  the CRB's** (dataset CRB was ~\u22648/14 in the 23-series; expect sandwich \u2265 11/14 and much better at
  T \u2265 40).  **Falsified if** the median ratio \u2248 1 (K \u2248 H \u2192 the model is well specified at the optimum
  and the misspecification is *bias*, not variance) or the sandwich coverage is no better.
- **If falsified:** the interval problem is **bias** (FM#8), not variance \u2192 the CRB/sandwich family is
  bounded by curvature \u2192 escalate to the **bias-aware profile likelihood** (C3).
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
for c in cells:
    for ln in txt(c).splitlines():
        if 'DO_FISHER  =' in ln or 'DO_BOOTSTRAP' in ln or 'K_single' in ln or 'sig_mu_sand' in ln:
            print('   ', ln.strip())
