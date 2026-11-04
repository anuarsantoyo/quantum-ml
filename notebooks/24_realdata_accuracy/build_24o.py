#!/usr/bin/env python3
"""Build 24o from 24n — Phase B5: estimator-matched sigma_fit (pseudo-Voigt sigma channel)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24o.ipynb')
if os.path.exists(p):
    sys.exit('24o.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24o', '24n',
                'estimator-matched sigma_fit: pseudo-Voigt sigma channel (Phase B5)',
                'B5: the SIGMA channel value comes from the estimator-matched pseudo-Voigt fit on the SAME '
                'photons (n_params=3, the 22g pseudo-Voigt estimator, 28%% median mismatch vs 62%% for the '
                'analytic Lorentzian CRLB) instead of the analytic CRLB; FWHM and dFWHM/dgamma stay with the '
                'frozen Lorentzian fit; the B4 t-kernel is switched OFF (branch measured)'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config ----
found = False
for c in cells:
    s = txt(c)
    s2 = s.replace(
        "    sigma_kernel='student_t', sigma_t_nu=3.0,    # B4: robust (heavy-tailed) sigma-channel kernel\n",
        "    sigma_kernel='gauss',                        # B4 t-kernel OFF (branch measured)\n"
        "    sigma_estimator='pseudo_voigt',              # B5: sigma channel from the pseudo-Voigt estimator\n")
    if s2 != s:
        set_txt(c, s2); found = True; break
if not found: sys.exit('config: sigma_kernel line not found')

# ---- helpers: pv fitters + a pv worker + a fn-switchable parallel map ----
done = False
for c in cells:
    s = txt(c)
    if "def _fit_fn(ph):  return fit_profile(ph, n_iters=80, model='lorentzian', uniform_bg=False)\n" in s:
        s = s.replace(
            "def _fit_fn(ph):  return fit_profile(ph, n_iters=80, model='lorentzian', uniform_bg=False)\n",
            "def _fit_fn(ph):  return fit_profile(ph, n_iters=80, model='lorentzian', uniform_bg=False)\n"
            "# --- B5 (24o): the estimator-matched (pseudo-Voigt) fits, used for the SIGMA channel only ---\n"
            "def _fit_pv(ph):   return fit_profile(ph, n_iters=80, model='pseudo-voigt', uniform_bg=False)\n"
            "def _fwhm_pv(th):  return fwhm_from_theta(th, model='pseudo-voigt')\n"
            "def _nll_pv(th, ph): return nll(th, ph, model='pseudo-voigt', uniform_bg=False)\n")
        s = s.replace(
            "def _init_worker(): torch.set_num_threads(1)\n"
            "def _parallel_map(pool, tasks): return list(pool.map(_run_one, tasks, chunksize=8))\n",
            "def _run_one_pv(args):\n"
            "    gamma_val, u, b = args\n"
            "    return compute_fwhm_and_dgamma(gamma_val, u, b, _fit_pv, _fwhm_pv, _nll_pv, n_params=3)\n\n"
            "def _init_worker(): torch.set_num_threads(1)\n"
            "def _parallel_map(pool, tasks, fn=None): return list(pool.map(fn or _run_one, tasks, chunksize=8))\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('helpers cell: _fit_fn line not found')

# ---- _sims: optional pv sigma channel on the SAME photons ----
done = False
for c in cells:
    s = txt(c)
    if "def _sims(pool, mu, sigma_prop, lam, gamma, n_runs, seed):\n" in s:
        s = s.replace(
            "def _sims(pool, mu, sigma_prop, lam, gamma, n_runs, seed):\n",
            "def _sims(pool, mu, sigma_prop, lam, gamma, n_runs, seed, sigma_est='lorentzian'):\n")
        s = s.replace(
            "    res = _parallel_map(pool, tasks)\n"
            "    return (torch.tensor([r[0] for r in res], dtype=torch.float32),\n"
            "            torch.tensor([r[1] for r in res], dtype=torch.float32),\n"
            "            torch.tensor(ns, dtype=torch.float32),\n"
            "            torch.tensor([r[2] for r in res], dtype=torch.float32),\n"
            "            torch.tensor([r[3] for r in res], dtype=torch.float32))\n",
            "    res = _parallel_map(pool, tasks)\n"
            "    ft = torch.tensor([r[0] for r in res], dtype=torch.float32)\n"
            "    st = torch.tensor([r[1] for r in res], dtype=torch.float32)\n"
            "    dgt = torch.tensor([r[2] for r in res], dtype=torch.float32)\n"
            "    dst = torch.tensor([r[3] for r in res], dtype=torch.float32)\n"
            "    if str(sigma_est) == 'pseudo_voigt':\n"
            "        # B5 (24o): take sigma_fit and dsigma/dgamma from the pseudo-Voigt fit on the SAME\n"
            "        # photons (estimator-matched); FWHM and dFWHM/dgamma stay with the Lorentzian fit.\n"
            "        rpv = _parallel_map(pool, tasks, fn=_run_one_pv)\n"
            "        st = torch.tensor([r[1] for r in rpv], dtype=torch.float32)\n"
            "        dst = torch.tensor([r[3] for r in rpv], dtype=torch.float32)\n"
            "    return ft, st, torch.tensor(ns, dtype=torch.float32), dgt, dst\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('_sims not found')

# ---- run_experiment: pass the sigma estimator to both _sims calls ----
done = 0
for c in cells:
    s = txt(c)
    a_old = "    _ft0, _si0, _nt0, _dg0, _ds0 = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED)\n"
    a_new = ("    _ft0, _si0, _nt0, _dg0, _ds0 = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED,\n"
             "                                    sigma_est=cfg.get('sigma_estimator', 'lorentzian'))\n")
    b_old = "        ft, si_t, nt, dg_t, ds_t = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED + step)\n"
    b_new = ("        ft, si_t, nt, dg_t, ds_t = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED + step,\n"
             "                                            sigma_est=cfg.get('sigma_estimator', 'lorentzian'))\n")
    if a_old in s:
        s = s.replace(a_old, a_new); done += 1
    if b_old in s:
        s = s.replace(b_old, b_new); done += 1
    if done:
        set_txt(c, s); break
if not done: sys.exit('run_experiment: _sims calls not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24o (Phase B5)

- **Change vs previous notebook (`24n`):** one knob — `sigma_estimator='pseudo_voigt'`. The **σ channel** is
  now produced by the **estimator-matched** pseudo-Voigt fit (`fit_profile(model='pseudo-voigt')`, `n_params=3`)
  applied to the **same photons** as the Lorentzian fit; the FWHM channel and `dFWHM/dγ` stay with the frozen
  Lorentzian fit. The B4 t-kernel is switched **off** (branch measured). Everything else is 24n verbatim.
  *(This is the cheap, channel-local version of the roadmap's D1: only the uncertainty is re-estimated.)*
- **Hypothesis (22g):** the σ mismatch is an **estimator** artefact, not missing physics: at truth, our
  Lorentzian analytic-CRLB σ is 1.6× too small (61.5 % median mismatch) while the pseudo-Voigt σ is only
  **28.1 %** off — and the pseudo-Voigt FWHM *spread* matches the real one (13.8 % vs 45.2 %). If the σ
  channel is fed a quantity that actually resembles the recorded one, the channel's μ pull should lose its
  systematic (−μ) component (FM#8) and become a *usable* signal instead of an attractor.
- **Falsifiable prediction:** the per-cell μ ratios move toward 1.0 (high-T 3nW T60/80/100 undershoot
  0.61/0.58/0.56 is the target) and `W_obj(all14)` ≤ 24g's 0.0747; γ ≈ unchanged (its chain is only
  re-weighted). **If falsified** — no systematic motion, or a *worse* μ — then "same estimator ⇒ same cloud"
  is false here too (22g's own headline) and the σ channel cannot be fixed without the forward model
  (D1/D2/D3).
- **Cost note:** two fits per run (Lorentzian + pseudo-Voigt) ⇒ ~2× runtime for this notebook only.
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
for c in cells:
    s = txt(c)
    for ln in s.splitlines():
        if 'sigma_est' in ln or '_run_one_pv' in ln or 'sigma_estimator' in ln:
            print('   ', ln.strip())
