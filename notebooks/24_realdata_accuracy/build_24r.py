#!/usr/bin/env python3
"""Build 24r from 24q — Phase B8: 1-D FWHM-only KDE with the count channel re-added as a prior."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24r.ipynb')
if os.path.exists(p):
    sys.exit('24r.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24r', '24q',
                'FWHM-only KDE + count channel as a prior (Phase B8)',
                'B8: drop the sigma_channel entirely (1-D FWHM-only KDE -- the sigma factor is removed from '
                'the reward AND from the gamma chain) and re-add the COUNT channel as a PRIOR -- a Gaussian '
                'prior on mu from the count statistic (mu ~ nbar, whose only protocol information is the init) '
                'whose gradient enters the mu update; HET_FRAC and the other B-branches are OFF'],
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
        "    het_frac=1.0,                                # B7: per-scan FWHM heterogeneity (20e HET_FRAC)\n",
        "    het_frac=0.0,                                # B7 OFF (branch measured)\n"
        "    sigma_channel='off', cnt_prior_w=0.25,       # B8: 1-D FWHM KDE + count channel as a prior\n")
    if s2 != s:
        set_txt(c, s2); found = True; break
if not found: sys.exit('config: het_frac line not found')

# ---- _kde_scores: the sigma channel switch ----
done = False
for c in cells:
    s = txt(c)
    a = "    _t = (d_s / h_s) ** 2\n    if str(cfg.get('sigma_kernel', 'gauss')) == 'student_t':\n"
    if a in s:
        s = s.replace(a,
            "    _t = (d_s / h_s) ** 2\n"
            "    if str(cfg.get('sigma_channel', 'on')) == 'off':\n"
            "        # --- B8 (24r): 1-D FWHM-only KDE.  The sigma channel is removed from the reward AND\n"
            "        #     from the gamma chain (rho=0 kills the d_s*sim_ds term).  Truth-free.\n"
            "        W = torch.exp(-0.5 * (d_f / h_f) ** 2)\n"
            "        _rho = torch.zeros_like(_t)\n"
            "    elif str(cfg.get('sigma_kernel', 'gauss')) == 'student_t':\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('_kde_scores: _t line not found')

# ---- run_experiment: the count prior on the mu step ----
done = False
for c in cells:
    s = txt(c)
    a = ("        frozen = bool(cfg.get('mu_freeze', False)) and (abs(grad_mu) < cfg['mu_freeze_frac'] * grad_mu_init)\n"
         "        if not frozen:\n"
         "            mu_val -= float(np.clip(lr_mu_eff_e * grad_mu, -mu_step_k[step], mu_step_k[step]))\n")
    if a in s:
        s = s.replace(a,
            "        frozen = bool(cfg.get('mu_freeze', False)) and (abs(grad_mu) < cfg['mu_freeze_frac'] * grad_mu_init)\n"
            "        _dmu = lr_mu_eff_e * grad_mu\n"
            "        if str(cfg.get('sigma_channel', 'on')) == 'off':\n"
            "            # --- B8 (24r): the COUNT CHANNEL re-added as a PRIOR ----------------------------\n"
            "            #  mu is proportional to the photon count nbar, so the count channel is a statement\n"
            "            #  about mu; on real data it carries no measurable count (19b: dead), so its only\n"
            "            #  information is the protocol's own init.  A Gaussian prior N(mu_init, mu_init^2)\n"
            "            #  has gradient -(mu-mu_init)/mu_init^2; a REINFORCE reward cannot see a mu-only\n"
            "            #  constant, so the prior acts DIRECTLY on the update, at most cnt_prior_w * step.\n"
            "            _dmu = _dmu + float(cfg.get('cnt_prior_w', 0.25)) * _shape[step] * (\n"
            "                -(mu_val - mu_init) / max(mu_init, 1e-9))\n"
            "        if not frozen:\n"
            "            mu_val -= float(np.clip(_dmu, -mu_step_k[step], mu_step_k[step]))\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: frozen block not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24r (Phase B8)

- **Change vs previous notebook (`24q`):** two coordinated parts of ONE designed change:
  (i) `sigma_channel='off'` — the σ factor is removed from the KDE **reward** and the σ term is removed from
  the **γ chain** (`ρ = 0`), i.e. a true **1-D FWHM-only KDE**;
  (ii) `cnt_prior_w = 0.25` — the **count channel re-added as a prior**: `μ ∝ n̄` (photon count), so the count
  channel is a statement about μ; a **Gaussian prior `N(μ_init, μ_init²)`** (the protocol's only count
  information — on real data the count channel is *dead*, 19b) whose gradient enters the **μ update** directly
  (a REINFORCE reward is blind to a μ-only constant), contributing at most `0.25 ×` the nominal step.
  HET_FRAC and the other Phase-B branches are off.
- **Hypothesis / purpose:** separate *"σ is the culprit"* from *"the 2-D form is the culprit"*. 22d showed
  that dropping σ alone fails (μ runs away upward at low count in the old optimiser). Here the same drop is
  tested (a) on the travel-normalised optimiser (μ cannot run away: total travel = μ_init) and (b) with the
  count prior providing a weak restoring force toward the protocol's count belief.
- **Falsifiable prediction:** if **σ** was the culprit, this arm should be *competitive* with 24g
  (`W_obj(all14)` ≤ ~0.080) with μ ratios inside [0.5, 1.2]; if the **2-D form** (i.e. having a second,
  band-limited channel at all) was load-bearing, this arm is clearly **worse** (W_obj > 0.085, μ pinned near
  the prior/init at 0.5–0.7 and the low-T FWHM-only runaway suppressed only by the prior). γ should be
  *slightly better* than 24g if the γ chain's σ contamination is real.
- **If falsified (2-D is load-bearing):** Phase B's σ-channel conclusion is that the channel must be
  *corrected* rather than removed → escalate to the forward-model phase (D1/D2/D3).
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
for c in cells:
    s = txt(c)
    for ln in s.splitlines():
        if 'sigma_channel' in ln or '_dmu' in ln or 'cnt_prior_w' in ln:
            print('   ', ln.strip())
