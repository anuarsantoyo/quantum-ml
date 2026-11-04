#!/usr/bin/env python3
"""Build 24i from 24g — Phase A9: coupled mu/gamma travel along the (init) valley direction.

Fork note: A8 (24h) was a negative branch (gamma travel normalisation broke the good high-T gamma),
so the main chain re-forks 24g; A9 *replaces* the gamma step rule, nothing of A8 is carried.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24i.ipynb')
if os.path.exists(p):
    sys.exit('24i.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24i', '24g',
                'coupled mu/gamma travel along the valley (Phase A9)',
                'A9: slave gamma to mu - fix dgamma/dmu to the valley direction rho taken once from the local '
                'Hessian of the reward at the init, so the path follows the (mu,gamma) valley instead of '
                'running the two channels independently (capped by the existing gamma trust region)'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: gamma LR -> coupling ----
done = False
for c in cells:
    s = txt(c)
    if "mu_sched_shape='two_phase'" in s:
        s = s.replace(
            "    lr_gamma=0.4716, gamma_anneal=0.4723,      # gamma channel frozen\n",
            "    lr_gamma=None, gamma_anneal=0.0,            # A9: the annealed gamma LR is replaced by the coupling\n")
        s = s.replace(
            "    mu_freeze=True, mu_freeze_frac=0.15,         # A5: stop mu when |grad| < frac * |grad_init|\n",
            "    mu_freeze=True, mu_freeze_frac=0.15,         # A5: stop mu when |grad| < frac * |grad_init|\n"
            "    gamma_coupled=True, gamma_rho_cap=3.0,       # A9: dgamma = rho*dmu (valley direction)\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: lr_gamma / mu_freeze line not found')

# ---- 2. run_experiment: valley ratio at the init + coupled step ----
done = False
for c in cells:
    s = txt(c)
    if "    mu_step = mu_init / n_iter                                 # A2 reference cap (diagnostic)\n" in s:
        s = s.replace(
            "    # ---------------------------------------------------------------------------------\n"
            "    history, diverged, diverged_at = [], False, None\n",
            "    # ---------------------------------------------------------------------------------\n"
            "    # --- A9 (24i): valley direction from the local Hessian of the reward at the init --------\n"
            "    #  Central differences of the scalar KDE NLL in (mu, gamma); the valley is the eigenvector\n"
            "    #  of the smallest |curvature|; rho = v_gamma/v_mu is FIXED for the run.  Truth-free: only\n"
            "    #  the init sim cloud and the target cloud enter.\n"
            "    _c = [0]\n"
            "    def _Rval(mu_, g_):\n"
            "        _c[0] += 1\n"
            "        _f, _s, _n, _dgf, _dgs = _sims(pool, mu_, sigma_prop, lam, g_, n_runs, SEED + 900 + _c[0])\n"
            "        return float(_kde_scores(_f, _s, _n, _dgf, _dgs, target_f, target_s,\n"
            "                                 H_F, H_S, mu_, sigma_prop, cfg)[2])\n"
            "    _e_mu, _e_g = 0.15 * mu_init, 0.15 * gamma_init\n"
            "    _R00 = _Rval(mu_val, gamma_val)\n"
            "    _Rpm = _Rval(mu_val + _e_mu, gamma_val); _Rmm = _Rval(mu_val - _e_mu, gamma_val)\n"
            "    _Rpg = _Rval(mu_val, gamma_val + _e_g); _Rmg = _Rval(mu_val, gamma_val - _e_g)\n"
            "    _Rc = _Rval(mu_val + _e_mu, gamma_val + _e_g)\n"
            "    _H = np.array([\n"
            "        [(_Rpm - 2 * _R00 + _Rmm) / _e_mu ** 2, (_Rc - _Rpm - _Rpg + _R00) / (_e_mu * _e_g)],\n"
            "        [(_Rc - _Rpm - _Rpg + _R00) / (_e_mu * _e_g), (_Rpg - 2 * _R00 + _Rmg) / _e_g ** 2]])\n"
            "    _w, _V = np.linalg.eigh(_H)\n"
            "    _v = _V[:, int(np.argmin(np.abs(_w)))]\n"
            "    rho = float(_v[1] / _v[0]) if abs(_v[0]) > 1e-12 else 0.0\n"
            "    _rho_lim = cfg['gamma_rho_cap'] * gamma_init / mu_init\n"
            "    rho = float(np.clip(rho, -_rho_lim, _rho_lim))\n"
            "    # -------------------------------------------------------------------------------------\n"
            "    history, diverged, diverged_at = [], False, None\n")
        s = s.replace(
            "        # A5 (24e): truth-free convergence freeze -- once |grad_mu| < frac*|grad_mu_init|,\n"
            "        # STOP updating mu (the cell has reached its stationary point; further steps only wander).\n"
            "        frozen = bool(cfg.get('mu_freeze', False)) and (abs(grad_mu) < cfg['mu_freeze_frac'] * grad_mu_init)\n"
            "        if not frozen:\n"
            "            mu_val -= float(np.clip(lr_mu_eff_e * grad_mu, -mu_step_k[step], mu_step_k[step]))\n",
            "        # A9 (24i): the SINGLE mu step, used by both channels (gamma follows rho*dmu)\n"
            "        _d_mu = float(np.clip(lr_mu_eff_e * grad_mu, -mu_step_k[step], mu_step_k[step]))\n"
            "        # A5 (24e): truth-free convergence freeze -- once |grad_mu| < frac*|grad_mu_init|,\n"
            "        # STOP updating mu (the cell has reached its stationary point; further steps only wander).\n"
            "        frozen = bool(cfg.get('mu_freeze', False)) and (abs(grad_mu) < cfg['mu_freeze_frac'] * grad_mu_init)\n"
            "        if not frozen:\n"
            "            mu_val -= _d_mu\n")
        s = s.replace(
            "        dgamma = lr_gamma * (1.0 - gamma_anneal * step / n_iter) * grad_gamma\n",
            "        # A9: gamma is SLAVED to the mu step along the fixed valley ratio rho\n"
            "        dgamma = rho * _d_mu if cfg.get('gamma_coupled', False) else \\\n"
            "            lr_gamma * (1.0 - gamma_anneal * step / n_iter) * grad_gamma\n")
        s = s.replace(
            "                lr_mu_eff=lr_mu_eff, lr_mu_eff_e=lr_mu_eff_e, lr_scale_g=_g_lr,\n",
            "                lr_mu_eff=lr_mu_eff, lr_mu_eff_e=lr_mu_eff_e, lr_scale_g=_g_lr,\n"
            "                valley_rho=rho,\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: A2 block not found')

# ---- 3. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24i (Phase A9)

- **Change vs previous notebook (`24g`):** γ is no longer driven by its own fixed annealed LR. The **ratio
  `Δγ/Δμ` is fixed to the valley direction `ρ`**, taken **once** from the local Hessian of the scalar KDE NLL
  at the init (central differences in `(μ, γ)`; `ρ = v_γ/v_μ` for the eigenvector of smallest `|curvature|`,
  clipped to `±3·γ_init/μ_init`). Each step `Δγ = ρ·Δμ`, with `Δμ` the A2/A3-clipped μ step, and the existing
  γ trust region (`gamma_rel_cap`) still bounding `Δγ`. Truth-free: only the init sim cloud + the target cloud
  enter. The μ schedule (two-phase, σ_prop² LR, freeze), the loss and the bandwidths are frozen.
  *(Fork note: `24h` = A8 γ-travel normalisation was a **negative branch** (it broke the good high-T γ), so the
  main chain re-forks `24g`; A9 replaces the γ step rule, so nothing of A8's failed rule is carried.)*
- **Hypothesis:** the two channels are **not independent** — the KDE reward has a narrow stiff direction and a
  shallow valley. Driving μ alone (A2–A7) lets the μ path cross the valley while γ rides the wrong side;
  fixing `Δγ/Δμ` to the valley tangent should keep the path **inside** the valley so both channels land closer
  to truth.
- **Falsifiable prediction:** μ ratios at least as good as 24g and γ improves, so `W_obj(all14)` **below 24g's
  0.0747** and ideally below **0.0730** (24b); 0 divergences. A `|ρ|` pinned at the cap is itself a diagnostic
  that the local valley is ill-conditioned.
- **If falsified** (no gain, or worse): the valley picture does not describe the real optimisation → the
  residual is the **σ-channel bias (FM#8)** and the fix is in the forward model → Phase B (`24k`).
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

# ---- 4. verdict placeholder ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Verdict'):
        set_txt(c, "## Verdict — *(fill in after the run)*\n")
        done = True
        break
if not done: sys.exit('verdict cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
for c in cells:
    s = txt(c)
    if 'valley_rho' in s and 'run_experiment' in s:
        print('--- patched lines ---')
        for ln in s.splitlines():
            if any(k in ln for k in ('rho', '_d_mu', 'mu_val -=', 'dgamma =', 'gamma_coupled')):
                print('   ', ln.strip())
        break
