#!/usr/bin/env python3
"""Build 24i from 24h — Phase A9: coupled mu/gamma travel along the (init) valley direction."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24i.ipynb')
if os.path.exists(p):
    sys.exit('24i.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24i', '24h',
                'coupled mu/gamma travel along the valley (Phase A9)',
                'A9: slaved gamma - fix dgamma/dmu to the valley direction rho taken once from the local '
                'Hessian of the reward at the init, so the path follows the (mu,gamma) valley instead of '
                'running the two channels independently (capped by the existing gamma trust region)'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config ----
done = False
for c in cells:
    s = txt(c)
    if "mu_sched_shape='two_phase'" in s:
        s = s.replace(
            "    mu_freeze=True, mu_freeze_frac=0.15,         # A5: stop mu when |grad| < frac * |grad_init|\n",
            "    mu_freeze=True, mu_freeze_frac=0.15,         # A5: stop mu when |grad| < frac * |grad_init|\n"
            "    gamma_coupled=True, gamma_rho_cap=3.0,       # A9: dgamma = rho*dmu (valley direction)\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: mu_freeze line not found')

# ---- 2. run_experiment: init valley ratio + coupled gamma step ----
done = False
for c in cells:
    s = txt(c)
    if "    gamma_step = gamma_init / n_iter" in s:
        s = s.replace(
            "    gamma_step = gamma_init / n_iter                                     # A8 travel/step\n",
            "    gamma_step = gamma_init / n_iter                                     # A8 travel/step\n"
            "    # --- A9 (24i): valley direction from the local Hessian of the reward at the init --------\n"
            "    #  Central differences of the scalar KDE NLL in (mu, gamma); the valley is the\n"
            "    #  eigenvector of the smallest |curvature|; rho = v_gamma/v_mu is FIXED for the run\n"
            "    #  (truth-free: only the init cloud and the target cloud enter).\n"
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
            "    # -------------------------------------------------------------------------------------\n")
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
            "        # A8: calibrated gamma step, clipped to +-gamma_init/n_iter (total travel <= gamma_init)\n"
            "        dgamma = float(np.clip(lr_gamma_eff * grad_gamma, -gamma_step, gamma_step))\n",
            "        # A9: gamma is SLAVED to the mu step along the fixed valley ratio rho\n"
            "        dgamma = rho * _d_mu if cfg.get('gamma_coupled', False) else \\\n"
            "            float(np.clip(lr_gamma_eff * grad_gamma, -gamma_step, gamma_step))\n")
        s = s.replace(
            "                lr_gamma_eff=lr_gamma_eff, grad_gamma_init=grad_gamma_init, gamma_step=gamma_step,\n",
            "                lr_gamma_eff=lr_gamma_eff, grad_gamma_init=grad_gamma_init, gamma_step=gamma_step,\n"
            "                valley_rho=rho,\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: gamma_step line not found')

# ---- 3. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24i (Phase A9)

- **Change vs previous notebook (`24h`):** γ is no longer driven by its own calibrated travel. The **ratio
  `Δγ/Δμ` is fixed to the valley direction `ρ`**, taken **once** from the local Hessian of the scalar KDE NLL
  at the init (central differences in `(μ, γ)`; `ρ = v_γ/v_μ` for the eigenvector of smallest `|curvature|`,
  clipped to `±3·γ_init/μ_init`). Each step `Δγ = ρ·Δμ`, with `Δμ` the usual A2/A3-clipped μ step, and the
  existing γ trust region (`gamma_rel_cap`) still bounding `Δγ`. Truth-free: only the init sim cloud + the
  target cloud enter. The μ schedule, σ_prop² LR, freeze rule, loss and bandwidths are frozen.
- **Hypothesis:** the two channels are **not independent** — the KDE reward has a narrow stiff direction and a
  shallow valley. Driving μ alone (A2–A7) lets the μ path cross the valley, and the γ channel then rides the
  wrong side; fixing `Δγ/Δμ` to the valley tangent should keep the path inside the valley so **both** channels
  land closer to truth.
- **Falsifiable prediction:** μ ratios stay at least as good as 24h and γ improves at low T; `W_obj(all14)`
  below 24h's and ideally below **0.0730** (24b); 0 divergences. (A large `|ρ|` at the cap is itself a
  diagnostic that the local valley is poorly conditioned.)
- **If falsified** (no gain, or the coupled path is worse): the valley picture does not describe the real
  optimisation → the residual is the **σ-channel bias (FM#8)** and the fix is in the forward model → Phase B.
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
