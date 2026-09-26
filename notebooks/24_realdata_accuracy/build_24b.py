#!/usr/bin/env python3
"""Build 24b from 24a — Phase A2: truth-free travel normalisation."""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, '24b.ipynb')):
    sys.exit('24b.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24b', '24a',
                'truth-free mu-travel calibration (Phase A2)',
                'A2: replace the fixed lr_mu anneal by a per-experiment LR calibrated at the init so '
                'the run covers exactly mu_init of travel (LR_e = (mu_init/n_iter)/|grad_mu|_init)'],
               check=True)

p = os.path.join(HERE, '24b.ipynb')
nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: neutralise the global lr_mu / anneal ----
for c in cells:
    s = txt(c)
    if 'lr_mu=0.1669' in s:
        s = s.replace(
            "    lr_mu=0.1669, mu_anneal=0.3455, lr_gamma=0.4716, gamma_anneal=0.4723,",
            "    lr_mu=None, mu_anneal=0.0,                 # A2: lr_mu REPLACED by the per-exp travel calibration\n"
            "    lr_gamma=0.4716, gamma_anneal=0.4723,      # gamma channel frozen")
        set_txt(c, s)

# ---- 2. run_experiment: travel calibration + mu step rule ----
for c in cells:
    s = txt(c)
    if "lr_mu, lr_gamma = cfg['lr_mu'], cfg['lr_gamma']" in s:
        s = s.replace(
            "    lr_mu, lr_gamma = cfg['lr_mu'], cfg['lr_gamma']\n",
            "    lr_gamma = cfg['lr_gamma']\n")
        s = s.replace(
            "    mu_val, gamma_val = float(mu_init), float(gamma_init)\n    history, diverged, diverged_at = [], False, None\n",
            "    mu_val, gamma_val = float(mu_init), float(gamma_init)\n\n"
            "    # --- A2 (24b): truth-free travel calibration -------------------------------------\n"
            "    #  The protocol starts at mu_init, so the distance to cover is exactly mu_init.  Set the\n"
            "    #  mu step = sign(grad) * (mu_init/n_iter): each cell travels exactly mu_init over the run.\n"
            "    #  |grad| is measured ONCE at the init (truth-free: it uses only mu_init and the reward).\n"
            "    _ft0, _si0, _nt0, _dg0, _ds0 = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED)\n"
            "    _r0 = torch.log(_kde_scores(_ft0, _si0, _nt0, _dg0, _ds0, target_f, target_s,\n"
            "                                H_F, H_S, mu_val, sigma_prop, cfg)[4].clamp_min(1e-30)).mean(dim=0)\n"
            "    _score0 = (_nt0 - mu_val) / sigma_prop ** 2\n"
            "    grad_mu_init = abs(float(-(_r0 - _r0.mean()) @ _score0))\n"
            "    lr_mu_eff = (mu_init / n_iter) / max(grad_mu_init, 1e-12)   # equivalent fixed LR (diagnostic)\n"
            "    mu_step = mu_init / n_iter                                 # A2 travel budget per step (total = mu_init)\n"
            "    # ---------------------------------------------------------------------------------\n"
            "    history, diverged, diverged_at = [], False, None\n")
        s = s.replace(
            "        mu_val -= lr_mu * (1.0 - mu_anneal * step / n_iter) * grad_mu\n",
            "        # A2: gradient step with the per-step travel CLIPPED to mu_init/n_iter (total <= mu_init)\n"
            "        mu_val -= float(np.clip(lr_mu_eff * grad_mu, -mu_step, mu_step))\n")
        s = s.replace(
            "    return dict(exp=exp['name'], power=exp['power'], mu_true=mu_true, gamma_true=gamma_true,\n",
            "    return dict(exp=exp['name'], power=exp['power'], mu_true=mu_true, gamma_true=gamma_true,\n"
            "                lr_mu_eff=lr_mu_eff, grad_mu_init=grad_mu_init, mu_step=mu_step,\n")
        set_txt(c, s)

# ---- 3. hypothesis markdown ----
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24b (Phase A2)

- **Change vs previous notebook (`24a`):** the fixed global `lr_mu` + its linear anneal is **replaced by a
  per-experiment μ step calibrated at the init** so the run travels exactly `μ_init`:
  `lr_mu_eff = (μ_init/n_iter)/|∇μ|_init` and the per-step travel is **clipped to ±μ_init/n_iter**, so each
  cell covers **at most** `μ_init` of travel over the run. γ, the loss and the bandwidths are frozen.
- **Hypothesis (from `23b/23c/23e`):** the high-T μ landing is **monotone in the step budget**, and the
  baseline's single global LR starves the high-`σ_prop` cells while over-driving the low-T cells (24a:
  3nW T100 ×0.56 vs 1nW T20 ×1.60). Tying each cell's travel to its own (known) `μ_init` should fix both
  regimes **without touching γ**.
- **Falsifiable prediction:** μ ratios move **toward 1** at both ends — the starved high-T cells rise
  (3nW T60/T80/T100 ×0.61/0.58/0.56 → ≈0.8–1.0) and the low-T overshoot falls (1nW T20 1.60 → <1.3);
  **`W_obj(T ≥ 40)` drops below 0.0675**; γ is **unchanged** (rel-RMSE ≈ 26 %); 0 divergences.
- **If falsified** (μ does not improve, or γ drifts): the step budget is not the binding constraint on this
  model → the residual is the σ-channel bias (FM#8) → go to Phase B (24k σ-shape matching).
""")
        break

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)

# sanity: show the patched step rule
for c in cells:
    s = txt(c)
    if 'lr_mu_eff' in s and 'run_experiment' in s:
        print('--- patched run_experiment (key lines) ---')
        for ln in s.splitlines():
            if any(k in ln for k in ('lr_mu_eff', 'grad_mu_init', 'mu_val -=', '_ft0', 'lr_gamma =')):
                print('   ', ln.strip())
        break
