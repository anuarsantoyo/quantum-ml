#!/usr/bin/env python3
"""Build 24c from 24b — Phase A3: cosine travel schedule at FIXED total travel (mu_init)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24c.ipynb')
if os.path.exists(p):
    sys.exit('24c.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24c', '24b',
                'cosine travel schedule at fixed total travel (Phase A3)',
                'A3: replace the constant per-step mu travel cap by a cosine-shaped schedule with the '
                'SAME total travel (sum_k c_k = mu_init); only the distribution over the run changes'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: add the schedule-shape knob ----
done_cfg = False
for c in cells:
    s = txt(c)
    if s.startswith('# ============================================================\n# 24a CONFIG') or ('CFG = dict(' in s and 'sigma_weight=' in s):
        s = s.replace("CFG = dict(\n",
                      "CFG = dict(\n"
                      "    mu_sched_shape='cosine',                     # A3: cosine per-step travel c_k (sum = mu_init)\n",
                      1)
        set_txt(c, s); done_cfg = True
        break
if not done_cfg:
    sys.exit('config cell not found')

# ---- 2. run_experiment: cosine per-step caps ----
done_run = False
for c in cells:
    s = txt(c)
    if 'def run_experiment' in s and 'mu_step = mu_init / n_iter' in s:
        s = s.replace(
            "    mu_step = mu_init / n_iter                                 # A2 travel budget per step (total = mu_init)\n",
            "    # --- A3 (24c): cosine-shaped per-step travel, SAME total travel -------------------\n"
            "    #  A2's total budget (sum_k c_k = mu_init) is kept EXACTLY; only its DISTRIBUTION\n"
            "    #  over the run changes: c_k = (mu_init/n_iter)*shape_k, sum_k shape_k = n_iter.\n"
            "    _shape_mode = cfg.get('mu_sched_shape', 'constant')\n"
            "    _kk = np.arange(n_iter)\n"
            "    if _shape_mode == 'cosine':\n"
            "        _shape = 1.0 + np.cos(np.pi * _kk / max(n_iter - 1, 1))    # front-loaded, mean 1\n"
            "    else:\n"
            "        _shape = np.ones(n_iter)\n"
            "    _shape = _shape * n_iter / _shape.sum()                    # normalise: sum -> n_iter\n"
            "    mu_step_k = (mu_init / n_iter) * _shape                    # per-step travel CAP, sum_k = mu_init\n"
            "    mu_step = mu_init / n_iter                                 # A2 reference cap (diagnostic)\n")
        s = s.replace(
            "        # A2: gradient step with the per-step travel CLIPPED to mu_init/n_iter (total <= mu_init)\n"
            "        mu_val -= float(np.clip(lr_mu_eff * grad_mu, -mu_step, mu_step))\n",
            "        # A3: same clip rule, but the cap is the cosine-shaped mu_step_k[step] (total <= mu_init)\n"
            "        mu_val -= float(np.clip(lr_mu_eff * grad_mu, -mu_step_k[step], mu_step_k[step]))\n")
        s = s.replace("                lr_mu_eff=lr_mu_eff, grad_mu_init=grad_mu_init, mu_step=mu_step,\n",
                      "                lr_mu_eff=lr_mu_eff, grad_mu_init=grad_mu_init, mu_step=mu_step,\n"
                      "                mu_step_total=float(mu_step_k.sum()),\n")
        set_txt(c, s); done_run = True
        break
if not done_run:
    sys.exit('run_experiment cell not found')

# ---- 3. hypothesis markdown ----
done_h = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24c (Phase A3)

- **Change vs previous notebook (`24b`):** the per-step μ travel cap (constant `μ_init/n_iter`) is replaced
  by a **cosine-shaped schedule** `c_k = (μ_init/n_iter)·shape_k` with `shape_k = 1+cos(πk/(n_iter−1))`
  normalised so `Σ_k shape_k = n_iter`. **The total travel budget is unchanged** (`Σ_k c_k = μ_init`); only
  its distribution over the run changes (front-loaded: ≈2.2× the A2 cap at step 0, →0 at the last step).
  γ, the loss, the bandwidths and `lr_mu_eff` are frozen.
- **Hypothesis (`23e`):** the *cumulative travel* sets the landing, not the per-step shape. If so, re-shaping
  a **fixed** total budget should barely move the optimum.
- **Falsifiable prediction:** `W_obj(all14)` stays inside the chaos band of 24b (`0.0730 ± 0.005`) and the
  per-cell μ ratios match 24b to within ±0.05 (high-T 3nW T60/T80/T100 ≈ 0.66/0.83/0.69; low-T unchanged);
  γ unchanged (rel-RMSE ≈ 34 %); 0 divergences.
  **Falsified if** the front-loading systematically moves the landing (e.g. low-T overshoots again because the
  travel is spent early) or the cosine beats the constant by > 0.005.
- **If falsified:** step *shape* matters → the landing is set by the trajectory, not only the total budget →
  prioritise A4 (per-cell scale) / A5 (freeze) over shape engineering.
""")
        done_h = True
        break
if not done_h:
    sys.exit('hypothesis cell not found')

# ---- 4. verdict placeholder (24b's stale verdict must not propagate) ----
done_v = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Verdict'):
        set_txt(c, "## Verdict — *(fill in after the run)*\n")
        done_v = True
        break
if not done_v:
    sys.exit('verdict cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)

for c in cells:
    s = txt(c)
    if 'mu_step_k' in s and 'run_experiment' in s:
        print('--- patched run_experiment (key lines) ---')
        for ln in s.splitlines():
            if any(k in ln for k in ('mu_sched_shape', '_shape', 'mu_step_k', 'mu_step =', 'mu_val -=', 'mu_step_total')):
                print('   ', ln.strip())
        break
