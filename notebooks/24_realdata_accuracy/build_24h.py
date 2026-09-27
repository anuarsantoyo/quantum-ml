#!/usr/bin/env python3
"""Build 24h from 24g — Phase A8: gamma travel normalisation (the A2 idea applied to gamma)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24h.ipynb')
if os.path.exists(p):
    sys.exit('24h.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24h', '24g',
                'gamma travel normalisation (Phase A8)',
                'A8: replace the fixed annealed gamma LR by a per-experiment travel calibration so the run '
                'covers exactly gamma_init of gamma travel (LR = (gamma_init/n_iter)/|grad_gamma|_init, step '
                'clipped to +-gamma_init/n_iter) - the A2 idea on the gamma channel (FM#6)'],
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
            "    lr_gamma=0.4716, gamma_anneal=0.4723,      # gamma channel frozen\n",
            "    lr_gamma=None, gamma_anneal=0.0,            # A8: replaced by the gamma travel calibration\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: lr_gamma line not found')

# ---- 2. run_experiment: gamma travel calibration + step rule ----
done = False
for c in cells:
    s = txt(c)
    if "    lr_gamma = cfg['lr_gamma']\n" in s:
        s = s.replace(
            "    lr_gamma = cfg['lr_gamma']\n",
            "    # A8 (24h): gamma channel -- the annealed global LR is replaced by a per-cell travel budget\n")
        # add the init-gradient measurement next to the mu one
        s = s.replace(
            "    grad_mu_init = abs(float(-(_r0 - _r0.mean()) @ _score0))\n",
            "    grad_mu_init = abs(float(-(_r0 - _r0.mean()) @ _score0))\n"
            "    _sg0 = _kde_scores(_ft0, _si0, _nt0, _dg0, _ds0, target_f, target_s,\n"
            "                       H_F, H_S, mu_val, sigma_prop, cfg)[1]\n"
            "    grad_gamma_init = abs(float(-_sg0.mean()))\n"
            "    lr_gamma_eff = (gamma_init / n_iter) / max(grad_gamma_init, 1e-12)   # A8 equivalent LR\n"
            "    gamma_step = gamma_init / n_iter                                     # A8 travel/step\n")
        s = s.replace(
            "        dgamma = lr_gamma * (1.0 - gamma_anneal * step / n_iter) * grad_gamma\n",
            "        # A8: calibrated gamma step, clipped to +-gamma_init/n_iter (total travel <= gamma_init)\n"
            "        dgamma = float(np.clip(lr_gamma_eff * grad_gamma, -gamma_step, gamma_step))\n")
        s = s.replace(
            "                lr_mu_eff=lr_mu_eff, lr_mu_eff_e=lr_mu_eff_e, lr_scale_g=_g_lr,\n",
            "                lr_mu_eff=lr_mu_eff, lr_mu_eff_e=lr_mu_eff_e, lr_scale_g=_g_lr,\n"
            "                lr_gamma_eff=lr_gamma_eff, grad_gamma_init=grad_gamma_init, gamma_step=gamma_step,\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: lr_gamma line not found')

# ---- 3. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24h (Phase A8)

- **Change vs previous notebook (`24g`):** the γ channel no longer uses the fixed annealed LR
  (`lr_gamma = 0.4716`, `gamma_anneal = 0.4723`). It gets the **same treatment A2 gave μ**: a per-cell LR
  calibrated once at the init, `lr_gamma_eff = (γ_init/n_iter)/|∇γ|_init`, whose step is **clipped to
  ±γ_init/n_iter**, so each cell covers **at most γ_init = 0.5·γ_true** of γ travel. The μ schedule
  (two-phase shape, σ_prop²-scaled LR, freeze rule), the loss and the bandwidths are frozen.
- **Hypothesis (`FM#6`):** the γ residual is the **1nW low-T deficit** (γ/true 0.46–0.64 there vs 0.93–1.00 at
  high T). The frozen annealed LR happens to be well-tuned for the high-T cells (where γ already lands) but
  **under-travels the low-T cells**, whose `|∇γ|` is much smaller. Tying each cell's γ travel to its own
  (protocol-known) `γ_init` should lift the low-T γ toward truth **without moving the already-good high-T γ**.
- **Falsifiable prediction:** 1nW T05/T10/T20 γ/true rise from 0.64/0.46/0.44 toward ≈0.7–1.0; the high-T
  cells (already 0.93–1.00) stay within ±0.05; `W_obj(T ≤ 20)` drops below 24g's, and `W_obj(all14)` drops
  below **24b's 0.0730** (the best value so far in the series); μ essentially unchanged; 0 divergences.
- **If falsified** (γ does not rise, or the high-T γ breaks): the γ channel is limited by the *line-shape*
  mismatch (FM#6) rather than by travel → the fix must be in the forward model, not the schedule → Phase B/D.
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
    if 'lr_gamma_eff' in s and 'run_experiment' in s:
        print('--- patched lines ---')
        for ln in s.splitlines():
            if any(k in ln for k in ('lr_gamma', 'gamma_step', 'grad_gamma_init', 'dgamma =')):
                print('   ', ln.strip())
        break
