#!/usr/bin/env python3
"""Build 24d from 24c — Phase A4: per-cell mu LR proportional to sigma_prop^2, with cap/floor."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24d.ipynb')
if os.path.exists(p):
    sys.exit('24d.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24d', '24c',
                'per-cell mu LR proportional to sigma_prop^2 (Phase A4)',
                'A4: multiply each cell mu LR by clip(sigma_prop^2 / median(sigma_prop^2), 1, 2) '
                '(23e finding: the right step is ~ sigma_prop^2), floored at 1x and capped at 2x so the '
                'well-calibrated cells are not over-driven; the travel caps stay at the A2/A3 budget'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: the LR-scale knobs ----
done = False
for c in cells:
    s = txt(c)
    if "mu_sched_shape='cosine'" in s:
        s = s.replace(
            "    mu_sched_shape='cosine',                     # A3: cosine per-step travel c_k (sum = mu_init)\n",
            "    mu_sched_shape='cosine',                     # A3: cosine per-step travel c_k (sum = mu_init)\n"
            "    mu_lr_sign2=True,                            # A4: per-cell LR x clip(sigma_prop^2/s2_med, floor, cap)\n"
            "    mu_lr_cap=2.0, mu_lr_floor=1.0,              # ... capped at 2x / floored at 1x the A2 LR\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: mu_sched_shape line not found')

# ---- 2. run_experiment: scale the mu LR ----
done = False
for c in cells:
    s = txt(c)
    if "    lr_mu_eff = (mu_init / n_iter) / max(grad_mu_init, 1e-12)" in s:
        s = s.replace(
            "    lr_mu_eff = (mu_init / n_iter) / max(grad_mu_init, 1e-12)   # equivalent fixed LR (diagnostic)\n",
            "    lr_mu_eff = (mu_init / n_iter) / max(grad_mu_init, 1e-12)   # equivalent fixed LR (diagnostic)\n"
            "    # --- A4 (24d): per-cell mu LR \\u221d sigma_prop^2 (23e), with cap/floor ----------------\n"
            "    #  23e: the correct mu step scales like sigma_prop^2.  Multiply the A2 LR by\n"
            "    #  clip(sigma_prop^2 / median_14(sigma_prop^2), floor, cap) -- truth-free (sigma_prop is a\n"
            "    #  data/protocol quantity).  floor 1.0 protects the already-calibrated (small-sigma_prop)\n"
            "    #  cells from being starved; cap 2.0 stops the largest-sigma_prop cells being over-driven.\n"
            "    if cfg.get('mu_lr_sign2', False):\n"
            "        _s2ref = float(np.median([e['sigma_prop'] ** 2 for e in EXPERIMENTS]))\n"
            "        _g_lr = float(sigma_prop ** 2 / _s2ref)\n"
            "        _g_lr = min(max(_g_lr, cfg['mu_lr_floor']), cfg['mu_lr_cap'])\n"
            "    else:\n"
            "        _g_lr = 1.0\n"
            "    lr_mu_eff_e = lr_mu_eff * _g_lr               # the LR the step rule actually uses\n")
        s = s.replace(
            "        mu_val -= float(np.clip(lr_mu_eff * grad_mu, -mu_step_k[step], mu_step_k[step]))\n",
            "        mu_val -= float(np.clip(lr_mu_eff_e * grad_mu, -mu_step_k[step], mu_step_k[step]))\n")
        s = s.replace(
            "                lr_mu_eff=lr_mu_eff, grad_mu_init=grad_mu_init, mu_step=mu_step,\n",
            "                lr_mu_eff=lr_mu_eff, lr_mu_eff_e=lr_mu_eff_e, lr_scale_g=_g_lr,\n"
            "                grad_mu_init=grad_mu_init, mu_step=mu_step,\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: lr_mu_eff line not found')

# ---- 2b. diagnostic cell: use the EFFECTIVE LR when present ----
done = False
for c in cells:
    s = txt(c)
    if "raw = [abs(r['lr_mu_eff'] * s['grad_mu']) for s in h]" in s:
        s = s.replace(
            "raw = [abs(r['lr_mu_eff'] * s['grad_mu']) for s in h]",
            "lre = r.get('lr_mu_eff_e', r['lr_mu_eff'])          # effective LR (A4-scaled when present)\n"
            "    raw = [abs(lre * s['grad_mu']) for s in h]")
        set_txt(c, s); done = True
        break
if not done: print('WARN: diagnostic cell not patched (ok if absent)')

# ---- 3. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24d (Phase A4)

- **Change vs previous notebook (`24c`):** the μ LR is no longer uniform across cells. Each cell's *effective*
  LR is `lr_mu_eff · clip(σ_prop²/median₁₄(σ_prop²), 1, 2)` — the `23e` finding that the correct μ step
  scales like `σ_prop²` — with a **floor of 1×** (already-calibrated, small-`σ_prop` cells are not starved)
  and a **cap of 2×** (the largest-`σ_prop` cells are not over-driven). The travel caps (`Σ_k c_k = μ_init`),
  the cosine shape, γ, loss and bandwidths are frozen.
- **Hypothesis (with the `24c` diagnostic FM#13):** the realised μ travel is only ≈0.42×μ_init because the
  **LR** — not the cap — is the binding constraint (the clip is idle on the median step). So the lever must be
  the **LR**: raising it for the high-`σ_prop` cells (which under-travel most) should let them reach the
  budget and land closer to truth, while the floor leaves the small-`σ_prop` cells exactly as in 24c.
- **Falsifiable prediction:** the high-`σ_prop` cells rise toward truth (3nW T40/T60/T80/T100 0.76/0.66/0.76/0.67
  → ≥0.85; 1nW T80/T100 up) and **`W_obj(T ≥ 40)` drops below 24c's 0.0505** (ideally below 24b's 0.0435);
  the small-`σ_prop` cells are **bit-identical to 24c** (they sit at the floor, LR unchanged); γ unchanged;
  0 divergences.
- **If falsified** (high-σ cells do not rise, or the low-σ cells move): the σ_prop² LR correction is not the
  lever → the residual is the σ-channel bias (**FM#8**) → Phase B (`24k` σ-shape matching).
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
    if 'lr_mu_eff_e' in s and 'run_experiment' in s:
        print('--- patched lines ---')
        for ln in s.splitlines():
            if any(k in ln for k in ('mu_lr_sign2', '_g_lr', 'lr_mu_eff_e', 'mu_val -=', 'lr_scale_g')):
                print('   ', ln.strip())
        break
