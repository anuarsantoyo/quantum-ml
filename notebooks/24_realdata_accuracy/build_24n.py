#!/usr/bin/env python3
"""Build 24n from 24m — Phase B4: robust Student-t kernel on the sigma channel."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24n.ipynb')
if os.path.exists(p):
    sys.exit('24n.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24n', '24m',
                'robust Student-t kernel on the sigma channel (Phase B4)',
                'B4: replace the Gaussian sigma-channel factor in the 2-D KDE by a heavy-tailed '
                'Student-t kernel (nu = 3): W = exp(-0.5(d_f/h_f)^2) * (1 + sw*(d_s/h_s)^2/nu)^(-(nu+1)/2), '
                'with the matching exact robustness factor rho = ((nu+1)/nu)/(1 + sw*t/nu) multiplying the '
                'sigma part of the gamma-score chain; the B3 sw-anneal is switched OFF (branch measured)'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: revert the B3 anneal, add the t-kernel ----
done = False
for c in cells:
    s = txt(c)
    if "    sw_sched='anneal'," in s:
        s = s.replace(
            "    sw_sched='anneal',                           # B3: anneal the sigma-channel weight over the run\n"
            "    sw_hi=1.7297725439197594, sw_lo=0.43244313597993985,   # 2x -> 0.5x the frozen sigma_weight\n",
            "    sw_sched='none',                             # B3 anneal OFF (branch measured)\n"
            "    sigma_kernel='student_t', sigma_t_nu=3.0,    # B4: robust (heavy-tailed) sigma-channel kernel\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: sw_sched line not found')

# ---- _kde_scores: kernel switch + exact robustness factor on the gamma-score sigma term ----
done = False
for c in cells:
    s = txt(c)
    old = "    W = torch.exp(-0.5 * (d_f / h_f) ** 2 - 0.5 * sw * (d_s / h_s) ** 2)\n"
    if old in s:
        s = s.replace(old,
            "    # --- B4 (24n): sigma-channel kernel switch.  gauss (frozen) | student_t (heavy tailed).\n"
            "    #   log K_s = -((nu+1)/2) log(1 + sw * t / nu)   with t = (d_s/h_s)^2;  nu -> inf  =  gauss.\n"
            "    #   The exact dlogK_s/dgamma in the frozen z-form carries the factor rho (-> 1 as nu -> inf).\n"
            "    _t = (d_s / h_s) ** 2\n"
            "    if str(cfg.get('sigma_kernel', 'gauss')) == 'student_t':\n"
            "        _nu = float(cfg.get('sigma_t_nu', 3.0))\n"
            "        W = torch.exp(-0.5 * (d_f / h_f) ** 2) * (1.0 + sw * _t / _nu) ** (-0.5 * (_nu + 1.0))\n"
            "        _rho = ((_nu + 1.0) / _nu) / (1.0 + sw * _t / _nu)\n"
            "    else:\n"
            "        W = torch.exp(-0.5 * (d_f / h_f) ** 2 - 0.5 * sw * _t)\n"
            "        _rho = torch.ones_like(_t)\n")
        s = s.replace(
            "        dlogG = ((d_f * sim_df[None, :]) / h_f + sw * (d_s * sim_ds[None, :]) / h_s) / H_REF\n",
            "        dlogG = ((d_f * sim_df[None, :]) / h_f + sw * _rho * (d_s * sim_ds[None, :]) / h_s) / H_REF\n")
        s = s.replace(
            "        dlogG = (d_f * sim_df[None, :]) / h_f ** 2 + sw * (d_s * sim_ds[None, :]) / h_s ** 2\n",
            "        dlogG = (d_f * sim_df[None, :]) / h_f ** 2 + sw * _rho * (d_s * sim_ds[None, :]) / h_s ** 2\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('_kde_scores: W line not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24n (Phase B4)

- **Change vs previous notebook (`24m`):** one knob — `sigma_kernel='student_t'` (`sigma_t_nu = 3`) for the
  **σ channel only**. The KDE factor becomes `W = exp(-0.5(d_f/h_f)²)·(1 + sw·(d_s/h_s)²/ν)^{−(ν+1)/2}` and the
  σ part of the γ-score chain gets the matching exact factor `ρ = ((ν+1)/ν)/(1 + sw·t/ν)` (→ 1 as ν → ∞, so the
  Gaussian is the ν→∞ limit). The B3 sw-anneal is switched **off** (its branch has been measured); the B1/B2
  map stays off. Everything else is 24m verbatim.
- **Hypothesis (23d):** the σ channel's contribution is **outlier-dominated** — a handful of heavy-tailed
  `σ_fit` values (22g: real σ_fit median ≈ 1 MHz but std 4–10 MHz, a few near-degenerate fits) dominate the
  KDE responsibilities, and it is exactly those outliers that drag μ. A heavy-tailed kernel *down-weights*
  large `|d_s|` (relative to the Gaussian) instead of letting them dominate → the μ bias should shrink while
  the channel's *bulk* information is retained. Unlike B1 this changes the **kernel**, not the marginal shape,
  and unlike B3 it is unconditional (no schedule).
- **Falsifiable prediction:** the per-cell `μ_final/μ_true` table moves toward 1.0 (especially the high-T 3nW
  T60/80/100 undershoot: 0.61/0.58/0.56) with `W_obj(all14)` ≤ 24g's 0.0747; γ should be at worst mildly
  perturbed (the γ chain is re-weighted but not re-shaped). If `W_obj` is within ±0.005 of 24g **and** the
  ratios do not move, then outlier domination is not the operative mechanism at these noise levels.
- **If falsified:** the σ channel's *rank/kernel* form is not the lever → the channel's information is
  wrong at the estimator level (B5: pseudo-Voigt σ; B6: drop σ for μ, keep it for γ).
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
for c in cells:
    s = txt(c)
    for ln in s.splitlines():
        if 'sigma_kernel' in ln or '_rho' in ln:
            print('   ', ln.strip())
