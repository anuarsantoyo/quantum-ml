#!/usr/bin/env python3
"""Build 24j from 24i — Phase A10: truth-free data-derived mu init (bisection on the model FWHM scale)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24j.ipynb')
if os.path.exists(p):
    sys.exit('24j.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24j', '24i',
                'truth-free mu init from the data (Phase A10)',
                'A10: replace the protocol init mu_init = 0.5*mu_true by a data-derived estimate - bisect '
                'the simulator so that its median simulated FWHM matches the median target FWHM (truth-free), '
                'then run the SAME 0.5x-init + travel-budget protocol on mu_hat instead of mu_true'],
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
            "    gamma_coupled=True, gamma_rho_cap=3.0,       # A9: dgamma = rho*dmu (valley direction)\n",
            "    gamma_coupled=True, gamma_rho_cap=3.0,       # A9: dgamma = rho*dmu (valley direction)\n"
            "    mu_init_from_data=True, mu_hat_tol=0.02,     # A10: mu_init = 0.5*mu_hat (data), not 0.5*mu_true\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: gamma_coupled line not found')

# ---- 2. helper: the data-derived mu estimate (append to the helpers cell) ----
done = False
for c in cells:
    s = txt(c)
    if 'def _sims(pool, mu, sigma_prop, lam, gamma, n_runs, seed):' in s:
        s = s.rstrip('\n') + '''

def _mu_hat_from_target(pool, sigma_prop, lam, gamma, tgt_f, cfg, n_runs, tol=0.02, hi_max=512.0):
    """A10 (24j): TRUTH-FREE mu estimate -- the model's FWHM is monotone in mu, so bisect mu until the
    simulated MEDIAN FWHM matches the target's MEDIAN FWHM.  Uses only the target cloud and the simulator."""
    tgt_med = float(np.median(tgt_f.numpy()))
    def med_sim(mu_):
        ft, _, _, _, _ = _sims(pool, mu_, sigma_prop, lam, gamma, n_runs, SEED + 500)
        return float(np.median(ft.numpy()))
    lo, hi = 1e-3, 1.0
    while med_sim(hi) < tgt_med and hi < hi_max:
        lo, hi = hi, hi * 2.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if med_sim(mid) < tgt_med:
            lo = mid
        else:
            hi = mid
        if (hi - lo) / max(hi, 1e-12) < tol:
            break
    return 0.5 * (lo + hi)
'''
        set_txt(c, s); done = True
        break
if not done: sys.exit('helpers cell not found')

# ---- 3. run_experiment: use mu_hat for the init ----
done = False
for c in cells:
    s = txt(c)
    if "    mu_init, gamma_init = 0.5 * mu_true, 0.5 * gamma_true\n" in s:
        s = s.replace(
            "    mu_init, gamma_init = 0.5 * mu_true, 0.5 * gamma_true\n",
            "    # --- A10 (24j): the protocol init uses a DATA-DERIVED mu_hat, not mu_true ---------------\n"
            "    mu_hat = (_mu_hat_from_target(pool, sigma_prop, lam, gamma_true, target_f, cfg, n_runs,\n"
            "                                  cfg.get('mu_hat_tol', 0.02))\n"
            "              if cfg.get('mu_init_from_data', False) else mu_true)\n"
            "    mu_init, gamma_init = 0.5 * mu_hat, 0.5 * gamma_true\n")
        s = s.replace(
            "                valley_rho=rho,\n",
            "                valley_rho=rho, mu_hat=mu_hat, mu_hat_ratio=mu_hat / mu_true,\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: mu_init line not found')

# ---- 4. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24j (Phase A10)

- **Change vs previous notebook (`24i`):** the protocol's init no longer needs the answer. `μ_init` becomes
  `0.5·μ̂` where **`μ̂` is estimated from the data**: the model's FWHM is monotone in μ, so `μ̂` is found by
  **bisecting the simulator** until its simulated **median FWHM** matches the target's median FWHM (only the
  target cloud + the simulator enter — no `mu_true`). Everything else — the `0.5×` init rule, the two-phase
  travel budget, the σ_prop²-scaled LR, the freeze, the coupled γ (A9) and the loss — is frozen.
  *(γ_init still uses `0.5·γ_true`: this notebook's one change is the μ init.)*
- **Hypothesis:** the `0.5×truth` init is a protocol *cheat*; a data-derived scale should recover it. If the
  FWHM↔μ map is well-behaved, `μ̂ ≈ μ_true` and the whole Phase-A machinery (which is all normalised to
  `μ_init`) transfers unchanged → the landing should be **near-identical**.
- **Falsifiable prediction:** `μ̂/μ_true` within ≈ ±20 % per cell; `W_obj(all14)` within ±0.005 of 24i's and
  the per-cell μ ratios within ±0.05 of 24i's; 0 divergences. **Falsified if** the estimate is badly biased
  (then the protocol genuinely needs the truth / the estimator must be improved).
- **If falsified:** report the `μ̂/μ_true` bias per cell as a new diagnostic (an estimator bias, not an
  optimiser failure) and keep the `0.5×truth` init in the protocol until the estimator improves.
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

# ---- 5. verdict placeholder ----
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
    if 'mu_hat_from_target' in s and 'run_experiment' in s:
        print('--- patched lines ---')
        for ln in s.splitlines():
            if any(k in ln for k in ('mu_hat', 'mu_init, gamma_init')):
                print('   ', ln.strip())
        break
