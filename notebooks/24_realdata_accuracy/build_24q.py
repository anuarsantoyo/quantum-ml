#!/usr/bin/env python3
"""Build 24q from 24p — Phase B7: per-scan gamma/FWHM heterogeneity (20e HET_FRAC) on the A2 travel."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24q.ipynb')
if os.path.exists(p):
    sys.exit('24q.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24q', '24p',
                'per-scan linewidth heterogeneity on the A2 travel (Phase B7)',
                "B7: re-test 20e's per-scan FWHM heterogeneity (HET_FRAC) on top of the A2 travel-normalised "
                "optimiser: add excess FWHM scatter N(0, (HET_FRAC*IQR(target_f)/1.349)^2) to every simulated "
                "cloud (gamma-independent, so the implicit derivatives are untouched); the B6 cvm reward is "
                "switched OFF (branch measured) so the arm is judged on the frozen KDE"],
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
        "    mu_reward='cvm_fwhm', cvm_scale=0.028428,    # B6: exp8 cvm_fwhm for mu, KDE kept for gamma\n",
        "    mu_reward='kde',                             # B6 OFF (branch measured)\n"
        "    het_frac=1.0,                                # B7: per-scan FWHM heterogeneity (20e HET_FRAC)\n")
    if s2 != s:
        set_txt(c, s2); found = True; break
if not found: sys.exit('config: mu_reward line not found')

# ---- run_experiment: jitter_std + the two cloud injections ----
done = False
for c in cells:
    s = txt(c)
    a = "    H_F, H_S = _bandwidths(target_f, target_s, cfg['h_f_scale'], cfg['h_s_scale'], cfg['h_s_min'])\n"
    if a in s:
        s = s.replace(a, a +
            "    # --- B7 (24q): per-scan linewidth heterogeneity (20e HET_FRAC) ----------------------\n"
            "    #   excess FWHM scatter calibrated from the REAL robust FWHM scale (truth-free); it is\n"
            "    #   independent of gamma, so dFWHM/dgamma and dsigma/dgamma are untouched.\n"
            "    _jit = float(cfg.get('het_frac', 0.0)) * float(\n"
            "        torch.quantile(target_f, 0.75) - torch.quantile(target_f, 0.25)) / 1.349\n")
        # init cloud
        s = s.replace(
            "    _ft0, _si0, _nt0, _dg0, _ds0 = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED",
            "    _ft0, _si0, _nt0, _dg0, _ds0 = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED")
        lines = s.split('\n')
        out = []
        for ln in lines:
            out.append(ln)
            if ln.startswith('    _ft0, _si0, _nt0, _dg0, _ds0 = _sims('):
                out.append("    if _jit > 0.0:                                                       # B7")
                out.append("        _ft0 = _ft0 + _jit * torch.tensor(np.random.default_rng(SEED + 991).standard_normal(n_runs), dtype=torch.float32)")
            elif ln.startswith('        ft, si_t, nt, dg_t, ds_t = _sims('):
                out.append("        if _jit > 0.0:                                                   # B7")
                out.append("            ft = ft + _jit * torch.tensor(np.random.default_rng(SEED + 991 + step).standard_normal(n_runs), dtype=torch.float32)")
        s = '\n'.join(out)
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: _bandwidths line not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24q (Phase B7)

- **Change vs previous notebook (`24p`):** one knob — `het_frac = 1.0`, 20e's **per-scan linewidth
  heterogeneity**: every simulated FWHM is augmented with excess scatter `N(0, (HET_FRAC·IQR(target_f)/1.349)²)`
  (the real FWHM *robust* scale, truth-free). The jitter is independent of γ, so `dFWHM/dγ` and `dσ/dγ` (and
  the implicit-differentiation machinery) are untouched — a pure **simulator** change. The B6 cvm reward is
  switched **off** (branch measured).
- **Hypothesis (FM#6 + 20e):** the real FWHM clouds are **heterogeneity-broadened** (robust σ ≈ 6–11 MHz,
  different NV sites/strain), not photon-noise-broadened (sim ≈ 1 MHz). 20e found this collapses the SCALE
  mismatch between the sim and real FWHM clouds and turned the γ channel into a robust *location* match,
  fixing the low-T γ deficit (FM#6) — but it was tested on the pre-travel (20-series) optimiser whose μ step
  was a raw LR. Here it is re-tested on the A2 travel-normalised optimiser, where the μ dynamics are
  *different* (each cell covers exactly μ_init over the run).
- **Falsifiable prediction:** the **1nW low-T γ** deficit improves (base γ/true at 1nW T05/T10/T20:
  0.75/0.41/0.64) so `W_obj(T≤20)` improves on 24g's 0.1614, and γ rel-RMSE improves on 24g's 32.5 %; the μ
  landing is **not** systematically worse (the FWHM marginal changes shape, so some μ motion is allowed).
  **Falsified if** the low-T γ deficit is unchanged (⇒ the deficit is a line-shape/FM#8 issue, not a
  cloud-width issue) or the μ channel collapses.
- **If falsified:** FM#6 survives model broadening on the travel-normalised optimiser ⇒ the line shape itself
  (Lorentzian vs Voigt) is the residual, pointing at the forward-model phase.
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
for c in cells:
    s = txt(c)
    for ln in s.splitlines():
        if '_jit' in ln or 'het_frac' in ln:
            print('   ', ln.strip())
