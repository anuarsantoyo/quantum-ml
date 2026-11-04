#!/usr/bin/env python3
"""Build 24p from 24o — Phase B6: hybrid two-role loss (exp8 cvm_fwhm for mu + KDE for gamma)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24p.ipynb')
if os.path.exists(p):
    sys.exit('24p.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24p', '24o',
                'hybrid two-role loss: cvm_fwhm for mu + KDE for gamma (Phase B6)',
                "B6: the MU reward is exp8's cvm_fwhm per-run loss (Cramer-von Mises squared-quantile "
                "distance on FWHM only, family scale 0.028428) instead of the KDE log-likelihood, while the "
                "GAMMA score stays the KDE chain; the B5 pseudo-Voigt sigma estimator is switched OFF "
                "(branch measured); one loss-form change, two roles"],
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
        "    sigma_estimator='pseudo_voigt',              # B5: sigma channel from the pseudo-Voigt estimator\n",
        "    sigma_estimator='lorentzian',                # B5 OFF (branch measured)\n"
        "    mu_reward='cvm_fwhm', cvm_scale=0.028428,    # B6: exp8 cvm_fwhm for mu, KDE kept for gamma\n")
    if s2 != s:
        set_txt(c, s2); found = True; break
if not found: sys.exit('config: sigma_estimator line not found')

# ---- helpers: exp8 quantile loss (verbatim in spirit, torch) ----
done = False
for c in cells:
    s = txt(c)
    if "def _run_one_pv(args):\n" in s:
        s = s.replace(
            "def _run_one_pv(args):\n",
            "# --- B6 (24p): exp8's cvm_fwhm per-run loss (Cramer-von Mises squared-quantile, FWHM only) ---\n"
            "def _sorted_target(t): return torch.sort(t)[0]\n\n"
            "def _quantile_match(sim_vals, tgt_sorted):\n"
            "    B = sim_vals.shape[0]; M = tgt_sorted.shape[0]\n"
            "    sv, idx = torch.sort(sim_vals)\n"
            "    pos = ((torch.arange(B, dtype=torch.float32) + 0.5) / B) * (M - 1)\n"
            "    lo = torch.floor(pos).long().clamp(0, M - 1); hi = torch.clamp(lo + 1, max=M - 1)\n"
            "    frac = pos - lo.float()\n"
            "    matched = tgt_sorted[lo] * (1.0 - frac) + tgt_sorted[hi] * frac\n"
            "    return sv, matched, idx\n\n"
            "def _quantile_loss(sim_vals, sim_dvals, tgt_sorted, power=2):\n"
            "    B = sim_vals.shape[0]\n"
            "    sv, matched, idx = _quantile_match(sim_vals, tgt_sorted)\n"
            "    d = sv - matched\n"
            "    l_s = d ** 2 if power == 2 else d.abs()\n"
            "    g_s = (2.0 * d * sim_dvals[idx]) if power == 2 else (torch.sign(d) * sim_dvals[idx])\n"
            "    out_l = torch.empty(B, dtype=sv.dtype); out_l[idx] = l_s\n"
            "    out_g = torch.empty(B, dtype=sv.dtype); out_g[idx] = g_s\n"
            "    return out_l, out_g\n\n"
            "def _run_one_pv(args):\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('helpers cell: _run_one_pv not found')

# ---- run_experiment: the hybrid reward ----
done = False
for c in cells:
    s = txt(c)
    a = ("    _r0 = torch.log(_kde_scores(_ft0, _si0, _nt0, _dg0, _ds0, target_f, target_s,\n"
         "                                H_F, H_S, mu_val, sigma_prop, _cfg_sw(0))[4].clamp_min(1e-30)).mean(dim=0)\n")
    if a in s:
        s = s.replace(a,
            "    # --- B6 (24p): the MU reward is exp8's cvm_fwhm (squared-quantile, FWHM only) ---------\n"
            "    _tgt_f_sorted = _sorted_target(target_f)\n"
            "    _cvm_s = float(cfg.get('cvm_scale', 1.0))\n"
            "    _W0 = _kde_scores(_ft0, _si0, _nt0, _dg0, _ds0, target_f, target_s,\n"
            "                      H_F, H_S, mu_val, sigma_prop, _cfg_sw(0))[4]\n"
            "    if str(cfg.get('mu_reward', 'kde')) == 'cvm_fwhm':\n"
            "        _r0 = -_quantile_loss(_ft0, _dg0, _tgt_f_sorted, power=2)[0] * _cvm_s\n"
            "    else:\n"
            "        _r0 = torch.log(_W0.clamp_min(1e-30)).mean(dim=0)\n")
        s = s.replace(
            "        r = torch.log(W.clamp_min(1e-30)).mean(dim=0)                 # Anuar's reward\n",
            "        if str(cfg.get('mu_reward', 'kde')) == 'cvm_fwhm':              # B6: cvm_fwhm | KDE\n"
            "            r = -_quantile_loss(ft, dg_t, _tgt_f_sorted, power=2)[0] * _cvm_s\n"
            "        else:\n"
            "            r = torch.log(W.clamp_min(1e-30)).mean(dim=0)             # Anuar's reward\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: _r0 line not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24p (Phase B6)

- **Change vs previous notebook (`24o`):** the **reward** driving the μ REINFORCE step changes family:
  `mu_reward='cvm_fwhm'` — exp8's **Cramér-von Mises squared-quantile distance on FWHM only**, per run
  (`_quantile_loss(f, dFWHM/dγ, sorted target FWHM, power=2)`, family scale 0.028428). The **γ score and the
  μ-score normalisation are the frozen KDE chain** (`_kde_scores`), so this is a genuine **two-role** loss: μ
  is scored by a **discrepancy** statistic (no bandwidth, no σ dependence), γ by the KDE likelihood. The B5
  pseudo-Voigt σ estimator is switched **off** (branch measured). Everything else is 24o verbatim.
- **Hypothesis (exp8):** exp8 found `cvm_fwhm` wins ~10 % on the point estimate of μ, and its advantage is
  structural — a quantile matching has **no bandwidth and no σ channel**, so it cannot suffer FM#8: the μ
  signal comes from the FWHM *distribution* shape via a comparison against the *real* quantiles (the real
  cloud enters the loss directly, not through a KDE bandwidth). The price is that CvM is a **discrepancy**,
  not a likelihood, so it carries no usable uncertainty/γ. Keeping the KDE for γ should combine the two.
- **Falsifiable prediction:** the per-cell μ ratios move toward 1.0 (target: high-T 3nW T60/80/100
  0.61/0.58/0.56) with `W_obj(all14)` ≤ 24g's 0.0747, **and** γ stays within ~0.01 W_obj of 24g's γ
  (frozen KDE chain untouched). **Falsified if** the FWHM-quantile μ signal is too weak on the real clouds
  (22d's reading: the FWHM marginal alone does not identify μ) — then μ stays near its init and W_obj is
  dominated by the γ channel, i.e. the σ channel really is load-bearing for μ even though it is biased.
- **If falsified:** Phase B's conclusion is that μ cannot be moved without a (corrected) σ channel → escalate
  to the forward-model phase (D: match the estimator end-to-end).
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
for c in cells:
    s = txt(c)
    for ln in s.splitlines():
        if '_mu_reward' in ln or 'mu_reward' in ln or '_quantile_loss' in ln:
            print('   ', ln.strip())
