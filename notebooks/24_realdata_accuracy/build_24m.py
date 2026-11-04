#!/usr/bin/env python3
"""Build 24m from 24l — Phase B3: sigma-weight ANNEAL (sw high early -> low late)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24m.ipynb')
if os.path.exists(p):
    sys.exit('24m.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24m', '24l',
                'sigma-weight anneal: sw high early -> low late (Phase B3)',
                'B3: the sigma-channel weight sw is ANNEALED over the run -- sw_hi = 2x the frozen value '
                'early (mu fuel) down to sw_lo = 0.5x late (kill the sigma-induced bias), instead of the '
                'constant sw; the B1/B2 sigma-remap is switched OFF (its branch was measured) so the arm '
                'is judged against the 24g base'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: map off, sw anneal on ----
done = False
for c in cells:
    s = txt(c)
    if "sigma_map='scale',                           # B2: SCALE-ONLY control" in s:
        s = s.replace(
            "    sigma_map='scale',                           # B2: SCALE-ONLY control (f(x)=a*x), no shape change\n",
            "    sigma_map='none',                            # B1/B2 map OFF (branch measured; base = 24g/24k)\n"
            "    sw_sched='anneal',                           # B3: anneal the sigma-channel weight over the run\n"
            "    sw_hi=1.7297725439197594, sw_lo=0.43244313597993985,   # 2x -> 0.5x the frozen sigma_weight\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: sigma_map scale line not found')

# ---- run_experiment: per-step sw (the helper must be defined BEFORE the init reward uses it) ----
done = False
for c in cells:
    s = txt(c)
    old0 = ("    _r0 = torch.log(_kde_scores(_ft0, _si0, _nt0, _dg0, _ds0, target_f, target_s,\n"
            "                                H_F, H_S, mu_val, sigma_prop, cfg)[4].clamp_min(1e-30)).mean(dim=0)\n")
    if old0 in s:
        new0 = (
            "    # --- B3 (24m): sigma-channel weight anneal ----------------------------------------\n"
            "    #   sw_k = sw_hi early -> sw_lo late (linear).  The KDE (reward + gamma score) is rebuilt\n"
            "    #   with sw_k each step; the A2 travel calibration below is unchanged (truth-free).\n"
            "    def _cfg_sw(k):\n"
            "        if str(cfg.get('sw_sched', 'none')) != 'anneal':\n"
            "            return cfg\n"
            "        _sw = cfg['sw_hi'] + (cfg['sw_lo'] - cfg['sw_hi']) * (k / max(n_iter - 1, 1))\n"
            "        _c = dict(cfg); _c['sigma_weight'] = _sw; return _c\n"
            + old0.replace("sigma_prop, cfg)[4]", "sigma_prop, _cfg_sw(0))[4]"))
        s = s.replace(old0, new0)
        set_txt(c, s)
        done = True
        break
if not done: sys.exit('run_experiment: _r0 line not found')

done = False
for c in cells:
    s = txt(c)
    oldl = ("        _, s_gamma, nll_val, w, W = _kde_scores(ft, si_t, nt, dg_t, ds_t,\n"
            "                                                target_f, target_s, H_F, H_S, mu_val, sigma_prop, cfg)\n")
    if oldl in s:
        s = s.replace(oldl,
            "        _ck = _cfg_sw(step)\n"
            "        _, s_gamma, nll_val, w, W = _kde_scores(ft, si_t, nt, dg_t, ds_t,\n"
            "                                                target_f, target_s, H_F, H_S, mu_val, sigma_prop, _ck)\n")
        done = True
        break
if not done: sys.exit('run_experiment: loop kde call not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24m (Phase B3)

- **Change vs previous notebook (`24l`):** one knob — `sw_sched='anneal'` with `sw_hi = 2 × σ_weight`
  (1.7298) early and `sw_lo = 0.5 × σ_weight` (0.4324) late, linear over the 30 steps. The KDE (reward **and**
  γ-score) is rebuilt each step with the current `sw_k`. The B1/B2 σ-remap is switched **off**: its branch has
  been measured, so this arm is judged against the same base as B1 (`24g`/`24l`-minus-map).
  *(Fork note: `24l` is the scale-only control for B1; B3 needs the map off, hence the explicit revert line.)*
- **Hypothesis:** FM#8 says the σ channel is the only live μ gradient but it also carries the (−μ) attractor
  (its mismatch grows with T). A **schedule** can use the same estimator for both jobs: run the σ channel
  *sharply* while μ still has to travel (the "fuel" phase — the μ reward is the KDE log-likelihood, whose
  sensitivity is set by `sw`), then *damp* it as the landing is approached, so the terminal point is set by the
  FWHM channel (which carries the γ information) rather than by the mismatched σ channel. This is B1's idea
  implemented in the *weight* instead of in the *kernel* — no new estimator, no new map.
- **Falsifiable prediction:** `W_obj(all14)` ≤ 24g's 0.0747 with the **high-T μ ratios** (3nW T60/80/100:
  0.61/0.58/0.56; 1nW T60–100: 0.72/0.60/0.73) rising, and the low-T μ overshoot (1nW T20 0.66) not worsening;
  γ ≈ unchanged. If instead `W_obj` ≈ 24g within ±0.005 and no μ ratio moves systematically, the *annealing
  schedule* is not a lever either — the σ channel's information content, not its weight, is what is wrong.
- **If falsified:** the σ channel must be *replaced* (B4 robust kernel / B5 matched estimator / B6 hybrid
  reward), not re-weighted.
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
for c in cells:
    s = txt(c)
    for ln in s.splitlines():
        if '_cfg_sw' in ln or 'sw_sched' in ln or 'sw_hi' in ln:
            print('   ', ln.strip())
