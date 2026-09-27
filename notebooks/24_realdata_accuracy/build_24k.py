#!/usr/bin/env python3
"""Build 24k from 24g — Phase B1: sigma(mu) SHAPE MATCHING via a monotone quantile map."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24k.ipynb')
if os.path.exists(p):
    sys.exit('24k.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24k', '24g',
                'sigma shape matching: monotone quantile map sim->target (Phase B1)',
                'B1: fit a monotone (quantile) map f from the INIT reference sim sigma_fit cloud to the '
                'target sigma_fit cloud and apply it inside _kde_scores to sim_s and, by the chain rule, '
                'to sim_ds (piecewise-linear f => piecewise-constant f-prime), preserving the ranking '
                '(forked from 24g, the best standing Phase-A config)'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: add the B1 keys next to the A-keys ----
done = False
for c in cells:
    s = txt(c)
    if "mu_freeze=True, mu_freeze_frac=0.15," in s:
        s = s.replace(
            "    mu_freeze=True, mu_freeze_frac=0.15,         # A5: stop mu when |grad| < frac * |grad_init|\n",
            "    mu_freeze=True, mu_freeze_frac=0.15,         # A5: stop mu when |grad| < frac * |grad_init|\n"
            "    sigma_map='quantile', si_q=65,               # B1: monotone quantile map sim sigma_fit -> target\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: mu_freeze line not found')

# ---- 2. _kde_scores: apply the map to sim_s and (chain rule) sim_ds ----
done = False
for c in cells:
    s = txt(c)
    if "    sw = float(cfg.get('sigma_weight', 1.0))\n" in s and 'def _kde_scores' in s:
        s = s.replace(
            "    sw = float(cfg.get('sigma_weight', 1.0))\n",
            "    # --- B1 (24k): sigma-remap.  Apply the monotone map f to sim_s and, by the chain rule,\n"
            "    #     to sim_ds (piecewise-linear f => piecewise-constant f').  Ranking is preserved.\n"
            "    _sm = cfg.get('_sigma_map')\n"
            "    if _sm is not None:\n"
            "        _raw = sim_s.detach().cpu().numpy()\n"
            "        if _sm['mode'] == 'quantile':\n"
            "            _xs, _ys, _sl = _sm['xs'], _sm['ys'], _sm['sl']\n"
            "            _mapped = np.interp(_raw, _xs, _ys)\n"
            "            _ii = np.clip(np.searchsorted(_xs, _raw, side='right') - 1, 0, len(_sl) - 1)\n"
            "            _fac = _sl[_ii]\n"
            "        else:                                            # 'scale': f(x) = a*x (B2 control)\n"
            "            _mapped = _sm['a'] * _raw; _fac = np.full(_raw.shape, _sm['a'])\n"
            "        sim_s = torch.as_tensor(_mapped, dtype=sim_s.dtype)\n"
            "        sim_ds = torch.as_tensor(_fac, dtype=sim_ds.dtype) * sim_ds\n"
            "    sw = float(cfg.get('sigma_weight', 1.0))\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('_kde_scores: sw line not found')

# ---- 3. run_experiment: build the map once from the init reference cloud + target ----
done = False
for c in cells:
    s = txt(c)
    if "    _ft0, _si0, _nt0, _dg0, _ds0 = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED)\n" in s:
        s = s.replace(
            "    _ft0, _si0, _nt0, _dg0, _ds0 = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED)\n",
            "    _ft0, _si0, _nt0, _dg0, _ds0 = _sims(pool, mu_val, sigma_prop, lam, gamma_val, n_runs, SEED)\n"
            "    # --- B1 (24k): build the monotone sigma-remap ONCE from the INIT reference cloud + the\n"
            "    #   target cloud (truth-free).  'quantile': sorted quantile matching (shape); 'scale': a*x.\n"
            "    _smap = None\n"
            "    if str(cfg.get('sigma_map', 'none')) != 'none':\n"
            "        if str(cfg['sigma_map']) == 'quantile':\n"
            "            _q = np.linspace(0.0, 1.0, int(cfg.get('si_q', 65)))\n"
            "            _xs = np.quantile(_si0.numpy().astype(np.float64), _q)\n"
            "            _ys = np.quantile(target_s.numpy().astype(np.float64), _q)\n"
            "            _dx = np.diff(_xs); _dx = np.where(_dx <= 0, 1e-9, _dx)\n"
            "            _smap = dict(mode='quantile', xs=_xs, ys=_ys, sl=np.diff(_ys) / _dx)\n"
            "        else:                                        # 'scale' (B2 control)\n"
            "            _smap = dict(mode='scale',\n"
            "                         a=float(np.median(target_s.numpy())) / max(float(np.median(_si0.numpy())), 1e-9))\n"
            "        cfg = dict(cfg); cfg['_sigma_map'] = _smap\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: _ft0 line not found')

# ---- 4. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24k (Phase B1)

- **Change vs previous notebook (`24g`):** one new knob — `sigma_map='quantile'`. A **monotone quantile map**
  `f: sim σ_fit → target σ_fit` is fitted **once, at the protocol init** (the reference sim cloud) against the
  **real target cloud** (both truth-free), as a piecewise-linear sorted-quantile (QQ) map on a 65-point
  quantile grid. `f` is applied inside `_kde_scores` to `sim_s` and — **by the chain rule**, since `f` is
  piecewise linear with piecewise-constant `f′` — to `sim_ds` (`sim_ds ← f′(sim_s_r)·sim_ds`). The map is
  strictly increasing, so the sim ranking is preserved. Everything else (two-phase A7 schedule, σ_prop²-scaled
  LR, freeze, γ channel, bandwidths, init, budget) is 24g verbatim.
  *(Fork note: B1 is forked from `24g`, the best standing Phase-A config; the Phase-A chain is exhausted.)*
- **Hypothesis (FM#8):** 19c showed σ **scale** calibration does not move the μ mode — the attractor is the
  **shape** of σ(μ). 24j measured the size of the effect: the σ-matching estimator lands at
  `μ̂/μ_true ≈ 1.02` (1nW T10) but **0.22** (3nW T100), i.e. the estimator's μ is pulled by the σ-channel
  mismatch with a factor that grows with T. If the sim σ *shape* is made to match the target σ shape, that
  pull should vanish and the μ landing should move back toward truth.
- **Falsifiable prediction:** the per-cell **`μ_final/μ_true` table moves toward 1.0 across the grid** — in
  particular the high-T 3nW cells (T60/80/100, base ratio 0.61/0.58/0.56) rise, and `W_obj(all14)` improves
  on 24g (0.0747) and on 24b (0.0730); the σ-mismatch diagnostic (sim σ median / target σ median) goes to
  ≈1 at the landing. γ should be roughly unchanged (the map is applied to the σ channel only).
- **If falsified:** the σ-channel *rank* information (not just its marginal shape) is the load-bearing part;
  the fix must be an estimator-level one (B5/D1: pseudo-Voigt) rather than a monotone remap. Record the
  per-cell ratios and move to B2 (the scale-only control) which isolates shape vs scale.
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
for c in cells:
    s = txt(c)
    if '_sigma_map' in s and 'def _kde_scores' in s:
        print('--- kde patch ok ---')
        break
