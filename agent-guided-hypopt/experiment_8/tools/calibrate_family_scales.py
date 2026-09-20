#!/usr/bin/env python3
"""Calibrate the per-family normalization constants for experiment_8.

Each loss family lives on a different scale, so under a FROZEN schedule the campaign
would otherwise measure "which loss has the biggest gradient", not "which loss FORM is
best". Two constants per family are calibrated ONCE, at the shared frozen start
(0.5 x truth, DEFAULT_CONFIG), and hard-coded into ag_hypopt.py:

  FAMILY_SCALE[family]     -> applied to BOTH the per-run reward r and the pathwise g,
                              so that |d(loss)/dgamma| matches the KDE control's.

The reward r is additionally rescaled PER STEP to the control's reward spread inside
_alt_scores' caller (ag_hypopt.py), so no per-family mu constant is needed here.
The KDE control is 1.0 by construction. Run from this experiment folder:

    python3 tools/calibrate_family_scales.py

Deterministic (SEED=42, same start/draws as a real trial). Not used at campaign time.
"""
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

import ag_hypopt as A

EXPS = ['1nW Trans05', '1nW Trans100', '3nW Trans60']
FAMILIES = ['kde', 'w1_2d', 'energy_2d', 'mmd_2d', 'w1_fwhm', 'cvm_fwhm']
SW = 0.4092   # a fixed moderate sigma-weight so the contrast is the FAMILY, not sw


def step0(family, exp, sw):
    """One synthetic step 0: return (|d(loss)/dgamma|, std(reward)) at the frozen start."""
    pool = ProcessPoolExecutor(max_workers=4, initializer=A._init_worker)
    cfg = dict(A.DEFAULT_CONFIG, loss_family=family, sigma_weight=sw)
    mu = 0.5 * exp['mu_true']; gam = 0.5 * exp['gamma_true']
    sp = exp['sigma_prop']; lam = exp['lam']
    rng = np.random.default_rng(A.SEED + 0)
    tasks, ns = [], []
    for _ in range(cfg['n_runs']):
        u, b, n = A.draw_fixed_noise(mu, sp, lam, rng)
        tasks.append((gam, u.numpy(), b.numpy())); ns.append(n)
    res = A._parallel_map(pool, tasks)
    ft = torch.tensor([r[0] for r in res], dtype=torch.float32)
    si = torch.tensor([r[1] for r in res], dtype=torch.float32)
    nt = torch.tensor(ns, dtype=torch.float32)
    dg = torch.tensor([r[2] for r in res], dtype=torch.float32)
    ds = torch.tensor([r[3] for r in res], dtype=torch.float32)
    tf, ts = A._load_real_target(exp)
    HF, HS = A._bandwidths('target', tf, ts, ft, si, cfg['h_f_scale'], cfg['h_s_scale'],
                           cfg['h_s_min'], exp['power'])
    if family == 'kde':
        s_mu, s_gamma, nll, w, W = A._kde_scores(ft, si, nt, dg, ds, tf, ts, HF, HS, mu, sp, cfg)
        r = torch.log(W.clamp_min(1e-30)).mean(dim=0)
        grad = abs(float(s_gamma.mean()))
    else:
        r, g, nll = A._alt_scores(family, ft, si, nt, dg, ds, tf, ts, HF, HS, sw, cfg)
        grad = abs(float(g.mean()))
    pool.shutdown()
    return grad, float(r.std())


def main():
    exps = [e for e in A.EXPERIMENTS if e['name'] in EXPS]
    grad = {}
    for f in FAMILIES:
        cur = float(A.FAMILY_SCALE.get(f, 1.0)) or 1.0
        gs = []
        for e in exps:
            g, _r = step0(f, e, SW)
            gs.append(g / cur)          # undo the in-file scale -> RAW |d(loss)/dgamma|
        grad[f] = gs
        print(f"{f:10} raw|g| " + " ".join(f"{v:8.4f}" for v in gs))
    g_ref = statistics.median(grad['kde'])
    print(f"\nkde ref |g| = {g_ref:.4f}\n")
    scale = {'kde': 1.0}
    for f in FAMILIES[1:]:
        gm = statistics.median(grad[f])
        scale[f] = round(g_ref / gm, 6) if gm > 0 else 1.0
    print("FAMILY_SCALE = {")
    for f in FAMILIES:
        print(f"    '{f}': {scale[f]},")
    print("}")


if __name__ == '__main__':
    main()
