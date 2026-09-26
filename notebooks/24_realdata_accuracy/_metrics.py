#!/usr/bin/env python3
"""_metrics.py <id> [prev1 prev2] — print the frozen metric + per-cell ratios vs predecessors."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
def R(t): return json.load(open(os.path.join(ROOT, 'data', 'processed', f'{t}_history.json')))

ids = sys.argv[1:]
runs = {t: R(t) for t in ids}
for t in ids:
    o = runs[t]['objective']
    print(f"{t}: all14 {o['all14']:.4f} | T>=40 {o['hiT']:.4f} | T<=20 {o['loT']:.4f} | "
          f"mu {o['mu_rel_rmse']:.1f}% | gamma {o['gamma_rel_rmse']:.1f}% | div {o['n_diverged']}")
base = runs[ids[0]]['results']
hdr = ''.join(f"{t:>9s}" for t in ids)
print(f"\n{'exp':14s}{hdr}   {'gam ' + hdr}")
for i, rb in enumerate(base):
    mu = ''.join(f"{runs[t]['results'][i]['mu_final']/runs[t]['results'][i]['mu_true']:9.3f}" for t in ids)
    ga = ''.join(f"{runs[t]['results'][i]['gamma_final']/runs[t]['results'][i]['gamma_true']:9.3f}" for t in ids)
    print(f"{rb['exp']:14s}{mu}   {ga}")
