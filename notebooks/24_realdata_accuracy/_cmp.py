#!/usr/bin/env python3
"""_cmp.py — print the frozen metrics + per-cell ratio table for a series-24 notebook vs references.

Usage: python3 _cmp.py NEW [REF1 REF2 ...]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def load(i):
    z = f'data/processed/{i}_history.json'
    p = z if os.path.isabs(z) else os.path.join(HERE, '..', '..', z)
    return json.load(open(p))

def metrics(d):
    o = d['objective']
    return (o['all14'], o['hiT'], o['loT'], o['mu_rel_rmse'], o['gamma_rel_rmse'], o['n_diverged'])

def rtab(d):
    return {r['exp']: (r['mu_final'] / r['mu_true'], r['gamma_final'] / r['gamma_true'],
                       float(np.median(r['history'][-1]['s'])) / float(np.median(r['target_s'])))
            for r in d['results']}

new = sys.argv[1]
refs = sys.argv[2:]
dn = load(new)
mn = metrics(dn)
print(f'{new:5s} W_obj {mn[0]:.4f} | T>=40 {mn[1]:.4f} | T<=20 {mn[2]:.4f} | mu {mn[3]:.2f}% | gam {mn[4]:.2f}% | div {mn[5]}')
ref_tabs = {}
for r in refs:
    dr = load(r); mr = metrics(dr)
    ref_tabs[r] = rtab(dr)
    print(f'{r:5s} W_obj {mr[0]:.4f} | T>=40 {mr[1]:.4f} | T<=20 {mr[2]:.4f} | mu {mr[3]:.2f}% | gam {mr[4]:.2f}% | div {mr[5]}')
tn = rtab(dn)
hdr = f"{'exp':13s}" + "".join(f"{'mu_'+k:>8s}" for k in [new] + refs) + "".join(f"{'g_'+k:>8s}" for k in [new] + refs) + f"{'sig_'+new:>9s}"
print(hdr)
for e in tn:
    line = f"{e:13s}"
    for k, t in [(new, tn)] + [(k, ref_tabs[k]) for k in refs]:
        line += f"{t[e][0]:8.3f}"
    for k, t in [(new, tn)] + [(k, ref_tabs[k]) for k in refs]:
        line += f"{t[e][1]:8.3f}"
    line += f"{tn[e][2]:9.2f}"
    print(line)
