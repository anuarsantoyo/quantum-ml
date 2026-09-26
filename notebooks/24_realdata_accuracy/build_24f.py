#!/usr/bin/env python3
"""Build 24f from 24e — Phase A6: n_iter 30 -> 60 at FIXED total travel (negative control)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24f.ipynb')
if os.path.exists(p):
    sys.exit('24f.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24f', '24e',
                'finer integration: n_iter 30 -> 60 at fixed total travel (Phase A6)',
                'A6: double n_iter (30 -> 60) at FIXED total travel (each per-step cap halves, sum_k c_k '
                'unchanged) - a negative control for the travel-budget hypothesis'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: n_iter 30 -> 60 (the ONE change) ----
done = False
for c in cells:
    s = txt(c)
    if 'n_runs=100, n_iter=30,' in s:
        s = s.replace(
            "    n_runs=100, n_iter=30,                       # series-21/exp5/6 protocol\n",
            "    n_runs=100, n_iter=60,                       # A6: 30 -> 60 (finer integration; total travel FIXED)\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: n_iter line not found')

# ---- 2. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24f (Phase A6)

- **Change vs previous notebook (`24e`):** **`n_iter` 30 → 60** (the single named knob). Because the μ
  schedule is *normalised* (`Σ_k c_k = μ_init · g`), doubling `n_iter` **halves each per-step cap** and
  leaves the **total travel budget unchanged**; each step is also re-noised with `SEED+step` (the first 30
  seeds are unchanged). γ, the loss, the bandwidths, the σ_prop² scale and the freeze rule are frozen.
  *(This is the one notebook deliberately touching the protocol's `N_ITER = 30`, exactly as roadmap A6
  specifies — it is the negative control.)*
- **Hypothesis:** if the landing is governed by the **total travel budget** (A2/A3) and not by step size,
  finer integration should **not move the optimum**. A change ⇒ the discretisation / noise-per-step matters.
- **Falsifiable prediction:** `W_obj(all14)` within the chaos band of 24e (±0.005), per-cell μ ratios match
  24e to ±0.05, γ unchanged, 0 divergences. **Falsified if** the finer integration systematically shifts the
  landing (⇒ step-count / noise schedule is itself a lever).
- **If falsified:** add a step-count sweep to the roadmap (a genuinely new Phase-A lever).
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

# ---- 3. verdict placeholder ----
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
    if 'n_iter=60' in txt(c):
        print('   cfg line:', [ln.strip() for ln in txt(c).splitlines() if 'n_iter=60' in ln])
        break
