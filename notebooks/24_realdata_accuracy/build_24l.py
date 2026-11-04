#!/usr/bin/env python3
"""Build 24l from 24k — Phase B2: SCALE-ONLY control for B1 (19c's rejected fix)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24l.ipynb')
if os.path.exists(p):
    sys.exit('24l.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24l', '24k',
                'scale-only control for B1 (Phase B2)',
                "B2: the SAME sigma_channel remap machinery as 24k, but with the map reduced to a single "
                "scalar f(x) = a*x (a = median(target sigma_fit)/median(init sim sigma_fit), f-prime = a) "
                "instead of the monotone quantile map -- the isolated SCALE control for B1 (19c's rejected "
                "scale fix)"],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: quantile -> scale ----
done = False
for c in cells:
    s = txt(c)
    if "sigma_map='quantile', si_q=65," in s:
        s = s.replace(
            "    sigma_map='quantile', si_q=65,               # B1: monotone quantile map sim sigma_fit -> target\n",
            "    sigma_map='scale',                           # B2: SCALE-ONLY control (f(x)=a*x), no shape change\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: sigma_map line not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24l (Phase B2)

- **Change vs previous notebook (`24k`):** the **same** σ-remap machinery, but the map is reduced from the
  monotone quantile map to a **single scalar** `f(x) = a·x` with `a = median(target σ_fit)/median(init sim σ_fit)`
  (so `f′ = a` constant and `sim_ds ← a·sim_ds`). One knob: `sigma_map: 'quantile' → 'scale'`. Everything else
  is 24k verbatim. This is the isolated **control** that separates **shape** from **scale** (the fix 19c
  already tried, on the score side, and rejected).
- **Hypothesis (19c):** matching the σ *scale* alone does not move the μ attractor — the attractor is the
  **shape** of σ(μ). So the per-cell `μ_final/μ_true` table should stay where 24g was (< 1 at high T,
  > 1 at low T) and `W_obj` should be ≈ 24g/24k.
- **Falsifiable prediction:** `W_obj(all14)` within ±0.005 of **24g's 0.0747** and the per-cell μ ratios
  within ±0.05 of 24g's — i.e. **no systematic motion toward 1.0**. If instead the scale arm moves μ as much
  as B1, then B1's gain (if any) is not attributable to shape.
- **If falsified:** a pure scale change does move μ → the 19c conclusion does not transfer to the current
  travel-normalised optimiser and B1's shape story must be re-examined.
- **Interpretation rule:** B2 is only meaningful read **against B1 and 24g** — it is a control, not a candidate.
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p)
