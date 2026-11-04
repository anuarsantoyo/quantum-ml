#!/usr/bin/env python3
"""Build 24g from 24f — Phase A7: two-phase mu schedule (wide explore, then damped)."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24g.ipynb')
if os.path.exists(p):
    sys.exit('24g.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24g', '24e',
                'two-phase mu schedule: explore then damp (Phase A7)',
                'A7: replace the cosine travel schedule by a two-phase one - a 2x cap for the first half of '
                'the run, a small (1/3) cap for the second half, at the SAME total travel (forked from 24e; '
                '24f was the n_iter=60 control branch, reverted)'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: shape mode -> two_phase ----
done = False
for c in cells:
    s = txt(c)
    if "mu_sched_shape='cosine'" in s:
        s = s.replace(
            "    mu_sched_shape='cosine',                     # A3: cosine per-step travel c_k (sum = mu_init)\n",
            "    mu_sched_shape='two_phase',                  # A7: 2x cap first half, 1/3 cap second half\n"
            "    mu_tp_hi=2.0, mu_tp_lo=(1.0 / 3.0),          # the two per-step cap multipliers\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: mu_sched_shape line not found')

# ---- 2. run_experiment: add the two_phase branch ----
done = False
for c in cells:
    s = txt(c)
    old = ("    if _shape_mode == 'cosine':\n"
           "        _shape = 1.0 + np.cos(np.pi * _kk / max(n_iter - 1, 1))    # front-loaded, mean 1\n"
           "    else:\n"
           "        _shape = np.ones(n_iter)\n")
    if old in s:
        new = ("    if _shape_mode == 'cosine':\n"
               "        _shape = 1.0 + np.cos(np.pi * _kk / max(n_iter - 1, 1))    # front-loaded, mean 1\n"
               "    elif _shape_mode == 'two_phase':\n"
               "        _shape = np.where(_kk < n_iter // 2, cfg['mu_tp_hi'], cfg['mu_tp_lo'])  # explore / damp\n"
               "    else:\n"
               "        _shape = np.ones(n_iter)\n")
        s = s.replace(old, new)
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: cosine branch not found')

# ---- 3. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24g (Phase A7)

- **Change vs previous notebook (`24e`):** the travel **shape** changes from cosine to **two-phase** —
  `shape_k = 2.0` for the first half of the run, `1/3` for the second half (renormalised so
  `Σ_k c_k = μ_init · g` is unchanged). Everything else (n_iter = 30, σ_prop²-scaled LR, freeze rule, γ,
  loss, bandwidths) is frozen.
  *(Fork note: `24f` was the roadmap's `n_iter 30→60` **control branch**; its verdict was “the aggregate is
  unchanged ⇒ travel time, not step count, sets the landing”, so the main chain resumes from `24e` at 30
  steps instead of inheriting 60 for every downstream notebook.)*
- **Hypothesis:** the `24e` verdict showed a hard stop (`freeze`) discards **real travel** (the cells still
  under-travel), while `24d` showed re-scaling the LR **saturates** (gradient sign flips, clip bursts). The
  remaining form of the same idea is **damping**: spend the budget early (2× cap, when the gradient is large
  and consistent) and then keep only a small 1/3-cap step (which still carries a direction estimate but cannot
  wander far). Prediction: the landing is set by the *early* trajectory → the under-travelling cells move
  toward truth at the **same total travel**.
- **Falsifiable prediction:** `W_obj(all14)` **< 24e's 0.0771** and below 24d's 0.0744 (outside the ±0.005
  chaos band), driven by the low-T and mid-T cells moving toward 1 (μ ratios 0.65–0.79 → higher); the high-T
  cells within ±0.05 of 24e; γ unchanged; 0 divergences.
- **If falsified** (no improvement over 24e, or a regression): the schedule *shape* is not a lever at all →
  close Phase A as exhausted and the residual is the σ-channel bias (**FM#8**) → Phase B (`24k` σ-shape
  matching).
""")
        done = True
        break
if not done: sys.exit('hypothesis cell not found')

# ---- 4. verdict placeholder ----
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
    if "two_phase" in s and 'run_experiment' in s:
        print('--- patched lines ---')
        for ln in s.splitlines():
            if any(k in ln for k in ('two_phase', '_shape = np.where', "mu_sched_shape")):
                print('   ', ln.strip())
        break
