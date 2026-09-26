#!/usr/bin/env python3
"""Build 24e from 24d — Phase A5: residual-aware truth-free mu freeze."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24e.ipynb')
if os.path.exists(p):
    sys.exit('24e.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24e', '24d',
                'residual-aware truth-free mu freeze (Phase A5)',
                'A5: freeze the mu update once |grad_mu| falls below a threshold (0.15x the init gradient) '
                '- a truth-free convergence test that stops late-step wandering'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: freeze knobs ----
done = False
for c in cells:
    s = txt(c)
    if "mu_lr_cap=2.0, mu_lr_floor=1.0," in s:
        s = s.replace(
            "    mu_lr_cap=2.0, mu_lr_floor=1.0,              # ... capped at 2x / floored at 1x the A2 LR\n",
            "    mu_lr_cap=2.0, mu_lr_floor=1.0,              # ... capped at 2x / floored at 1x the A2 LR\n"
            "    mu_freeze=True, mu_freeze_frac=0.15,         # A5: stop mu when |grad| < frac * |grad_init|\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config: mu_lr_cap line not found')

# ---- 2. run_experiment: freeze rule ----
done = False
for c in cells:
    s = txt(c)
    old = ("        # A3: same clip rule, but the cap is the cosine-shaped mu_step_k[step] (total <= mu_init)\n"
           "        mu_val -= float(np.clip(lr_mu_eff_e * grad_mu, -mu_step_k[step], mu_step_k[step]))\n")
    if old in s:
        new = ("        # A5 (24e): truth-free convergence freeze -- once |grad_mu| < frac*|grad_mu_init|,\n"
               "        # STOP updating mu (the cell has reached its stationary point; further steps only wander).\n"
               "        frozen = bool(cfg.get('mu_freeze', False)) and (abs(grad_mu) < cfg['mu_freeze_frac'] * grad_mu_init)\n"
               "        if not frozen:\n"
               "            mu_val -= float(np.clip(lr_mu_eff_e * grad_mu, -mu_step_k[step], mu_step_k[step]))\n")
        s = s.replace(old, new)
        s = s.replace(
            "                            grad_mu=grad_mu, grad_gamma=grad_gamma,\n",
            "                            grad_mu=grad_mu, grad_gamma=grad_gamma, frozen=bool(frozen),\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment: A3 step line not found')

# ---- 3. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction — 24e (Phase A5)

- **Change vs previous notebook (`24d`):** add a **truth-free convergence freeze** for μ — once the
  per-step `|∇μ|` falls below `frac · |∇μ|_init` (`frac = 0.15`, `|∇μ|_init` measured at the init), the μ
  update is **frozen** for the rest of the run. The LR (already σ_prop²-scaled in 24d), the cosine caps, γ,
  the loss and the bandwidths are frozen.
- **Hypothesis (`FM#14`):** the `24d` diagnostic shows the *net* travel saturates because the noisy
  REINFORCE gradient **flips sign** several times per run — late steps that fire when `|∇μ|` has already
  collapsed are pure waste. A residual-aware stop should remove exactly that waste and cost nothing where
  `|∇μ|` stays large.
- **Falsifiable prediction:** (i) cells whose `|∇μ|/|∇μ|_init` never falls below 0.15 stay **bit-identical**
  (from the 24d table: 1nW T80 and 3nW T80 have median `raw/cap` > 1); (ii) the freezing cells move by
  ≲0.03 in μ ratio, so **`|ΔW_obj(all14)| ≤ 0.005`** (a neutral control).
  **Falsified if** the freeze improves the objective by > 0.005 (then the late flips were pure waste and A7 is
  promoted) or worsens it by > 0.005 (then the late steps carry real μ signal).
- **If falsified:** record the direction and hand the conclusion to A7 (`24g`, two-phase damping).
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
    if 'frozen = bool(cfg.get' in s:
        print('--- patched lines ---')
        for ln in s.splitlines():
            if any(k in ln for k in ('mu_freeze', 'frozen =', 'if not frozen', 'mu_val -=', 'frozen=bool')):
                print('   ', ln.strip())
        break
