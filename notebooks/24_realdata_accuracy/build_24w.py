#!/usr/bin/env python3
"""Build 24w from 24v — Phase C5: coverage referee (one honest-interval table).

Adds DO_REFEREE=True -- no new campaign: inherit 24v's point estimate + Fisher (RESULTS) and
read the companion JSONs (24s_bootstrap, 24u_profile, 24v_bias); print ONE table comparing
CRB / sandwich / bootstrap / profile / bias-corrected across the 14 real experiments.
DO_BIAS/DO_PROFILE/DO_BOOTSTRAP off (all read from the JSONs); DO_FISHER stays True (cached).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24w.ipynb')
if os.path.exists(p):
    sys.exit('24w.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24w', '24v',
                'coverage referee — one honest-interval table (Phase C5)',
                'C5: DO_REFEREE=True -- ONE table comparing CRB / sandwich / bootstrap / profile / '
                'bias-corrected across the 14 real experiments; no new campaign (inherit 24v results + '
                'read data/processed/24s_bootstrap.json, 24u_profile.json, 24v_bias.json). DO_BIAS / '
                'DO_PROFILE / DO_BOOTSTRAP off.', ],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: turn the C-phase blocks off, add DO_REFEREE ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_FISHER' in s and 'CFG = dict(' in s:
        s = s.replace("DO_BIAS = True         # C4 (24v): bias-corrected CRB via the sim-at-truth bias",
                      "DO_BIAS = False        # C4 off (bias read from data/processed/24v_bias.json)\n"
                      "DO_REFEREE = True      # C5 (24w): one coverage table over CRB/sandwich/boot/profile/bias-corrected")
        set_txt(c, s); done = True; break
if not done: sys.exit('config cell not found')

# ---- cell 8: inherit the 24v results (no re-optimisation) when DO_REFEREE ----
done = False
for c in cells:
    s = txt(c)
    if 'RUN: all 14 real experiments' in s:
        old = ("N_WORKERS = 4\n"
               "CACHED = (not SMOKE) and os.path.exists(OUT_JSON) and os.environ.get('NB_RECOMPUTE') != '1'\n"
               "RESULTS = []\n"
               "if CACHED:\n"
               "    RESULTS = json.load(open(OUT_JSON))['results']\n"
               "    print(f'loaded {len(RESULTS)} experiments from {OUT_JSON} (set NB_RECOMPUTE=1 to recompute)')\n")
        new = ("N_WORKERS = 4\n"
               "REF_SRC = os.path.join(REPO_ROOT, 'data', 'processed', '24v_history.json')\n"
               "CACHED = (not SMOKE) and os.path.exists(OUT_JSON) and os.environ.get('NB_RECOMPUTE') != '1'\n"
               "RESULTS = []\n"
               "if DO_REFEREE and not SMOKE:\n"
               "    # C5 (24w): inherit the 24v point estimate + Fisher (the C-phase is a pure uncertainty side-car;\n"
               "    # no re-optimisation, so the referee is cheap and consistent with the 24s/24u/24v JSONs).\n"
               "    RESULTS = json.load(open(REF_SRC))['results']\n"
               "    CACHED = True\n"
               "    print(f'C5 referee: inherited {len(RESULTS)} experiments from {REF_SRC} (no re-optimisation)')\n"
               "elif CACHED:\n"
               "    RESULTS = json.load(open(OUT_JSON))['results']\n"
               "    print(f'loaded {len(RESULTS)} experiments from {OUT_JSON} (set NB_RECOMPUTE=1 to recompute)')\n")
        assert old in s, 'cell-8 anchor not found'
        set_txt(c, s.replace(old, new)); done = True; break
if not done: sys.exit('run cell not found')

# ---- cell 16: still save the 24w companion JSON in referee mode ----
done = False
for c in cells:
    s = txt(c)
    if 'save companion run data' in s:
        s = s.replace("if SMOKE or CACHED:\n    print('smoke/cached: not re-saving')",
                      "if SMOKE or (CACHED and not DO_REFEREE):\n    print('smoke/cached: not re-saving')")
        set_txt(c, s); done = True; break
if not done: sys.exit('save cell not found')

# ---- insert the referee cell AFTER the C4 (bias) cell ----
bias_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'BIAS-CORRECTED CRB' in txt(c):
        bias_idx = i; break
if bias_idx is None: sys.exit('bias cell not found')

ref_src = '''# ============================================================
# SERIES 24 / C5 (24w) \u2014 COVERAGE REFEREE: ONE honest-interval table
#   CRB | sandwich | bootstrap | profile | bias-corrected, over the 14 real experiments.
#   No new campaign: inherit 24v's point estimate + Fisher (RESULTS) and read the companion
#   JSONs 24s_bootstrap.json (C1), 24u_profile.json (C3), 24v_bias.json (C4).
#   mu_true enters ONLY to score coverage (the referee's yardstick) -- never to tune a knob.
# ============================================================
def _fin(v):
    try: v = float(v)
    except (TypeError, ValueError): return float('nan')
    return v if np.isfinite(v) else float('nan')

_bootJ = json.load(open(os.path.join(REPO_ROOT, 'data', 'processed', '24s_bootstrap.json')))
_profJ = json.load(open(os.path.join(REPO_ROOT, 'data', 'processed', '24u_profile.json')))['profile']
_biasJ = json.load(open(os.path.join(REPO_ROOT, 'data', 'processed', '24v_bias.json')))['bias']
_boot  = _bootJ['boot']

REF = {}
for r in RESULTS:
    k = r['exp']; fr = r.get('fisher_frozen') or {}
    b = _biasJ.get(k, {}); pf = _profJ.get(k, {}); bs = _boot.get(k)
    REF[k] = dict(T=int(str(k).split('Trans')[-1]), power=r['power'], N=int(r['n_target']),
                  mu_true=float(r['mu_true']), mu_hat=float(r['mu_final']),
                  crb=_fin(fr.get('sig_mu_data')), sand=_fin(fr.get('sig_mu_sand')),
                  boot=(_fin(bs['mu_sd']) if bs else float('nan')),
                  prof=_fin(pf.get('sig_prof')),
                  mu_corr=_fin(b.get('mu_corr')))

def _cov(v, sk, pk, z):
    if not (np.isfinite(v[sk]) and np.isfinite(v[pk])): return None
    return abs(v[pk] - v['mu_true']) <= z * v[sk]

CANDS = [
    ('CRB (raw)',                 'crb',  'mu_hat',  lambda v: True),
    ('sandwich (raw)',            'sand', 'mu_hat',  lambda v: True),
    ('bootstrap (raw)',           'boot', 'mu_hat',  lambda v: np.isfinite(v['boot'])),
    ('profile quad (raw)',        'prof', 'mu_hat',  lambda v: True),
    ('CRB (bias-corrected)',      'crb',  'mu_corr', lambda v: np.isfinite(v['mu_corr'])),
    ('sandwich (bias-corrected)', 'sand', 'mu_corr', lambda v: np.isfinite(v['mu_corr'])),
]

print('=' * 120)
print('C5 (24w) \\u2014 HONEST-INTERVAL REFEREE: mu coverage of mu_true across the 14 real experiments')
print('=' * 120)
print(f"{'interval':30s}{'n':>4s} | {'1sig':>7s} {'2sig':>7s} | {'med sig/CRB':>12s} {'med width/CRB':>14s}")
SUMM = {}
for label, sk, pk, sel in CANDS:
    n = h1 = h2 = 0
    for k, v in REF.items():
        if not sel(v): continue
        c1 = _cov(v, sk, pk, 1.0); c2 = _cov(v, sk, pk, 2.0)
        if c1 is None or c2 is None: continue
        n += 1; h1 += int(c1); h2 += int(c2)
    _rs = [v[sk] / v['crb'] for k, v in REF.items()
           if sel(v) and np.isfinite(v[sk]) and np.isfinite(v['crb'])]
    _mr = float(np.median(_rs)) if _rs else float('nan')
    SUMM[label] = dict(n=n, h1=h1, h2=h2, med_ratio=_mr)
    print(f'{label:30s}{n:4d} | {h1:3d}/{n:<3d} {h2:3d}/{n:<3d} | {_mr:12.2f} {_mr:14.2f}')

print()
print('per-experiment (|d| = |point - mu_true|; the honest interval is the one with |d|/sigma < 2):')
_hdr = (f"{'exp':14s}{'mu_true':>8s}{'mu_hat':>8s}{'|d|':>7s} | "
        f"{'CRB':>7s}{'d/C':>6s} | {'sand':>8s}{'d/S':>6s} | {'boot':>7s}{'d/B':>6s} | "
        f"{'prof':>7s}{'d/P':>6s} | {'mu_corr':>8s}{'|dc|':>7s}{'dc/C':>6s}")
print(_hdr)
for k, v in REF.items():
    d = abs(v['mu_hat'] - v['mu_true'])
    dc = abs(v['mu_corr'] - v['mu_true']) if np.isfinite(v['mu_corr']) else float('nan')
    r = lambda a, b: (a / b if (np.isfinite(a) and np.isfinite(b) and b > 0) else float('nan'))
    print(f"{k:14s}{v['mu_true']:8.2f}{v['mu_hat']:8.2f}{d:7.2f} | "
          f"{v['crb']:7.2f}{r(d, v['crb']):6.1f} | {v['sand']:8.1f}{r(d, v['sand']):6.1f} | "
          f"{v['boot']:7.2f}{r(d, v['boot']):6.1f} | {v['prof']:7.2f}{r(d, v['prof']):6.1f} | "
          f"{v['mu_corr']:8.2f}{dc:7.2f}{r(dc, v['crb']):6.1f}")

# ---- a compact gamma line (the C-phase was mu-centric; gamma reported for completeness) ----
_g2c = sum(1 for k, v in REF.items() if np.isfinite(v['crb']))
print()
print('gamma (completeness): the C-phase tools above were built for mu; the 24v history also stores the')
print('gamma CRB/sandwich widths -- see data/processed/24v_history.json (fisher_frozen).')

if not SMOKE:
    _ro = os.path.join(REPO_ROOT, 'data', 'processed', '24w_referee.json')
    json.dump(dict(notebook=NB_ID, method='coverage_referee', summary=SUMM,
                   cells={k: {kk: (None if not np.isfinite(vv) else float(vv)) for kk, vv in v.items()
                              if not isinstance(vv, str)} for k, v in REF.items()}),
              open(_ro, 'w'))
    print('saved', _ro)
'''
cells.insert(bias_idx + 1, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                                source=ref_src.splitlines(keepends=True)))

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24w (Phase C5)

- **Change vs previous notebook (`24v`):** `DO_REFEREE=True` \u2014 a **single consolidated coverage table**
  over the 14 real experiments comparing every honest-interval candidate built in Phase C: the raw
  dataset **CRB** (24t), the **sandwich** `H\u207b\u00b9KH\u207b\u00b9` (24t), the empirical **data bootstrap** (24s), the
  **profile-likelihood** quadratic-fit \u03c3 (24u), and the **bias-corrected** estimate (24v).  No new campaign:
  the notebook inherits 24v's point estimate + Fisher and reads the three companion JSONs.
- **Hypothesis (closing Phase C):** the C1\u2013C4 results compose into one statement \u2014 the residual \u03bc error is
  a **systematic bias**, so the *width* of the CRB is adequate but its *centring* is wrong; the only interval
  that both contains the truth and is not absurdly wide is the **bias-corrected** one.
- **Falsifiable prediction:** in the referee table the 2\u03c3 coverage ordering will be
  **biased-corrected \u2265 sandwich > CRB > profile > bootstrap**, with the bias-corrected CRB near 12/14
  (24v), the raw sandwich 10/14 (24t), the raw CRB 5/14 (24t), the profile 3/14 (24u) and the bootstrap
  0/8 (24s).  **Falsified if** any differently-centred candidate dominates the bias-corrected one, or if the
  bias-corrected coverage collapses when read from the frozen 24v artefacts (a sign the 24v correction was a
  one-draw artefact).
- **If falsified:** the honest interval is genuinely undetermined and Phase D must fix the estimator before any
  interval can be trusted.
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
for c in cells:
    for ln in txt(c).splitlines():
        if 'DO_BIAS' in ln or 'DO_REFEREE' in ln or 'DO_PROFILE' in ln:
            print('   ', ln.strip())
