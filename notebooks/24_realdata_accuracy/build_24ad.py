#!/usr/bin/env python3
"""Build 24ad from 24ac — Phase E4: held-out protocol (core tune / held-out report). CLOSING notebook.

ONE change: split the 14 real transmissions into a CORE selection split {20,60,100} and a HELD-OUT
report split {5,10,40,80}; select on the core, report on the held-out.  Final honest verdict on
whether the series' model generalises.  (E2_SEEDS stays [42] — single campaign of the E1 winner.)
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24ad.ipynb')
if os.path.exists(p):
    sys.exit('24ad.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24ad', '24ac',
                'E4: held-out protocol (tune on T20/60/100, report on T05/10/40/80) \u2014 closing notebook',
                'E4: held-out protocol \u2014 select the model on the CORE transmissions T in {20,60,100}, '
                'then report on the HELD-OUT transmissions T in {5,10,40,80}; the E1 winner is also this '
                'notebook\'s own freshly-run campaign.  Final honest verdict on whether the series\' model '
                'generalises (no knob in the series was ever selected on the held-out split).'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- insert the E4 held-out cell right AFTER the E3 referee cell ----
e3_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'E3 (24ac) \u2014 INDEPENDENT REFEREE' in txt(c):
        e3_idx = i; break
if e3_idx is None: sys.exit('E3 referee cell not located')

e4_src = '''# ============================================================
# SERIES 24 / E4 (24ad) \u2014 HELD-OUT PROTOCOL (core selection / held-out report)
#   SELECTION split = the CORE transmissions T in {20,60,100}; REPORT split = the HELD-OUT
#   transmissions T in {5,10,40,80}.  Candidate points are the series' saved configs; the E1
#   winner is ALSO this notebook's own freshly-run campaign (RESULTS = CFG, i.e. the winner).
#   Honesty: no knob anywhere in series 24 was ever selected on the held-out split (every
#   decision used the all-14 T-weighted objective), so the held-out numbers are untouched.
#   mu_true/gamma_true enter ONLY through the frozen metric (the scoring yardstick), never a knob.
# ============================================================
CORE_T, HOLD_T = {20, 60, 100}, {5, 10, 40, 80}
E4_CANDS = ['24a', '24g', '24i', '24n', '24p', '24r', '24y', '24aa']
E4 = {'24ad (winner, re-run)': dict(results=RESULTS)}
for _cid in E4_CANDS:
    _hp = os.path.join(REPO_ROOT, 'data', 'processed', f'{_cid}_history.json')
    if not os.path.exists(_hp):
        continue
    E4[_cid] = dict(results=json.load(open(_hp))['results'])

print('E4 (24ad) \u2014 held-out protocol: select on CORE {20,60,100}, report on HELD-OUT {5,10,40,80}')
print(f"{'candidate':>22}{'core':>10}{'held-out':>11}{'all14':>9}{'gap':>9}")
for _k, _v in E4.items():
    _v['core'] = tw_objective(_v['results'], Ts=CORE_T)
    _v['hold'] = tw_objective(_v['results'], Ts=HOLD_T)
    _v['all14'] = tw_objective(_v['results'], Ts=None)
for _k, _v in sorted(E4.items(), key=lambda kv: kv[1]['core']):
    print(f"{_k:>22}{_v['core']:>10.4f}{_v['hold']:>11.4f}{_v['all14']:>9.4f}"
          f"{_v['hold'] - _v['core']:>+9.4f}")
_bc = min(E4, key=lambda k: E4[k]['core'])
_bh = min(E4, key=lambda k: E4[k]['hold'])
print(f"\\nbest by CORE  (selection split)   : {_bc}  (core {E4[_bc]['core']:.4f})")
print(f"best by HELD-OUT (report split)   : {_bh}  (held-out {E4[_bh]['hold']:.4f})")
print(f"selection agreement: {'YES' if _bc == _bh else 'NO'}"
      f"   | winner {_bc} held-out {E4[_bc]['hold']:.4f} vs its core {E4[_bc]['core']:.4f}")

# ---- per-T breakdown of the winning model (this notebook's own run) ----
print()
print('per-experiment ratio (E1 winner, this run):')
print(f"{'T':>5}{'mu(1nW)':>10}{'mu(3nW)':>10}{'g(1nW)':>9}{'g(3nW)':>9}   split")
_SPL = {5: 'HELD', 10: 'HELD', 20: 'core', 40: 'HELD', 60: 'core', 80: 'HELD', 100: 'core'}
for _T in sorted({int(str(r['exp']).split('Trans')[-1]) for r in RESULTS}):
    _row = [_T]
    for _p in POWERS:
        _rs = [r for r in RESULTS if r['power'] == _p and int(str(r['exp']).split('Trans')[-1]) == _T]
        _row.append(round(_rs[0]['mu_final'] / _rs[0]['mu_true'], 3) if _rs else float('nan'))
    for _p in POWERS:
        _rs = [r for r in RESULTS if r['power'] == _p and int(str(r['exp']).split('Trans')[-1]) == _T]
        _row.append(round(_rs[0]['gamma_final'] / _rs[0]['gamma_true'], 3) if _rs else float('nan'))
    print(f"{_row[0]:>5}{_row[1]:>10}{_row[2]:>10}{_row[3]:>9}{_row[4]:>9}   {_SPL.get(_T, '?')}")

# ---- inline figure: core vs held-out per candidate ----
fig, ax = plt.subplots(figsize=(10, 5))
_ks = sorted(E4, key=lambda k: E4[k]['core'])
_x = np.arange(len(_ks))
ax.bar(_x - 0.2, [E4[k]['core'] for k in _ks], width=0.4, label='CORE (select)', color='#1d3557')
ax.bar(_x + 0.2, [E4[k]['hold'] for k in _ks], width=0.4, label='HELD-OUT (report)', color='#e76f51')
ax.set_xticks(_x); ax.set_xticklabels(_ks, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('W_obj (lower = better)'); ax.grid(alpha=0.3, axis='y')
ax.set_title('24ad \u2014 held-out protocol: core (T20/60/100) vs held-out (T05/10/40/80)')
ax.legend()
plt.tight_layout(); plt.show()

if not SMOKE:
    _eo = os.path.join(REPO_ROOT, 'data', 'processed', '24ad_holdout.json')
    E4_TABLE = {k: dict(core=v['core'], hold=v['hold'], all14=v['all14']) for k, v in E4.items()}
    json.dump(dict(notebook=NB_ID, core_T=sorted(CORE_T), hold_T=sorted(HOLD_T),
                   best_core=_bc, best_hold=_bh, agreement=bool(_bc == _bh), table=E4_TABLE),
              open(_eo, 'w'))
    print('saved', _eo)
else:
    E4_TABLE = {k: dict(core=v['core'], hold=v['hold'], all14=v['all14']) for k, v in E4.items()}
    print('smoke: held-out JSON not saved')
'''
cells.insert(e3_idx + 1, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                             source=e4_src.splitlines(keepends=True)))

# ---- save cell: add the held-out table ----
done = False
for c in cells:
    s = txt(c)
    if 'referee=REF_OBJ,' in s:
        s = s.replace("               referee=REF_OBJ,",
                      "               referee=REF_OBJ, holdout=E4_TABLE,")
        set_txt(c, s); done = True; break
if not done: sys.exit('save cell referee anchor not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24ad (Phase E4, CLOSING)

- **Change vs previous notebook (`24ac`):** add the **held-out protocol**.  The 14 real transmissions are
  split into a **CORE** selection split `T \u2208 {20, 60, 100}` and a **HELD-OUT** report split
  `T \u2208 {5, 10, 40, 80}`.  The series' candidates are ranked by the CORE objective only; the winner is
  then *reported* on the held-out split.  The E1 winner is also this notebook's own freshly-run campaign
  (`RESULTS`).  No knob in series 24 was ever selected on the held-out split \u2014 every decision used the
  all-14 (or a phase-level) objective \u2014 so the held-out numbers are honest.
- **Hypothesis (E4):** the series' model **generalises**: the candidate that wins the core split also wins
  the held-out split, and the winner's held-out objective is comparable to (not much worse than) its core
  objective.  This is the closing verdict on the whole series.
- **Falsifiable prediction:** `best_core == best_hold` (the E1 winner on both), and the winner's
  held-out `W_obj` is within **+0.02** of its core `W_obj`.  **Falsified if** the two selections disagree,
  or the held-out value is more than +0.02 worse than the core value \u2014 then the series' model is tuned to
  the specific transmissions in the core split and does not generalise.
- **If falsified:** declare the series' model **transmission-specific**, name the split responsible, and
  hand the failure to a series 25 (which should select on a random, not a T-based, split).
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
