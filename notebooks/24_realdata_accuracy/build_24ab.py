#!/usr/bin/env python3
"""Build 24ab from 24aa — Phase E2: repeat the E1 winner 3x at different SEED bases.

ONE change: the frozen campaign is repeated at SEED bases 42/43/44 (only the per-step noise
seed changes) to separate the E1 signal from the +/-0.01 chaos band.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, '24ab.ipynb')
if os.path.exists(p):
    sys.exit('24ab.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24ab', '24aa',
                'E2: repeat the E1 winner 3x at different SEED bases (signal vs chaos band)',
                'E2: repeat the frozen E1 winner (24aa config, byte-identical protocol) at three '
                'per-step noise SEED bases 42/43/44; report each run all14/T>=40 and the spread to '
                'separate the -0.0021 all14 margin over 24r from the +/-0.01 chaos band.'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- config: add the E2 seed list right after SEED ----
done = False
for c in cells:
    s = txt(c)
    if 'SEED  = 42' in s and 'CFG = dict(' in s:
        s = s.replace(
            "SEED  = 42             # per-step noise seed base (deterministic)\n",
            "SEED  = 42             # per-step noise seed base (deterministic)\n"
            "E2_SEEDS = [42, 43, 44]  # E2 (24ab): repeat the frozen campaign at these seed bases\n"
            "                         #   (only the per-step noise seed changes between repeats)\n")
        set_txt(c, s); done = True; break
if not done: sys.exit('config SEED line not found')

# ---- run cell: loop the campaign over E2_SEEDS ----
RUN_OLD = '''# ============================================================
# 23a — RUN: all 14 real experiments (the long cell)
# ============================================================
N_WORKERS = 4
CACHED = (not SMOKE) and os.path.exists(OUT_JSON) and os.environ.get('NB_RECOMPUTE') != '1'
RESULTS = []
if CACHED:
    RESULTS = json.load(open(OUT_JSON))['results']
    print(f'loaded {len(RESULTS)} experiments from {OUT_JSON} (set NB_RECOMPUTE=1 to recompute)')
t0 = time.time()
with (nullcontext() if CACHED else _PPE(max_workers=N_WORKERS, mp_context=_mp.get_context('fork'), initializer=_init_worker)) as pool:
    for exp in EXPERIMENTS:
        if CACHED:
            continue
        r = run_experiment(exp, CFG, pool)
        RESULTS.append(r)
        print(f"  {r['exp']:13s} mu {r['mu_true']:7.2f} -> {r['mu_final']:7.2f} | "
              f"gamma {r['gamma_true']:5.2f} -> {r['gamma_final']:6.2f} | NLL {r['nll_final']:.3f} | "
              f"{r['t_elapsed']/60:.1f} min{'  DIVERGED' if r['diverged'] else ''}", flush=True)
print(f'\\ntotal {len(RESULTS)} experiments in {(time.time()-t0)/60:.1f} min')'''

RUN_NEW = '''# ============================================================
# 23a — RUN: all 14 real experiments (the long cell)
#   E2 (24ab): the SAME frozen campaign is repeated for every seed base in E2_SEEDS.
#   The first seed's run is the primary RESULTS (figures + frozen metric); the other seeds
#   are the signal/chaos-band replicates.  ONLY the per-step noise seed base changes.
# ============================================================
N_WORKERS = 4
CACHED = (not SMOKE) and os.path.exists(OUT_JSON) and os.environ.get('NB_RECOMPUTE') != '1'
RESULTS = []
E2_RUNS = {}
if CACHED:
    _pl = json.load(open(OUT_JSON))
    RESULTS = _pl['results']
    E2_RUNS = {int(E2_SEEDS[0]): RESULTS}
    print(f'loaded {len(RESULTS)} experiments from {OUT_JSON} (set NB_RECOMPUTE=1 to recompute)')
t0 = time.time()
with (nullcontext() if CACHED else _PPE(max_workers=N_WORKERS, mp_context=_mp.get_context('fork'), initializer=_init_worker)) as pool:
    for _s in E2_SEEDS:
        if CACHED:
            continue
        globals()['SEED'] = int(_s)
        _rs = []
        for exp in EXPERIMENTS:
            r = run_experiment(exp, CFG, pool)
            _rs.append(r)
            print(f"  seed {_s} | {r['exp']:13s} mu {r['mu_true']:7.2f} -> {r['mu_final']:7.2f} | "
                  f"gamma {r['gamma_true']:5.2f} -> {r['gamma_final']:6.2f} | NLL {r['nll_final']:.3f} | "
                  f"{r['t_elapsed']/60:.1f} min{'  DIVERGED' if r['diverged'] else ''}", flush=True)
        E2_RUNS[int(_s)] = _rs
        print(f"  -> seed {_s} campaign complete  (total {(time.time()-t0)/60:.1f} min)", flush=True)
globals()['SEED'] = int(E2_SEEDS[0])
RESULTS = E2_RUNS[int(E2_SEEDS[0])]
print(f'\\ntotal {len(E2_SEEDS)} campaign(s) x {len(EXPERIMENTS)} experiments in {(time.time()-t0)/60:.1f} min')'''

hit = False
for c in cells:
    s = txt(c)
    if RUN_OLD in s:
        set_txt(c, s.replace(RUN_OLD, RUN_NEW)); hit = True; break
if not hit: sys.exit('run cell anchor not found')

# ---- insert the E2 signal/chaos cell right AFTER the frozen-metric cell ----
met_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'FROZEN PROTOCOL METRIC' in txt(c):
        met_idx = i; break
if met_idx is None: sys.exit('metric cell not found')

e2_src = '''# ============================================================
# SERIES 24 / E2 (24ab) \u2014 SIGNAL vs CHAOS BAND
#   The E1 winner (24aa config) is repeated at several noise SEED bases.  The frozen
#   T-weighted objective is computed per seed; the spread is the empirical chaos band in
#   this exact configuration, to be compared with 24aa's 0.0021 all14 margin over 24r.
# ============================================================
E2_OBJ = {}
for _s, _rs in E2_RUNS.items():
    E2_OBJ[_s] = dict(all14=tw_objective(_rs), hiT=tw_objective(_rs, Ts={40, 60, 80, 100}),
                      loT=tw_objective(_rs, Ts={5, 10, 20}),
                      mu_rel_rmse=float(np.sqrt(np.mean([((r['mu_final'] - r['mu_true']) / r['mu_true']) ** 2 for r in _rs]))) * 100,
                      gamma_rel_rmse=float(np.sqrt(np.mean([((r['gamma_final'] - r['gamma_true']) / r['gamma_true']) ** 2 for r in _rs]))) * 100,
                      n_div=int(sum(bool(r['diverged']) for r in _rs)))
print('E2 \u2014 repeat of the E1 winner at different SEED bases:')
print(f"{'seed':>6}{'all14':>10}{'T>=40':>10}{'T<=20':>10}{'mu%':>8}{'gam%':>8}{'div':>5}")
for _s, _o in E2_OBJ.items():
    print(f"{_s:>6}{_o['all14']:>10.4f}{_o['hiT']:>10.4f}{_o['loT']:>10.4f}"
          f"{_o['mu_rel_rmse']:>8.2f}{_o['gamma_rel_rmse']:>8.2f}{_o['n_div']:>5}")
_a14 = np.array([_o['all14'] for _o in E2_OBJ.values()])
_a40 = np.array([_o['hiT'] for _o in E2_OBJ.values()])
_a20 = np.array([_o['loT'] for _o in E2_OBJ.values()])
_dd = (lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0)
print(f"all14 : mean {_a14.mean():.4f}  sd {_dd(_a14):.4f}  range [{_a14.min():.4f}, {_a14.max():.4f}]")
print(f"T>=40 : mean {_a40.mean():.4f}  sd {_dd(_a40):.4f}  range [{_a40.min():.4f}, {_a40.max():.4f}]")
print(f"T<=20 : mean {_a20.mean():.4f}  sd {_dd(_a20):.4f}  range [{_a20.min():.4f}, {_a20.max():.4f}]")
for _lbl, _ref in [('24r', 0.0672), ('24p', 0.0684)]:
    _mg = _ref - _a14.mean()
    print(f"vs {_lbl} all14 {_ref:.4f}: mean margin {_mg:+.4f}  "
          f"(seed spread sd {_dd(_a14):.4f}; {'separated' if abs(_mg) > _dd(_a14) else 'INSIDE the chaos band'})")
print(f"=> signal test: winner all14 mean {_a14.mean():.4f} +/- {_dd(_a14):.4f} vs 24r 0.0672 "
      f"(margin {0.0672 - _a14.mean():+.4f})")
'''
cells.insert(met_idx + 1, dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                               source=e2_src.splitlines(keepends=True)))

# ---- save cell: add e2_by_seed ----
done = False
for c in cells:
    s = txt(c)
    if "payload = dict(notebook=NB_ID" in s:
        s = s.replace("payload = dict(notebook=NB_ID, one_change=ONE_CHANGE, config=CFG, DO_FISHER=DO_FISHER,",
                      "payload = dict(notebook=NB_ID, one_change=ONE_CHANGE, config=CFG, DO_FISHER=DO_FISHER,\n"
                      "               e2_seeds=list(E2_SEEDS), e2_by_seed=E2_OBJ,")
        set_txt(c, s); done = True; break
if not done: sys.exit('save cell anchor not found')

# ---- hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24ab (Phase E2)

- **Change vs previous notebook (`24aa`):** the frozen protocol is **repeated three times** at the
  per-step noise **SEED bases 42 / 43 / 44** (`E2_SEEDS`).  Nothing else changes \u2014 same model, same
  config, same metric.  The seed-42 repeat is the primary `RESULTS` (figures + frozen metric).
- **Hypothesis (E2):** 24aa's all14 win over 24r (`0.0651` vs `0.0672`, margin \u22120.0021) is a **signal**,
  not a lucky draw from the chaos band.  If real, the three repeats should cluster *below* 0.0672 with a
  spread much smaller than the 0.0021 margin; if the spread is \u2265 the margin, the win is not resolvable.
- **Falsifiable prediction:** the three seed replicates give `all14` \u2208 [0.060, 0.070] with an sd
  \u2264 0.0025, and their mean stays **below 24r's 0.0672**.  **Falsified if** the replicate sd exceeds the
  margin (mean margin inside \u00b11\u00b7sd) or the mean regresses to/above 24r \u2014 then 24r and 24aa are
  statistically tied and the series winner must be chosen on the secondary criteria (T\u226540, \u03b3).
- **If falsified:** report 24r/24aa as a tie and pick the winner by the T\u226540 and \u03b3 sub-metrics.
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, f'({len(cells)} cells)')
