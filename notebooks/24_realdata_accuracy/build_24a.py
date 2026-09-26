#!/usr/bin/env python3
"""Build notebooks/24_realdata_accuracy/24a.ipynb from series-23's 23a notebook.

Series-24 template = 23a (the frozen exp6 trial_07 KDE model) but LEAN:
  * adds the frozen T-weighted objective (identical in every series-24 notebook)
  * makes the heavy Fisher block optional behind DO_FISHER (off for the point-estimate phase)
  * adds NB_ID / ONE_CHANGE / a hypothesis markdown cell / a verdict placeholder
  * keeps the figures inline (FIG1 interactive -> HTML, FIG2-4 matplotlib) as per conventions
"""
import json, os, copy

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '23_fisher_uncertainties',
                   '23a-fisher-uncertainties-exp6-trial07.ipynb')
DST = os.path.join(HERE, '24a.ipynb')

src = json.load(open(SRC))
S = src['cells']

def md(text):  return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def co(text):  return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": text.splitlines(keepends=True)}

# ---------------------------------------------------------------- cell 0 : title
C0 = md("""# 24a — baseline: the `experiment_6 trial_07` KDE model reproduced (series-24 reference)

**Series 24 — real-data accuracy.** New folder, new cycle. Goal (Anuar, 2026-09-26): the model must
optimise **correctly to the real data**, with **higher transmissions weighted more**, but ideally all 14
real experiments approximated. Process = the same as always: **one change per notebook → hypothesis →
test → verdict → the next notebook follows this one.**

**Starting point (decided from the whole-project survey):** the project's best *KDE* model is
**AG-HYPOPT `experiment_6` `trial_07`** — the exp5 winning schedule + the exp6 winner
(`sigma_weight 0.8649`, `h_f_scale 4.0`, `h_s_scale 3.6745`, `gamma_rel_cap 0.10736`), i.e. the series-21
μ mechanism (Anuar's `loglik_mean` reward, `sigma_prop` score, **no clip, no clamps**, init `0.5 × truth`).
It is the only family that can also carry a Fisher/CRB (exp8's `cvm_fwhm`/`w1_2d` are discrepancies).

**This notebook (24a) changes NOTHING** — it re-runs that model on all 14 real experiments under the
**frozen series-24 metric** (T-weighted objective, CAP 1.0, W_FLOOR 0.25) and becomes the reference every
later notebook is compared against. Series-23's `23a` recorded **W_obj = 0.0811 (all 14), 0.0675 (T ≥ 40)**;
this notebook must reproduce it.
""")

# ---------------------------------------------------------------- cell 1 : hypothesis
C1 = md("""## Hypothesis / prediction — 24a (baseline)

- **Change vs previous notebook:** none. 24a *is* the frozen reference (forked from `23a`).
- **Hypothesis:** the frozen `experiment_6 trial_07` model, re-run under the series-24 protocol, is the
  best available real-data **KDE** point estimate and reproduces the series-23 number.
- **Falsifiable prediction:** `W_obj(all14) ≈ 0.081` (± the chaos band), 0 divergences, γ ≤ 25 % rel-RMSE,
  and μ still the dominant channel (≈ 36 % rel-RMSE).
- **If falsified:** the series baseline is wrong → fix the reproduction before any one-change notebook.
""")

# ---------------------------------------------------------------- cell 5 : config (new)
C5 = co('''# ============================================================
# 24a CONFIG — FROZEN experiment_6 trial_07 model (the series-24 baseline)
# ============================================================
NB_ID      = '24a'
ONE_CHANGE = 'none - baseline: AG-HYPOPT experiment_6 trial_07 reproduced exactly (best KDE model)'
DO_FISHER  = False     # lean series-24 notebooks: point estimate only (Phase C switches this on)

CFG = dict(
    n_runs=100, n_iter=30,                       # series-21/exp5/6 protocol
    # FROZEN schedule = experiment_5 winning trial (trial_21)
    lr_mu=0.1669, mu_anneal=0.3455, lr_gamma=0.4716, gamma_anneal=0.4723,
    sigma_ref=10.0, clip=float('inf'),           # no gradient clipping (21i)
    # FROZEN experiment_6 winner (trial_07)
    sigma_weight=0.8648862719598797,             # sigma-channel damping
    gamma_rel_cap=0.10735617789825944,           # relative gamma trust region
    h_f_scale=4.0, h_s_scale=3.674548492934102,  # inflated Scott bandwidths
    h_s_min=0.05,
)
H_REF = 1.0            # z-form gamma-score reference (optimizer only)
SEED  = 42             # per-step noise seed base (deterministic)
REAL_CSV = os.path.join(REPO_ROOT, 'data', 'processed', 'fwhm_linewidths.csv')

# ---- Fisher knobs (used ONLY when DO_FISHER is True) ----
M_FINAL, FISHER_SEEDS, FISHER_BASE_SEED = 500, 5, 7000

OUT_JSON = os.path.join(REPO_ROOT, 'data', 'processed', f'{NB_ID}_history.json')

SMOKE = os.environ.get('NB_SMOKE') == '1'
if SMOKE:
    CFG['n_runs'], CFG['n_iter'] = 20, 4
    EXPERIMENTS = EXPERIMENTS[4:5]
    M_FINAL, FISHER_SEEDS = 40, 1
CACHED = (not SMOKE) and os.path.exists(OUT_JSON) and os.environ.get('NB_RECOMPUTE') != '1'
print(f'{NB_ID} | one change: {ONE_CHANGE}')
print(f'config: {CFG}')
print(f'M_FINAL={M_FINAL}, seeds={FISHER_SEEDS} | DO_FISHER={DO_FISHER} | SMOKE={SMOKE} | CACHED={CACHED}')
''')

# ---------------------------------------------------------------- cell 9 : frozen metric (new)
C9 = co('''# ============================================================
# SERIES 24 — FROZEN PROTOCOL METRIC (identical in every series-24 notebook)
#   T-weighted objective (exp5/6 definition): per-cell rel-sq error capped at CAP,
#   weight w(T) = W_FLOOR + (1-W_FLOOR)*T/100.  Lower = better.
# ============================================================
CAP, W_FLOOR = 1.0, 0.25
def tw_objective(results, Ts=None):
    rows = []
    for r in results:
        T = int(str(r['exp']).split('Trans')[-1])
        if Ts is not None and T not in Ts:
            continue
        em = min(((r['mu_final'] - r['mu_true']) / r['mu_true']) ** 2, CAP)
        eg = min(((r['gamma_final'] - r['gamma_true']) / r['gamma_true']) ** 2, CAP)
        w  = W_FLOOR + (1.0 - W_FLOOR) * T / 100.0
        rows.append((w, em, eg))
    if not rows:
        return float('nan')
    wts  = np.array([x[0] for x in rows] * 2)
    errs = np.array([x[1] for x in rows] + [x[2] for x in rows])
    return float((wts * errs).sum() / wts.sum())

OBJ_ALL = tw_objective(RESULTS)
OBJ_HI  = tw_objective(RESULTS, Ts={40, 60, 80, 100})
OBJ_LO  = tw_objective(RESULTS, Ts={5, 10, 20})
MU_RMSE = float(np.sqrt(np.mean([((r['mu_final'] - r['mu_true']) / r['mu_true']) ** 2 for r in RESULTS]))) * 100
G_RMSE  = float(np.sqrt(np.mean([((r['gamma_final'] - r['gamma_true']) / r['gamma_true']) ** 2 for r in RESULTS]))) * 100
N_DIV   = int(sum(bool(r['diverged']) for r in RESULTS))
print('=' * 74)
print(f'FROZEN METRIC ({NB_ID})   W_obj(all14) = {OBJ_ALL:.4f} | T>=40 {OBJ_HI:.4f} | T<=20 {OBJ_LO:.4f}')
print(f'                          mu rel-RMSE {MU_RMSE:.1f}% | gamma rel-RMSE {G_RMSE:.1f}% | diverged {N_DIV}/14')
print('=' * 74)
print(f"{'exp':14s}{'mu_true':>9s}{'mu_fit':>9s}{'mu/true':>9s} | {'gam_true':>9s}{'gam_fit':>9s}{'gam/true':>9s}")
for r in RESULTS:
    print(f"{r['exp']:14s}{r['mu_true']:9.2f}{r['mu_final']:9.2f}{r['mu_final']/r['mu_true']:9.3f} | "
          f"{r['gamma_true']:9.2f}{r['gamma_final']:9.2f}{r['gamma_final']/r['gamma_true']:9.3f}"
          f"{'   DIV' if r['diverged'] else ''}")
print()
print(f"{'T':>6}{'mu(1nW)':>10}{'mu(3nW)':>10}{'gam(1nW)':>11}{'gam(3nW)':>11}")
for T in sorted({int(str(r['exp']).split('Trans')[-1]) for r in RESULTS}):
    vals = []
    for p in POWERS:
        rs = [r for r in RESULTS if r['power'] == p and int(str(r['exp']).split('Trans')[-1]) == T]
        vals.append(round(rs[0]['mu_final'] / rs[0]['mu_true'], 3) if rs else float('nan'))
    for p in POWERS:
        rs = [r for r in RESULTS if r['power'] == p and int(str(r['exp']).split('Trans')[-1]) == T]
        vals.append(round(rs[0]['gamma_final'] / rs[0]['gamma_true'], 3) if rs else float('nan'))
    print(f"{T:>6}{vals[0]:>10}{vals[1]:>10}{vals[2]:>11}{vals[3]:>11}")
''')

# ---------------------------------------------------------------- cell 10 : fisher (gated)
C10 = co('''# ============================================================
# SERIES 24 — FISHER at each optimum (ONLY when DO_FISHER):
#   (A) per-scan normalisation J = sum/N   -> sigma_*_single   (the historic convention)
#   (B) dataset correction  sig/sqrt(N)    -> sigma_*_data     (the whole experiment)
# ============================================================
t0 = time.time()
if not DO_FISHER:
    print('DO_FISHER=False - Fisher block skipped (series-24 point-estimate notebook)')
elif CACHED and all('fisher_frozen' in r for r in RESULTS):
    print('fisher block: cached')
else:
    with _PPE(max_workers=N_WORKERS, mp_context=_mp.get_context('fork'), initializer=_init_worker) as pool:
        for r in RESULTS:
            exp = next(e for e in EXPERIMENTS if e['name'] == r['exp'])
            tf = torch.tensor(r['target_f'], dtype=torch.float32)
            ts = torch.tensor(r['target_s'], dtype=torch.float32)
            r['fisher_frozen'] = _fisher_at(pool, r['mu_final'], r['gamma_final'], r['sigma_prop'],
                                            r['lam'], tf, ts, r['H_F'], r['H_S'], CFG)
            print(f"  {r['exp']:13s} N={r['n_target']:5d} | sig_mu per-scan "
                  f"{r['fisher_frozen']['sig_mu_single']:8.1f} -> dataset {r['fisher_frozen']['sig_mu_data']:7.2f}",
                  flush=True)
    print(f'fisher block: {(time.time()-t0)/60:.1f} min')
''')

# ---------------------------------------------------------------- cell 11 : table
C11 = co('''# ============================================================
# SERIES 24 — per-experiment summary table [+ Fisher coverage when DO_FISHER]
# ============================================================
TABLE = []
for r in RESULTS:
    dmu = abs(r['mu_final'] - r['mu_true']); dg = abs(r['gamma_final'] - r['gamma_true'])
    row = dict(exp=r['exp'], n=r['n_target'], mu_true=r['mu_true'], mu=r['mu_final'],
               gamma_true=r['gamma_true'], gamma=r['gamma_final'], dmu=dmu, dgamma=dg,
               ratio_mu=r['mu_final'] / r['mu_true'], ratio_gamma=r['gamma_final'] / r['gamma_true'])
    fr = r.get('fisher_frozen')
    if fr and fr.get('sig_mu_single'):
        row.update(sig_mu_single=fr['sig_mu_single'], sig_mu_data=fr['sig_mu_data'],
                   sig_g_single=fr['sig_g_single'], sig_g_data=fr['sig_g_data'], corr=fr.get('corr_data'),
                   dmu_sigma_single=dmu / fr['sig_mu_single'], dmu_sigma_data=dmu / fr['sig_mu_data'],
                   dg_sigma_single=dg / fr['sig_g_single'], dg_sigma_data=dg / fr['sig_g_data'])
    TABLE.append(row)

if DO_FISHER:
    cov = lambda k, thr=2.0: sum(1 for t in TABLE if t.get(k, 1e30) <= thr)
    print('coverage within 2 sigma (mu):  per-scan %d/14 | dataset %d/14'
          % (cov('dmu_sigma_single'), cov('dmu_sigma_data')))
    print('coverage within 1 sigma (mu):  per-scan %d/14 | dataset %d/14'
          % (cov('dmu_sigma_single', 1), cov('dmu_sigma_data', 1)))
    print('gamma: median |dgamma|/sigma  per-scan %.2f | dataset %.2f'
          % (np.median([t['dg_sigma_single'] for t in TABLE]),
             np.median([t['dg_sigma_data'] for t in TABLE])))
else:
    print('DO_FISHER=False - coverage table skipped')
''')

# ---------------------------------------------------------------- cell 12 : save (new)
C12 = co('''# ============================================================
# SERIES 24 — save companion run data (paths + per-step clouds [+ Fisher when DO_FISHER])
# ============================================================
if SMOKE or CACHED:
    print('smoke/cached: not re-saving')
else:
    payload = dict(notebook=NB_ID, one_change=ONE_CHANGE, config=CFG, DO_FISHER=DO_FISHER,
                   objective=dict(all14=OBJ_ALL, hiT=OBJ_HI, loT=OBJ_LO,
                                  mu_rel_rmse=MU_RMSE, gamma_rel_rmse=G_RMSE, n_diverged=N_DIV),
                   results=RESULTS)
    json.dump(payload, open(OUT_JSON, 'w'))
    print('saved', OUT_JSON, f'({os.path.getsize(OUT_JSON)/1e6:.2f} MB)')
''')

# ---------------------------------------------------------------- verdict + notes (new)
C_VERDICT = md("""## Verdict — *(fill in after the run)*

Replace this cell with: the frozen-metric table (W_obj all/≥40/≤20, μ & γ rel-RMSE, divergences), the
one-line scientific read, the HIT / PARTLY / FALSIFIED verdict on the prediction, and the FM tag.
""")

C_NOTES = md("""## Notes / caveats

- **Chaos band.** The objective is chaotically sensitive (~±0.01 for <1e-3 knob changes; integer photon
  rounding). Treat ≤0.005 differences as noise; repeat a winning config ≥2× before believing it.
- **Truth-free rule.** Every knob derived from data must come from the *target cloud* or the *sim cloud* —
  never from `mu_true`/`gamma_true`. (The protocol init `0.5 × truth` is the single documented exception
  and is a property of the protocol, not a tuned knob.)
- **One change per notebook**, stated in the markdown cell above the config; keep the rest byte-identical
  by **forking the previous notebook** (`cp 24x.ipynb 24y.ipynb`).
- Figures render inline (no `savefig`); FIG1 is exported to HTML for the browser. Run **in place**
  (`papermill 24x.ipynb 24x.ipynb`), never an `-executed` copy.
- Append the log row in `SERIES_STATE.md` §7 and commit+push after every notebook.
""")

# ---------------------------------------------------------------- assemble
cells = []
cells.append(C0)
cells.append(C1)
cells.append(S[1])          # panel markdown (22 conventions)
cells.append(S[2])          # imports
cells.append(S[3])          # EXPERIMENTS
cells.append(C5)            # config
cells.append(S[5])          # helpers
cells.append(S[6])          # run_experiment
cells.append(S[7])          # RUN
cells.append(C9)            # frozen metric
cells.append(C10)           # fisher (gated)
cells.append(C11)           # table
cells.append(C12)           # save
cells.append(S[11])         # FIG1 builder
cells.append(S[12])         # FIG1a
cells.append(S[13])         # FIG1b
cells.append(copy.deepcopy(S[14]))  # FIG2 (patched below)
cells.append(S[15])         # FIG3
cells.append(S[16])         # FIG4
cells.append(S[17])         # FIG5 (gated, patched below)
cells.append(C_VERDICT)
cells.append(C_NOTES)

nb = dict(cells=cells, metadata=src['metadata'],
          nbformat=src['nbformat'], nbformat_minor=src['nbformat_minor'])

# ---- patches: html paths + titles + gating ----
def sub(cell, pairs):
    s = "".join(cell['source'])
    for a, b in pairs:
        assert a in s, f'missing: {a[:60]}'
        s = s.replace(a, b)
    cell['source'] = s.splitlines(keepends=True)

# FIG1a/b: rename the exported HTML files
for _c, _tag in ((cells[14], 'fig1a'), (cells[15], 'fig1b')):
    sub(_c, [("'23_fisher_uncertainties'", "'24_realdata_accuracy'"),
             (f"'23a-{_tag}", f"'24a-{_tag}")])

# FIG2: optional whisker + generic title
sub(cells[16], [
    ("            err = [t['sig_mu_data'] / t['mu_true'] for t in ts]",
     "            err = [t['sig_mu_data'] / t['mu_true'] for t in ts] if DO_FISHER else None"),
    ("            err = [t['sig_g_data'] / t['gamma_true'] for t in ts]",
     "            err = [t['sig_g_data'] / t['gamma_true'] for t in ts] if DO_FISHER else None"),
    ("ax.errorbar(xs, ys, yerr=err, fmt=MK[power]",
     "ax.errorbar(xs, ys, yerr=err, fmt=MK[power]"),
    ("label=f'{power}   (whisker = Fisher σ_{key})')",
     "label=(f'{power}   (whisker = Fisher sigma_{key})' if DO_FISHER else power))"),
    ("ax.set_title(f'23a — {key} report: ratio to truth with honest Fisher σ_{key} whiskers (dataset CRB)')",
     "ax.set_title(f\"{NB_ID} - {key} report: ratio to truth\" + (\" with Fisher sigma whiskers (dataset CRB)\" if DO_FISHER else \"\"))"),
])

# FIG5: gate the whole Fisher-ellipse figure
s17 = "".join(cells[19]['source'])
gated = ("if not DO_FISHER:\n"
         "    print('DO_FISHER=False - FIG5 (Fisher ellipses) skipped')\n"
         "else:\n" + "\n".join(("    " + ln if ln.strip() else ln) for ln in s17.splitlines()))
cells[19]['source'] = gated.splitlines(keepends=True)

# strip inherited outputs/exec counts so 24a starts clean
for _c in nb['cells']:
    if _c['cell_type'] == 'code':
        _c['outputs'] = []
        _c['execution_count'] = None
json.dump(nb, open(DST, 'w'), indent=1, ensure_ascii=False)
print('wrote', DST, '|', len(cells), 'cells')
