#!/usr/bin/env python3
"""Build 24s from SRC (env FORK_SRC, default 24r) — Phase C1: empirical data bootstrap.

  * resample the RETAINED real scans (with replacement) B times
  * re-run the FROZEN optimiser on each resample
  * report honest mu/gamma intervals and compare them with the (dataset) CRB
Budget reduction (recorded in the verdict + history): BOOT_EXPS = 8 exps
(1nW/3nW x T05/20/60/100), B = 10 resamples, M_FINAL = 150 (Fisher for the CRB compare).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get('FORK_SRC', '24r')
p = os.path.join(HERE, '24s.ipynb')
if os.path.exists(p):
    sys.exit('24s.ipynb exists')

subprocess.run([sys.executable, os.path.join(HERE, '_fork.py'), '24s', SRC,
                'empirical data bootstrap of the real retained scans (Phase C1)',
                'C1: DO_BOOTSTRAP=True -- resample the retained real scans with replacement B times and '
                're-run the FROZEN optimiser on each resample, reporting honest per-cell mu/gamma intervals '
                'and comparing them with the dataset CRB (M_FINAL reduced 500->150 for the comparison). '
                'Subset: 8 exps (1nW/3nW x T05/20/60/100), B=10 (budget reduction, recorded).'],
               check=True)

nb = json.load(open(p))
cells = nb['cells']
def txt(c): return ''.join(c['source'])
def set_txt(c, s): c['source'] = s.splitlines(keepends=True)

# ---- 1. config: flags + M_FINAL ----
done = False
for c in cells:
    s = txt(c)
    if 'DO_FISHER' in s and 'M_FINAL' in s and 'CFG = dict(' in s:
        s = s.replace("M_FINAL, FISHER_SEEDS, FISHER_BASE_SEED = 500, 5, 7000",
                      "M_FINAL, FISHER_SEEDS, FISHER_BASE_SEED = 150, 1, 7000   # C1: reduced for the bootstrap CRB compare")
        s = s.replace("DO_FISHER  = False     # lean series-24 notebooks: point estimate only (Phase C switches this on)",
                      "DO_FISHER  = False     # C1 is a point-estimate + bootstrap notebook (Fisher only inside the bootstrap cell)\n"
                      "DO_BOOTSTRAP = True    # C1 (24s): empirical data bootstrap of the retained real scans")
        set_txt(c, s); done = True
        break
if not done: sys.exit('config cell not found')

# ---- 2. run_experiment: target override (minimal, additive) ----
done = False
for c in cells:
    s = txt(c)
    a = "    target_f, target_s = _load_real_target(exp)\n"
    if a in s:
        s = s.replace(a,
            "    _ov = cfg.get('_target_override')            # C1 (24s): bootstrap target override\n"
            "    target_f, target_s = _ov if _ov is not None else _load_real_target(exp)\n")
        set_txt(c, s); done = True
        break
if not done: sys.exit('run_experiment target line not found')

# ---- 3. insert the bootstrap cell right after the RUN cell ----
run_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and '23a \u2014 RUN: all 14 real experiments' in txt(c):
        run_idx = i; break
if run_idx is None: sys.exit('RUN cell not found')

boot_src = '''# ============================================================
# SERIES 24 / C1 (24s) \u2014 EMPIRICAL DATA BOOTSTRAP
#   Resample the RETAINED real scans (with replacement), re-run the FROZEN optimiser on each
#   resample, and report an honest per-cell interval for mu/gamma.  Budget reduction recorded
#   in the verdict/history: BOOT_EXPS = 8 exps (T05/20/60/100 x 1nW/3nW), B = 10.
# ============================================================
BOOT_T = [5, 20, 60, 100]
BOOT_EXPS = [e['name'] for e in EXPERIMENTS if int(str(e['name']).split('Trans')[-1]) in BOOT_T]
B_BOOT = 2 if SMOKE else 10
BOOT = {}
if DO_BOOTSTRAP:
    with _PPE(max_workers=N_WORKERS, mp_context=_mp.get_context('fork'), initializer=_init_worker) as _bpool:
        for enm in BOOT_EXPS:
            exp = next(e for e in EXPERIMENTS if e['name'] == enm)
            tf_full, ts_full = _load_real_target(exp)
            N = int(len(tf_full))
            rng = np.random.default_rng(SEED + 31337)
            mus, gams = [], []
            for b in range(B_BOOT):
                idx = torch.tensor(rng.integers(0, N, size=N), dtype=torch.long)
                c2 = dict(CFG); c2['_target_override'] = (tf_full[idx], ts_full[idx])
                rb = run_experiment(exp, c2, _bpool)
                mus.append(rb['mu_final']); gams.append(rb['gamma_final'])
            mus = np.array(mus); gams = np.array(gams)
            rr = next(r for r in RESULTS if r['exp'] == enm)
            fr = _fisher_at(_bpool, rr['mu_final'], rr['gamma_final'], rr['sigma_prop'], rr['lam'],
                            tf_full, ts_full, rr['H_F'], rr['H_S'], CFG)
            BOOT[enm] = dict(mu=[round(float(x), 3) for x in mus], gamma=[round(float(x), 3) for x in gams],
                             mu_lo=float(np.percentile(mus, 2.5)), mu_hi=float(np.percentile(mus, 97.5)),
                             g_lo=float(np.percentile(gams, 2.5)), g_hi=float(np.percentile(gams, 97.5)),
                             mu_sd=float(mus.std()), g_sd=float(gams.std()),
                             crb_mu=float(fr['sig_mu_data']), crb_g=float(fr['sig_g_data']),
                             mu_true=exp['mu_true'], gamma_true=exp['gamma_true'])
            print(f"  {enm:13s} mu_true {exp['mu_true']:7.2f} | boot mu sd {mus.std():6.2f} "
                  f"[{BOOT[enm]['mu_lo']:7.2f},{BOOT[enm]['mu_hi']:7.2f}] | CRB_mu {fr['sig_mu_data']:6.2f} "
                  f"| ratio boot/CRB {mus.std()/max(fr['sig_mu_data'],1e-9):5.2f}", flush=True)
    _bcov = sum(1 for v in BOOT.values() if v['mu_lo'] <= v['mu_true'] <= v['mu_hi'])
    _bcvg = sum(1 for v in BOOT.values() if v['g_lo'] <= v['gamma_true'] <= v['g_hi'])
    _rt = float(np.median([v['mu_sd'] / max(v['crb_mu'], 1e-9) for v in BOOT.values()]))
    print(f"bootstrap 95% coverage (subset N={len(BOOT)}): mu {_bcov}/{len(BOOT)} | gamma {_bcvg}/{len(BOOT)}")
    print(f"median bootstrap-sd / dataset-CRB (mu) = {_rt:.2f}")
    if not SMOKE:
        _bout = os.path.join(REPO_ROOT, 'data', 'processed', '24s_bootstrap.json')
        json.dump(dict(notebook=NB_ID, subset=BOOT_EXPS, B=B_BOOT,
                       coverage_mu=_bcov, coverage_gamma=_bcvg, median_ratio=_rt, boot=BOOT),
                  open(_bout, 'w'))
        print('saved', _bout)
else:
    print('DO_BOOTSTRAP=False - bootstrap block skipped')
'''
boot_cell = dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                 source=boot_src.splitlines(keepends=True))
cells.insert(run_idx + 1, boot_cell)

# ---- 4. hypothesis ----
done = False
for c in cells:
    if c['cell_type'] == 'markdown' and txt(c).startswith('## Hypothesis / prediction'):
        set_txt(c, """## Hypothesis / prediction \u2014 24s (Phase C1)

- **Change vs previous notebook:** add `DO_BOOTSTRAP=True` \u2014 an **empirical data bootstrap**.  For a
  representative subset (`BOOT_EXPS` = 8 exps: 1nW/3nW \u00d7 T05/20/60/100) we resample the **retained real
  scans with replacement** `B=10` times (`B=2` in smoke) and **re-run the frozen optimiser** on each resample
  (`run_experiment` gains a single additive `_target_override` hook; everything else is byte-identical).
  The per-cell `mu_final`/`gamma_final` spreads give an honest interval; `_fisher_at` (M_FINAL reduced
  500\u2192150, 1 seed) gives the **dataset CRB** for the comparison.  Budget reduction recorded.
- **Hypothesis (C1):** the CRB (a Fisher/curvature claim, FM#11/#12) is **bias-blind and too narrow** \u2014 it
  quotes a per-scan `\u03c3` divided by `\u221aN`, but the real fit error is 1.04\u20134.25\u00d7 wider (FM#8) and the
  estimator has a systematic bias.  A **data** bootstrap, which propagates the actual retained-scan
  resampling variability through the whole optimiser, should give intervals that are (a) wider than the
  dataset CRB everywhere and (b) **contain the truth far more often** than the CRB's implied 95 %.
- **Falsifiable prediction:** `median(bootstrap sd / CRB) \u226b 1` (expect \u2265 3\u20135, growing with T), and the 95 %
  bootstrap coverage of `mu_true`/`gamma_true` \u2265 ~6/8 on the subset \u2014 versus the CRB, which (24a/23a) covers
  the truth in \u226a 95 % of the high-T cells.
- **If falsified** (bootstrap \u2248 CRB, i.e. the intervals are already honest): the uncertainty problem is not
  a variance under-estimate \u2192 the residual is pure **bias** (FM#8) \u2192 escalate to the bias-aware profile (C3).
""")
        done = True; break
if not done: sys.exit('hypothesis cell not found')

json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print('built', p, 'from', SRC, f'({len(cells)} cells)')
for c in cells:
    for ln in txt(c).splitlines():
        if 'DO_BOOTSTRAP' in ln or '_target_override' in ln or 'B_BOOT' in ln:
            print('   ', ln.strip())
