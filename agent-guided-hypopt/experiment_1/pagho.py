"""PAGHO — Physics-informed Agent-Guided Hyperparameter Optimization.

experiment_1 — REAL implementation (2026-08-30). Replaces the 2026-08-29 draft
(uniform-random propose stub, NotImplementedError harness).

One module (Anuar's convention: ALL scripts in a single .py per experiment).
Sections:
  1. Protocol constants      — benchmark, seeds, fixed structural choices, runtime budget
  2. Search space            — DEFAULT_SPACE + feasibility cap
  3. Algorithm (propose)     — real TPE + EI (median split, gaussian KDEs, density-ratio
                               acquisition, LHS cold start)
  4. Harness (run/compute)   — faithful port of the 17g/18c notebook machinery:
                               synthetic targets at true params, per-photon fit +
                               implicit diff, 2D KDE likelihood, REINFORCE μ-score
                               (σ_ref) + z-form γ-score (H_REF), anneal, clip,
                               deterministic seeds. Objective = combined relative MSE
                               of (μ, γ) + sampling SE (Anuar 2026-08-29 11:15).

Interface contract (algorithm-swappable — future experiment_2 can ship GPBO/CMA-ES):
    propose(trials, space, n_candidates=10, seed=None) -> [{"config", "score"}]
      trials: list of dicts from trials.json (trial_id, config, objective, uncertainty)
      score  = TPE acquisition (log density ratio good/bad); the AGENT considers it
               and decides with physics reasoning on top.
    run_trial(config) -> results (per-experiment dicts)
    compute_objective(results) -> (objective, uncertainty, breakdown)
"""
import math
import os
import sys
import time

# ---- path bootstrap: climb to repo root (dir containing src/) ----
_REPO = os.getcwd()
while _REPO != os.path.dirname(_REPO) and not os.path.isdir(os.path.join(_REPO, 'src')):
    _REPO = os.path.dirname(_REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from scipy.stats import gaussian_kde

torch.set_default_dtype(torch.float32)

from src.fitting import nll, fwhm_from_theta, fit_profile
from src.samplers import draw_fixed_noise
from src.implicit import compute_fwhm_and_dgamma

# ============================================================
# 1. PROTOCOL — fixed for the campaign (lives here; changes need Anuar's OK)
# ============================================================
# Benchmark: 14 synthetic experiments, true params from Gregor's fits (16-series).
# Targets are generated at TRUE values with SYNTH_SEED -> identical targets across trials.
EXPERIMENTS = [
    dict(name='1nW Trans05',  power='1nW', mu_true=9.393,   sigma_prop=2.576,  lam=2.232, gamma_true=8.5,  n_target=61),
    dict(name='1nW Trans10',  power='1nW', mu_true=12.372,  sigma_prop=3.445,  lam=2.122, gamma_true=8.5,  n_target=358),
    dict(name='1nW Trans20',  power='1nW', mu_true=17.316,  sigma_prop=4.141,  lam=2.286, gamma_true=8.5,  n_target=1138),
    dict(name='1nW Trans40',  power='1nW', mu_true=38.405,  sigma_prop=7.198,  lam=2.351, gamma_true=8.5,  n_target=2428),
    dict(name='1nW Trans60',  power='1nW', mu_true=61.374,  sigma_prop=9.851,  lam=2.593, gamma_true=8.5,  n_target=2424),
    dict(name='1nW Trans80',  power='1nW', mu_true=79.365,  sigma_prop=12.627, lam=2.758, gamma_true=8.5,  n_target=2487),
    dict(name='1nW Trans100', power='1nW', mu_true=70.817,  sigma_prop=17.221, lam=2.636, gamma_true=8.5,  n_target=2455),
    dict(name='3nW Trans05',  power='3nW', mu_true=13.204,  sigma_prop=3.724,  lam=2.186, gamma_true=14.1, n_target=252),
    dict(name='3nW Trans10',  power='3nW', mu_true=24.476,  sigma_prop=5.639,  lam=2.158, gamma_true=14.1, n_target=1572),
    dict(name='3nW Trans20',  power='3nW', mu_true=34.279,  sigma_prop=8.319,  lam=2.264, gamma_true=14.1, n_target=2171),
    dict(name='3nW Trans40',  power='3nW', mu_true=84.892,  sigma_prop=24.013, lam=2.475, gamma_true=14.1, n_target=3742),
    dict(name='3nW Trans60',  power='3nW', mu_true=103.203, sigma_prop=23.95,  lam=2.741, gamma_true=14.1, n_target=2541),
    dict(name='3nW Trans80',  power='3nW', mu_true=137.537, sigma_prop=32.107, lam=2.911, gamma_true=14.1, n_target=2508),
    dict(name='3nW Trans100', power='3nW', mu_true=175.707, sigma_prop=40.975, lam=3.087, gamma_true=14.1, n_target=2516),
]

# Reduced benchmark option (open decision): None = full 14; or a list of names.
# Kept as protocol knob so trial_00 can flip it without touching code.
BENCHMARK_SUBSET = None

SYNTH_SEED = 12345    # target generation seed (identical targets every trial)
SEED = 42             # per-step noise seed base (deterministic runs)

# Fixed structural choices (17g/18c verdicts — NOT tunable this campaign):
GAMMA_SCALE = True    # z-form γ-score (bandwidth-normalized, fixed reference)
H_REF = 1.0           # fixed reference bandwidth for the γ-score
LAMBDA_MEAN = 0.0     # mean-matching anchor weight (0.0 = disabled)

# Parallelism (measured: 4 workers @ 17f speed ≈ 160k fit-calls/hour)
N_WORKERS = 4
N_EXP_PARALLEL = 1

# Runtime budget: must make BASELINE_CONFIG (200x200=40k, ~3.5h at measured speed) feasible.
# Full 14-exp benchmark at 17f speed: 40k inner steps x 14 exps / 4 workers ~= 3.5h (18c measured).
# BUDGET_HOURS is a protocol decision — lowering it requires a reduced benchmark or smaller space.
BUDGET_HOURS = 3.5
CALLS_PER_HOUR = 160_000   # total fit calls across all workers, measured 2026-08-29 (18c)

# Baseline config (17g = current best on synthetic): what every trial must beat.
BASELINE_CONFIG = dict(
    n_runs=200, n_iter=200, lr_mu=15.0, lr_gamma=0.5, sigma_ref=10.0,
    clip=10.0, gamma_anneal=0.5, h_s_min=0.05,
)

# ============================================================
# 2. SEARCH SPACE + FEASIBILITY
# ============================================================
DEFAULT_SPACE = {
    "n_runs":       {"type": "int",   "low": 100, "high": 500},   # runtime driver
    "n_iter":       {"type": "int",   "low": 100, "high": 400},   # runtime driver
    "lr_mu":        {"type": "float", "low": 5.0,  "high": 40.0},
    "lr_gamma":     {"type": "float", "low": 0.1,  "high": 1.5},
    "sigma_ref":    {"type": "float", "low": 5.0,  "high": 25.0},
    "clip":         {"type": "float", "low": 5.0,  "high": 20.0},
    "gamma_anneal": {"type": "float", "low": 0.0,  "high": 0.75},
    "h_s_min":      {"type": "float", "low": 0.0,  "high": 0.2},
}

def benchmark():
    """Effective benchmark list (full 14 or the configured subset)."""
    if BENCHMARK_SUBSET is None:
        return EXPERIMENTS
    return [e for e in EXPERIMENTS if e['name'] in BENCHMARK_SUBSET]

def runtime_cap(n_exps=None):
    """Max n_runs*n_iter for a ~BUDGET_HOURS trial on the effective benchmark."""
    n_exps = n_exps or len(benchmark())
    return int(BUDGET_HOURS * CALLS_PER_HOUR / n_exps)

def feasible(config, cap=None):
    return config.get('n_runs', 0) * config.get('n_iter', 0) <= (cap if cap is not None else runtime_cap())

# ============================================================
# 3. ALGORITHM — real TPE + EI
# ============================================================
def _lhs_sample(space, n, rng):
    """Latin hypercube spread over the space (cold start)."""
    out = []
    for _ in range(n):
        c = {}
        for k, s in space.items():
            u = (rng.permutation(n)[len(out)] + rng.uniform(0, 1)) / n
            if s["type"] == "int":
                lo, hi = s["low"], s["high"]
                c[k] = int(round(lo + u * (hi - lo)))
            elif s["type"] == "float":
                c[k] = float(s["low"] + u * (s["high"] - s["low"]))
            elif s["type"] == "choice":
                c[k] = str(rng.choice(s["values"]))
            else:
                raise ValueError(f"unknown type {s['type']} for {k}")
        out.append(c)
    return out

def _kde_or_uniform(values, rng):
    """gaussian_kde with degenerate fallback (all-equal or tiny spread)."""
    v = np.asarray(values, dtype=float)
    if v.size < 2 or np.ptp(v) < 1e-12:
        return None  # caller falls back to uniform
    try:
        return gaussian_kde(v)
    except Exception:
        return None

def propose(trials, space=None, n_candidates=10, seed=None):
    """Real TPE + EI.

    - cold start (fewer than 3 completed trials): LHS spread, exploration scores
    - otherwise: median split good/bad by objective; per-parameter gaussian KDEs on
      good vs bad trials; candidates drawn from the good KDE (truncated to bounds);
      acquisition score = log density ratio  sum_k log(l_k(x_k)/g_k(x_k))  (choice
      params: log(p_good/p_bad)); infeasible configs (n_runs*n_iter > cap) rejected.
      Scores min-max normalized to [0,1]; higher = more promising.
    """
    space = space or DEFAULT_SPACE
    rng = np.random.default_rng(seed)
    cap = runtime_cap()
    completed = [t for t in trials if t.get('objective') is not None and t.get('config')]

    if len(completed) < 3:
        cands = _lhs_sample(space, n_candidates, rng)
        out = []
        for i, c in enumerate(cands):
            if not feasible(c, cap):
                continue
            # exploration bonus: spread + jitter, higher for farther-from-visited configs
            score = 0.5 + 0.4 * (i + 1) / n_candidates + 0.1 * rng.uniform()
            out.append({"config": c, "score": float(min(score, 1.0))})
        return out

    objs = np.array([t['objective'] for t in completed])
    med = np.median(objs)
    good = [t for t in completed if t['objective'] <= med]
    bad = [t for t in completed if t['objective'] > med]

    # per-parameter densities
    kdes, freqs = {}, {}
    for k, s in space.items():
        gv = [t['config'].get(k) for t in good if k in t['config']]
        bv = [t['config'].get(k) for t in bad if k in t['config']]
        if s["type"] == "choice":
            vals = s["values"]
            pg = np.array([gv.count(v) for v in vals]) + 1e-6
            pb = np.array([bv.count(v) for v in vals]) + 1e-6
            freqs[k] = (pg / pg.sum(), pb / pb.sum())
        else:
            kdes[k] = (_kde_or_uniform(gv, rng), _kde_or_uniform(bv, rng))

    def draw_one(k, s, rng):
        if s["type"] == "choice":
            pg, pb = freqs[k]
            return str(rng.choice(s["values"], p=pg)), pg, pb
        kg, kb = kdes[k]
        lo, hi = s["low"], s["high"]
        if kg is None:
            x = rng.uniform(lo, hi)
        else:
            x = float(kg.resample(1)[0, 0])
            tries = 0
            while not (lo <= x <= hi) and tries < 50:
                x = float(kg.resample(1)[0, 0])
                tries += 1
            if not (lo <= x <= hi):
                x = rng.uniform(lo, hi)
        if s["type"] == "int":
            x = int(round(x))
        lg = math.log(float(kg(x)[0]) + 1e-30) if kg is not None else 0.0
        lb = math.log(float(kb(x)[0]) + 1e-30) if kb is not None else 0.0
        return x, lg, lb

    out = []
    attempts = 0
    while len(out) < n_candidates and attempts < 200 * n_candidates:
        attempts += 1
        c, log_ratio = {}, 0.0
        for k, s in space.items():
            x, lg, lb = draw_one(k, s, rng)
            c[k] = x
            log_ratio += lg - lb
        if not feasible(c, cap):
            continue
        out.append({"config": c, "score": float(log_ratio)})

    if not out:
        return []

    scores = np.array([o['score'] for o in out])
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        scores = np.full_like(scores, 0.5)
    else:
        scores = (scores - lo) / (hi - lo)
    for o, s in zip(out, scores):
        o['score'] = float(s)
    out.sort(key=lambda o: o['score'], reverse=True)
    return out

# ============================================================
# 4. HARNESS — run one trial + objective
# ============================================================
def _kde_scores(sim_f, sim_s, sim_n, sim_df, sim_ds, data_f, data_s, h_f, h_s, mu, sigma_prop, cfg):
    """2D KDE negative log-likelihood + per-data-point scores (identical to 17g)."""
    d_f = data_f[:, None] - sim_f[None, :]
    d_s = data_s[:, None] - sim_s[None, :]
    W = torch.exp(-0.5 * (d_f / h_f) ** 2 - 0.5 * (d_s / h_s) ** 2)
    w = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)
    score = (sim_n[None, :] - mu) / sigma_prop ** 2
    s_mu = (w * score).sum(dim=1)
    if GAMMA_SCALE:
        # z-form: ONE power of bandwidth (unitless distances), fixed reference.
        dlogG = ((d_f * sim_df[None, :]) / h_f + (d_s * sim_ds[None, :]) / h_s) / H_REF
    else:
        dlogG = (d_f * sim_df[None, :]) / h_f ** 2 + (d_s * sim_ds[None, :]) / h_s ** 2
    s_gamma = (w * dlogG).sum(dim=1)
    logp = torch.log((W.sum(dim=1) / len(sim_f)).clamp_min(1e-30))
    nll_val = -logp.mean()
    return s_mu, s_gamma, nll_val, w

def _fit_fn(ph):
    return fit_profile(ph, n_iters=80, model='lorentzian', uniform_bg=False)

def _fwhm_fn(th):
    return fwhm_from_theta(th, model='lorentzian')

def _nll_fn(th, ph):
    return nll(th, ph, model='lorentzian', uniform_bg=False)

import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor as _PPE

def _run_one(args):
    gamma_val, u, b = args
    return compute_fwhm_and_dgamma(gamma_val, u, b, _fit_fn, _fwhm_fn, _nll_fn, n_params=2)

def _init_worker():
    torch.set_num_threads(1)

def _parallel_map(pool, tasks):
    return list(pool.map(_run_one, tasks, chunksize=8))

def _run_experiment(exp, cfg, pool):
    """One experiment: joint μ + γ optimization (17g machinery, config-driven)."""
    mu_true, sigma_prop = exp['mu_true'], exp['sigma_prop']
    lam, gamma_true = exp['lam'], exp['gamma_true']
    mu_init, gamma_init = 0.5 * mu_true, 0.5 * gamma_true
    n_runs = cfg['n_runs']; n_iter = cfg['n_iter']
    lr_mu = cfg['lr_mu']; lr_gamma = cfg['lr_gamma']
    sigma_ref = cfg['sigma_ref']; clip = cfg['clip']
    gamma_anneal = cfg['gamma_anneal']; h_s_min = cfg['h_s_min']

    # ---- synthetic target at TRUE values (same seed as 16-series) ----
    rng_t = np.random.default_rng(SYNTH_SEED)
    tasks_t = []
    for _ in range(exp['n_target']):
        u, b, n = draw_fixed_noise(mu_true, sigma_prop, lam, rng_t)
        tasks_t.append((gamma_true, u.numpy(), b.numpy()))
    res_t = _parallel_map(pool, tasks_t)
    target_f = torch.tensor([r[0] for r in res_t], dtype=torch.float32)
    target_s = torch.tensor([r[1] for r in res_t], dtype=torch.float32)
    scott = exp['n_target'] ** (-1.0 / 6.0)
    H_F = float(target_f.std()) * scott
    H_S = max(float(target_s.std()) * scott, h_s_min)

    mu_val = float(mu_init)
    gamma_val = float(gamma_init)
    history = []
    t_start = time.time()

    for step in range(n_iter):
        rng2 = np.random.default_rng(SEED + step)
        tasks, ns = [], []
        for _ in range(n_runs):
            u, b, n = draw_fixed_noise(mu_val, sigma_prop, lam, rng2)
            tasks.append((gamma_val, u.numpy(), b.numpy()))
            ns.append(n)
        res = _parallel_map(pool, tasks)
        fwhms = [r[0] for r in res]; sigmas = [r[1] for r in res]
        dfs = [r[2] for r in res]; dsigmas = [r[3] for r in res]

        ft = torch.tensor(fwhms, dtype=torch.float32)
        si_t = torch.tensor(sigmas, dtype=torch.float32)
        nt = torch.tensor(ns, dtype=torch.float32)
        dg_t = torch.tensor(dfs, dtype=torch.float32)
        ds_t = torch.tensor(dsigmas, dtype=torch.float32)

        s_mu, s_gamma, nll_val, w = _kde_scores(
            ft, si_t, nt, dg_t, ds_t, target_f, target_s, H_F, H_S, mu_val, sigma_prop, cfg)

        # ---- μ: REINFORCE (σ_ref score, self-normalized, scale-invariant) ----
        B = w.mean(dim=0)
        score = (nt - mu_val) / sigma_ref ** 2
        grad_mu = float(max(min(-(B - B.mean()) @ score, clip), -clip))

        # ---- γ: KDE channel (z-form) ----
        grad_gamma = float(max(min(-s_gamma.mean(), clip), -clip))

        # ---- mean-matching anchor (kernel-free; disabled unless LAMBDA_MEAN > 0) ----
        mean_sim = float(ft.mean()); mean_tgt = float(target_f.mean())
        mean_loss = abs(mean_sim - mean_tgt)
        if LAMBDA_MEAN > 0:
            d_mean_dgamma = float(dg_t.mean())
            grad_gamma_mean = -math.copysign(1.0, mean_sim - mean_tgt) * d_mean_dgamma
            grad_gamma = float(max(min(grad_gamma + LAMBDA_MEAN * grad_gamma_mean, clip), -clip))

        # ---- updates (17f schedules: μ linear decay, no floor; γ anneal) ----
        lr_mu_decay = lr_mu * (1.0 - step / n_iter)
        mu_val -= lr_mu_decay * grad_mu
        mu_val = max(1.0, min(200.0, mu_val))
        gamma_val -= lr_gamma * (1.0 - gamma_anneal * step / n_iter) * grad_gamma
        gamma_val = max(0.1, min(100.0, gamma_val))

        history.append(dict(step=step, mu=mu_val, gamma=gamma_val, nll=float(nll_val),
                            grad_mu=grad_mu, grad_gamma=grad_gamma, mean_loss=mean_loss,
                            mean_n=float(nt.mean())))

    return dict(exp=exp['name'], power=exp['power'],
                mu_true=mu_true, gamma_true=gamma_true,
                mu_final=history[-1]['mu'], gamma_final=history[-1]['gamma'],
                nll_final=history[-1]['nll'], mean_loss_final=history[-1]['mean_loss'],
                history=history, H_F=H_F, H_S=H_S, t_elapsed=time.time() - t_start)

def run_trial(config, experiments=None, verbose=True):
    """Run one trial on the benchmark. Returns dict with per-experiment results.

    config keys (all optional, defaults from BASELINE_CONFIG):
        n_runs, n_iter, lr_mu, lr_gamma, sigma_ref, clip, gamma_anneal, h_s_min
    Deterministic per (config, benchmark): SEED + SYNTH_SEED fixed.
    """
    cfg = dict(BASELINE_CONFIG)
    cfg.update({k: v for k, v in config.items() if v is not None})
    exps = experiments if experiments is not None else benchmark()
    if not feasible(cfg):
        raise ValueError(
            f"infeasible config: n_runs*n_iter={cfg['n_runs']*cfg['n_iter']} "
            f"> runtime_cap={runtime_cap(len(exps))} (~{BUDGET_HOURS}h trial)")

    results = []
    t0 = time.time()
    with _PPE(max_workers=N_WORKERS, mp_context=_mp.get_context('fork'),
              initializer=_init_worker) as pool:
        for exp in exps:
            if verbose:
                print(f"Running {exp['name']:>12} ...", end=' ', flush=True)
            r = _run_experiment(exp, cfg, pool)
            r['t_elapsed'] = time.time() - t0
            results.append(r)
            if verbose:
                print(f"μ {r['mu_true']:.2f} -> {r['mu_final']:.2f} | "
                      f"γ {r['gamma_true']:.1f} -> {r['gamma_final']:.2f} | "
                      f"NLL {r['nll_final']:.2f}", flush=True)
    total = time.time() - t0
    if verbose:
        print(f"\nTotal: {total/60:.1f} min")
    return dict(config=cfg, benchmark=[e['name'] for e in exps],
                results=results, t_elapsed=total)

def compute_objective(run):
    """Objective = combined relative MSE of (μ, γ) over the benchmark + sampling SE.

    Per experiment: rel-sq error for μ and γ (each normalized by its true value).
    objective = mean over all 2*n_exps errors (lower is better).
    uncertainty = SE of those per-experiment errors (sampling uncertainty across
    experiments — Anuar 2026-08-29 11:15; NO Fisher in the objective).
    Returns (objective, uncertainty, breakdown) — breakdown carries the full
    per-channel MSE/RMSE/rel-RMSE/bias table for reporting.
    """
    results = run['results'] if isinstance(run, dict) else run
    rows = []
    for r in results:
        rel_mu = ((r['mu_final'] - r['mu_true']) / r['mu_true']) ** 2
        rel_g = ((r['gamma_final'] - r['gamma_true']) / r['gamma_true']) ** 2
        rows.append(dict(exp=r['exp'], rel_sq_mu=rel_mu, rel_sq_gamma=rel_g,
                         mu_err=r['mu_final'] - r['mu_true'],
                         gamma_err=r['gamma_final'] - r['gamma_true']))
    errs = np.array([v for row in rows for v in (row['rel_sq_mu'], row['rel_sq_gamma'])])
    objective = float(errs.mean())
    uncertainty = float(errs.std(ddof=1) / np.sqrt(len(errs))) if len(errs) > 1 else 0.0

    mu_e = np.array([row['mu_err'] for row in rows])
    g_e = np.array([row['gamma_err'] for row in rows])
    breakdown = dict(
        n_exps=len(results),
        mu=dict(bias=float(mu_e.mean()), mse=float((mu_e ** 2).mean()),
                rmse=float(np.sqrt((mu_e ** 2).mean())),
                rel_rmse=float(np.sqrt(((mu_e / np.array([r['mu_true'] for r in results])) ** 2).mean()))),
        gamma=dict(bias=float(g_e.mean()), mse=float((g_e ** 2).mean()),
                   rmse=float(np.sqrt((g_e ** 2).mean())),
                   rel_rmse=float(np.sqrt(((g_e / np.array([r['gamma_true'] for r in results])) ** 2).mean()))),
        rows=rows,
    )
    return objective, uncertainty, breakdown

def format_results(breakdown):
    """Compact table for the trial notebook / Telegram (no pandas needed)."""
    lines = [f"{'exp':<12} {'μ_true':>8} {'μ_final':>8} {'Δμ':>8} | "
             f"{'γ_true':>6} {'γ_final':>8} {'Δγ':>8}"]
    for r in breakdown['rows']:
        lines.append(f"{r['exp']:<12} ...")  # detailed rows live in the notebook
    m, g = breakdown['mu'], breakdown['gamma']
    lines.append(f"μ: RMSE {m['rmse']:.3f} (rel {m['rel_rmse']*100:.1f}%) | "
                 f"γ: RMSE {g['rmse']:.3f} (rel {g['rel_rmse']*100:.1f}%)")
    return "\n".join(lines)

if __name__ == '__main__':
    # quick self-test: tiny run on 1 experiment + objective + propose
    import json
    tiny = [EXPERIMENTS[0]]
    cfg = dict(BASELINE_CONFIG, n_runs=4, n_iter=3)
    run = run_trial(cfg, experiments=tiny, verbose=True)
    obj, unc, bd = compute_objective(run)
    print(f"\nSMOKE objective={obj:.6f} ± {unc:.6f} | μ RMSE {bd['mu']['rmse']:.3f} γ RMSE {bd['gamma']['rmse']:.3f}")
    print("propose cold start:", json.dumps(propose([], n_candidates=4, seed=1), indent=1)[:400])
    print("propose warm:", json.dumps(propose([{"config": dict(BASELINE_CONFIG, n_runs=100), "objective": 0.05},
                                                {"config": dict(BASELINE_CONFIG, n_runs=150), "objective": 0.12},
                                                {"config": dict(BASELINE_CONFIG, n_runs=200), "objective": 0.03}],
                                               n_candidates=4, seed=2), indent=1)[:400])
