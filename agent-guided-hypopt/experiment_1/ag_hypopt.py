"""AG-HYPOPT — Agent-Guided Hyperparameter Optimization.

experiment_1 module (algorithm + trial harness; per instructions.md, algorithm +
harness live in ag_hypopt.py + src/). Sections:
  1. Protocol constants   — 14-ex synthetic benchmark (true params from Gregor's
                            fits), seeds, fixed structural choices, runtime budget
  2. Space + feasibility  — declared space lives in space.json (consumed by the
                            AGHyperopt class); benchmark()/runtime_cap()/feasible()
  3. Algorithm            — AGHyperopt: uncertainty-aware tree-structured Parzen
                            Estimator (worst-case good/bad split, variable-
                            bandwidth KDEs, uniform prior, two phases:
                            uniform draws until n_initial, then trials-based proposals). Template contract:
                                opt = AGHyperopt()
                                opt.fit(SPACE_PATH, TRIALS_PATH)
                                cands = opt.propose_trials(10)   # prints table, returns batch
  4. Harness              — faithful 17g/18c port: synthetic targets at true
                            params, per-photon fit + implicit diff, 2D KDE
                            likelihood, REINFORCE mu-score (sigma_ref) + z-form
                            gamma-score (H_REF), anneal, clip, deterministic
                            seeds. Objective = combined relative MSE of (mu, gamma)
                            + sampling SE (no Fisher).

Interface (algorithm-swappable — future experiment_2 can ship GPBO/CMA-ES):
    AGHyperopt().fit(space, trials).propose_trials(n) -> [{'params','ei',...}]
    run_trial(config) -> results          (deterministic per config, SEED=42)
    compute_objective(results) -> (objective, uncertainty, breakdown)
    format_report(breakdown) -> str
"""
import json
import math
import os
import sys
import time

import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor as _PPE

# ---- path bootstrap: climb to repo root (dir containing src/) ----
_REPO = os.getcwd()
while _REPO != os.path.dirname(_REPO) and not os.path.isdir(os.path.join(_REPO, 'src')):
    _REPO = os.path.dirname(_REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from scipy.stats import norm

torch.set_default_dtype(torch.float32)

from src.fitting import nll, fwhm_from_theta, fit_profile
from src.samplers import draw_fixed_noise
from src.implicit import compute_fwhm_and_dgamma

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



def _budget_feasible(config):
    """Default feasibility for AGHyperopt: enforce the runtime budget."""
    return feasible(config)

def load_space_config(space_config):
    """Normalize space_config (path, full dict, or plain parameter map).

    Returns {'parameters': {...}, 'dependencies': {...}}.
    """
    if space_config is None:
        return {'parameters': None, 'dependencies': {}}
    if isinstance(space_config, (str, os.PathLike)):
        with open(space_config) as f:
            cfg = json.load(f)
    else:
        cfg = dict(space_config)
    if 'parameters' in cfg:
        return {'parameters': dict(cfg['parameters']),
                'dependencies': dict(cfg.get('dependencies', {}))}
    return {'parameters': cfg, 'dependencies': {}}


def load_trials(source):
    """Normalize trials to the AGHyperopt input format.

    Accepts a path to a JSON file (trials.json: config -> params, objective -> loss)
    or a list of dicts already in {'params', 'loss', 'uncertainty'} form.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source) as f:
            data = json.load(f)
        raw = data['trials'] if isinstance(data, dict) and 'trials' in data else data
    else:
        raw = list(source)
    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        params = t.get('params') or t.get('config')
        loss = t.get('loss', t.get('objective'))
        if not params or loss is None:
            continue
        out.append({'params': dict(params), 'loss': float(loss),
                    'uncertainty': float(t.get('uncertainty', 0.0) or 0.0)})
    return out

class _ContinuousModel:
    """Variable-bandwidth KDE + uniform prior over [lo, hi] for one parameter.

    h_base = magic-clipped Scott bandwidth; h_i = h_base * (sigma_i/sigma_med)^beta.
    g(x) = [prior_weight*U(x) + sum_i N(x|mu_i, h_i^2)] / (prior_weight + n).
    """

    def __init__(self, values, uncertainties, lo, hi, int_flag,
                 prior_weight, bandwidth_beta):
        v = np.asarray(values, dtype=float)
        u = np.asarray(uncertainties, dtype=float)
        self.lo, self.hi = float(lo), float(hi)
        self.int_flag = int_flag
        self.n = int(v.size)
        self.prior_weight = prior_weight

        if self.n == 0:
            self.mus = None
            return

        rng_obs = float(np.ptp(v)) if self.n > 1 else 0.0
        if rng_obs >= 1e-12:
            h_min = rng_obs / min(self.n, 100)
            h_max = rng_obs
            if self.n >= 2:
                sig = float(v.std(ddof=1))
                q1, q3 = np.percentile(v, [25, 75])
                iqr = float(q3 - q1)
                scott_sig = min(sig, iqr / 1.34) if iqr > 0 else sig
                h_scott = 1.059 * (self.n ** -0.2) * scott_sig
            else:
                h_scott = h_min
            h_base = min(max(h_scott, h_min), h_max)
        else:
            h_max = (self.hi - self.lo) if self.hi > self.lo else 1.0
            h_base = max(1e-6, 0.05 * h_max)
            h_min = h_scott = h_base

        u_med = float(np.median(u)) if self.n else 0.0
        if u_med >= 1e-12:
            h_i = h_base * np.power(np.maximum(u, 1e-12) / u_med, bandwidth_beta)
        else:
            h_i = np.full(self.n, h_base)
        if h_max > 1e-12:
            h_i = np.minimum(h_i, h_max)
        h_i = np.maximum(h_i, 1e-6)

        self.mus = v
        self.h_i = h_i
        self.h_base = float(h_base)
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.h_scott = float(h_scott)
        self.sigma_med = u_med
        self._denom = prior_weight + self.n
        self._u_val = 1.0 / (self.hi - self.lo) if self.hi > self.lo else 0.0

    def logpdf(self, x):
        if self.mus is None:
            return -float('inf')
        z = (float(x) - self.mus) / self.h_i
        pdfs = norm.pdf(z) / self.h_i
        dens = float(pdfs.sum())
        if self._u_val > 0 and self.lo <= float(x) <= self.hi:
            dens += self.prior_weight * self._u_val
        return float(np.log(dens / self._denom))

    def sample(self, rng):
        if self.mus is None:
            return None
        probs = np.full(self.n + 1, 1.0 / self._denom)
        probs[0] = self.prior_weight / self._denom
        comp = int(rng.choice(self.n + 1, p=probs))
        if comp == 0:
            x = rng.uniform(self.lo, self.hi) if self.hi > self.lo else self.lo
        else:
            x = rng.normal(self.mus[comp - 1], self.h_i[comp - 1])
        if self.int_flag:
            x = int(round(x))
            return int(max(int(self.lo), min(int(self.hi), x)))
        return float(max(self.lo, min(self.hi, x)))


class _CategoricalModel:
    """Laplace-smoothed multinomial over the category universe."""

    def __init__(self, values, categories):
        self.categories = [str(c) for c in categories]
        self.n = len(values)
        if self.n == 0:
            self.probs = None
            return
        counts = np.array([sum(1 for vv in values if str(vv) == c)
                           for c in self.categories], dtype=float)
        self.probs = (counts + 1.0) / (self.n + len(self.categories))

    def logpdf(self, x):
        if self.probs is None:
            return -float('inf')
        x = str(x)
        if x in self.categories:
            return float(np.log(self.probs[self.categories.index(x)]))
        return -float('inf')

    def sample(self, rng):
        if self.probs is None:
            return None
        return str(rng.choice(self.categories, p=self.probs))


class AGHyperopt:
    """Tree-structured Parzen Estimator (TPE), sklearn-style.

    - uncertainty-aware Good/Bad split: loss_adj = loss + lcb_lambda*uncertainty
      (worst-case comparison: a trial is good only if even its upper bound is low)
    - variable-bandwidth KDEs: magic-clipped Scott bandwidth, then uncertainty scaling
    - a uniform prior component in every density (stable EI everywhere + soft exploration)
    - tree-structured (conditional) params from the space dependencies
    - propose_trials returns the batch in draw order (never sorted), so the table does not bias the agent.

    Parameters
    ----------
    n_initial: completed trials needed before switching from uniform random draws
        to trials-based proposals (default 5).
    space: path to JSON or dict. Two accepted shapes:
        {'parameters': {name: {type, low, high | values}}, 'dependencies': {...}}
        or a plain parameters map (dependencies empty).
    feasible: optional callable(config) -> bool; rejects infeasible draws
        (e.g. n_runs*n_iter > runtime_cap).
    """

    def __init__(self, n_initial=5, space=None, quantile=0.25, lcb_lambda=0.5,
                 bandwidth_beta=0.5, prior_weight=1.0, explore_frac=0.1,
                 feasible=None, seed=42):
        self.n_initial = n_initial
        self.quantile = quantile
        self.lcb_lambda = lcb_lambda
        self.bandwidth_beta = bandwidth_beta
        self.prior_weight = prior_weight
        self.explore_frac = explore_frac
        self.seed = seed

        self._set_space(space)
        self.feasible = feasible if feasible is not None else _budget_feasible

        self.history_ = []
        self.good_trials_ = []
        self.bad_trials_ = []
        self.good_models_ = {}
        self.bad_models_ = {}
        self.param_meta_ = {}
        self.params_ = []
        self.is_fitted = False

    def _set_space(self, space):
        cfg = load_space_config(space)
        self.parameters_ = cfg['parameters']
        self.dependencies_ = cfg['dependencies']
        self._validate_dependencies()

    def _validate_dependencies(self):
        deps = self.dependencies_
        for child, spec in deps.items():
            if not isinstance(spec, dict) or 'parent' not in spec or 'parent_value' not in spec:
                raise ValueError(
                    f"dependency for '{child}' must be {{'parent', 'parent_value'}}")
            if spec['parent'] == child:
                raise ValueError(f"self-dependency for '{child}'")

    def _discover_params(self, trials):
        params = set()
        for t in trials:
            params.update(t['params'].keys())
        if self.parameters_:
            params.update(self.parameters_.keys())
        return sorted(params)

    def _param_meta(self, param):
        declared = (self.parameters_ or {}).get(param) or {}
        vals = []
        for t in self.good_trials_ + self.bad_trials_:
            if param in t['params']:
                vals.append(t['params'][param])
        ptype = declared.get('type')
        if ptype is None:
            if vals and all(isinstance(v, str) for v in vals):
                ptype = 'choice'
            elif vals and all(isinstance(v, int) for v in vals):
                ptype = 'int'
            else:
                ptype = 'float'
        if ptype == 'choice':
            values = declared.get('values') or sorted({str(v) for v in vals})
            return {'type': 'choice', 'values': values}
        num = [float(v) for v in vals if isinstance(v, (int, float))]
        lo = declared.get('low') if declared.get('low') is not None else \
            (min(num) if num else 0.0)
        hi = declared.get('high') if declared.get('high') is not None else \
            (max(num) if num else 1.0)
        return {'type': ptype, 'low': float(lo), 'high': float(hi)}

    def _build_model(self, meta, pairs):
        """pairs: list of (value, uncertainty). Returns a model, or None (no trials)."""
        if not pairs:
            return None
        vals = [p[0] for p in pairs]
        if meta['type'] == 'choice':
            return _CategoricalModel(vals, meta['values'])
        uncs = [p[1] for p in pairs]
        return _ContinuousModel(vals, uncs, meta['low'], meta['high'],
                                meta['type'] == 'int', self.prior_weight,
                                self.bandwidth_beta)

    def fit(self, space=None, trials=None):
        """Load space + trials and decide the proposal phase. Returns self.

        Template contract: fit(SPACE_PATH, TRIALS_PATH). Both arguments may be
        paths or preloaded dicts/lists; fit(trials) alone is accepted (space then
        falls back to the constructor's space).

        Phase rule (trials are assumed valid and completed; that check happens
        elsewhere):
            len(trials) <  n_initial  -> phase 'uniform' (random proposals)
            len(trials) >= n_initial  -> phase 'trials'  (proposals from trials)

        In the trials phase the Good/Bad split is uncertainty-aware: the ranking
        uses loss_adj = objective + lcb_lambda * uncertainty.
        """
        if trials is None:
            trials, space = space, None
        if space is not None:
            self._set_space(space)
        trials = load_trials(trials)
        self.history_ = trials
        self.n_completed_ = len(trials)
        self.phase = 'uniform' if len(trials) < self.n_initial else 'trials'
        self.is_fitted = False
        if self.phase != 'trials' or len(trials) < 2:
            return self

        losses = np.array([t['loss'] for t in trials], dtype=float)
        uncs = np.array([t.get('uncertainty', 0.0) for t in trials], dtype=float)
        loss_adj = losses + self.lcb_lambda * uncs
        n = len(trials)
        n_good = int(round(self.quantile * n))
        n_good = max(1, min(n - 1, n_good))
        order = np.argsort(loss_adj, kind='stable')
        good_idx = order[:n_good]
        bad_idx = order[n_good:]

        self.good_trials_ = [trials[i] for i in good_idx.tolist()]
        self.bad_trials_ = [trials[i] for i in bad_idx.tolist()]

        params = self._discover_params(trials)
        self.params_ = params
        self.good_models_, self.bad_models_ = {}, {}
        for param in params:
            meta = self._param_meta(param)
            self.param_meta_[param] = meta
            g_pairs = [(t['params'][param], t.get('uncertainty', 0.0))
                       for t in self.good_trials_ if param in t['params']]
            b_pairs = [(t['params'][param], t.get('uncertainty', 0.0))
                       for t in self.bad_trials_ if param in t['params']]
            self.good_models_[param] = self._build_model(meta, g_pairs)
            self.bad_models_[param] = self._build_model(meta, b_pairs)
        self.is_fitted = True
        return self

    def _sample_order(self):
        """Params sorted so every parent comes before its children (topological)."""
        deps = self.dependencies_
        order = []
        remaining = set(self.params_)
        while remaining:
            ready = [p for p in remaining
                     if p not in deps or deps[p]['parent'] not in remaining]
            if not ready:
                raise ValueError("dependency cycle detected among params")
            order.extend(sorted(ready))
            remaining -= set(ready)
        return order

    def _sample_one(self, rng):
        """Draw one config from the Good model, respecting conditional deps."""
        cfg = {}
        for param in self._sample_order():
            dep = self.dependencies_.get(param)
            if dep is not None and cfg.get(dep['parent']) != dep['parent_value']:
                continue
            model = self.good_models_.get(param)
            if model is None:
                continue
            x = model.sample(rng)
            if x is not None:
                cfg[param] = x
        return cfg

    def _joint_log_density(self, cfg, which):
        """Product of per-param densities over active params; inactive factor = 1."""
        models = self.good_models_ if which == 'g' else self.bad_models_
        logd = 0.0
        for param, x in cfg.items():
            model = models.get(param)
            if model is None:
                continue
            lp = model.logpdf(x)
            if lp == -float('inf'):
                return -float('inf')
            logd += lp
        return logd

    def _score(self, cfg):
        """Return (log_g, log_l, ei) for one config."""
        lg = self._joint_log_density(cfg, 'g')
        ll = self._joint_log_density(cfg, 'l')
        if lg <= -1e15 and ll <= -1e15:
            ratio = 1.0
        elif ll <= -1e15:
            ratio = 0.0
        elif lg <= -1e15:
            ratio = 1e15
        else:
            ratio = float(np.exp(np.clip(ll - lg, -700.0, 700.0)))
        ei = 1.0 / (self.quantile + (1.0 - self.quantile) * ratio)
        return lg, ll, float(ei)

    def _print_table(self, batch):
        """Compact candidate table: ID | ei | params JSON (same order as the list)."""
        phase = getattr(self, 'phase', 'uniform')
        print(f'phase = {phase}  ({self.n_completed_}/{self.n_initial} trials)')
        print(f"{'ID':>2} | {'ei':>6} | params")
        for i, c in enumerate(batch, 1):
            ei = '-' if c.get('ei') is None else f"{c['ei']:.4f}"
            print(f"{i:>2} | {ei:>6} | {json.dumps(c['params'])}")

    def propose_trials(self, n_candidates=10):
        """Propose n_candidates configs for the current phase (template contract).

        uniform phase: independent uniform draws from the declared space
            (ei = None, origin = 'uniform').
        trials phase: draws informed by the fitted trials, uncertainties included
            in the Good/Bad split (ei computed, origin = 'trials').

        The batch is never sorted: it is returned and printed in draw order, so
        the table does not push the agent toward any specific candidate.

        Returns [{'params', 'ei', 'origin'}, ...]  (row N == list[N-1]).
        """
        if getattr(self, 'phase', None) is None:
            raise RuntimeError('call fit(space, trials) before propose_trials()')
        if self.phase == 'uniform':
            batch = self._uniform_propose(n_candidates)
        else:
            batch = self._trials_propose(n_candidates)
        self._print_table(batch)
        return batch

    def _uniform_propose(self, n_candidates):
        """n_candidates independent uniform draws from the declared space."""
        rng = np.random.default_rng()
        out = []
        attempts = 0
        max_attempts = 200 * n_candidates
        while len(out) < n_candidates and attempts < max_attempts:
            attempts += 1
            cfg = self._sample_uniform_declared(rng)
            if self.feasible is not None and not self.feasible(cfg):
                continue
            out.append({'params': cfg, 'ei': None, 'origin': 'uniform'})
        return out

    def _sample_uniform_declared(self, rng):
        """One config sampled uniformly from the declared space (deps-aware)."""
        cfg = {}
        for param in self._declared_order():
            dep = self.dependencies_.get(param)
            if dep is not None and cfg.get(dep['parent']) != dep['parent_value']:
                continue
            spec = (self.parameters_ or {}).get(param)
            if spec is None:
                continue
            if spec.get('type') == 'choice':
                vals = spec.get('values') or []
                if not vals:
                    continue
                cfg[param] = str(rng.choice(list(vals)))
            else:
                lo = float(spec.get('low', 0.0))
                hi = float(spec.get('high', 1.0))
                x = float(rng.uniform(lo, hi)) if hi > lo else lo
                if spec.get('type') == 'int':
                    x = int(round(x))
                    x = int(max(int(lo), min(int(hi), x)))
                cfg[param] = x
        return cfg

    def _declared_order(self):
        """Declared params in dependency order (parents before children)."""
        deps = self.dependencies_
        order = []
        remaining = set((self.parameters_ or {}).keys())
        while remaining:
            ready = [p for p in remaining
                     if p not in deps or deps[p]['parent'] not in remaining]
            if not ready:
                raise ValueError('dependency cycle detected among params')
            order.extend(sorted(ready))
            remaining -= set(ready)
        return order

    def _trials_propose(self, n_candidates):
        """n_candidates draws from the fitted Good model (draw order, no sort)."""
        if not self.is_fitted:
            raise RuntimeError('fit() did not produce densities (trials phase)')
        rng = np.random.default_rng()
        out = []
        attempts = 0
        max_attempts = 200 * n_candidates
        while len(out) < n_candidates and attempts < max_attempts:
            attempts += 1
            cfg = self._sample_one(rng)
            if not cfg:
                continue
            if self.feasible is not None and not self.feasible(cfg):
                continue
            lg, ll, ei = self._score(cfg)
            out.append({'params': cfg, 'ei': float(ei), 'origin': 'trials'})
        return out


AGHyperopt.propose_candidates = AGHyperopt.propose_trials


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
    cfg['n_runs'] = int(cfg['n_runs'])
    cfg['n_iter'] = int(cfg['n_iter'])
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

def format_report(breakdown):
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
    # quick self-test: template API contract + tiny run + objective + cold propose
    opt = AGHyperopt()
    opt.fit('space.json', 'trials.json')
    cands = opt.propose_trials(4)
    assert len(cands) == 4 and all('params' in c for c in cands), 'propose_trials contract broken'

    tiny = [EXPERIMENTS[0]]
    cfg = dict(BASELINE_CONFIG, n_runs=4, n_iter=3)
    run = run_trial(cfg, experiments=tiny, verbose=True)
    obj, unc, bd = compute_objective(run)
    print(f"\nSMOKE objective={obj:.6f} +/- {unc:.6f} | "
          f"mu RMSE {bd['mu']['rmse']:.3f} | gamma RMSE {bd['gamma']['rmse']:.3f}")
    print(format_report(bd))
    print("\nag_hypopt.py self-test OK")
