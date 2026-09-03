"""AG-HYPOPT — Agent-Guided Hyperparameter Optimization.

sklearn-style optimizer (algorithm only; the trial harness lives in the
experiment template notebook, as it may change per experiment):

    opt = AGHyperopt(space='space.json', feasible=feasible)
    opt.fit('trials.json')          # path to trials.json or list of dicts
                                    #   {'params', 'loss', 'uncertainty'}
    candidates = opt.propose_candidates(n_candidates=10, n_draws=1000)
            -> [{'params', 'ei', 'g_density', 'l_density', 'explore'}]
"""
import json
import os

import numpy as np
from scipy.stats import norm

# ============================================================
# 3. ALGORITHM — AGHyperopt (tree-structured Parzen Estimator, sklearn-style)
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
            x = float(int(round(x)))
            return float(max(int(self.lo), min(int(self.hi), int(x))))
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
    - propose_candidates returns a batch ranked by EI, with reserved fully-uniform slots.

    Parameters
    ----------
    space: path to JSON or dict. Two accepted shapes:
        {'parameters': {name: {type, low, high | values}}, 'dependencies': {...}}
        or a plain parameters map (dependencies empty).
    feasible: optional callable(config) -> bool; rejects infeasible draws
        (e.g. n_runs*n_iter > runtime_cap).
    """

    def __init__(self, space=None, quantile=0.25, lcb_lambda=0.5,
                 bandwidth_beta=0.5, prior_weight=1.0, explore_frac=0.1,
                 feasible=None, seed=42):
        self.quantile = quantile
        self.lcb_lambda = lcb_lambda
        self.bandwidth_beta = bandwidth_beta
        self.prior_weight = prior_weight
        self.explore_frac = explore_frac
        self.feasible = feasible
        self.seed = seed

        cfg = load_space_config(space)
        self.parameters_ = cfg['parameters']
        self.dependencies_ = cfg['dependencies']
        self._validate_dependencies()

        self.history_ = []
        self.good_trials_ = []
        self.bad_trials_ = []
        self.good_models_ = {}
        self.bad_models_ = {}
        self.param_meta_ = {}
        self.params_ = []
        self.is_fitted = False

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

    def fit(self, trials):
        """Build density models from trial history. Returns self.

        trials: path to a JSON file (trials.json) or a list of
        {'params', 'loss', 'uncertainty'} dicts.
        """
        trials = load_trials(trials)
        completed = [t for t in trials if t.get('loss') is not None]
        self.history_ = completed
        self.is_fitted = False
        if len(completed) < 2:
            # too few trials for a Good/Bad split: propose_candidates cold-starts (LHS)
            return self

        losses = np.array([t['loss'] for t in completed], dtype=float)
        uncs = np.array([t.get('uncertainty', 0.0) for t in completed], dtype=float)
        loss_adj = losses + self.lcb_lambda * uncs
        n = len(completed)
        n_good = int(round(self.quantile * n))
        n_good = max(1, min(n - 1, n_good))
        order = np.argsort(loss_adj, kind='stable')
        good_idx = order[:n_good]
        bad_idx = order[n_good:]

        self.good_trials_ = [completed[i] for i in good_idx.tolist()]
        self.bad_trials_ = [completed[i] for i in bad_idx.tolist()]

        params = self._discover_params(completed)
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

    def _sample_uniform_cfg(self, rng):
        """Fully-uniform config over declared/observed bounds (reserved exploration)."""
        cfg = {}
        for param in self._sample_order():
            dep = self.dependencies_.get(param)
            if dep is not None and cfg.get(dep['parent']) != dep['parent_value']:
                continue
            meta = self.param_meta_.get(param)
            if meta is None:
                continue
            if meta['type'] == 'choice':
                if not meta['values']:
                    continue
                cfg[param] = str(rng.choice(meta['values']))
            else:
                lo, hi = meta['low'], meta['high']
                x = rng.uniform(lo, hi) if hi > lo else lo
                if meta['type'] == 'int':
                    x = float(int(round(x)))
                    x = float(max(int(lo), min(int(hi), x)))
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

    def propose_candidates(self, n_candidates=10, n_draws=1000):
        """Propose a batch of candidate configs.

        Cold start (fewer than COLD_START_MIN_TRIALS completed trials): LHS spread.
        Otherwise: draw from the Good model, rank by EI, with reserved explore slots.
        Returns [{'params', 'ei', 'g_density', 'l_density', 'explore'}] sorted by ei desc.
        """
        if len(self.history_) < COLD_START_MIN_TRIALS:
            return self._cold_start(n_candidates)
        if not self.is_fitted:
            raise RuntimeError("fit() did not produce densities")
        if n_draws < n_candidates:
            raise ValueError("n_draws must be >= n_candidates")
        rng = np.random.default_rng(self.seed)

        pool = []
        attempts = 0
        max_attempts = 50 * n_draws
        while len(pool) < n_draws and attempts < max_attempts:
            attempts += 1
            cfg = self._sample_one(rng)
            if self.feasible is not None and not self.feasible(cfg):
                continue
            pool.append(cfg)

        scored = []
        for cfg in pool:
            lg, ll, ei = self._score(cfg)
            scored.append({'params': cfg, 'ei': ei,
                           'g_density': float(np.exp(np.clip(lg, -700.0, 700.0))),
                           'l_density': float(np.exp(np.clip(ll, -700.0, 700.0))),
                           'explore': False})
        scored.sort(key=lambda c: c['ei'], reverse=True)

        n_explore = max(1, int(round(self.explore_frac * n_candidates)))
        n_explore = min(n_explore, n_candidates)
        batch = scored[:n_candidates - n_explore]

        reserved = []
        attempts = 0
        while len(reserved) < n_explore and attempts < 50 * n_explore:
            attempts += 1
            cfg = self._sample_uniform_cfg(rng)
            if self.feasible is not None and not self.feasible(cfg):
                continue
            lg, ll, ei = self._score(cfg)
            reserved.append({'params': cfg, 'ei': ei,
                             'g_density': float(np.exp(np.clip(lg, -700.0, 700.0))),
                             'l_density': float(np.exp(np.clip(ll, -700.0, 700.0))),
                             'explore': True})
        if len(reserved) < n_explore:
            for c in reversed(scored):
                if len(reserved) >= n_explore:
                    break
                c['explore'] = True
                reserved.append(c)

        batch = batch + reserved
        batch.sort(key=lambda c: c['ei'], reverse=True)
        return batch

    def _cold_start(self, n_candidates):
        """LHS spread over the declared space (used until enough trials exist)."""
        if self.parameters_ is None:
            raise ValueError("space required for the cold-start phase")
        rng = np.random.default_rng(self.seed)
        out = []
        attempts = 0
        max_attempts = 200 * n_candidates
        while len(out) < n_candidates and attempts < max_attempts:
            attempts += 1
            c = _lhs_sample(self.parameters_, 1, rng)[0]
            if self.feasible is not None and not self.feasible(c):
                continue
            score = 0.5 + 0.4 * (len(out) + 1) / n_candidates + 0.1 * rng.uniform()
            out.append({'params': c, 'ei': float(min(score, 1.0)),
                        'g_density': None, 'l_density': None, 'explore': False})
        return out


COLD_START_MIN_TRIALS = 4
