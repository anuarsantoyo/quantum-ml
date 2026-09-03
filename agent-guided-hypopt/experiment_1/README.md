# AG-HYPOPT optimizer: math reference (experiment_1)

This document explains, step by step, the mathematics implemented in
`ag_hypopt.py` (`AGHyperopt`): what `fit()` computes and how
`propose_trials(n)` turns that into candidate configurations. It is a
reference for the agent and for anyone reading the code. It documents the
code as it stands on branch `ag-hypopt-clean`.

## 1. Inputs and notation

**Search space.** The space is a set of declared parameters. Each continuous or
integer parameter carries a box `[lo, hi]`; each categorical parameter carries a
finite set of values. A dependency `child -> {parent, parent_value}` makes the
child conditional: it is only active when `parent == parent_value`.

**Trials.** The history is a list of completed trials. Trial `i` is a triplet

$$
(x_i,\; \ell_i,\; s_i)
$$

where `x_i` is the parameter vector that was run, `\ell_i` is its objective
(stored as `loss` / `objective`; lower is better), and `s_i` is the reported
uncertainty of that objective. Trials are assumed valid and completed before
`AGHyperopt` sees them.

**Algorithm constants** (constructor defaults in parentheses):

| symbol | code name | default | meaning |
|---|---|---|---|
| $n_{\text{init}}$ | `n_initial` | 5 | completed trials needed before switching from uniform to trials-based proposals |
| $q$ | `quantile` | 0.25 | fraction of trials treated as "good" in the Good/Bad split |
| $\lambda$ | `lcb_lambda` | 0.5 | weight of the uncertainty in the adjusted loss |
| $\beta$ | `bandwidth_beta` | 0.5 | exponent for uncertainty-scaled kernel bandwidths |
| $w_0$ | `prior_weight` | 1.0 | weight of the uniform prior component in each density |
| $s_{\text{explore}}$ | `explore_slots` | 2 | fully-uniform (entirely random) candidate slots reserved in every trials-phase batch |
| $\alpha$ | (Laplace) | 1 | Laplace smoothing count for categorical models |

## 2. The two phases

`AGHyperopt` has exactly two phases, decided by `fit()` from one number:

$$
\text{phase} =
\begin{cases}
\text{uniform}, & n < n_{\text{init}} \\
\text{trials},   & n \ge n_{\text{init}}
\end{cases}
\qquad n = \text{number of completed trials}
$$

Until `n_initial` trials exist there is not enough history to build densities, so
candidates are drawn uniformly from the declared space. Once enough trials
exist, proposals are generated from density models fitted on those trials.

## 3. `fit(space, trials)` step by step

### 3.1 Loading

`fit(space, trials)` normalizes both arguments. Trials are read with
`load_trials`: each entry keeps only its `params` (key `config` is accepted as
an alias), its `objective`/`loss`, and its `uncertainty` (default 0). Entries
without parameters or without an objective are dropped, because every trial
that reaches the optimizer is required to be complete.

### 3.2 Phase decision

The optimizer stores

$$
n_{\text{completed}} = n, \qquad \text{phase} \in \{\text{uniform},\;\text{trials}\}
$$

and exits early in the uniform phase: no densities are built there.

### 3.3 Trials phase: uncertainty-aware Good/Bad split

The first thing the trials phase does is rank trials. A trial with a good
objective but a large uncertainty is not as informative as one with the same
objective and a small uncertainty, so the ranking uses an adjusted loss

$$
\tilde{\ell}_i = \ell_i + \lambda\, s_i .
$$

This is a worst-case (upper-bound) comparison: a trial is called "good" only if
even its upper confidence bound is low. With $\lambda = 0.5$, half of each
trial's uncertainty is added before comparing.

Trials are sorted by $\tilde{\ell}$ ascending (stable sort), and the lowest
quantile becomes the good set:

$$
m = \text{clamp}\big(\text{round}(q\, n),\; 1,\; n-1\big),
\qquad
G = \text{first } m \text{ trials},
\qquad
B = \text{remaining } n - m \text{ trials}.
$$

With $q = 0.25$ and $n = 6$, for example, $m = \text{round}(1.5) = 2$: the two
lowest-adjusted-loss trials form the good set, the other four the bad set.

### 3.4 Parameter metadata

For every parameter (the union of declared parameters and parameters observed in
the trials), `fit` determines a metadata record:

- **type**: declared, or inferred from the observed values (all strings ->
  `choice`, all ints -> `int`, otherwise `float`);
- **range**: declared `[lo, hi]`, else `[min, max]` of the observed values,
  else `[0, 1]`;
- **categorical values**: declared list, else the sorted set of observed
  strings.

### 3.5 Per-side density models

For each parameter and for each side (good and bad), a one-dimensional model is
built from the observed `(value, uncertainty)` pairs of that side.

#### Continuous and integer parameters: variable-bandwidth KDE with prior

Given $k$ observed values $v_1,\dots,v_k$ with uncertainties $u_1,\dots,u_k$:

1. **Observed spread.** $R = \max(v) - \min(v)$ (0 if $k = 1$).
2. **Bandwidth floor and ceiling.** If $R \ge 10^{-12}$:

   $$
   h_{\min} = \frac{R}{\min(k,\,100)},\qquad h_{\max} = R.
   $$

3. **Magic-clipped Scott bandwidth.** If $k \ge 2$, compute the sample standard
   deviation $\sigma$, the interquartile range IQR, and the robust scale

   $$
   \sigma_{\text{scott}} =
   \begin{cases}
   \min\big(\sigma,\; \text{IQR} / 1.34\big), & \text{IQR} > 0 \\
   \sigma, & \text{otherwise}
   \end{cases}
   $$

   and Scott's rule for a univariate density,

   $$
   h_{\text{scott}} = 1.059\; \sigma_{\text{scott}}\; k^{-1/5}.
   $$

   If $k = 1$, $h_{\text{scott}} = h_{\min}$. The base bandwidth is then clipped
   to the feasible band:

   $$
   h_{\text{base}} = \text{clamp}(h_{\text{scott}},\; h_{\min},\; h_{\max}).
   $$

4. **Degenerate spread.** If $R < 10^{-12}$ (all values equal), the ceiling is
   the declared range $h_{\max} = hi - lo$ (or 1 if the range is empty) and

   $$
   h_{\text{base}} = \max\big(10^{-6},\; 0.05\, h_{\max}\big).
   $$

5. **Per-observation bandwidths (uncertainty scaling).** With
   $u_{\text{med}}$ the median of the uncertainties, each trial gets its own
   kernel width

   $$
   h_i = h_{\text{base}}\; \left( \frac{u_i}{u_{\text{med}}} \right)^{\beta},
   \qquad \beta = 0.5,
   $$

   clipped to $[10^{-6},\; h_{\max}]$. Trials whose objective was measured less
   reliably (larger $u_i$) therefore contribute wider, flatter kernels and pull
   the density less strongly. This is the second place uncertainty enters the
   model (the first is the adjusted loss in 3.3).

6. **Density.** The model is a normalized mixture of kernels plus a uniform
   prior over the declared range:

   $$
   g(x) = \frac{w_0\, U_{[lo,hi]}(x) + \sum_{i=1}^{k} \varphi\big(x \mid v_i,\, h_i^2\big)}
               {w_0 + k},
   $$

   where $\varphi(x \mid \mu, \sigma^2)$ is the normal density and
   $U_{[lo,hi]}(x) = 1/(hi - lo)$ inside the box, 0 outside. The prior keeps a
   nonzero probability mass everywhere inside the box, which stabilizes the
   expected-improvement ratio and acts as soft exploration.

7. **Log-density.** `logpdf(x)` returns $\log g(x)$ computed directly from the
   expression above. If no observations exist for the side, the log-density is
   $-\infty$.

8. **Sampling.** One draw from the mixture: choose the prior component with
   probability $w_0/(w_0 + k)$, otherwise kernel $i$ with probability
   $1/(w_0 + k)$. Prior draws are uniform on $[lo, hi]$; kernel draws are
   $\mathcal{N}(v_i,\, h_i^2)$. Integer parameters are rounded and clamped to
   $[lo, hi]$; continuous parameters are clamped to $[lo, hi]$.

#### Categorical parameters: Laplace-smoothed multinomial

Over the category universe $C$ with observed counts $c_j$ (total $k$ draws), the
probability of category $j$ is

$$
p_j = \frac{c_j + 1}{k + |C|}
$$

(Laplace smoothing with $\alpha = 1$; every category keeps a nonzero
probability). `logpdf(x) = log p_j`; sampling draws from this multinomial.

### 3.6 What `fit` stores

The result of `fit` is a fitted state: `phase`, `n_completed_`,
`good_trials_`, `bad_trials_`, `param_meta_`, and the dictionaries
`good_models_` / `bad_models_` mapping each parameter to its two densities
(`is_fitted = True`). In the uniform phase `fit` returns with `is_fitted =
False` and no models, by design.

## 4. `propose_trials(n)` step by step

### 4.1 Uniform phase (n < n_initial)

Each of the `n` candidates is an independent, identically distributed uniform
draw over the declared space:

- float: $x \sim U(lo,\; hi)$;
- int: $x = \text{round}\, U(lo,\; hi)$, then clamped to the integer box
  $[\lceil lo \rceil,\; \lfloor hi \rfloor]$;
- choice: one value uniformly from the declared set.

Parameters are drawn in dependency (topological) order: a parent is drawn
first, and a conditional child is only drawn if the already-drawn parent equals
the value the dependency requires. The trials play no role here. Every draw is
filtered by the feasibility rule (section 6); infeasible draws are discarded
and redrawn. Each returned candidate carries `ei = None` and
`origin = 'uniform'`.

This phase deliberately ignores the trials: with fewer than `n_initial`
completed trials there is nothing to fit, and the uniform draws give the first
trials good coverage of the whole box.

### 4.2 Trials phase: model draws + reserved random slots

The batch has two parts: `n_model = n - n_explore` model-informed candidates
followed by `n_explore = clamp(explore_slots, 0, n)` entirely random ones.

**Model draws.** One candidate is built by sampling every active parameter
from its good density (section 3.5), in dependency order:

$$
x_p \sim g_p(x), \qquad p = 1,\dots,d_{\text{active}}.
$$

The candidate is the assembled vector $x = (x_1, \dots, x_{d_{\text{active}}})$.
If the draw is infeasible it is discarded and redrawn. This loop repeats until
`n_model` feasible candidates exist (bounded by an attempt counter). Because
the good set is the *source* of candidates, these draws cluster where good
trials live; the bad density plays no role in generating them (it enters only
through the EI score, section 4.3, which is reported but never used to order
the batch).

**Explore slots.** The last `n_explore` candidates are independent fully-uniform
draws over the declared space (the same sampler as the uniform phase, section
4.1). They guarantee that every trials-phase batch still contains entirely
random members, no matter how peaked the good densities get. They carry
`ei = None`, `origin = 'explore'`, and the table shows `-` for them.

### 4.3 Scoring each candidate: expected improvement (EI)

Each proposed candidate is evaluated against both fitted densities using a
product (independent-marginals) assumption over the active parameters. The
log-densities are sums over per-parameter log-densities; inactive or
model-less parameters contribute a factor of 1, i.e. nothing:

$$
\log g(x) = \sum_{p} \log g_p(x_p),
\qquad
\log \ell(x) = \sum_{p} \log \ell_p(x_p),
$$

with the convention that a single zero-density factor makes the whole joint
log-density $-\infty$.

The Good/Bad density ratio is computed robustly in log space,

$$
r(x) = \frac{\ell(x)}{g(x)} = \exp\big(\log \ell(x) - \log g(x)\big),
$$

with the log difference clipped to $[-700, 700]$, plus explicit degenerate
cases: both densities zero -> $r = 1$; only $\ell$ zero -> $r = 0$; only $g$
zero -> $r$ huge.

The implemented EI score is the TPE-style quantity

$$
\text{EI}(x) = \frac{1}{q + (1 - q)\; r(x)}.
$$

Interpretation. A candidate that is much more probable under the good density
than under the bad density has $r \to 0$ and the score saturates at $1/q$
(4 with the default $q = 0.25$). A candidate equally plausible under both
($r \approx 1$) scores about 1. A candidate that is mostly bad has $r > 1$ and
scores below 1, decaying toward 0. So EI is high exactly where the model
thinks good trials are concentrated and bad trials are not, which is the TPE
heuristic for "where the next improving configuration probably lives".

### 4.4 Output

Candidates are returned and printed in **draw order, never sorted by EI**, so
the printed table does not steer the agent toward candidate 1. Each candidate is

```
{ 'params': { ... }, 'ei': float | None, 'origin': 'uniform' | 'trials' | 'explore' }
```

Row `N` of the printed table equals `proposed_trials[N-1]`. `ei` is `None`
(and the table shows `-`) for uniform-phase candidates and for explore slots.

## 5. Where uncertainty enters (summary)

Uncertainty is used in exactly two places, both in the trials phase:

1. **Ranking (fit):** the adjusted loss $\tilde{\ell}_i = \ell_i + \lambda s_i$
   decides who is good and who is bad. Uncertain good-looking trials are
   demoted.
2. **Density shape (fit):** the per-observation bandwidths
   $h_i \propto (u_i / u_{\text{med}})^{\beta}$ widen the kernels of
   high-uncertainty trials.

The uniform phase does not use the trials at all, so uncertainty does not enter
there.

## 6. Feasibility and the runtime budget

The default feasibility rule (experiment-specific, in `ag_hypopt.py`) enforces
that a config fits inside the trial time budget:

$$
n_{\text{runs}} \times n_{\text{iter}} \le
\text{cap}, \qquad
\text{cap} = \frac{\text{BUDGET\_HOURS} \times \text{CALLS\_PER\_HOUR}}
                    {\text{number of benchmark experiments}}.
$$

For experiment_1 (3.5 h, 160,000 calls/h, 14 experiments) this cap is 40,000.
Both proposal phases redraw any config that fails the feasibility test, so the
optimizer never proposes a trial that cannot run within budget.

## 7. Algorithm knobs and their effect

| knob | effect |
|---|---|
| `n_initial` | length of the pure-uniform warm-up; raise for more initial coverage, lower to enter trials-based proposals sooner |
| `quantile` | size of the good set; lower = more selective, higher = more tolerant good set |
| `lcb_lambda` | how strongly uncertainty demotes a trial in the Good/Bad ranking |
| `bandwidth_beta` | how strongly per-trial uncertainty widens its kernel (0 = no scaling) |
| `prior_weight` | weight of the uniform prior inside every density (stability / soft exploration) |
| `feasible` | callable that rejects configs that cannot run within budget |

## 8. A worked micro-example

Suppose six completed trials with objectives and uncertainties

$$
(\ell,\; s) = (0.10,0.01),\;(0.12,0.04),\;(0.15,0.02),\;(0.35,0.05),\;(0.40,0.06),\;(0.45,0.10)
$$

and $\lambda = 0.5$, $q = 0.25$. Adjusted losses:

$$
\tilde\ell = 0.105,\;0.140,\;0.160,\;0.375,\;0.430,\;0.500 .
$$

Stable sort preserves this order; $m = \text{round}(1.5) = 2$, so trials 1 and 2
are good and trials 3-6 are bad. For each parameter, a good density and a bad
density are built from those two groups (bandwidths per section 3.5).

Now suppose a candidate $x$ has joint log-densities
$\log g(x) = -1.0$ and $\log \ell(x) = -2.5$ under the fitted models. Then

$$
r(x) = e^{-2.5 - (-1.0)} = e^{-1.5} \approx 0.223,
\qquad
\text{EI}(x) = \frac{1}{0.25 + 0.75 \times 0.223} \approx 2.40 .
$$

A second candidate with $\log g = -1.0$, $\log \ell = -0.5$ gives
$r = e^{0.5} \approx 1.65$ and $\text{EI} \approx 0.67$: less probable under the
good model than under the bad one, hence a lower score. Both candidates are
shown to the agent in draw order, with these `ei` values as hints only.

## 9. References

The two-phase structure, the Good/Bad split, and the EI formula follow the TPE
family of models (Bergstra et al., "Algorithms for Hyper-Parameter
Optimization", 2011), adapted here with an uncertainty-adjusted ranking, a
uniform prior component, variable bandwidths scaled by trial uncertainty, and
explicit uniform draws during the first `n_initial` trials.
