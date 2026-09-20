# experiment_7 — AG-HYPOPT: the bandwidth RULE family (where "flatter" turns over)

## The idea
experiment_6 searched the **score design** (with the schedule frozen) and won: best objective **0.0803**
(trial_07: `sigma_weight 0.865`, `h_f_scale 4.0`, `h_s_scale 3.675`, `gamma_rel_cap 0.107`), the
**mu floor halved** (rel-RMSE 77.9 % → 35.9 %) and the **gamma divergence eliminated** (0/30 vs 9/40
in exp5). But the campaign's own correlation table gave an unambiguous next step:

- `corr(obj, h_f_scale) = −0.948`, `corr(obj, h_s_scale) = −0.809` — **bandwidth is the dominant lever**;
- **17/30 trials sat exactly at the `h_f` upper bound (4.0)** — the answer was *"flatter than the space allows"*;
- `sigma_weight` optimum was interior (~0.85–0.92); `gamma_rel_cap` value was irrelevant (any cap > 0 cures the blow-up).

The **physics reason** (19 Sep discussion): the KDE kernel lives over the **sim cloud** but its width is
computed from the **REAL target** (Scott) — structurally mismatched whenever the sim/target spreads
differ, which is always (FM#6/#8). A **flat** kernel makes the KDE responsibilities uniform → reward
std → 0 → the REINFORCE μ channel dies (a *bandwidth* mechanism for the dead count channel!); a
**knife-edge** kernel makes the likelihood spiky and H-sensitive. exp6 answered "flatter is better" but
ran out of room. **exp7 finds where it turns over — and whether the RULE itself matters.**

experiment_7 therefore freezes **both** the schedule (exp5 winner) and the score design (exp6 winner)
and searches the **bandwidth rule family × coefficient**:

| knob | values / range | meaning |
|---|---|---|
| `bandwidth_rule` | `target` / `sim` / `power` | **how** `H_F`, `H_S` are computed (see below) |
| `h_f_scale` | 1.0 – 12.0 | multiplier on the FWHM-channel bandwidth (extended from exp6's 4.0 ceiling) |
| `h_s_scale` | 1.0 – 12.0 | multiplier on the sigma-channel bandwidth (floored at `h_s_min = 0.05` MHz) |

The three rules (defaults reproduce the exp6 winner **exactly**: `target`, `4.0`, `3.675`):

- **`target`** — `H = h·scale·std(real target)·N_target^(−1/6)` — the exp5/exp6 design (Scott-from-target).
- **`sim`** — `H = h·scale·std(sim cloud)·N_sim^(−1/6)`, **recomputed every step** (sim-adaptive). The
  sim spread tracks `mu` (more photons → tighter cloud), so this bandwidth shrinks as μ grows.
- **`power`** — `H = h·scale·std(pooled real targets of the same laser power)·N^(−1/6)` — one **global /
  per-power** bandwidth, pooled across all 7 transmissions of that power.

Everything else is the frozen series-21 / exp5 / exp6 mechanism.

## Benchmark
8 of the 14 experiments (`BENCHMARK_SUBSET`): 1nW/3nW × {Trans05, Trans20, Trans60, Trans100}.
**Held-out** (`HELD_OUT`, validation only): 1nW/3nW × {Trans10, Trans40, Trans80} — the winner is
re-evaluated there at the end of the campaign.

## Data source
`USE_REAL_DATA = True` — real measured FWHM targets from `data/processed/fwhm_linewidths.csv`
(15-series load & filter: raw ×1000 → MHz, keep `fit_error/fwhm < 10`).

## Search space (3 tunables, `space.json`)
| tunable | range | note |
|---|---|---|
| `bandwidth_rule` | `target` / `sim` / `power` | the bandwidth RULE family |
| `h_f_scale` | 1.0 – 12.0 | multiplier on the FWHM-channel bandwidth |
| `h_s_scale` | 1.0 – 12.0 | multiplier on the sigma-channel bandwidth (floored at `h_s_min=0.05`) |

`n_runs`/`n_iter` are fixed by the protocol (100×30), not searched. The **schedule** (exp5 winner) and
the **score design** (exp6 winner) are **frozen**.

## Fixed parameters (`DEFAULT_CONFIG`, in `ag_hypopt.py`)
`n_runs=100`, `n_iter=30`, `lr_mu=0.1669`, `mu_anneal=0.3455`, `lr_gamma=0.4716`, `gamma_anneal=0.4723`,
`clip=inf`, `sigma_weight=0.865`, `gamma_rel_cap=0.107`, `h_s_min=0.05`, `sigma_ref=10.0` (retained;
unused under `MU_SCORE='sigma_prop'`). Defaults of the new knobs (`target`, `4.0`, `3.675`) reproduce the
**experiment_6 winner** exactly.

## Frozen structural choices
- z-form gamma-score (`GAMMA_SCALE=True`, `H_REF=1.0`), `LAMBDA_MEAN=0`
- Anuar's mu reward (`MU_REWARD='loglik_mean'`), mu score denominator `sigma_prop` (`MU_SCORE`)
- 2-D `(FWHM, sigma_fit)` KDE kernel, Scott exponent, Lorentzian fit (2 params)
- no gradient clipping, divergence **guard** (stop if mu∉(0.2, 1500) or gamma∉(0.02, 500))
- init at `0.5 × truth`

## Objective (pre-registered, unchanged from exp5/exp6)
```
err_i = min(rel_sq_i, CAP)              # CAP = 1.0
w(T)  = 0.25 + 0.75 * T/100             # graded toward high T, 0.25 floor
objective = Σ w_i * err_i / Σ w_i       # T-weighted mean over the 2*8 channel errors
uncertainty = std(err_i)/sqrt(n)        # sampling SE across cells (reporting only; NO Fisher)
```

## Running
Trial worksheet: `trial_01.ipynb` (unexecuted); registry/protocol: `trials.json`.
Cap: `MAX_TRIALS=30` (~7.5 min/trial at 100×30 on the 8-exp benchmark ⇒ ~3.75 h + analysis).
Cell-by-cell with `nbrun.py` (see `instructions.md`). Self-test: `python3 ag_hypopt.py`.

## Success bar
γ finite on all 8 cells **and** the T-weighted objective below the exp6 level (~0.086; beating the
0.0803 headline is a bonus). The winner must **also hold on the 6 held-out experiments** (post-campaign
validation): a bandwidth tuned to the benchmark that does not transfer is a failure, not a win.

## Authority
Hyperparameters are the agent's call within this space. The structural change (the configurable
bandwidth rule family + extended bounds) was **pre-authorized by Anuar on 2026-09-20** and is frozen
into this experiment's snapshot; the campaign itself only tunes the three knobs.
