# Series-24 campaign — delegation briefs (for spawned runners)

The parent spawns **isolated** subagents (`sessions_spawn`, `mode=run`, `context` omitted) — they must NOT
message the user; the parent owns reporting. Each child runs a **batch** of consecutive notebooks
(forked one after the other), commits+pushing after each, and returns one compact summary.

## Common preamble (prepend to every batch task)

> You are the **series-24 campaign runner** for the quantum-ml project. Work entirely in
> `/home/pukky/.openclaw/workspace/qm-ml`, branch `develop`. **Do NOT message the user.**
>
> FIRST read, in full: `notebooks/24_realdata_accuracy/SERIES_STATE.md` (goal, frozen protocol §2, frozen
> metric §3, failure modes §4, roadmap §5, log §7) and `notebooks/24_realdata_accuracy/RUNNER.md` (the loop).
> Then execute your assigned batch, one notebook at a time, forking each from the previous one.
>
> Per notebook: (1) `python3 _fork.py <new> <prev> "<short title>" "<one change>" [...]`; (2) edit the new
> notebook with python+json to implement the ONE change and fill the hypothesis cell (hypothesis + a
> falsifiable prediction, written BEFORE the run); (3) smoke `NB_SMOKE=1 papermill <new>.ipynb
> /tmp/_smoke_<new>.ipynb`; (4) full run in place `NB_RECOMPUTE=1 nohup papermill <new>.ipynb <new>.ipynb
> > /tmp/<new>_run.log 2>&1 &` and WAIT for `data/processed/<new>_history.json` (each run ~15-25 min —
> use `sleep` inside exec to wait, do not tight-poll); (5) replace the verdict placeholder with the real
> verdict (metric table, one-line read, HIT/PARTLY/FALSIFIED, FM tag, next step); (6) append the §7 log row;
> (7) `git add -A && git commit -m "series 24: <new> — <one change> (<verdict>)" && git push` **before** the
> next notebook.
>
> Rules: never edit `notes/JOURNAL.md`, `src/`, or other series. Never tune a knob against `mu_true`/
> `gamma_true`. Keep the frozen protocol + metric byte-identical. A worse result is a valid negative result —
> record it and move on (do not silently re-tune). If a cell errors, fix and re-run that notebook.
> Return ONE compact summary at the end: per notebook `id | one change | W_obj all14 / T>=40 | verdict one-liner`,
> then the best config so far.

## Batches

1. **B1 — 24c, 24d, 24e, 24f, 24g** (Phase A3–A7). Fork chain starts from **24b**.
2. **B2 — 24h, 24i, 24j, 24k** (A8 γ-travel, A9 coupled travel, A10 truth-free μ-init, B1 σ-shape matching).
3. **B3 — 24l, 24m, 24n, 24o, 24p** (B2 scale-only control, B3 σ-weight anneal, B4 robust kernel, B5 estimator-matched σ, B6 hybrid loss).
4. **B4 — 24q, 24r, 24s, 24t, 24u** (B7 heterogeneity, B8 FWHM-1D+count prior, C1 bootstrap, C2 sandwich, C3 profile interval).
5. **B5 — 24v, 24w, 24x, 24y, 24z** (C4 bias-corrected CRB, C5 coverage referee, D1 matched estimator, D2 differentiable forward model, D3 σ-bias table).
6. **B6 — 24aa, 24ab, 24ac, 24ad** (E1 combination, E2 repeat winner ×3, E3 independent referee, E4 held-out protocol) + series summary.

> The parent may reorder/insert batches based on the running verdicts; the roadmap §5 is the source of truth.
