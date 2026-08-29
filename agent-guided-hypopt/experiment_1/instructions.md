# instructions.md — PAGHO: the agent loop

> **PAGHO** = Physics-informed Agent-Guided Hyperparameter Optimization.
> This file is the **operational manual for the agent**. Read it together with `context.md`
> (physics + model background) and the experiment's `trial_00` notebook (experiment specifics
> and current state). It is deliberately **generic**: copy-pasteable into any experiment folder.
> *DRAFT — being reviewed with Anuar (2026-08-29).*

---

## 1. Your role

You are the agent running **one trial** of a PAGHO campaign. One trial ≈ 1 hour — expensive.
Be sample-efficient: every pick must be defensible in writing. You are the *physics-informed
selection layer* on top of a numerical optimizer:

- the **algorithm** (`pagho.py`) proposes candidates from the structured history (`trials.json`);
- **you** decide which candidate actually runs, using physics reasoning + the full narrative
  history (the trial notebooks).

## 2. The three sources of knowledge

| File | Content | Role |
|---|---|---|
| `context.md` | physics, model, failure modes | *background* — stable |
| `trial_00.ipynb` | this experiment: objective, benchmark, space, current state | *experiment anchor* — read FIRST |
| `trial_XXX.ipynb` | each trial's reasoning + results + reflection | *memory* — the full story |

- `trials.json` is **not** memory: it is the structured input for the algorithm
  (config, objective, uncertainty, one-line summary per trial).

## 3. Before each trial

1. Read `context.md` (skim — stable).
2. Read `trial_00.ipynb` (the current state — where the experiment stands).
3. Read all previous `trial_XXX.ipynb` notebooks: start at their **Key insight** line (the
   template's setup cell prints them all — the memory index), read fully where depth is needed.
4. Read `trials.json` (structured history only).

## 4. The loop (template.ipynb cells)

1. **Setup** (cell 1) — loads history + memory index; shows current best and parameter ranges.
2. **Propose** (cell 2) — `pagho.propose(trials, space, 10)` → 10 candidates + EI scores.
3. **Analyze** (cell 4, markdown) — assess each candidate with physics reasoning (context.md
   failure modes) + history. Rank all 10. Note disagreements with the EI ranking.
4. **Select** (cell 5, markdown) — pick ONE: why, which hypothesis it tests, what would
   confirm/refute it. Set `INDEX` in cell 6.
5. **Run** (cell 7, ~1h) — `pagho.run_trial(config)`; progress shown. Fail fast: if obviously
   broken, abort and record the failure.
6. **Results** (cell 8) — per-experiment table, plots, objective + uncertainty
   (`pagho.compute_objective`), comparison to previous trials.
7. **Reflect** (cell 9, markdown) — hypothesis right? what was learned? **ideas worth keeping?**
   next hypothesis? Ends with the **Key insight** sentence.
8. **Record** (cell 10) — append `{trial_id, config, objective, uncertainty, summary,
   key_insight, notebook}` to `trials.json`; update current best.

## 5. Conventions

- **Title cell:** `# Trial {N} — <hypothesis>` + `**Key insight:** <TBD — filled after trial>`.
  Fill the Key insight at the end; it must match cell 9 and the `trials.json` summary.
- **Current state at trial start:** the agent writes a short state summary near the top (before
  running), so each notebook is self-explanatory about where the campaign stood.
- **Ideas worth keeping:** the reflection (cell 9) collects ideas for future trials — even ones
  not acted on.
- **Notebooks are the full memory.** `trials.json` is data for the algorithm only.
- **Never "Run All"** — markdown cells 4, 5, 9 are written by the agent between execution
  phases. Watch the ⛔ STOP markers.
- **Execution is IN PLACE:** `papermill trial_XXX.ipynb trial_XXX.ipynb --log-output` — results
  land directly in the notebook; one file = the record. If a run fails mid-way, partial outputs
  stay in the notebook; record `objective: null` + error in `trials.json` and reflect on the cause.
- **Paths:** trials run with cwd = repo root (data/src resolve); the notebook's own folder is
  `NOTEBOOK_DIR` (pagho.py, trials.json, sibling notebooks). Set by the template's setup cell.

## 6. Authority

- **Hyperparameter choices:** your call, with written reasoning.
- **Structural changes** (score function, likelihood, bandwidths, model): PROPOSE in the
  notebook, do NOT run them without Anuar's explicit OK.
- **Protocol** (objective, benchmark, seeds, space): fixed per experiment — defined in
  `trial_00`. Do not change it per trial; propose protocol changes to Anuar.

## 7. The algorithm contract (how algorithms swap)

- ALL scripts live in ONE module per experiment: `pagho.py` (algorithm + harness).
- Contract: `propose(trials, space, n_candidates=10, seed=None) -> [{"config", "score"}]`.
  `score` = Expected Improvement (EI) from the surrogate. Consider it, but decide with physics.
- An experiment can ship any algorithm against this contract (TPE, GPBO, CMA-ES) — swapping
  `pagho.py` is the swap mechanism. Decision TPE vs GPBO: pending.
- `run_trial(config)` and `compute_objective(results)` define the trial and the metric — they
  must stay identical across trials of one experiment (a drifted metric poisons `trials.json`).

## 8. Failure handling

- Trial fails mid-run → record (`objective: null`, summary = error), keep the notebook,
  reflect on the cause. Do not silently skip.
- Candidate obviously broken before the hour is spent → abort it, record why.
- Any doubt about structural changes or protocol → ask Anuar.
