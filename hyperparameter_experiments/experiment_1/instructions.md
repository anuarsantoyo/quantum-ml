# instructions.md — the trial loop (DRAFT v1)

Role: you are the agent running ONE trial of the qm-ml hyperparameter campaign.
One trial ≈ 1h. Be sample-efficient: every pick must be defensible in writing.

## Before each trial
1. Read `context.md`, this file, and **all previous trial_XXX.ipynb notebooks**
   (cell 1 prints their "**Key insight:**" lines — the memory index; read fully if needed).
2. Read `trials.json` — structured history only (config, objective, uncertainty, summary).

## The loop (template.ipynb cells)
1. **Setup** (cell 1) — load trials.json + memory index; note current best + ranges.
2. **Propose** (cell 2) — `tpe_custom.propose(trials, space, 10)` → 10 candidates + EI scores.
3. **Analyze** (cell 4, markdown) — assess each candidate with physics reasoning
   (context.md failure modes) + history. Rank all 10. Note disagreements with the EI ranking.
4. **Select** (cell 5, markdown) — pick ONE: why, which hypothesis it tests, what would
   confirm/refute it. Set `INDEX` in cell 6.
5. **Run** (cell 7, ~1h) — evaluate via `eval_harness.run_trial(config)`; progress shown.
6. **Results** (cell 8) — per-exp table, plots, objective + uncertainty
   (`eval_harness.compute_objective`), comparison to previous trials.
7. **Reflect** (cell 9, markdown) — hypothesis right? learned? next hypothesis?
   Ends with the **Key insight** sentence — copy it into the title cell.
8. **Record** (cell 10) — append `{trial_id, config, objective, uncertainty, summary,
   key_insight, notebook}` to trials.json; update current best.

## Conventions
- Title cell starts with "**Key insight:** <TBD — filled after trial>" — fill it at the end;
  it must match the reflection's Key insight and the trials.json summary.
- Notebooks are the full memory. `trials.json` is data for the algorithm only.
- **Paths:** trials run with cwd = repo root (data/src resolve); the notebook's own folder is
  `NOTEBOOK_DIR` (tpe_custom, trials.json, sibling trial notebooks). Set by cell 1.

## Execution (Anuar's convention, 2026-08-29)
- Trial notebooks are executed **IN PLACE** — no `-executed` copies:
  `papermill trial_XXX.ipynb trial_XXX.ipynb --log-output`
- Results, plots, and reflection live directly in the trial notebook (one file = the record).
- If a run fails mid-way, partial outputs stay in the notebook — that's fine: record the
  failure in trials.json (objective: null) and reflect on the cause.
- Dev/exploration notebooks (17/18 series style) may keep the source + executed pair;
  trials never do.
- **Never "Run All"** — markdown cells 4–5 and 9 are written by the agent between
  execution phases. Watch the ⛔ STOP markers.
- If a trial fails mid-run: record it (`objective: null`, summary = error), keep the
  notebook, reflect on the cause. Fail fast — don't burn the hour on a broken candidate.

## Authority
- Hyperparameter choices: your call, with written reasoning.
- **Structural changes** (score function, likelihood, bandwidths, model): PROPOSE in the
  notebook, do NOT run them without Anuar's explicit OK.
- Trial protocol (benchmark, seeds, objective) is FIXED — do not change it per trial;
  propose protocol changes to Anuar instead.

## Algorithm swapping (why each experiment folder has its own tpe_custom.py)
- The interface `propose(trials, space, n_candidates) -> [{config, score}]` is the contract.
  experiment_1 ships the TPE+EI draft; a future experiment_2 can ship GPBO or CMA-ES
  against the same contract, same benchmark → clean A/B. Decision TPE vs GPBO: pending.
- The agent considers `score` but decides with physics reasoning on top.

## Open items (decide with Anuar before trial_002)
- Exact objective combination (MSE_μ + MSE_γ? normalized?) and uncertainty definition
- RUNTIME_CAP value (trial ~1h) — feasibility constraint in the search space
- Whether structural knobs enter the space as categoricals, or a first structure probe runs
- Whether n_runs/n_iter are tuned or fixed as protocol
