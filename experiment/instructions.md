# instructions.md — the trial loop (DRAFT v0)

Role: you are the agent running ONE trial of the qm-ml hyperparameter campaign.
One trial ≈ 1h. Be sample-efficient: every pick must be defensible in writing.

## Before each trial
1. Read `context.md`, this file, and **all previous trial_XXX.ipynb notebooks**
   (scan their "**Key insight:**" header first; read fully only if needed).
2. Read `trials.json` — structured history only (config, objective, uncertainty, one-line summary).

## The loop
1. **Propose** — run `tpe_custom.propose(trials, space, 10)` → 10 candidates from the surrogate.
2. **Analyze** — assess each candidate with physics reasoning (context.md failure modes) and
   notebook history. Rank all 10. Note explicitly when your ranking disagrees with the algorithm's.
3. **Select** — pick ONE. Write the reasoning: why this one, which hypothesis it tests, what to watch.
4. **Run** — evaluate the candidate (~1h). Show results + plots in this notebook.
5. **Reflect** — was the hypothesis right? what was learned? what should the next trial test?
6. **Record** — append `{trial_id, config, objective, uncertainty, summary}` to `trials.json`,
   update the current best. One-line summary only; the full story lives in this notebook.

## Conventions
- First content cell after the title: "**Key insight:** ..." — one line, the takeaway for scanners.
- Last analysis cell: "**Next hypothesis:** ..." — what the next trial should test.
- Notebooks are the full memory. `trials.json` is data for the algorithm only.
- If a trial fails mid-run: record it (`objective: null`, summary = error), keep the notebook,
  reflect on the cause. Fail fast — don't burn the hour on an obviously broken candidate.

## Authority
- Hyperparameter choices: your call, with written reasoning.
- **Structural changes** (score function, likelihood, bandwidths, model): PROPOSE in the notebook,
  do NOT run them without Anuar's explicit OK.
- Do not start trial_002 before the 18b real-data verdict is in (it may change the objective).

## Open items (decide with Anuar before trial_002)
- Objective definition (RMSE combo? NLL? + Fisher term?)
- Synthetic vs real benchmark for trials
- Full 14-exp trial vs reduced benchmark (runtime)
- Noise model: single seeded run (deterministic) vs multi-seed std → what "uncertainty" means
