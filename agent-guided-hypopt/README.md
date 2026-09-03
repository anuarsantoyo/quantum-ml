# AG-HYPOPT — Agent-Guided Hyperparameter Optimization

Agent-guided hyperparameter optimization for the qm-ml project: a probabilistic optimizer (TPE) proposes candidates, the agent (LLM) analyzes them with physics reasoning, picks one, runs it (~3.5h), and records the outcome in a worksheet notebook. The agent writes no code and creates no files; each trial notebook's last cell generates the next one. It sets only `INDEX`, `SUMMARY`, `KEY_INSIGHT` (`TRIAL_ID` is auto-stamped).

## Structure — total isolation

Each experiment is a **fully self-contained folder**: its own copy of context, instructions, algorithm module, template, trial records, registry, and figures. Nothing is shared at the top level — an experiment is a frozen snapshot of the model + method at the time it was created.

```
agent-guided-hypopt/
├── README.md          ← this file: index of experiments (brief description each)
└── experiment_N/      ← one experiment = one benchmark + one algorithm + one model snapshot
    ├── context.md         physics/model knowledge (frozen)
    ├── instructions.md    AG-HYPOPT recipe card (frozen)
    ├── ag_hypopt.py       AGHyperopt (TPE) + trial harness (frozen)
    ├── space.json         declared space + conditional dependencies (frozen)
    ├── template.ipynb     trial worksheet (frozen): its last cell generates the next trial
    ├── trial_01.ipynb     ships unexecuted: the first trial with results
    ├── trial_XX.ipynb     created by the previous trial's last cell, executed in place
    ├── trials.json        distilled registry + current best
    └── figures/
```

## Conventions

- **Numbered evolution**: experiment_N+1 forks experiment_N. Changes to the model (e.g. TPA → Gaussian), the method, or the docs happen in the NEW experiment. Past experiments are never edited — they are the record.
- **Trial execution**: papermill in place (`papermill x.ipynb x.ipynb`); one notebook = the trial record. Never "Run All" — ✍️ cells are written by the agent between execution phases. Each trial's last cell generates the next trial notebook and prompts the agent to run it.
- **Authority**: hyperparameters = agent's call within the protocol; structural changes (score, likelihood, model) need Anuar's OK (permission hook in instructions.md).
- **Start a new experiment**: `cp -r experiment_N experiment_N+1`, then rewrite context.md to match the new question and reset the trial chain: delete executed trial notebooks, and ship a fresh unexecuted `trial_01.ipynb` copied from the folder's template (with number `01` stamped in).

## Experiment index

| Experiment | Description |
|---|---|
| experiment_1 | Origin: 17/18-series lineage (synthetic 17g win, real-data 18b/18c verdicts). Algorithm: TPE. Model: KDE+REINFORCE inversion, Lorentzian-width (TPA) line shape. **Real machinery built 2026-08-30**: ag_hypopt.py (AGHyperopt TPE class: LCB split, magic-clipped variable-bandwidth KDEs, uniform-prior exploration, conditional tree), space.json (declared space), trial harness in ag_hypopt.py (run_trial/compute_objective, restored 17g port); self-generating chain in template.ipynb (trial_01 ships unexecuted), registry with baseline_17g (obj 0.001578 ± 0.000859). Budget ~3.5h/trial (cap 40k). Space provisional pending model analysis. |
