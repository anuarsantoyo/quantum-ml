# instructions.md: AG-HYPOPT, what the agent does

AG-HYPOPT = Agent-Guided Hyperparameter Optimization.

When you are told to work on an experiment, your job is mechanical and is fully defined by the
notebooks. Do exactly this, nothing more:

This is a **hyperparameter tuning algorithm**. Every trial you run is one hyperparameter
configuration of the frozen pipeline. Structural changes (score, likelihood, model, template,
protocol) are not looked for in this campaign.

## Recipe
1. **Read the instructions** (this file) and the experiment's `context.md`. Do not read the
   trial notebooks here: each trial notebook itself prompts you to read the previous trials
   when the time comes.
2. **Start with the first trial**: the experiment folder ships with an unexecuted
   `trial_01.ipynb`. Open it and follow its cells from the top.
3. **Follow the notebook**: go through the trial notebook cell by cell, top to bottom, and do
   exactly what each cell says. `▶` cells are executed as-is. `✍️` cells are written by you
   before you continue.
4. **Repeat the cycle**: each finished trial's last cell creates the next trial (or stops the
   campaign when the cap is reached) and tells you what to do. Open the notebook it names and
   follow it. Keep going until the campaign stops or you are told to stop.

## Rules: non-negotiable
- ⛔ **Create no files yourself.** The only files ever created are the trial notebooks, and
  each is produced by the notebook's own last cell, not by you.
- ⛔ **Write no code.** All code lives in `ag_hypopt.py`, `src/`, and the frozen template. You
  never write, edit, or delete code. You only write analysis text where a `✍️` cell asks for it
  and set the variables the notebook asks for (`INDEX`, `SUMMARY`, `KEY_INSIGHT`; `TRIAL_ID`
  is stamped automatically at generation).
- ⛔ **Never "Run All"**: `✍️` cells are written by you between execution phases.
- ⛔ **Execute in place** (`papermill x.ipynb x.ipynb`): results land in the notebook.
- ⛔ **Follow the notebook exactly**: do not skip cells, do not improvise, do not go beyond
  what the cells say.
- Fail fast: if a run is obviously broken, stop it and record `objective: null` with the
  reason, as the notebook instructs.
- Structural changes (score, likelihood, model, template, protocol) are not part of this
  campaign and you do not look for them. If an urgent idea comes up anyway, write it for Anuar
  as a short note in the notebook's `✍️` analysis, then keep going. Never stop, delay, or
  interrupt a trial because of an idea: this is a hyperparameter tuning algorithm.

Everything else lives where it belongs:
physics -> `context.md` | protocol + registry -> `trials.json` | algorithm + harness ->
`ag_hypopt.py` + `src/` | trial worksheet + per-trial instructions -> `template.ipynb`
