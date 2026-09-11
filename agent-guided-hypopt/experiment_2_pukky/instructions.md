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
- ⛔ **Run nothing but the notebook's `▶` cells.** Execute them in place (for example with
  `papermill`). Never run any other program. Never create or run a script, helper, or
  automation of any kind, in the experiment folder or anywhere else (including `/tmp`).
- ⛔ **No background or detached processes, with one exception.** Never use `&`, `nohup`,
  `setsid`, or any detach trick, and never start a long-running process to get around a
  command or time limit, except the trial notebook's own `▶` run cell, which is executed
  detached and polled as described in **Executing the trial cell** below.
- ⛔ **Never work around your own limits.** If a step does not fit within the tools and the
  time you have, that is a stop condition, not a problem to solve with a gadget
  you invent.
- ⛔ **Make only the decisions a cell asks for.** The only decisions in this campaign are the
  cell-3 choice and the cell-6/7 text. Do not decide anything else: not how to run, not how to
  wait, not how to choose beyond the cell's instruction, not what to record beyond it.
- ⛔ **Touch only the allowed files.** The allowed files are exactly the current trial
  notebook, the trial notebooks it generates, and `trials.json`. Any other file, anywhere, is a
  stop condition.
- Fail fast: if a run is obviously broken, stop it and record `objective: null` with the
  reason, as the notebook instructs.
- Structural changes (score, likelihood, model, template, protocol) are not part of this
  campaign and you do not look for them. If an urgent idea comes up anyway, write it for Anuar
  as a short note in the notebook's `✍️` analysis, then keep going. Never stop, delay, or
  interrupt a trial because of an idea: this is a hyperparameter tuning algorithm.

## Executing the trial cell (the one exception to the no-background rule)

One trial runs `objective()` and takes roughly 20 to 30 minutes, longer than the agent's
per-command time limit. The trial notebook's own `▶` run cell is the single sanctioned
exception to the no-background rule. Use the repository's `.venv` (its `papermill` and
`python3` kernel have torch; the plain system `papermill` starts a kernel without torch),
launch that cell with `setsid` in a new session and with stdin closed, so it survives the command runner's process-group cleanup, then poll:

```bash
cd <experiment folder> && setsid <repo>/.venv/bin/papermill --no-progress-bar trial_XX.ipynb trial_XX.ipynb </dev/null >/dev/null 2>&1 &
```

then wait with short commands, for example:

```bash
pgrep -af 'papermill trial_XX'
```

repeating until no matching process remains. `papermill` writes the notebook in place when it
exits, so once the process is gone you read the results and continue. This exception applies
only to running this notebook's own `▶` cells through `papermill`, never to any other program,
and never to more than one detached run at a time.

Everything else lives where it belongs:
physics -> `context.md` | idea of the experiment -> `README.md` | registry -> `trials.json`
| algorithm + harness -> `ag_hypopt.py` + `src/` | trial worksheet + per-trial instructions ->
`template.ipynb`
