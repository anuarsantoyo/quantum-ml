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
- ⛔ **Run the notebook cell by cell** — like pressing Run on one cell at a time in Jupyter,
  using `nbrun.py` (below). Never run the whole notebook in one pass: **no `papermill`, no
  "Run All"**. Those execute every cell at once and therefore cannot stop for you to fill a
  `✍️` cell after a `▶` cell produced its result. `✍️` cells are written by you between cells.
- ⛔ **Execute in place**: each cell's output must land in the trial notebook. `nbrun.py`
  does this.
- ⛔ **Follow the notebook exactly**: do not skip cells, do not improvise, do not go beyond
  what the cells say.
- ⛔ **Run nothing but the notebook's `▶` cells.** Execute them one at a time with `nbrun.py`.
  Never run any other program, and never create or run any other script, helper, or automation,
  in the experiment folder or anywhere else (including `/tmp`). `nbrun.py` is the one provided
  runner; it is not a script for you to modify or extend.
- ⛔ **No background or detached processes, with one exception.** Never use `&`, `nohup`,
  `setsid`, or any detach trick, and never start a long-running process to get around a
  command or time limit — except running your own trial's long `▶` cell (cell 5) detached via
  `nbrun.py`, as described in **Running the notebook** below.
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

## Running the notebook (cell by cell)

Run the trial notebook **one cell at a time**, exactly as a person would in Jupyter: run a `▶`
cell, read its output, write the next `✍️` cell, run the next `▶` cell, and so on. Use the
bundled runner, which executes a single cell in place against a persistent kernel:

```bash
python3 nbrun.py trial_XX.ipynb <cell_index>   # run one cell (0-based index), save in place, print its output
python3 nbrun.py trial_XX.ipynb --status       # is this notebook's kernel alive?
python3 nbrun.py trial_XX.ipynb --stop         # shut the kernel down (do this when the trial is finished)
```

Run `nbrun.py` with the repository's Python (the one whose `python3` kernel has `torch`).
`nbrun.py` starts a kernel for the notebook on first use and keeps it alive between calls, so a
later cell sees the variables the earlier cells created — the notebook runs exactly as it would
if you pressed Run on each cell yourself. It writes the cell's outputs back into the notebook
and echoes the cell's stdout/stderr to your terminal so you can read it.

Per trial, in order:
1. `python3 nbrun.py trial_XX.ipynb 2` — propose (cell 2). Read the candidate table it prints.
2. Write `✍️` cell 3 (your analysis) and `✍️` cell 4 (`INDEX = N`), then run cell 4:
   `python3 nbrun.py trial_XX.ipynb 4`.
3. Run cell 5 — the trial itself (`objective(CHOSEN)`), which takes ~20–30 minutes:
   `python3 nbrun.py trial_XX.ipynb 5`. This is longer than your per-command limit, so launch
   it detached and poll (see below).
4. Write `✍️` cell 6 (your analysis) and `✍️` cell 7 (`SUMMARY`, `KEY_INSIGHT`), then run cell 7:
   `python3 nbrun.py trial_XX.ipynb 7`.
5. Run cells 8 and 9 (record, then generate the next trial):
   `python3 nbrun.py trial_XX.ipynb 8` then `python3 nbrun.py trial_XX.ipynb 9`.
6. `python3 nbrun.py trial_XX.ipynb --stop`.

### The one long cell (cell 5): launch detached, then poll

Cell 5 is the only step longer than your per-command time limit. Launch just that cell detached,
with stdin closed, so it survives the command runner's process-group cleanup:

```bash
cd <experiment folder> && nohup python3 nbrun.py trial_XX.ipynb 5 </dev/null >/tmp/nbrun_XX.log 2>&1 &
```

then wait with short commands, for example:

```bash
pgrep -af 'nbrun.py trial_XX.ipynb 5'
```

repeating until no matching process remains. When the process is gone, `nbrun.py` has already
written the cell's output into the notebook — read it and continue with cell 6. This exception
applies only to running this notebook's own `▶` cell 5, never to any other program, and never
to more than one detached run at a time.

Everything else lives where it belongs:
physics -> `context.md` | idea of the experiment -> `README.md` | registry -> `trials.json`
| algorithm + harness -> `ag_hypopt.py` + `src/` | trial worksheet + per-trial instructions ->
`template.ipynb` | cell runner -> `nbrun.py`
