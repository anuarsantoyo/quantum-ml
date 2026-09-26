# RUNNER.md — mechanical run-book for the series-24 campaign

You are the **series-24 campaign runner**. Working dir: `qm-ml` (branch `develop`).
Read `SERIES_STATE.md` first (goal, protocol, metric, roadmap, log). Then execute the loop below for the
notebooks assigned to you. **Do not message the user** — the parent session owns reporting.

## Hard rules
- **One change per notebook**, stated in the markdown cell *above* the config cell, with a falsifiable
  prediction, **before** the run.
- **Fork the previous notebook** — never write a notebook from scratch.
- Keep the frozen protocol (§2) and the frozen metric (§3) untouched in every notebook.
- Only touch files inside `notebooks/24_realdata_accuracy/`, `data/processed/24*`, and the `SERIES_STATE.md`
  §7 log. **Never** edit `notes/JOURNAL.md`, `src/`, or any other series.
- **Truth-free:** never tune a knob against `mu_true`/`gamma_true`.
- Figures inline only (no `savefig`, no PNG on disk). Notebooks run **in place** (`papermill x x`).
- `git add -A && git commit && git push` after **every** notebook, before starting the next.
- If a run diverges (a cell errors), fix the notebook, re-run, and record the fix in the verdict.

## Per-notebook loop

```bash
cd notebooks/24_realdata_accuracy

# 1. fork the previous notebook (mechanical renames: NB_ID, title, hypothesis stub, html names)
python3 _fork.py 24b 24a "short title of the change" "ONE CHANGE description" \
    --config lr_mu=0.25 \                 # (optional) replace inside the CFG dict
    --add-config "my_new_knob=1.0" \      # (optional) add a config line
    --replace "old text==>new text"       # (optional) any other targeted text edit (e.g. a new code line)

# 2. edit the new notebook for the scientific change (python + json is the reliable way):
#    - fill the hypothesis markdown cell with hypothesis + prediction
#    - make the ONE code/config change (beyond what _fork.py did mechanically)
#    - if you add a new code cell, insert it in the SAME structural position as related cells

# 3. smoke-test (fast; must produce inline PNG outputs, no exceptions)
NB_SMOKE=1 papermill 24b.ipynb /tmp/_smoke_24b.ipynb
rm -f /tmp/_smoke_24b.ipynb

# 4. full run IN PLACE  (14 experiments, ~15-30 min; run it in the background and poll)
NB_RECOMPUTE=1 nohup papermill 24b.ipynb 24b.ipynb > /tmp/24b_run.log 2>&1 &

# 5. after it finishes: read the FROZEN METRIC print + the saved objective from
#    data/processed/24b_history.json, write the VERDICT markdown cell (replace the placeholder):
#    metrics table + one-line read + HIT/PARTLY/FALSIFIED + FM tag.

# 6. append the §7 log row in SERIES_STATE.md

# 7. commit + push
git add -A && git commit -m "series 24: 24b — <one change> (<verdict one-liner>)" && git push
```

## Verdict cell template
```markdown
## Verdict — <one-line headline>

`W_obj(all14) = X | T>=40 Y | T<=20 Z | mu rel-RMSE A% | gamma rel-RMSE B% | diverged D/14`
(vs previous notebook: …)

**(a)** … **(b)** … — the scientific read, with the per-cell ratios that matter.

**Prediction: HIT / PARTLY / FALSIFIED.** <why>

**FM tag:** FM#… . **Next:** <the notebook this verdict implies>.
```

## Notes
- The objective is chaotically sensitive (≤0.005 = noise). Repeat a winning config before believing it.
- New code cells: keep them small and self-contained; prefer a `cfg` key read by the existing
  `run_experiment` over rewriting the loop. If you must change `run_experiment`, change only the step rule
  and say so explicitly in the hypothesis cell.
- If `data/processed/<id>_history.json` already exists the notebook will load it as CACHED — always run
  with `NB_RECOMPUTE=1` so the notebook actually runs.
- Check `df -h .` occasionally; delete `/tmp/_smoke_*`.
