#!/usr/bin/env python3
"""_fork.py — mechanically fork a series-24 notebook (one-change rule).

Usage:
  python3 _fork.py NEW_ID SRC_ID "SHORT TITLE" "ONE CHANGE" \
      [--config KEY=VALUE ...] [--add-config "LINE" ...] [--replace "old==>new" ...]

What it does (mechanical only — keep the science edit to the config/one cell):
  * copies SRC_ID.ipynb -> NEW_ID.ipynb
  * sets NB_ID / ONE_CHANGE in the config cell
  * rewrites the title markdown cell (first cell) and the hypothesis cell
  * rewrites the FIG1 HTML filenames (SRC_ID-fig1a -> NEW_ID-fig1a)
  * applies --config substitutions inside the CFG dict (KEY=<old>, -> KEY=VALUE,)
  * applies --replace raw text substitutions over every cell's source
  * strips outputs / execution_count so the new notebook starts clean
"""
import argparse, json, os, re, sys

def cell_text(c): return "".join(c["source"])
def set_cell_text(c, s): c["source"] = s.splitlines(keepends=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_id"); ap.add_argument("src_id")
    ap.add_argument("short_title"); ap.add_argument("one_change")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--add-config", action="append", default=[])
    ap.add_argument("--replace", action="append", default=[])
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src_p = os.path.join(here, f"{a.src_id}.ipynb")
    dst_p = os.path.join(here, f"{a.new_id}.ipynb")
    if not os.path.exists(src_p):
        sys.exit(f"source notebook not found: {src_p}")
    if os.path.exists(dst_p):
        sys.exit(f"destination already exists: {dst_p}")

    nb = json.load(open(src_p))
    cells = nb["cells"]

    # --- 1. NB_ID / ONE_CHANGE in the config cell ---
    for c in cells:
        if c["cell_type"] == "code" and "NB_ID" in cell_text(c) and "=" in cell_text(c):
            s = cell_text(c)
            s = re.sub(r"NB_ID\s*=\s*'[^']*'", f"NB_ID = '{a.new_id}'", s)
            s = re.sub(r"ONE_CHANGE\s*=\s*'[^']*'",
                       "ONE_CHANGE = '" + a.one_change.replace("'", "\\'") + "'", s)
            set_cell_text(c, s)
            break

    # --- 2. config substitutions (inside the CFG dict lines) ---
    for kv in a.config:
        key, _, val = kv.partition("=")
        key, val = key.strip(), val.strip()
        hit = 0
        for c in cells:
            if c["cell_type"] != "code":
                continue
            s = cell_text(c)
            pat = re.compile(rf"^(\s*){re.escape(key)}\s*=\s*[^,\n]*,", re.M)
            s2, n = pat.subn(lambda m: f"{m.group(1)}{key}={val},", s)
            if n:
                set_cell_text(c, s2); hit += n
        if not hit:
            sys.exit(f"--config {key}: no matching 'KEY=<value>,' line found")

    # --- 3. add config lines (after the CFG dict opening) ---
    if a.add_config:
        for c in cells:
            if c["cell_type"] == "code" and re.search(r"^CFG\s*=\s*dict\(", cell_text(c), re.M):
                s = cell_text(c)
                s = re.sub(r"(CFG\s*=\s*dict\(\n)", r"\1" + "".join(
                    f"    {ln.strip()}\n" for ln in a.add_config), s, count=1)
                set_cell_text(c, s)
                break

    # --- 4. title + hypothesis markdown ---
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and re.match(r"^#\s*2\d\w*\s*[—-]", cell_text(c).strip()):
            set_cell_text(c, f"""# {a.new_id} — {a.short_title}

**Series 24 — real-data accuracy.** One change vs `{a.src_id}`:

> **{a.one_change}**

Full context and the frozen protocol/metric are in `SERIES_STATE.md`. Baseline reference:
`experiment_6 trial_07` (24a) = **W_obj(all14) 0.0811, T≥40 0.0675, μ rel-RMSE 35.9 %, γ 22.3 %**, 0/14 divergences.
""")
            break
    for c in cells:
        if c["cell_type"] == "markdown" and cell_text(c).startswith("## Hypothesis / prediction"):
            set_cell_text(c, f"""## Hypothesis / prediction — {a.new_id}

- **Change vs previous notebook (`{a.src_id}`):** {a.one_change}
- **Hypothesis:** *(the mechanism this change should exploit — fill in)*
- **Falsifiable prediction:** *(the numbers/direction expected; state it before the run)*
- **If falsified:** *(the conclusion to draw and the next step)*
""")
            break

    # --- 5. raw replacements (per cell) ---
    for spec in a.replace:
        old, _, new = spec.partition("==>")
        n = 0
        for c in cells:
            s = cell_text(c)
            if old in s:
                set_cell_text(c, s.replace(old, new)); n += s.count(old)
        if not n:
            sys.exit(f"--replace: pattern not found: {old[:70]}")

    # --- 6. FIG1 HTML names ---
    for c in cells:
        s = cell_text(c)
        s2 = re.sub(rf"{re.escape(a.src_id)}(-fig1[ab])", f"{a.new_id}\\1", s)
        if s2 != s:
            set_cell_text(c, s2)

    # --- 7. clean outputs ---
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"] = []; c["execution_count"] = None
        c.setdefault("metadata", {})
    json.dump(nb, open(dst_p, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {dst_p} ({len(cells)} cells)")

if __name__ == "__main__":
    main()
