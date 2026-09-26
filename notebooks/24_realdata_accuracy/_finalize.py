#!/usr/bin/env python3
"""_finalize.py — write the verdict markdown into a series-24 notebook's Verdict cell.

Usage:  python3 _finalize.py <NB_ID> <verdict.md>
Replaces the markdown cell whose source starts with '## Verdict' with the file's content.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
nb_id, vfile = sys.argv[1], sys.argv[2]
p = os.path.join(HERE, f'{nb_id}.ipynb')
nb = json.load(open(p))
s = open(vfile if os.path.isabs(vfile) else os.path.join(HERE, vfile)).read()
done = False
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## Verdict'):
        c['source'] = s.splitlines(keepends=True); done = True; break
if not done:
    sys.exit('verdict cell not found')
json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
print(f'{nb_id}: verdict written from {vfile}')
