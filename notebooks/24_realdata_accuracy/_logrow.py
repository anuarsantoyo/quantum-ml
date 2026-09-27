#!/usr/bin/env python3
"""_logrow.py — append one row to the SERIES_STATE.md §7 log table (before the 'Current best' marker).

Usage: python3 _logrow.py "<full markdown table row>"
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, 'SERIES_STATE.md')
row = sys.argv[1]
lines = open(p).read().split('\n')

mark = None
for i, ln in enumerate(lines):
    if ln.startswith('**Current best = '):
        mark = i
        break
if mark is None:
    sys.exit('marker "**Current best = " not found')
j = mark - 1
while j >= 0 and (not lines[j].startswith('|') or lines[j].strip() == ''):
    j -= 1
if j < 0:
    sys.exit('no log table row found before the marker')
lines.insert(j + 1, row)
open(p, 'w').write('\n'.join(lines))
print('row inserted after line', j)
