#!/usr/bin/env python3
"""_check.py — guard against silent no-op patches.

Usage: python3 _check.py <NB_ID> TOKEN1 [TOKEN2 ...]
Fails loudly if any TOKEN is missing from the notebook source, or if any code cell
does not compile.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
nb_id = sys.argv[1]
tokens = sys.argv[2:]
p = os.path.join(HERE, f'{nb_id}.ipynb')
nb = json.load(open(p))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])

bad = []
for t in tokens:
    n = src.count(t)
    print(f'  token {t!r}: {n}')
    if n == 0:
        bad.append(t)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c['source'])
    try:
        compile(s, f'<cell {i}>', 'exec')
    except SyntaxError as e:
        bad.append(f'SYNTAX cell {i}: {e}')

if bad:
    print('CHECK FAILED:', bad)
    sys.exit(1)
print(f'{nb_id}: check OK ({len(tokens)} tokens present, all code cells compile)')
