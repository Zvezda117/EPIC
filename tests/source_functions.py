"""Load real numerical functions without importing observatory/network clients."""
import ast
import csv
import datetime
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/python'))


def epic_functions(*names):
    path = ROOT / 'src/python/epic_utils.py'
    tree = ast.parse(path.read_text())
    module = ast.Module(body=[node for node in tree.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name in names], type_ignores=[])
    scope = dict(np=np, os=os, Path=Path, csv=csv, datetime=datetime,
                 __file__=str(path), CHAN_BW=25000.0, FS=196000000.0)
    exec(compile(module, str(path), 'exec'), scope)
    return scope
