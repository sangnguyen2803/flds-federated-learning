#!/usr/bin/env python
"""Quick smoke test: can we import VerticalFL from the installed fluke package?"""
import sys
import importlib

print("Python:", sys.executable)

try:
    mod = importlib.import_module("fluke.algorithms.vertical")
    print("Module loaded:", mod)
    cls = getattr(mod, "VerticalFL")
    print("Class found:", cls)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", type(e).__name__, e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
