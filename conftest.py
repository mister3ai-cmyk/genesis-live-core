"""Makes `python3 -m pytest research/tests/` work from the repository root.

The research modules import each other by top-level name (`sila2_bridge`,
`modules`), which resolves when pytest is invoked from inside `research/` and
fails from the root: collection stopped on
`ModuleNotFoundError: No module named 'sila2_bridge'` and took the whole run
with it. A reviewer should not have to know which directory to stand in.
"""
import os
import sys

RESEARCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
if RESEARCH not in sys.path:
    sys.path.insert(0, RESEARCH)
