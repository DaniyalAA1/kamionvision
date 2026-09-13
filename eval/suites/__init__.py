"""The six suites, and the budget arithmetic they share.

Every suite exposes the same three entry points, and the split between them is
the whole economics of this package:

    plan(tier, seed, ...)          -> a Budget or Plan. Spends nothing. Where it
                                      can, resolves every cache key so that
                                      `--estimate` is an exact count of new
                                      calls rather than a guess.
    run(plan, client, ...)         -> the unit index. The only step that can
                                      cost money, and it does not cost any when
                                      the cache already holds the keys.
    score(records, plan, cache, m) -> a SuiteResult. PURE, offline, replayable.
                                      Every change downstream of pass B is
                                      re-measured here for nothing.

`twin_fp` is implemented. The other five carry working `plan()`s - so
`--estimate --tier full` can budget the whole tier before anyone writes them -
and `run`/`score` that raise, with the design's metric list in the docstring
for whoever fills them in.
"""
from __future__ import annotations

from .base import Budget, NotImplementedSuite, calls_per_vehicle
from . import demo_gate, distribution, monotonic, panel, retest, twin_fp

SUITES = {
    twin_fp.NAME: twin_fp,
    retest.NAME: retest,
    monotonic.NAME: monotonic,
    distribution.NAME: distribution,
    panel.NAME: panel,
    demo_gate.NAME: demo_gate,
}

# Order matters: the cheap regression gate first so a broken build fails before
# the expensive suites spend anything, and `monotonic` after `distribution`
# because it scores off the same cached sweep.
DEFAULT_ORDER = ("demo_gate", "twin_fp", "retest", "distribution", "monotonic", "panel")

__all__ = ["SUITES", "DEFAULT_ORDER", "Budget", "NotImplementedSuite",
           "calls_per_vehicle"]
