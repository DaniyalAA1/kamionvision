"""The measurement instrument, kept outside the thing it measures.

Why this package exists, stated plainly because the reason is the design:
`app/vlm/bench.py` — the benchmark that selected the shipped model — scores a
backend on finding count, truck-vocabulary share and mean observation length,
on one fixture, with no ground truth of any kind. A model that hallucinates
twice as many long, truck-flavoured findings scores strictly better on every
column it has. The development loop was rewarding the bug it was meant to
catch, and every number it ever printed should be read that way.

There is no condition ground truth in this corpus either: `damaged` is null in
100% of 7,458 rows, there are no Turkish hasar/tramer/boyalı fields, and
`data/DATASET_CARD.md` states this as limitation 3. So the harness cannot score
against labels. It manufactures measurement out of structure instead:

  twin_fp       a degraded twin is the SAME truck, so a finding on the twin
                with no counterpart on the original is a false positive by
                construction. No labels needed - the pairing is the label.
  retest        no backend exposes a seed, so repeats vary. The spread IS the
                measured condition widening, not an assertion about it.
  monotonic     wear should not fall with kilometres. A direction check.
  distribution  the grade histogram against a prior derived from the dataset
                card, labelled as the assumption it is.
  panel         the only external reference, and always scored against the
                leave-one-panelist-out ceiling rather than against 1.0.
  demo_gate     the eight rehearsed cases as a regression gate.

`eval/` sits at the repo root rather than in `scripts/` (documented as the
dataset pipeline) or `app/` (the product). It is an instrument over the
product and it bears artifacts the way `models/` and `data/` do.

Two rules the rest of the package is built around:

  * The response cache stores raw response TEXT, never parsed objects. Pass B
    is 16 of 18 calls per vehicle, so caching it makes every change downstream
    of it - `parse_closeup`, `merge_duplicates`, the rollup, the grade ladder,
    the price multiplier - replayable at zero vision calls.
  * Every headline metric carries a confidence interval and `--diff` says
    `within noise` when a delta sits inside the before-run's interval. At
    n = 40-60 vehicles most deltas WILL be noise. A harness that reports noise
    as progress is worse than no harness.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Tiers are named here rather than in each suite so `--estimate --tier full`
# can budget suites nobody has implemented yet.
TIERS = ("smoke", "standard", "full")

# New vision calls above which a run refuses to start without `--yes`. Cost
# control belongs in the tool that spends the money, not in the operator's
# memory of what a tier costs.
CONFIRM_ABOVE_CALLS = 500
