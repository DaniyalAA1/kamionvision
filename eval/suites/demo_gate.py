"""Suite 6 - the eight rehearsed demo cases as a regression gate. STUB.

`app/demo.py:run_demo` prints and returns 0/1 rather than returning results, so
the first move here is twenty additive lines in that file: extract
`run_case(case, backend) -> dict`, and have `run_demo` and this suite both call
it. No behaviour change, and it is what stops the demo script and the regression
gate from drifting apart.

Existing status expectations stay. What this adds is condition-side assertions
that would have caught the whole bug class:

    tr_clean        grade in {good, excellent} AND multiplier >= 0.97
                    - a clean dealer set must not be punished
    tr_phone        |grade_rank - grade_rank(tr_clean)| <= 1 AND
                    multiplier >= multiplier(tr_clean) - 0.02
    closeups_only   C < 0.60, no premium, pricing still declined
    odometer_lie    unchanged: conflict surfaced, band widened
    unseen_brand    band widened, still prices

`tr_phone` is a DIFFERENT vehicle's twins, so it is unpaired and therefore a
weak check. Add a ninth case, `tr_clean_degraded`: the degraded twins of the
same listing that backs `tr_clean`. One `_copy` in `build_fixtures`, and it puts
the headline defence - same truck, worse photos, same grade - on stage in front
of judges instead of only in a scorecard.
"""
from __future__ import annotations

from .base import Budget, NotImplementedSuite, samples_for

NAME = "demo_gate"

# The ninth case is proposed by the design and does not exist yet; the budget
# assumes it, which over-estimates by one case until it lands.
N_CASES = 9


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    from .base import calls_per_vehicle
    # Two cases refuse at the gate and never reach a vision call; the rest are
    # small fixture sets rather than full 16-photo appraisals.
    priced, photos_each = N_CASES - 2, 8
    per = calls_per_vehicle(photos_each, samples=samples_for(tier))
    return Budget(name=NAME, tier=tier, seed=seed, n_calls=priced * per,
                  units=N_CASES, unit="case",
                  params={"cases": N_CASES, "reaching_vision": priced,
                          "photos_each": photos_each, "calls_per_case": per},
                  note="cheapest suite - run it FIRST so a broken build fails "
                       "before the expensive suites spend anything")


def run(plan_obj, client, **kwargs) -> list[dict]:
    raise NotImplementedSuite("demo_gate.run is not implemented yet")


def score(records, plan_obj, cache, model_id):
    raise NotImplementedSuite("demo_gate.score is not implemented yet")
