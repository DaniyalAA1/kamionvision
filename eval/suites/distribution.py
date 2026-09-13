"""Suite 4 - the grade histogram over the corpus. STUB: plan() works.

Runs the full pipeline over a stratified vehicle sample and reports the grade
distribution with Wilson intervals, split by market, by `quality_bucket`
(dealer / mixed / phone, using `gallery.py`'s own definition so the buckets mean
on the scorecard what they mean on screen) and by variant.

The prior, written as a falsifiable acceptance band:

    on the TR OEM subset:   P(good or excellent) >= 0.70   and   P(poor) <= 0.10

That is an ASSUMPTION derived from DATASET_CARD limitation 2 - the Turkish
corpus is reconditioned dealer stock offered for sale - and the scorecard must
label it that way. It is the strongest cheap prior available, and the panel
reference set is what converts it into a measurement. The expectation is that
the current system fails it badly; that failing number, before and after, is
the single most persuasive line the harness produces.

Also free and high value:
    grade_agreement_model_vs_rollup   how often the synthesis's own grade equals
                                      the deterministic one, and the SIGNED bias
                                      between them

This suite's cached findings are also what `monotonic` and `panel` score off,
so it is the one place where the budget is genuinely shared three ways.
"""
from __future__ import annotations

from .base import Budget, NotImplementedSuite, samples_for

NAME = "distribution"

TIER_VEHICLES = {"smoke": 8, "standard": 40, "full": 200}
STRATA = ("market", "quality_bucket", "km_tercile")


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    from .base import calls_per_vehicle
    n = TIER_VEHICLES.get(tier, TIER_VEHICLES["standard"])
    per = calls_per_vehicle(samples=samples_for(tier))
    return Budget(name=NAME, tier=tier, seed=seed, n_calls=n * per, units=n,
                  params={"vehicles": n, "calls_per_vehicle": per, "strata": list(STRATA),
                          "prior_tr_good_or_excellent": 0.70, "prior_tr_poor": 0.10},
                  note="the acceptance band is an ASSUMPTION from DATASET_CARD "
                       "limitation 2, not a measurement; the panel set is what "
                       "would convert it")


def run(plan_obj, client, **kwargs) -> list[dict]:
    raise NotImplementedSuite("distribution.run is not implemented yet")


def score(records, plan_obj, cache, model_id):
    raise NotImplementedSuite("distribution.score is not implemented yet")
