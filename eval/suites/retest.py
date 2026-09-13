"""Suite 2 - test-retest variance. STUB: plan() works, run/score do not.

No backend in `app/vlm/` exposes a temperature or a seed - verified in
`openai_backend.py` and `anthropic_backend.py`, both of which send neither - so
repeated calls genuinely vary at provider defaults. That variance is not a
nuisance to be averaged away here: it is the measurement.

The payoff is concrete and is why this suite pays for itself. `multiplier_sd_log`
is the spread of log(condition multiplier) across repeats of the SAME vehicle,
and it is written straight into `models/price_model.json` as the MEASURED
condition widening. Today that widening is a stated assumption.

Photo tier (24 photos x N=5 repeats):
    findings_count_cv        sd/mean of finding count per photo
    severity_disagreement    P(two repeats assign different severity to a
                             matched finding), over all C(5,2)=10 repeat pairs
    presence_instability     mean over findings of (1 - times_seen/N) - how
                             often does a finding simply vanish
    krippendorff_alpha       ordinal alpha over severity on matched findings

Vehicle tier (4 vehicles x N=3 full appraisals):
    grade_instability        P(two repeats give a different DETERMINISTIC grade)
    model_grade_instab       the same for the synthesis's own grade. The gap
                             between these two is the evidence for preferring
                             the deterministic ladder.
    multiplier_sd_log        sd of log(condition multiplier)  <- the shipped number

Implementation notes for whoever fills this in:
  * Repeats are `repeat_index` in the cache key. They are ALREADY separate
    entries, so a retest run over photos the twin suite has touched is partly
    free - but only for repeat indices below `config.CLOSEUP_SAMPLES`.
  * Match findings across repeats with `twin_fp.cluster_observations`; do not
    write a third matcher.
  * The vehicle tier must drive the real `pipeline.appraise` through a
    `CachedBackend`, because the whole point is the variance of the shipped
    path, not of pass B alone.
"""
from __future__ import annotations

from app import config

from .base import Budget, NotImplementedSuite, samples_for

NAME = "retest"

# (photos, photo repeats, vehicles, vehicle repeats)
TIER_PLAN = {"smoke": (8, 3, 1, 2), "standard": (24, 5, 4, 3), "full": (48, 5, 8, 3)}


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    from .base import calls_per_vehicle
    photos, photo_n, vehicles, vehicle_n = TIER_PLAN.get(tier, TIER_PLAN["standard"])
    photo_calls = photos * photo_n
    vehicle_calls = vehicles * vehicle_n * calls_per_vehicle(samples=samples_for(tier))
    return Budget(name=NAME, tier=tier, seed=seed,
                  n_calls=photo_calls + vehicle_calls,
                  units=photos + vehicles, unit="photo+vehicle",
                  params={"photos": photos, "photo_repeats": photo_n,
                          "vehicles": vehicles, "vehicle_repeats": vehicle_n,
                          "photo_calls": photo_calls, "vehicle_calls": vehicle_calls,
                          "calls_per_vehicle": calls_per_vehicle(samples=samples_for(tier))},
                  note="repeats are distinct repeat_index keys, so indices below "
                       f"CLOSEUP_SAMPLES={config.CLOSEUP_SAMPLES} may already be cached")


def run(plan_obj, client, **kwargs) -> list[dict]:
    raise NotImplementedSuite("retest.run is not implemented yet")


def score(records, plan_obj, cache, model_id):
    raise NotImplementedSuite("retest.score is not implemented yet")
