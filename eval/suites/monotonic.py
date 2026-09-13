"""Suite 3 - monotonicity of condition against kilometres and age. STUB.

Shares its corpus sweep with `distribution`: one pass over N vehicles produces
the cached findings and BOTH suites score off them. That halves the headline
cost of the pair, and the budget below reports zero incremental calls for
exactly that reason rather than hiding the sharing in a footnote.

    rho_km        Spearman(S, km) over vehicles, S = the rollup demerit
    rho_age       Spearman(S, age)
    rho_partial   Spearman(S, km | age, capture_quality) via rank residuals
    null          permute km across vehicles WITHIN (market, brand) strata,
                  2000 draws

`capture_quality` is not an optional covariate. Older trucks are photographed
worse, and without partialling it out this test measures photography rather
than wear. The column is already in `images.csv`.

The pre-registered interpretation, which MUST be printed in the scorecard so
nobody over-reads the number:

    The corpus is reconditioned OEM stock (DATASET_CARD limitation 2), so the
    true km -> visible-wear relation is attenuated by construction. This is a
    direction check and a regression guard, not a validation. A significantly
    NEGATIVE rho is a bug. A near-zero rho is expected and uninformative. A
    strongly positive rho is suspicious and should be checked for the
    photography confound first.

The stronger monotonicity evidence in this harness is the dose-response curve
inside `twin_fp.by_severity_band`, where the ordering of the stimulus is known
by construction rather than inferred from an odometer. Rank it above this.
"""
from __future__ import annotations

from .base import Budget, NotImplementedSuite

NAME = "monotonic"


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    from . import distribution
    shared = distribution.plan(tier, seed=seed, model_id=model_id, effort=effort,
                               structured=structured, samples=samples,
                               require_files=require_files)
    return Budget(name=NAME, tier=tier, seed=seed, n_calls=0, units=shared.units,
                  params={"shares_corpus_with": distribution.NAME,
                          "vehicles": shared.units},
                  note="zero incremental calls: scores off the cached findings the "
                       "distribution sweep already paid for")


def run(plan_obj, client, **kwargs) -> list[dict]:
    raise NotImplementedSuite("monotonic.run is not implemented yet")


def score(records, plan_obj, cache, model_id):
    raise NotImplementedSuite("monotonic.score is not implemented yet")
