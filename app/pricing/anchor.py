"""Price a truck from what it cost new, for when no comparable exists.

The hedonic fit is 84 Turkish listings and 78 of them are Ford. Ask it about a
Mercedes and it has nothing to compare against, so today it prices off the
`other` brand column and widens the band 1.85x - a measured number, but a
measurement of ignorance. The widest leave-one-brand-out holdout needed 3.25x
at 75% median error. Judges bring unseen photos; this is the likeliest way the
system fails in the room.

A second route avoids the problem. If we know what this model costs new, and
we know how heavy tractors retain value with age and distance, we can price it
without ever having seen one sold. The retention curve is fit on the corpus and
is a better model of it than the hedonic fit is - out-of-fold R^2 0.911 against
0.842 - because the published new price carries brand, segment and specification
that three brand dummies cannot.

Anchoring on TODAY's price for the CURRENT model is deliberate and is what makes
this safe in Turkiye. Nominal Turkish prices move with roughly 31% CPI, so a
2019 sticker price and a 2026 asking price are not the same units. Both sides of
`price / new_price` are current, which removes the problem instead of modelling
it.

The two routes are combined by inverse variance, so each speaks in proportion to
how well it has been shown to work. When the brand is unknown to the hedonic fit
its variance is inflated by the measured unknown-brand factor, and the anchor
takes over on its own merits rather than by a rule that says it should.

**The two routes are no longer independent, and that is a stated cost.** The
argument above - that a published price carries what three brand dummies cannot
- was strong enough that `log(list_price)` is now a COLUMN in the hedonic
design as well (`features.NEW_PRICE_COLUMNS`), where it takes out-of-fold R2
from 0.842 to 0.951 and halves the 80% band at the same measured coverage. The
hedonic fit with that column in it is a generalisation of this file: the
retention curve is the same regression with the coefficient on log(new price)
pinned at 1.0 and no brand or market terms, and the free coefficient comes out
near +2.4. So the same reference figure now reaches the estimate twice, and an
inverse-variance blend of two estimates that share an input is a blend that
believes itself more than it should. Two things hold the line, and neither is
optional: `estimate` floors the blended band factor at 1.0 so the blend can
claw a widening back but never cut below the interval whose coverage was
measured, and a row whose `source_type` is not `oem_official` now widens the
band itself rather than only this route's variance - a mistranscribed figure
used to be able to hurt only the anchor, and can now move the hedonic estimate
by roughly 2.4x its own error.
"""
from __future__ import annotations

import json
import math
import threading

from ..config import REPO
from ..schema import AnchorEstimate
from .features import KM_FLOOR, REF_YEAR, normalise_brand

NEW_PRICES = REPO / "data" / "reference" / "new_prices_tr.json"

# The corpus records twelve pre-F-MAX Ford tractors under the model name
# "TRUCKS". Their current equivalent is the Cargo-derived 1845T, not the F-MAX
# that the brand default would otherwise pick - a ~9% difference in the anchor.
#
# This is NOT alias folding and the two must not be merged. `modelspec`
# answers "what model is this, given a vision model's free text" - "F Max",
# "FMAX" and "F-MAX 500" are all the F-MAX. This table answers a different
# question: "which priced reference row is the current equivalent of a model
# that is no longer sold". A 2014 Cargo tractor really is a TRUCKS; it is
# simply not buyable new, so its anchor has to borrow the row of the truck that
# replaced it. Folding runs first and this runs on the folded id, so every
# spelling of the Cargo family - "CARGO", "1848T", "FORD TRUCKS" - reaches the
# 1845T row, where before only the literal string "TRUCKS" did.
MODEL_ALIASES = {("FORD", "TRUCKS"): "1845T"}

# A trade-press figure is a real number from a real list, but nobody at the
# manufacturer stands behind our transcription of it. Widen its contribution
# rather than either trusting it equally or throwing it away. Stated assumption.
SOURCE_WIDENING = {"oem_official": 1.0, "trade_press": 1.35}

_REFERENCE: dict | None = None
_LOCK = threading.Lock()


def reference() -> dict:
    global _REFERENCE
    with _LOCK:
        if _REFERENCE is None:
            if not NEW_PRICES.exists():
                raise FileNotFoundError(f"{NEW_PRICES} not found - it is hand-curated, see the "
                                        "'refresh' key in the file for what maintaining it means")
            _REFERENCE = json.loads(NEW_PRICES.read_text(encoding="utf-8"))
        return _REFERENCE


def lookup(make: str | None, model: str | None) -> dict | None:
    """(brand, model) -> a priced reference row, or the brand default, or None.

    The model string arrives from three places that spell it three ways: the
    corpus ("F-MAX", "TRUCKS"), a seller's form, and a vision model that has
    just read a badge and may answer "F Max" or "F-MAX 500". This used to match
    `str(model).strip().upper()` against the rows exactly, so two of those three
    fell through to the brand default without saying so.

    Two folds, in this order and not the other:

      1. `modelspec.normalise_model` folds free text onto the canonical model
         id the reference files agree on. Soft: with `models_tr.json` absent it
         returns the input uppercased, which is byte-for-byte what this
         function did before it existed.
      2. `MODEL_ALIASES` maps a canonical id with no current equivalent onto
         the row of the truck that replaced it.

    Order matters because step 2's keys are canonical ids. Run it first and
    "CARGO" never becomes "TRUCKS" and so never reaches the 1845T row; run it
    second and every spelling of the family does.
    """
    try:
        rows = [r for r in reference()["rows"] if r.get("list_price")]
    except FileNotFoundError:
        return None
    brand = normalise_brand(make)
    if brand == "other":
        return None
    from .. import modelspec
    wanted = modelspec.normalise_model(make, model) or str(model or "").strip().upper()
    wanted = MODEL_ALIASES.get((brand, wanted), wanted)
    for row in rows:
        if row["brand"] == brand and str(row["model"]).upper() == wanted:
            return row
    for row in rows:
        if row["brand"] == brand and row.get("default"):
            return row
    return None


def estimate(anchor_cfg: dict | None, *, year, km, make, model,
             market: str = "TR") -> AnchorEstimate:
    """New price x fitted retention -> a standalone estimate with its own error."""
    est = AnchorEstimate(currency="TRY")
    if market.upper() != "TR":
        est.reason = "the new-price reference covers the Turkish market only"
        return est
    if not anchor_cfg:
        est.reason = "no retention curve in the price model - refit with app.pricing.train"
        return est
    if year is None or km is None:
        est.reason = "needs a year and a distance"
        return est

    row = lookup(make, model)
    if row is None:
        est.reason = f"no published new price for {normalise_brand(make)}"
        return est

    age = max(0.0, REF_YEAR - float(year))
    x = [math.log1p(age), math.log(max(KM_FLOOR, float(km)))]
    log_retention = _predict(anchor_cfg, x)

    est.ok = True
    est.new_price = float(row["list_price"])
    est.retention = round(math.exp(log_retention), 4)
    est.point = round(est.new_price * est.retention, -3)
    est.matched = f"{row['brand']} {row['model']}"
    est.source = row.get("source", "")
    est.source_type = row.get("source_type", "")
    est.source_url = row.get("source_url", "")
    est.as_of = row.get("as_of", "")
    est.basis = (f"{est.new_price:,.0f} TRY new ({est.matched}, {est.source_type.replace('_', ' ')}, "
                 f"as of {est.as_of}) x {est.retention:.1%} retained at "
                 f"{age:.0f} years and {float(km):,.0f} km")
    return est


def _predict(cfg: dict, x: list[float]) -> float:
    mean, scale, coef = cfg["mean"], cfg["scale"], cfg["coef"]
    z = [(xi - m) / (s or 1.0) for xi, m, s in zip(x, mean, scale)]
    return sum(zi * c for zi, c in zip(z, coef)) + cfg["intercept"]


def sigma(anchor_cfg: dict, est: AnchorEstimate) -> float:
    """The anchor's own log-space error, widened for a weaker source."""
    base = float(anchor_cfg.get("residual_std", 0.10))
    return base * SOURCE_WIDENING.get(est.source_type, 1.35)


def blend(mu_regression: float, sd_regression: float,
          mu_anchor: float, sd_anchor: float) -> tuple[float, float, float]:
    """Inverse-variance combine in log space -> (mu, sd, anchor's weight).

    Nothing here privileges either route. A route with twice the error gets a
    quarter of the say, which is the whole reason the unknown-brand case works:
    inflating the hedonic variance by the measured 1.85x is what hands the
    decision to the anchor, rather than a rule that says to.

    Inverse variance is the right combination for INDEPENDENT estimates, and
    since the published new price became a column in the hedonic design these
    two are not independent - see the module docstring. The consequence is a
    `sd` that is too small, which is exactly why the caller floors the band
    factor it derives from this at 1.0 rather than trusting it downwards.
    """
    wr, wa = 1.0 / max(sd_regression, 1e-6) ** 2, 1.0 / max(sd_anchor, 1e-6) ** 2
    total = wr + wa
    return (wr * mu_regression + wa * mu_anchor) / total, math.sqrt(1.0 / total), wa / total
