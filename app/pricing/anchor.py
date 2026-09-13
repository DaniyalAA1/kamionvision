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
    """(brand, model) -> a priced reference row, or the brand default, or None."""
    try:
        rows = [r for r in reference()["rows"] if r.get("list_price")]
    except FileNotFoundError:
        return None
    brand = normalise_brand(make)
    if brand == "other":
        return None
    wanted = str(model or "").strip().upper()
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
    """
    wr, wa = 1.0 / max(sd_regression, 1e-6) ** 2, 1.0 / max(sd_anchor, 1e-6) ** 2
    total = wr + wa
    return (wr * mu_regression + wa * mu_anchor) / total, math.sqrt(1.0 / total), wa / total
