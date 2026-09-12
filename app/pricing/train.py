"""Fit the price model, measure it honestly, and write models/price_model.json.

    .venv/bin/python -m app.pricing.train

Two evaluation rules do the real work here.

**Folds are grouped by spec.** 87 of the 155 priced listings sit in 19 groups
of identical (market, brand, model, year, price) - a dealer listing the same
2022 F-MAX twenty-six times. Splitting on rows would put an identical truck on
both sides of the boundary and score memorisation, exactly as splitting images
by frame rather than by vehicle would.

**The interval is calibrated, not asserted.** Within every split the band is
derived from the TRAINING residuals only and then checked against the held-out
rows. The number that ends up on the report card - "our 80% band contained the
real asking price N% of the time across M held-out trucks" - is that
measurement, not a claim.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ..config import FX_AS_OF, LISTINGS_CSV, PRICE_MODEL, USD_TRY
from . import features as F
from .model import PriceModel

ALPHA = 1.0               # ridge penalty; the design matrix is small and correlated
LEVELS = (0.5, 0.8, 0.9)
N_CALIBRATION_SPLITS = 40
TEST_SIZE = 0.25
SEED = 7


def _fit(X: np.ndarray, y: np.ndarray, alpha: float = ALPHA):
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    model = Ridge(alpha=alpha).fit((X - mean) / scale, y)
    return model, mean, scale


def _predict(model, mean, scale, X: np.ndarray) -> np.ndarray:
    return model.predict((X - mean) / scale)


def _offsets(residuals: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    return (float(np.quantile(residuals, tail)), float(np.quantile(residuals, 1 - tail)))


def evaluate(df: pd.DataFrame, brands: list[str], *, score_market: str | None = None,
             drop_brand: bool = False) -> dict:
    """Grouped out-of-fold metrics plus measured interval coverage."""
    X = F.matrix(df, brands)
    if drop_brand:
        keep = len(["log1p_age", "log_km", "market_tr", "euro6"])
        X = X[:, :keep]
    y = df.y.to_numpy()
    groups = df.group.to_numpy()
    n_groups = len(set(groups))

    # --- out-of-fold point accuracy --------------------------------------
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups):
        model, mean, scale = _fit(X[tr], y[tr])
        oof[te] = _predict(model, mean, scale, X[te])

    mask = np.ones(len(df), dtype=bool)
    if score_market:
        mask = (df.market.str.upper() == score_market.upper()).to_numpy()

    resid = y[mask] - oof[mask]
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
    out = {
        "n": int(mask.sum()),
        "n_groups": n_groups,
        "r2_oof": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
        # Back in price space: a log residual of 0.10 is a ~10% miss.
        "mae_pct_oof": round(float(np.mean(np.abs(np.expm1(resid)))) * 100, 1),
        "median_ape_oof": round(float(np.median(np.abs(np.expm1(resid)))) * 100, 1),
        "residual_std_oof": round(float(np.std(resid)), 4),
    }

    # --- interval coverage ------------------------------------------------
    # The band is built from OUT-OF-FOLD residuals inside the training half,
    # never from in-sample ones. A ridge fit on ~18 spec groups has in-sample
    # residuals far tighter than its real prediction error, and building the
    # band from those produced an "80%" interval that covered 70% - the exact
    # failure this whole calibration step exists to catch.
    splitter = GroupShuffleSplit(n_splits=N_CALIBRATION_SPLITS, test_size=TEST_SIZE,
                                 random_state=SEED)
    hits = {level: [] for level in LEVELS}
    widths = {level: [] for level in LEVELS}
    for tr, te in splitter.split(X, y, groups):
        sel = mask[te]
        if not sel.any():
            continue
        model, mean, scale = _fit(X[tr], y[tr])
        pred_te = _predict(model, mean, scale, X[te])

        inner_groups = groups[tr]
        n_inner = min(5, len(set(inner_groups)))
        if n_inner < 2:
            continue
        inner_oof = np.full(len(tr), np.nan)
        for a, b in GroupKFold(n_splits=n_inner).split(X[tr], y[tr], inner_groups):
            m, mu_, sc_ = _fit(X[tr][a], y[tr][a])
            inner_oof[b] = _predict(m, mu_, sc_, X[tr][b])
        band_resid = y[tr] - inner_oof

        actual, pred = y[te][sel], pred_te[sel]
        for level in LEVELS:
            lo, hi = _offsets(band_resid, level)
            hits[level].extend(((actual >= pred + lo) & (actual <= pred + hi)).tolist())
            widths[level].extend([float(np.expm1(hi - lo))] * int(sel.sum()))
    for level in LEVELS:
        out[f"coverage_{level}"] = round(float(np.mean(hits[level])), 4) if hits[level] else None
        out[f"band_width_pct_{level}"] = round(float(np.mean(widths[level])) * 100, 1) if widths[level] else None
    out["coverage_n"] = len(hits[LEVELS[1]])
    out["coverage_splits"] = N_CALIBRATION_SPLITS
    return out



def leave_one_brand_out(df: pd.DataFrame, brands: list[str], level: float = 0.8) -> dict:
    """What actually happens when a make the model has never fitted walks in.

    The "refit without the brand columns" measurement is not informative on a
    corpus that is 93% one brand - it can only report that brand barely
    matters *here*. This does the honest version: hold out an entire make,
    fit on the rest, and price that make as an unknown. The widening factor is
    how much the band has to grow before it covers at its nominal rate.

    It is the single most demo-relevant number in the file, because the brief
    hands you photos of a truck you have never seen.
    """
    results = {}
    factors = []
    for held in [b for b in brands if b != "other"]:
        train = df[df.brand != held]
        test = df[df.brand == held]
        if len(test) < 5 or train.brand.nunique() < 2:
            continue
        train_brands = F.brand_vocabulary(train, min_n=F.MIN_BRAND_N)
        Xtr, ytr = F.matrix(train, train_brands), train.y.to_numpy()
        Xte, yte = F.matrix(test, train_brands), test.y.to_numpy()

        model, mean, scale = _fit(Xtr, ytr)
        # Band from out-of-fold residuals on the training brands only.
        gtr = train.group.to_numpy()
        oof = np.full(len(train), np.nan)
        for a, b in GroupKFold(n_splits=min(5, len(set(gtr)))).split(Xtr, ytr, gtr):
            m, mu_, sc_ = _fit(Xtr[a], ytr[a])
            oof[b] = _predict(m, mu_, sc_, Xtr[a.size and b])
        oof = np.full(len(train), np.nan)
        for a, b in GroupKFold(n_splits=min(5, len(set(gtr)))).split(Xtr, ytr, gtr):
            m, mu_, sc_ = _fit(Xtr[a], ytr[a])
            oof[b] = _predict(m, mu_, sc_, Xtr[b])
        band = ytr - oof
        lo, hi = _offsets(band, level)
        pred = _predict(model, mean, scale, Xte)

        covered = float(np.mean((yte >= pred + lo) & (yte <= pred + hi)))
        # Smallest symmetric inflation of the band that reaches the nominal rate.
        factor = 1.0
        for f in np.arange(1.0, 4.01, 0.05):
            if float(np.mean((yte >= pred + lo * f) & (yte <= pred + hi * f))) >= level:
                factor = float(f)
                break
        else:
            factor = 4.0
        results[held] = {
            "n_held_out": int(len(test)),
            "coverage_unwidened": round(covered, 3),
            "widening_needed": round(factor, 2),
            "median_ape_pct": round(float(np.median(np.abs(np.expm1(yte - pred)))) * 100, 1),
        }
        factors.append(factor)
    results["_summary"] = {
        "brands_tested": len(factors),
        "max_widening_needed": round(max(factors), 2) if factors else None,
        "median_widening_needed": round(float(np.median(factors)), 2) if factors else None,
    }
    return results


def main() -> None:
    listings = pd.read_csv(LISTINGS_CSV)
    df = F.build_frame(listings)
    brands = F.brand_vocabulary(df)
    tr = df[df.market.str.upper() == "TR"]

    print(f"priced listings: {len(df)}   distinct specs (fold groups): {df.group.nunique()}")
    print(f"brands with their own column: {brands}")
    print(f"TR: {len(tr)} listings, {tr.group.nunique()} specs\n")

    # --- which training set prices a Turkish truck best? ------------------
    candidates = {
        "pooled_tr_us": (df, brands),
        "tr_only": (tr, F.brand_vocabulary(tr, min_n=3)),
    }
    comparison = {}
    for name, (frame, brand_set) in candidates.items():
        comparison[name] = evaluate(frame, brand_set, score_market="TR")
        c = comparison[name]
        print(f"{name:14s} scored on TR held-out: n={c['n']:3d} R2={c['r2_oof']:.3f} "
              f"MAE={c['mae_pct_oof']:.1f}%  cov@0.8={c['coverage_0.8']:.3f} "
              f"(band +/-{c['band_width_pct_0.8'] / 2:.0f}%)")

    # Prefer the pooled fit unless TR-only is clearly better on TR trucks.
    # Pooling is not free - it assumes the age and km slopes transfer across
    # markets - so it has to earn its place on the market being demoed.
    pooled_better = (comparison["pooled_tr_us"]["r2_oof"] >= comparison["tr_only"]["r2_oof"] - 0.02)
    chosen = "pooled_tr_us" if pooled_better else "tr_only"
    fit_df, fit_brands = candidates[chosen]
    print(f"\nchosen training set: {chosen}")

    overall = evaluate(fit_df, fit_brands)
    no_brand = evaluate(fit_df, fit_brands, drop_brand=True)

    # How much wider does the band have to be for a make we never fitted?
    # Measured by holding out entire brands, pooled, because the TR-only
    # corpus has too few makes to answer the question at all.
    lobo = leave_one_brand_out(df, brands)
    print("\nleave-one-brand-out (pooled) - pricing a make the fit never saw:")
    for brand, r in lobo.items():
        if brand.startswith("_"):
            continue
        print(f"  {brand:14s} n={r['n_held_out']:3d}  80% band covered "
              f"{r['coverage_unwidened']:.2f} unwidened, needed {r['widening_needed']:.2f}x, "
              f"median error {r['median_ape_pct']:.0f}%")
    widen = lobo["_summary"]["median_widening_needed"] or 1.5
    widen = round(max(1.2, min(4.0, widen)), 2)
    print(f"  -> unseen-brand widening set to {widen:.2f}x (median across held-out makes)")
    print(f"brand columns are worth {overall['r2_oof'] - no_brand['r2_oof']:+.3f} R2 within the corpus")

    # --- final fit on everything -----------------------------------------
    X = F.matrix(fit_df, fit_brands)
    y = fit_df.y.to_numpy()
    ridge, mean, scale = _fit(X, y)

    oof = np.full(len(fit_df), np.nan)
    groups = fit_df.group.to_numpy()
    for a, b in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
        m, mu_, sc_ = _fit(X[a], y[a])
        oof[b] = _predict(m, mu_, sc_, X[b])
    oof_resid = y - oof

    tr_brand_counts = tr.brand.value_counts().to_dict()
    top_brand, top_n = max(tr_brand_counts.items(), key=lambda kv: kv[1])
    model = PriceModel(
        brands=list(fit_brands),
        columns=F.design_columns(list(fit_brands)),
        coef=[round(float(c), 6) for c in ridge.coef_],
        intercept=round(float(ridge.intercept_), 6),
        mean=[round(float(v), 6) for v in mean],
        scale=[round(float(v), 6) for v in scale],
        residual_std=round(float(np.std(oof_resid)), 4),
        offsets={str(level): [round(v, 4) for v in _offsets(oof_resid, level)]
                 for level in LEVELS},
        calibration={**overall, "scored_on": "all markets",
                     "tr_only_view": comparison["pooled_tr_us"] if chosen == "pooled_tr_us"
                     else comparison["tr_only"],
                     "without_brand_columns": no_brand},
        widening={"unknown_brand": widen,
                  "unknown_brand_basis": (
                      "measured by leave-one-brand-out: entire makes were held out of the "
                      "pooled fit and priced as unknowns; this is the median band inflation "
                      "needed to restore 80% coverage on them"),
                  "leave_one_brand_out": lobo},
        meta={
            "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "training_set": chosen,
            "n_listings": int(len(fit_df)),
            "n_groups": int(fit_df.group.nunique()),
            "target": "log(asking price in USD)",
            "fx": {"usd_try": USD_TRY, "as_of": FX_AS_OF},
            "markets": sorted(fit_df.market.str.upper().unique().tolist()),
            "tr_brand_note": f"{top_n} of {len(tr)} Turkish listings are {top_brand}",
            "alpha": ALPHA,
            "grouping": ("folds grouped on (market, brand, model, year, price); "
                         f"{len(fit_df)} listings collapse to {fit_df.group.nunique()} specs"),
        },
    )
    model.save(PRICE_MODEL)

    print(f"\nresidual std (out-of-fold): {model.residual_std:.4f} log "
          f"= +/-{(np.exp(model.residual_std) - 1) * 100:.1f}% - this is the condition cap")
    for level in LEVELS:
        lo, hi = model.offsets[str(level)]
        print(f"  {int(level * 100)}% band: {np.expm1(lo) * 100:+.0f}% .. {np.expm1(hi) * 100:+.0f}%   "
              f"measured coverage {overall[f'coverage_{level}']:.3f} "
              f"over {overall['coverage_n']} held-out trucks")
    print(f"\nwrote {PRICE_MODEL}")


if __name__ == "__main__":
    main()
