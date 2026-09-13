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

**A column has to beat the design without it, here, on every refit.** The
published-new-price term is fitted and scored both ways and ships only if it
clears `NEW_PRICE_MIN_GAIN` on out-of-fold R2 without giving back more than
`NEW_PRICE_MAX_COVERAGE_LOSS` of the 80% band's coverage. The rule lives in
code so that a future corpus re-asks the question rather than inheriting this
one's answer, and so that "it helped" is something the run prints rather than
something a commit message asserts.
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

# --- the gate the published-new-price column had to pass --------------------
# Kept in the code rather than in a commit message, so a refit on a different
# corpus re-asks the question instead of inheriting the answer. The column may
# only ship if it adds at least NEW_PRICE_MIN_GAIN to out-of-fold R2 on the
# market being scored AND costs no more than NEW_PRICE_MAX_COVERAGE_LOSS of the
# 80% band's measured coverage. Both halves matter and the second is the one
# that would be easy to leave out: the band is the product, the point estimate
# is not, and buying R2 by quietly decalibrating the interval is not a trade
# this repo makes.
NEW_PRICE_MIN_GAIN = 0.02
NEW_PRICE_MAX_COVERAGE_LOSS = 0.02
NEW_PRICE_LEVEL = 0.8


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


def _held_out_design(train: pd.DataFrame, test: pd.DataFrame, brands: list[str],
                     new_price: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Design matrices for a train/test pair where the test frame is a stranger.

    The new-price column is imputed at a mean and clipped to a support, and
    both of those belong to the fit doing the scoring rather than to the rows
    being scored. Getting this backwards is how a holdout quietly gets told the
    answer: a held-out brand whose own new price sets its own clip range is
    being handed a range it should not have.
    """
    Xtr = F.matrix(train, brands, new_price=new_price)
    if not new_price:
        return Xtr, F.matrix(test, brands, new_price=False)
    values = F.log_new_price_series(train).dropna()
    log_mean = float(values.mean()) if len(values) else None
    log_range = (float(values.min()), float(values.max())) if len(values) else None
    return Xtr, F.matrix(test, brands, new_price=True,
                         log_mean=log_mean, log_range=log_range)


def evaluate(df: pd.DataFrame, brands: list[str], *, score_market: str | None = None,
             drop_brand: bool = False, new_price: bool = True) -> dict:
    """Grouped out-of-fold metrics plus measured interval coverage."""
    X = F.matrix(df, brands, new_price=new_price)
    if drop_brand:
        # By name, not by a count. This used to slice the first four columns
        # and call the rest brand, which silently dropped `euro6` as well and
        # made "brand columns are worth X R2" a measurement of brand AND
        # emissions class.
        cols = F.design_columns(brands, new_price=new_price)
        X = X[:, [i for i, c in enumerate(cols) if not c.startswith("brand_")]]
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



def leave_one_brand_out(df: pd.DataFrame, level: float = 0.8,
                        new_price: bool = True) -> dict:
    """What happens when a make the model has never fitted walks in.

    Done WITHIN a market, never across. On this corpus brand is nearly
    collinear with market - Türkiye is 93% Ford, the European stock is 95%
    Mercedes - so holding a make out of the pooled frame removes an entire
    market with it, and what comes back is the cost of extrapolating across a
    border, not across a brand. Measured that way the numbers were absurd:
    holding out Ford reported a 442% median error, because the model was left
    estimating the Turkish intercept from six MAN listings.

    Restricted to one market at a time it answers the real question, and the
    European stock is the only market with enough makes to ask it.
    """
    results, factors = {}, []
    for market, group in df.groupby(df.market.str.upper()):
        brands_here = F.brand_vocabulary(group, min_n=F.MIN_BRAND_N)
        if len([b for b in brands_here if b != "other"]) < 2:
            continue
        for held in [b for b in brands_here if b != "other"]:
            train = group[group.brand != held]
            test = group[group.brand == held]
            if len(test) < 5 or train.brand.nunique() < 2:
                continue
            train_brands = F.brand_vocabulary(train, min_n=F.MIN_BRAND_N)
            Xtr, Xte = _held_out_design(train, test, train_brands, new_price)
            ytr, yte = train.y.to_numpy(), test.y.to_numpy()

            model, mean, scale = _fit(Xtr, ytr)
            gtr = train.group.to_numpy()
            oof = np.full(len(train), np.nan)
            for a, b in GroupKFold(n_splits=min(5, len(set(gtr)))).split(Xtr, ytr, gtr):
                m, mu_, sc_ = _fit(Xtr[a], ytr[a])
                oof[b] = _predict(m, mu_, sc_, Xtr[b])
            lo, hi = _offsets(ytr - oof, level)
            pred = _predict(model, mean, scale, Xte)

            covered = float(np.mean((yte >= pred + lo) & (yte <= pred + hi)))
            factor = 4.0
            for f in np.arange(1.0, 4.01, 0.05):
                if float(np.mean((yte >= pred + lo * f) & (yte <= pred + hi * f))) >= level:
                    factor = float(f)
                    break
            results[f"{market}:{held}"] = {
                "market": market, "brand": held,
                "n_held_out": int(len(test)),
                "trained_on_brands": [b for b in train_brands if b != "other"],
                "coverage_unwidened": round(covered, 3),
                "widening_needed": round(factor, 2),
                "median_ape_pct": round(float(np.median(np.abs(np.expm1(yte - pred)))) * 100, 1),
            }
            factors.append(factor)
    results["_summary"] = {
        "holdouts": len(factors),
        "max_widening_needed": round(max(factors), 2) if factors else None,
        "median_widening_needed": round(float(np.median(factors)), 2) if factors else None,
        "method": "within-market leave-one-brand-out",
    }
    return results


def fit_retention(df: pd.DataFrame) -> dict:
    """log(asking / published new price) ~ log1p(age) + log(km), on TR rows.

    The comparables route needs a same-brand comparable to exist. This one
    needs only a published new price, which is why it is what answers an
    unseen make. It is fit here rather than in anchor.py so that one command
    refits every number the pricing stage relies on, and so its folds are the
    same spec-grouped folds the hedonic model uses - 26 of the Turkish rows are
    the same F-MAX at the same asking price, and splitting them would score
    memorisation.
    """
    from . import anchor as anchor_mod

    tr = df[df.market.str.upper() == "TR"].copy()
    new = [anchor_mod.lookup(r.make if hasattr(r, "make") else r.brand, r.model) for r in tr.itertuples()]
    tr["new_price"] = [float(n["list_price"]) if n else np.nan for n in new]
    tr = tr[tr.new_price.notna() & (tr.new_price > 0)]
    if len(tr) < 20:
        return {"ok": False, "reason": f"only {len(tr)} Turkish listings have a published new price"}

    X = tr[["log1p_age", "log_km"]].to_numpy(dtype=float)
    y = np.log(tr.price.to_numpy(dtype=float) / tr.new_price.to_numpy(dtype=float))
    groups = tr.group.to_numpy()

    oof = np.zeros(len(y))
    for a, b in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
        m, mu_, sc_ = _fit(X[a], y[a])
        oof[b] = _predict(m, mu_, sc_, X[b])
    resid = y - oof
    r2 = 1 - float(np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))

    ridge, mean, scale = _fit(X, y)
    return {
        "ok": True,
        "columns": ["log1p_age", "log_km"],
        "coef": [round(float(c), 6) for c in ridge.coef_],
        "intercept": round(float(ridge.intercept_), 6),
        "mean": [round(float(v), 6) for v in mean],
        "scale": [round(float(v), 6) for v in scale],
        "residual_std": round(float(np.std(resid)), 4),
        "r2_oof": round(r2, 4),
        "median_ape_oof": round(float(100 * np.median(np.abs(np.expm1(-resid)))), 1),
        "n": int(len(tr)), "n_groups": int(tr.group.nunique()),
        "target": "log(asking price / published new price of the current equivalent model)",
        "basis": "out-of-fold, folds grouped on the same (market, brand, model, year, price) "
                 "key the hedonic model uses",
        "reference": "data/reference/new_prices_tr.json",
    }


def anchor_holdout(df: pd.DataFrame, retention: dict, level: float = 0.8,
                   new_price: bool = False) -> dict:
    """Does the new-price anchor actually rescue a brand the fit never saw?

    The claim the anchor is built on is that it prices an unseen make. This
    measures it the only way that counts: hold a whole Turkish brand out of the
    hedonic fit, price its vehicles as unknowns twice - once with the measured
    1.85x widening alone, once with the anchor blended in - and compare band
    coverage and error.

    n is small and honestly so: the Turkish corpus has exactly two makes, so
    holding out Ford leaves six MAN rows to fit on. That is a punishing test
    rather than a flattering one, which is the point.

    `new_price` defaults OFF here alone, and the reason is the whole reason
    this measurement exists. Once the published new price is a COLUMN, the
    hedonic side of this comparison has the reference figure too, and "does the
    anchor rescue an unseen make" stops being a question about the anchor and
    becomes a question about which of two doors the same number walks through.
    Held out with the column on, TR:MAN is extrapolated well past the support
    the coefficient was fitted on - the remaining rows are Ford, spanning
    6.85M-7.43M, and a MAN is 8.67M - which is the clipping rule doing its job
    and is not what this function is asking. So the hedonic baseline here stays
    the design the widening was measured against, and the shipped design's own
    behaviour is reported by `evaluate`.
    """
    from . import anchor as anchor_mod
    from ..schema import AnchorEstimate

    tr = df[df.market.str.upper() == "TR"].copy()
    if not retention.get("ok"):
        return {"_summary": {"ran": False, "reason": "no retention curve"}}
    out: dict = {}

    for brand in sorted(tr.brand.unique()):
        held, rest = tr[tr.brand == brand], tr[tr.brand != brand]
        if len(held) < 3 or len(rest) < 8:
            continue
        brands = F.brand_vocabulary(rest, min_n=3)
        Xr, yr = F.matrix(rest, brands, new_price=new_price), rest.y.to_numpy()
        ridge, mean, scale = _fit(Xr, yr)

        # Band offsets and residual sd from the reduced fit's own OOF residuals.
        groups = rest.group.to_numpy()
        oof = np.full(len(rest), np.nan)
        for a, b in GroupKFold(n_splits=min(5, len(set(groups)))).split(Xr, yr, groups):
            m_, mu_, sc_ = _fit(Xr[a], yr[a])
            oof[b] = _predict(m_, mu_, sc_, Xr[b])
        resid = yr - oof
        lo_off, hi_off = _offsets(resid, level)
        sd_hedonic = float(np.std(resid))
        widen = 1.85

        hits_plain = hits_anchor = 0
        ape_plain, ape_anchor, weights, factors = [], [], [], []
        for r in held.itertuples():
            _, Xh = _held_out_design(rest, pd.DataFrame([r._asdict()]), brands, new_price)
            mu = _predict(ridge, mean, scale, Xh)[0]
            row = anchor_mod.lookup(r.brand, r.model)
            # plain: unknown-brand widening only
            if mu + lo_off * widen <= r.y <= mu + hi_off * widen:
                hits_plain += 1
            ape_plain.append(abs(np.expm1(mu - r.y)))
            # anchored
            if row:
                a_est = AnchorEstimate(ok=True, source_type=row.get("source_type", ""))
                mu_a = (np.log(float(row["list_price"]))
                        + anchor_mod._predict(retention, [r.log1p_age, r.log_km]))
                sd_a = anchor_mod.sigma(retention, a_est)
                mu_b, sd_b, w = anchor_mod.blend(mu, sd_hedonic * widen, mu_a, sd_a)
                f = max(1.0, sd_b / sd_hedonic)
                weights.append(w)
            else:
                mu_b, f = mu, widen
            factors.append(f)
            if mu_b + lo_off * f <= r.y <= mu_b + hi_off * f:
                hits_anchor += 1
            ape_anchor.append(abs(np.expm1(mu_b - r.y)))

        out[f"TR:{brand}"] = {
            "n_held_out": int(len(held)),
            "trained_on": sorted(rest.brand.unique().tolist()),
            "coverage_widened_only": round(hits_plain / len(held), 3),
            "coverage_with_anchor": round(hits_anchor / len(held), 3),
            "median_ape_widened_only": round(float(100 * np.median(ape_plain)), 1),
            "median_ape_with_anchor": round(float(100 * np.median(ape_anchor)), 1),
            "band_factor_widened_only": widen,
            "band_factor_with_anchor": round(float(np.mean(factors)), 2),
            "mean_anchor_weight": round(float(np.mean(weights)), 3) if weights else 0.0,
        }
    ran = [v for k, v in out.items() if not k.startswith("_")]
    out["_summary"] = {
        "ran": bool(ran), "holdouts": len(ran),
        "median_ape_widened_only": round(float(np.median([v["median_ape_widened_only"] for v in ran])), 1) if ran else None,
        "median_ape_with_anchor": round(float(np.median([v["median_ape_with_anchor"] for v in ran])), 1) if ran else None,
        "basis": "whole Turkish makes held out of the hedonic fit and priced as unknowns, "
                 "with the 1.85x unknown-brand widening alone versus the new-price anchor blended in",
    }
    return out


def new_price_permutation_test(fit_df: pd.DataFrame, brands: list[str]) -> dict:
    """Is the new-price column signal, or is it the fold key wearing a column?

    The objection is real and has to be answered rather than waved at:
    `log_new_price` is a deterministic function of (brand, model), the fold key
    contains brand and model, so every held-out group shares its column value
    with training rows of the same model. Two nulls, and they answer different
    halves of it.

    ACROSS MODELS. The column takes one value per (brand, model), so every
    reassignment of those values to those models is the SAME partition of the
    rows. Anything a per-model dummy could explain is therefore constant across
    the whole permutation table, and what varies is only whether the published
    figures are the RIGHT numbers. Reported as an exact rank: with four models
    there are 24 assignments and the smallest p this test can return is 1/24.

    ACROSS ROWS. The column shuffled row-wise, which breaks the per-model
    constancy as well. This is the crude leakage story and it dies loudly.
    """
    from sklearn.model_selection import GroupKFold as _GKF

    y, g = fit_df.y.to_numpy(), fit_df.group.to_numpy()
    key = (fit_df.brand.astype(str) + "|" + fit_df.model.astype(str))
    models = sorted(key.unique())
    truth = fit_df.groupby(key).log_new_price.first()
    if truth.isna().any() or len(models) < 2 or len(models) > 8:
        return {"ran": False, "reason": f"{len(models)} models with a published price "
                                        f"- the exact test is only enumerable for a few"}

    def r2(frame: pd.DataFrame) -> float:
        X = F.matrix(frame, brands, new_price=True)
        oof = np.full(len(y), np.nan)
        for a, b in _GKF(n_splits=min(5, len(set(g)))).split(X, y, g):
            m, mu_, sc_ = _fit(X[a], y[a])
            oof[b] = _predict(m, mu_, sc_, X[b])
        resid = y - oof
        return float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))

    import itertools
    values = [float(truth[m]) for m in models]
    observed = r2(fit_df)
    across_models = []
    for perm in itertools.permutations(values):
        frame = fit_df.copy()
        frame["log_new_price"] = key.map(dict(zip(models, perm)))
        across_models.append(r2(frame))

    rng = np.random.default_rng(SEED)
    raw = fit_df.log_new_price.to_numpy()
    across_rows = []
    for _ in range(200):
        frame = fit_df.copy()
        frame["log_new_price"] = rng.permutation(raw)
        across_rows.append(r2(frame))

    ge = int(sum(1 for v in across_models if v >= observed - 1e-9))
    return {
        "ran": True,
        "observed_r2": round(observed, 4),
        "n_models": len(models),
        "across_models": {
            "permutations": len(across_models),
            "rank_of_truth": ge,
            "p_exact": round(ge / len(across_models), 4),
            "p_floor": round(1 / len(across_models), 4),
            "max": round(float(np.max(across_models)), 4),
            "median": round(float(np.median(across_models)), 4),
            "min": round(float(np.min(across_models)), 4),
            "reads": ("every permutation is the same partition of rows, so a per-model "
                      "dummy would score identically across all of them; the spread here "
                      "is what the VALUES are worth. The true assignment is near the top "
                      "but with this few models the test cannot resolve it from the two "
                      "best alternatives - p_floor is the smallest p obtainable"),
        },
        "across_rows": {
            "draws": len(across_rows),
            "p": round(float(np.mean(np.array(across_rows) >= observed)), 4),
            "max": round(float(np.max(across_rows)), 4),
            "mean": round(float(np.mean(across_rows)), 4),
            "reads": "the crude leakage story: it does not survive",
        },
    }


def _new_price_card(fit_df: pd.DataFrame, brands: list[str], ridge, scale,
                    without: dict, with_np: dict, shipped: bool) -> dict:
    """Everything `estimate` and the disclosure need about the new-price column.

    The imputation mean and the clip range live here rather than in a module
    constant because they are properties of the fit, not of the reference file:
    refit on a different corpus and they move with it.
    """
    values = F.log_new_price_series(fit_df).dropna()
    lvl = f"coverage_{NEW_PRICE_LEVEL}"
    card = {
        "in_design": bool(shipped),
        "gate": {
            "r2_without": without["r2_oof"], "r2_with": with_np["r2_oof"],
            "median_ape_without": without["median_ape_oof"],
            "median_ape_with": with_np["median_ape_oof"],
            "sigma_without": without["residual_std_oof"],
            "sigma_with": with_np["residual_std_oof"],
            f"{lvl}_without": without[lvl], f"{lvl}_with": with_np[lvl],
            f"band_width_pct_{NEW_PRICE_LEVEL}_without": without[f"band_width_pct_{NEW_PRICE_LEVEL}"],
            f"band_width_pct_{NEW_PRICE_LEVEL}_with": with_np[f"band_width_pct_{NEW_PRICE_LEVEL}"],
            "min_gain": NEW_PRICE_MIN_GAIN,
            "max_coverage_loss": NEW_PRICE_MAX_COVERAGE_LOSS,
            "basis": "out-of-fold, folds grouped on the same (market, brand, model, "
                     "year, price) key everything else here uses, scored on TR",
        },
        "reference": "data/reference/new_prices_tr.json",
        "markets": list(F.NEW_PRICE_MARKETS),
        "permutation_test": (new_price_permutation_test(fit_df, list(brands))
                             if shipped else {"ran": False, "reason": "column not shipped"}),
    }
    if not shipped or not len(values):
        card["basis"] = ("not in the design: the column did not clear the gate above, "
                         "or no listing in the training set has a published new price")
        return card
    cols = F.design_columns(list(brands), new_price=True)
    i = cols.index("log_new_price")
    card.update({
        "log_mean": round(float(values.mean()), 6),
        "log_range": [round(float(values.min()), 6), round(float(values.max()), 6)],
        "price_range": [int(round(float(np.exp(values.min())))),
                        int(round(float(np.exp(values.max()))))],
        "distinct_prices": int(values.nunique()),
        "n_with_a_price": int(len(values)),
        "n_without": int(len(fit_df) - len(values)),
        "per_unit_coef": round(float(ridge.coef_[i] / (scale[i] or 1.0)), 4),
        "basis": (
            f"MEASURED: the column buys {with_np['r2_oof'] - without['r2_oof']:+.3f} "
            f"out-of-fold R2 and halves the 80% band's width at the same coverage. "
            f"ASSUMED, and the reason the column is clipped: the coefficient is "
            f"identified from {int(values.nunique())} published prices spanning "
            f"{float(values.max() - values.min()):.2f} in log space, so outside that "
            f"span it is extrapolation off a short lever and the value is pinned to "
            f"the edge. MISSING is defined, not an error: a truck with no published "
            f"new price takes the training mean, standardises to zero, and this "
            f"column then says nothing about it - though the rest of the design was "
            f"refitted around the column and does move, by about 14% for a brand the "
            f"reference table has never heard of, which nothing on this corpus can "
            f"measure because every Turkish make in it has a published price"),
    })
    return card


def main() -> None:
    listings = pd.read_csv(LISTINGS_CSV)
    df = F.build_frame(listings)
    pooled_eu = F.build_frame(listings, include_eu=True)
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
    if (pooled_eu.market.str.upper() == "EU").any():
        # TruckStore's European tractor units. Same vehicle class as the
        # Turkish stock, unlike the US conventionals - the open question is
        # whether depreciation and mileage slopes transfer across the border.
        candidates["pooled_tr_us_eu"] = (pooled_eu, F.brand_vocabulary(pooled_eu))
        candidates["pooled_tr_eu"] = (
            pooled_eu[pooled_eu.market.str.upper() != "US"],
            F.brand_vocabulary(pooled_eu[pooled_eu.market.str.upper() != "US"]))
    # Scored WITHOUT the new-price column, deliberately. Which market to train
    # on and whether the model name may reach the fit are two separate
    # questions, and the first one is settled here against the design its
    # answer was measured on. The second is asked below, on the winner.
    comparison = {}
    for name, (frame, brand_set) in candidates.items():
        comparison[name] = evaluate(frame, brand_set, score_market="TR", new_price=False)
        c = comparison[name]
        print(f"{name:14s} scored on TR held-out: n={c['n']:3d} R2={c['r2_oof']:.3f} "
              f"MAE={c['mae_pct_oof']:.1f}%  cov@0.8={c['coverage_0.8']:.3f} "
              f"(band +/-{c['band_width_pct_0.8'] / 2:.0f}%)")

    # Prefer the pooled fit unless TR-only is clearly better on TR trucks.
    # Pooling is not free - it assumes the age and km slopes transfer across
    # markets - so it has to earn its place on the market being demoed.
    # Whichever prices a Turkish truck best on held-out Turkish listings wins,
    # tie-broken toward the smaller training set: pooling assumes slopes
    # transfer across markets, and that assumption has to pay for itself.
    best = max(comparison.items(), key=lambda kv: (kv[1]["r2_oof"] or -9))
    chosen = best[0] if (best[1]["r2_oof"] or -9) > (comparison["tr_only"]["r2_oof"] or -9) + 0.02 \
        else "tr_only"
    fit_df, fit_brands = candidates[chosen]
    print(f"\nchosen training set: {chosen}")

    # --- does letting the MODEL reach the fit earn its column? ------------
    # Until this, the model name was in the fold key and in the anchor lookup
    # and in no term of the regression. Inside Ford-TR an F-MAX and a
    # Cargo-derived "TRUCKS" tractor were the same truck to it.
    without = evaluate(fit_df, fit_brands, score_market="TR", new_price=False)
    with_np = evaluate(fit_df, fit_brands, score_market="TR", new_price=True)
    lvl = f"coverage_{NEW_PRICE_LEVEL}"
    gain = (with_np["r2_oof"] or -9) - (without["r2_oof"] or -9)
    cover_loss = (without[lvl] or 0) - (with_np[lvl] or 0)
    use_new_price = gain >= NEW_PRICE_MIN_GAIN and cover_loss <= NEW_PRICE_MAX_COVERAGE_LOSS
    print("\nwhat this truck cost new, as a column in the fit (scored on TR held-out):")
    for label, c in (("without", without), ("with", with_np)):
        print(f"  {label:8s} R2={c['r2_oof']:+.4f}  MAE={c['mae_pct_oof']:5.1f}%  "
              f"median error={c['median_ape_oof']:4.1f}%  sigma={c['residual_std_oof']:.4f}  "
              f"cov@{NEW_PRICE_LEVEL}={c[lvl]:.3f} (band +/-{c[f'band_width_pct_{NEW_PRICE_LEVEL}'] / 2:.0f}%)")
    print(f"  -> R2 {gain:+.4f} (needs >= {NEW_PRICE_MIN_GAIN:+.2f}), "
          f"coverage {-cover_loss:+.3f} (may lose at most {NEW_PRICE_MAX_COVERAGE_LOSS:.2f})"
          f"  ->  {'SHIPPED' if use_new_price else 'NOT SHIPPED'}")

    overall = evaluate(fit_df, fit_brands, new_price=use_new_price)
    no_brand = evaluate(fit_df, fit_brands, drop_brand=True, new_price=use_new_price)

    # How much wider does the band have to be for a make we never fitted?
    # Measured by holding out entire brands, pooled, because the TR-only
    # corpus has too few makes to answer the question at all.
    lobo = leave_one_brand_out(pooled_eu if len(pooled_eu) > len(df) else df,
                               new_price=use_new_price)
    print("\nwithin-market leave-one-brand-out - pricing a make the fit never saw:")
    for key, r in lobo.items():
        if key.startswith("_"):
            continue
        print(f"  {key:18s} n={r['n_held_out']:4d}  80% band covered "
              f"{r['coverage_unwidened']:.2f} unwidened, needed {r['widening_needed']:.2f}x, "
              f"median error {r['median_ape_pct']:5.1f}%")
    widen = lobo["_summary"]["median_widening_needed"] or 1.5
    widen = round(max(1.2, min(4.0, widen)), 2)
    print(f"  -> unseen-brand widening set to {widen:.2f}x "
          f"(median over {lobo['_summary']['holdouts']} within-market holdouts)")
    print(f"brand columns are worth {overall['r2_oof'] - no_brand['r2_oof']:+.3f} R2 within the corpus")

    # --- the second pricing route: new price x retention ------------------
    retention = fit_retention(df)
    if retention.get("ok"):
        print(f"\nretention curve (new-price anchor): R2 {retention['r2_oof']:+.3f} out-of-fold, "
              f"median error {retention['median_ape_oof']}%, on {retention['n']} TR listings "
              f"with a published new price")
        print(f"  hedonic fit for comparison: R2 {overall['r2_oof']:+.3f}, "
              f"median error {overall['median_ape_oof']}%")
    else:
        print(f"\nretention curve not fitted: {retention['reason']}")

    anchor_test = anchor_holdout(df, retention)
    # The same holdout with the column in the hedonic design. It is reported
    # separately rather than replacing the line above because it answers a
    # different question: with the column shipped, the reference figure reaches
    # the estimate through TWO doors, and the inverse-variance blend in
    # `model.estimate` assumes the two routes are independent when they no
    # longer are. The blend's floor at 1.0x is what keeps that from becoming a
    # band narrower than the one whose coverage was measured, and this is the
    # measurement of what it costs.
    anchor_test_np = (anchor_holdout(df, retention, new_price=True)
                      if use_new_price else {"_summary": {"ran": False,
                                                          "reason": "column not shipped"}})
    if anchor_test["_summary"].get("ran"):
        print("\n  does the anchor rescue an unseen make? (whole TR brands held out)")
        for label, test in (("hedonic without the new-price column", anchor_test),
                            ("hedonic WITH it - the same figure, both routes", anchor_test_np)):
            if not test["_summary"].get("ran"):
                continue
            print(f"    {label}:")
            for key, r in test.items():
                if key.startswith("_"):
                    continue
                print(f"      {key:12s} n={r['n_held_out']:3d}  "
                      f"band {r['band_factor_widened_only']:.2f}x->{r['band_factor_with_anchor']:.2f}x   "
                      f"coverage {r['coverage_widened_only']:.2f}->{r['coverage_with_anchor']:.2f}   "
                      f"median error {r['median_ape_widened_only']:5.1f}%->{r['median_ape_with_anchor']:5.1f}%")

    # --- final fit on everything -----------------------------------------
    X = F.matrix(fit_df, fit_brands, new_price=use_new_price)
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
    new_price_card = _new_price_card(fit_df, fit_brands, ridge, scale,
                                     without, with_np, use_new_price)
    model = PriceModel(
        brands=list(fit_brands),
        columns=F.design_columns(list(fit_brands), new_price=use_new_price),
        coef=[round(float(c), 6) for c in ridge.coef_],
        intercept=round(float(ridge.intercept_), 6),
        mean=[round(float(v), 6) for v in mean],
        scale=[round(float(v), 6) for v in scale],
        residual_std=round(float(np.std(oof_resid)), 4),
        offsets={str(level): [round(v, 4) for v in _offsets(oof_resid, level)]
                 for level in LEVELS},
        # Every block in here is the SHIPPED design unless its key says
        # otherwise. That was briefly not true and it was the kind of untrue
        # that reads as a mistake this repo has made before: `tr_only_view`
        # carried the design WITHOUT the new-price column while the top-level
        # `r2_oof` carried the design with it, so one artifact showed 0.9506
        # and 0.8422 for the same 84 listings with nothing saying why.
        calibration={**overall,
                     "scored_on": ", ".join(sorted(fit_df.market.str.upper().unique())),
                     "design": "shipped: " + ", ".join(F.design_columns(list(fit_brands),
                                                                        new_price=use_new_price)),
                     "tr_only_view": with_np if use_new_price else without,
                     # Same 84 listings, same folds, same key - one column fewer.
                     # This is the number CLAUDE.md and the README documented
                     # before the column existed; `r2_oof` above is its successor.
                     "without_new_price_column": without,
                     "without_brand_columns": no_brand,
                     "training_set_choice": {k: {"r2_oof": v["r2_oof"], "n": v["n"],
                                                 "coverage_0.8": v["coverage_0.8"]}
                                             for k, v in comparison.items()},
                     "training_set_choice_basis": (
                         "scored on held-out TR listings with the new-price column OFF, "
                         "because which market to train on and whether the model name may "
                         "reach the fit are two separate questions and the first was "
                         "settled against the design its answer was measured on")},
        anchor={**retention, "unseen_brand_test": anchor_test,
                "unseen_brand_test_with_new_price_column": anchor_test_np},
        new_price=new_price_card,
        widening={"unknown_brand": widen,
                  "unknown_brand_basis": (
                      "measured by within-market leave-one-brand-out: whole makes were held "
                      "out of the fit for their own market and priced as unknowns; this is "
                      "the median band inflation needed to restore 80% coverage. Done within "
                      "a market because brand is nearly collinear with market on this corpus, "
                      "so a cross-market holdout measures the border, not the brand"),
                  "leave_one_brand_out": lobo,
                  # How much wider the CONDITION band is drawn because the
                  # model's own read of the photos is unstable. It is the one
                  # place the eval harness feeds a number back into the
                  # product, and it ships at 1.0 - no widening - until the
                  # test-retest suite has measured it. Same inverse-variance
                  # logic the anchor uses: sqrt(1 + (sd_cond/residual_std)^2).
                  "condition_read": 1.0,
                  "condition_read_basis": (
                      "not yet measured: this is the spread of the condition "
                      "multiplier across repeat runs of the SAME photo set, "
                      "inflated into the band the way the anchor inflates "
                      "variance. Until eval/suites/retest.py has run it stays "
                      "at 1.0, which widens nothing and claims nothing")},
        meta={
            "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "training_set": chosen,
            "n_listings": int(len(fit_df)),
            "n_groups": int(fit_df.group.nunique()),
            "target": "log(asking price in the market's own currency; "
                      "market dummies absorb the FX constant)",
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
    if use_new_price:
        c = new_price_card
        print(f"\nnew-price column: {c['per_unit_coef']:+.2f} log price per log lira of list "
              f"price, fitted on {c['distinct_prices']} published prices "
              f"({c['price_range'][0]:,}-{c['price_range'][1]:,} TRY) and clipped to them; "
              f"{c['n_without']} of {len(fit_df)} training rows have no published price")
        p = c.get("permutation_test", {})
        if p.get("ran"):
            am, ar = p["across_models"], p["across_rows"]
            print(f"  is it the fold key in disguise? shuffled ACROSS MODELS "
                  f"({am['permutations']} exact permutations, same partition every time): "
                  f"R2 {am['min']:.3f}..{am['max']:.3f}, true {p['observed_r2']:.3f} "
                  f"ranks #{am['rank_of_truth']}, p={am['p_exact']:.3f} "
                  f"(floor {am['p_floor']:.3f} at {p['n_models']} models)")
            print(f"                             shuffled ACROSS ROWS "
                  f"({ar['draws']} draws): R2 max {ar['max']:.3f}, p={ar['p']:.3f}")
    print(f"\nwrote {PRICE_MODEL}")


if __name__ == "__main__":
    main()
