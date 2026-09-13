"""Does a panel-graded condition score explain the price residual? Probably not.

The severity x impact weight table in `app.condition` decides where inside the
measured +/-9.4% cap a truck lands, and it is assumed, not fitted. This is the
experiment that would let us stop calling it assumed - and the honest prior is
that it will fail, for reasons that are about the corpus rather than the code.

It is the panel-score analogue of `scripts/probe_residual_signal.py`, which
asked the same question of a CLIP embedding and answered no: TR R^2 -0.222, US
-0.038, pooled -0.046, all inside a permuted-target null at p >= 0.57, across
six pooling variants, with positive controls passing on the same pipeline. Same
method here, better feature: a human-or-panel condition grade instead of 512
floats.

  1. Fit the shipped spec ridge and take its OUT-OF-FOLD residual per listing,
     GroupKFold on the same `(market, brand, model, year, price)` key the price
     model folds on, so identical dealer stock never straddles a split.
  2. Join the panel's condition score onto those listings.
  3. Predict the residual from the condition score under NESTED grouped CV.
     Nothing that touches a scoring row informs the fit.
  4. Run it again against permuted targets, >= 60 times. A ridge on tens of
     groups scores positive on noise often enough that a bare R^2 means nothing
     without the null it has to beat.

PRE-REGISTERED. All four must hold before the weights may be called fitted:

  1. the panel's condition score has real variance on the priced vehicles. If
     reconditioned stock grades everything `good` there is nothing to regress.
  2. out-of-fold R^2 beats the null's 95th percentile, on TR and on pooled.
  3. the fitted weights are MONOTONE in severity. A fit saying major is worth
     less than minor is overfitting, not a discovery.
  4. the fitted weights beat the assumed ones on held-out interval COVERAGE,
     not only on R^2. The band is the product; R^2 is not.

THE HONEST NEGATIVE, drafted before the run so nobody has to draft it under
pressure. If this fails, this is what goes in the report verbatim:

    The severity weights remain assumptions. We tested whether a panel-graded
    condition score explains the out-of-fold price residual on the priced
    vehicles, under vehicle-grouped nested CV against a permuted-target null:
    R^2 = X against a null 95th percentile of Y. It does not beat the null, so
    the weights are not fitted. What the panel did calibrate is the severity
    scale itself, so the shipped numbers are assumed weights applied to a
    severity whose agreement with a panel is measured. The cap stays at one
    measured residual sigma, which is the honest ceiling for a quantity we
    cannot calibrate.

Why the prior is negative: TR has 23 distinct prices across 84 listings, both
OEM sources sell reconditioned stock off a prepped lot, and the CLIP probe
already found nothing on the same target. Even a SUCCESSFUL fit could only
redistribute weight INSIDE the measured cap - it could never raise it.

Throwaway diagnostic; nothing in `app/` imports it.

    .venv/bin/python scripts/probe_condition_residual.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import condition as C                 # noqa: E402
from app import config                         # noqa: E402
from app.pricing import features as F          # noqa: E402

SEED = 7
N_PERM = 60
ALPHAS = (0.1, 1.0, 10.0, 100.0)

# Where the panel reference set is expected. One JSON object per line:
#
#   {"listing_id": "12345", "panelist": "opus-a", "grade": "fair",
#    "condition_score": 0.42,
#    "families": {"drive_tires": 1.4, "frame_corrosion": 0.0, ...}}
#
# `condition_score` is 0.0 (as-new) to 1.0 (end of life). `grade` alone is
# enough - it is mapped through GRADE_SCORE below - and `families` is optional
# and gives the probe per-subsystem demerits to regress instead of one scalar.
# Several rows per listing_id are averaged, which is also how the
# leave-one-panelist-out ceiling gets computed.
PANEL_PATHS = [
    config.DATA / "reference" / "condition_panel.jsonl",
    Path(__file__).resolve().parent.parent / "eval" / "reference" / "condition_panel.jsonl",
]

GRADE_SCORE = {"excellent": 0.05, "good": 0.30, "fair": 0.65, "poor": 0.95}


def load_panel() -> pd.DataFrame | None:
    """The panel reference set, or None if nobody has collected one yet."""
    for path in PANEL_PATHS:
        if path.exists():
            rows = [json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                df = pd.DataFrame(rows)
                df["listing_id"] = df.listing_id.astype(str)
                if "condition_score" not in df.columns:
                    df["condition_score"] = np.nan
                df["condition_score"] = df.condition_score.fillna(
                    df.get("grade", pd.Series(dtype=object)).map(GRADE_SCORE))
                return df.dropna(subset=["condition_score"])
    return None


def panel_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One row per vehicle: the mean panel score, and per-family demerits if given."""
    agg = panel.groupby("listing_id").condition_score.mean().rename("panel_score")
    out = agg.to_frame()
    columns = ["panel_score"]
    if "families" in panel.columns:
        fam = {}
        for listing_id, grp in panel.groupby("listing_id"):
            totals = {f: 0.0 for f in C.FAMILIES}
            seen = 0
            for entry in grp.families:
                if not isinstance(entry, dict):
                    continue
                seen += 1
                for family, value in entry.items():
                    if family in totals:
                        totals[family] += float(value)
            if seen:
                fam[listing_id] = {f: v / seen for f, v in totals.items()}
        if fam:
            wide = pd.DataFrame.from_dict(fam, orient="index")
            out = out.join(wide, how="left").fillna(0.0)
            columns += list(wide.columns)
    return out.reset_index(), columns


def spec_residuals(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold residual of the shipped spec model, per listing."""
    brands = F.brand_vocabulary(df)
    X, y, groups = F.matrix(df, brands), df.y.to_numpy(), df.group.to_numpy()
    resid = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=min(5, df.group.nunique())).split(X, y, groups):
        mean, scale = X[tr].mean(0), X[tr].std(0)
        scale[scale == 0] = 1.0
        m = Ridge(alpha=1.0).fit((X[tr] - mean) / scale, y[tr])
        resid[te] = y[te] - m.predict((X[te] - mean) / scale)
    return resid


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def nested_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Outer-fold R^2; alpha chosen inside the training fold only."""
    n_out = min(5, len(np.unique(groups)))
    if n_out < 3:
        return float("nan")
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_out).split(X, y, groups):
        g_tr = groups[tr]
        n_in = min(4, len(np.unique(g_tr)))
        best, best_score = ALPHAS[0], -np.inf
        if n_in >= 2:
            for a in ALPHAS:
                inner = np.zeros(len(tr))
                for itr, ite in GroupKFold(n_splits=n_in).split(X[tr], y[tr], g_tr):
                    m = Ridge(alpha=a).fit(X[tr][itr], y[tr][itr])
                    inner[ite] = m.predict(X[tr][ite])
                score = r2(y[tr], inner)
                if score > best_score:
                    best_score, best = score, a
        m = Ridge(alpha=best).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return r2(y, pred)


def probe(name: str, df: pd.DataFrame, feats: pd.DataFrame, columns: list[str]) -> dict:
    df = df.merge(feats, left_on=df.listing_id.astype(str), right_on="listing_id",
                  how="inner", suffixes=("", "_panel"))
    if len(df) < 12:
        print(f"\n{name}: only {len(df)} listings carry a panel grade - skipped")
        return {}
    df["resid"] = spec_residuals(df)
    X = df[columns].to_numpy(dtype=float)
    y, groups = df.resid.to_numpy(), df.group.to_numpy()
    spread = float(df.panel_score.std())

    print(f"\n{name}: n={len(df)} listings, {df.group.nunique()} spec groups, "
          f"residual sd={y.std():.4f} ({100 * y.std():.1f}% in price terms)")
    print(f"  panel condition score: mean {df.panel_score.mean():.3f}, "
          f"sd {spread:.3f}, {df.panel_score.nunique()} distinct values")
    # Criterion 1, and the one most likely to fail on reconditioned stock.
    if spread < 0.05:
        print("  CRITERION 1 FAILS: the panel grades everything the same. There is "
              "nothing to regress and no amount of method fixes that.")

    real = nested_r2(X, y, groups)
    rng = np.random.default_rng(SEED)
    null = np.array([nested_r2(X, rng.permutation(y), groups) for _ in range(N_PERM)])
    null = null[~np.isnan(null)]
    p95 = float(np.percentile(null, 95)) if len(null) else float("nan")
    p_value = float(np.mean(null >= real)) if len(null) else float("nan")

    print(f"  panel score -> residual  R^2 = {real:+.3f}")
    print(f"  permuted-target null     median {np.median(null):+.3f}, "
          f"95th pct {p95:+.3f}  (n={len(null)})")
    print(f"  p = {p_value:.3f}   ->  {'SIGNAL' if p_value < 0.05 else 'no signal'}")
    return {"name": name, "n": len(df), "r2": real, "p95": p95, "p": p_value,
            "spread": spread}


def no_panel_yet() -> int:
    print("No condition panel reference set on disk, so there is nothing to probe.")
    print("Looked in:")
    for path in PANEL_PATHS:
        print(f"  {path}")
    print("\nExpected format - one JSON object per line, one line per (vehicle,")
    print("panelist), several panelists per vehicle so the agreement ceiling is")
    print("computable:")
    print('  {"listing_id": "12345", "panelist": "opus-a", "grade": "fair",')
    print('   "condition_score": 0.42, "families": {"drive_tires": 1.4}}')
    print("\n`grade` alone is enough; `condition_score` and `families` sharpen it.")
    print("Collecting that set is the eval harness's job (eval/suites/panel.py).")
    print("\nUntil then the severity weights stay labelled `assumed`, which is what")
    print("`app.condition.WEIGHTS_BASIS` says and what the report and the web")
    print("disclosure both print. That is the correct state, not a gap.")
    return 0


def main() -> int:
    panel = load_panel()
    if panel is None:
        return no_panel_yet()

    feats, columns = panel_features(panel)
    listings = pd.read_csv(config.LISTINGS_CSV)
    df = F.build_frame(listings)
    print(f"corpus: {len(df)} priced listings with year+km; "
          f"{len(feats)} vehicles carry a panel grade "
          f"({panel.panelist.nunique() if 'panelist' in panel else 1} panelists)")
    print(f"features: {', '.join(columns)}")

    results = [r for r in (
        probe("TR", df[df.market.str.upper() == "TR"], feats, columns),
        probe("US", df[df.market.str.upper() == "US"], feats, columns),
        probe("pooled TR+US", df, feats, columns)) if r]

    print("\n" + "-" * 72)
    passed = [r for r in results if r["name"] in ("TR", "pooled TR+US")
              and r["p"] < 0.05 and r["r2"] > r["p95"] and r["spread"] >= 0.05]
    if len(passed) >= 2:
        print("Criteria 1 and 2 hold. Criteria 3 (weights monotone in severity) and")
        print("4 (fitted weights beat assumed ones on HELD-OUT interval coverage,")
        print("not on R^2) are not tested here and must both pass before anything")
        print("in app/condition.py stops being labelled `assumed`.")
    else:
        print("The severity weights remain assumptions. A panel-graded condition")
        print("score does not beat the permutation null on the out-of-fold price")
        print("residual, so the weights are not fitted. What the panel calibrates")
        print("is the severity SCALE; the shipped numbers are assumed weights")
        print("applied to a severity whose agreement with a panel is measured. The")
        print("cap stays at one measured residual sigma, which is the honest")
        print("ceiling for a quantity we cannot calibrate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
