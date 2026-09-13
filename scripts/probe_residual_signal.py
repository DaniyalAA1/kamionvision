"""Step 0: is there any image signal in the price residual at all?

The learned condition multiplier only makes sense if what a photo shows moves
the asking price in a way `log1p(age) + log(km) + brand + euro6` does not
already capture. This probe asks that question directly and cheaply, before
anything is built on top of the answer.

Method, deliberately conservative:

  1. Fit the shipped spec ridge and take its OUT-OF-FOLD residual per listing
     (GroupKFold on the same `(market, brand, model, year, price)` key the
     price model uses, so identical dealer stock never straddles a split).
  2. Mean-pool the cached CLIP embedding over that vehicle's ORIGINAL photos.
     Degraded twins are synthetic and would trip the same vehicle twice.
  3. Predict the residual from the embedding under nested grouped CV - the
     inner fold picks alpha and the PCA width, the outer fold scores. Nothing
     that touches a scoring row is allowed to inform the fit.
  4. Run the whole thing again against permuted targets. With 512 dimensions
     and tens of groups a ridge can score positive on pure noise, so a bare
     R^2 means nothing without the null it has to beat.

Reports R^2 for TR, US and pooled against the 95th percentile of that null.
Throwaway diagnostic - it is not imported by the app.

    .venv/bin/python scripts/probe_residual_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                       # noqa: E402
from app.pricing import features as F        # noqa: E402

SEED = 7
N_PERM = 60
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
DIMS = (4, 8, 16, 32)


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


def vehicle_embeddings() -> dict[str, np.ndarray]:
    """listing_id -> mean of its original photos' embeddings."""
    z = np.load(config.META / "embeddings.npz", allow_pickle=False)
    emb = dict(zip(z["image_id"].tolist(), z["emb"]))
    img = pd.read_csv(config.IMAGES_CSV)
    img = img[img.variant == "original"]
    out = {}
    for listing_id, grp in img.groupby("listing_id"):
        vecs = [emb[i] for i in grp.image_id if i in emb]
        if vecs:
            v = np.mean(vecs, axis=0)
            out[str(listing_id)] = v / (np.linalg.norm(v) + 1e-9)
    return out


def nested_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray, rng) -> float:
    """Outer-fold R^2; alpha and PCA width chosen inside the training fold only."""
    n_out = min(5, len(np.unique(groups)))
    if n_out < 3:
        return float("nan")
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_out).split(X, y, groups):
        best, best_score = None, -np.inf
        g_tr = groups[tr]
        n_in = min(4, len(np.unique(g_tr)))
        for k in DIMS:
            if k >= len(tr):
                continue
            for a in ALPHAS:
                if n_in < 2:
                    continue
                inner = np.zeros(len(tr))
                for itr, ite in GroupKFold(n_splits=n_in).split(X[tr], y[tr], g_tr):
                    p = PCA(n_components=min(k, len(itr) - 1), random_state=SEED).fit(X[tr][itr])
                    m = Ridge(alpha=a).fit(p.transform(X[tr][itr]), y[tr][itr])
                    inner[ite] = m.predict(p.transform(X[tr][ite]))
                score = r2(y[tr], inner)
                if score > best_score:
                    best_score, best = score, (k, a)
        if best is None:
            continue
        k, a = best
        p = PCA(n_components=min(k, len(tr) - 1), random_state=SEED).fit(X[tr])
        m = Ridge(alpha=a).fit(p.transform(X[tr]), y[tr])
        pred[te] = m.predict(p.transform(X[te]))
    return r2(y, pred)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def probe(name: str, df: pd.DataFrame, embs: dict) -> None:
    df = df[df.listing_id.astype(str).isin(embs)].copy()
    if len(df) < 12:
        print(f"\n{name}: only {len(df)} listings with embeddings - skipped")
        return
    df["resid"] = spec_residuals(df)
    X = np.stack([embs[str(i)] for i in df.listing_id])
    y, groups = df.resid.to_numpy(), df.group.to_numpy()
    rng = np.random.default_rng(SEED)

    print(f"\n{name}: n={len(df)} listings, {df.group.nunique()} spec groups, "
          f"residual sd={y.std():.4f} ({100 * y.std():.1f}% in price terms)")

    real = nested_r2(X, y, groups, rng)
    null = np.array([nested_r2(X, rng.permutation(y), groups, rng) for _ in range(N_PERM)])
    null = null[~np.isnan(null)]
    p95 = float(np.percentile(null, 95)) if len(null) else float("nan")
    p_value = float(np.mean(null >= real)) if len(null) else float("nan")

    print(f"  embedding -> residual   R^2 = {real:+.3f}")
    print(f"  permuted-target null    median {np.median(null):+.3f}, 95th pct {p95:+.3f}  "
          f"(n={len(null)})")
    print(f"  p = {p_value:.3f}   ->  {'SIGNAL' if p_value < 0.05 else 'no signal'}")


def main() -> int:
    listings = pd.read_csv(config.LISTINGS_CSV)
    df = F.build_frame(listings)
    embs = vehicle_embeddings()
    print(f"corpus: {len(df)} priced listings with year+km, {len(embs)} vehicles with embeddings")
    probe("TR", df[df.market.str.upper() == "TR"], embs)
    probe("US", df[df.market.str.upper() == "US"], embs)
    probe("pooled TR+US", df, embs)
    print("\nDecision rule: Part 5 (fitted condition multiplier) proceeds only if the "
          "US or pooled probe beats its null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
