"""Fit the three perception heads and write models/perception.json.

    .venv/bin/python scripts/cache_embeddings.py    # once, ~2 min
    .venv/bin/python -m app.perception.train

Every score printed here is held out on a vehicle-grouped split. One tractor
contributes 15-40 near-identical frames, so an image-level split would score
memorisation - the same reason `scripts/package_dataset.py` groups its splits
by `listing_id` and `app/pricing/train.py` groups its folds by spec.

What each head can honestly claim differs, and the artifact records which:

  degradation  REAL labels. 3,729 degraded images against their exact paired
               originals, with the 14 applied transforms known. Held-out AUC
               means what it says.
  view         The labels are CLIP zero-shot pseudo-labels, so accuracy against
               them measures agreement with a noisy teacher, not correctness.
               The number worth reporting is different: a degraded twin
               inherits its original's label, so we can measure whether a head
               keeps its answer when the photo is corrupted, and compare that
               to the zero-shot tagger doing the same thing. That is a real
               robustness gain even though the teacher is the ceiling.
  identity     REAL labels from listing metadata, but the corpus is 78/84 Ford
               in TR and Freightliner-dominated in the US, so this is close to
               a two-class problem. It ships as a cross-check on the VLM's
               badge read, never as the primary identification.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .. import config
from ..pricing import features as F
from .heads import PERCEPTION_MODEL, SCALARS, scalar_row

N_SPLITS = 5
SEED = 7
# Share of genuinely clean photos we accept flagging as unfit for fine detail.
# The same posture calibrate_gate.py takes: fix the false-positive rate on
# real photos, then report what that buys.
FALSE_FLAG_RATE = 0.05
ROUND = 5


def load() -> tuple[pd.DataFrame, np.ndarray]:
    """images.csv joined to the cached embeddings, aligned row for row."""
    npz = config.META / "embeddings.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} not found - run: .venv/bin/python scripts/cache_embeddings.py")
    z = np.load(npz, allow_pickle=False)
    emb = dict(zip(z["image_id"].tolist(), z["emb"]))
    df = pd.read_csv(config.IMAGES_CSV)
    df = df[df.image_id.isin(emb)].reset_index(drop=True)
    return df, np.stack([emb[i] for i in df.image_id]).astype(float)


def scalars_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array([
        scalar_row(blur_laplacian_var=r.blur_laplacian_var, brightness=r.brightness,
                   contrast_rms=r.contrast_rms, dark_clipped_frac=r.dark_clipped_frac,
                   bright_clipped_frac=r.bright_clipped_frac,
                   colourfulness=r.colourfulness, capture_quality=r.capture_quality,
                   width=r.width, height=r.height)
        for r in df.itertuples()])


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    return mean, scale


def _pack(estimators, classes, mean, scale) -> dict:
    """A fitted sklearn linear model -> plain JSON."""
    coef = np.atleast_2d(np.vstack([e.coef_.ravel() for e in estimators])
                         if isinstance(estimators, list) else estimators.coef_)
    intercept = (np.array([float(np.ravel(e.intercept_)[0]) for e in estimators])
                 if isinstance(estimators, list)
                 else np.atleast_1d(estimators.intercept_))
    return {"classes": list(classes),
            "coef": np.round(coef, ROUND).tolist(),
            "intercept": np.round(intercept, ROUND).tolist(),
            "mean": np.round(mean, ROUND).tolist(),
            "scale": np.round(scale, ROUND).tolist()}


def fit_degradation(df, X, groups) -> dict:
    """14 binary heads + a severity regressor. The labels here are real."""
    atoms = sorted({a for v in df.degradations.dropna() for a in str(v).split("|")})
    Y = np.zeros((len(df), len(atoms)))
    for j, atom in enumerate(atoms):
        Y[:, j] = df.degradations.fillna("").astype(str).str.split("|").map(
            lambda parts, a=atom: float(a in parts))
    sev = df.severity.fillna(0.0).to_numpy(dtype=float)

    cv = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros_like(Y)
    oof_sev = np.zeros(len(df))
    for tr, te in cv.split(X, sev, groups):
        mean, scale = _standardise(X[tr])
        Ztr, Zte = (X[tr] - mean) / scale, (X[te] - mean) / scale
        for j in range(len(atoms)):
            if len(np.unique(Y[tr, j])) < 2:
                continue
            m = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, Y[tr, j])
            oof[te, j] = m.predict_proba(Zte)[:, 1]
        oof_sev[te] = Ridge(alpha=10.0).fit(Ztr, sev[tr]).predict(Zte)

    aucs = {a: round(float(roc_auc_score(Y[:, j], oof[:, j])), 3)
            for j, a in enumerate(atoms) if len(np.unique(Y[:, j])) > 1}
    is_deg = (df.variant == "degraded").to_numpy(dtype=float)
    binary_auc = float(roc_auc_score(is_deg, oof.max(axis=1)))
    ss = 1 - float(np.sum((sev - oof_sev) ** 2) / np.sum((sev - sev.mean()) ** 2))

    # Calibrate the "too corrupted for a fine-detail claim" threshold the same
    # way calibrate_gate.py sets its capture floors: from a quantile of the
    # real population, not from a round number. Anchored on ORIGINALS so the
    # stated quantity is a false-flag rate on clean photos, then reported
    # against how much genuine degradation it actually catches.
    clean_sev = oof_sev[df.variant.to_numpy() == "original"]
    deg_sev = oof_sev[df.variant.to_numpy() == "degraded"]
    fine_detail_severity = float(np.quantile(clean_sev, 1 - FALSE_FLAG_RATE))
    caught = float(np.mean(deg_sev >= fine_detail_severity))

    mean, scale = _standardise(X)
    Z = (X - mean) / scale
    heads = [LogisticRegression(max_iter=2000, C=1.0).fit(Z, Y[:, j]) for j in range(len(atoms))]
    packed = _pack(heads, atoms, mean, scale)
    packed["atoms"] = atoms
    sev_model = Ridge(alpha=10.0).fit(Z, sev)
    packed["severity"] = {"coef": np.round(np.atleast_2d(sev_model.coef_), ROUND).tolist(),
                          "intercept": [round(float(sev_model.intercept_), ROUND)],
                          "mean": packed["mean"], "scale": packed["scale"], "classes": ["severity"]}
    packed["fine_detail_severity"] = round(fine_detail_severity, 4)
    packed["metrics"] = {"per_atom_auc": aucs,
                         "mean_atom_auc": round(float(np.mean(list(aucs.values()))), 3),
                         "degraded_vs_original_auc": round(binary_auc, 3),
                         "severity_r2": round(ss, 3),
                         "fine_detail_severity": round(fine_detail_severity, 4),
                         "fine_detail_false_flag_rate": FALSE_FLAG_RATE,
                         "fine_detail_degraded_caught": round(caught, 3),
                         "n": int(len(df)), "basis": "held out, vehicle-grouped 5-fold"}
    return packed


def fit_view(df, emb, groups) -> dict:
    """Robustness, not accuracy: does the head hold its answer under degradation?"""
    labels = df.view.astype(str).to_numpy()
    classes = sorted(set(labels))
    cv = GroupKFold(n_splits=N_SPLITS)
    pred = np.empty(len(df), dtype=object)
    for tr, te in cv.split(emb, labels, groups):
        mean, scale = _standardise(emb[tr])
        m = LogisticRegression(max_iter=3000, C=1.0).fit((emb[tr] - mean) / scale, labels[tr])
        pred[te] = m.predict((emb[te] - mean) / scale)

    # Twin agreement. The degraded row's `view` is inherited from its original,
    # so it cannot serve as the comparison - we recompute what the zero-shot
    # tagger itself says about each degraded frame and ask the same question of
    # both: original label in, degraded label out, do they match?
    zs = zero_shot_views(emb)
    pairs = twin_pairs(df)
    head_agree = float(np.mean([pred[i] == pred[j] for i, j in pairs])) if pairs else float("nan")
    zs_agree = float(np.mean([zs[i] == zs[j] for i, j in pairs])) if pairs else float("nan")
    acc = float(np.mean(pred == labels))

    mean, scale = _standardise(emb)
    m = LogisticRegression(max_iter=3000, C=1.0).fit((emb - mean) / scale, labels)
    packed = _pack(m, m.classes_, mean, scale)
    packed["metrics"] = {
        "agreement_with_teacher": round(acc, 3),
        "twin_agreement_head": round(head_agree, 3),
        "twin_agreement_zeroshot": round(zs_agree, 3),
        "n_pairs": len(pairs),
        "basis": "held out, vehicle-grouped 5-fold; twin agreement is "
                 "original-vs-degraded label stability on the same photo",
    }
    return packed


def zero_shot_views(emb: np.ndarray) -> np.ndarray:
    """What CLIP zero-shot says, recomputed from the cached embeddings."""
    from .. import vision
    tagger = vision.clip()
    bank = tagger.view_bank.cpu().numpy()
    scores = emb @ bank.T
    return np.array([vision.VIEW_LABELS[i] for i in scores.argmax(axis=1)])


def twin_pairs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Row indices of (original, its degraded twin)."""
    by_path = {p: i for i, p in enumerate(df.path)}
    out = []
    for i, r in enumerate(df.itertuples()):
        if r.variant == "degraded" and isinstance(r.clean_path, str):
            j = by_path.get(r.clean_path)
            if j is not None:
                out.append((j, i))
    return out


def fit_identity(df, emb) -> dict:
    """Brand from mean-pooled per-vehicle embedding. Real labels, thin classes."""
    orig = df.variant == "original"
    sub, sub_emb = df[orig].reset_index(drop=True), emb[orig.to_numpy()]
    rows, labels = [], []
    for lid, idx in pd.Series(range(len(sub))).groupby(sub.listing_id).groups.items():
        v = sub_emb[list(idx)].mean(axis=0)
        rows.append(v / (np.linalg.norm(v) + 1e-9))
        labels.append(F.normalise_brand(sub.make.iloc[list(idx)[0]]))
    X, y = np.stack(rows), np.array(labels)
    groups = np.arange(len(y))          # one row per vehicle already

    cv = GroupKFold(n_splits=N_SPLITS)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y, groups):
        mean, scale = _standardise(X[tr])
        m = LogisticRegression(max_iter=3000, C=1.0).fit((X[tr] - mean) / scale, y[tr])
        pred[te] = m.predict((X[te] - mean) / scale)
    acc = float(np.mean(pred == y))
    majority = float(pd.Series(y).value_counts(normalize=True).max())

    mean, scale = _standardise(X)
    m = LogisticRegression(max_iter=3000, C=1.0).fit((X - mean) / scale, y)
    packed = _pack(m, m.classes_, mean, scale)
    packed["metrics"] = {"accuracy": round(acc, 3), "majority_baseline": round(majority, 3),
                         "n_vehicles": int(len(y)),
                         "class_counts": pd.Series(y).value_counts().to_dict(),
                         "basis": "held out, one row per vehicle, 5-fold"}
    return packed


def main() -> int:
    df, emb = load()
    groups = df.listing_id.astype(str).to_numpy()
    X = np.concatenate([emb, scalars_matrix(df)], axis=1)
    print(f"{len(df)} images, {df.listing_id.nunique()} vehicles, "
          f"{emb.shape[1]}-d embedding + {len(SCALARS)} scalars")

    print("\nfitting degradation head (real labels)...")
    degradation = fit_degradation(df, X, groups)
    dm = degradation["metrics"]
    print(f"  degraded-vs-original AUC {dm['degraded_vs_original_auc']}   "
          f"mean per-atom AUC {dm['mean_atom_auc']}   severity R2 {dm['severity_r2']}")
    print(f"  fine-detail cutoff {dm['fine_detail_severity']} "
          f"(flags {100 * dm['fine_detail_false_flag_rate']:.0f}% of clean photos, "
          f"catches {100 * dm['fine_detail_degraded_caught']:.0f}% of degraded)")
    worst = sorted(dm["per_atom_auc"].items(), key=lambda kv: kv[1])[:3]
    print(f"  weakest atoms: {', '.join(f'{a} {v}' for a, v in worst)}")

    print("\nfitting view head (robustness under degradation)...")
    view = fit_view(df, emb, groups)
    vm = view["metrics"]
    print(f"  twin agreement: head {vm['twin_agreement_head']} vs "
          f"zero-shot {vm['twin_agreement_zeroshot']} over {vm['n_pairs']} pairs")
    print(f"  agreement with the zero-shot teacher's own label: {vm['agreement_with_teacher']}")

    print("\nfitting identity head (cross-check only)...")
    identity = fit_identity(df, emb)
    im = identity["metrics"]
    print(f"  brand accuracy {im['accuracy']} vs majority baseline {im['majority_baseline']} "
          f"on {im['n_vehicles']} vehicles")

    meta = {
        "fitted_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_images": int(len(df)), "n_vehicles": int(df.listing_id.nunique()),
        "embedding": {"model": "ViT-B-32-quickgelu", "pretrained": "openai", "dim": int(emb.shape[1])},
        "grouping": "all folds grouped by listing_id (vehicle)",
        "degradation_auc": dm["degraded_vs_original_auc"],
        "fine_detail_severity": dm["fine_detail_severity"],
        "fine_detail_degraded_caught": dm["fine_detail_degraded_caught"],
        "severity_r2": dm["severity_r2"],
        "view_agreement_head": vm["twin_agreement_head"],
        "view_agreement_zeroshot": vm["twin_agreement_zeroshot"],
        "identity_accuracy": im["accuracy"], "identity_majority": im["majority_baseline"],
        "label_provenance": {
            "degradation": "measured - known synthetic transforms, 3729 exact clean/degraded pairs",
            "view": "pseudo-label - CLIP zero-shot teacher; only the twin-agreement delta is measured",
            "identity": "measured - listing metadata, but 78/84 TR vehicles are Ford",
        },
    }
    # Written compactly on purpose: indented, this is 23k lines of coefficients
    # and the diff is unreadable. Nobody inspects a weight matrix by eye - the
    # numbers a human wants are the held-out metrics, and `app.cli doctor`
    # prints those from the meta block.
    PERCEPTION_MODEL.write_text(json.dumps(
        {"scalars": SCALARS, "degradation": degradation, "view": view,
         "identity": identity, "meta": meta}, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {PERCEPTION_MODEL} ({PERCEPTION_MODEL.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
