"""Does international-haul usage improve the Turkish asking-price model?

This is a pre-registered, throwaway diagnostic.  It does not alter the pricing
model.  The candidate feature is deliberately narrow:

    usage_international = 1 when usage_class == "Yurt Dışı Lojistik", else 0

That means the two construction listings collapse into domestic and missing or
unknown usage is zero.  The probe compares spec-grouped nested-CV R² with a
permutation null, checks the fitted sign, and compares held-out 80% interval
coverage with the shipped feature set on exactly the same grouped folds.

All four ship rules must pass before this column may enter app/pricing:

  1. nested-CV R² beats the permutation null's 95th percentile;
  2. the international coefficient is negative;
  3. held-out 80% band coverage does not drop versus the base model; and
  4. leave-one-brand-out within TR is informative rather than merely flattering.

    .venv/bin/python scripts/probe_usage_class.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                         # noqa: E402
from app.pricing import features as F          # noqa: E402

SEED = 7
N_PERM = 100
ALPHAS = (0.1, 1.0, 10.0, 100.0)
INTERNATIONAL = "Yurt Dışı Lojistik"


def usage_international(values: pd.Series) -> np.ndarray:
    """One only for the named international class; construction/unknown are zero."""
    return (values.fillna("").astype(str).str.strip() == INTERNATIONAL).to_numpy(
        dtype=float
    )


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit(X: np.ndarray, y: np.ndarray, alpha: float):
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    model = Ridge(alpha=alpha).fit((X - mean) / scale, y)
    return model, mean, scale


def predict(fitted, X: np.ndarray) -> np.ndarray:
    model, mean, scale = fitted
    return model.predict((X - mean) / scale)


def choose_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    n_splits = min(4, len(np.unique(groups)))
    if n_splits < 2:
        return ALPHAS[0]
    best_alpha, best_score = ALPHAS[0], -np.inf
    for alpha in ALPHAS:
        inner = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            inner[te] = predict(fit(X[tr], y[tr], alpha), X[te])
        score = r2(y, inner)
        if score > best_score:
            best_alpha, best_score = alpha, score
    return best_alpha


def nested_predictions(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Outer predictions with ridge alpha selected only inside each training fold."""
    pred = np.full(len(y), np.nan)
    n_splits = min(5, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        alpha = choose_alpha(X[tr], y[tr], groups[tr])
        pred[te] = predict(fit(X[tr], y[tr], alpha), X[te])
    return pred


def offsets(residuals: np.ndarray, level: float = 0.8) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return (
        float(np.quantile(residuals, tail)),
        float(np.quantile(residuals, 1.0 - tail)),
    )


def grouped_coverage(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, int]:
    """Held-out coverage; each outer fold gets offsets from training-fold OOF errors."""
    hits: list[bool] = []
    n_splits = min(5, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        alpha = choose_alpha(X[tr], y[tr], groups[tr])
        outer_fit = fit(X[tr], y[tr], alpha)

        inner_pred = np.full(len(tr), np.nan)
        inner_groups = groups[tr]
        inner_splits = min(5, len(np.unique(inner_groups)))
        for a, b in GroupKFold(n_splits=inner_splits).split(
            X[tr], y[tr], inner_groups
        ):
            inner_alpha = choose_alpha(
                X[tr][a], y[tr][a], inner_groups[a]
            )
            inner_pred[b] = predict(
                fit(X[tr][a], y[tr][a], inner_alpha), X[tr][b]
            )
        lo, hi = offsets(y[tr] - inner_pred)
        held_pred = predict(outer_fit, X[te])
        hits.extend(((y[te] >= held_pred + lo) & (y[te] <= held_pred + hi)).tolist())
    return float(np.mean(hits)), len(hits)


def leave_one_brand_out_note(
    df: pd.DataFrame, base: np.ndarray, dummy: np.ndarray
) -> None:
    """Report whether either Turkish brand can validate the proposed effect."""
    print("\nwithin-TR leave-one-brand-out (Ford vs MAN):")
    for brand in sorted(df.brand.unique()):
        held = df.brand.to_numpy() == brand
        train = ~held
        train_international = int(dummy[train].sum())
        held_international = int(dummy[held].sum())
        print(
            f"  hold out {brand:4s}: n={int(held.sum())}, "
            f"train n={int(train.sum())}, international "
            f"train={train_international}, held={held_international}"
        )
        if len(np.unique(dummy[train])) < 2:
            print("    not estimable: the training brand has no usage-class variation")
        elif held_international == 0:
            print(
                "    does not validate the international effect: every held-out "
                "listing is domestic"
            )
        else:
            X = np.column_stack([base[train], dummy[train]])
            alpha = choose_alpha(X, df.y.to_numpy()[train], df.group.to_numpy()[train])
            fitted = fit(X, df.y.to_numpy()[train], alpha)
            pred = predict(fitted, np.column_stack([base[held], dummy[held]]))
            median_ape = 100 * float(
                np.median(np.abs(np.expm1(pred - df.y.to_numpy()[held])))
            )
            print(f"    median held-out error: {median_ape:.1f}%")
    print(
        "  MAN is only six listings and has zero international examples; "
        "leave-one-brand-out cannot independently validate this dummy."
    )


def main() -> int:
    listings = pd.read_csv(config.LISTINGS_CSV)
    df = F.build_frame(listings)
    tr = df[df.market.str.upper() == "TR"].copy().reset_index(drop=True)
    brands = F.brand_vocabulary(tr, min_n=3)
    base = F.matrix(tr, brands)
    dummy = usage_international(tr.usage_class)
    augmented = np.column_stack([base, dummy])
    y, groups = tr.y.to_numpy(), tr.group.to_numpy()

    counts = tr.usage_class.fillna("<unknown>").value_counts()
    print(
        f"TR corpus: n={len(tr)}, {tr.group.nunique()} spec groups; "
        f"international={int(dummy.sum())}, other={int((1 - dummy).sum())}"
    )
    print("usage classes (construction collapses into domestic; unknown is 0):")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    base_r2 = r2(y, nested_predictions(base, y, groups))
    real_r2 = r2(y, nested_predictions(augmented, y, groups))

    rng = np.random.default_rng(SEED)
    null = np.array(
        [
            r2(
                y,
                nested_predictions(
                    np.column_stack([base, rng.permutation(dummy)]), y, groups
                ),
            )
            for _ in range(N_PERM)
        ]
    )
    null_p95 = float(np.percentile(null, 95))
    p_value = float(np.mean(null >= real_r2))

    alpha = choose_alpha(augmented, y, groups)
    fitted = fit(augmented, y, alpha)
    coefficient = float(fitted[0].coef_[-1])

    base_coverage, base_n = grouped_coverage(base, y, groups)
    usage_coverage, usage_n = grouped_coverage(augmented, y, groups)

    print(f"\nbase nested CV R^2:             {base_r2:+.4f}")
    print(f"with usage dummy nested CV R^2: {real_r2:+.4f}")
    print(
        f"permuted-dummy null: median {np.median(null):+.4f}, "
        f"95th pct {null_p95:+.4f} (n={len(null)}), p={p_value:.3f}"
    )
    print(
        f"international coefficient:      {coefficient:+.6f} "
        "(standardized ridge coefficient)"
    )
    print(
        f"held-out 80% band coverage:      base {base_coverage:.3f} "
        f"(n={base_n}) -> usage {usage_coverage:.3f} (n={usage_n})"
    )

    leave_one_brand_out_note(tr, base, dummy)

    rules = {
        "R2 beats permutation-null 95th percentile": real_r2 > null_p95,
        "international coefficient is negative": coefficient < 0,
        "80% coverage does not drop": usage_coverage >= base_coverage,
        "within-TR leave-one-brand-out validates effect": False,
    }
    print("\npre-registered ship rule:")
    for label, passed in rules.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")

    ship = all(rules.values())
    print("\nSHIP RULE:", "PASS" if ship else "FAIL")
    if ship:
        print("All four rules pass; the feature is eligible for implementation.")
    else:
        print(
            "Negative result: do not add usage_international to app/pricing. "
            "Weights and features remain unchanged."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
