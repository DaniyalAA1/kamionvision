# 0001 — The served price is the measured blend, with its own band (2026-09-12)

The screen's price was the hedonic fit blended with the new-price anchor by inverse variance, but
only the hedonic fit had ever been scored, and the band drawn around the blend was the hedonic
band. The coverage printed next to the price therefore described a different estimator. This
decision makes `train.fit_blend` score the served combination and `estimate()` use its band.

## What was measured

84 Turkish listings, 24 spec groups; 5-fold grouped OOF for accuracy, 40 grouped shuffle splits
for coverage, band from inner OOF residuals (the same protocol as `train.evaluate`).

| Estimator | R² | median error | 80% band covers | band width (expm1 hi−lo) |
|---|---|---|---|---|
| hedonic only (was reported) | 0.842 | 4.0% | 0.803 | 31.8% |
| inverse-variance blend (was served) | 0.924 | 3.6% | 0.874 | hedonic band |
| **least-squares blend (now served)** | **0.938** | **3.1%** | **0.809** | **16.7%** |

Learned anchor share: 1.00 in all 5 folds and on the full fit. Re-run: `.venv/bin/python -m
app.pricing.train` (prints the "served blend" block).

## Scope — what this does NOT claim

- Not that the hedonic fit is useless. It prices every make with no official new-price row, and
  its brand columns are worth +0.017 R² (the −0.024 printed before was a column-slicing bug).
- Not measured for unseen makes or `trade_press` rows — those keep the old inverse-variance path
  and its floor at the hedonic band. Mercedes and Scania are unchanged.
- Not a statement about realised sale prices. These are asking prices.
- n is 24 spec groups, 2 makes. A weight of 1.00 on this corpus is a finding about this corpus.

## Decision

`estimate()` uses the measured blend only when the make has its own hedonic column **and** the
matched new-price row is `oem_official`. `model_card.estimator` names which path ran, and the
card's figures belong to that path. The condition cap stays the hedonic `residual_std` on both.

## What this supersedes

The `CLAUDE.md` invariant "the anchor may claw back a widening; it may never narrow below the
measured band" — now scoped to the unmeasured path.

## Open, deliberately not answered here

- With weight 1.00 the Ford/MAN price rests on 5 hand-curated new-price rows, oldest 219 days.
  How fast does that go stale at ~31% Turkish inflation? Unmeasured.
- The 90% band under-covers on both paths (0.86 / 0.87). A finite-sample quantile correction
  was not tried.
