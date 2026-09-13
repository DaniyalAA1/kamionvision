"""Should `dashboard_odometer` be split into a dashboard class and a steering class?

`app/vision.py`'s `VIEW_PROMPTS["dashboard_odometer"]` is one class covering
three things at once - an odometer readout, a gauge cluster, and a steering
wheel - and one of its own templates says so out loud ("a steering wheel and
dash panel photographed from inside"). Downstream, `evidence/prompts.py` already
tracks `steering_wheel_controls` and `dashboard_instruments` as separate
findings, but the coarse VIEW label is what drives two things that cannot see
that distinction:

  * `web/js/elevation.js`'s VIEW_ZONES lights `dashboard_instruments`,
    `warning_lights` AND `steering_wheel_controls` the instant any one
    dashboard-tagged frame exists, whichever of the three it actually shows
  * `dashboard_odometer` is one of three REQUIRED views in
    models/gate_thresholds.json, so whatever wins that label earns the coverage

So: does a separate `steering_wheel` class separate these cleanly enough to be
worth wiring through five files? This script answers that against hand labels
rather than against a hunch, the same way the per-class prompt ensemble itself
was settled (see the comment at the top of app/vision.py): candidates are
chosen on `dev` and the number that counts is `test`.

    .venv/bin/python scripts/probe_steering_wheel_view.py

Reads data/reference/steering_wheel_labels.jsonl (160 hand-labelled frames) and
data/metadata/embeddings.npz (scripts/cache_embeddings.py). Scoring goes through
`vision.ClipTagger.score_bank`, which is the same max-pooled path `tag()` uses -
a second implementation of the pooling or of the logit_scale multiply is exactly
the kind of thing that would make a candidate look better than it is.
"""
from __future__ import annotations

import json
import pathlib as _pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

from app import config, vision  # noqa: E402

LABELS = config.DATA / "reference" / "steering_wheel_labels.jsonl"
EMB = config.META / "embeddings.npz"

# `dashboard_odometer` with its steering-wheel template removed. Every candidate
# below is scored against this trimmed class, because leaving the wheel template
# in both places would be measuring nothing.
DASH_TRIMMED = [
    "a close-up photo of a truck dashboard, gauges and odometer",
    "an instrument cluster with speedometer and warning lights",
    "the dashboard controls and switches of a lorry",
]

# Six candidate `steering_wheel` banks, deliberately including the obvious
# phrasing, a cab-scoped one, one that names the column stalks as their own
# template, and a one-liner - the ensemble comment in app/vision.py says one
# template scored worse for the existing classes, so don't assume for this one.
CANDIDATES = {
    "v1_plain": [
        "a close-up photo of a truck steering wheel",
        "the steering wheel rim and hub of a lorry photographed from the driver's seat",
        "a steering wheel with control buttons on its spokes",
        "the steering column stalks and wheel of a heavy vehicle",
    ],
    "v2_cab_scoped": [
        "a close-up photo of a steering wheel inside a truck cab",
        "the rim and centre boss of a lorry's steering wheel, seen from the driver's seat",
        "a steering wheel held by the driver, its spoke buttons in view",
        "the indicator and wiper stalks on a truck's steering column",
    ],
    "v3_contrastive": [
        "a close-up photo of the steering wheel inside a truck cab, not a road wheel",
        "a lorry's steering wheel rim and hub, upholstered and hand-worn",
        "the steering wheel spokes and their control buttons in a truck cab",
        "the stalk levers beside a truck steering wheel",
    ],
    "v4_minimal": [
        "a close-up photo of a truck steering wheel",
    ],
    "v5_stalks_named": [
        "a close-up photo of a truck steering wheel",
        "a steering wheel rim, hub and spoke controls",
        "a gear selector or indicator stalk on a steering column",
        "the wiper and cruise control stalks of a lorry",
        "a worn steering wheel rim photographed from the driver's seat",
    ],
    "v6_wheel_only_no_stalks": [
        "a close-up photo of a truck steering wheel",
        "the rim and hub of a lorry steering wheel filling the frame",
        "a steering wheel with buttons on its spokes",
    ],
}

# How a hand label maps onto the view id that SHOULD win, under each taxonomy.
WANT_SPLIT = {"steering": "steering_wheel", "gauges": "dashboard_odometer",
              "dash_general": "dashboard_odometer"}
WANT_LUMPED = {"steering": "dashboard_odometer", "gauges": "dashboard_odometer",
               "dash_general": "dashboard_odometer"}


def load():
    rows = [json.loads(l) for l in open(LABELS, encoding="utf-8")]
    z = np.load(EMB, allow_pickle=True)
    pos = {iid: i for i, iid in enumerate(z["image_id"])}
    missing = [r["image_id"] for r in rows if r["image_id"] not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} labelled frames are not in the embedding "
                         f"cache; run scripts/cache_embeddings.py. First: {missing[0]}")
    emb = z["emb"][[pos[r["image_id"]] for r in rows]]
    return rows, emb


def taxonomy(steering: list[str] | None):
    banks = dict(vision.VIEW_PROMPTS)
    if steering is None:
        return banks
    banks["dashboard_odometer"] = DASH_TRIMMED
    banks["steering_wheel"] = steering
    return banks


def score(tagger, emb, banks):
    labels = list(banks)
    probs = tagger.score_bank(emb, banks)
    return [labels[i] for i in probs.argmax(1)]


def report(rows, pred, split, want, new_class):
    """The three numbers a split has to win on, and the one it has to not lose."""
    sub = [(r, p) for r, p in zip(rows, pred) if r["split"] == split]
    steer = [(r, p) for r, p in sub if r["label"] == "steering"]
    claimed = [(r, p) for r, p in sub if p == new_class]
    interior = [(r, p) for r, p in sub if r["label"] in want]

    recall = sum(p == want.get(r["label"]) for r, p in steer) / max(1, len(steer))
    prec = sum(r["label"] == "steering" for r, _ in claimed) / max(1, len(claimed))
    # An interior frame whose correct answer is `dashboard_odometer` and which
    # no longer gets it has lost its coverage credit and its question bank.
    kept = sum(p == want[r["label"]] for r, p in interior) / max(1, len(interior))
    stolen = sum(1 for r, p in sub
                 if r["label"] == "not_interior_dash" and p == new_class)
    n_non = sum(1 for r, _ in sub if r["label"] == "not_interior_dash")
    return {"n": len(sub), "recall_steering": recall, "precision_new": prec,
            "n_claimed": len(claimed), "dash_kept": kept, "n_interior": len(interior),
            "stolen_non_interior": stolen, "n_non_interior": n_non,
            "wrong": [(r["image_id"], r["label"], p) for r, p in claimed
                      if r["label"] != "steering"]}


def line(tag, m):
    return (f"  {tag:24s} n={m['n']:3d}  steering recall {m['recall_steering']:6.1%}"
            f"  |  new-class precision {m['precision_new']:6.1%} over "
            f"{m['n_claimed']:3d} claimed  |  dashboard frames keeping their label "
            f"{m['dash_kept']:6.1%}  |  non-interior frames stolen "
            f"{m['stolen_non_interior']:2d}/{m['n_non_interior']:2d}")


def main() -> None:
    rows, emb = load()
    print(f"{len(rows)} hand-labelled frames from {LABELS.name}")
    for s in ("dev", "test"):
        c = Counter(r["label"] for r in rows if r["split"] == s)
        print(f"  {s:4s} n={sum(c.values()):3d} " + " ".join(
            f"{k}={v}" for k, v in sorted(c.items())))
    rand = [r for r in rows if r["sample"] == "random"]
    c = Counter(r["label"] for r in rand)
    print(f"\nBASE RATE, pass A only ({len(rand)} frames sampled at random from the "
          f"interior classes, not retrieved):")
    for k, v in sorted(c.items()):
        print(f"  {k:18s} {v:3d}  {v/len(rand):6.1%}")

    tagger = vision.clip()

    print("\nLUMPED - the taxonomy as shipped, 11 classes, no steering class.")
    print("What does it call the frames a steering class would want?")
    lumped = score(tagger, emb, taxonomy(None))
    for s in ("dev", "test"):
        got = Counter(p for r, p in zip(rows, lumped)
                      if r["split"] == s and r["label"] == "steering")
        n = sum(got.values())
        print(f"  {s:4s} n={n:3d}  " + "  ".join(
            f"{k}={v} ({v/n:.0%})" for k, v in got.most_common()))
    for s in ("dev", "test"):
        m = report(rows, lumped, s, WANT_LUMPED, "steering_wheel")
        print(line(f"lumped/{s}", m))

    print("\nSPLIT - 12 classes. Candidates are chosen on dev; test is the number.")
    dev_scores = {}
    preds = {}
    for name, bank in CANDIDATES.items():
        preds[name] = score(tagger, emb, taxonomy(bank))
        m = report(rows, preds[name], "dev", WANT_SPLIT, "steering_wheel")
        dev_scores[name] = m
        print(line(name, m))

    best = max(dev_scores, key=lambda k: (dev_scores[k]["precision_new"]
                                          + dev_scores[k]["recall_steering"]))
    print(f"\nbest on dev by precision+recall: {best}")
    for s in ("dev", "test"):
        m = report(rows, preds[best], s, WANT_SPLIT, "steering_wheel")
        print(line(f"{best}/{s}", m))
    m = report(rows, preds[best], "test", WANT_SPLIT, "steering_wheel")
    print(f"\nEvery frame {best} wrongly claims on test, and what it really is:")
    for iid, truth, _ in m["wrong"]:
        note = next(r["note"] for r in rows if r["image_id"] == iid)
        print(f"  {iid:32s} really {truth:18s} {note}")

    coverage_cost(tagger, CANDIDATES[best], best)


def coverage_cost(tagger, steering: list[str], name: str) -> None:
    """The vehicle-level consequence, corpus-wide, on all 200 vehicles.

    Truck detection and view coverage are set-level in this repo, so the number
    that decides a taxonomy change is not per-frame accuracy - it is how many
    VEHICLES lose a required view. `dashboard_odometer` and `tire_wheel` are two
    of the three entries in `coverage.required`, and a vehicle that loses one
    gets asked for a photograph its seller already sent.
    """
    import pandas as pd

    z = np.load(EMB, allow_pickle=True)
    pos = {iid: i for i, iid in enumerate(z["image_id"])}
    df = pd.read_csv(config.META / "images.csv")
    df = df[(df.variant == "original") & df.image_id.isin(pos)]
    emb = z["emb"][[pos[i] for i in df.image_id]]

    before = score(tagger, emb, taxonomy(None))
    after = score(tagger, emb, taxonomy(steering))
    print(f"\nCORPUS-WIDE, all {len(df)} original frames of {df.listing_id.nunique()} "
          f"vehicles, with {name}:")
    claimed = sum(p == "steering_wheel" for p in after)
    print(f"  frames the new class claims: {claimed} ({claimed/len(df):.1%})")
    print("  what the shipped taxonomy called them: " + ", ".join(
        f"{k}={v}" for k, v in Counter(
            b for b, a in zip(before, after) if a == "steering_wheel").most_common()))

    out = df.assign(before=before, after=after)
    for view in ("dashboard_odometer", "tire_wheel", "exterior_front_34"):
        had = out[out.before == view].groupby("listing_id").size()
        still = out[out.after == view].groupby("listing_id").size()
        lost = sorted(set(had.index) - set(still.index))
        print(f"  vehicles with a `{view}` frame: {len(had)} -> {len(still)}"
              f"   LOSE THE VIEW ENTIRELY: {len(lost)}")
        for lid in lost[:6]:
            print(f"      {lid} had {had[lid]} such frame(s), now 0")


if __name__ == "__main__":
    main()
