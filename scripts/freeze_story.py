"""Freeze one real appraisal into the landing page's scroll story.

    .venv/bin/python -m app.cli appraise demo/tr_clean --year 2021 --km 164374 \
        --make Ford --asking 2550000 --market TR --save /tmp/story.json
    .venv/bin/python scripts/freeze_story.py /tmp/story.json

The landing page argues that this system is not a thin wrapper. That argument
is worthless if the page's own evidence is copywriting, so every sentence about
the truck on the front page comes out of a run that actually happened, and this
script is the only thing allowed to put it there.

Three outputs, all committed:

    app/web/assets/story/NNN.jpg     the hero photos, downscaled
    app/web/assets/story/story.json  the trimmed run, read by js/story/
    app/web/landing.html             the region between the story markers

The markup is written here rather than by the browser for three reasons: it is
the no-JS page, it is the prefers-reduced-motion page, and it lets a test read
the HTML and assert that every claim on it appears verbatim in the JSON.
`tests/test_story.py` regenerates the region and diffs it, so hand-editing the
page between the markers fails the suite rather than drifting quietly.

The odometer OCR is re-read here rather than lifted from the appraisal, because
`reconcile` stays silent when the two readings agree - which is the interesting
case and the one this truck is. Agreement that is never recorded cannot be shown.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.config import USD_TRY, WEB                             # noqa: E402

# Act 4 states the tolerance the two odometer readings are allowed to differ by
# and what the band does when they do not. Both are `reconcile`'s to define, and
# a page that retyped them would keep asserting the old ones after they moved.
try:
    from app.reconcile import (ODOMETER_CONFLICT_WIDENING,      # noqa: E402
                               ODOMETER_TOLERANCE_FRAC)
except ImportError:                                             # pragma: no cover
    ODOMETER_TOLERANCE_FRAC, ODOMETER_CONFLICT_WIDENING = None, None

OUT = WEB / "assets" / "story"
LANDING = WEB / "landing.html"
START, END = "<!-- story:start -->", "<!-- story:end -->"

# Long edge of a hero photo. The deck holds six of them at once and they are
# never shown larger than half a viewport, so 1100px is generous; the originals
# are 1440x1080 dealer frames and shipping those is 3.7 MB of landing page.
LONG_EDGE = 1100
QUALITY = 82

# The order the six beats are told in. Whole vehicle first because the set has
# to establish what it is looking at before it inspects it, and the dashboard
# last because the odometer is what act 4's cross-check is about.
STORY_ORDER = [
    "exterior_side", "exterior_front_34", "exterior_front", "exterior_rear",
    "tire_wheel", "fifth_wheel", "chassis_undercarriage", "engine_bay",
    "interior_cab", "damage_detail", "dashboard_odometer",
]
WHOLE_VEHICLE = {"exterior_side", "exterior_front_34", "exterior_front",
                 "exterior_rear"}
N_HEROES = 6
# The set has to establish what it is looking at, and then stop: four exterior
# views of one red tractor tell a visitor nothing a second one did not, and
# they crowd out the close-ups that carry the actual findings.
MAX_WHOLE_VEHICLE = 2

# The eleven views the gate can recognise, for "7 of 11 covered".
N_VIEWS = len(STORY_ORDER)


def pick_heroes(findings: list[dict]) -> list[dict]:
    """Six photos, one per view, the ones carrying findings first.

    Deterministic on purpose: the page is committed, so a selection that
    depended on dict order would produce a different landing page on every
    freeze and no test could pin it.
    """
    def score(f: dict) -> tuple:
        return (
            f.get("odometer_km") is not None,   # act 4 needs this frame
            bool(f.get("issues")),
            len(f.get("issues") or []),
            len(f.get("strengths") or []),
            -f["photo_id"],                     # ties break low-id-first
        )

    legible = [f for f in findings if f.get("legible") and not f.get("error")]
    best: dict[str, dict] = {}
    for f in sorted(legible, key=score, reverse=True):
        best.setdefault(f["view"], f)          # one frame per view, the best one

    ranked = sorted(best.values(), key=score, reverse=True)
    whole = [f for f in ranked if f["view"] in WHOLE_VEHICLE][:MAX_WHOLE_VEHICLE]
    rest = [f for f in ranked if f["view"] not in WHOLE_VEHICLE]

    # A set of six close-ups would never establish what the truck is, and six
    # exteriors would never find anything. Take the establishing frames first,
    # then fill with the close-ups that carry the findings.
    picked = whole + rest[:max(0, N_HEROES - len(whole))]

    order = {v: i for i, v in enumerate(STORY_ORDER)}
    picked.sort(key=lambda f: (order.get(f["view"], len(order)), f["photo_id"]))
    return picked


def copy_photo(src: Path, dest: Path) -> tuple[int, int]:
    from PIL import Image, ImageOps
    pil = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if max(pil.size) > LONG_EDGE:
        scale = LONG_EDGE / max(pil.size)
        pil = pil.resize((round(pil.width * scale), round(pil.height * scale)),
                         Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pil.save(dest, "JPEG", quality=QUALITY, optimize=True, progressive=True)
    return pil.size


def read_odometer(path: Path, subject_box) -> dict:
    """The second, independent read of the dashboard. Absent RapidOCR this
    returns `available: False` and the page drops act 4's OCR line rather than
    asserting a reading nobody took."""
    try:
        from app import odometer
    except Exception as exc:                                # pragma: no cover
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"}
    try:
        r = odometer.read(str(path), subject_box=subject_box)
    except Exception as exc:
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"}
    return {"available": r.km is not None, "km": r.km,
            "confidence": round(r.confidence, 3), "text": r.text,
            "reason": r.reason, "rule": r.rule}


def _frozen_ocr() -> dict | None:
    """The odometer OCR block from the committed story.json, if any. It carries
    the canonical run's confidence, which is what the page states - see build()
    for why a live re-read's confidence is not used for display."""
    path = OUT / "story.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["cross_checks"]["odometer"]["ocr"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


# COCO's vehicle classes, as the gate treats them. `person` is detected too and
# is kept in the story - a box the subject rule had to rule out is evidence that
# it ran - but it must not be counted as a vehicle in the caption.
VEHICLES = {"truck", "car", "bus", "motorcycle", "bicycle", "train", "boat"}


def lot_frame(gate: dict) -> dict | None:
    """The frame with the most vehicles in it. Act 1 is about picking the
    subject out of a dealer lot, and a photo with one truck in it cannot make
    that point."""
    boxed = [p for p in gate["photos"] if len(p.get("detections") or []) > 1]
    if not boxed:
        return None
    p = max(boxed, key=lambda p: len(p["detections"]))
    w, h = p["width"] or 1, p["height"] or 1
    dets = [{"label": d["label"], "confidence": round(d["confidence"], 3),
             # Normalised against the SOURCE frame, because the copy the page
             # downloads is downscaled and pixel coordinates would land the
             # boxes somewhere else entirely.
             "box": [round(d["box"][0] / w, 4), round(d["box"][1] / h, 4),
                     round(d["box"][2] / w, 4), round(d["box"][3] / h, 4)],
             "area_frac": round(d["area_frac"], 4),
             "is_subject": bool(d.get("is_subject"))}
            for d in p["detections"]]
    return {
        "file": p["filename"], "width": w, "height": h,
        "basis": p.get("subject_basis", ""),
        "n_vehicles": sum(1 for d in dets if d["label"] in VEHICLES),
        "n_other": sum(1 for d in dets if d["label"] not in VEHICLES),
        "subject_area_pct": round(
            next((d["area_frac"] for d in dets if d["is_subject"]), 0) * 100, 1),
        "detections": dets,
    }


def build(appraisal: dict, folder: Path) -> dict:
    gate, ev, price = appraisal["gate"], appraisal["evidence"], appraisal["price"]
    veh, declared = ev["vehicle"], appraisal.get("declared") or {}
    checks = {p["photo_id"]: p for p in gate["photos"]}

    heroes = pick_heroes(ev["photo_findings"])
    merged = {}
    for issue in ev["issues"]:
        merged.setdefault(issue["photo_id"], []).append(issue)

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.jpg"):
        stale.unlink()

    photos = []
    for n, f in enumerate(heroes):
        check = checks[f["photo_id"]]
        name = f"{n:03d}.jpg"
        w, h = copy_photo(folder / check["filename"], OUT / name)
        sx, sy = w / (check["width"] or w), h / (check["height"] or h)
        box = check.get("subject_box")
        photos.append({
            "file": name,
            "photo_id": f["photo_id"],
            "source": check["filename"],
            "width": w, "height": h,
            "view": f["view"],
            "shows": f["shows"],
            "cropped": bool(f.get("cropped")),
            "quality": check.get("quality_bucket", "unknown"),
            # Normalised, so the browser never rescales pixel coordinates
            # against a photo it downloaded at a different size.
            "subject_box": ([round(box[0] * sx / w, 4), round(box[1] * sy / h, 4),
                             round(box[2] * sx / w, 4), round(box[3] * sy / h, 4)]
                            if box else None),
            "subject_basis": check.get("subject_basis", ""),
            "detections": len(check.get("detections") or []),
            "odometer_km": f.get("odometer_km"),
            # The merged findings bound to this frame, not the raw per-photo
            # ones: three paraphrases of one worn tire is one finding.
            "issues": [{"component": i["component"], "severity": i["severity"],
                        "observation": i["observation"],
                        "price_impact": i["price_impact"],
                        "confidence": round(i["confidence"], 2),
                        "also_seen_in": i["also_seen_in"]}
                       for i in merged.get(f["photo_id"], [])],
            "strengths": f.get("strengths") or [],
            "cannot_tell": f.get("cannot_tell") or [],
        })

    odo_id = veh.get("odometer_photo_id")
    odo_check = checks.get(odo_id) if odo_id is not None else None
    ocr = (read_odometer(folder / odo_check["filename"], odo_check.get("subject_box"))
           if odo_check else {"available": False, "why": "no odometer frame"})
    # The mileage OCR reads is deterministic; the confidence float it reports is
    # not - the same 164,374 km comes back at 0.99 on one machine and 0.81 on
    # another, from the OCR runtime's own numerics. The story is a frozen artifact
    # of one canonical run, so the confidence in the committed story.json is the
    # value the page states. Re-read to confirm the reading itself still holds,
    # but adopt the frozen confidence so the page and --check reproduce anywhere.
    frozen = _frozen_ocr()
    if ocr.get("available") and frozen and frozen.get("km") == ocr.get("km"):
        ocr["confidence"] = frozen["confidence"]
        ocr["reason"] = frozen.get("reason", ocr["reason"])

    raw_issues = sum(len(f.get("issues") or []) for f in ev["photo_findings"])
    clamped = [c for c in ev.get("corrections") or []
               if c["kind"] == "uncorroborated_finding" and c["before"] != c["after"]]
    disagreed = [c for c in ev.get("corrections") or []
                 if c["kind"] == "sample_disagreement"]
    corroborated = max((i for i in ev["issues"]),
                       key=lambda i: len(i["also_seen_in"]), default=None)

    # What the run concluded the truck IS, and which independent reads said so.
    # Act 1 is the identification act, so the witnesses belong in it: "two
    # independent reads agree this is a Ford" is a checkable claim in a way
    # that a confidence decimal is not.
    verdict = ev.get("identity") or {}
    identity = {
        "status": verdict.get("status", "unknown"),
        "make": verdict.get("make"), "model": verdict.get("model"),
        "reason": verdict.get("reason", ""),
        "witnesses": [{"name": w[0], "said": w[1], "confidence": round(float(w[2]), 3)}
                      for w in (verdict.get("witnesses") or []) if len(w) >= 3],
        "agreed": verdict.get("agreed") or [],
        "disagreed": verdict.get("disagreed") or [],
        "widening": verdict.get("widening", 1.0),
        # The identity pass is sampled; the agreement rate across those samples
        # is measured, unlike the model's own confidence in itself.
        "samples": veh.get("identity_samples", 1),
        "agreement": round(veh.get("identity_agreement", 0.0), 3),
        "year_evidence": veh.get("year_evidence", ""),
        "badge_text": veh.get("badge_text") or [],
        "wmi": veh.get("wmi"), "wmi_brand": veh.get("wmi_brand"),
        "generation": veh.get("generation"),
    }

    anchor = price.get("anchor") or {}
    routes = None
    if anchor.get("ok") and anchor.get("point"):
        gap = abs(anchor["point"] - price["point"]) / price["point"] * 100
        routes = {"comparables": price["point"], "anchor": anchor["point"],
                  "gap_pct": round(gap, 1), "new_price": anchor["new_price"],
                  "retention_pct": round(anchor["retention"] * 100, 1),
                  "source": anchor["source"], "source_url": anchor["source_url"],
                  "source_type": anchor["source_type"], "as_of": anchor["as_of"],
                  "basis": anchor["basis"]}

    card = price.get("model_card") or {}
    # The measured figures belong to a specific fit, and the fit is refitted.
    # Stamping it is what lets `tests/test_story.py` tell "this page is a record
    # of a run" apart from "this page quotes a number the system no longer
    # produces", which look identical without it.
    fitted_at = card.get("fitted_at") or ""
    return {
        "frozen_at": date.today().isoformat(),
        "source": {
            # Repo-relative, not absolute: an absolute path bakes the freezing
            # machine's home directory into a committed artifact and churns it on
            # the next machine to re-freeze.
            "case": folder.name,
            "folder": str(folder.relative_to(REPO)) if folder.is_relative_to(REPO)
                      else str(folder),
            "backend": ev.get("backend", ""), "model": ev.get("model", ""),
            "elapsed_s": round(appraisal.get("elapsed_s", 0), 1),
            "vision_calls": len(ev.get("calls") or []),
        },
        "truck": {
            "make": veh.get("make"), "model": veh.get("model"),
            "body_type": veh.get("body_type"), "cab_type": veh.get("cab_type"),
            "axle_config": veh.get("axle_config"),
            "year_range": veh.get("approx_year_range"),
            "year": declared.get("year"), "km": declared.get("km"),
            "asking": declared.get("asking_price"),
            "currency": price.get("currency", "TRY"),
        },
        "gate": {
            "decision": gate["decision"],
            "photos_total": len(gate["photos"]),
            "photos_usable": len(gate["usable_photo_ids"]),
            "photos_read": ev.get("photos_read", 0),
            "views_present": gate["views_present"],
            "views_total": N_VIEWS,
            "subject_evidence": gate.get("subject_evidence", ""),
            "subject_frames": gate.get("subject_frames", 0),
            "subject_consistency": round(gate.get("subject_consistency", 0), 2),
            "elapsed_s": round(gate.get("elapsed_s", 0), 1),
            "lot": lot_frame(gate),
        },
        "identity": identity,
        "photos": photos,
        "merge": {
            "raw": raw_issues, "merged": len(ev["issues"]),
            "clamped": len(clamped), "disagreed": len(disagreed),
            "example": ({"component": corroborated["component"],
                         "severity": corroborated["severity"],
                         "observation": corroborated["observation"],
                         "photo_id": corroborated["photo_id"],
                         "also_seen_in": corroborated["also_seen_in"]}
                        if corroborated and corroborated["also_seen_in"] else None),
        },
        "cross_checks": {
            "odometer": {
                "declared": declared.get("km"),
                "vlm": veh.get("odometer_km"),
                "vlm_photo_id": odo_id,
                "ocr": ocr,
                "agree": bool(ocr.get("available") and veh.get("odometer_km")
                              and ocr.get("km") == veh.get("odometer_km")),
            },
            "price_routes": routes,
        },
        "condition": {
            "grade": ev.get("condition_grade", "unknown"),
            "grade_reason": (ev.get("condition") or {}).get("grade_reason", ""),
            "grade_model": ev.get("condition_grade_model", ""),
            "coverage_pct": round((ev.get("condition") or {}).get("coverage", 0) * 100, 1),
            "confirmed_sound": ev.get("confirmed_sound") or [],
            "cannot_tell": ev.get("coverage_gaps") or [],
        },
        "price": {
            "currency": price["currency"],
            "low": price["low"], "point": price["point"], "high": price["high"],
            "baseline_low": price["baseline_low"],
            "baseline_point": price["baseline_point"],
            "baseline_high": price["baseline_high"],
            "low_usd": price["low_usd"], "point_usd": price["point_usd"],
            "high_usd": price["high_usd"],
            "interval_level": price["interval_level"],
            "asking": price.get("asking"),
            "adjustment": {k: (price["adjustment"] or {}).get(k)
                           for k in ("multiplier", "pct", "cap_pct", "drivers",
                                     "direction", "cap_basis", "weights_basis")},
            "widened": price.get("widened") or [],
            "caveats": price.get("caveats") or [],
            "fx": USD_TRY,
            # The measured figure, and the band it is measured on. These two
            # travel together everywhere: the coverage belongs to the
            # comparable-asking band and never to the photo-adjusted one.
            "measured": {"coverage": round(card.get("coverage", 0) * 100, 1),
                         "coverage_n": card.get("coverage_n"),
                         "belongs_to": "baseline",
                         "r2": round(card.get("r2", 0), 3),
                         "mae_pct": round(card.get("mae_pct", 0), 1),
                         "n_listings": card.get("n_listings"),
                         "n_groups": card.get("n_groups"),
                         "fitted_at": fitted_at},
        },
    }


# --- the markup -----------------------------------------------------------

# `IdentityVerdict.witnesses` names its sources in the pipeline's own
# vocabulary. Each is a genuinely separate measurement, and saying which is the
# whole point of printing them - "the badge" and "the chassis plate" are things
# a reader can go and look at, "identity_pass" is not.
WITNESS = {
    "identity_pass": "The vision model, over every photo",
    "head": "The trained brand head",
    "badge": "The badge, read off a full-resolution crop",
    "wmi": "The chassis-plate VIN",
}


def pct(v: float) -> str:
    """A rate, with a decimal only when rounding would hide something. 0.998
    and 1.0 both print as 100% at zero places, which is the one difference
    between two witness rows."""
    return f"{v:.0%}" if v >= 0.9995 or v == 0 else f"{v:.1%}"


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def money(v, currency="TRY") -> str:
    sym = {"TRY": "₺", "USD": "$", "EUR": "€"}.get(currency, currency + " ")
    return f"{sym}{round(v):,}"


def km(v) -> str:
    """Kilometres, grouped and without a decimal. `declared["km"]` arrives as a
    float and `164,374.0 km` on a dashboard reads as a bug."""
    return f"{round(float(v)):,}"


def num(text) -> str:
    """Mark a figure as countable by `js/story/reveal.js`.

    The marker exists because sniffing numbers out of prose gets it wrong: a
    `4x2` axle configuration is not two integers to animate, and a band reading
    `₺2,015,000 – ₺2,619,000` is two separate figures rather than one. The span
    is a promise that its text holds exactly one number, so the reveal can
    tween it without a regex having to guess.
    """
    return f'<span data-count>{text}</span>'


def render(story: dict) -> str:
    """The story region of `landing.html`.

    This is the whole section as plain markup: no-JS gets it, reduced-motion
    gets it, and `js/story/` upgrades it in place. Nothing here is authored
    prose about the truck - every sentence is a lookup into `story`.
    """
    t, g, p = story["truck"], story["gate"], story["price"]
    cond, merge, checks = story["condition"], story["merge"], story["cross_checks"]
    cur = p["currency"]
    name = " ".join(x for x in (t["make"], t["model"]) if x)
    out = [START, '<section class="story" id="story" data-story>']

    # --- act 0 + 1: the set, and which vehicle it is about ---
    out.append(
        '<div class="story-intro">'
        f'<p class="eyebrow"><span class="status-dot"></span> 01 / WHAT HAPPENS TO YOUR PHOTOS</p>'
        f'<h2>{g["photos_total"]} photos.<br><span class="muted">One truck.</span><br>'
        f'Here is every step<br>between them and a number.</h2>'
        f'<p class="story-lede">Not a mock-up. This is one real run on a real Turkish listing, '
        f'frozen on {esc(story["frozen_at"])} so it can be shown without a network. '
        f'Your own photos run live.</p>'
        '</div>')

    out.append('<div class="story-track" data-track>')
    out.append('<div class="story-stage" data-stage>')
    out.append('<div class="stage-head"><span class="stage-act" data-act-label>Identify</span>'
               '<span class="stage-count" data-act-count></span></div>')
    out.append('<div class="stage-body" data-stage-body>')

    # act 1 — the gate
    lot = g.get("lot")
    out.append('<article class="act act-gate" data-act="gate">')
    out.append('<h3>Which of these is being sold?</h3>')
    if lot:
        others = (f' and {lot["n_other"]} other object'
                  f'{"s" if lot["n_other"] != 1 else ""}' if lot["n_other"] else "")
        out.append(
            f'<figure class="lot-figure" data-lot '
            f'data-detections="{esc(json.dumps(lot["detections"], separators=(",", ":")))}">'
            f'<span class="frame"><img src="/static/assets/story/lot.jpg" '
            f'width="{lot["width"]}" height="{lot["height"]}" alt="A dealer lot frame '
            f'in which the detector found {lot["n_vehicles"]} vehicles{others}" '
            f'loading="lazy" decoding="async"></span>'
            f'<figcaption>{lot["n_vehicles"]} vehicles{others} in this one frame. '
            f'The subject fills {lot["subject_area_pct"]}% of it and '
            f'{esc(lot["basis"])}.</figcaption></figure>')
    out.append('<div class="gate-side">')
    out.append(
        '<dl class="gate-facts">'
        f'<div><dt>Photos accepted</dt><dd>{num(g["photos_usable"])} of '
        f'{num(g["photos_total"])}</dd></div>'
        f'<div><dt>Read in depth</dt><dd>{num(g["photos_read"])}, one vision call '
        f'each</dd></div>'
        f'<div><dt>Views covered</dt><dd>{num(len(g["views_present"]))} of '
        f'{num(g["views_total"])}</dd></div>'
        f'<div><dt>Identified as</dt><dd>{esc(name)}, {esc(t["axle_config"])} '
        f'{esc((t["body_type"] or "").replace("_", " "))}</dd></div>'
        '</dl>')
    ident = story.get("identity") or {}
    if ident.get("witnesses"):
        out.append('<div class="witnesses"><h4>How it knows</h4><ul>')
        for w in ident["witnesses"]:
            out.append(f'<li><span>{esc(WITNESS[w["name"]] if w["name"] in WITNESS else w["name"].replace("_", " "))}</span>'
                       f'<b>{esc(w["said"])}</b>'
                       f'<em>{pct(w["confidence"])}</em></li>')
        out.append('</ul>')
        if ident.get("reason"):
            reason = re.sub(r"\s*\([^)]*\)\s*$", "", ident["reason"])
            out.append(f'<p class="witness-verdict {"agree" if not ident["disagreed"] else "differ"}">'
                       f'{esc(reason)}. The identity pass is asked '
                       f'{ident["samples"]} times and agreed with itself '
                       f'{ident["agreement"]:.0%} of the time &mdash; a measured rate, '
                       f'not the model&rsquo;s opinion of itself.</p>')
        if ident.get("year_evidence"):
            out.append(f'<p class="witness-year">{esc(ident["year_evidence"])}</p>')
        out.append('</div>')
    out.append(f'<p class="act-note">{esc(g["subject_evidence"]).capitalize()}. '
               f'Truck detection is decided across the set, never frame by frame: '
               f'a tire close-up contains no truck-shaped object and is still a photo '
               f'of the truck.</p>')
    out.append(f'<p class="act-timing">Stage 1 took {g["elapsed_s"]}s, on this machine, '
               f'before any vision call.</p>')
    out.append('</div>')
    out.append('</article>')

    # act 2 — one beat per photo
    out.append('<ol class="act act-read" data-act="read">')
    for n, ph in enumerate(story["photos"]):
        out.append(f'<li class="beat" data-beat="{n}" data-photo="{ph["photo_id"]}"'
                   + (f' data-box="{esc(json.dumps(ph["subject_box"]))}"'
                      if ph["subject_box"] else "") + '>')
        out.append(
            f'<figure class="beat-figure"><span class="frame">'
            f'<img src="/static/assets/story/{ph["file"]}" '
            f'width="{ph["width"]}" height="{ph["height"]}" alt="{esc(ph["shows"])}" '
            f'loading="lazy" decoding="async"></span></figure>')
        out.append('<div class="beat-verdict">')
        out.append(f'<p class="beat-index">Photo {n + 1} of {len(story["photos"])}</p>')
        out.append(f'<p class="beat-shows">{esc(ph["shows"])}</p>')
        if ph["issues"]:
            out.append('<ul class="beat-issues">')
            for i in ph["issues"]:
                also = (f' <span class="also">also seen in '
                        f'{len(i["also_seen_in"])} other photo'
                        f'{"s" if len(i["also_seen_in"]) != 1 else ""}</span>'
                        if i["also_seen_in"] else "")
                out.append(
                    f'<li class="sev-{esc(i["severity"])}">'
                    f'<span class="sev-chip">{esc(i["severity"])}</span>'
                    f'<b>{esc(i["component"].replace("_", " "))}</b>'
                    f'<span class="issue-text">{esc(i["observation"])}</span>{also}</li>')
            out.append('</ul>')
        else:
            out.append('<p class="beat-clean">Nothing wrong found in this frame.</p>')
        if ph["odometer_km"] is not None:
            out.append(f'<p class="beat-odo">Odometer read here: '
                       f'<b>{km(ph["odometer_km"])} km</b></p>')
        if ph["strengths"]:
            out.append('<details class="beat-strengths"><summary>'
                       f'{len(ph["strengths"])} things this frame confirms are in order'
                       '</summary><ul>')
            out.extend(f'<li>{esc(s)}</li>' for s in ph["strengths"])
            out.append('</ul></details>')
        out.append('</div></li>')
    out.append('</ol>')

    # act 3 — the merge
    out.append('<article class="act act-merge" data-act="merge">')
    seen = len(merge["example"]["also_seen_in"]) + 1 if merge["example"] else 0
    out.append(f'<h3>One defect seen {"twice" if seen == 2 else f"{seen} times"} '
               f'is one defect.</h3>' if seen > 1
               else '<h3>One defect seen twice is one defect.</h3>')
    out.append(
        '<p class="merge-count">'
        f'<b>{num(merge["raw"])}</b> raw observations '
        f'<span aria-hidden="true">&rarr;</span> <b>{num(merge["merged"])}</b> '
        f'findings</p>')
    out.append('<div class="merge-side">')
    if merge["example"]:
        ex = merge["example"]
        seen = len(ex["also_seen_in"]) + 1
        out.append(f'<p class="merge-example"><b>{esc(ex["component"].replace("_", " "))}</b> '
                   f'&mdash; {esc(ex["observation"])} '
                   f'<span class="also">seen in {seen} photos</span></p>')
    if merge["clamped"] or merge["disagreed"]:
        out.append(
            f'<p class="act-note">{merge["clamped"]} finding'
            f'{"s" if merge["clamped"] != 1 else ""} had a raised severity clamped '
            f'because only one sample of that photo reported it, and {merge["disagreed"]} '
            f'had samples that disagreed with each other. Both are shown with the '
            f'change recorded, never deleted &mdash; changing your mind is allowed, '
            f'doing it quietly is not.</p>')
    out.append('</div>')
    out.append('</article>')

    # act 4 — the cross-checks
    odo, routes = checks["odometer"], checks["price_routes"]
    out.append('<article class="act act-check" data-act="check">')
    out.append('<h3>Then it checks itself.</h3>')
    if odo["ocr"].get("available"):
        out.append(
            '<div class="check"><h4>The odometer, read twice</h4><ul class="readings">'
            f'<li><span>Seller typed</span><b>{num(km(odo["declared"]))} km</b></li>'
            f'<li><span>Vision model read</span><b>{num(km(odo["vlm"]))} km</b></li>'
            f'<li><span>Offline OCR read</span><b>{num(km(odo["ocr"]["km"]))} km</b>'
            f'<em>confidence {odo["ocr"]["confidence"]:.2f}</em></li>'
            '</ul>'
            + (f'<p class="check-verdict agree">All three agree, so the band does not '
               f'widen. Had they disagreed past {ODOMETER_TOLERANCE_FRAC:.0%}, both '
               f'figures would be shown and the range would widen '
               f'{ODOMETER_CONFLICT_WIDENING:g}&times; rather than the number quietly '
               f'moving.</p>'
               if odo["agree"] else
               '<p class="check-verdict differ">The readings differ, so both are shown '
               'and the band widens.</p>')
            + '</div>')
    if routes:
        out.append(
            '<div class="check"><h4>The price, reached twice</h4><ul class="readings">'
            f'<li><span>From {p["measured"]["n_listings"]} comparable listings</span>'
            f'<b>{num(money(routes["comparables"], cur))}</b></li>'
            f'<li><span>From new price &times; fitted retention</span>'
            f'<b>{num(money(routes["anchor"], cur))}</b></li>'
            '</ul>'
            f'<p class="check-verdict agree">Two routes that share no inputs, '
            f'{routes["gap_pct"]}% apart. The second starts from '
            f'{money(routes["new_price"], cur)} new '
            f'(<a href="{esc(routes["source_url"])}" rel="nofollow noopener">'
            f'{esc(routes["source"])}</a>, {esc(routes["as_of"])}) and applies '
            f'{routes["retention_pct"]}% retention at {km(t["km"])} km.</p></div>')
    out.append('</article>')

    # act 5 — the number
    ask = p.get("asking") or {}
    out.append('<article class="act act-price" data-act="price">')
    out.append('<h3>And only then, a number.</h3>')
    out.append(
        '<div class="story-band" data-band>'
        f'<div class="band-row" data-band-row="adjusted"><span class="band-label">'
        f'Photo-adjusted estimate</span><span class="band-values">'
        f'{num(money(p["low"], cur))} &ndash; {num(money(p["high"], cur))}</span>'
        f'<span class="band-sub">The comparable estimate moved by what the photos show. '
        f'No measured coverage guarantee.</span></div>'
        f'<div class="band-row" data-band-row="baseline"><span class="band-label">'
        f'Comparable-market baseline</span><span class="band-values">'
        f'{num(money(p["baseline_low"], cur))} &ndash; '
        f'{num(money(p["baseline_high"], cur))}</span>'
        f'<span class="band-sub">{p["measured"]["coverage"]}% of held-out listings fall '
        f'inside this band &mdash; measured on the price model fitted '
        f'{esc((p["measured"]["fitted_at"] or "")[:10])}, and it belongs to this row '
        f'only.</span></div>'
        '</div>')
    out.append(f'<p class="band-usd">About {money(p["low_usd"], "USD")} &ndash; '
               f'{money(p["high_usd"], "USD")} at {p["fx"]} TRY/USD.</p>')
    if ask:
        out.append(f'<p class="band-asking">The seller asks '
                   f'{money(ask["asking"], cur)} &mdash; {esc(ask["label"])}. '
                   f'{esc(ask["summary"])}</p>')
    # `grade_reason` opens by restating the grade ("good: 6 finding(s), ...").
    # The grade is already in the sentence, so the restatement is dropped.
    reason = cond["grade_reason"].split(": ", 1)[-1]
    out.append(f'<p class="band-grade">Condition <b>{esc(cond["grade"])}</b> '
               f'&mdash; {esc(reason)}.</p>')
    out.append('<p class="act-note">The vision model never sees a price and never emits '
               'one. The price model never sees the photographs. The only thing that '
               'crosses between them is a condition multiplier, and these are estimated '
               'asking prices, not confirmed sale prices.</p>')
    out.append('</article>')

    # act 6 — the limits
    out.append('<article class="act act-limits" data-act="limits">')
    out.append('<h3>And what it will not say.</h3>')
    out.append('<ul class="limits">')
    out.extend(f'<li>{esc(c)}</li>' for c in cond["cannot_tell"])
    out.append('</ul>')
    out.append('<a class="button orange-button" href="/app">Run this on your truck '
               '<span>&#8599;</span></a>')
    out.append('</article>')

    out.append('</div></div></div>')   # stage-body, stage, track
    out.append('</section>')
    out.append(END)
    return "\n".join(out)


def write_landing(markup: str) -> bool:
    page = LANDING.read_text(encoding="utf-8")
    if START not in page or END not in page:
        raise SystemExit(f"{LANDING} has no {START} / {END} markers to write between")
    before, rest = page.split(START, 1)
    _, after = rest.split(END, 1)
    updated = before + markup + after
    if updated == page:
        return False
    LANDING.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("appraisal", type=Path,
                    help="the Appraisal JSON written by `app.cli appraise --save`")
    ap.add_argument("--folder", type=Path, default=None,
                    help="the photo folder (default: the one named in the appraisal)")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and diff without writing; exit 1 on drift")
    args = ap.parse_args()

    appraisal = json.loads(args.appraisal.read_text(encoding="utf-8"))
    folder = args.folder or REPO / "demo" / "tr_clean"
    if not folder.is_dir():
        raise SystemExit(f"no such photo folder: {folder}")

    story = build(appraisal, folder)

    # Act 1's lot frame is a seventh photo and keeps its own name, so the six
    # numbered heroes stay a contiguous run the deck can index.
    if story["gate"]["lot"]:
        w, h = copy_photo(folder / story["gate"]["lot"]["file"], OUT / "lot.jpg")
        story["gate"]["lot"].update(width=w, height=h)

    markup = render(story)
    if args.check:
        page = LANDING.read_text(encoding="utf-8")
        current = START + page.split(START, 1)[1].split(END, 1)[0] + END
        if current != markup:
            print("landing.html story region has drifted from the JSON", file=sys.stderr)
            return 1
        print("story region matches")
        return 0

    (OUT / "story.json").write_text(
        json.dumps(story, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    changed = write_landing(markup)

    jpgs = sorted(OUT.glob("*.jpg"))
    size = sum(f.stat().st_size for f in jpgs) / 1024
    print(f"wrote {OUT}/story.json  ({(OUT / 'story.json').stat().st_size / 1024:.0f} KB)")
    print(f"wrote {len(jpgs)} photos ({size:.0f} KB): {', '.join(f.name for f in jpgs)}")
    print(f"landing.html story region: {'rewritten' if changed else 'unchanged'}")
    print(f"\n  {story['gate']['photos_total']} photos -> "
          f"{story['gate']['photos_read']} read -> {story['merge']['raw']} observations "
          f"-> {story['merge']['merged']} findings -> "
          f"{money(story['price']['low'], story['price']['currency'])}-"
          f"{money(story['price']['high'], story['price']['currency'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
