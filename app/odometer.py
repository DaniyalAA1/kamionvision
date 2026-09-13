"""Pretrained-OCR odometer reader - a specialist that grounds the mileage.

This is the "layer a pretrained model on top of the vision" idea, applied where
it actually moves the price: kilometres, not cosmetics. The condition adjustment
is capped at one residual sigma by design, so a damage model can only nudge the
number - but mileage is a first-class term in the hedonic model, so an
independent, checkable odometer read is worth more than any condition signal.

What this is and is not:
  * It is a pretrained scene-text model (RapidOCR / PP-OCR, bundled ONNX, fully
    offline) plus a domain selector. No training, no labels, no network.
  * It is NOT trusted blindly. It reads the digits off the cluster; the pipeline
    still prefers a seller's stated, document-backed figure. Its real job is to
    give the odometer-vs-stated fraud check in pricing a second, independent
    witness, and to supply a km when the seller left it blank.

The selector is the entire point. A truck cluster shows a dozen numbers - speed,
coolant temp, a clock, trip meters, range-to-empty, gauge ticks. The odometer is
the one that is a plain 4-7 digit integer written immediately before a bare "km"
(not "km/h", not "km to E", not a decimal trip figure). That single rule picks it
out of every distractor on the F-MAX and Actros clusters in the corpus.

Standalone by design: it imports nothing from the rest of app/, so it can be
wired into gate/evidence/pricing later without a circular import, and demoed on
its own with `python -m app.odometer <image>...`.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Plausible total mileage for a heavy tractor, in km. Below the floor is a trip
# meter or a gauge tick; above the ceiling is an OCR run-together of two numbers.
KM_FLOOR = 1_000
KM_CEILING = 2_000_000

# A token like "242780km" or "0499150km": optional leading zero, 4-7 digits, then
# "km" with nothing after it. The trailing-end anchor is what rejects "km/h",
# "kmtoE" and "km/l"; requiring pure digits rejects the decimal trip figures
# ("976.6km", "32.7l/100km").
_JOINED = re.compile(r"^0?(\d{4,7})km$")
_BARE_INT = re.compile(r"^0?(\d{4,7})$")
_LONE_KM = re.compile(r"^km$")

_LOCK = threading.Lock()
_OCR = None


def _engine():
    """Lazy RapidOCR singleton. ~0.3 s import, models bundled in the wheel."""
    global _OCR
    with _LOCK:
        if _OCR is None:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        return _OCR


@dataclass
class Candidate:
    km: int
    text: str
    ocr_conf: float
    box: list[float]          # xyxy in pixels
    rule: str                 # "joined" | "adjacent"


@dataclass
class OdometerRead:
    """Result of reading one dashboard photo. `km is None` means abstain."""
    km: int | None = None
    confidence: float = 0.0        # 0-1, the selected token's OCR confidence
    text: str = ""                 # the raw token the reading came from
    box: list[float] = field(default_factory=list)
    rule: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    n_text_tokens: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _norm(text: str) -> str:
    """Collapse whitespace and unicode lookalikes so the regexes see plain ASCII."""
    t = (text or "").strip().lower().replace(" ", "")
    # OCR sometimes emits fullwidth / OCR-B digits; map the common ones.
    trans = str.maketrans("oOlLiISB", "00111158")
    return t.translate(trans)


def _tokens(image: str | Path):
    """RapidOCR -> [(text, confidence, xyxy_box)]. Empty on an unreadable frame."""
    result, _ = _engine()(str(image))
    out = []
    for box, text, conf in (result or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append((text, float(conf), [min(xs), min(ys), max(xs), max(ys)]))
    return out


def read(image: str | Path) -> OdometerRead:
    """Read the odometer from a single dashboard photo.

    Returns an `OdometerRead`; `.km is None` when nothing plausible was found,
    which is a deliberate abstention, not a zero. A wrong six-figure mileage is
    far more damaging than no reading, because km is a primary price term.
    """
    tokens = _tokens(image)
    res = OdometerRead(n_text_tokens=len(tokens))
    if not tokens:
        res.reason = "no text found in the frame"
        return res

    cands: list[Candidate] = []

    # Rule 1 - joined token "NNNNNNkm". This is how the F-MAX and Actros clusters
    # in the corpus render the odometer, and it is unambiguous.
    for text, conf, box in tokens:
        m = _JOINED.match(_norm(text))
        if not m:
            continue
        km = int(m.group(1))
        if KM_FLOOR <= km <= KM_CEILING:
            cands.append(Candidate(km=km, text=text, ocr_conf=round(conf, 3),
                                   box=[round(v, 1) for v in box], rule="joined"))

    # Rule 2 - a bare integer token sitting just left of a lone "km" token on the
    # same line. Covers phone photos where OCR splits the digits and the unit.
    if not cands:
        km_boxes = [box for text, _, box in tokens if _LONE_KM.match(_norm(text))]
        for text, conf, box in tokens:
            m = _BARE_INT.match(_norm(text))
            if not m:
                continue
            km = int(m.group(1))
            if not (KM_FLOOR <= km <= KM_CEILING):
                continue
            cy = (box[1] + box[3]) / 2
            for kb in km_boxes:
                same_line = abs((kb[1] + kb[3]) / 2 - cy) < (box[3] - box[1])
                to_the_right = 0 <= kb[0] - box[2] < 3 * (box[3] - box[1])
                if same_line and to_the_right:
                    cands.append(Candidate(km=km, text=f"{text} km", ocr_conf=round(conf, 3),
                                           box=[round(v, 1) for v in box], rule="adjacent"))
                    break

    res.candidates = sorted(cands, key=lambda c: -c.ocr_conf)
    if not res.candidates:
        res.reason = ("no token matched an odometer reading "
                      "(a plain 4-7 digit integer followed by 'km')")
        return res

    best = res.candidates[0]
    res.km, res.confidence, res.text = best.km, best.ocr_conf, best.text
    res.box, res.rule = best.box, best.rule
    res.reason = (f"read {best.km:,} km from a '{best.rule}' token at OCR confidence "
                  f"{best.ocr_conf:.2f}"
                  + (f"; {len(res.candidates)} candidates, highest kept"
                     if len(res.candidates) > 1 else ""))
    return res


def read_best(images: list[str | Path]) -> OdometerRead:
    """Read several dashboard photos, return the highest-confidence agreement.

    A seller sends more than one dashboard shot; the cluster reading should be
    stable across them. Picks the highest-confidence read, and records in
    `reason` whether the other frames agreed - a disagreement is itself worth
    surfacing rather than hiding behind an average.
    """
    reads = [(p, read(p)) for p in images]
    hits = [(p, r) for p, r in reads if r.km is not None]
    if not hits:
        out = OdometerRead(reason=f"no odometer read from {len(images)} photo(s)")
        return out
    hits.sort(key=lambda pr: -pr[1].confidence)
    best_path, best = hits[0]
    agree = [r.km for _, r in hits if abs(r.km - best.km) <= max(500, 0.01 * best.km)]
    best.reason = (f"{best.km:,} km from {Path(best_path).name} "
                   f"(conf {best.confidence:.2f}); "
                   f"{len(agree)}/{len(hits)} legible dashboard photos agree")
    return best


def _demo(argv: list[str]) -> None:
    import json
    import sys
    paths = argv or []
    if not paths:
        print("usage: python -m app.odometer <dashboard_image>...", file=sys.stderr)
        raise SystemExit(2)
    for p in paths:
        r = read(p)
        verdict = f"{r.km:,} km" if r.km is not None else "ABSTAIN"
        print(f"{Path(p).name:32} -> {verdict:>14}   {r.reason}")
    if len(paths) > 1:
        best = read_best(paths)
        print("-" * 60)
        print("read_best:", best.reason)


if __name__ == "__main__":
    import sys
    _demo(sys.argv[1:])
