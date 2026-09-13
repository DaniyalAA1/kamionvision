"""Offline chassis-plate VIN reader and deterministic VIN checks.

Two things live here. The reader is OCR over a photograph of a plate. The rest
is arithmetic on the string it produced - a check digit, a model-year code, and
the world manufacturer identifier that the first three characters are.

The WMI half is the one worth explaining. `read` has returned `wmi=vin[:3]`
since it was written and nothing consumed it, which was a waste of the only
deterministic, standards-backed identity signal in the whole pipeline. It
matters most in the case nothing else covers: `reconcile._check_identity` may
only dispute brands in the trained head's own class list, and the corpus has no
Scania, no DAF and no Volvo. A judge arriving with an unseen brand is this
system's measured 1.85x failure mode, and a stamped chassis plate is the
cheapest second opinion available on it.

`data/reference/wmi.json` is the table, hand-curated and cited row by row in
the same posture as `new_prices_tr.json`. Absence from it means UNKNOWN, never
CONFLICT - the lookup returns None and the rule above it stays silent, because
three characters nobody has verified must never be allowed to contradict a
badge the vision model can plainly read.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import REPO

WMI_TABLE = REPO / "data" / "reference" / "wmi.json"

_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
_VALUES = {
    **{str(n): n for n in range(10)},
    **dict(zip("ABCDEFGH", range(1, 9))),
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5,
    "P": 7, "R": 9, "S": 2, "T": 3, "U": 4,
    "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}
_YEAR_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"
_OCR = None
_LOCK = threading.Lock()

# The WMI table gets its own lock rather than sharing the OCR one. Same shape
# as the `_OCR` singleton below - read once, cached, guarded - but the OCR lock
# is held for the several seconds it takes RapidOCR to load its ONNX graphs,
# and a brand lookup has no business queueing behind that.
_WMI: dict | None = None
_WMI_LOCK = threading.Lock()


@dataclass
class VinRead:
    vin: str | None = None
    year: int | None = None
    wmi: str | None = None
    check_ok: bool | None = None
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def normalise(text: str) -> str:
    """Normalise the three characters forbidden in a VIN but common in OCR."""
    return (text or "").strip().upper().translate(str.maketrans({"I": "1",
                                                                 "O": "0",
                                                                 "Q": "0"}))


def check_digit_ok(vin: str) -> bool:
    """Return whether a 17-character VIN has the standard position-9 digit."""
    value = normalise(vin)
    if len(value) != 17 or any(ch not in _VALUES for ch in value):
        return False
    remainder = sum(_VALUES[ch] * weight
                    for ch, weight in zip(value, _WEIGHTS)) % 11
    expected = "X" if remainder == 10 else str(remainder)
    return value[8] == expected


def north_american(vin: str) -> bool:
    """Whether ISO 3780 puts this VIN in the region that builds to FMVSS 565.

    The first character is the geographic area: 1-5 is North America, 6-7
    Oceania, 8-9 South America, A-H Africa, J-R Asia, S-Z Europe. Only the
    North American block is required to encode a model year in character 10
    and a check digit in character 9; everywhere else those two positions mean
    whatever the manufacturer decided they mean.
    """
    return normalise(vin)[:1] in "12345"


def model_year(vin: str) -> int | None:
    """Decode position 10 using position 7 to select the VIN's 30-year cycle.

    North American VINs only, and that restriction is the whole point. The
    position-10 model-year code is an FMVSS 565 convention, not an ISO 3779
    one, and European heavy trucks do not follow it. Measured: the DAF VINs
    published with Australian recall REC-006624, on trucks built between 2019
    and 2025, decode to 1994 under this scheme. Roughly one European VIN in
    eleven also passes `check_digit_ok` by chance, so a check-digit test alone
    would let ~9% of them through with a confidently wrong year - which then
    reaches the price model, because `pricing/model.py` falls back to
    `vin_year` when nothing else supplies one.
    """
    value = normalise(vin)
    if len(value) != 17 or value[9] not in _YEAR_CODES or not north_american(value):
        return None
    offset = _YEAR_CODES.index(value[9])
    if value[6].isalpha():
        return 2010 + offset
    if value[6].isdigit():
        return 1980 + offset
    return None


# --- the world manufacturer identifier ------------------------------------

def _table() -> dict:
    """The parsed WMI table, cached.

    A missing or unreadable file is an EMPTY table, never an exception. This is
    a soft dependency in exactly the sense the odometer OCR is: without it the
    brand cross-check no-ops and the appraisal runs as it did before the table
    existed. Failing an appraisal because a reference file was not checked out
    would be a worse outcome than not performing one extra cross-check.
    """
    global _WMI
    with _WMI_LOCK:
        if _WMI is None:
            try:
                _WMI = json.loads(WMI_TABLE.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logging.getLogger(__name__).warning(
                    "WMI table unavailable (%s); the brand cross-check is inert", exc)
                _WMI = {"rows": []}
        return _WMI


def _reset_wmi() -> None:
    """Drop the cache. Tests that point WMI_TABLE elsewhere call this."""
    global _WMI
    with _WMI_LOCK:
        _WMI = None


def wmi_brands(wmi: str | None, *, allow_unverified: bool = False) -> list[str]:
    """Every brand that may legitimately be badged on a vehicle with this WMI.

    The primary brand first, then `also_badged`. Two different things end up in
    that list and the caller treats them identically, because the right
    response to both is silence:

      * one plant, several marques - Navistar's WMIs really do carry Chevrolet,
        IC Bus and Caterpillar badges as well as International ones;
      * one brand, several spellings - `pricing.features.normalise_brand` folds
        FORD TRUCKS into FORD but has no rule for RENAULT TRUCKS, and a rule
        that disputed the badge on a Renault Trucks T because of a gap in a
        normaliser would be inventing a conflict out of its own vocabulary.

    Empty list for an unknown WMI, an unverified row, or an absent table. The
    caller cannot tell those three apart and must not need to: all three mean
    "this witness has no opinion", which is not the same as "the badge is wrong".
    """
    key = re.sub(r"[^A-Z0-9]", "", normalise(wmi or ""))
    if len(key) != 3:
        return []
    for row in _table().get("rows", []):
        if str(row.get("wmi", "")).strip().upper() != key:
            continue
        if not row.get("verified") and not allow_unverified:
            return []
        brand = str(row.get("brand") or "").strip()
        if not brand:
            return []
        return [brand] + [str(b).strip() for b in (row.get("also_badged") or [])
                          if str(b).strip()]
    return []


def wmi_brand(wmi: str | None, *, allow_unverified: bool = False) -> str | None:
    """The manufacturer's own brand for a verified WMI, else None."""
    brands = wmi_brands(wmi, allow_unverified=allow_unverified)
    return brands[0] if brands else None


def _engine():
    global _OCR
    with _LOCK:
        if _OCR is None:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        return _OCR


def _tokens(path: str | Path) -> list[tuple[str, float]]:
    result, _ = _engine()(str(path))
    return [(str(text), float(conf)) for _, text, conf in (result or [])]


def _clean(text: str) -> str:
    without_label = re.sub(r"^\s*VIN\s*[:#-]?\s*", "", text or "",
                           flags=re.IGNORECASE)
    return re.sub(r"[^A-Z0-9]", "", normalise(without_label))


def _candidates(tokens: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Build exact-length candidates from one token or adjacent OCR fragments."""
    chunks = [(_clean(text), conf) for text, conf in tokens]
    chunks = [(text, conf) for text, conf in chunks if text]
    found: dict[str, float] = {}
    for start in range(len(chunks)):
        joined = ""
        confidences: list[float] = []
        for text, confidence in chunks[start:start + 4]:
            joined += text
            confidences.append(confidence)
            if len(joined) == 17:
                found[joined] = max(found.get(joined, 0.0), min(confidences))
                break
            if len(joined) > 17:
                break
    return sorted(found.items(), key=lambda item: (not check_digit_ok(item[0]), -item[1]))


def read(path: str | Path) -> VinRead:
    """Read a VIN from a chassis plate, abstaining unless OCR yields 17 chars."""
    tokens = _tokens(path)
    candidates = _candidates(tokens)
    if not candidates:
        return VinRead(reason=f"no cleaned 17-character VIN in {len(tokens)} OCR token(s)")

    vin, confidence = candidates[0]
    valid = check_digit_ok(vin)
    reason = f"read 17-character VIN at OCR confidence {confidence:.2f}"
    if not valid:
        reason += "; check digit failed, so verify it against the chassis plate"
        logging.getLogger(__name__).warning("VIN check digit failed for %s", vin)
    return VinRead(vin=vin, year=model_year(vin), wmi=vin[:3],
                   check_ok=valid, confidence=round(confidence, 3), reason=reason)
