"""Offline chassis-plate VIN reader and deterministic VIN checks."""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

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


def model_year(vin: str) -> int | None:
    """Decode position 10 using position 7 to select the VIN's 30-year cycle."""
    value = normalise(vin)
    if len(value) != 17 or value[9] not in _YEAR_CODES:
        return None
    offset = _YEAR_CODES.index(value[9])
    if value[6].isalpha():
        return 2010 + offset
    if value[6].isdigit():
        return 1980 + offset
    return None


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
