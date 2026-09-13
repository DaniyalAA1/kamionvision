"""Inference for the trained perception heads.

Three small linear heads sit between the deterministic gate and the VLM. They
exist because the gate can only measure a photo and the VLM can only describe
one; neither can say "this photo has been degraded in a way that makes the
claim you are about to read unreliable". These can, because they were trained
on 3,729 exactly-paired clean/degraded images where the degradation is known.

    degradation  is this photo corrupted, how badly, and in what way
    view         which of the 11 canonical views this is, robustly
    identity     which brand the vehicle is, as a cross-check on the VLM

All three are linear models over the CLIP embedding `vision.ClipTagger.tag`
already computes, so scoring a set costs a few matrix multiplies and no extra
model. Parameters live in `models/perception.json`; refit with

    .venv/bin/python -m app.perception.train
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass

import numpy as np

from ..config import MODELS
from ..schema import PerceptionReport, PhotoPerception

PERCEPTION_MODEL = MODELS / "perception.json"

# Scalar capture features, in artifact order. The single source of truth for
# both the training matrix and the inference vector - a test asserts the two
# agree, because a silent reordering here would be invisible and wrong.
SCALARS = ["log_blur", "brightness", "contrast_rms", "dark_clipped_frac",
           "bright_clipped_frac", "colourfulness_100", "capture_quality",
           "log_megapixels"]

# Fallback only. The real cutoff is calibrated on the corpus and carried in
# the artifact as degradation.fine_detail_severity - a quantile of predicted
# severity on genuinely clean photos, so the number it fixes is a false-flag
# rate. This constant is what an artifact predating that calibration gets.
FINE_DETAIL_SEVERITY = 0.55
ATOM_THRESHOLD = 0.60

_MODEL: "PerceptionModel | None" = None
_LOCK = threading.Lock()


def scalar_row(*, blur_laplacian_var: float, brightness: float, contrast_rms: float,
               dark_clipped_frac: float, bright_clipped_frac: float,
               colourfulness: float, capture_quality: float,
               width: float, height: float) -> list[float]:
    """The scalar half of the feature vector, in SCALARS order."""
    return [
        math.log1p(max(0.0, float(blur_laplacian_var))),
        float(brightness),
        float(contrast_rms),
        float(dark_clipped_frac),
        float(bright_clipped_frac),
        float(colourfulness) / 100.0,
        float(capture_quality),
        math.log(max(1e-6, float(width) * float(height) / 1e6)),
    ]


@dataclass
class PerceptionModel:
    scalars: list[str]
    degradation: dict
    view: dict
    identity: dict
    meta: dict

    @classmethod
    def load(cls, path=PERCEPTION_MODEL) -> "PerceptionModel":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run: .venv/bin/python -m app.perception.train")
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(scalars=d["scalars"], degradation=d["degradation"],
                   view=d["view"], identity=d["identity"], meta=d["meta"])


def model() -> PerceptionModel:
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            _MODEL = PerceptionModel.load()
        return _MODEL


def available() -> bool:
    return PERCEPTION_MODEL.exists()


def _apply(head: dict, X: np.ndarray) -> np.ndarray:
    """Standardise with the head's own stats, then a linear map."""
    mean = np.asarray(head["mean"], dtype=float)
    scale = np.asarray(head["scale"], dtype=float)
    coef = np.asarray(head["coef"], dtype=float)
    intercept = np.asarray(head["intercept"], dtype=float)
    return ((X - mean) / scale) @ coef.T + intercept


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def photo_matrix(checks: list) -> np.ndarray:
    """PhotoChecks -> the (n, 512 + len(SCALARS)) matrix the heads expect."""
    rows = []
    for c in checks:
        emb = c._embedding
        if emb is None:
            emb = np.zeros(512, dtype=np.float32)
        rows.append(np.concatenate([np.asarray(emb, dtype=float),
                                    scalar_row(blur_laplacian_var=c.blur_laplacian_var,
                                               brightness=c.brightness,
                                               contrast_rms=c.contrast_rms,
                                               dark_clipped_frac=c.dark_clipped_frac,
                                               bright_clipped_frac=c.bright_clipped_frac,
                                               colourfulness=c.colourfulness,
                                               capture_quality=c.capture_quality,
                                               width=c.width or 1, height=c.height or 1)]))
    return np.stack(rows) if rows else np.zeros((0, 512 + len(SCALARS)))


def run(checks: list) -> PerceptionReport:
    """Score a photo set. Returns an empty report if the artifact is absent."""
    import time
    t0 = time.time()
    report = PerceptionReport()
    scored = [c for c in checks if c._embedding is not None]
    if not scored or not available():
        report.elapsed_s = round(time.time() - t0, 3)
        report.model_card = {"trained": available()}
        return report

    m = model()
    X = photo_matrix(scored)
    emb_only = X[:, :512]

    deg_p = _sigmoid(_apply(m.degradation, X))
    sev = np.clip(_apply(m.degradation["severity"], X).ravel(), 0.0, 1.0)
    view_p = _softmax(_apply(m.view, emb_only))
    atoms = m.degradation["atoms"]
    views = m.view["classes"]
    cutoff = float(m.degradation.get("fine_detail_severity", FINE_DETAIL_SEVERITY))

    for i, c in enumerate(scored):
        vi = int(np.argmax(view_p[i]))
        report.photos.append(PhotoPerception(
            photo_id=c.photo_id,
            degraded_prob=round(float(deg_p[i].max()), 3),
            severity=round(float(sev[i]), 3),
            degradations=[atoms[j] for j in np.argsort(-deg_p[i])[:3]
                          if deg_p[i][j] >= ATOM_THRESHOLD],
            view=views[vi],
            view_conf=round(float(view_p[i][vi]), 3),
            fine_detail_ok=bool(sev[i] < cutoff),
        ))

    # Identity is a property of the vehicle, not of one frame: pool the
    # embeddings the same way the head was trained.
    pooled = emb_only.mean(axis=0)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-9)
    ident = _softmax(_apply(m.identity, pooled[None, :]))[0]
    bi = int(np.argmax(ident))
    report.brand = m.identity["classes"][bi]
    report.brand_conf = round(float(ident[bi]), 3)
    report.model_card = {k: m.meta.get(k) for k in
                         ("fitted_at", "n_images", "n_vehicles", "view_agreement_head",
                          "view_agreement_zeroshot", "degradation_auc", "identity_accuracy",
                          "identity_majority")}
    report.elapsed_s = round(time.time() - t0, 3)
    return report
