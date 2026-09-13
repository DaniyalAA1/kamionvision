"""Lazily-loaded local vision models shared by the gate.

Two models, two jobs, deliberately not one:

  YOLOv8n (COCO, zero fine-tuning)  "is there a truck-shaped object here"
  CLIP ViT-B-32-quickgelu           "which canonical view is this photo"

Both are loaded once per process and pinned to MPS when it is available, so
the web server pays the ~2 s warm-up on boot rather than on the first
appraisal a judge watches.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch

from .config import MODELS

os.environ.setdefault("YOLO_VERBOSE", "false")

_LOCK = threading.Lock()
_YOLO = None
_CLIP = None

COCO_VEHICLES = {"truck", "bus", "car", "motorcycle", "bicycle", "train", "boat", "airplane"}
# Detections that mean "this is confidently not the truck you are selling".
COCO_DISQUALIFYING = {"motorcycle", "bicycle", "boat", "airplane", "train", "bird",
                      "cat", "dog", "horse", "pizza", "laptop", "cell phone", "tv"}

# Verbatim from scripts/clean_dataset.py so a gate decision and a corpus row
# mean the same thing. Divergence here would silently retrain the thresholds.
VIEW_PROMPTS = [
    ("exterior_front",    "a photo of the front of a semi truck tractor unit"),
    ("exterior_front_34", "a three-quarter front view photo of a semi truck"),
    ("exterior_side",     "a photo of the side profile of a semi truck"),
    ("exterior_rear",     "a photo of the back of a semi truck tractor unit"),
    ("interior_cab",      "a photo inside a truck cab showing seats and sleeper bunk"),
    ("dashboard_odometer", "a close-up photo of a truck dashboard, gauges and odometer"),
    ("tire_wheel",        "a close-up photo of a truck tire and wheel rim"),
    ("engine_bay",        "a close-up photo of a diesel truck engine"),
    ("chassis_undercarriage", "a photo of the chassis, frame rails or undercarriage of a truck"),
    ("fifth_wheel",       "a close-up photo of a truck fifth wheel coupling plate"),
    ("damage_detail",     "a close-up photo of damage, a dent, rust or a scratch on a vehicle panel"),
]

CONTENT_PROMPTS = [
    ("keep", "a photograph of a truck or lorry"),
    ("keep", "a close-up photograph of part of a vehicle"),
    ("keep", "a photograph of the inside of a vehicle cab"),
    ("reject_document", "a scanned paper document, invoice or window sticker with printed text"),
    ("reject_screen", "a photograph of a computer screen, laptop or diagnostic tablet display"),
    ("reject_graphic", "a company logo, watermark, banner or graphic design"),
    ("reject_placeholder", "a blank grey placeholder image meaning no photo available"),
    ("reject_scene", "a photograph of a building, office, person or landscape with no vehicle"),
]

VIEW_LABELS = [t for t, _ in VIEW_PROMPTS]


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def yolo():
    """COCO detector. Weights are vendored in models/ so demo day needs no network."""
    global _YOLO
    with _LOCK:
        if _YOLO is None:
            from ultralytics import YOLO
            weights = MODELS / "yolov8n.pt"
            _YOLO = YOLO(str(weights) if weights.exists() else "yolov8n.pt")
        return _YOLO


class ClipTagger:
    """Zero-shot view + content tagging.

    The `logit_scale` multiply is load-bearing, not decoration: raw CLIP
    cosine similarities live in ~0.15-0.35, and softmaxing them gives a
    near-uniform distribution whose argmax is noise. The trained scale
    (~100) is what makes the decision decisive.
    """

    def __init__(self) -> None:
        import open_clip
        self.device = device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai", device=self.device)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

        def bank(prompts):
            toks = tokenizer([p for _, p in prompts]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(toks)
            return feats / feats.norm(dim=-1, keepdim=True)

        self.view_bank = bank(VIEW_PROMPTS)
        self.content_bank = bank(CONTENT_PROMPTS)
        self.logit_scale = self.model.logit_scale.exp().item()

    def tag(self, pil_images: list) -> list[dict]:
        if not pil_images:
            return []
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in pil_images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            view = (self.logit_scale * feats @ self.view_bank.T).softmax(dim=-1).cpu().numpy()
            content = (self.logit_scale * feats @ self.content_bank.T).softmax(dim=-1).cpu().numpy()
        emb = feats.cpu().numpy().astype(np.float32)
        out = []
        for v, c, e in zip(view, content, emb):
            vi, ci = int(np.argmax(v)), int(np.argmax(c))
            out.append({
                "view": VIEW_PROMPTS[vi][0],
                "view_conf": float(v[vi]),
                "content": CONTENT_PROMPTS[ci][0],
                "content_conf": float(c[ci]),
                # The embedding these tags were derived from. Already computed;
                # app.perception trains on it, so returning it costs nothing and
                # saves a second CLIP pass over the same pixels.
                "embedding": e,
                # keep-mass, not top-1: the three "keep" prompts split the
                # probability of a genuine vehicle photo between them, so a
                # legitimate cab interior can top out at 0.4 on any one of them.
                "keep_mass": float(sum(c[i] for i, (t, _) in enumerate(CONTENT_PROMPTS) if t == "keep")),
            })
        return out


def clip() -> ClipTagger:
    global _CLIP
    with _LOCK:
        if _CLIP is None:
            _CLIP = ClipTagger()
        return _CLIP


def warm() -> dict:
    """Pre-load both models. Called on server boot; returns a status dict."""
    import time
    out = {}
    t = time.time()
    yolo()
    out["yolo_s"] = round(time.time() - t, 2)
    t = time.time()
    clip()
    out["clip_s"] = round(time.time() - t, 2)
    out["device"] = device()
    return out
