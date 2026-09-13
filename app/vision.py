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
# Several templates per class, max-pooled at inference. One template per class
# was measured at 83% on the only question that matters here - is the whole
# tractor in this frame, or one part of it - and, worse, 19% of genuine
# component close-ups came back as an exterior view. That is not a cosmetic
# mislabel: a frame tagged `exterior_front` is handed the exterior question
# bank ("grille, bumper and valance... headlamp clouding... windscreen chips"),
# counts toward whole-vehicle coverage, and is exempt from every part-view rule
# in app/subject.py. A Ford dashboard at `exterior_front` 0.38 and a stripped
# engine bay at `exterior_rear` 0.39 are both in the corpus.
#
# Measured against 90 hand-labelled frames (60 dev, 30 held out), the ensemble
# takes whole-vs-part from 83.3%/86.7% to 91.7%/96.7% and components miscalled
# whole from 18.9%/15.8% to 10.8%/0.0%.
#
# MAX-pooled, not mean: the templates for one class are alternative phrasings of
# the same thing, so the best-matching one is the evidence and averaging it
# against five worse phrasings only dilutes it. A margin requirement on top
# ("whole vehicle has to be earned") scored better on dev and worse on held-out,
# so it was dropped - it was fitting 60 frames.
VIEW_PROMPTS = {
    "exterior_front": [
        "a photo of the front of a semi truck tractor unit",
        "the front grille and bumper of a whole lorry seen head on",
        "a truck photographed from directly in front, the whole cab visible"],
    "exterior_front_34": [
        "a three-quarter front view photo of a semi truck",
        "a whole tractor unit seen from the front corner",
        "an angled view of a complete truck showing front and side"],
    "exterior_side": [
        "a photo of the side profile of a semi truck",
        "the whole length of a lorry seen from the side",
        "a complete tractor unit photographed side on"],
    "exterior_rear": [
        "a photo of the back of a semi truck tractor unit",
        "a whole truck seen from behind showing the rear wheels",
        "the rear three-quarter view of a complete lorry"],
    "interior_cab": [
        "a photo inside a truck cab showing seats and sleeper bunk",
        "the interior of a lorry cabin, upholstery and trim",
        "inside a truck sleeper compartment",
        "a view from the driver's seat of a truck"],
    "dashboard_odometer": [
        "a close-up photo of a truck dashboard, gauges and odometer",
        "an instrument cluster with speedometer and warning lights",
        "the dashboard controls and switches of a lorry",
        "a steering wheel and dash panel photographed from inside"],
    "tire_wheel": [
        "a close-up photo of a truck tire and wheel rim",
        "a heavy vehicle tyre tread photographed close up",
        "a truck wheel hub and rim in close-up"],
    "engine_bay": [
        "a close-up photo of a diesel truck engine",
        "an engine bay with hoses, belts and turbocharger",
        "the engine compartment of a lorry, tipped cab"],
    "chassis_undercarriage": [
        "a photo of the chassis, frame rails or undercarriage of a truck",
        "the underside of a lorry showing axles and air tanks",
        "truck frame rails and crossmembers photographed from below"],
    "fifth_wheel": [
        "a close-up photo of a truck fifth wheel coupling plate",
        "the coupling plate and jaws behind a tractor cab",
        "a fifth wheel hitch photographed close up"],
    "damage_detail": [
        "a close-up photo of damage, a dent, rust or a scratch on a vehicle panel",
        "a detail photograph of corrosion or a crack on bodywork",
        "a small area of a vehicle panel showing a defect"],
}

# Zero-shot body type. Separate from views: a dashboard is not a body-type
# signal, and pooling here is only meaningful on whole-vehicle frames.
BODY_PROMPTS = {
    "tractor_unit": [
        "a semi truck tractor unit with a fifth wheel coupling and no cargo box",
        "a cab-over lorry tractor, chassis ending at the fifth wheel",
        "a truck-tractor designed to pull a semi trailer"],
    "rigid": [
        "a rigid box truck with a cargo body behind the cab",
        "a straight truck with an integrated box van body",
        "a dump truck or tipper with a cargo bed, not a tractor unit"],
    "other": [
        "a passenger car, van or motorcycle, not a heavy truck"],
}
BODY_LABELS = list(BODY_PROMPTS)

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

VIEW_LABELS = list(VIEW_PROMPTS)


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
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

        def bank(prompts):
            toks = self.tokenizer([p for _, p in prompts]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(toks)
            return feats / feats.norm(dim=-1, keepdim=True)

        self.view_bank, self.view_owner = self.grouped_bank(VIEW_PROMPTS)
        self.body_bank, self.body_owner = self.grouped_bank(BODY_PROMPTS)
        self.content_bank = bank(CONTENT_PROMPTS)
        self.logit_scale = self.model.logit_scale.exp().item()

    def grouped_bank(self, banks: dict):
        """One row block per class, and the index that says where each ends."""
        flat, owner = [], []
        for i, prompts in enumerate(banks.values()):
            flat.extend(prompts)
            owner.extend([i] * len(prompts))
        toks = self.tokenizer(flat).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(toks)
        return (feats / feats.norm(dim=-1, keepdim=True),
                torch.tensor(owner, device=self.device))

    def pooled_softmax(self, feats, text_bank, owner, n_classes: int):
        """Max over each class's templates, THEN softmax over the classes.

        The one scoring path. `tag` calls it for the view and body banks, and
        a measurement that wants to try a *candidate* taxonomy calls it with a
        bank from `grouped_bank` rather than re-deriving the arithmetic - the
        logit_scale multiply and the max-not-mean pooling are the two things a
        second implementation would get subtly wrong.
        """
        sims = feats @ text_bank.T                       # (n, templates)
        per_class = torch.full((sims.shape[0], n_classes), float("-inf"),
                               device=sims.device)
        per_class = per_class.index_reduce(1, owner, sims, "amax",
                                           include_self=False)
        return (self.logit_scale * per_class).softmax(dim=-1)

    def score_bank(self, feats, banks: dict) -> np.ndarray:
        """Probabilities over an arbitrary prompt-bank dict, same path as `tag`.

        `feats` is already-L2-normalised image features - a torch tensor, or
        the numpy rows out of `embed` / the corpus embedding cache.
        """
        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(np.asarray(feats), device=self.device)
        text_bank, owner = self.grouped_bank(banks)
        with torch.no_grad():
            return self.pooled_softmax(feats, text_bank, owner,
                                       len(banks)).cpu().numpy()

    def _features(self, pil_images: list):
        batch = torch.stack([self.preprocess(im.convert("RGB"))
                             for im in pil_images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            return feats / feats.norm(dim=-1, keepdim=True)

    def embed(self, pil_images: list) -> np.ndarray:
        """L2-normalised image embeddings, no text banks.

        One implementation, three callers: `tag` below, the corpus cache in
        scripts/cache_embeddings.py, and app.subject, which embeds every
        candidate crop in a photo set to work out which vehicle recurs.
        """
        if not pil_images:
            return np.zeros((0, 512), dtype=np.float32)
        return self._features(pil_images).cpu().numpy().astype(np.float32)

    def tag(self, pil_images: list) -> list[dict]:
        if not pil_images:
            return []
        feats = self._features(pil_images)
        with torch.no_grad():
            # Max over each class's templates, THEN softmax over the classes.
            # The logit_scale multiply stays where it was: raw cosines sit at
            # 0.15-0.35 and soften to near-uniform without it.
            view = self.pooled_softmax(feats, self.view_bank, self.view_owner,
                                       len(VIEW_LABELS)).cpu().numpy()
            body = self.pooled_softmax(feats, self.body_bank, self.body_owner,
                                       len(BODY_LABELS)).cpu().numpy()
            content = (self.logit_scale * feats @ self.content_bank.T).softmax(dim=-1).cpu().numpy()
        emb = feats.cpu().numpy().astype(np.float32)
        out = []
        for v, b, c, e in zip(view, body, content, emb):
            vi, bi, ci = int(np.argmax(v)), int(np.argmax(b)), int(np.argmax(c))
            out.append({
                "view": VIEW_LABELS[vi],
                "view_conf": float(v[vi]),
                "body": BODY_LABELS[bi],
                "body_conf": float(b[bi]),
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
