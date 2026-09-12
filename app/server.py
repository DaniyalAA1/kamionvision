"""FastAPI app behind the demo screen.

Upload and appraisal are two steps on purpose. The gate finishes in a couple
of seconds and the vision call takes half a minute, so the browser streams
the run over SSE and paints the gate verdict - which photos were kept, which
were dropped and why, whether a truck was actually found - long before the
price exists. On a four-minute demo clock that turns a 40-second wait into a
sequence the room can watch.
"""
from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from . import pipeline, report
from .config import IMAGES, REPO, WEB

app = FastAPI(title="KamionVision", docs_url="/api/docs")

SESSIONS = Path(tempfile.gettempdir()) / "kamionvision-sessions"
SESSIONS.mkdir(parents=True, exist_ok=True)
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_PHOTOS = 60
MAX_BYTES = 25 * 1024 * 1024


@app.on_event("startup")
def warm() -> None:
    """Pay the model load on boot, not on the first appraisal a judge watches."""
    from . import vision
    threading.Thread(target=vision.warm, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"))


@app.get("/static/{name}")
def static(name: str) -> FileResponse:
    path = (WEB / name).resolve()
    if path.parent != WEB.resolve() or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/health")
def health() -> dict:
    from . import vlm
    from .config import MAX_EVIDENCE_PHOTOS
    from .pricing import load_model

    out: dict = {"backends": [
        {"name": s.name, "ready": s.ready, "detail": s.detail,
         "account_blocked": s.account_blocked, "model": s.model}
        for s in vlm.probe_all()]}
    out["evidence_photos"] = MAX_EVIDENCE_PHOTOS
    try:
        m = load_model()
        out["price_model"] = {
            "training_set": m.meta.get("training_set"),
            "n_listings": m.meta.get("n_listings"),
            "n_groups": m.meta.get("n_groups"),
            "r2": m.calibration.get("r2_oof"),
            "median_ape": m.calibration.get("median_ape_oof"),
            "coverage_80": m.calibration.get("coverage_0.8"),
            "coverage_n": m.calibration.get("coverage_n"),
            "unknown_brand_widening": m.widening.get("unknown_brand"),
            "fitted_at": m.meta.get("fitted_at"),
        }
    except Exception as exc:
        out["price_model"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


@app.get("/api/samples")
def samples() -> list[dict]:
    from .demo import resolved_cases
    out = []
    for case in resolved_cases():
        folder = REPO / case["folder"]
        out.append({"id": case["id"], "title": case["title"], "blurb": case["blurb"],
                    "expect": case["expect"], "declared": case.get("declared") or {},
                    "available": folder.exists(),
                    "n_photos": len(pipeline.collect_photos(folder)) if folder.exists() else 0})
    return out


def _new_session(paths: list[Path]) -> dict:
    sid = uuid.uuid4().hex[:12]
    dest = SESSIONS / sid
    dest.mkdir(parents=True, exist_ok=True)
    kept = []
    for i, src in enumerate(paths[:MAX_PHOTOS]):
        target = dest / f"{i:03d}{src.suffix.lower()}"
        shutil.copyfile(src, target)
        kept.append(target)
    return {"session": sid, "photos": [{"name": p.name,
                                        "url": f"/api/photo/{sid}/{p.name}"} for p in kept]}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    sid = uuid.uuid4().hex[:12]
    dest = SESSIONS / sid
    dest.mkdir(parents=True, exist_ok=True)
    kept, skipped = [], []
    for i, f in enumerate(files[:MAX_PHOTOS]):
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in ALLOWED:
            skipped.append({"name": f.filename, "why": f"unsupported type {suffix or '?'}"})
            continue
        raw = await f.read()
        if len(raw) > MAX_BYTES:
            skipped.append({"name": f.filename, "why": "larger than 25 MB"})
            continue
        target = dest / f"{i:03d}{suffix}"
        target.write_bytes(raw)
        kept.append({"name": target.name, "original": f.filename,
                     "url": f"/api/photo/{sid}/{target.name}"})
    if not kept:
        raise HTTPException(400, "No usable image files in that upload.")
    return {"session": sid, "photos": kept, "skipped": skipped}


@app.post("/api/upload-sample")
def upload_sample(case: str = Form(...)) -> dict:
    from .demo import resolved_cases
    match = next((c for c in resolved_cases() if c["id"] == case), None)
    if not match:
        raise HTTPException(404, f"unknown sample {case!r}")
    folder = REPO / match["folder"]
    paths = pipeline.collect_photos(folder)
    if not paths:
        raise HTTPException(404, f"{match['folder']} has no photos - run "
                                 f"'python -m app.demo --build' to create the fixtures")
    out = _new_session(paths)
    out["declared"] = match.get("declared") or {}
    out["title"] = match["title"]
    return out


@app.get("/api/photo/{session}/{name}")
def photo(session: str, name: str) -> FileResponse:
    path = (SESSIONS / session / name).resolve()
    if not str(path).startswith(str(SESSIONS.resolve())) or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/appraise/{session}")
def appraise(session: str, year: int | None = None, km: float | None = None,
             make: str | None = None, market: str = "TR",
             backend: str | None = None) -> StreamingResponse:
    folder = (SESSIONS / session).resolve()
    if not str(folder).startswith(str(SESSIONS.resolve())) or not folder.is_dir():
        raise HTTPException(404, "unknown session")
    photos = pipeline.collect_photos(folder)
    if not photos:
        raise HTTPException(400, "session has no photos")

    declared = {k: v for k, v in (("year", year), ("km", km), ("make", make))
                if v not in (None, "")}
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            def note(step, detail):
                events.put({"type": "stage", "step": step, "detail": detail})

            result = pipeline.appraise(photos, declared, market=market,
                                       backend=backend, on_step=note)
            payload = result.to_dict()
            payload["photo_urls"] = {c.photo_id: f"/api/photo/{session}/{Path(c.path).name}"
                                     for c in result.gate.photos}
            payload["text_report"] = report.render_text(result)
            events.put({"type": "result", "appraisal": payload})
        except Exception as exc:
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        yield "retry: 10000\n\n"
        while True:
            item = events.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
