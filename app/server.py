"""FastAPI app behind the demo screen.

Upload and appraisal are two steps on purpose. The gate finishes in a couple
of seconds and the vision call takes half a minute, so the browser streams
the run over SSE and paints the gate verdict - which photos were kept, which
were dropped and why, whether a truck was actually found - long before the
price exists. On a four-minute demo clock that turns a 40-second wait into a
sequence the room can watch.
"""
from __future__ import annotations

import importlib.util
import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from . import evidence, gallery, pipeline, report
from .config import HEIF_SUPPORT, IMAGE_SUFFIXES, IMAGES, REPO, WEB

app = FastAPI(title="KamionVision", docs_url="/api/docs")

SESSIONS = Path(tempfile.gettempdir()) / "kamionvision-sessions"
SESSIONS.mkdir(parents=True, exist_ok=True)
ALLOWED = IMAGE_SUFFIXES
MAX_PHOTOS = 60
MAX_BYTES = 25 * 1024 * 1024
SESSION_TTL_SECONDS = 6 * 60 * 60

_APPRAISAL_SLOT = threading.Semaphore(1)
_APPRAISAL_COUNT_LOCK = threading.Lock()
_APPRAISALS_IN_FLIGHT = 0


def _expire_sessions(now: float | None = None) -> None:
    """Remove session directories that have been idle for more than six hours."""
    cutoff = (time.time() if now is None else now) - SESSION_TTL_SECONDS
    for path in SESSIONS.iterdir():
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path)
        except OSError:
            pass


def _appraisals_in_flight() -> int:
    with _APPRAISAL_COUNT_LOCK:
        return _APPRAISALS_IN_FLIGHT


@app.on_event("startup")
def warm() -> None:
    """Pay the model load on boot, not on the first appraisal a judge watches."""
    from . import vision
    _expire_sessions()
    threading.Thread(target=vision.warm, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB / "landing.html").read_text(encoding="utf-8"))


@app.get("/app", response_class=HTMLResponse)
def appraisal() -> HTMLResponse:
    return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"))


@app.get("/static/{name:path}")
def static(name: str) -> FileResponse:
    # Nested, because the fonts, the schematic and the js modules live in
    # subdirectories. The guard is `is_relative_to` on the resolved path rather
    # than a parent comparison, so `..` still cannot climb out of app/web/.
    path = (WEB / name).resolve()
    if not path.is_relative_to(WEB.resolve()) or not path.is_file():
        raise HTTPException(404)
    # No caching. The browser's module map holds an ES module until the tab is
    # closed, so an edited js/ file kept serving the old one all through a
    # working session and every "that fix did not land" was a stale bundle.
    # These files are local and tiny; there is nothing to gain by caching them.
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health() -> dict:
    from . import vlm
    from .config import MAX_EVIDENCE_PHOTOS, BACKEND_OVERRIDE
    from .perception import heads
    from .pricing import load_model

    out: dict = {"backends": [
        {"name": s.name, "ready": s.ready, "detail": s.detail,
         "account_blocked": s.account_blocked, "model": s.model}
        for s in vlm.probe_all()]}
    out["selected_backend"] = BACKEND_OVERRIDE
    out["evidence_photos"] = MAX_EVIDENCE_PHOTOS
    out["rapidocr"] = importlib.util.find_spec("rapidocr_onnxruntime") is not None
    out["perception"] = heads.available()
    out["appraisals_in_flight"] = _appraisals_in_flight()
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


@app.get("/api/gallery")
def gallery_cards() -> dict:
    """Every truck the screen can appraise, in one grid.

    Rehearsed cases first - three of them are not corpus vehicles at all, and a
    gallery that could only offer real tractors could not demonstrate a refusal.
    """
    cards = gallery.cards()
    declared = {}
    from .demo import resolved_cases
    for case in resolved_cases():
        declared[case["id"]] = case.get("declared") or {}
    return {"cards": cards, "facets": gallery.facets(), "declared": declared}


@app.get("/api/corpus-photo/{source_key}/{listing_id}/{index}")
def corpus_photo(source_key: str, listing_id: str, index: int) -> FileResponse:
    paths = gallery.photo_paths(source_key, listing_id)
    if not 0 <= index < len(paths):
        raise HTTPException(404)
    return FileResponse(gallery.thumb(paths[index]))


@app.get("/api/case-photo/{case_id}/{index}")
def case_photo(case_id: str, index: int) -> FileResponse:
    from .demo import resolved_cases
    match = next((c for c in resolved_cases() if c["id"] == case_id), None)
    if not match:
        raise HTTPException(404)
    paths = pipeline.collect_photos(REPO / match["folder"])
    if not 0 <= index < len(paths):
        raise HTTPException(404)
    return FileResponse(gallery.thumb(paths[index]))


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
            why = f"unsupported type {suffix or '?'}"
            if suffix in (".heic", ".heif") and not HEIF_SUPPORT:
                why = "HEIC needs pillow-heif installed (uv pip install pillow-heif)"
            skipped.append({"name": f.filename, "why": why})
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


@app.post("/api/upload-truck")
def upload_truck(truck: str = Form(...)) -> dict:
    """A gallery card into a session, through the same door as an upload.

    One code path from here on: the gate cannot tell a corpus vehicle from a
    phone full of photos, and it should not be able to.
    """
    source_key, _, listing_id = truck.partition(":")
    paths = gallery.photo_paths(source_key, listing_id)
    if not paths:
        raise HTTPException(404, f"unknown truck {truck!r}")
    card = next((c for c in gallery.cards() if c["id"] == truck), {})
    out = _new_session(paths)
    out["declared"] = {k: v for k, v in (("year", card.get("year")),
                                         ("km", card.get("km")),
                                         ("make", card.get("make")))
                       if v not in (None, "", "—")}
    out["title"] = f"{card.get('make', '')} {card.get('model', '')}".strip()
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
             asking: float | None = None,
             backend: str | None = None) -> StreamingResponse:
    folder = (SESSIONS / session).resolve()
    if not str(folder).startswith(str(SESSIONS.resolve())) or not folder.is_dir():
        raise HTTPException(404, "unknown session")
    photos = pipeline.collect_photos(folder)
    if not photos:
        raise HTTPException(400, "session has no photos")
    if not _APPRAISAL_SLOT.acquire(blocking=False):
        raise HTTPException(429, "another appraisal is already running",
                            headers={"Retry-After": "5"})
    global _APPRAISALS_IN_FLIGHT
    with _APPRAISAL_COUNT_LOCK:
        _APPRAISALS_IN_FLIGHT += 1

    declared = {k: v for k, v in (("year", year), ("km", km), ("make", make),
                                  ("asking_price", asking))
                if v not in (None, "")}
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            def note(step, detail):
                events.put({"type": "stage", "step": step, "detail": detail})

            def urls_for(checks):
                return {c.photo_id: f"/api/photo/{session}/{Path(c.path).name}"
                        for c in checks}

            def gate_done(gate):
                # Which frames the vision call will actually be given. It is a
                # view-diverse subset capped at MAX_EVIDENCE_PHOTOS, not every
                # usable frame, and the screen names them one by one while it
                # waits - so it has to be the real list, not a guess.
                evidence_ids = [c.photo_id for c in evidence.select_photos(gate)]
                events.put({"type": "gate", "gate": gate.to_dict(),
                            "photo_urls": urls_for(gate.photos),
                            "evidence_photo_ids": evidence_ids})

            def photo_read(finding):
                # One finished vision call, pushed the moment it lands. They
                # arrive out of order because the close-ups run concurrently,
                # and the screen is built to show that rather than hide it.
                events.put({"type": "photo", "finding": finding.to_dict()})

            result = pipeline.appraise(photos, declared, market=market,
                                       backend=backend, on_step=note,
                                       on_gate=gate_done, on_photo=photo_read,
                                       on_activity=lambda activity: events.put({"type": "activity", **activity}))
            payload = result.to_dict()
            payload["photo_urls"] = urls_for(result.gate.photos)
            payload["text_report"] = report.render_text(result)
            events.put({"type": "result", "appraisal": payload})
            _expire_sessions()
        except Exception as exc:
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            global _APPRAISALS_IN_FLIGHT
            with _APPRAISAL_COUNT_LOCK:
                _APPRAISALS_IN_FLIGHT -= 1
            _APPRAISAL_SLOT.release()
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
