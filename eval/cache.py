"""The keystone: a content-addressed cache of raw vision responses.

Pass B is 16 of the 18 calls an appraisal makes, so caching it is what decides
the whole architecture of this package. Once a response text is on disk under a
key that covers everything the provider was shown, any change *downstream* of
the call - `parse_closeup`, `merge_duplicates`, the rollup, the grade ladder,
the price multiplier, the scorecard itself - is re-scorable for free. A merge
threshold sweep and a grade-ladder sweep become grid searches rather than
budget decisions.

The key, and why each term is in it:

    sha256( image_bytes_as_sent || prompt || system || model_id || effort
            || schema_json || repeat_index )

  image_bytes_as_sent  the output of `app.vlm.base.encode_jpeg`, not the file
                       on disk. That is what the provider actually receives,
                       and it is deterministic given the file plus the crop
                       decision - so a cropped call and a whole-frame call of
                       the same photograph are correctly different keys.
  prompt / system      a prompt edit MUST bust the cache. That is the point:
                       the scorecard then reports 0% hits and a real bill, and
                       the workstream that changed the prompt is the one that
                       pays for re-collection.
  model_id / effort    `effort` is not in the design's key list because it was
                       a per-backend setting when that list was written and is
                       now per call. Two efforts are two different models as
                       far as a cached answer is concerned, so it is keyed.
  schema_json          constrained decoding changes the output shape.
  repeat_index         `config.CLOSEUP_SAMPLES` reads each photo more than
                       once on purpose. Without this term the three samples
                       would collapse to one cached answer and the
                       self-consistency measurement would silently read zero
                       variance.

Deliberately NOT keyed: `max_tokens`. It only truncates, a truncation already
surfaces as a parse warning, and keying it would throw away a 15 MB cache every
time a token budget is nudged. It is recorded in the stored payload instead, so
a mismatch is visible to anyone who looks.

Nothing here parses. The cache stores text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import EVIDENCE_IMAGE_LONG_EDGE, REPO
from app.vlm.base import VLMBackend, VLMError, VLMResponse, encode_jpeg

CACHE_ROOT = REPO / "eval" / "cache"

# Cache payloads carry this. Bumping it is how a storage-format change
# invalidates old entries without a key change.
PAYLOAD_VERSION = 1

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class CacheMiss(VLMError):
    """Raised by an offline `CachedBackend` asked for something it does not hold.

    A subclass of VLMError so `evidence.stage`'s existing failure posture - one
    lost photo is not a lost appraisal - applies unchanged when a replay is
    driven through the real pipeline.
    """


def _sanitise(name: str) -> str:
    return _SAFE.sub("-", (name or "unknown").strip()) or "unknown"


# --- image digests ---------------------------------------------------------

_DIGEST_MEMO: dict[tuple, str] = {}


def image_digest(path: str | Path, long_edge: int = EVIDENCE_IMAGE_LONG_EDGE) -> str:
    """sha256 of the JPEG bytes this photo becomes on the wire.

    Memoised on (path, mtime, size, long_edge) because a 480-pair twin run
    would otherwise decode, resize and re-encode the same originals once per
    sample. The mtime is in the memo key so editing a fixture is not silently
    ignored within a process.
    """
    p = Path(path)
    try:
        st = p.stat()
        memo = (str(p.resolve()), st.st_mtime_ns, st.st_size, long_edge)
    except OSError as exc:
        raise VLMError(f"cannot read image {p}: {exc}") from exc
    hit = _DIGEST_MEMO.get(memo)
    if hit:
        return hit
    raw, _ = encode_jpeg(p, long_edge)
    digest = hashlib.sha256(raw).hexdigest()
    _DIGEST_MEMO[memo] = digest
    return digest


def _field(label: str, value: str) -> bytes:
    """Length-prefixed, so no concatenation of two fields can equal another.

    Without this, prompt="ab" + system="c" and prompt="a" + system="bc" hash
    the same, and a cache that confuses those is worse than no cache.
    """
    body = value.encode("utf-8")
    return f"{label}:{len(body)}:".encode("ascii") + body + b"\x00"


def cache_key(*, image_digests: list[str], prompt: str, system: str,
              model_id: str, effort: str | None, schema: dict | None,
              repeat_index: int) -> str:
    h = hashlib.sha256()
    h.update(_field("n_images", str(len(image_digests))))
    for i, digest in enumerate(image_digests):
        h.update(_field(f"image{i}", digest))
    h.update(_field("prompt", prompt))
    h.update(_field("system", system))
    h.update(_field("model", model_id))
    h.update(_field("effort", effort or ""))
    h.update(_field("schema", json.dumps(schema, sort_keys=True) if schema else ""))
    h.update(_field("repeat", str(int(repeat_index))))
    return h.hexdigest()


# --- statistics ------------------------------------------------------------

@dataclass
class CacheStats:
    """What a run cost, in the only units that matter.

    `new` is money. A prompt change busts every key it touches and must show up
    here as a hit rate of zero rather than as a surprise on an invoice.
    """
    new: int = 0
    cached: int = 0
    failed: int = 0
    bytes_written: int = 0
    seconds_spent: float = 0.0

    @property
    def total(self) -> int:
        return self.new + self.cached + self.failed

    @property
    def hit_rate(self) -> float:
        served = self.new + self.cached
        return round(self.cached / served, 4) if served else 0.0

    def to_dict(self) -> dict:
        return {"new": self.new, "cached": self.cached, "failed": self.failed,
                "hit_rate": self.hit_rate,
                "bytes_written": self.bytes_written,
                "seconds_spent": round(self.seconds_spent, 1)}

    def merge(self, other: "CacheStats") -> None:
        self.new += other.new
        self.cached += other.cached
        self.failed += other.failed
        self.bytes_written += other.bytes_written
        self.seconds_spent += other.seconds_spent


# --- the store -------------------------------------------------------------

class ResponseCache:
    """`eval/cache/<model-id>/<hash>.json`, one response per file.

    One file per response rather than a database because the access pattern is
    random single-key reads from sixteen threads, the corpus is ~15 MB per
    model, and a file that can be `cat`-ed is a file whose contents can be
    argued with.
    """

    def __init__(self, root: Path | str = CACHE_ROOT):
        self.root = Path(root)
        self.stats = CacheStats()

    def path_for(self, model_id: str, key: str) -> Path:
        return self.root / _sanitise(model_id) / f"{key}.json"

    def has(self, model_id: str, key: str) -> bool:
        return self.path_for(model_id, key).exists()

    def get(self, model_id: str, key: str) -> dict | None:
        path = self.path_for(model_id, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A half-written entry is a miss, not a crash. The run pays for one
            # call and overwrites it.
            return None

    def put(self, model_id: str, key: str, payload: dict) -> int:
        path = self.path_for(model_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(payload, ensure_ascii=False, indent=1)
        # Write-then-rename: sixteen concurrent close-up calls share this root,
        # and a reader must never see half an entry.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)
        return len(body.encode("utf-8"))

    def size(self) -> tuple[int, int]:
        """(entries, bytes) currently on disk, across every model."""
        n = total = 0
        for f in self.root.rglob("*.json"):
            n += 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
        return n, total


# --- the backend wrapper ---------------------------------------------------

class CachedBackend(VLMBackend):
    """A `VLMBackend` that answers from disk first.

    Wraps a real backend, or none at all: `CachedBackend(None, model_id=...)`
    is the offline replay client, and it raises `CacheMiss` rather than
    reaching the network. That is what makes `--replay` and the committed
    golden fixtures provably zero-call instead of merely cheap.
    """

    name = "cached"

    def __init__(self, inner: VLMBackend | None, *, cache: ResponseCache | None = None,
                 model_id: str | None = None, read_only: bool = False,
                 record=None):
        self.inner = inner
        self.cache = cache or ResponseCache()
        self.model_id = model_id or getattr(inner, "model", "") or getattr(inner, "name", "unknown")
        self.read_only = read_only or inner is None
        # Called with (key, payload, hit: bool) after every served call, so a
        # suite can write its unit index without re-deriving keys.
        self.record = record

    @property
    def supports_structured_output(self) -> bool:
        return bool(self.inner and self.inner.supports_structured_output)

    @property
    def stats(self) -> CacheStats:
        return self.cache.stats

    def probe(self):
        from app.vlm.base import BackendStatus
        if self.inner is None:
            entries, _ = self.cache.size()
            return BackendStatus(name=self.name, ready=True, model=self.model_id,
                                 detail=f"offline replay over {entries} cached responses")
        return self.inner.probe()

    def key_for(self, prompt: str, images: list[Path], *, system: str = "",
                json_schema: dict | None = None, effort: str | None = None,
                repeat_index: int = 0) -> str:
        return cache_key(image_digests=[image_digest(p) for p in images],
                         prompt=prompt, system=system, model_id=self.model_id,
                         effort=effort, schema=json_schema, repeat_index=repeat_index)

    def complete(self, prompt: str, images: list[Path], *, system: str = "",
                 max_tokens: int = 4096, json_schema: dict | None = None,
                 effort: str | None = None, repeat_index: int = 0) -> VLMResponse:
        key = self.key_for(prompt, images, system=system, json_schema=json_schema,
                           effort=effort, repeat_index=repeat_index)
        hit = self.cache.get(self.model_id, key)
        if hit is not None:
            self.cache.stats.cached += 1
            if self.record:
                self.record(key, hit, True)
            return VLMResponse(text=hit.get("text", ""),
                               backend=hit.get("backend", "cached"),
                               model=hit.get("model", self.model_id),
                               elapsed_s=float(hit.get("elapsed_s") or 0.0),
                               usage=hit.get("usage") or {})

        if self.inner is None:
            self.cache.stats.failed += 1
            raise CacheMiss(
                f"no cached response for key {key[:12]}... under model "
                f"{self.model_id!r}; this client is offline by construction")

        t0 = time.time()
        try:
            response = self.inner.complete(prompt, images, system=system,
                                           max_tokens=max_tokens,
                                           json_schema=json_schema, effort=effort)
        except Exception:
            self.cache.stats.failed += 1
            raise
        elapsed = time.time() - t0

        payload = {
            "payload_version": PAYLOAD_VERSION,
            "key": key,
            "text": response.text,
            "backend": response.backend,
            "model": response.model,
            "elapsed_s": round(response.elapsed_s or elapsed, 2),
            "usage": response.usage or {},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # Recorded, not keyed - see the module docstring.
            "max_tokens": max_tokens,
            "n_images": len(images),
            "effort": effort or "",
        }
        self.cache.stats.new += 1
        self.cache.stats.seconds_spent += elapsed
        if not self.read_only:
            self.cache.stats.bytes_written += self.cache.put(self.model_id, key, payload)
        if self.record:
            self.record(key, payload, False)
        return VLMResponse(text=response.text, backend=response.backend,
                           model=response.model, elapsed_s=payload["elapsed_s"],
                           usage=payload["usage"])


@dataclass
class PlannedCall:
    """One call a suite intends to make, resolved to a key before any spending.

    `--estimate` is exact rather than a guess because a suite that forces the
    view tag, the crop decision and the soft-focus line (which `twin_fp` does,
    for confound reasons) can build its own prompts offline. The key then
    follows from the image file and the prompt with no provider involved.
    """
    key: str
    unit: str
    label: str
    cached: bool = False
    meta: dict = field(default_factory=dict)


def resolve_cached(planned: list[PlannedCall], model_id: str,
                   cache: ResponseCache | None = None) -> tuple[int, int]:
    """Mark each planned call as cached or not. Returns (cached, new)."""
    store = cache or ResponseCache()
    cached = 0
    for call in planned:
        call.cached = store.has(model_id, call.key)
        cached += bool(call.cached)
    return cached, len(planned) - cached
