from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from ..config import EVIDENCE_IMAGE_LONG_EDGE


class VLMError(RuntimeError):
    pass


@dataclass
class BackendStatus:
    name: str
    ready: bool
    detail: str = ""
    model: str = ""
    # Set when the backend is installed and configured but the account itself
    # is refusing work (unpaid invoice, rate limit, suspended key). Worth
    # separating from "not configured" because the fix is different.
    account_blocked: bool = False


@dataclass
class VLMResponse:
    text: str
    backend: str
    model: str
    elapsed_s: float = 0.0
    usage: dict = field(default_factory=dict)


def encode_jpeg(path: str | Path, long_edge: int = EVIDENCE_IMAGE_LONG_EDGE) -> tuple[bytes, tuple[int, int]]:
    """Downscale to `long_edge` and re-encode as JPEG.

    Uploading 4000 px dealer photos costs ~4x the tokens and buys nothing:
    the claims we ask for (tread grooves, rust pitting, seat bolster wear)
    are all legible at 1024 px, and the cap keeps a 14-photo call inside a
    single request on both backends.
    """
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(pil.size) > long_edge:
        scale = long_edge / max(pil.size)
        pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue(), pil.size


def encode_b64(path: str | Path, long_edge: int = EVIDENCE_IMAGE_LONG_EDGE) -> tuple[str, tuple[int, int]]:
    raw, size = encode_jpeg(path, long_edge)
    return base64.standard_b64encode(raw).decode("ascii"), size


class VLMBackend:
    """Interface every backend implements.

    `json_schema` is honoured natively where the backend supports constrained
    decoding and ignored otherwise - the prompt always carries the schema in
    prose too, so a backend without it still returns the right shape, just
    without the guarantee.
    """

    name: str = ""
    supports_structured_output: bool = False

    def probe(self) -> BackendStatus:
        raise NotImplementedError

    def complete(self, prompt: str, images: list[Path], *,
                 system: str = "", max_tokens: int = 4096,
                 json_schema: dict | None = None,
                 effort: str | None = None,
                 long_edge: int | None = None) -> VLMResponse:
        """`effort` and `long_edge` are per-call, not per-backend.

        The passes are not one kind of work. Reading a tread block off a
        photograph is perception; deciding whether that tread block is
        "moderate" against a written rubric is deduction, and the second wants
        a budget the first does not. None means the backend's configured
        default, so every existing caller keeps its behaviour.

        `long_edge` is the same idea for pixels. The close-ups are downscaled to
        `EVIDENCE_IMAGE_LONG_EDGE` because sixteen frames at full size is a
        twenty-thousand-token call for no gain on a tread block. Identity is the
        opposite problem: a model badge is small in frame, and 1024 px across a
        whole tractor leaves "F-MAX 500" a few pixels tall. None means the
        module default, so every existing caller keeps its behaviour.
        """
        raise NotImplementedError
