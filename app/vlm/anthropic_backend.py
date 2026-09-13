"""Anthropic Messages API backend.

Two things here are load-bearing and were both measured, not assumed:

`output_config.format` - constrained decoding against the evidence schema.
Before it, ~1 call in 3 came back as an unterminated JSON object and paid for
a repair round trip; with it the first block is valid JSON by construction.

`output_config.effort` - thinking is ON BY DEFAULT on Opus 5, and at the
default effort the evidence call spent its whole token budget thinking and
got truncated mid-object. This is a perception task, not a reasoning one, so
it runs at low effort. That is a latency decision for a live demo, and it is
the one knob to turn if the notes ever look shallow.

Images are interleaved with `photo_id=N` text blocks so the model's citations
land on indices we assigned rather than an ordering it had to infer.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import register
from ..config import ANTHROPIC_API_KEY, ANTHROPIC_EFFORT, ANTHROPIC_MODEL
from .base import BackendStatus, VLMBackend, VLMError, VLMResponse, encode_b64


@register
class AnthropicBackend(VLMBackend):
    name = "anthropic"
    supports_structured_output = True

    def __init__(self) -> None:
        self.model = ANTHROPIC_MODEL
        self.effort = ANTHROPIC_EFFORT
        self._client = None

    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise VLMError("anthropic package is not installed") from exc
            if not ANTHROPIC_API_KEY:
                raise VLMError("ANTHROPIC_API_KEY is not set")
            self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

    def probe(self) -> BackendStatus:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return BackendStatus(self.name, False, "anthropic package not installed",
                                 model=self.model)
        if not ANTHROPIC_API_KEY:
            return BackendStatus(self.name, False, "ANTHROPIC_API_KEY is not set",
                                 model=self.model)
        return BackendStatus(self.name, True, f"ready ({self.model}, effort={self.effort})",
                             model=self.model)

    def complete(self, prompt: str, images: list[Path], *,
                 system: str = "", max_tokens: int = 16000,
                 json_schema: dict | None = None,
                 effort: str | None = None) -> VLMResponse:
        content: list[dict] = []
        for i, path in enumerate(images):
            data, _ = encode_b64(path)
            content.append({"type": "text", "text": f"photo_id={i}"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": data}})
        content.append({"type": "text", "text": prompt})

        output_config: dict = {"effort": effort or self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}

        t0 = time.time()
        try:
            # Streamed: a 14-image request with a 16k ceiling is exactly the
            # shape that trips the SDK's non-streaming HTTP timeout.
            with self.client().messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system or None,
                output_config=output_config,
                messages=[{"role": "user", "content": content}],
            ) as stream:
                msg = stream.get_final_message()
        except Exception as exc:
            raise VLMError(f"anthropic call failed: {type(exc).__name__}: {exc}") from exc

        if getattr(msg, "stop_reason", None) == "refusal":
            raise VLMError(f"model declined the request: {getattr(msg, 'stop_details', None)}")

        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        usage = {"input_tokens": getattr(msg.usage, "input_tokens", 0),
                 "output_tokens": getattr(msg.usage, "output_tokens", 0)}
        if getattr(msg, "stop_reason", None) == "max_tokens":
            usage["truncated"] = True
        return VLMResponse(text=text, backend=self.name, model=self.model,
                           elapsed_s=round(time.time() - t0, 2), usage=usage)
