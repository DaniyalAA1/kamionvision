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
from ..config import EVIDENCE_IMAGE_LONG_EDGE
from .base import BackendStatus, VLMBackend, VLMError, VLMResponse, encode_b64


# Constraints the Messages API 400s on. Anthropic accepts array `minItems`
# of 0 or 1 only; everything else in this set is documented as unsupported
# and is moved onto `description` so the model still sees the intent.
_UNSUPPORTED = frozenset({
    "maxItems", "minLength", "maxLength", "minimum", "maximum",
    "multipleOf", "exclusiveMinimum", "exclusiveMaximum", "uniqueItems",
})


def schema_for_anthropic(schema: dict) -> dict:
    """A copy Anthropic's structured-output validator will accept.

    Two constructs in the shared evidence schemas 400 here, and both are
    legal on OpenAI's strict decoder:

    * A nullable enum (`type: ["string", "null"]` plus `enum` including
      null) on the identity `model` field:
          Enum value 'F-MAX' does not match declared type '['string', 'null']'
    * Array `maxItems` / `minItems` other than 0 or 1 on the close-up
      schema (`component_regions` capped at 8, a box at 4 numbers).

    The shared schema stays as it is — OpenAI strict mode wants that shape.
    This walks a copy and rewrites only the fields that would be rejected.
    """
    return _adapt(schema)


def _adapt(node):
    if isinstance(node, list):
        return [_adapt(item) for item in node]
    if not isinstance(node, dict):
        return node
    node = {key: _adapt(value) for key, value in node.items()}
    node = _reconcile_enum(node)
    return _strip_constraints(node)


def _reconcile_enum(node: dict) -> dict:
    enum, declared = node.get("enum"), node.get("type")
    if not isinstance(enum, list) or declared is None:
        return node
    kinds = [_json_type(value) for value in enum]
    if isinstance(declared, str):
        allowed = {declared, "integer"} if declared == "number" else {declared}
        if set(kinds) <= allowed:
            return node
    elif not isinstance(declared, list):
        return node
    grouped: dict[str, list] = {}
    for kind, value in zip(kinds, enum):
        grouped.setdefault(kind, []).append(value)
    variants = []
    for kind, values in grouped.items():
        if kind == "null":
            variants.append({"type": "null"})
        else:
            variants.append({"type": kind, "enum": values})
    rewritten = {key: value for key, value in node.items()
                 if key not in {"type", "enum"}}
    rewritten["anyOf"] = variants
    return rewritten


def _strip_constraints(node: dict) -> dict:
    dropped = {}
    if node.get("minItems") not in (None, 0, 1):
        dropped["minItems"] = node.pop("minItems")
    for key in list(node):
        if key in _UNSUPPORTED:
            dropped[key] = node.pop(key)
    if not dropped:
        return node
    extra = ", ".join(f"{key}: {value}" for key, value in dropped.items())
    prior = node.get("description")
    node["description"] = f"{prior}\n\n{extra}" if prior else extra
    return node


def _json_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"unsupported enum value {value!r}")


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
                 effort: str | None = None,
                 long_edge: int | None = None) -> VLMResponse:
        content: list[dict] = []
        for i, path in enumerate(images):
            data, _ = encode_b64(path, long_edge or EVIDENCE_IMAGE_LONG_EDGE)
            content.append({"type": "text", "text": f"photo_id={i}"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": data}})
        content.append({"type": "text", "text": prompt})

        output_config: dict = {"effort": effort or self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema",
                                       "schema": schema_for_anthropic(json_schema)}

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
