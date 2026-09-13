"""OpenAI GPT-5.6 backend, via the Responses API.

Uses `text.format = json_schema` with `strict: true`, which is the Responses
API's constrained decoding - the same guarantee the Anthropic backend gets
from `output_config.format`, so the evidence schema is enforced by the
decoder on both paths rather than by hopeful prompting.

`reasoning.effort` is the latency dial. The evidence call is perception, not
deduction: at low effort a 14-photo call returns in about the time a judge
will tolerate watching, and raising it mostly buys longer prose.

There is no bare `gpt-5.6` model id. The family ships as three variants -
luna, sol and terra - so the id is configurable and the default is whichever
`python -m app.vlm.bench` measured best on this task.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import register
from ..config import OPENAI_API_KEY, OPENAI_EFFORT, OPENAI_MODEL
from ..config import EVIDENCE_IMAGE_LONG_EDGE
from .base import BackendStatus, VLMBackend, VLMError, VLMResponse, encode_b64


@register
class OpenAIBackend(VLMBackend):
    name = "openai"
    supports_structured_output = True

    def __init__(self) -> None:
        self.model = OPENAI_MODEL
        self.effort = OPENAI_EFFORT
        self._client = None

    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise VLMError("openai package is not installed") from exc
            if not OPENAI_API_KEY:
                raise VLMError("OPENAI_API_KEY is not set")
            self._client = OpenAI(api_key=OPENAI_API_KEY)
        return self._client

    def probe(self) -> BackendStatus:
        try:
            import openai  # noqa: F401
        except ImportError:
            return BackendStatus(self.name, False, "openai package not installed",
                                 model=self.model)
        if not OPENAI_API_KEY:
            return BackendStatus(self.name, False, "OPENAI_API_KEY is not set",
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
            content.append({"type": "input_text", "text": f"photo_id={i}"})
            content.append({"type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{data}"})
        content.append({"type": "input_text", "text": prompt})

        kwargs: dict = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": effort or self.effort},
            "max_output_tokens": max_tokens,
        }
        if system:
            kwargs["instructions"] = system
        if json_schema is not None:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "truck_evidence",
                                         "schema": json_schema, "strict": True}}

        t0 = time.time()
        try:
            response = self.client().responses.create(**kwargs)
        except Exception as exc:
            raise VLMError(f"openai call failed: {type(exc).__name__}: {exc}") from exc

        if response.status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "?")
            # Surfaced rather than swallowed: a run that hit the token ceiling
            # returns a partial object, and a partial evidence report that
            # silently loses half its findings is worse than a visible failure.
            raise VLMError(f"openai response incomplete ({reason}) - raise max_output_tokens "
                           f"or lower KAMION_OPENAI_EFFORT")

        usage = {}
        if getattr(response, "usage", None):
            usage = {"input_tokens": response.usage.input_tokens,
                     "output_tokens": response.usage.output_tokens}
            details = getattr(response.usage, "output_tokens_details", None)
            if details is not None and getattr(details, "reasoning_tokens", None):
                usage["reasoning_tokens"] = details.reasoning_tokens
        return VLMResponse(text=response.output_text or "", backend=self.name,
                           model=self.model, elapsed_s=round(time.time() - t0, 2),
                           usage=usage)
