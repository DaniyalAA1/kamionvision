"""Cursor Agent SDK backend.

Uses the SDK's native image support (`SDKImage` / `UserMessage.images`) so
photos go up as image content, not as file paths the agent would have to open
itself. The agent is created with `tools=[]`: this stage is pure inference and
must not read the repo, run a shell, or otherwise wander - that keeps latency
predictable on stage and keeps the evidence grounded in the uploaded photos.

One agent per call, never reused. An Agent is a conversation, and reusing it
across appraisals would carry one truck's photos into the next truck's
evidence - a silent, very hard to notice correctness bug.

Auth note: the SDK wants a Cursor *API key* (from
cursor.com/dashboard -> API & SSH Keys). The `cursor-agent` CLI's
stored login is a session JWT and the SDK rejects it with
`unauthenticated: Invalid User API Key`, so being logged in to the CLI is
not sufficient.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import register
from ..config import CURSOR_API_KEY, CURSOR_MODEL, REPO
from .base import BackendStatus, VLMBackend, VLMError, VLMResponse, encode_jpeg

# Substrings that mean "the credentials are fine, the account is not".
_ACCOUNT_BLOCK_HINTS = ("unpaid invoice", "action required", "actionrequired",
                        "suspended", "quota", "billing", "payment")


def _classify(exc: Exception) -> tuple[str, bool]:
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    return text, any(h in low for h in _ACCOUNT_BLOCK_HINTS)


@register
class CursorBackend(VLMBackend):
    name = "cursor"

    def __init__(self) -> None:
        self.model = CURSOR_MODEL

    def probe(self) -> BackendStatus:
        try:
            import cursor_sdk  # noqa: F401
        except ImportError:
            return BackendStatus(self.name, False,
                                 "cursor-sdk not installed (uv pip install cursor-sdk)",
                                 model=self.model)
        if not CURSOR_API_KEY:
            return BackendStatus(
                self.name, False,
                "CURSOR_API_KEY is not set - mint one at cursor.com/dashboard -> "
                "Integrations -> API Keys and put it in .env "
                "(the cursor-agent CLI login is a session token and will not work)",
                model=self.model)
        from cursor_sdk import Cursor
        try:
            user = Cursor.me(api_key=CURSOR_API_KEY)
        except Exception as exc:
            detail, blocked = _classify(exc)
            return BackendStatus(self.name, False, detail, model=self.model,
                                 account_blocked=blocked)
        who = getattr(user, "user_email", None) or getattr(user, "user_id", "") or "authenticated"
        return BackendStatus(self.name, True, f"ready ({self.model}, {who})", model=self.model)

    def complete(self, prompt: str, images: list[Path], *,
                 system: str = "", max_tokens: int = 16000,
                 json_schema: dict | None = None) -> VLMResponse:
        # The Agent SDK has no constrained-decoding hook, so the schema is
        # carried in the prompt and enforced by evidence.parse. json_schema is
        # accepted and ignored to keep the two backends interchangeable.
        try:
            from cursor_sdk import Agent, AgentOptions, SDKImage, UserMessage, ModelSelection, ModelParameterValue
        except ImportError as exc:
            raise VLMError("cursor-sdk is not installed") from exc
        if not CURSOR_API_KEY:
            raise VLMError("CURSOR_API_KEY is not set")

        sdk_images = []
        for path in images:
            raw, _ = encode_jpeg(path)
            sdk_images.append(SDKImage.from_data(raw, "image/jpeg"))

        # photo_id is communicated in the prompt header rather than as
        # per-image text blocks: the SDK carries images as a list on one
        # message, so ordinal position is the only stable handle.
        header = ("The photos attached to this message are numbered by position: "
                  f"photo_id 0 through {len(images) - 1}, in the order attached.\n\n")
        text = (f"{system}\n\n{header}{prompt}" if system else header + prompt)

        t0 = time.time()
        try:
            with Agent.create(AgentOptions(
                model=ModelSelection(id=self.model, params=[
                    ModelParameterValue(id="reasoning", value="low"),
                ]),
                api_key=CURSOR_API_KEY,
                local={"cwd": str(REPO)},
                tools=[],          # pure inference: no shell, no file reads
            )) as agent:
                run = agent.send(UserMessage(text=text, images=sdk_images))
                out = run.text()
                result = run.wait()
                if result.status != "finished":
                    raise VLMError(f"Cursor run ended with status {result.status}")
                actual_model = getattr(result.model, "id", None)
                if actual_model != self.model:
                    raise VLMError(
                        f"Cursor model mismatch: requested {self.model}, returned {actual_model}")
                if not out.strip():
                    raise VLMError("Cursor returned an empty inference response")
        except Exception as exc:
            detail, blocked = _classify(exc)
            hint = (" - the Cursor account is blocked, not the key; clear it at "
                    "cursor.com/dashboard" if blocked else "")
            raise VLMError(f"cursor call failed: {detail}{hint}") from exc

        usage = {}
        if getattr(result, "usage", None) is not None:
            for field in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                value = getattr(result.usage, field, None)
                if value is not None:
                    usage[field] = value
        return VLMResponse(text=out, backend=self.name, model=actual_model,
                           elapsed_s=round(time.time() - t0, 2), usage=usage)
