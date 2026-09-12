"""Vision-model backends behind one interface.

Two implementations ship: `cursor` (the Cursor Agent SDK, which this build
targets) and `anthropic` (the Messages API). They are interchangeable because
`evidence.py` only ever asks for "this prompt, these images, give me back
text" - all the structure lives in the prompt and the parser, not the vendor.

Resolution order is config.BACKEND_CHAIN, first available wins, unless
KAMION_VLM_BACKEND pins one. A backend that cannot authenticate reports *why*
through `probe()` rather than raising at import, so `kamion doctor` can show
the reason and the demo can fall through to the one that works.
"""
from __future__ import annotations

from .base import BackendStatus, VLMBackend, VLMError, VLMResponse

_REGISTRY: dict[str, type[VLMBackend]] = {}


def register(cls: type[VLMBackend]) -> type[VLMBackend]:
    _REGISTRY[cls.name] = cls
    return cls


def _load() -> None:
    if _REGISTRY:
        return
    from . import anthropic_backend, cursor_backend  # noqa: F401


def available_names() -> list[str]:
    _load()
    return list(_REGISTRY)


def make(name: str) -> VLMBackend:
    _load()
    if name not in _REGISTRY:
        raise VLMError(f"unknown VLM backend {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def probe_all() -> list[BackendStatus]:
    """Status of every backend, in resolution order. Never raises."""
    from ..config import BACKEND_CHAIN
    _load()
    order = [n for n in BACKEND_CHAIN if n in _REGISTRY] + \
            [n for n in _REGISTRY if n not in BACKEND_CHAIN]
    out = []
    for name in order:
        try:
            out.append(make(name).probe())
        except Exception as exc:  # a broken backend must not hide a working one
            out.append(BackendStatus(name=name, ready=False,
                                     detail=f"{type(exc).__name__}: {exc}"))
    return out


def resolve(preferred: str | None = None) -> VLMBackend:
    """The backend an appraisal should use, or raise with every reason listed."""
    from ..config import BACKEND_CHAIN, BACKEND_OVERRIDE
    _load()
    pin = preferred or BACKEND_OVERRIDE
    if pin:
        backend = make(pin)
        status = backend.probe()
        if not status.ready:
            raise VLMError(f"backend {pin!r} was pinned but is not usable: {status.detail}")
        return backend

    problems = []
    for name in BACKEND_CHAIN:
        if name not in _REGISTRY:
            continue
        backend = make(name)
        status = backend.probe()
        if status.ready:
            return backend
        problems.append(f"  {name}: {status.detail}")
    raise VLMError("no vision backend is usable:\n" + "\n".join(problems))
