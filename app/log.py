"""Structured, persistent logging and execution run tracing for KamionVision.

Provides:
  1. Standard logger hierarchy: `get_logger(name)` under the `kamion` root,
     writing to `runs/kamion.log` (rotating) and stderr.
  2. Structured execution run tracing: `ExecutionTracker` writes detailed JSONL
     records of each appraisal run to `runs/executions/<run_id>.jsonl` so any
     run can be audited, debugged, and inspected later.
  3. CLI helper functions for viewing recent execution logs and errors.
"""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid
from typing import Any

from .config import REPO

LOG_DIR = REPO / "runs"
LOG_FILE = LOG_DIR / "kamion.log"
EXECUTIONS_DIR = LOG_DIR / "executions"
INDEX_FILE = EXECUTIONS_DIR / "index.jsonl"

_DEFAULT_LOG_LEVEL = os.environ.get("KAMION_LOG_LEVEL", "INFO").upper()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EXECUTIONS_DIR.mkdir(parents=True, exist_ok=True)


def init_logging(level: str | None = None) -> None:
    """Initialize root kamion logger with console and rotating file handlers."""
    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        _ensure_dirs()
        log_level_name = (level or _DEFAULT_LOG_LEVEL).strip().upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        root = logging.getLogger("kamion")
        root.setLevel(log_level)
        root.propagate = False

        # Clear existing handlers to prevent duplicate output on reloads
        root.handlers.clear()

        # Rotating file handler (10 MB per file, 5 backups)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            str(LOG_FILE), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

        # Stderr handler for interactive development and CLI runs
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        root.addHandler(console_handler)

        _INITIALIZED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger childed under `kamion`."""
    if not _INITIALIZED:
        init_logging()
    if not name or name == "kamion":
        return logging.getLogger("kamion")
    if name.startswith("kamion."):
        return logging.getLogger(name)
    if name.startswith("app."):
        return logging.getLogger(f"kamion.{name[4:]}")
    return logging.getLogger(f"kamion.{name}")


class ExecutionTracker:
    """Tracks a single appraisal execution and persists structured event telemetry.

    Writes events to `runs/executions/<run_id>.jsonl` and adds a summary line to
    `runs/executions/index.jsonl` on finish.
    """

    def __init__(self, run_id: str | None = None, source: str = "pipeline",
                 metadata: dict[str, Any] | None = None) -> None:
        _ensure_dirs()
        self.run_id = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.source = source
        self.file_path = EXECUTIONS_DIR / f"{self.run_id}.jsonl"
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.logger = get_logger(f"execution.{self.run_id}")
        self.status = "in_progress"
        self.error_count = 0

        self.log_event("init", {
            "source": source,
            "start_time": self.start_time,
            "metadata": metadata or {},
        })
        self.logger.info("Started appraisal execution [%s] from %s", self.run_id, source)

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - self.start_time, 3),
            "event": event_type,
            "data": data,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self.lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def stage_started(self, stage: str, detail: str = "") -> None:
        self.logger.info("[%s] Stage started: %s (%s)", self.run_id, stage, detail)
        self.log_event("stage_started", {"stage": stage, "detail": detail})

    def stage_completed(self, stage: str, duration_s: float, summary: dict[str, Any] | None = None) -> None:
        self.logger.info("[%s] Stage completed: %s in %.2fs", self.run_id, stage, duration_s)
        self.log_event("stage_completed", {
            "stage": stage,
            "duration_s": round(duration_s, 3),
            "summary": summary or {},
        })

    def photo_call(self, photo_id: int, view: str, sample_idx: int = 0,
                   backend: str = "", model: str = "", duration_s: float = 0.0,
                   findings_count: int = 0, error: str | None = None,
                   tokens: dict[str, Any] | None = None) -> None:
        if error:
            self.error_count += 1
            self.logger.warning("[%s] Photo %d read error on %s: %s", self.run_id, photo_id, backend, error)
        else:
            self.logger.debug(
                "[%s] Photo %d (%s) finished via %s (%s) in %.2fs with %d findings",
                self.run_id, photo_id, view, backend, model, duration_s, findings_count,
            )
        self.log_event("photo_call", {
            "photo_id": photo_id,
            "view": view,
            "sample_idx": sample_idx,
            "backend": backend,
            "model": model,
            "duration_s": round(duration_s, 3),
            "findings_count": findings_count,
            "tokens": tokens or {},
            "error": error,
        })

    def record_error(self, stage: str, exc: Exception | str, context: dict[str, Any] | None = None) -> None:
        self.error_count += 1
        tb = traceback.format_exc() if isinstance(exc, Exception) else ""
        msg = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
        self.logger.error("[%s] Error in stage %s: %s", self.run_id, stage, msg)
        self.log_event("error", {
            "stage": stage,
            "error": msg,
            "traceback": tb,
            "context": context or {},
        })

    def complete(self, status: str = "ok", summary: dict[str, Any] | None = None) -> None:
        total_time = round(time.time() - self.start_time, 2)
        self.status = status
        summary = summary or {}
        summary["total_time_s"] = total_time
        summary["error_count"] = self.error_count

        self.logger.info(
            "[%s] Appraisal execution finished with status '%s' in %.2fs (%d errors)",
            self.run_id, status, total_time, self.error_count,
        )
        self.log_event("finish", {"status": status, "summary": summary})

        # Record in global index
        index_entry = {
            "run_id": self.run_id,
            "source": self.source,
            "start_time": self.start_time,
            "end_time": time.time(),
            "total_time_s": total_time,
            "status": status,
            "error_count": self.error_count,
            "summary": summary,
        }
        with open(INDEX_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False, default=str) + "\n")


def list_recent_executions(limit: int = 10, errors_only: bool = False) -> list[dict[str, Any]]:
    """Query recent execution entries from `runs/executions/index.jsonl`."""
    if not INDEX_FILE.exists():
        return []
    runs = []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if errors_only and entry.get("error_count", 0) == 0 and entry.get("status") == "ok":
                    continue
                runs.append(entry)
            except Exception:
                continue
    runs.reverse()
    return runs[:limit]


def get_execution_events(run_id: str) -> list[dict[str, Any]]:
    """Retrieve full trace of events for a specific execution run."""
    path = EXECUTIONS_DIR / f"{run_id}.jsonl"
    if not path.exists():
        # Try matching prefix
        matches = list(EXECUTIONS_DIR.glob(f"*{run_id}*.jsonl"))
        if not matches:
            return []
        path = matches[0]
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return events
