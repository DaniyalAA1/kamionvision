"""Tests for logging and execution tracing infrastructure."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.log import (
    ExecutionTracker,
    get_logger,
    init_logging,
    list_recent_executions,
)


class TestLogging(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_logger(self):
        logger = get_logger("test_module")
        self.assertEqual(logger.name, "kamion.test_module")
        # Should not raise
        logger.info("Test log message")

    def test_execution_tracker_lifecycle(self):
        with patch("app.log.EXECUTIONS_DIR", self.runs_path / "executions"), \
             patch("app.log.INDEX_FILE", self.runs_path / "executions" / "index.jsonl"):

            tracker = ExecutionTracker(run_id="test_run_123", source="test", metadata={"year": 2020, "km": 300000})
            self.assertEqual(tracker.run_id, "test_run_123")

            tracker.stage_started("evidence", detail="5 photos")
            tracker.photo_call(
                photo_id=1,
                view="front_34_left",
                sample_idx=0,
                backend="cursor",
                model="gpt-5.6-sol",
                duration_s=1.25,
                findings_count=3,
                tokens={"prompt": 500, "completion": 200},
            )
            tracker.stage_completed("evidence", duration_s=1.25, summary={"findings": 3})
            tracker.complete(status="ok", summary={"verdict": "ok"})

            # Check that run file exists and contains events
            self.assertTrue(tracker.file_path.exists())
            events = []
            with open(tracker.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))

            event_types = [e["event"] for e in events]
            self.assertIn("init", event_types)
            self.assertIn("stage_started", event_types)
            self.assertIn("photo_call", event_types)
            self.assertIn("stage_completed", event_types)
            self.assertIn("finish", event_types)

            # Check photo call details
            photo_event = next(e for e in events if e["event"] == "photo_call")
            self.assertEqual(photo_event["data"]["photo_id"], 1)
            self.assertEqual(photo_event["data"]["view"], "front_34_left")
            self.assertEqual(photo_event["data"]["model"], "gpt-5.6-sol")
            self.assertAlmostEqual(photo_event["data"]["duration_s"], 1.25)

            # Check index file
            index_file = self.runs_path / "executions" / "index.jsonl"
            self.assertTrue(index_file.exists())
            with open(index_file, "r", encoding="utf-8") as f:
                index_lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(index_lines), 1)
            self.assertEqual(index_lines[0]["run_id"], "test_run_123")
            self.assertEqual(index_lines[0]["status"], "ok")
            self.assertEqual(index_lines[0]["error_count"], 0)

    def test_execution_tracker_error_handling(self):
        with patch("app.log.EXECUTIONS_DIR", self.runs_path / "executions"), \
             patch("app.log.INDEX_FILE", self.runs_path / "executions" / "index.jsonl"):

            tracker = ExecutionTracker(run_id="err_run_456", source="test")
            tracker.stage_started("perception")

            try:
                raise ValueError("Simulated network timeout")
            except Exception as e:
                tracker.record_error("perception", e)

            tracker.complete(status="failed", summary={"error": "Simulated network timeout"})

            # Check index recorded the error
            runs = list_recent_executions(limit=10)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], "err_run_456")
            self.assertEqual(runs[0]["status"], "failed")
            self.assertEqual(runs[0]["error_count"], 1)

            # Test errors_only filter
            err_runs = list_recent_executions(errors_only=True)
            self.assertEqual(len(err_runs), 1)

    def test_list_recent_executions_filtering(self):
        with patch("app.log.EXECUTIONS_DIR", self.runs_path / "executions"), \
             patch("app.log.INDEX_FILE", self.runs_path / "executions" / "index.jsonl"):

            for i in range(5):
                t = ExecutionTracker(run_id=f"run_{i}")
                if i == 2:
                    t.record_error("stage", Exception("Boom"))
                    t.complete(status="failed", summary={"error": "Boom"})
                else:
                    t.complete(status="ok")

            all_runs = list_recent_executions(limit=10)
            self.assertEqual(len(all_runs), 5)

            err_runs = list_recent_executions(errors_only=True)
            self.assertEqual(len(err_runs), 1)
            self.assertEqual(err_runs[0]["run_id"], "run_2")

            limited = list_recent_executions(limit=2)
            self.assertEqual(len(limited), 2)


if __name__ == "__main__":
    unittest.main()
