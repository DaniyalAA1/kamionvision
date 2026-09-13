"""Concurrency and lifecycle tests for the live demo server."""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from app import server


class _Result:
    gate = SimpleNamespace(photos=[])
    status: str = "ok"
    headline: str = "Appraisal complete"

    def to_dict(self):
        return {"status": "ok"}


class AppraisalSerialization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sessions = Path(self.tmp.name)
        self.session = self.sessions / "session-one"
        self.session.mkdir()
        self.client = TestClient(server.app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_appraisal_is_rejected_while_first_is_running(self):
        entered = threading.Event()
        release = threading.Event()
        first_response = []

        def fake_appraise(photos, declared, *, market, backend, on_step,
                         on_gate, on_photo, **kwargs):
            on_step("gate", "started")
            entered.set()
            self.assertTrue(release.wait(5), "test did not release appraisal")
            return _Result()

        def first_request():
            first_response.append(self.client.get("/api/appraise/session-one"))

        with mock.patch.object(server, "SESSIONS", self.sessions), \
             mock.patch.object(server.pipeline, "collect_photos",
                               return_value=[Path("photo.jpg")]), \
             mock.patch.object(server.pipeline, "appraise",
                               side_effect=fake_appraise), \
             mock.patch.object(server.report, "render_text", return_value="report"):
            thread = threading.Thread(target=first_request)
            thread.start()
            self.assertTrue(entered.wait(5), "first appraisal did not start")

            busy = self.client.get("/api/appraise/session-one")
            self.assertEqual(busy.status_code, 429)
            self.assertEqual(busy.headers["retry-after"], "5")
            self.assertEqual(server._appraisals_in_flight(), 1)

            release.set()
            thread.join(5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(first_response[0].status_code, 200)
        self.assertEqual(server._appraisals_in_flight(), 0)


class SessionExpiry(unittest.TestCase):
    def test_removes_only_session_directories_older_than_six_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions = Path(tmp)
            stale = sessions / "stale"
            fresh = sessions / "fresh"
            stale.mkdir()
            fresh.mkdir()
            marker = sessions / "not-a-session"
            marker.write_text("keep", encoding="utf-8")
            now = time.time()
            os.utime(stale, (now - server.SESSION_TTL_SECONDS - 1,) * 2)

            with mock.patch.object(server, "SESSIONS", sessions):
                server._expire_sessions(now)

            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(marker.exists())

    def test_startup_expires_sessions(self):
        with mock.patch.object(server, "_expire_sessions") as expire, \
             mock.patch("app.vision.warm"), \
             mock.patch.object(server.threading, "Thread") as thread:
            server.warm()
        expire.assert_called_once_with()
        thread.assert_called_once_with(target=mock.ANY, daemon=True)
        thread.return_value.start.assert_called_once_with()


class Health(unittest.TestCase):
    def test_reports_optional_components_and_active_appraisals(self):
        model = SimpleNamespace(
            meta={}, calibration={}, widening={})
        with mock.patch("app.vlm.probe_all", return_value=[]), \
             mock.patch("app.pricing.load_model", return_value=model), \
             mock.patch("app.perception.heads.available", return_value=True), \
             mock.patch.object(server.importlib.util, "find_spec",
                               return_value=object()), \
             mock.patch.object(server, "_appraisals_in_flight", return_value=1):
            response = TestClient(server.app).get("/api/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIs(body["rapidocr"], True)
        self.assertIs(body["perception"], True)
        self.assertEqual(body["appraisals_in_flight"], 1)


if __name__ == "__main__":
    unittest.main()
