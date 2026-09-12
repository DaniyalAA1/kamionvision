"""Cursor routing and inference-result verification, without network calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import vlm
from app.vlm.base import BackendStatus, VLMError
from app.vlm.cursor_backend import CursorBackend


class CursorInference(unittest.TestCase):
    def test_pinned_backend_never_falls_back(self):
        backend = MagicMock()
        backend.probe.return_value = BackendStatus('cursor', False, 'missing key')
        with patch('app.config.BACKEND_OVERRIDE', 'cursor'), patch.object(vlm, 'make', return_value=backend) as make:
            with self.assertRaises(VLMError):
                vlm.resolve()
            make.assert_called_once_with('cursor')

    def _complete(self, model='gpt-5.6-sol', status='finished'):
        result = SimpleNamespace(model=SimpleNamespace(id=model), status=status, usage=None)
        agent = MagicMock()
        agent.send.return_value.text.return_value = '{"ok":true}'
        agent.send.return_value.wait.return_value = result
        with patch('app.vlm.cursor_backend.CURSOR_API_KEY', 'test-only'), patch('cursor_sdk.Agent.create') as create:
            create.return_value.__enter__.return_value = agent
            backend = CursorBackend()
            backend.model = 'gpt-5.6-sol'
            response = backend.complete('test', [])
            options = create.call_args.args[0]
            self.assertEqual(options.model.id, 'gpt-5.6-sol')
            self.assertEqual(options.tools, [])
            create.return_value.__exit__.assert_called_once()
            return response

    def test_response_uses_verified_model(self):
        self.assertEqual(self._complete().model, 'gpt-5.6-sol')

    def test_wrong_model_rejected(self):
        with self.assertRaisesRegex(VLMError, 'model mismatch'):
            self._complete(model='another-model')

    def test_failed_run_rejected(self):
        with self.assertRaisesRegex(VLMError, 'status failed'):
            self._complete(status='failed')
