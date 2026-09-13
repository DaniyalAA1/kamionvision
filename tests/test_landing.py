"""Offline landing-page contract checks (no model calls)."""
import unittest
from html.parser import HTMLParser
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / 'app' / 'web'

class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tags = []
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

class LandingTests(unittest.TestCase):
    def setUp(self):
        self.source = (WEB / 'landing.html').read_text()
        self.page = Page(self.source)

    def test_assets_resolve(self):
        for tag, attrs in self.page.tags:
            for key in ('src', 'href'):
                value = attrs.get(key, '')
                if value.startswith('/static/'):
                    self.assertTrue((WEB / value.removeprefix('/static/')).is_file(), value)

    def test_appraisal_links_and_unique_ids(self):
        ids = [attrs['id'] for _, attrs in self.page.tags if 'id' in attrs]
        self.assertEqual(len(ids), len(set(ids)))
        links = [attrs.get('href') for tag, attrs in self.page.tags if tag == 'a']
        self.assertGreaterEqual(links.count('/app'), 4)
        self.assertIn('id="dropzone"', (WEB / 'index.html').read_text())

    def test_guided_demo_is_explicit(self):
        self.assertIn('not a live appraisal', self.source)
        controls = [attrs for _, attrs in self.page.tags if 'data-inspect' in attrs]
        self.assertEqual(len(controls), 6)
        self.assertTrue(all('aria-pressed' in attrs for attrs in controls))

    def test_motion_preference_is_supported(self):
        self.assertIn('prefers-reduced-motion:reduce', (WEB / 'styles/landing.css').read_text())
