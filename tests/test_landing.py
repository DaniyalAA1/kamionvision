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

    def test_the_scroll_story_is_labelled_as_a_frozen_run(self):
        # The page's whole argument is that its evidence is real. It has to say
        # both halves out loud: this one is frozen, and yours runs live.
        self.assertIn('one real run on a real Turkish listing', self.source)
        self.assertIn('Your own photos run live', self.source)

    def test_motion_preference_is_supported(self):
        for sheet in ('styles/landing.css', 'styles/story.css'):
            self.assertIn('prefers-reduced-motion', (WEB / sheet).read_text(), sheet)

    def test_every_in_page_anchor_has_a_target(self):
        ids = {attrs['id'] for _, attrs in self.page.tags if 'id' in attrs}
        for tag, attrs in self.page.tags:
            href = attrs.get('href', '')
            if tag == 'a' and href.startswith('#') and len(href) > 1:
                self.assertIn(href[1:], ids, href)


if __name__ == '__main__':
    unittest.main()
