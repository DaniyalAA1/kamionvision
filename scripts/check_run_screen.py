"""Zero-inference browser regression for the real photo-review renderer.

Start the local app, then run:
  .venv/bin/python scripts/check_run_screen.py [http://127.0.0.1:8000]
Requires Playwright and Chrome. Saved appraisal + local photos only; no uploads
or appraisal requests are allowed. Screenshots go to /tmp/kamion-run-check/.
"""
import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/tmp/kamion-run-check')
OUT.mkdir(exist_ok=True)
appraisal = json.loads((ROOT / 'data/reference/story_appraisal.json').read_text())
photos = appraisal['gate']['photos']
findings = appraisal['evidence']['photo_findings']
appraisal['photo_urls'] = {str(c['photo_id']): '/qa-photo/' + c['filename'] for c in photos}
base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    page = browser.new_page(viewport={'width': 1366, 'height': 768}, reduced_motion='reduce')
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.route('**/api/appraise/**', lambda r: r.abort())
    page.route('**/api/upload**', lambda r: r.abort())
    page.route('**/qa-photo/*', lambda r: r.fulfill(path=str(
        ROOT / 'demo/tr_clean' / r.request.url.split('/')[-1])))
    page.route('**/qa-missing.jpg', lambda r: r.fulfill(status=404, body='missing'))
    page.goto(base + '/app')
    page.wait_for_function("document.querySelector('#case-shelf button') !== null")
    page.evaluate('''async a => {
      window.qaRun = await import('/static/js/run.js');
      window.qaFrames = await import('/static/js/frames.js');
      window.qaAppraisal = a;
      document.querySelector('#intake').hidden = true;
      document.querySelector('#topbar-state').textContent = 'QA REPLAY · saved appraisal, not a new live run';
      qaRun.begin();
      qaRun.onGate({gate:a.gate, photo_urls:a.photo_urls,
        evidence_photo_ids:a.evidence.photo_findings.map(f=>f.photo_id)});
      qaRun.onStage({step:'evidence'});
    }''', appraisal)

    def displayed(photo_id):
        page.wait_for_function('id => qaFrames.currentPhotoId() === id && document.querySelector("#frame-img").naturalWidth > 0', arg=photo_id)

    displayed(photos[0]['photo_id'])
    for finding in findings[:3]:
        page.evaluate('finding => qaRun.onPhoto({finding})', finding)
        displayed(finding['photo_id'])
    # Manual selection holds, even while the stream advances; resume catches up.
    page.locator('#cell-' + str(photos[0]['photo_id'])).click()
    displayed(photos[0]['photo_id'])
    page.evaluate('finding => qaRun.onPhoto({finding})', findings[3])
    assert page.evaluate('qaFrames.currentPhotoId()') == photos[0]['photo_id']
    assert page.locator('#follow-live').get_attribute('aria-pressed') == 'false'
    page.locator('#follow-live').click()
    displayed(findings[3]['photo_id'])

    page.evaluate("window.scrollTo(0, document.querySelector('#run').offsetTop)")
    page.screenshot(path=str(OUT / 'review-normal.png'))

    # A selected photo without an individual call must not claim to be reading.
    not_read = next(c for c in photos if c['photo_id'] not in {f['photo_id'] for f in findings})
    page.evaluate('c => qaFrames.showFrame(c)', not_read)
    displayed(not_read['photo_id'])
    assert 'Not selected' in page.locator('#frame-intel').inner_text()

    # Adversarially long content uses the same schema, explicitly QA-only.
    long_finding = dict(findings[-1])
    long_finding['cannot_tell'] = ['QA stress content: ' + 'Cannot assess from this photograph. ' * 8] * 24
    page.evaluate('f => qaRun.onPhoto({finding:f})', long_finding)
    displayed(long_finding['photo_id'])
    for width, height in [(1366, 768), (1024, 600), (1440, 900), (390, 844)]:
        page.set_viewport_size({'width': width, 'height': height})
        page.evaluate("window.scrollTo(0, document.querySelector('#run').offsetTop)")
        metrics = page.evaluate('''() => {
          const box = s => document.querySelector(s).getBoundingClientRect().toJSON();
          const intel = document.querySelector('#frame-intel');
          return {frame:box('#frame-stage'), foot:box('.stage-foot'),
            intel:box('#frame-intel'), overflow:intel.scrollHeight > intel.clientHeight,
            pageOverflow:document.documentElement.scrollWidth > innerWidth};
        }''')
        assert not metrics['pageOverflow'], (width, metrics)
        assert metrics['overflow'], (width, metrics)
        assert metrics['frame']['height'] >= 176, metrics
        if width > 940:
            assert metrics['foot']['bottom'] <= height + 1, (width, metrics)
        before = page.evaluate('scrollY')
        page.evaluate('qaFrames.markCell(0)')
        assert page.evaluate('scrollY') == before, 'Contact strip dragged the page vertically'
        page.screenshot(path=str(OUT / f'review-{width}.png'))

    # New frame starts at its heading, not the previous photo's scrolled prose.
    page.locator('#frame-intel').evaluate('e => e.scrollTop = 200')
    page.evaluate('c => qaFrames.showFrame(c)', photos[0])
    displayed(photos[0]['photo_id'])
    assert page.locator('#frame-intel').evaluate('e => e.scrollTop') == 0

    # Missing image clears old evidence; switching back recovers.
    page.evaluate('''() => {
      const a=qaAppraisal;
      qaFrames.setSource({...a.photo_urls, 0:'/qa-missing.jpg'}, a.gate.photos, a.gate.decision);
      qaFrames.showFrame(a.gate.photos[0]);
    }''')
    page.wait_for_function("document.querySelector('#frame-img').alt === 'Photo could not be loaded'")
    assert page.locator('#frame-boxes').inner_html() == ''
    assert page.locator('#frame-intel').inner_html() == ''
    page.evaluate('''() => {
      const a=qaAppraisal;
      qaFrames.setSource(a.photo_urls,a.gate.photos,a.gate.decision);
      qaFrames.showFrame(a.gate.photos[0]);
      qaFrames.showFrame(a.gate.photos[1]);
    }''')
    displayed(photos[1]['photo_id'])

    # Interruption + restart must discard the old image and pending callback.
    page.evaluate("qaRun.onError('QA simulated interruption')")
    assert page.locator('#scan-status').inner_text() == 'Inspection interrupted'
    page.evaluate('qaFrames.showFrame(qaAppraisal.gate.photos[0]); qaRun.begin()')
    page.wait_for_timeout(100)
    assert page.locator('#frame-img').get_attribute('src') is None
    assert page.locator('#frame-intel').inner_html() == ''

    # Exercise actual animation mode too, including outgoing image cleanup.
    page.emulate_media(reduced_motion='no-preference')
    page.evaluate('''() => {
      const a=qaAppraisal;
      qaRun.onGate({gate:a.gate,photo_urls:a.photo_urls,evidence_photo_ids:[]});
    }''')
    displayed(photos[0]['photo_id'])
    page.evaluate('c => qaFrames.showFrame(c)', photos[1])
    displayed(photos[1]['photo_id'])
    page.wait_for_function("document.querySelectorAll('.frame-outgoing').length === 0")
    assert not errors, errors
    browser.close()
print(f'PASS: photo loading, follow/resume, four viewports, long content, failure/recovery, restart, motion. Screenshots: {OUT}')
