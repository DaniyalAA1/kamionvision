/* Local browser regression checks. No inference calls are made.
   Install playwright separately, run the app on :8000, then:
   NODE_PATH=/path/to/node_modules node tests/browser_ui.cjs
   Uses the saved appraisal; maps its older per_photo records to stream events. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const origin = process.env.KAMION_TEST_URL || 'http://127.0.0.1:8000';

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    const a = JSON.parse(fs.readFileSync(path.join(root, 'runs/cursor-sol-verification.json')));
    a.evidence.photo_findings ||= a.evidence.per_photo.map((f) => ({
      ...f, shows: f.notes, issues: [], strengths: [],
    }));
    a.evidence.photos_read = a.evidence.photo_findings.length;
    // Serve fixture photos from disk even if the original session has expired.
    for (const photo of a.gate.photos) {
      if (fs.existsSync(photo.path)) {
        a.photo_urls[photo.photo_id] = 'data:image/jpeg;base64,' + fs.readFileSync(photo.path).toString('base64');
      }
    }
    await page.addInitScript((a) => { window.KAMION_APPRAISAL = a; }, a);
    await page.goto(`${origin}/app`);
    await page.locator('.range-row').first().waitFor();
    await page.waitForFunction(() => document.querySelector('#frame-img').naturalWidth > 0);
    assert.equal(await page.locator('.range-row').count(), 2);
    assert.match(await page.locator('#band').innerText(), /not confirmed sale prices/);
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `overflow at ${width}`);
    }
    await page.evaluate(async (price) => {
      window.qaBand = await import('/static/js/band.js');
      qaBand.drawBand(document.querySelector('#band'), { ...price, asking: { asking: price.high * 10 } });
    }, a.price);
    assert.equal(await page.locator('.off-right').count(), 2);
    assert.match(await page.locator('.seller-key').innerText(), /above chart scale/);
    await page.evaluate((price) => qaBand.drawBand(document.querySelector('#band'), {
      ...price, asking: { asking: 1 },
    }), a.price);
    assert.equal(await page.locator('.off-left').count(), 2);
    await page.evaluate((price) => qaBand.drawBand(document.querySelector('#band'), {
      ...price, low: 100, high: 100, point: 100, baseline_low: 100, baseline_high: 100,
      baseline_point: 100, asking: null,
    }), a.price);
    assert.equal(await page.locator('.range-seller').count(), 0);
    assert.equal(await page.locator('[style*="NaN"]').count(), 0);

    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.evaluate(async (a) => {
      window.qaRun = await import('/static/js/run.js');
      window.qaFrames = await import('/static/js/frames.js');
      qaRun.begin();
      qaRun.onGate({ gate: a.gate, photo_urls: a.photo_urls,
        evidence_photo_ids: a.evidence.photo_findings.map((f) => f.photo_id) });
      qaRun.onStage({ step: 'evidence' });
    }, a);
    assert.equal(await page.locator('#scan.on').count(), 1);
    const before = await page.locator('#frame-stage').boundingBox();
    for (const finding of a.evidence.photo_findings.slice(0, 5)) {
      await page.evaluate((finding) => qaRun.onPhoto({ finding }), finding);
    }
    assert.equal(await page.locator('.thought').count(), 5);
    await page.waitForFunction((id) => qaFrames.currentPhotoId() === id, a.evidence.photo_findings[4].photo_id);
    const after = await page.locator('#frame-stage').boundingBox();
    assert.ok(Math.abs(before.height - after.height) < 1, 'photo changes must not resize the stage');
    await page.locator('.thought-summary').first().click();
    assert.equal(await page.locator('.thought[open]').count(), 1);
    await page.locator('.thought-source').first().click();
    assert.equal(await page.locator('#lightbox').isVisible(), true);
    await page.locator('#lightbox-close').click();
    await page.locator('#follow-live').click();
    const held = await page.evaluate(() => qaFrames.currentPhotoId());
    await page.evaluate((finding) => qaRun.onPhoto({ finding }), a.evidence.photo_findings[5]);
    assert.equal(await page.evaluate(() => qaFrames.currentPhotoId()), held);
    await page.locator('#follow-live').click();
    await page.waitForFunction((id) => qaFrames.currentPhotoId() === id, a.evidence.photo_findings[5].photo_id);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    assert.equal(await page.locator('#scan').evaluate((n) => getComputedStyle(n, '::before').animationName), 'none');
    await page.evaluate(() => qaRun.onStage({ step: 'price' }));
    assert.equal(await page.locator('#scan.on').count(), 0);
    await page.evaluate(() => qaRun.onError('Test interruption'));
    assert.match(await page.locator('#scan-status').innerText(), /interrupted/);
    await page.evaluate(() => qaRun.begin());
    assert.equal(await page.locator('.thought').count(), 0);
    assert.equal(await page.locator('#frame-img').getAttribute('src'), null);
    assert.equal(await page.locator('#progress').isVisible(), true);
    await page.evaluate((a) => qaRun.onResult({ ...a, evidence: null, price: null }), a);
    assert.equal(await page.locator('[data-step="evidence"]').getAttribute('data-state'), 'skipped');
    assert.equal(await page.locator('[data-step="price"]').getAttribute('data-state'), 'skipped');
    assert.deepEqual(errors, []);
    console.log('PASS: result/mobile layout, price edge cases, burst arrivals, stable frames, photo lightbox, follow/pause, reduced motion, error, restart, refusal.');
  } finally { await browser.close(); }
})().catch((e) => { console.error(e); process.exitCode = 1; });
