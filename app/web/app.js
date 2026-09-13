/* Entry point: wiring, the verdict block, and the title block.

   The run is streamed. The gate report now arrives as its own event about a
   second in, so the drawing, the frames and the detector's boxes are all on
   screen long before the price exists - see app/web/js/run.js. */

import { $, el, money, fixed, reduced, animate } from './js/dom.js';
import * as net from './js/net.js';
import * as run from './js/run.js';
import * as frames from './js/frames.js';
import * as elevation from './js/elevation.js';
import { drawGauge } from './js/gauge.js';
import { renderPanels } from './js/panels.js';

let stream = null;

/* ---------- title block ---------- */

async function loadHealth() {
  try {
    const h = await net.getHealth();
    const ready = h.backends.find(
      (b) => b.ready && (!h.selected_backend || b.name === h.selected_backend));
    const vision = $('tb-vision');
    if (ready) {
      vision.textContent = `${ready.name} / ${ready.model}`;
      vision.className = 'mono ready';
      $('masthead-state').textContent = '';
    } else {
      const blocked = h.backends.find((b) => b.account_blocked);
      vision.textContent = blocked ? `${blocked.name} blocked` : 'none configured';
      vision.className = 'mono broken';
      $('masthead-state').textContent = blocked
        ? `${blocked.name} account is blocked — ${blocked.detail}`
        : 'No vision backend is configured, so only the gate will run.';
      $('masthead-state').className = 'masthead-state broken';
    }
    const pm = h.price_model || {};
    if (pm.n_listings) {
      $('tb-comps').textContent = `${pm.n_listings} listings, ${pm.n_groups} specs`;
      $('tb-cov').replaceChildren(
        el('b', null, `${(pm.coverage_80 * 100).toFixed(1)}%`),
        document.createTextNode(` of ${pm.coverage_n}`));
      $('tb-cov').title =
        `The 80% comparable-asking band held the real asking price in `
        + `${(pm.coverage_80 * 100).toFixed(1)}% of ${pm.coverage_n} held-out `
        + `evaluations. Fit R² ${pm.r2}, median error ${pm.median_ape}%.`;
    }
  } catch {
    $('tb-vision').textContent = 'unreachable';
    $('tb-vision').className = 'mono broken';
  }
}

async function loadSamples() {
  const list = $('sample-list');
  try {
    const cases = await net.getSamples();
    list.replaceChildren(...cases.map((c) => {
      const b = el('button', 'sample');
      b.type = 'button';
      b.dataset.expect = c.expect;
      b.disabled = !c.available;
      b.append(el('span', 'sample-n', c.available ? String(c.n_photos) : '—'),
               el('span', 'sample-title', c.title),
               el('span', 'sample-blurb', c.available
                 ? c.blurb
                 : 'fixtures missing — run python -m app.demo --build'));
      b.addEventListener('click', () => runSample(c));
      return b;
    }));
  } catch {
    list.replaceChildren(el('p', 'sample-blurb', 'Could not load the rehearsed cases.'));
  }
}

/* ---------- intake ---------- */

function declaredParams() {
  const p = new URLSearchParams();
  const set = (k, v) => { if (v) p.set(k, v); };
  set('year', $('f-year').value);
  set('km', $('f-km').value);
  set('make', $('f-make').value.trim());
  set('asking', $('f-asking').value);
  p.set('market', $('f-market').value);
  return p;
}

function showSkipped(skipped) {
  const box = $('skipped');
  if (!skipped || !skipped.length) { box.hidden = true; return; }
  box.hidden = false;
  box.textContent = skipped.length === 1
    ? `${skipped[0].name} was not used: ${skipped[0].why}.`
    : `${skipped.length} files were not used: `
      + skipped.map((s) => `${s.name} (${s.why})`).join(', ') + '.';
}

async function runSample(c) {
  try {
    const data = await net.uploadSample(c.id);
    const d = c.declared || {};
    $('f-year').value = d.year || '';
    $('f-km').value = d.km || '';
    $('f-make').value = d.make || '';
    $('f-asking').value = d.asking_price || '';
    showSkipped(null);
    start(data.session);
  } catch (e) { fail(e.message); }
}

async function uploadFiles(files) {
  try {
    const data = await net.uploadFiles(files);
    showSkipped(data.skipped);
    start(data.session);
  } catch (e) { fail(e.message); }
}

function fail(message) {
  $('masthead-state').textContent = message;
  $('masthead-state').className = 'masthead-state broken';
}

/* ---------- the run ---------- */

function start(session) {
  if (stream) stream.close();
  run.begin();
  stream = net.openStream(session, declaredParams(), {
    gate: (m) => run.onGate(m),
    stage: (m) => run.onStage(m),
    result: (m) => { stream.close(); render(m.appraisal); },
    error: (m) => {
      stream.close();
      run.onError('Something broke while appraising.');
      showRefusal('Something broke while appraising.', m.message);
    },
  });
}

/* ---------- the verdict ---------- */

function showRefusal(headline, detail, evidence) {
  $('result').hidden = false;
  $('headline').textContent = headline;
  $('headline-usd').hidden = true;
  $('gauge-wrap').hidden = true;
  const box = $('refusal');
  box.hidden = false;
  box.replaceChildren(el('p', null, detail));
  if (evidence) box.append(el('p', 'refusal-evidence', evidence));
}

function render(a) {
  frames.setSource(a.photo_urls, a.gate.photos, a.gate.decision);
  $('result').hidden = false;
  $('refusal').hidden = true;
  $('gauge-wrap').hidden = true;
  $('headline-usd').hidden = true;
  $('headline').textContent = a.headline;

  run.onResult(a);
  frames.renderGrid(a.gate, frames.openLightbox);

  const ev = a.evidence, price = a.price;

  const fb = $('fallback-note');
  if (ev && ev.fell_back_from && ev.fell_back_from.length) {
    fb.hidden = false;
    fb.replaceChildren(
      document.createTextNode('Answered by '),
      el('b', null, `${ev.backend} / ${ev.model}`),
      document.createTextNode(` after ${ev.fell_back_from.length} backend(s) failed: `
                              + ev.fell_back_from[0]));
  } else fb.hidden = true;

  const askEl = $('asking-verdict');
  if (price && price.ok && price.asking) {
    const v = price.asking;
    askEl.hidden = false;
    askEl.className = 'asking-verdict '
      + (v.inside_comparable_band ? 'inline' : (v.vs_comparables_pct > 0 ? 'above' : 'below'));
    askEl.replaceChildren(
      document.createTextNode('The seller is asking '),
      el('b', null, money(v.asking, v.currency)),
      document.createTextNode(' — '),
      el('b', null, v.label),
      document.createTextNode(`. ${v.summary}`));
  } else askEl.hidden = true;

  if (a.status === 'refused') {
    showRefusal(a.headline,
                'No price was produced. The gate stopped this before any vision call.',
                a.gate.truck_evidence);
  } else if (price && price.ok) {
    $('gauge-wrap').hidden = false;
    drawGauge(price);
    writeGaugeNote(price);
    writeUsdBand(price);
  } else if (price && !price.ok) {
    showRefusal(a.headline, price.reason, a.gate.truck_evidence);
  } else {
    showRefusal(a.headline, a.gate.headline, a.gate.truck_evidence);
  }

  renderPanels(a, run.elevationRoot());
  stampTitleBlock(a);
  $('result').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
}

/* The same band a second time, in US dollars. The gauge and the headline are
   in Turkish lira - the market the model was fit on - but a used tractor is a
   cross-border purchase and a buyer thinks in both. The dollar figures already
   travel on the wire (price.*_usd, converted at the one stamped FX rate); this
   only renders them, with the rate and its date shown so the conversion is
   checkable rather than implied. A US-market appraisal is already in dollars,
   so the line is suppressed there. */
function writeUsdBand(price) {
  const box = $('headline-usd');
  if (price.currency !== 'TRY' || !price.low_usd || !price.high_usd) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const fx = (price.model_card && price.model_card.fx) || null;
  const range = `${money(price.low_usd, 'USD')} – ${money(price.high_usd, 'USD')}`;
  const parts = [
    document.createTextNode('≈ '),
    el('b', null, range),
    el('span', 'usd-point', `  ·  ${money(price.point_usd, 'USD')} point`),
  ];
  if (fx) parts.push(el('span', 'usd-fx', `   ₺${fx.usd_try}/$ · ${fx.as_of}`));
  box.replaceChildren(...parts);
}

/* The calibration figure is pinned to the band it was measured on, and says
   so in words, because the adjusted band carries no such guarantee. */
function writeGaugeNote(price) {
  const card = price.model_card || {};
  const note = $('gauge-note');
  const parts = [];
  const push = (s) => parts.push(document.createTextNode(s));
  const fig = (s) => parts.push(el('b', null, s));

  if (Math.abs(price.point - price.baseline_point) > 1) {
    push('What the photos found moved the estimate ');
    fig(`${price.adjustment.pct >= 0 ? '+' : ''}${fixed(price.adjustment.pct, 1)}%`);
    push(`, capped at ±${fixed(price.adjustment.cap_pct, 1)}% — one residual standard `
         + 'deviation of the price model, which is a stated assumption. ');
  }
  push('The comparable-asking band is an ');
  push(`${Math.round(price.interval_level * 100)}% interval, and that is the band `
       + 'whose accuracy was measured: on held-out listings it contained the real '
       + 'asking price ');
  fig(`${(card.coverage * 100).toFixed(1)}%`);
  push(` of the time over ${card.coverage_n} evaluations. Fit R² ${card.r2}, `
        + `median error ${card.mae_pct}% across ${card.n_listings} listings `
        + `collapsing to ${card.n_groups} distinct specs.`);
  note.replaceChildren(...parts);
}

function stampTitleBlock(a) {
  const v = a.evidence ? a.evidence.vehicle : null;
  if (v && (v.make || v.model)) {
    const bits = [[v.make, v.model].filter(Boolean).join(' ')];
    const spec = [v.cab_type, v.axle_config].filter(Boolean).join(' ');
    if (spec) bits.push(spec);
    if (v.approx_year_range) bits.push(v.approx_year_range);
    $('tb-subject').textContent = bits.join(', ');
  }
  const fx = a.price && a.price.model_card ? a.price.model_card.fx : null;
  if (fx) $('tb-fx').textContent = `${fx.usd_try} TRY/USD, ${fx.as_of}`;
  if (a.version) $('tb-sheet').textContent = a.version;
}

/* ---------- wiring ---------- */

const dz = $('dropzone');
dz.addEventListener('click', () => $('filepicker').click());
dz.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('filepicker').click(); }
});
$('filepicker').addEventListener('change', (e) => {
  if (e.target.files.length) uploadFiles(e.target.files);
});
['dragenter', 'dragover'].forEach((type) => dz.addEventListener(type, (e) => {
  e.preventDefault(); dz.classList.add('over');
}));
['dragleave', 'drop'].forEach((type) => dz.addEventListener(type, (e) => {
  e.preventDefault(); dz.classList.remove('over');
}));
dz.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

$('reset').addEventListener('click', () => {
  if (stream) stream.close();
  run.stop();
  $('run').hidden = true;
  $('result').hidden = true;
  elevation.reset(run.elevationRoot());
  window.scrollTo({ top: 0, behavior: reduced() ? 'auto' : 'smooth' });
});

$('lightbox-close').addEventListener('click', () => { $('lightbox').hidden = true; });
$('lightbox').addEventListener('click', (e) => {
  if (e.target === $('lightbox')) $('lightbox').hidden = true;
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') $('lightbox').hidden = true;
});

/* A frozen export embeds the appraisal and has no API behind it: render it
 * straight away and drop the parts that would call a server. The drawing, the
 * frames and the findings all still work, which is the point - a judge can
 * click through it when the live run has failed. */
if (window.KAMION_APPRAISAL) {
  $('intake').hidden = true;
  run.mount().then(() => {
    run.showFrozen(window.KAMION_APPRAISAL);
    render(window.KAMION_APPRAISAL);
  });
} else {
  run.mount();
  loadHealth();
  loadSamples();
}
