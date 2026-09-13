/* Entry point: wiring, the verdict, and the frozen-export path.

   The run is streamed. The gate report arrives about a second in, and every
   photo arrives as its own event while the rest are still being read - see
   js/run.js and js/reasoning.js. */

import { $, el, money, fixed, reduced } from './js/dom.js';
import * as net from './js/net.js';
import * as run from './js/run.js';
import * as frames from './js/frames.js';
import * as gallery from './js/gallery.js';
import * as elevation from './js/elevation.js';
import { drawBand } from './js/band.js';
import { renderPanels } from './js/panels.js';

let stream = null;

/* ---------- boot ---------- */

/* The top bar is silent when everything works. It is not a status dashboard;
   the only thing a person needs from it is to be told when the thing they are
   about to press will not work. */
async function loadHealth() {
  try {
    const h = await net.getHealth();
    const ready = h.backends.find(
      (b) => b.ready && (!h.selected_backend || b.name === h.selected_backend));
    if (ready) { $('topbar-state').textContent = ''; return; }
    const blocked = h.backends.find((b) => b.account_blocked);
    $('topbar-state').textContent = blocked
      ? `Vision is unavailable — ${blocked.detail}`
      : 'No vision model is configured, so photos can be checked but not read.';
    $('topbar-state').className = 'topbar-state broken';
  } catch {
    $('topbar-state').textContent = 'Cannot reach the server.';
    $('topbar-state').className = 'topbar-state broken';
  }
}

async function loadGallery() {
  try {
    const data = await net.getGallery();
    gallery.render(data, (card) => pick(card, data.declared || {}));
  } catch {
    gallery.fail('Could not load the trucks.');
  }
}

/* ---------- starting a run ---------- */

function declaredParams() {
  const p = new URLSearchParams();
  const set = (k, v) => { if (v) p.set(k, v); };
  set('year', $('f-year').value);
  set('km', $('f-km').value);
  set('make', $('f-make').value.trim());
  set('asking', $('f-asking').value);
  /* One market. The price model is fitted on Turkish listings, so a truck from
     anywhere is priced in lira and an unfamiliar make widens the range and
     says so - which beats a dropdown that let someone ask a Turkish fit for a
     number in dollars. */
  p.set('market', 'TR');
  return p;
}

function fillDeclared(d) {
  $('f-year').value = d.year || '';
  $('f-km').value = d.km ? Math.round(d.km) : '';
  $('f-make').value = d.make || '';
  $('f-asking').value = d.asking_price || '';
}

async function pick(card, declaredByCase) {
  try {
    const data = card.demo
      ? await net.uploadSample(card.case_id)
      : await net.uploadTruck(card.id);
    fillDeclared(card.demo ? (declaredByCase[card.case_id] || {}) : (data.declared || {}));
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

function showSkipped(skipped) {
  const box = $('skipped');
  if (!skipped || !skipped.length) { box.hidden = true; return; }
  box.hidden = false;
  box.textContent = skipped.length === 1
    ? `${skipped[0].name} was not used: ${skipped[0].why}.`
    : `${skipped.length} files were not used: `
      + skipped.map((s) => `${s.name} (${s.why})`).join(', ') + '.';
}

function fail(message) {
  $('topbar-state').textContent = message;
  $('topbar-state').className = 'topbar-state broken';
}

function start(session) {
  if (stream) stream.close();
  $('intake').hidden = true;
  run.begin();
  stream = net.openStream(session, declaredParams(), {
    gate: (m) => run.onGate(m),
    stage: (m) => run.onStage(m),
    photo: (m) => run.onPhoto(m),
    activity: (m) => run.onActivity(m),
    result: (m) => { stream.close(); render(m.appraisal); },
    error: (m) => {
      stream.close();
      run.onError('Something broke while appraising.');
      showRefusal('Something broke while appraising.', m.message);
      $('intake').hidden = false;
    },
  });
}

/* ---------- the answer ---------- */

/* `pipeline.pricing_blocker` returns a headline and a reason that deliberately
   share their explanatory tail, because the CLI prints only one of them. On
   screen both are shown, so printing them in full says the same sentence
   twice. */
const normalise = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

function showRefusal(headline, detail) {
  $('result').hidden = false;
  $('price').textContent = '';
  $('estimate-label').hidden = true;
  $('price-sub').textContent = '';
  $('price-usd').hidden = true;
  $('verdict-kicker').textContent = '';
  $('band-wrap').hidden = true;
  const box = $('refusal');
  box.hidden = false;
  box.replaceChildren(el('p', null, headline));
  const a = normalise(headline), b = normalise(detail);
  if (b && b !== a && !a.includes(b) && !b.includes(a)) box.append(el('p', null, detail));
}

function priceLine(price) {
  const wrap = $('price');
  $('estimate-label').hidden = false;
  wrap.replaceChildren(
    el('span', 'price-bound', money(price.low, price.currency)),
    el('span', 'dash', '–'),
    el('span', 'price-bound', money(price.high, price.currency)));
}

function render(a) {
  frames.setSource(a.photo_urls, a.gate.photos, a.gate.decision);
  (a.evidence?.photo_findings || []).forEach(frames.setFinding);
  $('result').hidden = false;
  $('refusal').hidden = true;
  $('band-wrap').hidden = true;
  $('price-usd').hidden = true;

  run.onResult(a);
  frames.showFrame(frames.checkFor(frames.currentPhotoId())
    || (a.gate.decision === 'refuse_not_a_truck' && frames.smokingGun())
    || a.gate.photos.find((photo) => photo.usable) || a.gate.photos[0]);
  frames.renderGrid(a.gate, frames.openLightbox);

  const ev = a.evidence, price = a.price;

  /* Who the truck is, before what it is worth. */
  const v = ev ? ev.vehicle : null;
  const named = v ? [v.make, v.model].filter(Boolean).join(' ') : '';
  const kicker = [];
  if (named) kicker.push(named);
  if (v && v.odometer_km) {
    kicker.push(`${v.odometer_km.toLocaleString('en-US')} km on the clock`);
  }
  $('verdict-kicker').textContent = kicker.join(', ');

  /* Falling back is allowed. Doing it quietly is not - so it stays on the
     answer, in words, rather than moving into the disclosure with the rest of
     the machinery. */
  const fb = $('fallback');
  if (ev && ev.fell_back_from && ev.fell_back_from.length) {
    fb.hidden = false;
    fb.textContent = 'One of the vision providers was unavailable, so a backup '
      + 'read the photos instead. The details are under "How I worked this out".';
  } else fb.hidden = true;

  const askEl = $('asking');
  if (price && price.ok && price.asking) {
    const ask = price.asking;
    askEl.hidden = false;
    askEl.className = 'asking '
      + (ask.inside_comparable_band ? 'inline' : (ask.vs_comparables_pct > 0 ? 'above' : 'below'));
    askEl.replaceChildren(
      document.createTextNode('The seller is asking '),
      el('b', null, money(ask.asking, ask.currency)),
      document.createTextNode(`, which is ${ask.label}. ${ask.summary}`));
  } else askEl.hidden = true;

  if (a.status === 'refused') {
    showRefusal(a.headline,
                'No photo was sent to a vision model. The checks that run first — '
                + 'on your machine, in about a second — stopped this before anything '
                + 'was spent on it.');
  } else if (price && price.ok) {
    $('band-wrap').hidden = false;
    priceLine(price);
    $('price-sub').textContent = subLine(a, ev, price);
    drawBand($('band'), price);
    $('band-note').textContent = bandNote(price);
    writeUsdBand(price);
  } else if (price && !price.ok) {
    showRefusal(a.headline, price.reason);
  } else {
    showRefusal(a.headline, a.gate.headline);
  }

  renderPanels(a, run.elevationRoot());
  // A reader inspecting an earlier photo keeps their place when the answer lands.
  if (run.isFollowing() && !run.hasPendingReview()) {
    $('result').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
  }
}

/* The same band a second time, in US dollars. The price and the bar are in
   Turkish lira - the market the model was fit on - but a used tractor is a
   cross-border purchase and a buyer thinks in both. The dollar figures already
   travel on the wire (price.*_usd, converted at the one stamped FX rate); this
   only renders them, with the rate and its date shown so the conversion is
   checkable rather than implied. A US-market appraisal is already in dollars,
   so the line is suppressed there. */
function writeUsdBand(price) {
  const box = $('price-usd');
  if (price.currency !== 'TRY' || !price.low_usd || !price.high_usd) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const fx = (price.model_card && price.model_card.fx) || null;
  const parts = [
    document.createTextNode('About '),
    el('b', null, `${money(price.low_usd, 'USD')} – ${money(price.high_usd, 'USD')}`),
  ];
  if (fx) {
    parts.push(el('span', 'usd-fx', ` at ₺${fx.usd_try} to the dollar, ${fx.as_of}`));
  }
  box.replaceChildren(...parts);
}

function subLine(a, ev, price) {
  const grade = ev ? ev.condition_grade : 'unknown';
  const read = ev ? ev.photos_read : 0;
  const bits = [];
  if (grade && grade !== 'unknown') bits.push(`Condition ${grade}`);
  if (read) bits.push(`read from ${read} photos, one at a time`);
  if (a.status === 'ok_with_requests') {
    bits.push('and the range would narrow with the photos listed below');
  }
  return bits.join(', ') + '.';
}

/* The two bands are not interchangeable, so the note that explains them says
   which is which without quoting the measured figure for the one it does not
   belong to. That number lives in the disclosure, attached to its own band. */
function bandNote(price) {
  if (Math.abs(price.point - price.baseline_point) <= 1) {
    return 'This is what comparable trucks of this age and mileage are being '
      + 'asked for. Nothing in the photos moved it.';
  }
  const dir = price.adjustment.pct >= 0 ? 'up' : 'down';
  const why = price.adjustment.pct > 0
    /* A number above the comparables is surprising, so it is explained where
       it is shown rather than only inside the disclosure. The baseline is
       average-condition ASKING prices, so a truck that is demonstrably better
       than average belongs above it - and only a truck graded excellent gets
       there. */
    ? ' The comparables are what an average-condition truck is asked for, and '
      + 'this one was photographed well enough to show it is better than that.'
    : '';
  return `What the photos found moved the estimate ${dir} `
    + `${fixed(Math.abs(price.adjustment.pct), 1)}% from what comparable trucks `
    + 'are being asked for.' + why;
}

/* ---------- wiring ---------- */

/* The upload panel. The gallery is the screen now, so the drop target lives
   one press behind a button - but it is still the path a judge's own photos
   take, so it opens in place, focuses itself, and the seller fields inside it
   keep feeding `declaredParams()` whether the panel is open or shut. */
const own = $('own-toggle');
own.addEventListener('click', () => {
  const open = own.getAttribute('aria-expanded') !== 'true';
  own.setAttribute('aria-expanded', String(open));
  $('own-panel').hidden = !open;
  if (open) $('dropzone').focus();
});

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

$('show-all').addEventListener('click', (e) => {
  const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
  e.currentTarget.setAttribute('aria-pressed', String(on));
  e.currentTarget.textContent = on ? 'Show only the truck being sold'
                                   : 'Show everything it detected';
  frames.setShowAll(on);
});

function startOver() {
  if (stream) stream.close();
  run.stop();
  $('intake').hidden = false;
  $('run').hidden = true;
  $('result').hidden = true;
  elevation.reset(run.elevationRoot());
  window.scrollTo({ top: 0, behavior: reduced() ? 'auto' : 'smooth' });
}
$('reset').addEventListener('click', startOver);
$('again').addEventListener('click', startOver);

$('lightbox-close').addEventListener('click', () => { $('lightbox').hidden = true; });
$('lightbox').addEventListener('click', (e) => {
  if (e.target === $('lightbox')) $('lightbox').hidden = true;
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') $('lightbox').hidden = true;
  if (!$('run').hidden && $('lightbox').hidden && !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      frames.stepPhoto(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      frames.stepPhoto(1);
    }
  }
});

/* A frozen export embeds the appraisal and has no API behind it: render it
 * straight away and drop the parts that would call a server. The drawing, the
 * frames, the per-photo rail and the findings all still work, which is the
 * point - a judge can click through it when the live run has failed. */
if (window.KAMION_APPRAISAL) {
  $('intake').hidden = true;
  run.mount().then(() => {
    run.showFrozen(window.KAMION_APPRAISAL);
    render(window.KAMION_APPRAISAL);
  });
} else {
  run.mount();
  loadHealth();
  loadGallery();
}
