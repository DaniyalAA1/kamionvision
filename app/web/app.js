/* KamionVision demo screen.
 *
 * One job: make the pipeline legible while it runs. The gate finishes in a
 * couple of seconds and the vision call takes half a minute, so the run is
 * streamed over SSE and each stage lands as it completes rather than the
 * screen sitting blank until a price exists.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const SYMBOL = { TRY: '₺', USD: '$', EUR: '€' };
const money = (v, cur) => (SYMBOL[cur] || cur + ' ') + Math.round(v).toLocaleString('en-US');
const titleise = (s) => s.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

let session = null;
let stream = null;
let photoUrls = {};

/* ---------- rig status ---------- */

async function loadHealth() {
  try {
    const h = await (await fetch('/api/health')).json();
    const ready = h.backends.find((b) => b.ready);
    const backendEl = $('rig-backend');
    if (ready) {
      backendEl.textContent = `${ready.name} · ${ready.model}`;
      backendEl.className = 'ready';
    } else {
      const blocked = h.backends.find((b) => b.account_blocked);
      backendEl.textContent = blocked
        ? `${blocked.name} account blocked` : 'no backend configured';
      backendEl.className = 'broken';
      backendEl.title = h.backends.map((b) => `${b.name}: ${b.detail}`).join('\n');
    }
    const pm = h.price_model || {};
    if (pm.n_listings) {
      $('rig-comps').innerHTML =
        `<b>${pm.n_listings}</b> listings · ${pm.n_groups} specs`;
      $('rig-cov').innerHTML =
        `80% band held <b>${(pm.coverage_80 * 100).toFixed(1)}%</b>`;
      $('rig-cov').title = `Measured on ${pm.coverage_n} held-out evaluations. `
        + `R² ${pm.r2}, median error ${pm.median_ape}%.`;
    }
  } catch (e) { $('rig-backend').textContent = 'unreachable'; }
}

async function loadSamples() {
  const list = $('sample-list');
  try {
    const cases = await (await fetch('/api/samples')).json();
    list.replaceChildren(...cases.map((c) => {
      const b = el('button', 'sample');
      b.type = 'button';
      b.dataset.expect = c.expect;
      b.disabled = !c.available;
      b.append(el('span', 'sample-n', c.available ? `${c.n_photos}` : '—'),
               el('span', 'sample-title', c.title),
               el('span', 'sample-blurb',
                  c.available ? c.blurb : 'fixtures missing — run python -m app.demo --build'));
      b.addEventListener('click', () => runSample(c));
      return b;
    }));
  } catch (e) {
    list.replaceChildren(el('p', 'sample-blurb', 'Could not load the rehearsed cases.'));
  }
}

/* ---------- intake ---------- */

function declaredParams() {
  const p = new URLSearchParams();
  const year = $('f-year').value, km = $('f-km').value, make = $('f-make').value.trim();
  if (year) p.set('year', year);
  if (km) p.set('km', km);
  if (make) p.set('make', make);
  p.set('market', $('f-market').value);
  return p;
}

async function runSample(c) {
  const body = new FormData();
  body.set('case', c.id);
  const r = await fetch('/api/upload-sample', { method: 'POST', body });
  if (!r.ok) { alert((await r.json()).detail); return; }
  const data = await r.json();
  const d = c.declared || {};
  $('f-year').value = d.year || '';
  $('f-km').value = d.km || '';
  $('f-make').value = d.make || '';
  start(data.session);
}

async function uploadFiles(files) {
  const body = new FormData();
  [...files].forEach((f) => body.append('files', f));
  const r = await fetch('/api/upload', { method: 'POST', body });
  if (!r.ok) { alert((await r.json()).detail); return; }
  start((await r.json()).session);
}

/* ---------- run ---------- */

function setStage(step, state, detail, time) {
  const li = document.querySelector(`.stages li[data-step="${step}"]`);
  if (!li) return;
  li.dataset.state = state;
  if (detail != null) li.querySelector('.stage-detail').textContent = detail;
  li.querySelector('.stage-time').textContent = time != null ? `${time.toFixed(2)}s` : '';
}

function start(sid) {
  session = sid;
  if (stream) stream.close();
  $('run').hidden = false;
  $('result').hidden = true;
  ['gate', 'evidence', 'price'].forEach((s) => setStage(s, 'waiting', 'waiting', null));
  $('run').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  stream = new EventSource(`/api/appraise/${sid}?${declaredParams()}`);
  stream.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'stage') setStage(msg.step, 'running', msg.detail, null);
    else if (msg.type === 'result') { stream.close(); render(msg.appraisal); }
    else if (msg.type === 'error') {
      stream.close();
      ['gate', 'evidence', 'price'].forEach((s) => setStage(s, 'skipped', '', null));
      showRefusal('Something broke while appraising.', msg.message);
    }
  };
  stream.onerror = () => { if (stream) stream.close(); };
}

/* ---------- the gauge ---------- */

const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

/* The price band drawn as a measuring scale: the comparables it was fit
 * against sit on the same axis as ticks, so "what is it comparing to" is
 * answered by the picture rather than by a sentence under it. */
function drawGauge(price) {
  const g = $('gauge');
  g.replaceChildren();
  const W = 1000, axisY = 108;
  const comps = (price.comparables || [])
    .map((c) => (price.currency === 'TRY' ? c.price_try : c.price_usd) ?? c.price)
    .filter((v) => v != null);

  const lo = Math.min(price.low, ...comps), hi = Math.max(price.high, ...comps);
  const pad = (hi - lo) * 0.12 || hi * 0.1;
  const min = lo - pad, max = hi + pad;
  const x = (v) => 40 + ((v - min) / (max - min)) * (W - 80);

  g.append(svg('line', { x1: 40, y1: axisY, x2: W - 40, y2: axisY,
                         stroke: 'var(--steel-700)', 'stroke-width': 1 }));

  // the band
  g.append(svg('rect', { x: x(price.low), y: axisY - 26, width: x(price.high) - x(price.low),
                         height: 52, fill: 'var(--sodium)', opacity: .14 }));
  for (const v of [price.low, price.high]) {
    g.append(svg('line', { x1: x(v), y1: axisY - 26, x2: x(v), y2: axisY + 26,
                           stroke: 'var(--sodium)', 'stroke-width': 2 }));
    const t = svg('text', { x: x(v), y: axisY - 36, fill: 'var(--sodium)',
                            'text-anchor': 'middle', 'font-family': 'var(--cond)',
                            'font-size': 30, 'font-weight': 600 });
    t.textContent = money(v, price.currency);
    g.append(t);
  }

  // the estimate before condition, as a ghost, so the adjustment is visible
  if (price.baseline_point && Math.abs(price.baseline_point - price.point) > (max - min) / 200) {
    g.append(svg('line', { x1: x(price.baseline_point), y1: axisY - 16,
                           x2: x(price.baseline_point), y2: axisY + 16,
                           stroke: 'var(--haze-dim)', 'stroke-width': 1.5,
                           'stroke-dasharray': '3 3' }));
    const t = svg('text', { x: x(price.baseline_point), y: axisY + 34,
                            fill: 'var(--haze-dim)', 'text-anchor': 'middle',
                            'font-family': 'var(--cond)', 'font-size': 17 });
    t.textContent = 'before condition';
    g.append(t);
  }

  // the needle
  const nx = x(price.point);
  const needle = svg('path', { d: `M ${nx} ${axisY - 34} L ${nx - 9} ${axisY - 50} L ${nx + 9} ${axisY - 50} Z`,
                               fill: 'var(--paper)' });
  g.append(needle);
  g.append(svg('line', { x1: nx, y1: axisY - 34, x2: nx, y2: axisY + 34,
                         stroke: 'var(--paper)', 'stroke-width': 2 }));

  // comparables as ticks on the same scale
  (price.comparables || []).forEach((c) => {
    const v = (price.currency === 'TRY' ? c.price_try : c.price_usd) ?? c.price;
    if (v == null) return;
    g.append(svg('line', { x1: x(v), y1: axisY + 30, x2: x(v), y2: axisY + 46,
                           stroke: 'var(--signal)', 'stroke-width': 2 }));
    const t = svg('text', { x: x(v), y: axisY + 66, fill: 'var(--signal)',
                            'text-anchor': 'middle', 'font-family': 'var(--cond)',
                            'font-size': 18 });
    t.textContent = `${c.year} · ${Math.round(c.km / 1000)}k`;
    g.append(t);
  });
  const legend = svg('text', { x: 40, y: axisY + 86, fill: 'var(--haze-dim)',
                               'font-family': 'var(--cond)', 'font-size': 17 });
  legend.textContent = 'teal ticks are the real listings this was priced against';
  g.append(legend);

  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    g.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }],
              { duration: 420, easing: 'cubic-bezier(.2,.7,.3,1)' });
  }
}

/* ---------- render ---------- */

function showRefusal(headline, detail, evidence) {
  $('result').hidden = false;
  $('headline').textContent = headline;
  $('gauge-wrap').hidden = true;
  const box = $('refusal');
  box.hidden = false;
  box.replaceChildren(el('p', null, detail));
  if (evidence) box.append(el('p', 'refusal-evidence', evidence));
}

function renderPhotos(a) {
  const grid = $('photo-grid');
  const checks = a.gate.photos || [];
  $('photo-count').textContent =
    `${a.gate.usable_photo_ids.length} used of ${checks.length}`;
  grid.replaceChildren(...checks.map((c) => {
    const b = el('button', 'thumb' + (c.usable ? '' : ' dropped'));
    b.type = 'button';
    b.id = `thumb-${c.photo_id}`;
    b.title = c.usable
      ? `${c.filename} — ${titleise(c.view)} (${c.quality_bucket})`
      : `${c.filename} — dropped: ${c.reasons.join('; ')}`;
    const img = el('img');
    img.src = photoUrls[c.photo_id] || '';
    img.alt = c.usable ? `${titleise(c.view)}` : `dropped: ${c.reasons.join('; ')}`;
    img.loading = 'lazy';
    b.append(img, el('span', 'thumb-tag',
                     c.usable ? titleise(c.view) : 'dropped'));
    b.addEventListener('click', () => openLightbox(c));
    return b;
  }));
}

function citePhoto(id, a) {
  const check = (a.gate.photos || []).find((c) => c.photo_id === id);
  const node = $(`thumb-${id}`);
  if (!node) return;
  node.classList.remove('flash');
  void node.offsetWidth;
  node.classList.add('flash');
  node.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  if (check) openLightbox(check);
}

function openLightbox(check) {
  $('lightbox-img').src = photoUrls[check.photo_id] || '';
  $('lightbox-img').alt = titleise(check.view);
  $('lightbox-cap').textContent = check.usable
    ? `${check.filename} — ${titleise(check.view)} · ${check.quality_bucket} capture`
      + (check.truck_dominant ? ` · truck detected ${check.truck_conf.toFixed(2)}` : '')
    : `${check.filename} — dropped: ${check.reasons.join('; ')}`;
  $('lightbox').hidden = false;
}

function render(a) {
  photoUrls = a.photo_urls || {};
  $('result').hidden = false;
  $('refusal').hidden = true;
  $('gauge-wrap').hidden = true;
  $('headline').textContent = a.headline;

  for (const step of a.trace || []) setStage(step.step, 'done', step.detail, step.elapsed_s);
  for (const s of ['gate', 'evidence', 'price']) {
    const li = document.querySelector(`.stages li[data-step="${s}"]`);
    if (li && li.dataset.state !== 'done') {
      li.dataset.state = 'skipped';
      li.querySelector('.stage-detail').textContent = 'not reached';
    }
  }

  renderPhotos(a);

  const ev = a.evidence, price = a.price;

  if (a.status === 'refused') {
    showRefusal(a.headline,
                'No price was produced. The gate stopped this before any vision call.',
                a.gate.truck_evidence);
  } else if (price && price.ok) {
    $('gauge-wrap').hidden = false;
    drawGauge(price);
    const card = price.model_card;
    $('gauge-note').innerHTML =
      `This is an ${Math.round(price.interval_level * 100)}% band. On held-out listings it `
      + `contained the real asking price <b>${(card.coverage * 100).toFixed(1)}%</b> of the time `
      + `(${card.coverage_n} evaluations). Fit R² ${card.r2}, median error ${card.mae_pct}% `
      + `across ${card.n_listings} listings collapsing to ${card.n_groups} distinct specs.`;
  } else if (price && !price.ok) {
    showRefusal(a.headline, price.reason, a.gate.truck_evidence);
  } else {
    showRefusal(a.headline, a.gate.headline, a.gate.truck_evidence);
  }

  // identity
  const idPanel = $('panel-identity');
  if (ev && (ev.vehicle.make || ev.vehicle.odometer_km)) {
    const v = ev.vehicle, dl = $('identity');
    dl.replaceChildren();
    const add = (k, val, read) => {
      if (!val) return;
      dl.append(el('dt', null, k));
      dl.append(el('dd', read ? 'read' : null, val));
    };
    add('Make', [v.make, v.model].filter(Boolean).join(' '), true);
    add('Cab', v.cab_type);
    add('Axles', v.axle_config);
    add('Generation', v.approx_year_range);
    if (v.odometer_km) add('Odometer', `${v.odometer_km.toLocaleString('en-US')} km`, true);
    if (v.badges_seen && v.badges_seen.length) add('Badges', v.badges_seen.join(', '));
    idPanel.hidden = false;
  } else idPanel.hidden = true;

  // findings
  const fPanel = $('panel-findings');
  if (ev && ev.issues.length) {
    $('findings-sub').textContent =
      `Condition graded ${ev.condition_grade}, confidence ${ev.confidence.toFixed(2)}. `
      + `Every line names the photo it came from — select one to see it.`;
    const order = { major: 0, moderate: 1, minor: 2, cosmetic: 3 };
    const list = [...ev.issues].sort((x, y) => order[x.severity] - order[y.severity]);
    $('findings').replaceChildren(...list.map((i) => {
      const li = el('li');
      const b = el('button', 'finding');
      b.type = 'button';
      b.dataset.sev = i.severity;
      const part = el('span', 'finding-part', titleise(i.component) + ' ');
      part.append(el('em', null, `${i.severity} · ${i.price_impact} price impact`));
      const check = (a.gate.photos || []).find((c) => c.photo_id === i.photo_id);
      const cite = el('span', 'finding-cite');
      cite.append(document.createTextNode('seen in '));
      cite.append(el('b', null, check ? check.filename : `photo ${i.photo_id}`));
      cite.append(document.createTextNode(` · confidence ${i.confidence.toFixed(2)}`));
      b.append(el('span', 'finding-bar'), part,
               el('span', 'finding-text', i.observation), cite);
      b.addEventListener('click', () => citePhoto(i.photo_id, a));
      li.append(b);
      return li;
    }));
    fPanel.hidden = false;
  } else fPanel.hidden = true;

  // systems
  const sPanel = $('panel-systems');
  if (ev && Object.keys(ev.condition_summary).length) {
    const dl = $('systems');
    dl.replaceChildren();
    for (const [k, v] of Object.entries(ev.condition_summary)) {
      dl.append(el('dt', null, titleise(k)), el('dd', null, v));
    }
    sPanel.hidden = false;
  } else sPanel.hidden = true;

  // asks + gaps
  const asks = a.requests || [];
  $('panel-asks').hidden = !asks.length;
  $('asks').replaceChildren(...asks.map((t) => el('li', null, t)));
  const gaps = ev ? ev.coverage_gaps : [];
  $('panel-gaps').hidden = !gaps.length;
  $('gaps').replaceChildren(...gaps.map((t) => el('li', null, t)));

  // why
  const wPanel = $('panel-why');
  if (price && price.ok) {
    const dt = $('drivers');
    dt.replaceChildren();
    const head = el('tr');
    head.append(el('th', null, 'Factor'), el('th', 'num', 'Value'),
                el('th', 'num', 'Effect vs. the average comparable'));
    dt.append(head);
    for (const d of price.drivers.slice(0, 6)) {
      const tr = el('tr');
      tr.append(el('td', null, titleise(d.feature)),
                el('td', 'num', d.value.toFixed(2)),
                el('td', 'num ' + (d.pct_effect >= 0 ? 'pos' : 'neg'),
                   `${d.pct_effect >= 0 ? '+' : ''}${d.pct_effect.toFixed(1)}%`));
      dt.append(tr);
    }
    if (price.adjustment && price.adjustment.pct) {
      const tr = el('tr');
      tr.append(el('td', null, 'Condition (from the photos)'), el('td', 'num', ''),
                el('td', 'num ' + (price.adjustment.pct >= 0 ? 'pos' : 'neg'),
                   `${price.adjustment.pct >= 0 ? '+' : ''}${price.adjustment.pct.toFixed(1)}%`));
      tr.title = price.adjustment.basis;
      dt.append(tr);
    }

    const ct = $('comps');
    ct.replaceChildren();
    const ch = el('tr');
    ch.append(el('th', null, 'Year'), el('th', null, 'Listing'),
              el('th', 'num', 'km'), el('th', 'num', 'Asking'));
    ct.append(ch);
    for (const c of price.comparables) {
      const tr = el('tr');
      tr.append(el('td', null, String(c.year)),
                el('td', null, `${c.make} ${c.model}`),
                el('td', 'num', `${Math.round(c.km / 1000)}k`),
                el('td', 'num price', money(c.price, c.currency)));
      ct.append(tr);
    }

    const prov = $('provenance');
    prov.replaceChildren();
    for (const [k, v] of Object.entries(price.inputs_provenance || {})) {
      const row = el('div', k.includes('conflict') ? 'conflict' : null);
      row.append(el('span', null, titleise(k)), el('span', null, v));
      prov.append(row);
    }
    $('widened').replaceChildren(...(price.widened || []).map((w) => el('li', null, w)));
    wPanel.hidden = false;
  } else wPanel.hidden = true;

  const cav = price ? price.caveats || [] : [];
  $('panel-caveats').hidden = !cav.length;
  $('caveats').replaceChildren(...cav.map((t) => el('li', null, t)));

  $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
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
['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => {
  e.preventDefault(); dz.classList.add('over');
}));
['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => {
  e.preventDefault(); dz.classList.remove('over');
}));
dz.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

$('reset').addEventListener('click', () => {
  if (stream) stream.close();
  $('run').hidden = true;
  $('result').hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

$('lightbox-close').addEventListener('click', () => { $('lightbox').hidden = true; });
$('lightbox').addEventListener('click', (e) => {
  if (e.target === $('lightbox')) $('lightbox').hidden = true;
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') $('lightbox').hidden = true;
});

loadHealth();
loadSamples();
