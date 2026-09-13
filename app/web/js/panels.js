/* The seven plates, in the order the report card uses them. Content and copy
   are carried over unchanged - this module only draws them. */

import { $, el, money, titleise, fixed, kkm, animate, reduced } from './dom.js';
import * as elevation from './elevation.js';
import { citePhoto, checkFor } from './frames.js';

const SEV_ORDER = { major: 0, moderate: 1, minor: 2, cosmetic: 3 };

export function renderPanels(a, runElev) {
  const ev = a.evidence, price = a.price;

  identity(ev);
  findings(a, ev, runElev);
  systems(ev);
  asksAndGaps(a, ev);
  why(price);
  caveats(price);
}

function identity(ev) {
  const panel = $('panel-identity');
  if (!(ev && (ev.vehicle.make || ev.vehicle.odometer_km))) { panel.hidden = true; return; }
  const v = ev.vehicle, dl = $('identity');
  dl.replaceChildren();
  const add = (k, val, cls) => {
    if (!val) return;
    dl.append(el('dt', null, k), el('dd', cls, val));
  };
  add('Make', [v.make, v.model].filter(Boolean).join(' '), 'read');
  add('Cab', v.cab_type);
  add('Axles', v.axle_config);
  add('Generation', v.approx_year_range);
  if (v.odometer_km) {
    add('Odometer', `${v.odometer_km.toLocaleString('en-US')} km`, 'read fig');
  }
  if (v.badges_seen && v.badges_seen.length) add('Badges', v.badges_seen.join(', '));
  panel.hidden = false;
}

function findings(a, ev, runElev) {
  const panel = $('panel-findings');
  if (!(ev && ev.issues.length)) { panel.hidden = true; return; }
  $('findings-sub').textContent =
    `Condition graded ${ev.condition_grade}, confidence ${fixed(ev.confidence, 2)}. `
    + 'Every line names the photo it came from — select one to see it.';

  const list = [...ev.issues].sort(
    (x, y) => (SEV_ORDER[x.severity] ?? 9) - (SEV_ORDER[y.severity] ?? 9));

  const rows = list.map((i) => {
    const li = el('li');
    const b = el('button', 'finding');
    b.type = 'button';
    b.dataset.sev = i.severity;
    const part = el('span', 'finding-part', titleise(i.component));
    part.append(el('em', null, `${i.severity} · ${i.price_impact} price impact`));
    const check = checkFor(i.photo_id);
    const cite = el('span', 'finding-cite');
    cite.append(document.createTextNode('seen in '));
    cite.append(el('b', null, check ? check.filename : `photo ${i.photo_id}`));
    cite.append(document.createTextNode(` · confidence ${fixed(i.confidence, 2)}`));
    b.append(el('span', 'finding-bar'), part,
             el('span', 'finding-text', i.observation), cite);
    b.addEventListener('click', () => citePhoto(i.photo_id));
    /* hovering a finding lights the part of the drawing it is about */
    b.addEventListener('pointerenter', () => elevation.focus(runElev, i.component, true));
    b.addEventListener('pointerleave', () => elevation.focus(runElev, i.component, false));
    b.addEventListener('focus', () => elevation.focus(runElev, i.component, true));
    b.addEventListener('blur', () => elevation.focus(runElev, i.component, false));
    li.append(b);
    return li;
  });
  $('findings').replaceChildren(...rows);
  panel.hidden = false;
  if (!reduced()) {
    rows.forEach((r, i) => animate(r, { opacity: [0, 1], transform: ['translateY(6px)', 'none'] },
                                   { duration: 0.35, delay: 0.05 * i, ease: [0.16, 1, 0.3, 1] }));
  }
}

function systems(ev) {
  const panel = $('panel-systems');
  if (!(ev && Object.keys(ev.condition_summary).length)) { panel.hidden = true; return; }
  const dl = $('systems');
  dl.replaceChildren();
  for (const [k, v] of Object.entries(ev.condition_summary)) {
    dl.append(el('dt', null, titleise(k)), el('dd', null, v));
  }
  panel.hidden = false;
}

function asksAndGaps(a, ev) {
  const asks = a.requests || [];
  $('panel-asks').hidden = !asks.length;
  $('asks').replaceChildren(...asks.map((t) => el('li', null, t)));
  const gaps = ev ? ev.coverage_gaps : [];
  $('panel-gaps').hidden = !gaps.length;
  $('gaps').replaceChildren(...gaps.map((t) => el('li', null, t)));
}

function why(price) {
  const panel = $('panel-why');
  if (!(price && price.ok)) { panel.hidden = true; return; }

  const dt = $('drivers');
  dt.replaceChildren();
  const head = el('tr');
  head.append(el('th', null, 'Factor'), el('th', 'num', 'Value'),
              el('th', 'num', 'Effect vs. the average comparable'));
  dt.append(head);
  for (const d of (price.drivers || []).slice(0, 6)) {
    const tr = el('tr');
    tr.append(el('td', null, titleise(d.feature)),
              el('td', 'num', fixed(d.value, 2)),
              el('td', 'num ' + (d.pct_effect >= 0 ? 'pos' : 'neg'),
                 `${d.pct_effect >= 0 ? '+' : ''}${fixed(d.pct_effect, 1)}%`));
    dt.append(tr);
  }
  if (price.adjustment && price.adjustment.pct) {
    const tr = el('tr');
    tr.append(el('td', null, 'Condition (from the photos)'), el('td', 'num', ''),
              el('td', 'num ' + (price.adjustment.pct >= 0 ? 'pos' : 'neg'),
                 `${price.adjustment.pct >= 0 ? '+' : ''}${fixed(price.adjustment.pct, 1)}%`));
    tr.title = price.adjustment.basis;
    dt.append(tr);
  }

  const ct = $('comps');
  ct.replaceChildren();
  const ch = el('tr');
  ch.append(el('th', null, 'Year'), el('th', null, 'Listing'),
            el('th', 'num', 'km'), el('th', 'num', 'Asking'));
  ct.append(ch);
  for (const c of price.comparables || []) {
    const tr = el('tr');
    tr.append(el('td', 'num', c.year == null ? '—' : String(c.year)),
              el('td', null, `${c.make || ''} ${c.model || ''}`.trim()),
              el('td', 'num', kkm(c.km)),
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
  panel.hidden = false;
}

function caveats(price) {
  const cav = price ? price.caveats || [] : [];
  $('panel-caveats').hidden = !cav.length;
  $('caveats').replaceChildren(...cav.map((t) => el('li', null, t)));
}
