/* Everything below the price.

   The rule the old screen broke: a seller is not reading a model card. The
   numbers that belong to the model - R squared, out-of-fold coverage, feature
   contributions in log space, backend ids, stage timings - are all still on
   the page, because the honesty rules require them to be, but they are inside
   `How I worked this out` rather than between a person and their answer.

   What survives in the open is what a buyer would ask a mechanic: what is
   wrong with it, what is right with it, what could you not see, and what is
   it. */

import { $, el, money, titleise, fixed, kkm, animate, reduced } from './dom.js';
import { partIcon } from './icons.js';
import * as elevation from './elevation.js';
import { citePhoto, checkFor, photoOrdinal } from './frames.js';

const SEV_ORDER = { major: 0, moderate: 1, minor: 2, cosmetic: 3 };

const SEV_WORD = {
  major: 'Major', moderate: 'Worth fixing', minor: 'Minor', cosmetic: 'Cosmetic',
};
const IMPACT_WORD = {
  high: 'moves the price', medium: 'moves the price',
  low: 'small effect on price', none: 'no effect on price',
};

/* A decimal is not a confidence a person can act on. These three are. */
function certainty(value) {
  if (value >= 0.85) return 'clear in the photo';
  if (value >= 0.6) return 'likely';
  return 'possible';
}

export function renderPanels(a, runElev) {
  const ev = a.evidence, price = a.price;
  findings(a, ev, runElev);
  strengths(a, ev);
  asksAndGaps(a, ev);
  identity(ev);
  systems(ev);
  working(a, ev, price);
}

/* ---------- what I found ---------- */

function findings(a, ev, runElev) {
  const block = $('block-findings');
  if (!(ev && ev.issues.length)) { block.hidden = true; return; }

  const n = ev.issues.length;
  const major = ev.issues.filter((i) => i.severity === 'major').length;
  $('findings-sub').textContent =
    `${n} thing${n === 1 ? '' : 's'} worth knowing about`
    + (major ? `, ${major} of them serious. ` : '. ')
    + 'Each one names the photo it came from — press it to see that photo.';

  const list = [...ev.issues].sort(
    (x, y) => (SEV_ORDER[x.severity] ?? 9) - (SEV_ORDER[y.severity] ?? 9));

  const rows = list.map((i) => {
    const li = el('li');
    const b = el('button', 'finding');
    b.type = 'button';
    b.dataset.sev = i.severity;

    const part = el('span', 'finding-part', titleise(i.component));
    part.append(el('span', 'finding-weight',
                   `${SEV_WORD[i.severity] || i.severity} — ${IMPACT_WORD[i.price_impact] || ''}`));
    b.append(partIcon(i.component), part, el('span', 'finding-text', i.observation));

    /* An uploaded set is renamed 000.jpg upwards on the way in, so a filename
       is not something a seller can resolve. The position in their own set is,
       and pressing the line opens that photo. The text report still cites the
       filename, where it is the only handle a reader has. */
    const check = checkFor(i.photo_id);
    const seen = (i.also_seen_in || []).length;
    const cite = el('span', 'finding-cite');
    cite.append(document.createTextNode('Seen in '));
    cite.append(el('b', null, seen ? `${seen + 1} photos`
                                   : `photo ${photoOrdinal(i.photo_id)}`));
    cite.append(document.createTextNode(` — ${certainty(i.confidence)}`));
    if (check) b.title = check.filename;
    b.append(cite);

    b.addEventListener('click', () => citePhoto(i.photo_id));
    /* hovering a finding lights the part of the drawing it is about */
    const light = (on) => elevation.focus(runElev, i.component, on);
    b.addEventListener('pointerenter', () => light(true));
    b.addEventListener('pointerleave', () => light(false));
    b.addEventListener('focus', () => light(true));
    b.addEventListener('blur', () => light(false));
    li.append(b);
    return li;
  });
  $('findings').replaceChildren(...rows);
  block.hidden = false;
  if (!reduced()) {
    rows.forEach((r, i) => animate(r,
      { opacity: [0, 1], transform: ['translateY(6px)', 'none'] },
      { duration: 0.35, delay: 0.04 * i, ease: [0.16, 1, 0.3, 1] }));
  }
}

/* ---------- what looks right ---------- */

function strengths(a, ev) {
  const block = $('block-strengths');
  const all = [];
  for (const f of (ev ? ev.photo_findings : []) || []) {
    for (const good of f.strengths || []) all.push({ photo_id: f.photo_id, text: good });
  }
  if (!all.length) { block.hidden = true; return; }
  $('strengths').replaceChildren(...all.slice(0, 12).map((s) => {
    const li = el('li');
    const wrap = el('span');
    wrap.append(document.createTextNode(s.text));
    wrap.append(el('span', null, ` — seen in photo ${photoOrdinal(s.photo_id)}`));
    li.append(wrap);
    return li;
  }));
  block.hidden = false;
}

/* ---------- what is still needed ---------- */

function asksAndGaps(a, ev) {
  const asks = a.requests || [];
  $('block-asks').hidden = !asks.length;
  $('asks').replaceChildren(...asks.map(bullet));
  const gaps = ev ? ev.coverage_gaps : [];
  $('block-gaps').hidden = !gaps.length;
  $('gaps').replaceChildren(...gaps.map(bullet));
}

/* ---------- what it is ---------- */

function identity(ev) {
  const block = $('block-identity');
  if (!(ev && (ev.vehicle.make || ev.vehicle.odometer_km))) { block.hidden = true; return; }
  const v = ev.vehicle, dl = $('identity');
  dl.replaceChildren();
  const add = (k, val, cls) => {
    if (!val) return;
    dl.append(el('dt', null, k), el('dd', cls, val));
  };
  add('Make and model', [v.make, v.model].filter(Boolean).join(' '), 'read');
  add('Cab', v.cab_type);
  add('Axles', v.axle_config);
  add('Looks like a', v.approx_year_range);
  if (v.odometer_km) {
    add('Odometer', `${v.odometer_km.toLocaleString('en-US')} km`, 'read fig');
  }
  if (v.badges_seen && v.badges_seen.length) {
    add('Badges on the truck', v.badges_seen.join(', '));
  }
  block.hidden = false;
}

function systems(ev) {
  const block = $('block-systems');
  if (!(ev && Object.keys(ev.condition_summary).length)) { block.hidden = true; return; }
  const dl = $('systems');
  dl.replaceChildren();
  for (const [k, v] of Object.entries(ev.condition_summary)) {
    const card = el('div', 'system-card');
    const term = el('dt');
    term.append(partIcon(k), el('span', null, titleise(k)));
    const definition = el('dd');
    const detail = el('details', 'system-detail');
    const summary = el('summary', null, 'Read assessment');
    detail.append(summary, el('p', null, v));
    definition.append(detail);
    card.append(term, definition);
    dl.append(card);
  }
  block.hidden = false;
}

/* ---------- the disclosure ---------- */

function section(title, ...nodes) {
  const s = el('section', 'work-section');
  s.append(el('h3', null, title), ...nodes.filter(Boolean));
  return s;
}

function table(head, rows) {
  const t = el('table', 'work-table');
  const hr = el('tr');
  head.forEach(([text, cls]) => hr.append(el('th', cls, text)));
  t.append(hr);
  rows.forEach((cells) => {
    const tr = el('tr');
    cells.forEach(([text, cls]) => tr.append(el('td', cls, text)));
    t.append(tr);
  });
  return t;
}

function bullet(text) {
  const li = el('li');
  li.append(el('span', null, text));
  return li;
}

function list(items) {
  const ul = el('ul', 'work-list');
  items.forEach((t) => ul.append(bullet(t)));
  return ul;
}

function working(a, ev, price) {
  const body = $('working-body');
  body.replaceChildren();

  if (price && price.ok) {
    const card = price.model_card || {};
    const drivers = (price.drivers || []).slice(0, 6).map((d) => [
      [titleise(d.feature)], [fixed(d.value, 2), 'num'],
      [`${d.pct_effect >= 0 ? '+' : ''}${fixed(d.pct_effect, 1)}%`,
       'num ' + (d.pct_effect >= 0 ? 'pos' : 'neg')]]);
    if (price.adjustment && price.adjustment.pct) {
      drivers.push([['Condition, from the photos'], ['', 'num'],
                    [`${price.adjustment.pct >= 0 ? '+' : ''}${fixed(price.adjustment.pct, 1)}%`,
                     'num ' + (price.adjustment.pct >= 0 ? 'pos' : 'neg')]]);
    }
    body.append(section(
      'What moves this number',
      el('p', null, 'A ridge regression over the comparable listings. The vision '
        + 'model never sees a price and the regression never sees the photos; the '
        + 'only thing that crosses between them is the condition multiplier on the '
        + 'last row.'),
      table([['Factor'], ['Value', 'num'], ['Effect vs. the average truck', 'num']],
            drivers)));

    if (price.comparables && price.comparables.length) {
      body.append(section(
        'The trucks it was compared against',
        table([['Year'], ['Listing'], ['km', 'num'], ['Asking', 'num']],
              price.comparables.map((c) => [
                [c.year == null ? '—' : String(c.year), 'num'],
                [`${c.make || ''} ${c.model || ''}`.trim()],
                [kkm(c.km), 'num'],
                [money(c.price, c.currency), 'num']])),
        el('p', null, 'These are asking prices, not prices anything sold for.')));
    }

    const accuracy = [];
    if (card.coverage != null) {
      const p = el('p', 'work-note measured');
      p.append(document.createTextNode('Measured: the comparable-asking band is an '
        + `${Math.round(price.interval_level * 100)}% interval, and on held-out `
        + 'listings it contained the real asking price '));
      p.append(el('b', null, `${(card.coverage * 100).toFixed(1)}%`));
      p.append(document.createTextNode(` of the time over ${card.coverage_n} `
        + `evaluations. Fit R² ${card.r2}, median error ${card.mae_pct}% across `
        + `${card.n_listings} listings collapsing to ${card.n_groups} distinct specs.`));
      accuracy.push(p);
      accuracy.push(el('p', 'work-note', 'That figure belongs to the pale bar only. '
        + 'The red bar is that range moved by what the photos found, and no one '
        + 'has measured how often that one is right.'));
    }
    if (price.adjustment && price.adjustment.cap_pct) {
      const p = el('p', 'work-note assumed');
      p.append(document.createTextNode('Assumed: the condition adjustment is capped at '));
      p.append(el('b', null, `±${fixed(price.adjustment.cap_pct, 1)}%`));
      p.append(document.createTextNode(', one residual standard deviation of the '
        + 'price model. That cap is a stated assumption, not a measurement.'));
      accuracy.push(p);
    }
    if (accuracy.length) body.append(section('How accurate this is', ...accuracy));

    if (price.widened && price.widened.length) {
      body.append(section('Why the range is wider than usual', list(price.widened)));
    }
    if (price.inputs_provenance && Object.keys(price.inputs_provenance).length) {
      body.append(section('Where the inputs came from',
        table([['Input'], ['Source']],
              Object.entries(price.inputs_provenance).map(
                ([k, v]) => [[titleise(k)], [v]]))));
    }
    const anchor = price.anchor;
    if (anchor && anchor.ok) {
      body.append(section(
        'A second opinion, from what it cost new',
        el('p', null, `${money(anchor.point, anchor.currency)} — ${anchor.basis}`),
        el('p', 'work-note', `Source: ${anchor.source} `
          + `(${String(anchor.source_type).replace(/_/g, ' ')}), stamped ${anchor.as_of}. `
          + `Weight in the estimate ${Math.round(anchor.weight * 100)}%, by inverse variance.`)));
    }
    if (price.caveats && price.caveats.length) {
      body.append(section('Caveats', list(price.caveats)));
    }
  }

  if (ev) {
    const calls = ev.calls || [];
    const bits = [];
    bits.push(el('p', null, `Answered by ${ev.backend} / ${ev.model}. Each photo got `
      + `its own call: ${ev.photos_read} read`
      + (ev.photos_failed ? `, ${ev.photos_failed} could not be` : '')
      + `, then a synthesis pass over the results.`));
    if (calls.length) {
      bits.push(table([['Call'], ['Seconds', 'num']],
                      calls.map(([name, secs]) => [[String(name)], [fixed(secs, 2), 'num']])));
    }
    if (ev.parse_warnings && ev.parse_warnings.length) {
      bits.push(el('p', 'work-note', 'Problems the parser found and recorded:'));
      bits.push(list(ev.parse_warnings));
    }
    body.append(section('What read the photos', ...bits));
  }

  const r = a.reconcile;
  if (r && r.corrections && r.corrections.length) {
    body.append(section(
      `${r.corrections.length} correction(s) from the trained heads`,
      el('p', null, 'Small models fitted on the corpus cross-check the vision '
        + 'model. A downgraded finding is still shown; it is never deleted.'),
      list(r.corrections.map((c) => c.before
        ? `${c.detail} — ${c.before} → ${c.after}`
        : c.detail))));
  }

  if (a.trace && a.trace.length) {
    body.append(section('Stages',
      table([['Stage'], ['Seconds', 'num'], ['What it did']],
            a.trace.map((s) => [[titleise(s.step)], [fixed(s.elapsed_s, 2), 'num'],
                                [s.detail]])),
      el('p', 'work-note', `Total ${fixed(a.elapsed_s, 2)}s`
        + (a.version ? ` · build ${a.version}` : ''))));
  }
}
