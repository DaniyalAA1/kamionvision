/* Everything below the price.

   The rule the old screen broke: a seller is not reading a model card. The
   numbers that belong to the model - R squared, out-of-fold coverage, feature
   contributions in log space, backend ids, stage timings - are all still on
   the page, because the honesty rules require them to be, but they are inside
   `How this number was reached` rather than between a person and their answer.

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
  vehicleHistory(a);
  systems(ev);
  working(a, ev, price);
}

/* ---------- what I found ---------- */

function findings(a, ev, runElev) {
  const block = $('block-findings');
  if (!(ev && ev.issues.length)) { block.hidden = true; return; }

  const n = ev.issues.length;
  const major = ev.issues.filter((i) => i.severity === 'major').length;
  $('findings-sub').textContent = major
    ? `${n} findings, ${major} major. Press one to mark it on the photo.`
    : `${n} finding${n === 1 ? '' : 's'}. Press one to mark it on the photo.`;

  const list = [...ev.issues].sort(
    (x, y) => (SEV_ORDER[x.severity] ?? 9) - (SEV_ORDER[y.severity] ?? 9));

  const rows = list.map((i) => {
    const li = el('li');
    const b = el('button', 'finding');
    b.type = 'button';
    b.dataset.sev = i.severity;

    const part = el('span', 'finding-part', titleise(i.component));
    /* An unreadable severity is not a minor defect. It is shown with its photo
       and weighted at zero rather than rounded up, so it says so here too. */
    part.append(el('span', 'finding-weight', i.ungraded
      ? 'reported without a grade'
      : `${SEV_WORD[i.severity] || i.severity} — ${IMPACT_WORD[i.price_impact] || ''}`));
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
    /* Frames that disagreed about how bad this is. The merge keeps the level
       two frames independently support, and prints the ones that lost rather
       than quietly dropping them. */
    if ((i.severity_span || []).length > 1) {
      cite.append(document.createTextNode(
        `, called ${i.severity_span.join(' and ')} by different frames`));
    }
    if (check) b.title = check.filename;
    b.append(cite);

    b.addEventListener('click', () => citePhoto(i.photo_id, i));
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
    wrap.append(el('span', 'cite', ` — seen in photo ${photoOrdinal(s.photo_id)}`));
    li.append(partIcon('check'), wrap);
    return li;
  }));
  block.hidden = false;
}

/* ---------- what is still needed ---------- */

function asksAndGaps(a, ev) {
  const asks = a.requests || [];
  $('block-asks').hidden = !asks.length;
  $('asks').replaceChildren(...asks.map((t) => bullet(t, 'camera')));
  const gaps = ev ? ev.coverage_gaps : [];
  $('block-gaps').hidden = !gaps.length;
  $('gaps').replaceChildren(...gaps.map((t) => bullet(t, 'search')));
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

const SYSTEM_TITLE = {
  tires: 'Tires',
  wheels_brakes: 'Wheels & brakes',
  fifth_wheel_coupling: 'Fifth-wheel coupling',
  chassis_corrosion: 'Chassis',
  body_paint: 'Body & paint',
  cab_interior: 'Cab interior',
  engine_driveline: 'Engine & driveline',
  glass_lights: 'Glass & lights',
};

function systems(ev) {
  const block = $('block-systems');
  if (!(ev && Object.keys(ev.condition_summary).length)) { block.hidden = true; return; }
  const dl = $('systems');
  dl.replaceChildren();
  for (const [k, v] of Object.entries(ev.condition_summary)) {
    const card = el('div', 'system-card');
    const term = el('dt');
    term.append(partIcon(k), el('span', null, SYSTEM_TITLE[k] || titleise(k)));
    const definition = el('dd');
    const line = gist(v);
    if (line === String(v || '').trim() || !v) {
      definition.append(el('p', 'system-gist', line || 'Unseen in this set'));
    } else {
      const detail = el('details', 'system-detail');
      detail.append(el('summary', null, line), el('p', null, v));
      definition.append(detail);
    }
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

function gist(text) {
  const t = String(text || '').trim();
  if (!t || /^not visible/i.test(t)) return 'Unseen in this set';
  const sentence = t.split(/(?<=[.!?])\s+/)[0] || t;
  return sentence.length > 120 ? `${sentence.slice(0, 116)}…` : sentence;
}

function bullet(text, icon) {
  const li = el('li');
  if (icon) li.append(partIcon(icon));
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
                [c.year == null ? '—' : String(c.year)],
                [`${c.make || ''} ${c.model || ''}`.trim()],
                [kkm(c.km), 'num'],
                [money(c.price, c.currency), 'num']])),
        el('p', null, 'These figures are estimated asking prices rather than prices anything sold for.')));
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
        + 'The blue bar includes condition and any confirmed history adjustment, and no one '
        + 'has measured how often that one is right.'));
    }
    // The cap and the weights are not the same kind of number. This panel used
    // to call the measured sigma "a stated assumption, not a measurement" while
    // app/pricing/model.py called the 1-sigma CHOICE "a measured quantity
    // rather than a chosen one". Both were wrong, in opposite directions. The
    // honest split is three-way and it is printed as three sentences.
    const adj = price.adjustment;
    if (adj && adj.cap_pct) {
      const m = el('p', 'work-note measured');
      m.append(document.createTextNode('Measured: one out-of-fold residual standard '
        + 'deviation of the price model is '));
      m.append(el('b', null, `±${fixed(adj.cap_pct, 1)}%`));
      m.append(document.createTextNode(' — the variation in asking price that year, '
        + 'kilometres, brand and market do not explain.'));
      accuracy.push(m);
      accuracy.push(el('p', 'work-note assumed', 'Assumed: that condition is capped at '
        + 'exactly one of those sigmas. The reasoning — condition cannot be worth '
        + 'more than everything we cannot see — is sound, and no experiment picks '
        + 'one sigma over half or two.'));
      if (adj.weights_basis) {
        accuracy.push(el('p', 'work-note assumed', 'Assumed: every weight that decides '
          + 'where inside that cap this truck lands — the severity × impact table, '
          + 'the per-subsystem saturation decay, the family value weights and the '
          + 'coverage thresholds. This corpus carries no condition ground truth to fit '
          + 'them against, so they are stated rather than measured.'));
      }
      if (adj.coverage_pct) {
        const c = el('p', 'work-note');
        c.append(document.createTextNode('This run photographed '));
        c.append(el('b', null, `${fixed(adj.coverage_pct, 0)}%`));
        c.append(document.createTextNode(' of the truck by value legibly, and positively '
          + 'called '));
        c.append(el('b', null, `${fixed(adj.merit_pct, 0)}%`));
        c.append(document.createTextNode(' of it sound. Merit can never exceed coverage, '
          + 'so a photo nobody could read earns nothing either way.'));
        accuracy.push(c);
      }
      (adj.notes || []).forEach((n) => accuracy.push(el('p', 'work-note', n)));
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
      + (ev.photos_failed ? `, ${ev.photos_failed} unread` : '')
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

/* Database strings are rendered as text, never HTML. */
function vehicleHistory(a) {
  const block = $('block-history'), h = a.history;
  block.hidden = !h;
  if (!h) return;
  $('history-summary').textContent = (h.notes || []).join(' ') ||
    'Records are matched by plate and country. Only VIN-confirmed history of the appraised truck can change its price.';
  $('history-observations').replaceChildren(...h.observations.map(o => {
    const row = el('li');
    row.append(el('strong', null, `${o.country || ''} ${o.plate || 'Plate unreadable'} · ${o.is_subject ? 'Appraised truck' : 'Other vehicle'} · ${titleise(o.status)}`));
    row.append(el('p', null, o.reason));
    const photo = el('button', null, `View photo ${photoOrdinal(o.photo_id)}`);
    photo.type = 'button';
    photo.addEventListener('click', () => citePhoto(o.photo_id));
    row.append(photo);
    if (o.record && o.record.source) {
      row.append(el('p', null, `${o.record.source} · Record ${o.record.record_id} · As of ${o.record.as_of}`));
      for (const e of o.record.events || []) {
        row.append(el('p', null, `${e.date} · ${e.type} · ${e.description} [${e.id}]`));
      }
      if (!o.record.events.length) row.append(el('p', null, 'This source lists zero events. Absence of a record leaves accident history unproven.'));
    }
    return row;
  }));
  $('history-reasoning').replaceChildren(...(h.reasoning || []).map(line => el('li', null, line)));
}
