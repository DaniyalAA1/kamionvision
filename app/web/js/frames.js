/* The photographs: the contact strip, the featured frame with the detector's
   box over it, the result grid, and the lightbox.

   One box, not eight. The subject is decided in app/subject.py, not here, so
   the box drawn and the pixels the vision model was given are the same
   vehicle. The old overlay drew every detection above 0.25 with equal weight,
   which on a dealer-lot photo meant five trucks outlined and no answer to
   which one was for sale - and the model was handed the whole frame, so it did
   not know either.

   Which detection is the subject is read off `is_subject`, never recomputed.
   This used to be an exact float comparison on all four coordinates, which
   worked only because `subject_box` was literally an element of `detections`;
   the moment the gate computed the subject anywhere else the match would have
   failed silently, `shown` would have been empty and no box would have been
   drawn at all. `sameBox` survives only so an older frozen export still
   renders.

   Everything outside the subject is dimmed rather than deleted: a person can
   still see what else was in the frame, and `show everything it detected`
   brings the rest of the boxes back for anyone who wants them. */

import { $, el, svg, viewName, fixed, animate, reduced, titleise } from './dom.js';
import { partIcon } from './icons.js';

const VEHICLE = new Set(['truck', 'bus', 'car', 'train', 'motorcycle', 'bicycle',
                         'boat', 'airplane']);

const CAPTURE_WORD = { good: 'sharp', fair: 'usable', poor: 'soft' };

let urls = {};
let checks = [];
let findings = {};
let evidencePhotos = null;
let refusedAsNotATruck = false;
let showAll = false;
let current = null;
let frameVersion = 0;
let partTimers = [];
const partName = (id) => ({engine_bay:'Engine bay', windscreen_glass:'Windshield',
  grille_headlights:'Grille & headlights', cab_exterior_panels:'Cab panels',
  steer_tires:'Front tires', drive_tires:'Drive tires', dashboard_instruments:'Dashboard'}[id] || titleise(id));

function clearPartTour() { partTimers.forEach(clearTimeout); partTimers = []; }

function locatedParts(finding) {
  if (!finding || finding.error) return [];
  const items = [...(finding.issues || []).map(i => ({...i, kind:'finding'})),
    ...(finding.component_regions || []).filter(r => !(finding.issues || []).some(i => i.component === r.component && i.box))
      .map(r => ({...r, kind:'visible'}))];
  return items.filter(p => Array.isArray(p.box) && p.box.length === 4 &&
    p.box.every(Number.isFinite) && p.box[0] >= 0 && p.box[1] >= 0 &&
    p.box[2] > 0 && p.box[3] > 0 && p.box[0]+p.box[2] <= 1.001 && p.box[1]+p.box[3] <= 1.001);
}

function focusPart(index) {
  document.querySelectorAll('[data-part-index]').forEach(n => {
    n.dataset.active = String(Number(n.dataset.partIndex) === index);
    if (n.tagName === 'BUTTON') n.setAttribute('aria-pressed', n.dataset.active);
  });
}


export function setSource(photoUrls, photoChecks, decision, evidenceIds = null) {
  frameVersion += 1;
  if (!current || urls[current.photo_id] !== photoUrls?.[current.photo_id]) current = null;
  urls = photoUrls || {};
  checks = photoChecks || [];
  findings = {};
  evidencePhotos = evidenceIds === null ? null : new Set(evidenceIds);
  const intel = $('frame-intel');
  if (intel) intel.replaceChildren();
  /* Truck detection is a set-level rule, never per-photo: a tire close-up
     contains no truck-shaped object and is still a photo of the truck. So a
     box only earns the refusal colour when the whole set was refused for not
     being a truck, and it is the thing the gate named instead. */
  refusedAsNotATruck = decision === 'refuse_not_a_truck';
}

export function setFinding(finding) {
  if (!finding) return;
  findings[finding.photo_id] = finding;
  updateStripCellStatus(finding);
  if (current && current.photo_id === finding.photo_id) {
    renderIntel(current, finding);
    drawBoxes($('frame-boxes'), current);
  }
}

export const findingFor = (id) => findings[id] || null;

export function cancelPendingFrame() {
  clearPartTour();
  frameVersion += 1;
  $('frame-stage').querySelectorAll('.frame-outgoing').forEach((n) => n.remove());
}

export const urlFor = (id) => urls[id] || '';
export const currentPhotoId = () => current?.photo_id;
export const checkFor = (id) => checks.find((c) => c.photo_id === id);

/* Where this photo sits in the set the person actually sent, 1-based. */
export const photoOrdinal = (id) => {
  const i = checks.findIndex((c) => c.photo_id === id);
  return i < 0 ? id : i + 1;
};

/* The frame the refusal is about: whichever one the gate named a non-truck
   subject in, most confident first. */
export function smokingGun() {
  const conf = (c) => {
    const m = /\(([\d.]+)\)/.exec(c.non_truck_subject || '');
    return m ? parseFloat(m[1]) : -1;
  };
  return checks.filter((c) => c.non_truck_subject)
    .sort((a, b) => conf(b) - conf(a))[0] || null;
}

export function setShowAll(on) {
  showAll = on;
  if (current) drawBoxes($('frame-boxes'), current);
}

/* Is there anything to reveal? A frame with one box has no "everything else". */
export const hasHiddenBoxes = (check) =>
  !!check && (check.detections || []).filter((d) => VEHICLE.has(d.label)).length > 1;

/* ---------- featured frame ---------- */

export function showFrame(check) {
  if (!check) return;
  clearPartTour();
  const version = ++frameVersion;
  const source = urlFor(check.photo_id);
  const preload = new Image();
  preload.onload = () => {
    if (version !== frameVersion) return;
    const img = $('frame-img');
    const stage = $('frame-stage');
    stage.querySelectorAll('.frame-outgoing').forEach((n) => n.remove());
    if (img.getAttribute('src') && !reduced()) {
      const previous = img.cloneNode();
      previous.removeAttribute('id');
      previous.alt = '';
      previous.setAttribute('aria-hidden', 'true');
      previous.className = 'frame-outgoing in';
      stage.insertBefore(previous, img);
      previous.addEventListener('animationend', () => previous.remove(), { once: true });
    }
    current = check;
    img.alt = check.usable ? viewName(check.view) : 'frame the gate dropped';
    img.src = source;
    img.classList.add('in');
    if (check.width && check.height) {
      $('frame-boxes').setAttribute('viewBox', `0 0 ${check.width} ${check.height}`);
    }
    fitChips();
    drawBoxes($('frame-boxes'), check);
    writeMeta(check);
    renderIntel(check, findings[check.photo_id] || null);
    const toggle = $('show-all');
    if (toggle) toggle.hidden = !hasHiddenBoxes(check);
  };
  preload.onerror = () => {
    if (version !== frameVersion) return;
    current = null;
    $('show-all').hidden = true;
    $('frame-img').removeAttribute('src');
    $('frame-img').alt = 'Photo could not be loaded';
    $('frame-boxes').replaceChildren();
    $('frame-chips').replaceChildren();
    $('frame-meta').textContent = 'Photo could not be loaded. Select another photo to continue.';
    const intel = $('frame-intel');
    if (intel) intel.replaceChildren();
  };
  preload.src = source;
}

// Keep HTML labels aligned with the contained image, including portrait photos.
function fitChips() {
  if (!current?.width || !current?.height) return;
  const stage = $('frame-stage');
  const scale = Math.min(stage.clientWidth / current.width, stage.clientHeight / current.height);
  $('frame-chips').style.inset = `${(stage.clientHeight - current.height * scale) / 2}px ${(stage.clientWidth - current.width * scale) / 2}px`;
}
new ResizeObserver(fitChips).observe($('frame-stage'));

/* Fallback only: a frozen export written before `is_subject` existed carries
   the subject as coordinates and nothing else. */
const sameBox = (a, b) =>
  !!a && !!b && a.length === 4 && b.length === 4 && a.every((v, i) => v === b[i]);

const subjectOf = (all, check) =>
  all.find((d) => d.is_subject) ||
  (check.subject_box ? all.find((d) => sameBox(d.box, check.subject_box)) : undefined);

function drawBoxes(root, check) {
  root.replaceChildren();
  const chips = $('frame-chips');
  chips.replaceChildren();
  const all = (check.detections || []).filter(
    (d) => d.box && d.box.length === 4 && check.width && check.height);

  const subject = subjectOf(all, check);
  const blockedLabel = (refusedAsNotATruck && check.non_truck_subject)
    ? String(check.non_truck_subject).split(' (')[0] : null;
  const disqualifying = blockedLabel
    ? all.filter((d) => d.label === blockedLabel) : [];

  /* Default: the subject, plus whatever the gate refused on. Everything else
     is behind the toggle. */
  const shown = showAll ? all
    : [subject, ...disqualifying].filter((d, i, list) => d && list.indexOf(d) === i);

  /* Darken the frame outside the subject, drawn as one even-odd path so the
     subject stays at full brightness without a second image. */
  if (subject && !showAll && check.width && check.height) {
    const [x1, y1, x2, y2] = subject.box;
    const outer = `M0 0H${check.width}V${check.height}H0Z`;
    const inner = `M${x1} ${y1}H${x2}V${y2}H${x1}Z`;
    const dim = svg('path', { class: 'dim', d: `${outer} ${inner}`, 'fill-rule': 'evenodd' });
    root.append(dim);
    animate(dim, { opacity: [0, 1] }, { duration: 0.45, delay: 0.1 });
  }

  shown.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.box;
    const w = Math.max(1, x2 - x1), h = Math.max(1, y2 - y1);
    const g = svg('g', { class: 'box' });
    const isSubject = d === subject;
    g.dataset.subject = String(isSubject);
    if (blockedLabel && d.label === blockedLabel) g.dataset.disqualifying = 'true';

    const r = svg('rect', { x: x1, y: y1, width: w, height: h });
    const perim = 2 * (w + h);
    if (!reduced()) {
      r.setAttribute('stroke-dasharray', perim);
      r.setAttribute('stroke-dashoffset', perim);
    }
    g.append(r);

    if (isSubject && !reduced()) {
      const cLen = Math.min(22, Math.max(8, w * 0.08, h * 0.08));
      const corners = `M ${x1} ${y1 + cLen} L ${x1} ${y1} L ${x1 + cLen} ${y1} ` +
                      `M ${x2 - cLen} ${y1} L ${x2} ${y1} L ${x2} ${y1 + cLen} ` +
                      `M ${x1} ${y2 - cLen} L ${x1} ${y2} L ${x1 + cLen} ${y2} ` +
                      `M ${x2 - cLen} ${y2} L ${x2} ${y2} L ${x2} ${y2 - cLen}`;
      const reticle = svg('path', { class: 'reticle-corner', d: corners });
      g.append(reticle);
    }

    root.append(g);

    /* The label is HTML over the frame, not inside the scaled viewBox, so it
       renders at its real size whatever the frame is shown at. */
    const text = isSubject ? 'the truck being appraised'
      : (blockedLabel && d.label === blockedLabel) ? d.label
      : `${d.label} ${fixed(d.confidence, 2)}`;
    const chip = el('span', 'box-chip', text);
    chip.dataset.subject = String(isSubject);
    if (blockedLabel && d.label === blockedLabel) chip.dataset.disqualifying = 'true';
    chip.style.left = `${(x1 / check.width) * 100}%`;
    chip.style.top = `${(y1 / check.height) * 100}%`;
    if (y1 / check.height > 0.07) chip.classList.add('above');
    chips.append(chip);

    animate(r, { strokeDashoffset: [perim, 0] },
            { duration: 0.55, delay: 0.12 + i * 0.07, ease: [0.2, 0.7, 0.3, 1] });
    animate(chip, { opacity: [0, 1] }, { duration: 0.25, delay: 0.5 + i * 0.07 });
  });

  // Visible parts and defects are distinct: green/neutral means located,
  // never "healthy". No geometry is inferred from a view name or truck diagram.
  clearPartTour();
  locatedParts(findings[check.photo_id]).forEach((part, index) => {
    const [x,y,w,h] = part.box;
    const g = svg('g', {class:'scan-part'});
    g.dataset.partIndex = String(index);
    g.dataset.kind = part.kind;
    const rect = svg('rect', {x:x*check.width, y:y*check.height,
      width:w*check.width, height:h*check.height, pathLength:1});
    g.append(rect); root.append(g);
    const chip = el('span', 'box-chip scan-part-label',
      `${String(index+1).padStart(2,'0')} · ${partName(part.component)}${part.kind === 'finding' ? ' · '+part.severity : ' · visible'}`);
    chip.dataset.partIndex = String(index);
    chip.style.left = `${Math.min(x, .65)*100}%`;
    chip.style.top = `${y*100}%`;
    if (y > .07) chip.classList.add('above');
    chips.append(chip);
    if (index) partTimers.push(setTimeout(() => focusPart(index), index*1600));
  });
  focusPart(0);
}

function writeMeta(check) {
  const meta = $('frame-meta');
  meta.replaceChildren();
  meta.append(el('span', 'fm-view',
                 check.usable ? viewName(check.view) : 'dropped by the gate'));
  if (!check.usable && (check.reasons || []).length) {
    meta.append(el('span', 'fm-why', check.reasons[0]));
  }
  /* Why that box, or why none. A close-up of an engine bay with a lorry in
     the yard behind it gets no box at all, and saying so out loud is the
     difference between a decision and a silence. */
  if (check.usable && check.subject_basis) {
    meta.append(el('span', 'fm-why', check.subject_basis));
  }
  /* A capture score of 0.82 means nothing to a seller holding a phone. The
     word does, and the number is still in the disclosure for anyone who wants
     to check it. */
  const word = CAPTURE_WORD[check.quality_bucket];
  if (word) {
    const cap = el('span', 'fm-capture', word);
    cap.dataset.bucket = check.quality_bucket;
    meta.append(cap);
  }
}

/* ---------- the briefing under the frame ---------- */

/* What the model said about the frame on screen, while it is on screen.

   Every class here is `intel-`-prefixed and every variant travels as a
   `data-` attribute, because these stylesheets share one global cascade.
   The column that read the model's blind spots was classed `gaps`, which
   `result.css` styles as a two-column icon row with a 1.35rem first track -
   so each line collapsed to its longest word, one item became 440px tall,
   the card became 3,637px and `.stage-main` became 4,255px against a 768px
   viewport. Every photograph loaded; all of them were below the fold, with
   the contact strip and the progress bar. `tests.test_offline.LiveBriefing`
   pins both halves of that: the namespace, and the height cap.

   Nothing here states a condition the call did not state. An empty issue
   list means nothing was flagged, which is a fact about the answer - it is
   not a finding that the component is sound, and the screen does not get to
   promote one into the other on the model's behalf. */

const WORST = (issues) => issues.some((i) => i.severity === 'major') ? 'major'
  : issues.some((i) => i.severity === 'moderate') ? 'moderate'
  : issues.some((i) => i.severity === 'minor') ? 'minor' : 'cosmetic';

function pill(kind, text) {
  const node = el('span', 'intel-pill', text);
  node.dataset.kind = kind;
  return node;
}

/* A titled column. `body` is appended as-is; `empty` is the line shown when
   there is nothing to list, and it says what the model did not say rather
   than filling the space with a claim. */
function column(kind, glyph, title, count, items, render, empty) {
  const col = el('div', 'intel-col');
  col.dataset.kind = kind;
  const head = el('h4', 'intel-col-title');
  head.append(partIcon(glyph), el('span', null, title));
  if (count !== null) head.append(el('span', 'intel-col-count', String(count)));
  col.append(head);
  if (items.length) {
    const list = el('ul', 'intel-list');
    items.forEach((item, i) => list.append(render(item, i)));
    col.append(list);
  } else {
    col.append(el('p', 'intel-quiet', empty));
  }
  return col;
}

function issueItem(check, issue, idx) {
  const li = el('li', 'intel-issue');
  li.dataset.sev = issue.severity || 'minor';
  li.dataset.issueId = `${check.photo_id}-${idx}`;

  const head = el('div', 'intel-issue-head');
  const tag = el('span', 'sev-tag', issue.severity || 'minor');
  tag.dataset.sev = issue.severity || 'minor';
  head.append(partIcon(issue.component),
              el('strong', 'intel-issue-comp', titleise(issue.component)), tag);
  li.append(head, el('p', 'intel-issue-obs', issue.observation));

  /* The structured half of the finding, in the model's own vocabulary. */
  const bits = [];
  if (issue.magnitude) {
    if (issue.magnitude.state) bits.push(issue.magnitude.state.replace(/_/g, ' '));
    if (issue.magnitude.extent) bits.push(issue.magnitude.extent.replace(/_/g, ' '));
    if (issue.magnitude.consumable) bits.push('consumable');
  }
  if (issue.price_impact && issue.price_impact !== 'none') {
    bits.push(`${issue.price_impact} price impact`);
  }
  /* Corroboration is a better signal than a confidence decimal, and it is the
     same one the report uses. */
  const also = (issue.also_seen_in || []).length;
  if (also) bits.push(`also in ${also} other photo${also === 1 ? '' : 's'}`);
  if (bits.length) li.append(el('span', 'intel-mag', bits.join(' · ')));

  /* Hovering a finding lights its box on the frame above, when it has one. */
  const mark = (on) => {
    const box = $('frame-boxes')
      ?.querySelector(`[data-issue-id="${check.photo_id}-${idx}"]`);
    if (box) box.classList.toggle('highlight', on);
  };
  li.addEventListener('mouseenter', () => mark(true));
  li.addEventListener('mouseleave', () => mark(false));
  return li;
}

export function renderIntel(check, finding) {
  const container = $('frame-intel');
  if (!container) return;
  if (container.dataset.photoId !== String(check?.photo_id)) container.scrollTop = 0;
  container.dataset.photoId = String(check?.photo_id);
  container.replaceChildren();
  if (!check) return;

  const card = el('div', 'intel-card');
  const issues = (finding && !finding.error && finding.issues) || [];
  const notSelected = !finding && evidencePhotos !== null && !evidencePhotos.has(check.photo_id);

  /* ---- the status line: where this frame is in the run ---- */
  const top = el('div', 'intel-top');
  const badge = el('span', 'intel-badge');
  if (!check.usable) {
    badge.dataset.level = 'dropped';
    badge.textContent = 'Not sent to the model';
  } else if (notSelected) {
    badge.dataset.level = 'dropped';
    badge.textContent = 'Photo checked';
  } else if (!finding) {
    badge.dataset.level = 'scanning';
    badge.append(el('span', 'intel-pulse'), document.createTextNode('Reading this frame'));
  } else if (finding.error) {
    badge.dataset.level = 'error';
    badge.textContent = 'This frame could not be read';
  } else if (!issues.length) {
    badge.dataset.level = 'clean';
    badge.textContent = 'Nothing flagged';
  } else {
    badge.dataset.level = WORST(issues);
    badge.textContent = `${issues.length} finding${issues.length === 1 ? '' : 's'}`;
  }
  top.append(badge, pill('view',
    `${viewName(check.view)} · photo ${photoOrdinal(check.photo_id)} of ${checks.length || 1}`));

  if (finding?.odometer_km) {
    top.append(pill('odo',
      `${Math.round(finding.odometer_km).toLocaleString('en-US')} km on the dash`));
  }
  if (finding?.cropped) top.append(pill('crop', 'cropped to the subject'));
  /* How the answer was obtained, not how good it is: the close-up is sampled
     CLOSEUP_SAMPLES times and a fallback names the photo it happened on. */
  if (finding?.backend) {
    top.append(pill('backend', finding.samples > 1
      ? `${finding.samples}× on ${finding.backend}` : finding.backend));
  }
  const word = CAPTURE_WORD[check.quality_bucket];
  if (check.usable && word) top.append(pill('quality', `${word} photo`));
  card.append(top);

  const parts = locatedParts(finding);
  if (parts.length) {
    const nav = el('div', 'intel-parts');
    nav.setAttribute('aria-label', 'Located parts in this photo');
    parts.forEach((part, index) => {
      const button = el('button', 'intel-part', `${index+1} · ${partName(part.component)}`);
      button.type = 'button'; button.dataset.partIndex = String(index);
      button.setAttribute('aria-pressed', String(index === 0));
      button.dataset.active = String(index === 0);
      button.addEventListener('click', () => { clearPartTour(); focusPart(index); });
      nav.append(button);
    });
    card.append(nav);
  } else if (finding && !finding.error) {
    card.append(el('p', 'intel-quiet', 'No reliable part locations returned for this photo. Findings below are not spatially marked.'));
  }


  /* ---- one line: what the frame is of ---- */
  const headline = el('p', 'intel-headline');
  if (!check.usable) {
    headline.dataset.state = 'pending';
    headline.textContent = `Dropped by the gate: ${(check.reasons || []).join('; ')
      || 'it did not pass the photo checks'}.`;
  } else if (notSelected) {
    headline.dataset.state = 'pending';
    headline.textContent = 'Not selected for an individual close-up reading.';
  } else if (!finding) {
    headline.dataset.state = 'pending';
    headline.textContent = 'Waiting for this photo’s inspection findings.';
  } else if (finding.error) {
    headline.dataset.state = 'pending';
    headline.textContent = finding.error;
  } else {
    headline.textContent = finding.shows
      || (issues.length ? `${issues.length} observation${issues.length === 1 ? '' : 's'} recorded on this frame.`
                        : 'Nothing was flagged on this frame.');
  }
  card.append(headline);

  /* ---- the three lists ---- */
  if (finding && !finding.error) {
    const grid = el('div', 'intel-grid');
    grid.append(column('issues', 'warning', 'Flagged', issues.length, issues,
                       (issue, i) => issueItem(check, issue, i),
                       'Nothing flagged on this frame.'));

    const aside = el('div', 'intel-aside');
    const strengths = finding.strengths || [];
    aside.append(column('sound', 'check', 'Named as sound', strengths.length,
                        strengths, (s) => el('li', null, s),
                        'Nothing named as sound on this frame.'));
    const gaps = finding.cannot_tell || [];
    aside.append(column('gaps', 'search', "Can't tell from this angle",
                        gaps.length, gaps, (g) => el('li', null, g),
                        'Nothing recorded as out of view.'));
    grid.append(aside);
    card.append(grid);
  }

  container.append(card);
}

/* ---------- contact strip ---------- */

export function buildStrip(photos, onPick) {
  const strip = $('strip');
  const cells = photos.map((c) => {
    const b = el('button', 'strip-cell');
    b.type = 'button';
    b.id = `cell-${c.photo_id}`;
    b.dataset.usable = String(!!c.usable);
    b.title = c.usable
      ? viewName(c.view)
      : `dropped: ${(c.reasons || []).join('; ')}`;
    const img = el('img');
    img.src = urlFor(c.photo_id);
    img.alt = '';
    img.loading = 'lazy';
    b.append(img);
    b.addEventListener('click', () => onPick(c));
    return b;
  });
  strip.replaceChildren(...cells);
  if (reduced()) { cells.forEach((c) => c.classList.add('in')); return; }
  /* The frames deal in rather than appearing all at once, which is how you see
     how many there are. */
  cells.forEach((c, i) => setTimeout(() => c.classList.add('in'), 40 + i * 30));
}

export function markCell(id, cls) {
  const c = $(`cell-${id}`);
  if (!c) return;
  c.parentElement.querySelectorAll('.strip-cell.current')
    .forEach((n) => n.classList.remove('current'));
  if (cls) c.classList.add(cls);
  c.classList.add('current');
  // Follow horizontally inside the contact strip, never drag the page away
  // from the photograph (especially on mobile) when a new finding arrives.
  const strip = c.parentElement;
  const cellBox = c.getBoundingClientRect();
  const stripBox = strip.getBoundingClientRect();
  strip.scrollTo({
    left: strip.scrollLeft + cellBox.left - stripBox.left - (strip.clientWidth - cellBox.width) / 2,
    behavior: reduced() ? 'auto' : 'smooth',
  });
}

export function updateStripCellStatus(finding) {
  const c = $(`cell-${finding.photo_id}`);
  if (!c) return;
  c.classList.remove('reading');
  c.classList.add('read');
  if (finding.error) {
    c.dataset.state = 'error';
  } else if (finding.issues && finding.issues.length) {
    c.dataset.state = WORST(finding.issues);
  } else {
    c.dataset.state = 'clean';
  }
}

export function stepPhoto(delta) {
  if (!checks.length || !current) return null;
  const index = checks.findIndex((c) => c.photo_id === current.photo_id);
  if (index < 0) return null;
  const nextIndex = Math.max(0, Math.min(checks.length - 1, index + delta));
  if (nextIndex !== index) {
    const nextCheck = checks[nextIndex];
    showFrame(nextCheck);
    markCell(nextCheck.photo_id);
    return nextCheck;
  }
  return null;
}

/* ---------- result grid + lightbox ---------- */

export function renderGrid(gate, onPick) {
  const grid = $('photo-grid');
  const all = gate.photos || [];
  const used = (gate.usable_photo_ids || []).length;
  $('photo-count').textContent = used === all.length
    ? `${all.length}`
    : `${used} used of ${all.length}`;
  grid.replaceChildren(...all.map((c) => {
    const b = el('button', 'thumb' + (c.usable ? '' : ' dropped'));
    b.type = 'button';
    b.id = `thumb-${c.photo_id}`;
    b.title = c.usable ? viewName(c.view)
                       : `dropped: ${(c.reasons || []).join('; ')}`;
    const img = el('img');
    img.src = urlFor(c.photo_id);
    img.alt = c.usable ? viewName(c.view) : `dropped: ${(c.reasons || []).join('; ')}`;
    img.loading = 'lazy';
    b.append(img, el('span', 'thumb-tag', c.usable ? viewName(c.view) : 'not used'));
    b.addEventListener('click', () => onPick(c));
    return b;
  }));
}

function paintMark(mark) {
  const overlay = $('lightbox-mark');
  const rect = $('lightbox-box');
  if (!overlay) return;
  const box = mark && Array.isArray(mark.box) && mark.box.length === 4 ? mark.box : null;
  overlay.classList.toggle('on', !!(box && rect));
  if (!box || !rect) return;
  const [x, y, w, h] = box;
  rect.setAttribute('x', x);
  rect.setAttribute('y', y);
  rect.setAttribute('width', w);
  rect.setAttribute('height', h);
}

export function openLightbox(check, mark) {
  $('lightbox-img').src = urlFor(check.photo_id);
  $('lightbox-img').alt = viewName(check.view);
  paintMark(mark);
  let cap;
  if (mark && mark.observation) {
    cap = mark.observation;
  } else if (check.usable) {
    cap = `${viewName(check.view)} — ${CAPTURE_WORD[check.quality_bucket] || 'unrated'} photo`;
  } else {
    cap = `Not used: ${(check.reasons || []).join('; ')}`;
  }
  $('lightbox-cap').textContent = cap;
  $('lightbox').hidden = false;
}

/* Flash the thumbnail a finding cites, then open it — marked if the finding
   pointed at pixels. */
export function citePhoto(id, mark) {
  const node = $(`thumb-${id}`);
  const check = checkFor(id);
  if (node) {
    node.classList.remove('flash');
    void node.offsetWidth;
    node.classList.add('flash');
    node.scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'nearest' });
  }
  if (check) openLightbox(check, mark);
}
