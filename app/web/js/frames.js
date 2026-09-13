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

import { $, el, svg, viewName, fixed, animate, reduced } from './dom.js';

const VEHICLE = new Set(['truck', 'bus', 'car', 'train', 'motorcycle', 'bicycle',
                         'boat', 'airplane']);

const CAPTURE_WORD = { good: 'sharp', fair: 'usable', poor: 'soft' };

let urls = {};
let checks = [];
let refusedAsNotATruck = false;
let showAll = false;
let current = null;

export function setSource(photoUrls, photoChecks, decision) {
  urls = photoUrls || {};
  checks = photoChecks || [];
  /* Truck detection is a set-level rule, never per-photo: a tire close-up
     contains no truck-shaped object and is still a photo of the truck. So a
     box only earns the refusal colour when the whole set was refused for not
     being a truck, and it is the thing the gate named instead. */
  refusedAsNotATruck = decision === 'refuse_not_a_truck';
}

export const urlFor = (id) => urls[id] || '';
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
  current = check;
  const stage = $('frame-stage');
  const img = $('frame-img');

  if (check.width && check.height) {
    stage.style.aspectRatio = `${check.width} / ${check.height}`;
    $('frame-boxes').setAttribute('viewBox', `0 0 ${check.width} ${check.height}`);
  }
  img.classList.remove('in');
  img.alt = check.usable ? viewName(check.view) : 'frame the gate dropped';
  img.src = urlFor(check.photo_id);
  img.onload = () => img.classList.add('in');
  drawBoxes($('frame-boxes'), check);
  writeMeta(check);

  const toggle = $('show-all');
  if (toggle) toggle.hidden = !hasHiddenBoxes(check);
}

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
  if (!all.length) return;

  const subject = subjectOf(all, check);
  const blockedLabel = (refusedAsNotATruck && check.non_truck_subject)
    ? String(check.non_truck_subject).split(' (')[0] : null;
  const disqualifying = blockedLabel
    ? all.filter((d) => d.label === blockedLabel) : [];

  /* Default: the subject, plus whatever the gate refused on. Everything else
     is behind the toggle. */
  const shown = showAll ? all
    : [subject, ...disqualifying].filter((d, i, list) => d && list.indexOf(d) === i);
  if (!shown.length) return;

  /* Darken the frame outside the subject, drawn as one even-odd path so the
     subject stays at full brightness without a second image. */
  if (subject && !showAll) {
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
    root.append(g);

    /* The label is HTML over the frame, not inside the scaled viewBox, so it
       renders at its real size whatever the frame is shown at. It names the
       subject in words rather than quoting a COCO class and a decimal, which
       told a seller nothing they wanted to know. */
    const text = isSubject ? 'the truck being appraised'
      : (blockedLabel && d.label === blockedLabel) ? d.label
      : `${d.label} ${fixed(d.confidence, 2)}`;
    const chip = el('span', 'box-chip', text);
    chip.dataset.subject = String(isSubject);
    if (blockedLabel && d.label === blockedLabel) chip.dataset.disqualifying = 'true';
    chip.style.left = `${(x1 / check.width) * 100}%`;
    chip.style.top = `${(y1 / check.height) * 100}%`;
    /* a box hugging the top edge has no room above it for its own label */
    if (y1 / check.height > 0.07) chip.classList.add('above');
    chips.append(chip);

    animate(r, { strokeDashoffset: [perim, 0] },
            { duration: 0.55, delay: 0.12 + i * 0.07, ease: [0.2, 0.7, 0.3, 1] });
    animate(chip, { opacity: [0, 1] }, { duration: 0.25, delay: 0.5 + i * 0.07 });
  });
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
