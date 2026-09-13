/* The photographs: the contact strip, the featured frame with the detector's
   own boxes over it, the result-view grid, and the lightbox.

   The boxes are not decoration. `detections[].box` is xyxy in source pixels
   and `width`/`height` ship beside it, so the overlay is the detector's
   actual output at the actual scale. Amber marks the box the gate treated as
   the subject; everything else is a hairline, because "there is a truck in
   this frame" and "this frame is of a truck" are the distinction the whole
   gate rests on. */

import { $, el, svg, viewName, fixed, animate, reduced } from './dom.js';

const VEHICLE = new Set(['truck', 'bus', 'car', 'train', 'motorcycle', 'bicycle',
                         'boat', 'airplane']);

let urls = {};
let checks = [];
let refusedAsNotATruck = false;

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

/* ---------- featured frame ---------- */

export function showFrame(check) {
  if (!check) return;
  const stage = $('frame-stage');
  const img = $('frame-img');
  const boxes = $('frame-boxes');

  if (check.width && check.height) {
    stage.style.aspectRatio = `${check.width} / ${check.height}`;
    boxes.setAttribute('viewBox', `0 0 ${check.width} ${check.height}`);
  }
  img.classList.remove('in');
  img.alt = check.usable ? viewName(check.view) : 'frame the gate dropped';
  img.src = urlFor(check.photo_id);
  img.onload = () => img.classList.add('in');
  drawBoxes(boxes, check);
  writeMeta(check);
}

function drawBoxes(root, check) {
  root.replaceChildren();
  const chips = $('frame-chips');
  chips.replaceChildren();
  const dets = (check.detections || []).filter(
    (d) => d.box && d.box.length === 4 && check.width && check.height);
  if (!dets.length) return;

  const vehicles = dets.filter((d) => VEHICLE.has(d.label));
  const biggest = vehicles.reduce(
    (a, b) => (a && a.area_frac >= b.area_frac ? a : b), null);
  const blockedLabel = (refusedAsNotATruck && check.non_truck_subject)
    ? String(check.non_truck_subject).split(' (')[0] : null;

  dets.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.box;
    const w = Math.max(1, x2 - x1), h = Math.max(1, y2 - y1);
    const g = svg('g', { class: 'box' });
    const subject = check.truck_dominant && d === biggest;
    g.dataset.subject = String(!!subject);
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
       renders at its real size whatever the frame is shown at. */
    const chip = el('span', 'box-chip', `${d.label} ${fixed(d.confidence, 2)}`);
    chip.dataset.subject = String(!!subject);
    if (blockedLabel && d.label === blockedLabel) chip.dataset.disqualifying = 'true';
    chip.style.left = `${(x1 / check.width) * 100}%`;
    chip.style.top = `${(y1 / check.height) * 100}%`;
    /* a box hugging the top edge has no room above it for its own label */
    if (y1 / check.height > 0.07) chip.classList.add('above');
    chips.append(chip);

    animate(r, { strokeDashoffset: [perim, 0] },
            { duration: 0.55, delay: 0.12 + i * 0.09, ease: [0.2, 0.7, 0.3, 1] });
    animate(chip, { opacity: [0, 1] }, { duration: 0.25, delay: 0.55 + i * 0.09 });
  });
}

function writeMeta(check) {
  const meta = $('frame-meta');
  const q = check.capture_quality;
  meta.replaceChildren();
  meta.append(el('span', 'fm-name', check.filename || `photo ${check.photo_id}`));
  meta.append(el('span', 'fm-view',
                 check.usable ? viewName(check.view) : 'dropped by the gate'));

  const row = el('span', 'fm-q');
  row.append(el('span', 'fm-q-label', 'capture'));
  const bar = el('span', 'fm-q-bar');
  bar.dataset.bucket = check.quality_bucket || 'unknown';
  const fill = el('i');
  bar.append(fill);
  row.append(bar);
  row.append(el('span', 'fm-q-val', fixed(q, 2)));
  meta.append(row);
  requestAnimationFrame(() => {
    fill.style.width = `${Math.round(Math.max(0, Math.min(1, q || 0)) * 100)}%`;
  });
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
      ? `${c.filename} — ${viewName(c.view)} (${c.quality_bucket})`
      : `${c.filename} — dropped: ${(c.reasons || []).join('; ')}`;
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
  /* One class, one CSS transition, staggered - the frames deal in rather than
     appearing all at once, which is how you see how many there are. */
  cells.forEach((c, i) => setTimeout(() => c.classList.add('in'), 40 + i * 35));
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
  $('photo-count').textContent =
    `${(gate.usable_photo_ids || []).length} used of ${all.length}`;
  grid.replaceChildren(...all.map((c) => {
    const b = el('button', 'thumb' + (c.usable ? '' : ' dropped'));
    b.type = 'button';
    b.id = `thumb-${c.photo_id}`;
    b.title = c.usable
      ? `${c.filename} — ${viewName(c.view)} (${c.quality_bucket})`
      : `${c.filename} — dropped: ${(c.reasons || []).join('; ')}`;
    const img = el('img');
    img.src = urlFor(c.photo_id);
    img.alt = c.usable ? viewName(c.view) : `dropped: ${(c.reasons || []).join('; ')}`;
    img.loading = 'lazy';
    b.append(img, el('span', 'thumb-tag', c.usable ? viewName(c.view) : 'dropped'));
    b.addEventListener('click', () => onPick(c));
    return b;
  }));
}

export function openLightbox(check) {
  $('lightbox-img').src = urlFor(check.photo_id);
  $('lightbox-img').alt = viewName(check.view);
  $('lightbox-cap').textContent = check.usable
    ? `${check.filename} — ${viewName(check.view)} · ${check.quality_bucket} capture`
      + (check.truck_dominant ? ` · truck detected ${fixed(check.truck_conf, 2)}` : '')
    : `${check.filename} — dropped: ${(check.reasons || []).join('; ')}`;
  $('lightbox').hidden = false;
}

/* Flash the thumbnail a finding cites, then open it. */
export function citePhoto(id) {
  const node = $(`thumb-${id}`);
  const check = checkFor(id);
  if (!node) return;
  node.classList.remove('flash');
  void node.offsetWidth;
  node.classList.add('flash');
  node.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  if (check) openLightbox(check);
}
