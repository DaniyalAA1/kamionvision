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
   brings the rest of the boxes back for anyone who wants them.

   The subject chip names the camera angle (front, side, rear ¾), not a
   generic "being appraised" caption. Inner boxes are findings only, and
   only the active one is drawn - a stack of healthy component regions
   inside the subject is how the overlay became unreadable.

   A close-up often has no truck-shaped object. Those frames still get an
   angle box: the photograph itself is the subject, inset so the stroke
   does not sit on the clip edge.

   A detected box can also fill nearly the whole frame - a tire or dashboard
   close-up where the "vehicle" YOLO boxed is the close-up itself
   (app/subject.py measures this: 455 of 773 boxed part-view frames sit
   above 80% of the frame). Past SUBJECT_FILLS_FRAME that box draws the same
   way as the no-detection case above, the photo's own inset edges, rather
   than its real and usually lopsided coordinates - past that size a
   rectangle distinguishes nothing from a background. */

import { $, el, svg, viewName, viewAngle, fixed, animate, reduced, titleise } from './dom.js';

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

const SEV_RANK = { cosmetic: 0, minor: 1, moderate: 2, major: 3 };

function validBox(box) {
  return Array.isArray(box) && box.length === 4 &&
    box.every(Number.isFinite) && box[0] >= 0 && box[1] >= 0 &&
    box[2] > 0 && box[3] > 0 && box[0] + box[2] <= 1.001 && box[1] + box[3] <= 1.001;
}

/* Findings only, one box per component. Healthy `component_regions` used to
   stack eight faded rectangles inside the subject and drown the angle mark. */
function locatedParts(finding) {
  if (!finding || finding.error) return [];
  const best = new Map();
  for (const issue of finding.issues || []) {
    if (!validBox(issue.box)) continue;
    const prev = best.get(issue.component);
    if (!prev || (SEV_RANK[issue.severity] || 0) > (SEV_RANK[prev.severity] || 0)) {
      best.set(issue.component, { ...issue, kind: 'finding' });
    }
  }
  return [...best.values()].sort(
    (a, b) => (SEV_RANK[b.severity] || 0) - (SEV_RANK[a.severity] || 0));
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
    drawBoxes($('frame-boxes'), current);
  }
}

export function applyViews(photos) {
  for (const patch of photos || []) {
    const check = checkFor(patch.photo_id);
    if (!check) continue;
    if (patch.view) check.view = patch.view;
    if (patch.subject_box === null) {
      check.subject_box = null;
      (check.detections || []).forEach((d) => { d.is_subject = false; });
    }
  }
  if (current) {
    drawBoxes($('frame-boxes'), current);
    writeMeta(current);
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

function paintFrame(check, source) {
  current = check;
  const img = $('frame-img');
  img.alt = check.usable ? viewName(check.view) : 'Dropped';
  img.src = source;
  img.classList.add('in');
  if (check.width && check.height) {
    $('frame-boxes').setAttribute('viewBox', `0 0 ${check.width} ${check.height}`);
  }
  fitChips();
  drawBoxes($('frame-boxes'), check);
  writeMeta(check);
  const toggle = $('show-all');
  if (toggle) toggle.hidden = !hasHiddenBoxes(check);
}

export function showFrame(check) {
  if (!check) return Promise.resolve(false);
  const source = urlFor(check.photo_id);
  const img = $('frame-img');
  /* Same photograph already on the stage: redraw the boxes without a
     dissolve, so a finding can land on the frame that was being scanned. */
  if (current && current.photo_id === check.photo_id && img.getAttribute('src') === source) {
    clearPartTour();
    frameVersion += 1;
    paintFrame(check, source);
    return Promise.resolve(true);
  }
  clearPartTour();
  const version = ++frameVersion;
  return new Promise((resolve) => {
    let settled = false;
    const settle = (ok) => {
      if (settled) return;
      settled = true;
      resolve(ok);
    };
    const preload = new Image();
    preload.onload = () => {
      if (version !== frameVersion) return settle(false);
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
        setTimeout(() => previous.remove(), 600);
      }
      paintFrame(check, source);
      settle(true);
    };
    preload.onerror = () => {
      if (version !== frameVersion) return settle(false);
      current = null;
      $('show-all').hidden = true;
      img.removeAttribute('src');
      img.alt = 'This photo failed to load';
      $('frame-boxes').replaceChildren();
      $('frame-chips').replaceChildren();
      $('frame-meta').textContent = 'This photo failed to load. Select another photo to continue.';
      settle(false);
    };
    preload.src = source;
  });
}

// Keep HTML labels aligned with the contained image, including portrait photos.
function fitChips() {
  if (!current?.width || !current?.height) return;
  const stage = $('frame-stage');
  const scale = Math.min(stage.clientWidth / current.width, stage.clientHeight / current.height);
  const inset = `${(stage.clientHeight - current.height * scale) / 2}px ${(stage.clientWidth - current.width * scale) / 2}px`;
  $('frame-chips').style.inset = inset;
  const field = $('scan-field');
  if (field) field.style.inset = inset;
}
new ResizeObserver(fitChips).observe($('frame-stage'));

/* Fallback only: a frozen export written before `is_subject` existed carries
   the subject as coordinates and nothing else. */
const sameBox = (a, b) =>
  !!a && !!b && a.length === 4 && b.length === 4 && a.every((v, i) => v === b[i]);

const subjectOf = (all, check) =>
  all.find((d) => d.is_subject) ||
  (check.subject_box ? all.find((d) => sameBox(d.box, check.subject_box)) : undefined);

/* Every photo gets an angle mark. YOLO's subject wins; then the gate's
   subject_box; then the frame itself, because a tire close-up is still
   a photo of the truck and used to render with no box at all. */
function angleBox(check, detected) {
  if (detected && detected.box) return detected.box;
  if (Array.isArray(check.subject_box) && check.subject_box.length === 4)
    return check.subject_box;
  if (check.width && check.height) {
    const pad = Math.max(8, Math.min(check.width, check.height) * 0.015);
    return [pad, pad, check.width - pad, check.height - pad];
  }
  return null;
}

/* The photo's own edges, inset so the stroke does not sit on the clip edge -
   the same fallback `angleBox` reaches for when nothing was detected at all,
   reused below for a detected box so large there is nothing left for it to
   distinguish. */
function wholeFrameBox(check) {
  if (!check.width || !check.height) return null;
  const pad = Math.max(8, Math.min(check.width, check.height) * 0.015);
  return [pad, pad, check.width - pad, check.height - pad];
}

/* Share of the frame a box covers. */
function boxFrac(box, check) {
  if (!box || !check.width || !check.height) return 0;
  const [x1, y1, x2, y2] = box;
  return (Math.max(1, x2 - x1) * Math.max(1, y2 - y1)) / (check.width * check.height);
}

/* Above this share of the frame, a box stops distinguishing anything and
   starts being the photo's own edges - most often a tire or dashboard
   close-up where the "vehicle" box IS the close-up (app/subject.py: 455 of
   773 boxed part-view frames sit above 80% of the frame). Also the cutoff
   below for skipping the dim overlay, which would otherwise paint a bare
   rim around a box this size. */
const SUBJECT_FILLS_FRAME = 0.85;

function drawBoxes(root, check) {
  const shownSrc = $('frame-img')?.getAttribute('src') || '';
  const want = urlFor(check.photo_id);
  if (!want || shownSrc !== want) return;

  root.replaceChildren();
  const chips = $('frame-chips');
  chips.replaceChildren();
  const all = (check.detections || []).filter(
    (d) => d.box && d.box.length === 4 && check.width && check.height);

  const detected = subjectOf(all, check);
  const box = angleBox(check, detected);
  const subject = detected || (box ? { box } : null);
  const blockedLabel = (refusedAsNotATruck && check.non_truck_subject)
    ? String(check.non_truck_subject).split(' (')[0] : null;
  const disqualifying = blockedLabel
    ? all.filter((d) => d.label === blockedLabel) : [];

  /* Default: the subject, plus whatever the gate refused on. Everything else
     is behind the toggle. */
  const extras = showAll ? all : disqualifying;
  const shown = [subject, ...extras].filter((d, i, list) => d && list.indexOf(d) === i);

  /* Only the drawn geometry changes here - `subject` keeps its identity, so
     the dedup above and the "show everything it detected" toggle still see
     the real detection at its real coordinates. */
  const subjectFillsFrame = !!subject && boxFrac(subject.box, check) >= SUBJECT_FILLS_FRAME;
  const subjectBox = subject && (subjectFillsFrame ? wholeFrameBox(check) : subject.box);

  /* Darken the frame outside the subject, drawn as one even-odd path so the
     subject stays at full brightness without a second image. Skipped when
     the subject already fills the frame - dimming that would only paint a
     rim. */
  if (subject && !showAll && !subjectFillsFrame && check.width && check.height) {
    const [x1, y1, x2, y2] = subjectBox;
    const outer = `M0 0H${check.width}V${check.height}H0Z`;
    const inner = `M${x1} ${y1}H${x2}V${y2}H${x1}Z`;
    const dim = svg('path', { class: 'dim', d: `${outer} ${inner}`, 'fill-rule': 'evenodd' });
    root.append(dim);
    animate(dim, { opacity: [0, 1] }, { duration: 0.45, delay: 0.1 });
  }

  shown.forEach((d, i) => {
    const isSubject = d === subject;
    const [x1, y1, x2, y2] = isSubject ? subjectBox : d.box;
    const w = Math.max(1, x2 - x1), h = Math.max(1, y2 - y1);
    const g = svg('g', { class: 'box' });
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
    const text = isSubject ? viewAngle(check.view)
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
    if (!animate(chip, { opacity: [0, 1] }, { duration: 0.25, delay: 0.5 + i * 0.07 })) {
      chip.style.opacity = '1';
    }
  });

  /* One inner box at a time: the worst finding, then the next if the viewer
     stays on the frame. Healthy regions are not drawn. */
  clearPartTour();
  locatedParts(findings[check.photo_id]).forEach((part, index) => {
    const [x, y, w, h] = part.box;
    const g = svg('g', { class: 'scan-part' });
    g.dataset.partIndex = String(index);
    g.dataset.kind = part.kind;
    const rect = svg('rect', { x: x * check.width, y: y * check.height,
      width: w * check.width, height: h * check.height, pathLength: 1 });
    g.append(rect); root.append(g);
    const chip = el('span', 'box-chip scan-part-label',
      `${partName(part.component)} · ${part.severity}`);
    chip.dataset.partIndex = String(index);
    chip.style.left = `${Math.min(x, .65) * 100}%`;
    chip.style.top = `${y * 100}%`;
    if (y > .07) chip.classList.add('above');
    chips.append(chip);
    if (index) partTimers.push(setTimeout(() => focusPart(index), index * 1600));
  });
  focusPart(0);
}

function writeMeta(check) {
  const meta = $('frame-meta');
  meta.replaceChildren();
  meta.append(el('span', 'fm-view',
                 check.usable ? viewName(check.view) : 'Dropped'));
  if (!check.usable && (check.reasons || []).length) {
    meta.append(el('span', 'fm-why', check.reasons[0]));
  }
  if (check.usable && evidencePhotos && !evidencePhotos.has(check.photo_id)) {
    meta.append(el('span', 'fm-why', 'Checked, not sent for a close-up'));
  }
  const word = CAPTURE_WORD[check.quality_bucket];
  if (word) {
    const cap = el('span', 'fm-capture', word);
    cap.dataset.bucket = check.quality_bucket;
    meta.append(cap);
  }
}

const WORST = (issues) => issues.some((i) => i.severity === 'major') ? 'major'
  : issues.some((i) => i.severity === 'moderate') ? 'moderate'
  : issues.some((i) => i.severity === 'minor') ? 'minor' : 'cosmetic';

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
    b.append(img, el('span', 'thumb-tag', c.usable ? viewName(c.view) : 'dropped'));
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
    cap = `Dropped: ${(check.reasons || []).join('; ')}`;
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
