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
let refusedAsNotATruck = false;
let showAll = false;
let current = null;
let frameVersion = 0;

export function setSource(photoUrls, photoChecks, decision) {
  frameVersion += 1;
  if (!current || urls[current.photo_id] !== photoUrls?.[current.photo_id]) current = null;
  urls = photoUrls || {};
  checks = photoChecks || [];
  findings = {};
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

  /* Defect annotations from the vision analysis */
  const finding = findings[check.photo_id];
  if (finding && finding.issues && check.width && check.height) {
    finding.issues.forEach((issue, idx) => {
      if (issue.box && Array.isArray(issue.box) && issue.box.length === 4) {
        const [bx, by, bw, bh] = issue.box;
        const ix = bx * check.width, iy = by * check.height;
        const iw = bw * check.width, ih = bh * check.height;
        if (iw > 6 && ih > 6) {
          const dg = svg('g', { class: 'defect-box-group' });
          dg.dataset.sev = issue.severity || 'minor';
          dg.dataset.issueId = `${check.photo_id}-${idx}`;
          const dr = svg('rect', {
            x: ix, y: iy, width: iw, height: ih,
            class: `defect-rect ${issue.severity || 'minor'}`
          });
          dg.append(dr);
          root.append(dg);

          const dchip = el('span', `box-chip defect-chip ${issue.severity || 'minor'}`,
                           `${titleise(issue.component)}: ${issue.severity}`);
          dchip.dataset.issueId = `${check.photo_id}-${idx}`;
          dchip.style.left = `${(ix / check.width) * 100}%`;
          dchip.style.top = `${(iy / check.height) * 100}%`;
          if (iy / check.height > 0.07) dchip.classList.add('above');
          chips.append(dchip);
        }
      }
    });
  }
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

export function renderIntel(check, finding) {
  const container = $('frame-intel');
  if (!container) return;
  container.replaceChildren();
  if (!check) return;

  const card = el('div', 'intel-card');

  // Top bar with live status badge and telemetry
  const top = el('div', 'intel-top');
  const badge = el('span', 'intel-badge');
  if (!finding) {
    badge.className = 'intel-badge scanning';
    const dot = el('span', 'pulse-dot');
    badge.append(dot, document.createTextNode(check.usable ? 'Scanning in progress…' : 'Dropped by gate'));
  } else if (finding.error) {
    badge.className = 'intel-badge error';
    badge.textContent = 'Unreadable frame';
  } else if (!finding.issues || !finding.issues.length) {
    badge.className = 'intel-badge clean';
    badge.textContent = 'Verified Sound · Clear';
  } else {
    const worst = (finding.issues.some((i) => i.severity === 'major')) ? 'major'
      : (finding.issues.some((i) => i.severity === 'moderate')) ? 'moderate' : 'minor';
    badge.className = `intel-badge ${worst}`;
    const n = finding.issues.length;
    badge.textContent = `${n} ${worst} finding${n === 1 ? '' : 's'}`;
  }

  const metaPill = el('span', 'intel-pill view',
    `${viewName(check.view)} · Photo ${photoOrdinal(check.photo_id)} of ${checks.length || 1}`);
  top.append(badge, metaPill);

  if (finding?.odometer_km) {
    const odo = el('span', 'intel-pill odo', `Odometer: ${Math.round(finding.odometer_km).toLocaleString('en-US')} km`);
    top.append(odo);
  }
  if (finding?.cropped) {
    const crop = el('span', 'intel-pill crop', 'Cropped to vehicle');
    top.append(crop);
  }
  if (check.usable && check.quality_bucket) {
    const cap = el('span', 'intel-pill quality', `${CAPTURE_WORD[check.quality_bucket] || check.quality_bucket} photo`);
    top.append(cap);
  }
  card.append(top);

  // Executive summary headline
  const headlineWrap = el('div', 'intel-headline');
  if (finding) {
    const summary = finding.shows || (finding.issues && finding.issues.length
      ? `${finding.issues.length} observation${finding.issues.length === 1 ? '' : 's'} flagged in this frame.`
      : 'Visual inspection confirms this component is in sound working order with no structural defects.');
    headlineWrap.append(el('p', 'intel-summary-text', summary));
  } else {
    headlineWrap.append(el('p', 'intel-summary-text pending',
      check.usable
        ? `Model examining ${viewName(check.view).toLowerCase()} for component wear, surface condition, and alignment.`
        : `Frame dropped: ${(check.reasons || []).join('; ') || 'insufficient vehicle presence'}.`
    ));
  }
  card.append(headlineWrap);

  // Structured diagnostics grid
  if (finding && !finding.error) {
    const grid = el('div', 'intel-grid');

    // Column 1: Verified Sound / Strengths
    const colSound = el('div', 'intel-col sound');
    const soundHead = el('h4', 'intel-col-title');
    soundHead.append(partIcon('check'), el('span', null, 'Verified Sound'));
    colSound.append(soundHead);

    const strengths = finding.strengths || [];
    if (strengths.length) {
      const soundList = el('ul', 'intel-list sound-list');
      strengths.forEach((st) => soundList.append(el('li', null, st)));
      colSound.append(soundList);
    } else if (!finding.issues || !finding.issues.length) {
      colSound.append(el('p', 'intel-quiet', 'All visible surfaces, seals, and mountings appear in expected operational condition.'));
    } else {
      colSound.append(el('p', 'intel-quiet', 'Component integrity acceptable outside the specific items noted.'));
    }
    grid.append(colSound);

    // Column 2: Observations / Issues
    const colIssues = el('div', 'intel-col issues');
    const issuesHead = el('h4', 'intel-col-title');
    const issueCount = (finding.issues || []).length;
    issuesHead.append(partIcon('warning'), el('span', null, `Findings (${issueCount})`));
    colIssues.append(issuesHead);

    if (issueCount) {
      const issuesList = el('ul', 'intel-list issues-list');
      finding.issues.forEach((issue, idx) => {
        const li = el('li', 'intel-issue-item');
        li.dataset.sev = issue.severity || 'minor';
        li.dataset.issueId = `${check.photo_id}-${idx}`;

        const headRow = el('div', 'intel-issue-head');
        headRow.append(
          partIcon(issue.component),
          el('strong', 'intel-issue-comp', titleise(issue.component)),
          el('span', `sev-tag ${issue.severity || 'minor'}`, issue.severity || 'minor')
        );
        li.append(headRow);

        const desc = el('p', 'intel-issue-obs', issue.observation);
        li.append(desc);

        if (issue.magnitude) {
          const magBits = [];
          if (issue.magnitude.state) magBits.push(issue.magnitude.state.replace(/_/g, ' '));
          if (issue.magnitude.extent) magBits.push(issue.magnitude.extent.replace(/_/g, ' '));
          if (issue.magnitude.consumable) magBits.push('consumable');
          if (magBits.length) {
            li.append(el('span', 'intel-mag', magBits.join(' · ')));
          }
        }
        if (issue.price_impact && issue.price_impact !== 'none') {
          li.append(el('span', 'intel-impact', `Price impact: ${issue.price_impact}`));
        }

        li.addEventListener('mouseenter', () => {
          const rect = $('frame-boxes')?.querySelector(`[data-issue-id="${check.photo_id}-${idx}"]`);
          if (rect) rect.classList.add('highlight');
        });
        li.addEventListener('mouseleave', () => {
          const rect = $('frame-boxes')?.querySelector(`[data-issue-id="${check.photo_id}-${idx}"]`);
          if (rect) rect.classList.remove('highlight');
        });

        issuesList.append(li);
      });
      colIssues.append(issuesList);
    } else {
      colIssues.append(el('p', 'intel-quiet', 'No wear, damage, or degradation flagged in this frame.'));
    }
    grid.append(colIssues);

    // Column 3: Angle Blindspots / Limits
    const colGaps = el('div', 'intel-col gaps');
    const gapsHead = el('h4', 'intel-col-title');
    gapsHead.append(partIcon('search'), el('span', null, 'Inspection Limits'));
    colGaps.append(gapsHead);

    const gaps = finding.cannot_tell || [];
    if (gaps.length) {
      const gapsList = el('ul', 'intel-list gaps-list');
      gaps.forEach((gap) => gapsList.append(el('li', null, gap)));
      colGaps.append(gapsList);
    } else {
      colGaps.append(el('p', 'intel-quiet', 'Full component view visible; no significant angle blindspots.'));
    }
    grid.append(colGaps);

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
  c.scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', inline: 'center', block: 'nearest' });
}

export function updateStripCellStatus(finding) {
  const c = $(`cell-${finding.photo_id}`);
  if (!c) return;
  c.classList.remove('reading');
  c.classList.add('read');
  if (finding.error) {
    c.dataset.state = 'error';
  } else if (finding.issues && finding.issues.length) {
    const worst = (finding.issues.some((i) => i.severity === 'major')) ? 'major'
      : (finding.issues.some((i) => i.severity === 'moderate')) ? 'moderate' : 'minor';
    c.dataset.state = worst;
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
