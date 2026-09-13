/* The run, against the real clock.

   Measured on a 21-photo set: gate 1.2 s, then sixteen concurrent close-up
   calls, then a synthesis. The gate report lands about a second in and every
   photo lands as its own event, so for the first time this screen has enough
   real material to fill the wait without inventing any.

   Three channels, all of them sourced from something that has actually
   returned:

     the drawing   fills in as the gate's view coverage binds to it, then
                   again as findings arrive
     the frame     stays on the photograph being read; when that call
                   returns the yellow box (if any) lands, then the next
     the rail      one card per finished vision call, newest on top
                   plus a pop on the photograph itself

   The progress figure is a count of finished calls, not an easing curve. The
   old bar eased asymptotically towards 95% and was labelled an estimate
   because nothing knew when the single call would answer; sixteen calls know
   exactly how many of them are done. */

import { $, el, timers, reduced, viewName } from './dom.js';
import * as elevation from './elevation.js';
import * as frames from './frames.js';
import * as reasoning from './reasoning.js';

let runElev = null;
let t = timers();
let gate = null;
let evidenceIds = [];
let read = 0;
let following = true;
let latest = null;
let reviewQueue = [];
let reviewTimer = null;
let sleepResolve = null;
let reviewed = 0;
let finished = false;
let presenting = false;
let reviewGen = 0;
const received = new Set();
const activePhotos = new Set();
let activityDetail = '';

function ensureScanDots() {
  const field = $('scan-field');
  if (!field) return;
  if (field.dataset.kind === 'pixels') return;
  field.replaceChildren();
  field.dataset.kind = 'pixels';
  const cols = 26;
  const rows = 18;
  const span = cols + rows - 2;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const nx = (x + 0.5) / cols;
      const ny = (y + 0.5) / rows;
      const dist = Math.min(1, Math.hypot(nx - 0.5, ny - 0.5) / 0.72);
      const dot = el('i', 'scan-dot');
      dot.style.setProperty('--x', `${(nx * 100).toFixed(2)}%`);
      dot.style.setProperty('--y', `${(ny * 100).toFixed(2)}%`);
      dot.style.setProperty('--s', `${(7.4 - dist * 5.1).toFixed(2)}px`);
      dot.style.setProperty('--diag', ((x + y) / span).toFixed(4));
      field.append(dot);
    }
  }
}

function setScanning(on) {
  const scan = $('scan');
  if (on) {
    ensureScanDots();
    scan.classList.add('on');
  } else {
    scan.classList.remove('on');
  }
}

function showActivePhoto() {
  if (!following || hasPendingReview() || !activePhotos.size) return;
  const current = frames.currentPhotoId();
  const id = activePhotos.has(current) ? current : [...activePhotos][0];
  const check = frames.checkFor(id);
  if (check) {
    frames.showFrame(check); frames.markCell(id);
    setScanning(true);
    const n = activePhotos.size;
    $('scan-status').textContent = n > 1
      ? `${viewName(check.view)} · ${n} active`
      : viewName(check.view);
  }
}

export function onActivity(msg) {
  if (msg.phase === 'views') {
    frames.applyViews(msg.photos);
    return;
  }
  if (msg.phase === 'identity') {
    reasoning.identity(msg);
    const who = [msg.make, msg.model].filter(Boolean).join(' ');
    const n = msg.sample;
    const of = msg.of;
    if (msg.status === 'reading' && n && of) {
      activityDetail = `Identity ${n} of ${of}`;
      progress(activityDetail, 0.12 + 0.08 * ((n - 1) / of));
    } else if (who) {
      activityDetail = who;
      if (n && of) progress(who, 0.12 + 0.08 * (n / of));
    } else if (msg.detail) {
      activityDetail = msg.detail;
    }
    if (!hasPendingReview()) $('scan-status').textContent = activityDetail;
    return;
  }
  if (msg.phase === 'photo') {
    activePhotos.add(msg.photo_id);
    reasoning.reading(msg.photo_id);
    if (hasPendingReview()) return;
    const current = frames.currentPhotoId();
    if (!current || !activePhotos.has(current) || activePhotos.size === 1) {
      showActivePhoto();
    } else {
      setScanning(true);
    }
    return;
  }
  activityDetail = msg.detail || '';
  if (msg.phase === 'synthesis') {
    activePhotos.clear();
  }
  if (!hasPendingReview()) $('scan-status').textContent = activityDetail;
}


export const hasPendingReview = () => reviewQueue.length > 0 || presenting;

function cancelPresentation() {
  reviewGen += 1;
  clearTimeout(reviewTimer);
  reviewTimer = null;
  if (sleepResolve) {
    const done = sleepResolve;
    sleepResolve = null;
    done(false);
  }
  presenting = false;
  reasoning.hidePop();
  reasoning.clearPlayhead();
}

function sleep(gen, ms) {
  return new Promise((resolve) => {
    clearTimeout(reviewTimer);
    sleepResolve = (ok) => resolve(ok && gen === reviewGen);
    reviewTimer = setTimeout(() => {
      reviewTimer = null;
      const done = sleepResolve;
      sleepResolve = null;
      if (done) done(gen === reviewGen);
    }, ms);
  });
}

async function presentNext() {
  if (!following || presenting || !reviewQueue.length) return;
  presenting = true;
  const gen = ++reviewGen;
  const finding = reviewQueue.shift();
  const check = frames.checkFor(finding.photo_id);
  if (!check) {
    presenting = false;
    return presentNext();
  }
  reviewed += 1;
  const painted = await frames.showFrame(check);
  if (gen !== reviewGen) return;
  if (!following) { presenting = false; return; }
  frames.markCell(check.photo_id, 'read');
  reasoning.markPlayhead(finding.photo_id);
  $('scan-status').textContent = `${viewName(check.view)} · photo ${frames.photoOrdinal(check.photo_id)}`;
  setScanning(false);
  const boxed = (finding.issues || []).filter((i) => Array.isArray(i.box) && i.box.length === 4).length;
  /* The yellow box traces on before the card pops, and the next photograph
     waits until that has been seen — unless the frame is clean. */
  if (boxed && painted && !reduced()) {
    if (!await sleep(gen, 420)) return;
    if (!following) { presenting = false; return; }
  }
  elevation.setFindings(runElev, finding.issues || []);
  reasoning.showPop(finding);
  const hold = boxed
    ? Math.max(2600, Math.min(7600, boxed * 1600 + 700))
    : 1100;
  if (!await sleep(gen, hold)) return;
  if (!following) { presenting = false; return; }
  reasoning.hidePop();
  presenting = false;
  if (reviewQueue.length) presentNext();
  else {
    reasoning.clearPlayhead();
    $('scan-status').textContent = finished
      ? 'Done'
      : activityDetail || `${read} of ${evidenceIds.length}`;
    showActivePhoto();
  }
}


function liveBtn() { return $('follow-live'); }

function setStep(step) {
  const order = ['gate', 'evidence', 'price'];
  const here = order.indexOf(step);
  document.querySelectorAll('[data-step]').forEach((node) => {
    const index = order.indexOf(node.dataset.step);
    const state = index < here ? 'done'
      : node.dataset.step === step ? 'active' : 'waiting';
    node.dataset.state = state;
    if (state === 'active') node.setAttribute('aria-current', 'step');
    else node.removeAttribute('aria-current');
  });
}

/* Which stops actually happened, read off the appraisal rather than assumed.
   A refusal stops after the gate, so closing the trail by marking all three
   finished told a viewer the photographs had been sent on the one screen whose
   whole point is that they were not. */
function closeSteps(a) {
  const ran = {
    gate: true,
    evidence: !!a.evidence,
    price: !!(a.price && a.price.ok),
  };
  document.querySelectorAll('[data-step]').forEach((node) => {
    node.dataset.state = ran[node.dataset.step] ? 'done' : 'skipped';
    node.removeAttribute('aria-current');
  });
}

function follow(value) {
  following = value;
  const button = liveBtn();
  if (!button) return;
  button.setAttribute('aria-pressed', String(value));
  button.textContent = value ? 'Following' : 'Paused';
  /* The overlay names the photo being presented. Clicking a thumb pauses
     that, so a leftover "dashboard · photo 5" on a side exterior would be
     describing the wrong frame. */
  $('scan-status').hidden = !value;
  if (!value) {
    cancelPresentation();
    setScanning(false);
  }
}
function selectFrame(c) { follow(false); frames.showFrame(c); frames.markCell(c.photo_id); }

export async function mount() {
  runElev = await elevation.mount($('run-elev'));
  const live = liveBtn();
  if (live) {
    live.addEventListener('click', () => {
      follow(!following);
      if (following && latest) {
        if (reviewQueue.length) presentNext();
        else { frames.showFrame(latest); frames.markCell(latest.photo_id); }
      }
    });
  }
  return runElev;
}
export const elevationRoot = () => runElev;
export const isFollowing = () => following;

function progress(label, fraction) {
  $('progress-label').textContent = label;
  $('progress-fill').style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
}

/* ---------- lifecycle ---------- */

export function begin() {
  stop();
  gate = null;
  evidenceIds = [];
  read = 0;
  latest = null;
  reviewed = 0;
  finished = false;
  received.clear(); activePhotos.clear(); activityDetail = '';
  follow(true);
  setStep('gate');
  if (liveBtn()) liveBtn().hidden = false;
  $('run').hidden = false;
  $('run').classList.remove('is-done');
  $('progress').hidden = false;
  $('scan-status').textContent = 'Checking photos';
  $('scan-status').hidden = false;
  $('result').hidden = true;
  $('strip').replaceChildren();
  $('frame-boxes').replaceChildren();
  $('frame-chips').replaceChildren();
  $('frame-meta').replaceChildren();
  $('frame-img').removeAttribute('src');
  $('frame-img').classList.remove('in');
  $('show-all').hidden = true;
  $('show-all').setAttribute('aria-pressed', 'false');
  frames.setSource({}, [], null);
  frames.setShowAll(false);
  elevation.reset(runElev);
  reasoning.clear();
  progress('Checking photos', 0.04);
  $('run').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
}

export function stop() {
  activePhotos.clear();
  reviewQueue = [];
  cancelPresentation();
  t.clear();
  setScanning(false);
  frames.cancelPendingFrame();
  document.querySelectorAll('.strip-cell.reading').forEach((cell) => cell.classList.remove('reading'));
}

/* ---------- act 1: the gate has landed ---------- */

export function onGate(msg) {
  gate = msg.gate;
  evidenceIds = msg.evidence_photo_ids || [];
  frames.setSource(msg.photo_urls, gate.photos, gate.decision, evidenceIds);

  elevation.setCoverage(runElev, gate.views_present);
  frames.buildStrip(gate.photos, selectFrame);

  /* Lead with the frame the verdict turns on: on a not-a-truck refusal that is
     whatever the gate identified instead, which is the whole explanation. */
  const first = (gate.decision === 'refuse_not_a_truck' && frames.smokingGun())
    || (gate.photos || []).find((c) => c.usable)
    || gate.photos[0];
  if (first) { frames.showFrame(first); frames.markCell(first.photo_id); }

  const usable = (gate.usable_photo_ids || []).length;
  const total = (gate.photos || []).length;
  progress(usable === total ? `${total} photos` : `${usable} of ${total} usable`, 0.1);
}

/* ---------- act 2: each photo's own call ---------- */

export function onStage(msg) {
  setStep(msg.step);
  if (msg.step === 'evidence') {
    reasoning.begin(evidenceIds);
    evidenceIds.forEach((id) => {
      const c = $(`cell-${id}`);
      if (c) c.classList.add('reading');
    });
    setScanning(true);
    $('scan-status').textContent = 'Waiting for the first frame';
    progress(`0 of ${evidenceIds.length}`, 0.12);
  }
  if (msg.step === 'price') {
    if (!hasPendingReview()) $('scan-status').textContent = 'Pricing';
    setScanning(false);
    progress(`${read} of ${evidenceIds.length || read}`, 0.96);
  }
}

/* One photo's own vision call has returned. */
export function onPhoto(msg) {
  const finding = msg.finding;
  if (!finding || received.has(finding.photo_id)) return;
  received.add(finding.photo_id);
  activePhotos.delete(finding.photo_id);
  frames.setFinding(finding);
  reasoning.add(finding);
  read += 1;
  if (!hasPendingReview()) {
    $('scan-status').textContent = `${read} of ${evidenceIds.length || read}`;
  }

  const check = frames.checkFor(finding.photo_id);
  if (check) {
    latest = check;
    const cell = $(`cell-${finding.photo_id}`);
    if (cell) { cell.classList.add('read'); cell.classList.remove('reading'); }
    reviewQueue.push(finding);
    presentNext();
  }
  /* The truck drawing updates when presentNext actually shows this finding,
     not on raw arrival - see elevation.setFindings inside presentNext. Only
     the rail (reasoning.add, above) stays eager; the diagram reads as "the
     verdict" and is the thing most damaged by finishing the story early. */

  const total = evidenceIds.length || read;
  progress(`${read} of ${total}`, 0.12 + 0.8 * (read / Math.max(1, total)));
}

/* ---------- act 3: it answered ---------- */

export function onResult(a) {
  finished = true;
  activePhotos.clear();
  t.clear();
  setScanning(false);
  $('scan-status').textContent = a.evidence ? 'Done' : 'Stopped';
  $('run').classList.add('is-done');
  progress(read ? `${read} photos` : 'Stopped', 1);
  closeSteps(a);
  if (liveBtn()) liveBtn().hidden = !a.evidence;
  const ev = a.evidence;
  if (ev) {
    elevation.setSummarised(runElev, ev.condition_summary);
    elevation.setFindings(runElev, ev.issues);
  } else {
    reasoning.note(a.gate && a.gate.headline
      ? a.gate.headline
      : 'The checks in front of the vision model stopped this first.');
  }
  reasoning.done(ev);
}

/* A frozen export has no server and no stream: paint the finished state
   directly. Same functions the live run uses, so the export cannot drift from
   what the demo shows. */
export function showFrozen(a) {
  stop();
  gate = a.gate;
  evidenceIds = [];
  $('run').hidden = false;
  $('progress').hidden = true;
  if (liveBtn()) liveBtn().hidden = true;
  frames.setSource(a.photo_urls, gate.photos, gate.decision);
  elevation.reset(runElev);
  elevation.setCoverage(runElev, gate.views_present, { stagger: 0 });
  const lead = (gate.decision === 'refuse_not_a_truck' && frames.smokingGun())
    || (gate.photos || []).find((c) => c.usable) || gate.photos[0];
  frames.buildStrip(gate.photos, selectFrame);
  /* The rail is rebuilt from the findings the export carries, so a frozen file
     shows the same per-photo reasoning the live run showed. */
  const ev = a.evidence;
  if (ev && (ev.photo_findings || []).length) {
    ev.photo_findings.forEach((f) => frames.setFinding(f));
    reasoning.begin(ev.photo_findings.map((f) => f.photo_id));
    [...ev.photo_findings].reverse().forEach((f) => reasoning.add(f));
  }
  if (lead) { frames.showFrame(lead); frames.markCell(lead.photo_id); }
  /* render() calls onResult itself, so the drawing and the rail's closing
     state both land through the one path. */
}

export function onError(message) {
  stop();
  $('scan-status').textContent = 'Inspection interrupted';
  reasoning.note(message);
  if (liveBtn()) liveBtn().hidden = true;
  progress(message, 0);
}
