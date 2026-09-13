/* The run, against the real clock.

   Measured on a 21-photo set: gate 1.2 s, then sixteen concurrent close-up
   calls, then a synthesis. The gate report lands about a second in and every
   photo lands as its own event, so for the first time this screen has enough
   real material to fill the wait without inventing any.

   Three channels, all of them sourced from something that has actually
   returned:

     the drawing   fills in as the gate's view coverage binds to it, then
                   again as findings arrive
     the frame     jumps to whichever photo was just read, with one box on it
     the rail      one card per finished vision call, newest on top

   The progress figure is a count of finished calls, not an easing curve. The
   old bar eased asymptotically towards 95% and was labelled an estimate
   because nothing knew when the single call would answer; sixteen calls know
   exactly how many of them are done. */

import { $, timers, reduced } from './dom.js';
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
  const now = $('run-now');
  if (now) {
    now.textContent = {
      gate: 'Checking the photographs',
      evidence: 'Looking at each photo',
      price: 'Matching it against listings',
    }[step] || now.textContent;
  }
}

/* Which stops actually happened, read off the appraisal rather than assumed.
   A refusal stops after the gate, so closing the trail by marking all three
   finished told a viewer the photographs had been read on the one screen whose
   whole point is that they were not - and the refusal saying "no photo was
   sent to a vision model" sat two inches below it. */
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
  const now = $('run-now');
  if (now) {
    now.textContent = a.evidence
      ? 'The photographs are read'
      : 'Stopped before the photographs were read';
  }
}

function follow(value) {
  following = value;
  const button = liveBtn();
  if (!button) return;
  button.setAttribute('aria-pressed', String(value));
  button.textContent = value ? 'Following live' : 'Resume live view';
}
function selectFrame(c) { follow(false); frames.showFrame(c); frames.markCell(c.photo_id); }

export async function mount() {
  runElev = await elevation.mount($('run-elev'));
  const live = liveBtn();
  if (live) {
    live.addEventListener('click', () => {
      follow(!following);
      if (following && latest) {
        frames.showFrame(latest);
        frames.markCell(latest.photo_id);
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
  follow(true);
  setStep('gate');
  if (liveBtn()) liveBtn().hidden = false;
  $('run').hidden = false;
  $('progress').hidden = false;
  $('scan-status').textContent = 'Checking photo quality';
  $('scan-status').hidden = false;
  $('result').hidden = true;
  $('view-result').hidden = true;
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
  $('rail-title').textContent = 'Checking the photos';
  $('rail-sub').textContent = 'before anything is sent anywhere';
  progress('checking the photos', 0.04);
  $('run').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
}

export function stop() {
  t.clear();
  $('scan').classList.remove('on');
  frames.cancelPendingFrame();
  document.querySelectorAll('.strip-cell.reading').forEach((cell) => cell.classList.remove('reading'));
}

/* ---------- act 1: the gate has landed ---------- */

export function onGate(msg) {
  gate = msg.gate;
  evidenceIds = msg.evidence_photo_ids || [];
  frames.setSource(msg.photo_urls, gate.photos, gate.decision);

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
  $('rail-title').textContent = usable === total
    ? `${total} photos, all usable`
    : `${usable} of ${total} photos are usable`;
  $('rail-sub').textContent = gate.truck_evidence || '';
  progress(usable === total ? `${total} photos checked`
                            : `${usable} of ${total} photos are usable`, 0.1);
}

/* ---------- act 2: the photos are read, one call each ---------- */

export function onStage(msg) {
  setStep(msg.step);
  if (msg.step === 'evidence') {
    reasoning.begin(evidenceIds.length);
    evidenceIds.forEach((id) => {
      const c = $(`cell-${id}`);
      if (c) c.classList.add('reading');
    });
    $('scan').classList.add('on');
    $('scan-status').textContent = 'Inspecting photos · findings appear as they return';
    progress(`reading ${evidenceIds.length} photos`, 0.12);
  }
  if (msg.step === 'price') {
    $('scan-status').textContent = 'Inspection complete · comparing market prices';
    $('scan').classList.remove('on');
    $('rail-title').textContent = 'Matching it against real listings';
    $('rail-sub').textContent = 'the photos are read';
    progress('pricing it against comparable trucks', 0.96);
  }
}

/* One photo's own vision call has returned. */
export function onPhoto(msg) {
  const finding = msg.finding;
  frames.setFinding(finding);
  reasoning.add(finding);
  read += 1;
  $('scan-status').textContent = `${read} of ${evidenceIds.length || read} photos read`;

  const check = frames.checkFor(finding.photo_id);
  if (check) {
    latest = check;
    const cell = $(`cell-${finding.photo_id}`);
    if (cell) { cell.classList.add('read'); cell.classList.remove('reading'); }
    if (following) { frames.showFrame(check); frames.markCell(finding.photo_id, 'read'); }
  }
  elevation.setFindings(runElev, finding.issues || []);

  const total = evidenceIds.length || read;
  if (read >= total) {
    progress('putting the findings together', 0.92);
  } else {
    progress(`${read} of ${total} photos read`, 0.12 + 0.8 * (read / Math.max(1, total)));
  }
}

/* ---------- act 3: it answered ---------- */

export function onResult(a) {
  stop();
  $('view-result').hidden = false;
  $('scan-status').textContent = a.evidence ? 'Inspection complete' : 'Photo checks complete';
  progress(read ? `${read} photos read` : 'the checks finished', 1);
  closeSteps(a);
  if (liveBtn()) liveBtn().hidden = true;
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
    reasoning.begin(ev.photo_findings.length);
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
  $('rail-title').textContent = 'Something broke';
  $('rail-sub').textContent = message;
}
