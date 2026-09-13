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
function setStep(step) {
  const steps = ['gate', 'evidence', 'price'];
  document.querySelectorAll('[data-step]').forEach(node => {
    const index = steps.indexOf(node.dataset.step);
    node.dataset.state = step === 'done' || index < steps.indexOf(step) ? 'done' : node.dataset.step === step ? 'active' : 'waiting';
    if (node.dataset.state === 'active') node.setAttribute('aria-current', 'step');
    else node.removeAttribute('aria-current');
  });
}
function follow(value) {
  following = value;
  const button = $('follow-live');
  button.setAttribute('aria-pressed', String(value));
  button.textContent = value ? 'Following live' : 'Resume live view';
}
function selectFrame(c) { follow(false); frames.showFrame(c); frames.markCell(c.photo_id); }

export async function mount() {
  runElev = await elevation.mount($('run-elev'));
  $('follow-live').addEventListener('click', () => {
    follow(!following);
    if (following && latest) { frames.showFrame(latest); frames.markCell(latest.photo_id); }
  });
  return runElev;
}
export const elevationRoot = () => runElev;

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
  $('follow-live').hidden = false;
  $('run').hidden = false;
  $('result').hidden = true;
  $('strip').replaceChildren();
  $('frame-boxes').replaceChildren();
  $('frame-chips').replaceChildren();
  $('frame-meta').replaceChildren();
  $('frame-img').removeAttribute('src');
  $('frame-img').classList.remove('in');
  $('show-all').hidden = true;
  $('show-all').setAttribute('aria-pressed', 'false');
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
    progress(`reading ${evidenceIds.length} photos`, 0.12);
  }
  if (msg.step === 'price') {
    $('scan').classList.remove('on');
    $('rail-title').textContent = 'Matching it against real listings';
    $('rail-sub').textContent = 'the photos are read';
    progress('pricing it against comparable trucks', 0.96);
  }
}

/* One photo's own vision call has returned. */
export function onPhoto(msg) {
  const finding = msg.finding;
  reasoning.add(finding);
  read += 1;

  const check = frames.checkFor(finding.photo_id);
  if (check) {
    latest = check;
    const cell = $(`cell-${finding.photo_id}`);
    if (cell) { cell.classList.add('read'); cell.classList.remove('reading'); }
    if (following) { frames.showFrame(check); frames.markCell(finding.photo_id, 'read'); }
  }
  elevation.setFindings(runElev, finding.issues || []);

  const total = evidenceIds.length || read;
  progress(`${read} of ${total} photos read`, 0.12 + 0.8 * (read / Math.max(1, total)));
}

/* ---------- act 3: it answered ---------- */

export function onResult(a) {
  stop();
  progress('done', 1);
  setStep('done');
  $('follow-live').hidden = true;
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
  frames.setSource(a.photo_urls, gate.photos, gate.decision);
  elevation.reset(runElev);
  elevation.setCoverage(runElev, gate.views_present, { stagger: 0 });
  const lead = (gate.decision === 'refuse_not_a_truck' && frames.smokingGun())
    || (gate.photos || []).find((c) => c.usable) || gate.photos[0];
  frames.buildStrip(gate.photos, selectFrame);
  if (lead) { frames.showFrame(lead); frames.markCell(lead.photo_id); }
  /* The rail is rebuilt from the findings the export carries, so a frozen file
     shows the same per-photo reasoning the live run showed. */
  const ev = a.evidence;
  if (ev && (ev.photo_findings || []).length) {
    reasoning.begin(ev.photo_findings.length);
    [...ev.photo_findings].reverse().forEach((f) => reasoning.add(f));
  }
  /* render() calls onResult itself, so the drawing and the rail's closing
     state both land through the one path. */
}

export function onError(message) {
  stop();
  reasoning.note(message);
  $('follow-live').hidden = true;
  progress(message, 0);
  $('rail-title').textContent = 'Something broke';
  $('rail-sub').textContent = message;
}
