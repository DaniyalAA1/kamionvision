/* The run, choreographed against the real clock.

   Measured on a 21-photo set: gate 1.2 s, vision call 49.5 s, regression
   0.03 s. So about 97% of a run is one opaque call, and this module exists to
   spend that minute showing work that has actually been done rather than a
   spinner. Three channels, all sourced from data already in hand:

     the drawing      fills in as the gate's view coverage binds to it
     the frames       the detector's own boxes, over the frames being sent
     the narration    one line at a time, never a claim that has not returned

   The clock is the truth. The rule beside it is labelled an estimate and
   eases towards 95% without ever arriving, because nothing here knows when
   the model will answer. */

import { $, el, titleise, viewName, fixed, timers, reduced, animate } from './dom.js';
import * as elevation from './elevation.js';
import * as frames from './frames.js';

/* The three stages that always report while the run is in flight. */
const STEPS = ['gate', 'evidence', 'price'];
/* Every stage the pipeline can put in a trace, in the order it runs them.
   `perception` and `reconcile` only appear when their artifacts are present,
   so the rail is rebuilt from the trace rather than assuming three rows -
   a hardcoded three silently dropped their measured timings. */
const RAIL_ORDER = ['gate', 'perception', 'evidence', 'reconcile', 'price'];
const DWELL_MS = 2600;          // one frame per scan sweep
const EST_TAU = 18;             // seconds; the rule's time constant

let runElev = null;
let t = timers();
let t0 = 0;
let clock = 0;
let gate = null;
let evidenceIds = [];
let reading = false;

export async function mount() {
  await elevation.mount($('intake-elev'));
  runElev = await elevation.mount($('run-elev'));
  return runElev;
}
export const elevationRoot = () => runElev;

/* ---------- stage rail ---------- */

function setStage(step, state, detail, secs) {
  const li = document.querySelector(`.stages li[data-step="${step}"]`);
  if (!li) return;
  li.dataset.state = state;
  if (detail != null) li.querySelector('.stage-detail').textContent = detail;
  li.querySelector('.stage-time').textContent =
    secs != null ? `${fixed(secs, 2)}s` : '';
}

/* On completion the rail shows one row per stage that actually ran, plus a
   `not reached` row for any of the three core stages that did not - refusing
   and re-asking are different answers and the rail should say which. */
function rebuildRail(trace) {
  const byStep = new Map(trace.map((s) => [s.step, s]));
  const rows = RAIL_ORDER.filter((s) => byStep.has(s) || STEPS.includes(s));
  $('stages').replaceChildren(...rows.map((step) => {
    const done = byStep.get(step);
    const li = el('li');
    li.dataset.step = step;
    li.dataset.state = done ? 'done' : 'skipped';
    li.append(el('span', 'stage-name', titleise(step)),
              el('span', 'stage-time num', done ? `${fixed(done.elapsed_s, 2)}s` : ''),
              el('span', 'stage-detail', done ? done.detail : 'not reached'));
    return li;
  }));
}

/* ---------- narration ---------- */

function narrate(...parts) {
  const n = $('narration');
  n.replaceChildren(...parts.filter(Boolean).map(
    (p) => (typeof p === 'string' ? document.createTextNode(p) : p)));
  n.classList.remove('swap');
  void n.offsetWidth;
  n.classList.add('swap');
}
const mono = (s) => el('span', 'mono', s);

function narrateSeq(lines, gap = 2400) {
  lines.forEach((parts, i) => t.after(i * gap, () => narrate(...parts)));
  return lines.length * gap;
}

/* ---------- lifecycle ---------- */

export function begin() {
  stop();
  gate = null;
  evidenceIds = [];
  $('run').hidden = false;
  $('result').hidden = true;
  $('progress').hidden = true;
  $('strip').replaceChildren();
  $('frame-boxes').replaceChildren();
  $('frame-meta').replaceChildren();
  $('coverage-read').replaceChildren();
  $('frame-img').removeAttribute('src');
  $('frame-img').classList.remove('in');
  elevation.reset(runElev);
  STEPS.forEach((s) => setStage(s, 'waiting', 'waiting', null));
  narrate('checking the photographs');

  t0 = performance.now();
  clock = setInterval(tick, 100);
  $('run').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
}

export function stop() {
  t.clear();
  clearInterval(clock);
  reading = false;
  $('scan').classList.remove('on');
}

const elapsed = () => (performance.now() - t0) / 1000;

function tick() {
  $('clock').textContent = `${elapsed().toFixed(1)}s`;
  if (!reading) return;
  /* Asymptotic, capped: an honest "still going" rather than a fake finish. */
  const p = Math.min(0.95, 1 - Math.exp(-elapsed() / EST_TAU));
  $('progress-fill').style.width = `${(p * 100).toFixed(1)}%`;
}

/* ---------- act 1: the gate has landed ---------- */

export function onGate(msg) {
  gate = msg.gate;
  evidenceIds = msg.evidence_photo_ids || [];
  frames.setSource(msg.photo_urls, gate.photos, gate.decision);

  elevation.setCoverage(runElev, gate.views_present);
  writeCoverage();
  frames.buildStrip(gate.photos, (c) => { frames.showFrame(c); frames.markCell(c.photo_id); });

  /* Lead with the frame the verdict turns on: on a not-a-truck refusal that is
     whatever the gate identified instead, which is the whole explanation. */
  const first = (gate.decision === 'refuse_not_a_truck' && frames.smokingGun())
    || (gate.photos || []).find((c) => c.usable)
    || gate.photos[0];
  if (first) { frames.showFrame(first); frames.markCell(first.photo_id); }

  const lines = [[gate.headline]];
  if (gate.truck_evidence) lines.push([gate.truck_evidence]);
  if ((gate.missing_views || []).length) {
    const pretty = gate.missing_views.map((v) => viewName(v).toLowerCase()).join(', ');
    lines.push([`nothing covering ${pretty} — the band will widen for that, `
                + 'which is a stated assumption rather than a measurement']);
  } else {
    lines.push(['every canonical view the gate asks for is present']);
  }
  narrateSeq(lines, 2600);
}

function writeCoverage(issues) {
  if (!gate) return;
  const dl = $('coverage-read');
  dl.replaceChildren();
  const add = (k, v, cls) => {
    dl.append(el('dt', null, k));
    dl.append(el('dd', cls, v));
  };
  const usable = (gate.usable_photo_ids || []).length;
  const total = (gate.photos || []).length;
  add('Frames', `${usable} usable of ${total}`, 'seen');
  const names = (list) => list.map((v) => viewName(v).toLowerCase()).join(', ');
  add('Views', names(gate.views_present || []) || 'none identified', 'seen');
  add('Not photographed',
      names(gate.missing_views || []) || 'nothing the gate asks for',
      (gate.missing_views || []).length ? 'unseen' : 'seen');
  if (issues) {
    const major = issues.filter((i) => i.severity === 'major').length;
    add('Flagged', `${issues.length} finding${issues.length === 1 ? '' : 's'}`
        + (major ? `, ${major} major` : ''), major ? 'unseen' : null);
  }
}

/* ---------- act 2: the long read ---------- */

export function onStage(msg) {
  /* Whatever came before this step is finished; its real time arrives with
     the result, so leave the figure blank rather than invent one. */
  const i = STEPS.indexOf(msg.step);
  for (const s of STEPS.slice(0, Math.max(0, i))) {
    const li = document.querySelector(`.stages li[data-step="${s}"]`);
    if (li && li.dataset.state === 'running') li.dataset.state = 'done';
  }
  setStage(msg.step, 'running', msg.detail, null);

  if (msg.step === 'evidence') startReading(evidenceIds);
  if (msg.step === 'price') {
    stopReading();
    narrate('matching against the priced listings');
  }
}

function startReading(ids) {
  const list = ids.length ? ids : (gate ? gate.usable_photo_ids : []) || [];
  if (!list.length) return;
  reading = true;
  $('progress').hidden = false;
  $('progress-label').textContent = 'vision model reading, typically 30–50s';
  $('scan').classList.add('on');
  list.forEach((id) => {
    const c = $(`cell-${id}`);
    if (c) c.classList.add('reading');
  });

  let i = 0;
  const step = () => {
    if (!reading) return;
    const id = list[i % list.length];
    const check = frames.checkFor(id);
    if (check) {
      frames.showFrame(check);
      frames.markCell(id, 'read');
      narrate('reading ', mono(check.filename), `, ${viewName(check.view).toLowerCase()}`);
    }
    i += 1;
    /* Past one pass the model is simply still working; say so and keep the
       head moving rather than pretending there are more frames. */
    if (i === list.length) {
      t.after(DWELL_MS, () => {
        if (reading) narrate(`all ${list.length} frames sent — waiting on the model`);
      });
    }
    t.after(DWELL_MS, step);
  };
  step();
}

function stopReading() {
  reading = false;
  $('scan').classList.remove('on');
  $('progress-fill').style.width = '100%';
  t.clear();
}

/* ---------- act 3: it answered ---------- */

export function onResult(a) {
  stopReading();
  clearInterval(clock);
  $('clock').textContent = `${fixed(a.elapsed_s, 2)}s`;
  $('progress-label').textContent = 'done';

  rebuildRail(a.trace || []);

  const ev = a.evidence;
  if (ev) {
    elevation.setSummarised(runElev, ev.condition_summary);
    elevation.setFindings(runElev, ev.issues);
    writeCoverage(ev.issues);
    const n = (ev.issues || []).length;
    narrate(n
      ? `${n} finding${n === 1 ? '' : 's'}, each pinned to the frame it came from`
      : 'nothing flagged in the frames supplied');
  } else {
    narrate('stopped before the vision call');
  }
}

/* A frozen export has no server and no clock: paint the finished state
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
  /* onResult only writes the coverage readout when there is evidence; on a
     refusal the live run had already written it from the gate event. */
  writeCoverage();
  const lead = (gate.decision === 'refuse_not_a_truck' && frames.smokingGun())
    || (gate.photos || []).find((c) => c.usable) || gate.photos[0];
  frames.buildStrip(gate.photos, (c) => { frames.showFrame(c); frames.markCell(c.photo_id); });
  if (lead) { frames.showFrame(lead); frames.markCell(lead.photo_id); }
  /* render() calls onResult itself, so the stage times, the findings and the
     revision marks all land through the one path. */
}

export function onError(message) {
  stop();
  STEPS.forEach((s) => {
    const li = document.querySelector(`.stages li[data-step="${s}"]`);
    if (li && li.dataset.state !== 'done') setStage(s, 'skipped', 'not reached', null);
  });
  narrate(message);
}
