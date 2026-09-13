/* The scroll story: wiring.

   `scripts/freeze_story.py` has already written the whole section into
   `landing.html` as a plain, readable document - every photograph, every
   verdict, both price bands. Nothing here fetches anything, and nothing here
   writes a sentence about the truck. This file's entire job is to measure the
   acts it finds, give the track a height, and turn one scroll position into
   which act is on stage and how far through it the reader is.

   That ordering is deliberate. If this module throws, fails to import, or
   never runs, the page above it is unchanged and complete. `is-driven` is
   added only once the measuring has succeeded, and every choreographed rule in
   `story.css` is scoped to it. */

import { track, refresh, clamp01, dwell } from './scroll.js';
import { Deck } from './deck.js';
import { makeReveals } from './reveal.js';

/* How much scroll each act is worth, in viewport heights. The reading act is
   sized from the number of photographs it actually contains, so adding a
   seventh hero photo in the freeze script does not also mean editing a
   number here. */
const WEIGHT = { gate: 1.25, merge: 1.0, check: 1.3, price: 1.5, limits: 0.9 };
const PER_BEAT = 0.92;
const UNIT_VH = 92;
const FALLBACK_WEIGHT = 1.0;

const LABEL = {
  gate: 'Identify', read: 'Read the photos', merge: 'Merge',
  check: 'Cross-check', price: 'Price', limits: 'Limits',
};

function setup(section) {
  const trackEl = section.querySelector('[data-track]');
  const stageBody = section.querySelector('[data-stage-body]');
  const actLabel = section.querySelector('[data-act-label]');
  const actCount = section.querySelector('[data-act-count]');
  const acts = [...section.querySelectorAll('[data-act]')];
  if (!trackEl || !stageBody || !acts.length) return null;

  const readAct = acts.find((a) => a.dataset.act === 'read');
  const deck = readAct ? new Deck(readAct) : null;

  const weights = acts.map((a) => (
    a === readAct ? Math.max(1, deck?.length || 1) * PER_BEAT
      : WEIGHT[a.dataset.act] ?? FALLBACK_WEIGHT));
  const total = weights.reduce((a, b) => a + b, 0);

  // Cumulative starts, so a progress value maps to an act with one scan.
  const starts = [];
  let acc = 0;
  for (const w of weights) {
    starts.push(acc / total);
    acc += w;
  }

  const reveal = makeReveals();
  let current = -1;
  let tone = '';

  function setTone(next) {
    if (next === tone) return;
    tone = next;
    section.dataset.tone = next;
  }

  function apply(p) {
    // Which act, and how far through it.
    let i = starts.length - 1;
    while (i > 0 && p < starts[i]) i -= 1;
    const span = (i + 1 < starts.length ? starts[i + 1] : 1) - starts[i];
    const local = clamp01(span > 0 ? (p - starts[i]) / span : 1);
    const act = acts[i];

    if (i !== current) {
      acts.forEach((a, n) => {
        a.classList.toggle('is-live', n === i);
        a.classList.toggle('is-done', n < i);
        // Only the act on stage is in the accessibility tree. Six overlapping
        // layers of prose read aloud in sequence is not a page anyone can use.
        a.setAttribute('aria-hidden', n === i ? 'false' : 'true');
      });
      current = i;
      if (actLabel) actLabel.textContent = LABEL[act.dataset.act] || act.dataset.act;
      reveal(act);
    }

    if (deck && act === readAct) {
      const n = deck.length;
      const scaled = local * n;
      const index = Math.min(n - 1, Math.floor(scaled));
      deck.update(index + dwell(scaled - index));
      if (actCount) actCount.textContent = `${index + 1} / ${n}`;
      setTone(`photo-${index}`);
    } else {
      if (actCount) actCount.textContent = '';
      setTone(act.dataset.act === 'gate' ? 'identify' : act.dataset.act);
    }
  }

  function height() {
    trackEl.style.setProperty('--track-height', `${(total * UNIT_VH).toFixed(0)}vh`);
  }

  return { trackEl, stageBody, deck, apply, height, acts };
}

function start() {
  const section = document.querySelector('[data-story]');
  if (!section) return;

  const wired = setup(section);
  if (!wired) return;

  // Below this width the stage does not pin at all - `story.css` unpins it -
  // so driving it would only fight the stylesheet.
  const narrow = window.matchMedia('(max-width: 819px)');
  const still = window.matchMedia('(prefers-reduced-motion: reduce)');
  let stop = null;

  function engage() {
    const off = narrow.matches || still.matches;
    if (off) {
      if (stop) { stop(); stop = null; }
      section.classList.remove('is-driven');
      delete section.dataset.tone;
      wired.deck?.reset();
      wired.acts.forEach((a) => {
        a.classList.remove('is-live', 'is-done');
        a.removeAttribute('aria-hidden');
      });
      return;
    }
    if (stop) return;
    section.classList.add('is-driven');
    wired.height();
    wired.deck?.measure();
    stop = track(wired.trackEl, wired.apply);
    refresh();
  }

  engage();
  narrow.addEventListener('change', engage);
  still.addEventListener('change', engage);

  addEventListener('resize', () => {
    if (!stop) return;
    wired.height();
    wired.deck?.measure();
    refresh();
  });

  // The deck's filed row is measured off a rendered photograph. Until the
  // images have laid out, that measurement is of an empty box.
  section.querySelectorAll('.beat-figure img').forEach((img) => {
    if (img.complete) return;
    img.addEventListener('load', () => {
      wired.deck?.measure();
      refresh();
    }, { once: true });
  });
  if (document.fonts?.ready) {
    document.fonts.ready.then(() => { wired.deck?.measure(); refresh(); });
  }
}

if (document.readyState === 'loading') {
  addEventListener('DOMContentLoaded', start, { once: true });
} else {
  start();
}
