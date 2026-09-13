/* The reveals that are numbers rather than positions.

   Act 1's detection boxes, act 3's fold from raw observations to findings, and
   act 5's two price bands. Each fires once, when its act first becomes live,
   and is idempotent afterwards - scrolling back up and down again must not
   restart a count that has already landed on the right figure.

   Every number these animate is read out of the markup that
   `scripts/freeze_story.py` already wrote, never out of a literal here. The
   final state of a count-up is the text that was in the DOM before it started,
   so a reader with the animation disabled, or one who scrolls past mid-tween,
   still sees the value the run produced. */

import { tween, reduced } from '../dom.js';
import { drawBoxes } from './boxes.js';

/* Count one `[data-count]` element up to the value already in it.

   The element is a promise from `freeze_story.py` that its text holds exactly
   one number; any currency symbol or unit around it is preserved verbatim.
   The tween always finishes by restoring the original string, so the figure a
   reader ends up looking at is the one the run produced rather than one this
   file re-formatted. */
function countUp(node, ms) {
  const text = node.dataset.value || node.textContent;
  node.dataset.value = text;
  const match = text.match(/-?[\d,]*\d(?:\.\d+)?/);
  if (!match) return;
  const target = Number(match[0].replace(/,/g, ''));
  if (!Number.isFinite(target)) return;
  const decimals = (match[0].split('.')[1] || '').length;
  const grouped = match[0].includes(',');
  const render = (v) => {
    const out = grouped
      ? v.toLocaleString('en-US', {
        minimumFractionDigits: decimals, maximumFractionDigits: decimals,
      })
      : v.toFixed(decimals);
    node.textContent = text.replace(match[0], out);
  };
  tween(target * 0.72, target, ms, render, () => { node.textContent = text; });
}

/* Every marked figure in an act, counted up together. */
function countAll(act, ms) {
  act.querySelectorAll('[data-count]').forEach((n) => countUp(n, ms));
}

export function makeReveals() {
  const done = new Set();

  const once = (key, fn) => {
    if (done.has(key)) return;
    done.add(key);
    fn();
  };

  // How long each act's figures take to land. The price is the slowest
  // deliberately: it is the answer, and an answer that snaps into place reads
  // as a number that was already decided.
  const MS = { gate: 700, merge: 1100, check: 900, price: 1400 };

  return function reveal(act) {
    const key = act.dataset.act;
    if (reduced()) return;         // the markup is already the finished state
    once(key, () => {
      if (key === 'gate') {
        const figure = act.querySelector('[data-lot]');
        let dets = [];
        try { dets = JSON.parse(figure?.dataset.detections || '[]'); } catch { /* keep [] */ }
        if (figure) drawBoxes(figure, dets);
      }
      if (MS[key]) countAll(act, MS[key]);
    });
  };
}
