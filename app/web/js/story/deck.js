/* The deck: six photographs, lifted off a stack one at a time and filed.

   The whole act is one number. `live` is a float across the beats - 2.0 means
   photo 2 is being read, 2.4 means it is on its way out and photo 3 on its way
   in - and every card's position is a pure function of its distance from it.
   There is no per-card state, no timeline and nothing to keep in sync, which
   is what makes it safe to drive from a scroll position that can jump
   backwards, skip, or arrive mid-way from a restored scroll offset.

   Three zones, by distance `d = index - live`:

     d < 0     read already; travelling to its slot in the filed row
     d ~ 0     live; centred, full size, its subject box drawing
     d > 0     still in the deck; stacked behind, smaller, dimmer

   Cards more than DEPTH behind the live one are not drawn. A stack of six
   visible edges reads as clutter, and three reads as a deck. */

import { drawSubject } from './boxes.js';

const DEPTH = 3;          // how many queued cards stay visible behind the live one
const STACK_X = 40;       // px right per card of depth
const STACK_Y = -22;      // px up per card of depth
const STACK_SCALE = 0.07; // size lost per card of depth
const STACK_TILT = -1.6;  // degrees per card of depth
const FILED_SCALE = 0.17;
const FILED_GAP = 10;     // px between filed thumbnails

const ease = (t) => 1 - Math.pow(1 - t, 3);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

export class Deck {
  constructor(root) {
    this.root = root;
    this.beats = [...root.querySelectorAll('.beat')];
    this.figures = this.beats.map((b) => b.querySelector('.beat-figure'));
    this.verdicts = this.beats.map((b) => b.querySelector('.beat-verdict'));
    this.boxes = this.beats.map((b) => {
      const raw = b.dataset.box;
      if (!raw) return null;
      try { return JSON.parse(raw); } catch { return null; }
    });
    this.live = -1;
    this.measure();
  }

  get length() { return this.beats.length; }

  /* The filed row's geometry depends on the rendered size of a card, so it is
     measured rather than assumed. Called again on resize. */
  measure() {
    const fig = this.figures[0];
    const stage = this.root.parentElement || this.root;
    this.cardW = fig?.offsetWidth || 0;
    this.cardH = fig?.offsetHeight || 0;
    this.stageH = stage.clientHeight || 0;
    const thumbW = this.cardW * FILED_SCALE;
    this.filedStep = thumbW + FILED_GAP;
    // The filed row sits along the bottom of the stage. The figure's own
    // origin is the stage's vertical centre, so the offset is measured from
    // there, and the scale happens about the card's centre.
    this.filedY = this.stageH / 2 - (this.cardH * FILED_SCALE) / 2 - 8;
  }

  /* Where card `i` sits once it has been read. */
  filedSlot(i) {
    const inset = (this.cardW - this.cardW * FILED_SCALE) / 2;
    return { x: -inset + i * this.filedStep, y: this.filedY };
  }

  place(i, live) {
    const figure = this.figures[i];
    const verdict = this.verdicts[i];
    if (!figure) return;
    const d = i - live;

    let x = 0, y = 0, scale = 1, tilt = 0, opacity = 1, z = 100;

    if (d >= 0) {
      // Still in the deck, or live. `k` saturates so a card ten deep does not
      // fly off the stage; it simply is not drawn.
      const k = Math.min(d, DEPTH);
      x = k * STACK_X;
      y = k * STACK_Y;
      scale = 1 - k * STACK_SCALE;
      tilt = k * STACK_TILT;
      opacity = d > DEPTH + 0.25 ? 0 : clamp01(1 - k * 0.2);
      z = 100 - Math.round(d * 10);
    } else {
      // Filing away. `t` runs 0 to 1 over one beat of travel, after which the
      // card rests in its slot for the remainder of the act.
      const t = ease(clamp01(-d));
      const slot = this.filedSlot(i);
      x = slot.x * t;
      y = slot.y * t;
      scale = 1 + (FILED_SCALE - 1) * t;
      tilt = 0;
      opacity = 1;
      z = 10 + i;
    }

    figure.style.transform =
      `translate(${x.toFixed(2)}px, calc(-50% + ${y.toFixed(2)}px)) ` +
      `scale(${scale.toFixed(4)}) rotate(${tilt.toFixed(2)}deg)`;
    figure.style.opacity = opacity.toFixed(3);
    figure.style.zIndex = String(z);

    if (verdict) {
      // A hard-ish cross-fade. Two verdicts legible in the same column at the
      // same time is worse than a moment with neither.
      const vis = clamp01(1 - Math.abs(d) * 2);
      verdict.style.opacity = vis.toFixed(3);
      verdict.style.transform = `translateY(calc(-50% + ${(d * 26).toFixed(1)}px))`;
      verdict.style.zIndex = String(vis > 0 ? 200 : 1);
    }

    const isLive = Math.abs(d) < 0.5;
    this.beats[i].classList.toggle('is-live', isLive);
    this.beats[i].classList.toggle('is-filed', d < -0.5);
    // Only the live verdict is in the accessibility tree. Six simultaneous
    // copies of a condition report read aloud is worse than none.
    this.beats[i].setAttribute('aria-hidden', isLive ? 'false' : 'true');

    // The box draws itself as the card settles, and is gone before it files.
    if (this.boxes[i]) {
      const drawn = d >= 0 ? clamp01(1 - d * 2.2) : clamp01(1 + d * 3);
      drawSubject(figure, drawn > 0.01 ? this.boxes[i] : null, drawn);
    }
  }

  /* `live` is a float across the beats. */
  update(live) {
    if (live === this.live) return;
    this.live = live;
    for (let i = 0; i < this.beats.length; i += 1) this.place(i, live);
  }

  /* Hand every card back to the stylesheet, for the unpinned fallback. */
  reset() {
    this.live = -1;
    for (let i = 0; i < this.beats.length; i += 1) {
      for (const node of [this.figures[i], this.verdicts[i]]) {
        if (!node) continue;
        node.style.transform = '';
        node.style.opacity = '';
        node.style.zIndex = '';
      }
      this.beats[i].classList.remove('is-live', 'is-filed');
      this.beats[i].removeAttribute('aria-hidden');
      this.figures[i]?.querySelector('.beat-box')?.remove();
    }
  }
}
