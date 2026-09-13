/* One scroll reader for the whole page.

   Every act of the story is driven by the same number: how far the page has
   scrolled through one tall element. Reading that per-act would mean a scroll
   listener and a layout read for each of them, so there is exactly one
   listener here, one `requestAnimationFrame` per scroll burst, and one
   `getBoundingClientRect` per tracked element per frame.

   Progress is 0 when the track's top reaches the top of the viewport and 1
   when its bottom reaches the bottom, which is precisely the span over which a
   `position: sticky` child is pinned. That correspondence is the whole trick:
   the sticky stage is on screen for exactly 0 to 1. */

const tracked = [];
let frame = 0;

function read() {
  frame = 0;
  const vh = window.innerHeight;
  for (const t of tracked) {
    const rect = t.el.getBoundingClientRect();
    const span = rect.height - vh;
    // A track shorter than the viewport can never pin. Report the endpoints
    // rather than dividing by zero, so a very short window still gets a
    // coherent story instead of NaN transforms.
    const raw = span <= 0 ? (rect.top <= 0 ? 1 : 0) : -rect.top / span;
    const p = raw < 0 ? 0 : raw > 1 ? 1 : raw;
    if (p !== t.last) {
      t.last = p;
      t.on(p);
    }
  }
}

function schedule() {
  if (!frame) frame = requestAnimationFrame(read);
}

/* Register `el` and call `on(progress)` whenever its progress changes.
   Returns an unsubscribe. */
export function track(el, on) {
  const entry = { el, on, last: -1 };
  tracked.push(entry);
  if (tracked.length === 1) {
    addEventListener('scroll', schedule, { passive: true });
    addEventListener('resize', schedule);
  }
  schedule();
  return () => {
    const i = tracked.indexOf(entry);
    if (i >= 0) tracked.splice(i, 1);
  };
}

/* Re-read on the next frame. Callers use this after changing a layout the
   progress depends on - loading a font, an image settling, the track height
   being written - because none of those fire a scroll event. */
export const refresh = schedule;

export const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/* Ease-in-out over [0, 1]. Used for the hand-off between two photographs,
   where a linear cross-fade reads as a dissolve and this reads as a cut. */
export const smoothstep = (t) => t * t * (3 - 2 * t);

/* A beat holds still for most of its slice and then moves. Scroll-driven
   animation that maps progress linearly onto position never lets the reader
   rest on anything: the dwell is what makes it legible. */
export function dwell(frac, hold = 0.62) {
  if (frac <= hold) return 0;
  return smoothstep((frac - hold) / (1 - hold));
}
