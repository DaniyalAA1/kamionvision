/* Shared helpers. */

export const $ = (id) => document.getElementById(id);

export const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

export const NS = 'http://www.w3.org/2000/svg';
export const svg = (tag, attrs = {}) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};
export const svgText = (attrs, text) => {
  const n = svg('text', attrs);
  n.textContent = text;
  return n;
};

const SYMBOL = { TRY: '₺', USD: '$', EUR: '€' };
export const money = (v, cur) =>
  (SYMBOL[cur] || cur + ' ') + Math.round(v).toLocaleString('en-US');
export const titleise = (s) =>
  String(s || '').replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

/* The gate's view ids read as machine names otherwise: `exterior_front_34` is
   a three-quarter view, not a thirty-fourth one. */
export const viewName = (s) =>
  titleise(String(s || '').replace(/_34$/, '_three_quarter'))
    .replace(/three quarter/, '¾');

/* Comparable.km and several confidences are nullable on the wire, and the old
   screen did bare arithmetic on them. These return an em dash instead of NaN. */
export const kkm = (v) =>
  (typeof v === 'number' && isFinite(v)) ? `${Math.round(v / 1000)}k` : '—';
export const fixed = (v, n = 2) =>
  (typeof v === 'number' && isFinite(v)) ? v.toFixed(n) : '—';

export const reduced = () =>
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* Motion (vendored, window.Motion) handles the element animations: it takes a
   NodeList in one call, and it animates SVG presentation attributes like
   stroke-dashoffset without caring whether they are attributes or style. The
   numeric tweens below - the band redraw, the clock - are a short rAF loop,
   because they animate values rather than elements.

   Both respect prefers-reduced-motion by skipping to the end: the caller
   still gets the final state, just immediately. */
const M = window.Motion || null;

export const animate = (target, keyframes, options) => {
  if (reduced() || !M) return null;
  try { return M.animate(target, keyframes, options); } catch { return null; }
};
export function tween(from, to, ms, onUpdate, onDone) {
  if (reduced()) { onUpdate(to); onDone && onDone(); return () => {}; }
  const t0 = performance.now();
  let raf = 0, live = true;
  const ease = (p) => 1 - Math.pow(1 - p, 3);
  const step = (now) => {
    if (!live) return;
    const p = Math.min(1, (now - t0) / ms);
    onUpdate(from + (to - from) * ease(p));
    if (p < 1) raf = requestAnimationFrame(step);
    else onDone && onDone();
  };
  raf = requestAnimationFrame(step);
  return () => { live = false; cancelAnimationFrame(raf); };
}

/* A cancellable timer bag, so a run that finishes early can drop every
   pending step of the sequence it was in the middle of. */
export function timers() {
  let ids = [];
  return {
    after(ms, fn) { ids.push(setTimeout(fn, ms)); },
    every(ms, fn) { ids.push(setInterval(fn, ms)); },
    clear() { ids.forEach((i) => { clearTimeout(i); clearInterval(i); }); ids = []; },
  };
}
