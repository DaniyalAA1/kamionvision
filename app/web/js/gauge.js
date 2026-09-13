/* The price band drawn the way a drawing dimensions a length.

   Two dimensions, stacked, on one scale. The upper one is the basic
   dimension: what comparable trucks are asking, in hairline, with its
   description set into the gap in the line. The lower one is that same
   dimension re-drawn after the photographs moved it, in amber, with the point
   estimate in its gap and the cap stated as a tolerance.

   Keeping them as two separate dimensions is not a stylistic choice. The
   measured 80.3% coverage belongs to the upper one only; a single merged bar
   would put a measured number under an unmeasured band.

   The comparables sit on the same scale as ticks, so "what is it comparing
   against" is answered by the picture.

   The viewBox is set to the element's own pixel width at draw time rather
   than a fixed 1000 with preserveAspectRatio="none", which used to stretch
   every glyph horizontally by whatever the window happened to be. */

import { $, svg, svgText, money, kkm, fixed, tween, animate, reduced } from './dom.js';

const H = 250;
const Y = {
  askText: 16, askTop: 24, askBottom: 48,
  basic: 62, basicFig: 57, basicDesc: 84,
  adj: 124, adjFig: 115, adjNote: 153,
  axis: 176, tick: 188, label: 201, caption: 240,
};

let last = null;

export function drawGauge(price) {
  last = price;
  const g = $('gauge');
  const W = Math.max(520, Math.round(g.clientWidth || 1000));
  g.setAttribute('viewBox', `0 0 ${W} ${H}`);
  g.replaceChildren();

  const comps = (price.comparables || [])
    .map((c) => ({ c, v: (price.currency === 'TRY' ? c.price_try : c.price_usd) ?? c.price }))
    .filter((d) => typeof d.v === 'number' && isFinite(d.v))
    .sort((a, b) => a.v - b.v);

  const asking = price.asking ? price.asking.asking : null;
  const hasBasic = price.baseline_low && price.baseline_high
    && Math.abs(price.baseline_point - price.point) > 1;

  const values = comps.map((d) => d.v);
  if (asking) values.push(asking);
  const lo = Math.min(price.low, hasBasic ? price.baseline_low : price.low, ...values);
  const hi = Math.max(price.high, hasBasic ? price.baseline_high : price.high, ...values);
  const pad = (hi - lo) * 0.12 || hi * 0.12;
  const min = lo - pad, max = hi + pad;
  const L = 150, R = W - 150;               // room for the end figures
  const x = (v) => L + ((v - min) / (max - min)) * (R - L);

  /* ---- the scale ---- */
  g.append(svg('line', { class: 'axis', x1: 40, y1: Y.axis, x2: W - 40, y2: Y.axis }));

  /* ---- basic dimension: the comparable-asking band ---- */
  if (hasBasic) {
    dimension(g, {
      cls: 'dim', term: 'term', y: Y.basic,
      x1: x(price.baseline_low), x2: x(price.baseline_high),
      gapText: 'what comparable trucks are asking', gapCls: 'label',
      endLow: money(price.baseline_low, price.currency),
      endHigh: money(price.baseline_high, price.currency),
      endCls: 'figure-basic', endY: Y.basicFig, dashed: true,
    });
    extension(g, x(price.baseline_low), Y.basic, Y.axis);
    extension(g, x(price.baseline_high), Y.basic, Y.axis);
  }

  /* ---- adjusted dimension, animated off the basic one ---- */
  const adj = svg('g', {});
  g.append(adj);
  const from = hasBasic
    ? { l: price.baseline_low, h: price.baseline_high, p: price.baseline_point }
    : { l: price.low, h: price.high, p: price.point };
  const to = { l: price.low, h: price.high, p: price.point };

  const place = (l, h, p) => {
    adj.replaceChildren();
    dimension(adj, {
      cls: 'dim-adj', term: 'term-adj', y: Y.adj, x1: x(l), x2: x(h),
      gapText: money(p, price.currency), gapCls: 'figure',
      endLow: money(l, price.currency), endHigh: money(h, price.currency),
      endCls: 'figure-basic', endY: Y.adjFig,
    });
    adj.append(svg('rect', {
      class: 'band-adj', x: x(l), y: Y.adj - 13,
      width: Math.max(1, x(h) - x(l)), height: 26,
    }));
    extension(adj, x(l), Y.adj, Y.axis);
    extension(adj, x(h), Y.adj, Y.axis);
    const px = x(p);
    adj.append(svg('path', {
      class: 'point-tri',
      d: `M ${px} ${Y.axis - 10} L ${px - 6} ${Y.axis - 21} L ${px + 6} ${Y.axis - 21} Z`,
    }));
    adj.append(svg('line', { class: 'point', x1: px, y1: Y.axis - 10, x2: px, y2: Y.axis + 8 }));
  };

  if (hasBasic && !reduced()) {
    place(from.l, from.h, from.p);
    setTimeout(() => tween(0, 1, 900, (t) => place(
      from.l + (to.l - from.l) * t,
      from.h + (to.h - from.h) * t,
      from.p + (to.p - from.p) * t)), 420);
  } else {
    place(to.l, to.h, to.p);
  }

  /* the cap, stated as a tolerance because that is what it is */
  if (price.adjustment && price.adjustment.pct) {
    const mid = (x(price.low) + x(price.high)) / 2;
    g.append(svgText({ class: 'tol', x: mid, y: Y.adjNote, 'text-anchor': 'middle' },
      `condition ${price.adjustment.pct >= 0 ? '+' : ''}`
      + `${fixed(price.adjustment.pct, 1)}%  ±${fixed(price.adjustment.cap_pct, 1)}% cap`));
  }

  /* ---- the seller's own number ---- */
  if (asking) {
    const ax = x(asking);
    const cls = price.asking.inside_comparable_band ? 'ask-in' : 'ask-out';
    g.append(svg('line', { class: `ask ${cls}`, x1: ax, y1: Y.askTop, x2: ax, y2: Y.askBottom }));
    g.append(svg('circle', { class: cls, cx: ax, cy: Y.askBottom, r: 3.5,
                             fill: 'currentColor', stroke: 'none' }));
    g.append(svgText({ class: `ask-label ${cls}`, x: ax, y: Y.askText,
                       'text-anchor': 'middle', fill: 'currentColor' },
                     `seller asks ${money(asking, price.currency)}`));
  }

  /* ---- comparables on the same scale ----
     Five F-MAXes at similar mileage land almost on top of each other, so
     labels go greedily into the first row that clears the previous one; a
     label with nowhere to sit is dropped and its tick stays, which beats an
     unreadable pile. */
  const rowEnds = [-Infinity, -Infinity, -Infinity];
  comps.forEach(({ c, v }) => {
    const px = x(v);
    g.append(svg('line', { class: 'comp', x1: px, y1: Y.axis, x2: px, y2: Y.tick }));
    const label = `${c.year ?? '—'} ${kkm(c.km)}`;
    const half = label.length * 3.9 + 6;
    const row = rowEnds.findIndex((end) => px - half > end);
    if (row < 0) return;
    rowEnds[row] = px + half;
    g.append(svgText({ class: 'comp-label', x: px, y: Y.label + row * 15,
                       'text-anchor': 'middle' }, label));
  });

  g.append(svgText({ class: 'label', x: 40, y: Y.caption },
                   'teal ticks: the real listings this was priced against'));

  if (!reduced()) {
    animate(g, { opacity: [0, 1] }, { duration: 0.45, ease: [0.2, 0.7, 0.3, 1] });
  }
}

/* A dimension line: two segments with a gap for the figure, an arrow
   terminator at each end pointing out at its extension line, and the endpoint
   values set outside the arrows. */
function dimension(parent, o) {
  const mid = (o.x1 + o.x2) / 2;

  /* Set the figure first and measure it, so the line really does break for it
     rather than running through the glyphs - which is what estimating the
     advance width from the character count got wrong. */
  const gapNode = svgText(
    { class: o.gapCls, x: mid, y: o.y + (o.gapCls === 'figure' ? 8 : 5),
      'text-anchor': 'middle' }, o.gapText);
  parent.append(gapNode);
  let gapHalf = 40;
  try {
    const w = gapNode.getComputedTextLength();
    if (w) gapHalf = w / 2 + 10;
  } catch { /* not laid out yet; the default is wide enough to look deliberate */ }

  const seg = (a, b) => {
    if (b - a < 6) return;
    const l = svg('line', { class: o.cls, x1: a, y1: o.y, x2: b, y2: o.y });
    if (o.dashed) l.setAttribute('stroke-dasharray', '5 4');
    parent.append(l);
  };
  seg(o.x1 + 7, mid - gapHalf);
  seg(mid + gapHalf, o.x2 - 7);

  parent.append(svg('path', { class: o.term,
    d: `M ${o.x1} ${o.y} L ${o.x1 + 9} ${o.y - 4} L ${o.x1 + 9} ${o.y + 4} Z` }));
  parent.append(svg('path', { class: o.term,
    d: `M ${o.x2} ${o.y} L ${o.x2 - 9} ${o.y - 4} L ${o.x2 - 9} ${o.y + 4} Z` }));

  parent.append(svgText({ class: o.endCls, x: o.x1 - 12, y: o.endY + 5,
                          'text-anchor': 'end' }, o.endLow));
  parent.append(svgText({ class: o.endCls, x: o.x2 + 12, y: o.endY + 5,
                          'text-anchor': 'start' }, o.endHigh));
}

function extension(parent, x, fromY, toY) {
  parent.append(svg('line', { class: 'ext', x1: x, y1: fromY + 14, x2: x, y2: toY - 2,
                              'stroke-dasharray': '2 4' }));
}

/* Redraw on resize: the layout is in real pixels now, so it has to be. */
let pending = 0;
new ResizeObserver(() => {
  if (!last || $('gauge-wrap').hidden) return;
  clearTimeout(pending);
  pending = setTimeout(() => drawGauge(last), 120);
}).observe(document.body);
