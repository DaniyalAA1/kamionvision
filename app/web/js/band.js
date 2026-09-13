/* The price band.

   Two ranges live here and they are not interchangeable, so the drawing keeps
   them apart rather than averaging them into one bar:

     the pale bar    what comparable trucks are being asked for. This is the
                     band whose accuracy was measured, and the measured figure
                     is attached to it and to nothing else.
     the red bar     that range moved by what the photos found. It carries no
                     coverage guarantee, so it never wears the measured number.

   The seller's own asking price, when they gave one, is a marker on the same
   scale. That comparison is the single most useful thing on the screen for
   someone who already has a number in their head. */

import { svg, svgText, money } from './dom.js';

const W = 1000, H = 168;
const X0 = 34, X1 = 966;
const BAR_Y = 64, BAR_H = 30;

export function drawBand(root, price) {
  root.replaceChildren();
  root.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const points = [price.low, price.high, price.baseline_low, price.baseline_high,
                  price.point, price.baseline_point]
    .filter((v) => typeof v === 'number' && isFinite(v) && v > 0);
  for (const c of price.comparables || []) {
    if (typeof c.price === 'number' && isFinite(c.price)) points.push(c.price);
  }
  if (price.asking && price.asking.asking) points.push(price.asking.asking);
  if (points.length < 2) return;

  const lo = Math.min(...points), hi = Math.max(...points);
  const pad = (hi - lo) * 0.1 || Math.max(1, hi * 0.05);
  const min = lo - pad, max = hi + pad;
  const x = (v) => X0 + ((v - min) / (max - min)) * (X1 - X0);

  /* the axis the bars sit on */
  root.append(svg('line', {
    x1: X0, y1: BAR_Y + BAR_H + 15, x2: X1, y2: BAR_Y + BAR_H + 15,
    stroke: 'var(--edge)', 'stroke-width': 1 }));

  /* what comparable trucks are asking */
  const baseW = Math.max(2, x(price.baseline_high) - x(price.baseline_low));
  root.append(svg('rect', {
    x: x(price.baseline_low), y: BAR_Y - 5, width: baseW, height: BAR_H + 10,
    rx: 6, fill: 'var(--lot-deep)' }));

  /* every comparable that was actually used, as a tick on the same scale */
  for (const c of price.comparables || []) {
    if (typeof c.price !== 'number' || !isFinite(c.price)) continue;
    root.append(svg('line', {
      x1: x(c.price), y1: BAR_Y + BAR_H + 10, x2: x(c.price), y2: BAR_Y + BAR_H + 20,
      stroke: 'var(--steel-dim)', 'stroke-width': 1.5 }));
  }

  /* the estimate, after the photos */
  const adjW = Math.max(3, x(price.high) - x(price.low));
  const bar = svg('rect', {
    x: x(price.low), y: BAR_Y, width: adjW, height: BAR_H,
    rx: 4, fill: 'var(--signal)' });
  root.append(bar);
  root.append(svg('line', {
    x1: x(price.point), y1: BAR_Y - 6, x2: x(price.point), y2: BAR_Y + BAR_H + 6,
    stroke: 'var(--ink)', 'stroke-width': 2 }));

  /* the seller's number */
  const asking = price.asking && price.asking.asking ? price.asking.asking : null;
  if (asking) {
    const ax = x(asking);
    root.append(svg('line', {
      x1: ax, y1: 28, x2: ax, y2: BAR_Y + BAR_H + 8,
      stroke: 'var(--ink)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }));
    root.append(svgText({
      x: Math.min(X1 - 4, Math.max(X0 + 4, ax)), y: 20,
      'text-anchor': ax > (X0 + X1) / 2 ? 'end' : 'start',
      fill: 'var(--ink)', 'font-size': 23, 'font-weight': 600,
      'font-family': 'var(--sans)' },
      `asking ${money(asking, price.currency)}`));
  }

  /* Endpoints only. A full axis of tick labels would be a chart; this is one
     range and the two numbers that bound it. */
  const label = (value, anchor, xPos) => root.append(svgText({
    x: xPos, y: BAR_Y + BAR_H + 38, 'text-anchor': anchor,
    fill: 'var(--steel)', 'font-size': 20, 'font-family': 'var(--sans)' },
    money(value, price.currency)));
  label(price.baseline_low, 'start', x(price.baseline_low));
  label(price.baseline_high, 'end', x(price.baseline_high));

  root.append(svgText({
    x: X0, y: H - 8, fill: 'var(--steel-dim)', 'font-size': 16,
    'font-family': 'var(--sans)' },
    'The pale bar is what similar trucks are being asked for, and each tick '
    + 'is one of the listings behind it.'));

  return bar;
}
