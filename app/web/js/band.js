/* Two separate, directly labelled ranges on one shared scale. The baseline
   describes asking prices, not sales; the photo adjustment is not calibrated. */
import { el, money } from './dom.js';

export function drawBand(root, price) {
  root.replaceChildren();
  const valid = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0;
  const values = [price.low, price.high, price.point, price.baseline_low,
    price.baseline_high, price.baseline_point].filter(valid);
  if (values.length < 2) return;
  // Outlying individual listings must not compress the estimate to a sliver.
  const lo = Math.min(...values), hi = Math.max(...values);
  const pad = (hi - lo) * .12 || hi * .05;
  const min = lo - pad, max = hi + pad;
  const position = (v) => Math.max(0, Math.min(100, (v - min) / (max - min) * 100));
  const amount = (v) => money(v, price.currency);
  const asking = price.asking?.asking;
  const row = (label, description, low, high, point, kind) => {
    if (![low, high, point].every(valid)) return;
    const section = el('div', `range-row ${kind}`);
    const head = el('div', 'range-heading');
    head.append(el('strong', null, label), el('span', 'range-values', `${amount(low)} – ${amount(high)}`));
    section.append(head, el('p', 'range-description', description));
    const track = el('div', 'range-track');
    track.setAttribute('aria-hidden', 'true');
    const bar = el('span', 'range-interval');
    bar.style.left = `${position(low)}%`;
    bar.style.width = `${Math.max(.3, position(high) - position(low))}%`;
    const mark = el('i', 'range-point');
    mark.style.left = `${position(point)}%`;
    track.append(bar, mark);
    if (valid(asking)) {
      const seller = el('i', 'range-seller');
      seller.style.left = `${position(asking)}%`;
      if (asking < min || asking > max) seller.classList.add(asking < min ? 'off-left' : 'off-right');
      track.append(seller);
    }
    const foot = el('div', 'range-foot');
    foot.append(el('span', null, 'Lower estimate'), el('span', null, `Model estimate ${amount(point)}`), el('span', null, 'Upper estimate'));
    section.append(track, foot);
    root.append(section);
    return bar;
  };
  const bar = row('Photo-adjusted estimate', 'The expected asking-price range, adjusted for visible condition.', price.low, price.high, price.point, 'adjusted');
  row('Comparable-market baseline', 'Before photo adjustments · similar age and mileage.', price.baseline_low, price.baseline_high, price.baseline_point, 'baseline');
  const legend = el('div', 'range-legend');
  legend.append(el('span', 'midpoint-key', 'Solid line: model estimate'));
  if (valid(asking)) {
    legend.append(el('span', 'seller-key', `Diamond: seller asks ${amount(asking)}${asking < min ? ' · below chart scale' : asking > max ? ' · above chart scale' : ''}`));
  }
  root.append(legend, el('p', 'range-disclaimer', 'Both rows use the same price scale. These are estimated asking prices, not confirmed sale prices. The photo-adjusted range has no measured coverage guarantee.'));
  return bar;
}
