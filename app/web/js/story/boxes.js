/* Detection boxes, drawn over a photograph.

   The coordinates arrive normalised to 0-1 of the SOURCE frame, because the
   copy the browser downloaded is downscaled and pixel coordinates would land
   the boxes somewhere else entirely. `scripts/freeze_story.py` does that
   division; nothing here rescales anything.

   Two rules from the appraisal screen carry over unchanged. Only the box the
   gate chose gets the accent colour - the subject is decided server-side and
   the browser must not re-derive it. And no box ever gets the refusal colour
   here, because this set was not refused: colouring a box red because one
   frame looks odd would contradict a gate that decides truck-ness across the
   whole set. */

import { svg } from '../dom.js';

const VIEWBOX = { viewBox: '0 0 1 1', preserveAspectRatio: 'none' };

/* Draw `detections` over the <img> inside `figure`. Returns the <svg>, or null
   when there is nothing to draw. */
export function drawBoxes(figure, detections, className = 'lot-boxes') {
  const frame = figure.querySelector('.frame') || figure;
  frame.querySelector(`.${className}`)?.remove();
  if (!detections?.length) return null;

  const layer = svg('svg', { ...VIEWBOX, class: className, 'aria-hidden': 'true' });
  // Subject last, so it paints over the boxes it beat rather than under them.
  const order = [...detections].sort((a, b) => (a.is_subject ? 1 : 0) - (b.is_subject ? 1 : 0));
  for (const d of order) {
    const [x1, y1, x2, y2] = d.box;
    layer.append(svg('rect', {
      x: x1, y: y1, width: Math.max(0, x2 - x1), height: Math.max(0, y2 - y1),
      class: d.is_subject ? 'subject' : 'other',
    }));
  }
  frame.append(layer);
  return layer;
}

/* The one box the gate picked, on one photograph.

   `stroke-dasharray` is set to the box's own perimeter so the outline draws
   itself from nothing as `progress` runs 0 to 1. The perimeter is in the 0-1
   user space of the viewBox, which is what the dash units are measured in. */
export function drawSubject(figure, box, progress = 1) {
  const frame = figure.querySelector('.frame') || figure;
  let layer = frame.querySelector('.beat-box');
  if (!box) {
    layer?.remove();
    return null;
  }
  const [x1, y1, x2, y2] = box;
  const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
  if (!layer) {
    layer = svg('svg', { ...VIEWBOX, class: 'beat-box', 'aria-hidden': 'true' });
    layer.append(svg('rect', { x: x1, y: y1, width: w, height: h }));
    frame.append(layer);
  }
  const rect = layer.firstChild;
  const perimeter = 2 * (w + h);
  rect.setAttribute('stroke-dasharray', perimeter);
  rect.setAttribute('stroke-dashoffset', perimeter * (1 - progress));
  return layer;
}
