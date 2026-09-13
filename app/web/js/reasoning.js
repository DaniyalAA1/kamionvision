/* The reasoning rail: one card per photo, as each photo's own vision call
   returns.

   Every line on this rail is something the model actually said about the frame
   beside it. The old screen filled the same minute with a scripted narration
   and a progress bar easing towards 95% - honest about being an estimate, but
   still a picture of work rather than the work. Sixteen concurrent calls give
   the rail real events to draw, so it draws those instead.

   Cards land out of order, because concurrent calls finish out of order. The
   newest goes on top and resolves out of defocus; the one it displaced softens
   back. Nothing here re-sorts, because re-sorting would hide the one honest
   thing the rail has to say, which is that this is arriving as it happens. */

import { $, el, viewName, reduced } from './dom.js';
import { urlFor } from './frames.js';

const SEV_RANK = { major: 3, moderate: 2, minor: 1, cosmetic: 0 };

let expected = 0;
let landed = 0;

export function begin(total) {
  expected = total;
  landed = 0;
  const rail = $('thoughts');
  rail.replaceChildren();
  for (let i = 0; i < Math.min(total, 3); i += 1) {
    rail.append(el('div', 'thought-pending', 'reading a photo…'));
  }
  $('rail-title').textContent = 'Looking at the photos';
  $('rail-sub').textContent = total
    ? `${total} frames, one at a time`
    : 'waiting on the gate';
}

export function clear() {
  expected = 0;
  landed = 0;
  $('thoughts').replaceChildren();
}

/* One finished call. `finding` is the PhotoFinding straight off the wire. */
export function add(finding) {
  const rail = $('thoughts');
  const pending = rail.querySelector('.thought-pending');
  if (pending) pending.remove();

  rail.querySelectorAll('.thought').forEach((n) => n.classList.add('settled'));

  const card = el('div', 'thought');
  card.dataset.worst = worstOf(finding);
  if (finding.error) card.dataset.error = 'true';

  const thumb = el('img', 'thought-thumb');
  thumb.src = urlFor(finding.photo_id);
  thumb.alt = '';
  thumb.loading = 'lazy';
  card.append(thumb);

  const body = el('div', 'thought-body');
  const head = el('p', 'thought-view', viewName(finding.view));
  if (finding.cropped) {
    head.append(el('span', 'thought-cropped', 'cropped to the truck'));
  }
  body.append(head);

  if (finding.error) {
    body.append(el('p', 'thought-quiet', 'this frame could not be read'));
  } else {
    if (finding.shows) body.append(el('p', 'thought-shows', finding.shows));

    const list = el('ul', 'thought-list');
    const issues = [...(finding.issues || [])].sort(
      (a, b) => (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0));
    issues.slice(0, 4).forEach((issue) => {
      const li = el('li', null, issue.observation);
      li.dataset.sev = issue.severity;
      list.append(li);
    });
    (finding.strengths || []).slice(0, issues.length ? 1 : 3).forEach((good) => {
      const li = el('li', null, good);
      li.dataset.sev = 'ok';
      list.append(li);
    });
    if (list.children.length) body.append(list);

    const more = issues.length - 4;
    if (more > 0) body.append(el('p', 'thought-quiet', `and ${more} more on this frame`));
    else if (!list.children.length) {
      body.append(el('p', 'thought-quiet', 'nothing to flag on this frame'));
    }
  }

  card.append(body);
  rail.prepend(card);
  if (!reduced()) card.classList.add('resolve');

  landed += 1;
  $('rail-sub').textContent = expected
    ? `${landed} of ${expected} frames read`
    : `${landed} frames read`;
  return card;
}

function worstOf(finding) {
  if (finding.error) return 'error';
  const issues = finding.issues || [];
  if (!issues.length) return 'clean';
  const worst = issues.reduce(
    (a, b) => ((SEV_RANK[a.severity] ?? 0) >= (SEV_RANK[b.severity] ?? 0) ? a : b));
  return worst.severity;
}

/* The rail's closing state. It keeps every card - a reader who wants to know
   where a finding came from can scroll back through the frames it came from. */
export function done(evidence) {
  $('thoughts').querySelectorAll('.thought-pending').forEach((n) => n.remove());
  $('thoughts').querySelectorAll('.thought').forEach((n) => n.classList.add('settled'));
  if (!evidence) {
    $('rail-title').textContent = 'Stopped before the photos were read';
    $('rail-sub').textContent = '';
    return;
  }
  const n = (evidence.issues || []).length;
  $('rail-title').textContent = n
    ? `${n} thing${n === 1 ? '' : 's'} to know about it`
    : 'Nothing to flag in these photos';
  const bits = [`${evidence.photos_read} frames read`];
  if (evidence.photos_failed) bits.push(`${evidence.photos_failed} unreadable`);
  $('rail-sub').textContent = bits.join(', ');
}
