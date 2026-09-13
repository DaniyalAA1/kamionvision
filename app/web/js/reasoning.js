/* The reasoning rail: one card per photo, as each photo's own vision call
   returns.

   Every line on this rail is something the model actually said about the frame
   beside it.    Cards land out of order, because concurrent calls finish out of
   order. The newest goes on top. Nothing here re-sorts, because re-sorting
   would hide the one honest thing the rail has to say, which is that this is
   arriving as it happens.

   The body of each card is behind a summary: the photograph, the view, and a
   row of part glyphs. The observations are there when you open it. A rail of
   sixteen open paragraphs is unreadable in the minute the calls take. */

import { $, el, viewName, reduced, titleise } from './dom.js';
import { urlFor, citePhoto } from './frames.js';
import { partIcon } from './icons.js';

const SEV_RANK = { major: 3, moderate: 2, minor: 1, cosmetic: 0 };

let expected = 0;
let landed = 0;

export function begin(total) {
  expected = total;
  landed = 0;
  const rail = $('thoughts');
  rail.replaceChildren();
  for (let i = 0; i < Math.min(total, 3); i += 1) {
    rail.append(pendingCard());
  }
  $('rail-title').textContent = 'Looking at the photos';
  $('rail-sub').textContent = total
    ? `${total} frames, arriving as they finish`
    : 'waiting on the gate';
}

export function clear() {
  expected = 0;
  landed = 0;
  $('thoughts').replaceChildren();
}

function pendingCard() {
  const n = el('div', 'thought-pending');
  n.append(partIcon('camera'), el('span', null, 'reading a photo…'));
  return n;
}

export function note(message) {
  const rail = $('thoughts');
  rail.querySelectorAll('.thought-pending').forEach((n) => n.remove());
  const card = el('div', 'thought thought-note');
  card.append(el('p', 'thought-shows', message));
  rail.append(card);
}

/* One finished call. `finding` is the PhotoFinding straight off the wire. */
export function add(finding) {
  const rail = $('thoughts');
  const atTop = rail.scrollTop < 12;
  const previousHeight = rail.scrollHeight;
  const previousCards = atTop && !reduced()
    ? [...rail.querySelectorAll('.thought')].map((node) => [node, node.getBoundingClientRect().top]) : [];
  const pending = rail.querySelector('.thought-pending');
  if (pending) pending.remove();

  rail.querySelectorAll('.thought').forEach((n) => n.classList.add('settled'));

  const issues = [...(finding.issues || [])].sort(
    (a, b) => (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0));

  const card = el('details', 'thought');
  card.dataset.worst = worstOf(finding);
  if (finding.error) card.dataset.error = 'true';

  const summary = el('summary', 'thought-summary');
  const thumb = el('img', 'thought-thumb');
  thumb.src = urlFor(finding.photo_id);
  thumb.alt = '';
  thumb.loading = 'lazy';

  const heading = el('span', 'thought-heading');
  const view = el('strong', 'thought-view', viewName(finding.view));
  if (finding.cropped) {
    view.append(el('span', 'thought-cropped', 'cropped to the truck'));
  }
  heading.append(view);
  heading.append(el('span', 'thought-status', statusLine(finding, issues)));

  const tags = el('span', 'thought-parts');
  uniqueParts(issues).forEach((part) => {
    const tag = el('span', 'part-tag');
    tag.append(partIcon(part), el('span', null, titleise(part)));
    tags.append(tag);
  });
  if (tags.children.length) heading.append(tags);

  summary.append(thumb, heading);
  card.append(summary);

  const body = el('div', 'thought-body');
  if (finding.shows) body.append(el('p', 'thought-shows', finding.shows));
  if (finding.error) {
    body.append(el('p', 'thought-quiet', 'this frame could not be read'));
  } else {
    const list = el('ul', 'thought-list');
    issues.forEach((issue) => list.append(citeIssue(finding.photo_id, issue)));
    (finding.strengths || []).slice(0, issues.length ? 2 : 4).forEach((good) => {
      const row = el('li');
      row.dataset.sev = 'ok';
      row.append(partIcon('check'), el('p', null, good));
      list.append(row);
    });
    if (list.children.length) body.append(list);
    else body.append(el('p', 'thought-quiet', 'nothing to flag on this frame'));
  }

  const source = el('button', 'thought-source', 'Open this photo');
  source.type = 'button';
  source.addEventListener('click', (ev) => {
    ev.preventDefault();
    citePhoto(finding.photo_id);
  });
  body.append(source);
  card.append(body);

  rail.prepend(card);
  if (!atTop) rail.scrollTop += rail.scrollHeight - previousHeight;
  if (!reduced()) {
    card.classList.add('resolve');
    for (const [node, top] of previousCards) {
      const delta = top - node.getBoundingClientRect().top;
      node.getAnimations().forEach((animation) => animation.cancel());
      node.animate([{ transform: `translateY(${delta}px)` }, { transform: 'translateY(0)' }],
        { duration: 480, easing: 'cubic-bezier(.22,1,.36,1)' });
    }
  }

  landed += 1;
  $('rail-sub').textContent = expected
    ? `${landed} of ${expected} frames read`
    : `${landed} frames read`;
  return card;
}

function citeIssue(photoId, issue) {
  const row = el('li', 'thought-cite');
  row.dataset.sev = issue.severity;
  const copy = el('div');
  copy.append(el('strong', null, titleise(issue.component)));
  copy.append(el('p', null, issue.observation));
  row.append(partIcon(issue.component), copy);
  row.tabIndex = 0;
  row.setAttribute('role', 'button');
  const open = (ev) => { ev.stopPropagation(); citePhoto(photoId, issue); };
  row.addEventListener('click', open);
  row.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); open(ev); }
  });
  return row;
}

function statusLine(finding, issues) {
  if (finding.error) return 'Could not read this photo';
  if (!issues.length) return 'Nothing to flag';
  const n = issues.length;
  return `${n} observation${n === 1 ? '' : 's'} · ${issues[0].severity}`;
}

function uniqueParts(issues) {
  const seen = [];
  issues.forEach((issue) => {
    if (issue.component && !seen.includes(issue.component)) seen.push(issue.component);
  });
  return seen.slice(0, 4);
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
    /* The stage headline beside this one already says the run stopped, so
       saying it twice in two headings wastes the only line the rail has left.
       This one says the part that is worth knowing: nothing was spent. */
    $('rail-title').textContent = 'No photo went to a vision model';
    $('rail-sub').textContent = 'the checks that run first stopped it';
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
