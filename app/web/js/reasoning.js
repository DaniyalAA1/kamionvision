/* The reasoning rail: one card per photo, as each photo's own vision call
   returns.

   Every line on this rail is something the model actually said about the frame
   beside it.    Cards land out of order, because concurrent calls finish out of
   order. The newest goes on top. Nothing here re-sorts, because re-sorting
   would hide the one honest thing the rail has to say, which is that this is
   arriving as it happens.

   The card is the same object as a result finding: a thumb, a part, one
   observation. Open it for the rest. A rail of sixteen open paragraphs is
   unreadable in the minute the calls take. */

import { $, el, viewAngle, reduced, titleise } from './dom.js';
import { urlFor, citePhoto, checkFor, showFrame, markCell } from './frames.js';
import { partIcon } from './icons.js';

const SEV_RANK = { major: 3, moderate: 2, minor: 1, cosmetic: 0 };

export function begin(total) {

  const rail = $('thoughts');
  rail.replaceChildren();
  for (let i = 0; i < Math.min(total, 3); i += 1) {
    rail.append(pendingCard());
  }
}

export function clear() {
  $('thoughts').replaceChildren();
}

function pendingCard() {
  const n = el('div', 'thought-pending');
  n.append(partIcon('camera'), el('span', null, 'Reading…'));
  return n;
}

export function note(message) {
  const rail = $('thoughts');
  rail.querySelectorAll('.thought-pending').forEach((n) => n.remove());
  const card = el('div', 'thought thought-note');
  card.append(el('p', 'thought-preview', message));
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

  const issues = [...(finding.issues || [])].sort(
    (a, b) => (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0));

  const card = el('details', 'thought');
  const worst = worstOf(finding);
  card.dataset.worst = worst;
  if (finding.error) card.dataset.error = 'true';

  const summary = el('summary', 'thought-summary');
  const thumb = el('img', 'thought-thumb');
  thumb.src = urlFor(finding.photo_id);
  thumb.alt = '';
  thumb.loading = 'lazy';

  const heading = el('span', 'thought-heading');
  const title = el('strong', 'thought-view', viewAngle(finding.view));
  if (finding.cropped) {
    title.append(el('span', 'thought-cropped', ' · cropped'));
  }
  heading.append(title);

  const lead = issues[0];
  const previewText = finding.error
    ? finding.error
    : (lead ? lead.observation : (finding.shows || ''));
  if (previewText) heading.append(el('p', 'thought-preview', previewText));

  const tags = el('span', 'thought-parts');
  uniqueParts(issues).forEach((part) => {
    const tag = el('span', 'part-tag');
    tag.append(partIcon(part), el('span', null, titleise(part)));
    tags.append(tag);
  });
  if (tags.children.length) heading.append(tags);

  summary.append(thumb, heading);
  summary.addEventListener('click', () => {
    const check = checkFor(finding.photo_id);
    if (check) { showFrame(check); markCell(finding.photo_id); }
  });
  card.append(summary);

  const rest = issues.slice(lead ? 1 : 0);
  if (finding.error || rest.length) {
    const body = el('div', 'thought-body');
    if (!finding.error) {
      const list = el('ul', 'thought-list');
      rest.forEach((issue) => list.append(citeIssue(finding.photo_id, issue)));
      body.append(list);
    }
    const source = el('button', 'thought-source', 'Open this photo');
    source.type = 'button';
    source.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      citePhoto(finding.photo_id);
    });
    body.append(source);
    card.append(body);
  }

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
  void evidence;
}
