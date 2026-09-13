/* The reasoning rail: a desk of prints.

   Waiting photographs sit as a fanned deck. The identity read is one plate
   on top of that pile, updated as each sample lands. A close-up in flight
   is the photograph itself, not a placeholder row. A finished call is the
   same print with the observation written into the frame.

   Cards land out of order, because concurrent calls finish out of order.
   Newest work sits above the unread deck. Nothing here re-sorts. */

import { $, el, viewAngle, reduced, titleise } from './dom.js';
import { urlFor, citePhoto, checkFor, showFrame, markCell } from './frames.js';
import { partIcon } from './icons.js';

const SEV_RANK = { major: 3, moderate: 2, minor: 1, cosmetic: 0 };

export function begin(ids) {
  const rail = $('thoughts');
  rail.replaceChildren();
  const list = Array.isArray(ids) ? ids : [];
  if (!list.length) return;
  const deck = el('div', 'thought-deck');
  deck.id = 'thought-deck';
  list.forEach((id, i) => {
    const img = el('img', 'thought-deck-print');
    img.src = urlFor(id);
    img.alt = '';
    img.dataset.photo = String(id);
    img.style.setProperty('--i', String(i));
    deck.append(img);
  });
  rail.append(deck);
}

export function clear() {
  hidePop();
  $('thoughts').replaceChildren();
}

export function identity(msg) {
  const rail = $('thoughts');
  let plate = $('thought-ident');
  if (!plate) {
    plate = el('article', 'thought thought-ident');
    plate.id = 'thought-ident';
    const fan = el('div', 'thought-fan');
    (msg.photo_ids || []).forEach((id) => {
      const img = el('img');
      img.src = urlFor(id);
      img.alt = '';
      fan.append(img);
    });
    const cap = el('div', 'thought-caption');
    cap.id = 'thought-ident-cap';
    plate.append(fan, cap);
    rail.prepend(plate);
  } else if (!(plate.querySelectorAll('.thought-fan img').length)
             && (msg.photo_ids || []).length) {
    const fan = plate.querySelector('.thought-fan') || el('div', 'thought-fan');
    fan.replaceChildren();
    msg.photo_ids.forEach((id) => {
      const img = el('img');
      img.src = urlFor(id);
      img.alt = '';
      fan.append(img);
    });
    if (!fan.parentNode) plate.prepend(fan);
  }
  const who = [msg.make, msg.model].filter(Boolean).join(' ');
  const n = (msg.sample && msg.of) ? `${msg.sample} of ${msg.of}` : '';
  const cap = $('thought-ident-cap');
  if (!cap) return;
  cap.replaceChildren();
  if (msg.status === 'reading') {
    cap.append(el('strong', 'thought-view', n ? `Identity ${n}` : 'Identity'));
    cap.append(el('p', 'thought-preview', 'Same vehicle, from the photographs'));
  } else if (who) {
    cap.append(el('strong', 'thought-view', who));
    cap.append(el('p', 'thought-preview',
      msg.same_vehicle === false
        ? 'These photographs may not be one vehicle'
        : (n ? n : 'From the photographs')));
  } else {
    cap.append(el('strong', 'thought-view', 'Identity'));
    cap.append(el('p', 'thought-preview', n || 'From the photographs'));
  }
  if (msg.status === 'done') plate.classList.add('settled');
  if (msg.same_vehicle === false) plate.dataset.worst = 'major';
}

export function reading(photoId) {
  const rail = $('thoughts');
  if (rail.querySelector(`.thought-live[data-photo="${photoId}"]`)) return;
  const waiting = rail.querySelector(`.thought-deck-print[data-photo="${photoId}"]`);
  if (waiting) waiting.classList.add('is-up');
  const plate = el('article', 'thought thought-live');
  plate.dataset.photo = String(photoId);
  const print = el('div', 'thought-print');
  const img = el('img');
  img.src = urlFor(photoId);
  img.alt = '';
  const cap = el('div', 'thought-caption');
  const check = checkFor(photoId);
  cap.append(el('strong', 'thought-view', viewAngle(check && check.view)));
  print.append(img, cap);
  plate.append(print);
  insertWork(rail, plate);
}

export function note(message) {
  const rail = $('thoughts');
  const card = el('div', 'thought thought-note');
  card.append(el('p', 'thought-preview', message));
  insertWork(rail, card);
}

/* One finished call. `finding` is the PhotoFinding straight off the wire. */
export function add(finding) {
  const rail = $('thoughts');
  const atTop = rail.scrollTop < 12;
  const previousHeight = rail.scrollHeight;
  const previousCards = atTop && !reduced()
    ? [...rail.querySelectorAll('.thought:not(.thought-ident)')].map((node) => [node, node.getBoundingClientRect().top]) : [];

  const waiting = rail.querySelector(`.thought-deck-print[data-photo="${finding.photo_id}"]`);
  if (waiting) waiting.classList.add('is-up');
  const live = rail.querySelector(`.thought-live[data-photo="${finding.photo_id}"]`);

  const issues = [...(finding.issues || [])].sort(
    (a, b) => (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0));

  const card = el('details', 'thought');
  const worst = worstOf(finding);
  card.dataset.worst = worst;
  if (finding.error) card.dataset.error = 'true';
  card.dataset.photo = String(finding.photo_id);

  const summary = el('summary', 'thought-summary');
  const print = el('div', 'thought-print');
  const thumb = el('img');
  thumb.src = urlFor(finding.photo_id);
  thumb.alt = '';
  thumb.loading = 'lazy';
  const cap = el('div', 'thought-caption');
  const title = el('strong', 'thought-view', viewAngle(finding.view));
  if (finding.cropped) {
    title.append(el('span', 'thought-cropped', ' · cropped'));
  }
  cap.append(title);
  const lead = issues[0];
  const previewText = finding.error
    ? finding.error
    : (lead ? lead.observation : (finding.shows || ''));
  if (previewText) cap.append(el('p', 'thought-preview', previewText));
  print.append(thumb, cap);
  summary.append(print);
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

  if (live) live.replaceWith(card);
  else insertWork(rail, card);

  if (!atTop) rail.scrollTop += rail.scrollHeight - previousHeight;
  if (!reduced()) {
    card.classList.add('resolve');
    for (const [node, top] of previousCards) {
      if (!node.isConnected) continue;
      const delta = top - node.getBoundingClientRect().top;
      node.getAnimations().forEach((animation) => animation.cancel());
      node.animate([{ transform: `translateY(${delta}px)` }, { transform: 'translateY(0)' }],
        { duration: 480, easing: 'cubic-bezier(.22,1,.36,1)' });
    }
  }

  return card;
}

function insertWork(rail, node) {
  const deck = $('thought-deck');
  if (deck) rail.insertBefore(node, deck);
  else rail.append(node);
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

function worstOf(finding) {
  if (finding.error) return 'error';
  const issues = finding.issues || [];
  if (!issues.length) return 'clean';
  const worst = issues.reduce(
    (a, b) => ((SEV_RANK[a.severity] ?? 0) >= (SEV_RANK[b.severity] ?? 0) ? a : b));
  return worst.severity;
}

function boxedIssues(finding) {
  return (finding.issues || []).filter((issue) =>
    Array.isArray(issue.box) && issue.box.length === 4);
}

function popSide(finding) {
  const boxed = boxedIssues(finding);
  if (!boxed.length) return 'right';
  const lead = boxed.reduce(
    (a, b) => ((SEV_RANK[a.severity] ?? 0) >= (SEV_RANK[b.severity] ?? 0) ? a : b));
  const cx = lead.box[0] + (lead.box[2] || 0) / 2;
  return cx > 0.62 ? 'left' : 'right';
}

/* The finding for the photograph on the stage. It sits on the photo's
   right unless that would cover the yellow box. A clean frame is silent. */
export function showPop(finding) {
  const pop = $('scan-pop');
  if (!pop) return false;
  const issues = [...(finding.issues || [])].sort(
    (a, b) => (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0));
  const lead = issues[0];
  const text = finding.error
    ? finding.error
    : (lead ? lead.observation : '');
  if (!text) {
    hidePop();
    return false;
  }
  pop.replaceChildren();
  pop.dataset.side = popSide(finding);
  pop.dataset.worst = worstOf(finding);
  const row = el('div', 'scan-pop-lead');
  if (lead && !finding.error) row.append(partIcon(lead.component));
  const copy = el('div');
  const title = el('strong', 'scan-pop-view', viewAngle(finding.view));
  if (lead && lead.severity) {
    title.append(el('span', 'scan-pop-sev', ` · ${lead.severity}`));
  }
  copy.append(title, el('p', 'scan-pop-copy', text));
  row.append(copy);
  pop.append(row);
  pop.hidden = false;
  pop.classList.remove('is-out');
  void pop.offsetWidth;
  pop.classList.add('is-in');
  return true;
}

export function hidePop() {
  const pop = $('scan-pop');
  if (!pop || pop.hidden) {
    if (pop) {
      pop.replaceChildren();
      pop.classList.remove('is-in', 'is-out');
    }
    return;
  }
  pop.classList.remove('is-in');
  if (reduced()) {
    pop.hidden = true;
    pop.replaceChildren();
    pop.classList.remove('is-out');
    return;
  }
  pop.classList.add('is-out');
  setTimeout(() => {
    if (!pop.classList.contains('is-out')) return;
    pop.hidden = true;
    pop.replaceChildren();
    pop.classList.remove('is-out');
  }, 280);
}

/* The rail's closing state. It keeps every print - a reader who wants to know
   where a finding came from can scroll back through the frames it came from. */
export function done(evidence) {
  hidePop();
  const deck = $('thought-deck');
  if (deck && !deck.querySelector('.thought-deck-print:not(.is-up)')) deck.remove();
  $('thoughts').querySelectorAll('.thought').forEach((n) => n.classList.add('settled'));
  $('thoughts').querySelectorAll('.thought-live').forEach((n) => n.remove());
  void evidence;
}
