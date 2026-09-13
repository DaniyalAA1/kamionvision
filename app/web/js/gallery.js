/* The truck gallery: one grid, every vehicle, filtered in the browser.

   Two hundred rows is small enough that filtering is a `filter()` over an
   array held in memory - no request per keystroke, and the count updates in
   the same frame as the chip. The covers are server-side thumbnails; the
   originals are 3-5 MP dealer photographs and painting all 200 at once is
   slower than the gate, so the first screenful is 24 and the rest arrive
   when you ask.

   A card is still a photograph with its specification sitting on it. The
   search, the tabs and the sort sit above the wall; they do not turn it into
   a table.

   The lot switch (Both / Türkiye / US) is a browse filter. It does not change
   how a truck is priced - that stays `market=TR` in app.js.

   Nothing here is pressed-by-default. No chip selected means no filter, so the
   grid opens as the whole corpus and narrowing it is always something the
   person did. */

import { $, el, reduced } from './dom.js';
import { partIcon } from './icons.js';

const PAGE = 24;
const EASE = 'cubic-bezier(.16, 1, .3, 1)';

const KM_BANDS = [
  { id: 'low',  label: 'Under 300k', test: (km) => km != null && km < 300000 },
  { id: 'mid',  label: '300–500k',   test: (km) => km != null && km >= 300000 && km < 500000 },
  { id: 'high', label: 'Over 500k',   test: (km) => km != null && km >= 500000 },
];

const YEAR_BANDS = [
  { id: 'y20', label: '2020 and newer', test: (y) => y != null && y >= 2020 },
  { id: 'y15', label: '2015–2019',      test: (y) => y != null && y >= 2015 && y < 2020 },
  { id: 'y10', label: '2010–2014',      test: (y) => y != null && y >= 2010 && y < 2015 },
  { id: 'y00', label: 'Before 2010',    test: (y) => y != null && y < 2010 },
];

const CAPTURE = [
  { id: 'dealer', label: 'Dealer photos' },
  { id: 'phone',  label: 'Phone photos' },
];

/* What each rehearsed case will actually do. The only thing worth knowing
   about one before you press it, so it is the only thing the card says. */
const EXPECT_LABEL = {
  ok: 'prices it',
  'ok|ok_with_requests': 'prices it',
  need_more_photos: 'asks for more',
  refused: 'refuses',
};

let cards = [];
let onPick = () => {};
const picked = { make: new Set(), year: new Set(), km: new Set(), capture: new Set() };
let dealt = false;
let query = '';
let scope = 'all';
let market = 'all';
let sort = 'default';
let limit = PAGE;
const nodes = new Map();
const thumbs = new Set();

const kkm = (km) => (km == null ? null : `${Math.round(km / 1000)}k km`);

function inLot(card) {
  if (market === 'all') return true;
  if (card.market) return card.market === market;
  /* A refusal case has no listing behind it, so it stays reachable on both
     lots - otherwise switching to the US wall would hide "not a truck". */
  return !!card.demo;
}

function matches(card) {
  if (!inLot(card)) return false;
  if (scope === 'demo' && !card.demo) return false;
  if (scope === 'listings' && card.demo) return false;
  if (query) {
    const hay = `${card.make} ${card.model} ${card.title || ''} ${card.year || ''}`.toLowerCase();
    if (!hay.includes(query)) return false;
  }
  if (card.demo) return true;
  if (picked.make.size && !picked.make.has(card.make)) return false;
  if (picked.year.size &&
      !YEAR_BANDS.some((b) => picked.year.has(b.id) && b.test(card.year))) return false;
  if (picked.km.size &&
      !KM_BANDS.some((b) => picked.km.has(b.id) && b.test(card.km))) return false;
  if (picked.capture.size && !picked.capture.has(card.capture)) return false;
  return true;
}

function chip(label, count, pressed, onToggle) {
  const b = el('button', 'chip');
  b.type = 'button';
  b.setAttribute('aria-pressed', String(pressed));
  b.append(document.createTextNode(label));
  if (count != null) b.append(el('b', null, String(count)));
  b.addEventListener('click', () => onToggle(b));
  return b;
}

function group(label, items, bucket) {
  const wrap = el('div', 'filter-group');
  wrap.append(el('span', 'filter-label', label));
  items.forEach(({ id, label: text, count }) => {
    wrap.append(chip(text, count, picked[bucket].has(id), (b) => {
      picked[bucket].has(id) ? picked[bucket].delete(id) : picked[bucket].add(id);
      b.setAttribute('aria-pressed', String(picked[bucket].has(id)));
      $('filter-clear').hidden = !anyPicked();
      limit = PAGE;
      paint();
    }));
  });
  return wrap;
}

const anyPicked = () => Object.values(picked).some((s) => s.size);

function makesForLot() {
  const counts = {};
  for (const card of cards) {
    if (card.demo || !inLot(card) || card.make === 'Unknown') continue;
    counts[card.make] = (counts[card.make] || 0) + 1;
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([name, n]) => ({ id: name, label: name, count: n }));
}

function pruneMakes() {
  const valid = new Set(makesForLot().map((m) => m.id));
  for (const name of [...picked.make]) {
    if (!valid.has(name)) picked.make.delete(name);
  }
}

function buildFilters() {
  const row = $('filters');
  let lot = $('filter-lot');
  if (!lot) {
    lot = el('div', 'filter-group filter-lot');
    lot.id = 'filter-lot';
    lot.append(el('span', 'filter-label', 'Lot'));
    lot.append(segmented({
      id: 'gallery-market',
      label: 'Where the truck was listed',
      options: [['all', 'Both'], ['TR', 'Türkiye'], ['US', 'US']],
      get: () => market,
      set: (value) => {
        market = value;
        pruneMakes();
        buildFilters();
      },
    }));
  }
  const clear = el('button', 'filter-clear', 'Clear filters');
  clear.type = 'button';
  clear.id = 'filter-clear';
  clear.hidden = !anyPicked();
  clear.addEventListener('click', () => {
    Object.values(picked).forEach((s) => s.clear());
    buildFilters();
    limit = PAGE;
    paint();
  });
  if (!lot.parentNode) row.append(lot);
  [...row.children].forEach((n) => { if (n !== lot) n.remove(); });
  row.append(
    group('Make', makesForLot(), 'make'),
    group('Year', YEAR_BANDS.map((b) => ({ id: b.id, label: b.label })), 'year'),
    group('Distance', KM_BANDS.map((b) => ({ id: b.id, label: b.label })), 'km'),
    group('Photos', CAPTURE.map((c) => ({ id: c.id, label: c.label })), 'capture'),
    clear);
}

function specItem(kind, text) {
  const item = el('span', 'truck-spec-item');
  item.append(partIcon(kind), document.createTextNode(String(text)));
  return item;
}

function build(card) {
  const b = el('button', 'truck');
  b.type = 'button';
  b.dataset.id = card.id;

  if (card.demo && !card.available) {
    b.disabled = true;
    b.append(el('div', 'truck-missing',
                'Fixtures missing — run python -m app.demo --build'));
    return b;
  }

  const img = el('img');
  img.loading = 'lazy';
  img.decoding = 'async';
  img.alt = '';
  img.src = card.cover;
  img.addEventListener('error', () => {
    img.remove();
    b.classList.add('photo-unavailable');
    b.prepend(el('span', 'photo-fallback', 'Photo unavailable'));
  }, { once: true });
  img.addEventListener('load', () => img.classList.add('in'), { once: true });
  b.append(img);

  if (card.demo) {
    const flag = el('span', 'truck-flag', EXPECT_LABEL[card.expect] || 'rehearsed');
    flag.dataset.expect = card.expect.includes('refused') ? 'refused'
      : card.expect.includes('need_more') ? 'need_more_photos' : 'ok';
    b.append(flag);
  }

  const scrim = el('div', 'truck-scrim');
  scrim.append(el('span', 'truck-name',
                  card.demo ? card.title
                            : `${card.make} ${card.model}`.trim() || card.make));
  const specs = el('span', 'truck-spec');
  if (card.year) specs.append(specItem('gauge', card.year));
  const km = kkm(card.km);
  if (km) specs.append(specItem('wheel', km));
  specs.append(specItem('camera', `${card.n_photos} photos`));
  scrim.append(specs);
  b.append(scrim);

  b.title = card.demo ? card.blurb : `${card.n_photos} photos of this vehicle`;
  b.addEventListener('click', () => onPick(card));
  return b;
}

function nodeFor(card) {
  if (!nodes.has(card.id)) nodes.set(card.id, build(card));
  return nodes.get(card.id);
}

function cancelMotion(node) {
  if (typeof node.getAnimations === 'function') {
    node.getAnimations().forEach((a) => a.cancel());
  }
}

function exitLayer() {
  let layer = $('gallery-exit');
  if (!layer) {
    layer = el('div', 'gallery-exit');
    layer.id = 'gallery-exit';
    layer.setAttribute('aria-hidden', 'true');
    $('gallery').after(layer);
  }
  return layer;
}

function ghostOut(node, rect) {
  const ghost = node.cloneNode(true);
  ghost.classList.add('truck-exit');
  ghost.tabIndex = -1;
  ghost.disabled = true;
  ghost.setAttribute('aria-hidden', 'true');
  Object.assign(ghost.style, {
    position: 'fixed',
    left: `${rect.left}px`,
    top: `${rect.top}px`,
    width: `${rect.width}px`,
    height: `${rect.height}px`,
    margin: '0',
    zIndex: '2',
    pointerEvents: 'none',
  });
  exitLayer().append(ghost);
  const fade = ghost.animate(
    [{ opacity: 1, transform: 'none' },
     { opacity: 0, transform: 'scale(.97)' }],
    { duration: 280, easing: EASE, fill: 'forwards' });
  fade.onfinish = () => ghost.remove();
}

function flipIn(node, was, index) {
  const now = node.getBoundingClientRect();
  cancelMotion(node);
  if (was) {
    const dx = was.left - now.left;
    const dy = was.top - now.top;
    if (dx || dy) {
      node.animate(
        [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }],
        { duration: 420, easing: EASE });
    }
    return;
  }
  node.animate(
    [{ opacity: 0, transform: 'translateY(10px) scale(.985)' },
     { opacity: 1, transform: 'none' }],
    { duration: 380, delay: Math.min(index, 10) * 18, easing: EASE, fill: 'backwards' });
}

function paint() {
  const grid = $('gallery');
  const shown = cards.filter(matches);
  if (sort === 'year') {
    shown.sort((a, b) => (b.year || 0) - (a.year || 0));
  } else if (sort === 'km') {
    shown.sort((a, b) => (a.km ?? Infinity) - (b.km ?? Infinity));
  }
  const visible = shown.slice(0, limit);
  const nextIds = new Set(visible.map((c) => c.id));
  const oldNodes = [...grid.children];
  const prev = new Map(oldNodes.map((n) => [n.dataset.id, n.getBoundingClientRect()]));
  const moving = dealt && !reduced() && oldNodes.length > 0;

  if (moving) {
    oldNodes.forEach((n) => {
      if (!nextIds.has(n.dataset.id)) ghostOut(n, prev.get(n.dataset.id));
    });
  }

  grid.replaceChildren(...visible.map(nodeFor));

  const more = $('gallery-more');
  const leftover = shown.length - limit;
  more.hidden = leftover <= 0;
  more.textContent = leftover > 0
    ? `Show ${Math.min(PAGE, leftover)} more trucks` : '';
  $('gallery-visible').textContent = shown.length
    ? `${visible.length} of ${shown.length} shown` : '';
  $('gallery-empty').hidden = shown.length > 0;

  const real = shown.filter((c) => !c.demo).length;
  const demos = shown.length - real;
  const totalReal = cards.filter((c) => !c.demo && inLot(c)).length;
  if (scope === 'demo') {
    $('gallery-count').textContent =
      `${demos} guided case${demos === 1 ? '' : 's'}`;
  } else if (market === 'TR') {
    $('gallery-count').textContent = `${real} trucks listed in Türkiye`;
  } else if (market === 'US') {
    $('gallery-count').textContent = `${real} trucks listed in the US`;
  } else if (anyPicked() || query || scope !== 'all') {
    $('gallery-count').textContent = `${real} of ${totalReal} trucks`;
  } else {
    $('gallery-count').textContent =
      `${real} real listings, and ${demos} worth watching it get wrong`;
  }

  /* One deal-in, on the first paint only. Later paints FLIP the cards that
     stayed and fade the ones that arrived, so a tab click feels like the wall
     rearranging rather than a new page. */
  if (!dealt && !reduced()) {
    dealt = true;
    [...grid.children].slice(0, 12).forEach((node, i) => {
      node.classList.add('deal');
      node.style.animationDelay = `${i * 28}ms`;
      node.addEventListener('animationend', () => {
        node.classList.remove('deal');
        node.style.animationDelay = '';
      }, { once: true });
    });
    return;
  }
  dealt = true;
  if (moving) {
    [...grid.children].forEach((node, i) => flipIn(node, prev.get(node.dataset.id), i));
  }
}

function syncSeg(wrap, value, instant) {
  if (!wrap) return;
  wrap.querySelectorAll('.seg-btn').forEach((b) => {
    const on = b.dataset.value === value;
    b.setAttribute('aria-checked', String(on));
    b.tabIndex = on ? 0 : -1;
  });
  const thumb = wrap.querySelector('.seg-thumb');
  const active = [...wrap.querySelectorAll('.seg-btn')]
    .find((b) => b.dataset.value === value);
  if (!thumb || !active) return;
  const track = wrap.getBoundingClientRect();
  const at = active.getBoundingClientRect();
  const x = at.left - track.left;
  const y = at.top - track.top;
  const jump = instant || reduced() || !thumbs.has(wrap);
  if (jump) {
    thumb.style.transition = 'none';
    thumb.style.width = `${at.width}px`;
    thumb.style.height = `${at.height}px`;
    thumb.style.transform = `translate(${x}px, ${y}px)`;
    void thumb.offsetWidth;
    thumb.style.transition = '';
    thumbs.add(wrap);
    return;
  }
  thumb.style.width = `${at.width}px`;
  thumb.style.height = `${at.height}px`;
  thumb.style.transform = `translate(${x}px, ${y}px)`;
}

function segmented({ id, label, options, get, set }) {
  const wrap = el('div', 'seg');
  wrap.id = id;
  wrap.setAttribute('role', 'radiogroup');
  wrap.setAttribute('aria-label', label);
  const thumb = el('span', 'seg-thumb');
  thumb.setAttribute('aria-hidden', 'true');
  wrap.append(thumb);

  const apply = (value, instant) => {
    set(value);
    syncSeg(wrap, value, instant);
    limit = PAGE;
    paint();
  };

  options.forEach(([value, text]) => {
    const b = el('button', 'seg-btn', text);
    b.type = 'button';
    b.dataset.value = value;
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(get() === value));
    b.addEventListener('click', () => {
      if (get() === value) return;
      apply(value);
    });
    wrap.append(b);
  });

  wrap.addEventListener('keydown', (e) => {
    const keys = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 1, ArrowUp: -1 };
    const dir = keys[e.key];
    if (!dir) return;
    e.preventDefault();
    const i = options.findIndex(([value]) => value === get());
    const next = options[(i + dir + options.length) % options.length][0];
    apply(next);
    wrap.querySelector(`[data-value="${next}"]`)?.focus();
  });

  new ResizeObserver(() => syncSeg(wrap, get(), true)).observe(wrap);
  requestAnimationFrame(() => syncSeg(wrap, get(), true));
  return wrap;
}

function buildTools() {
  if ($('gallery-tools')) return;
  const tools = el('div', 'gallery-tools');
  tools.id = 'gallery-tools';

  const search = el('label', 'gallery-search');
  search.append(partIcon('search'));
  const input = el('input');
  input.type = 'search';
  input.placeholder = 'Make, model or year';
  input.setAttribute('aria-label', 'Search trucks by make, model or year');
  input.autocomplete = 'off';
  input.addEventListener('input', () => {
    query = input.value.trim().toLowerCase();
    limit = PAGE;
    paint();
  });
  search.append(input);

  const tabs = segmented({
    id: 'gallery-tabs',
    label: 'Which trucks',
    options: [['all', 'All trucks'], ['listings', 'Listings'], ['demo', 'Guided cases']],
    get: () => scope,
    set: (value) => { scope = value; },
  });
  tabs.classList.add('gallery-tabs');

  const sorting = el('select');
  sorting.setAttribute('aria-label', 'Sort trucks');
  [['default', 'Recommended'], ['year', 'Newest first'], ['km', 'Lowest mileage']].forEach(([value, label]) => {
    const opt = el('option', null, label);
    opt.value = value;
    sorting.append(opt);
  });
  sorting.addEventListener('change', () => {
    sort = sorting.value;
    limit = PAGE;
    paint();
  });

  tools.append(search, tabs, sorting);
  $('filters').before(tools);

  const foot = el('div', 'gallery-foot');
  const count = el('span');
  count.id = 'gallery-visible';
  const more = el('button', 'ghost-btn');
  more.id = 'gallery-more';
  more.type = 'button';
  more.addEventListener('click', () => {
    limit += PAGE;
    paint();
  });
  foot.append(count, more);
  $('gallery').after(foot);

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      syncSeg($('gallery-tabs'), scope, true);
      syncSeg($('gallery-market'), market, true);
    });
  }
}

export function render(data, pick) {
  cards = data.cards || [];
  onPick = pick;
  buildTools();
  buildFilters();
  paint();
}

export function fail(message) {
  $('gallery-count').textContent = message;
  $('gallery').replaceChildren();
}
