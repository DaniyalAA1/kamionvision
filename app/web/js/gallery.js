/* The truck gallery: a shelf of rehearsed cases, then a wall you can page.

   Two hundred rows is small enough that filtering is a `filter()` over an
   array held in memory - no request per keystroke, and the count updates in
   the same frame as the checkbox. The covers are server-side thumbnails; the
   originals are 3-5 MP dealer photographs and painting all 197 at once is
   slower than the gate, so a page is 24 and the rest are a page number away.

   The nine rehearsed cases sit in their own shelf above the wall instead of
   pinned inside it. They used to be the first cards of the grid AND exempt
   from every filter, which meant narrowing to Ford left a motorcycle sitting
   among the Fords - the single most confusing thing on the screen. Separating
   them lets the filters below apply to every card with no exceptions, and a
   refusal stays one click away on stage, which is what the pinning was for.

   A card is still a photograph with its specification sitting on it. The
   search, the filters and the sort sit above the wall; they do not turn it
   into a table.

   Country is a browse filter. It does not change how a truck is priced - that
   stays market=TR in app.js, because the price model is tr_only. */

import { $, el, reduced } from './dom.js';
import { partIcon } from './icons.js';
import * as filters from './filters.js';
import * as pager from './pager.js';

const PAGE = pager.PAGE_SIZE;

/* What each rehearsed case will actually do. The only thing worth knowing
   about one before you press it, so it is the only thing the card says. */
const EXPECT_LABEL = {
  ok: 'prices it',
  'ok|ok_with_requests': 'prices it',
  need_more_photos: 'asks for more',
  refused: 'refuses',
};

/* Short enough to sit in a scrim beside a year and a mileage. The filter that
   selects them is spelled out in full - Türkiye, United States. */
const MARKET_TAG = { TR: 'Türkiye', US: 'USA' };

const SORTS = [
  ['default', 'Recommended'],
  ['year', 'Newest first'],
  ['km', 'Lowest mileage'],
  ['photos', 'Most photos'],
];

let listings = [];
let demos = [];
let onPick = () => {};
let facets = null;
let query = '';
let sort = 'default';
let page = 1;
let dealt = false;
const nodes = new Map();

const kkm = (km) => (km == null ? null : `${Math.round(km / 1000)}k km`);

/* Search and the checkboxes are the same narrowing to a reader, but only the
   search is shared with the facet counts - a count has to answer "how many if
   I also tick this", which means every other filter applies to it and the
   group being counted does not. */
function inSearch(card) {
  if (!query) return true;
  const hay = `${card.make} ${card.model} ${card.title || ''} ${card.year || ''}`;
  return hay.toLowerCase().includes(query);
}

const shownCards = () => listings.filter((c) => inSearch(c) && facets.matches(c));

function sorted(rows) {
  if (sort === 'year') return rows.sort((a, b) => (b.year || 0) - (a.year || 0));
  if (sort === 'km') return rows.sort((a, b) => (a.km ?? Infinity) - (b.km ?? Infinity));
  if (sort === 'photos') return rows.sort((a, b) => b.n_photos - a.n_photos);
  return rows;
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
  } else if (MARKET_TAG[card.market]) {
    /* Where it was advertised, on the card, so the Country filter is visibly
       doing something rather than silently thinning the wall. */
    b.append(el('span', 'truck-lot', MARKET_TAG[card.market]));
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

/* One deal-in, on the first paint of the wall only. A later page arrives as a
   short rise so a page change reads as a page change - the old version FLIPed
   every surviving card to a new position, which is the right animation for a
   list growing in place and the wrong one for a wall being replaced. */
function arrive(grid, first) {
  if (reduced()) return;
  [...grid.children].slice(0, first ? 12 : 24).forEach((node, i) => {
    node.classList.remove('deal');
    void node.offsetWidth;
    node.style.animationDelay = `${Math.min(i, 11) * (first ? 28 : 14)}ms`;
    node.classList.add('deal');
    node.addEventListener('animationend', () => {
      node.classList.remove('deal');
      node.style.animationDelay = '';
    }, { once: true });
  });
}

/* The empty state below the grid already says "no trucks match those filters",
   so this one stays a tally rather than repeating the sentence back. */
function countLine(shown) {
  const total = listings.length;
  if (!shown.length) return `0 of ${total} trucks`;
  if (shown.length === total) {
    return `${total} real listings, and ${demos.length} rehearsed cases above`;
  }
  return `${shown.length} of ${total} trucks`;
}

function paint(opts = {}) {
  const grid = $('gallery');
  const shown = sorted(shownCards());
  const pages = pager.pageCount(shown.length);
  page = Math.min(Math.max(1, page), pages);
  const from = (page - 1) * PAGE;
  const visible = shown.slice(from, from + PAGE);

  grid.replaceChildren(...visible.map(nodeFor));
  arrive(grid, !dealt);
  dealt = true;

  $('gallery-count').textContent = countLine(shown);
  $('gallery-empty').hidden = shown.length > 0;
  $('gallery-range').textContent = shown.length
    ? `Showing ${from + 1}–${from + visible.length} of ${shown.length}` : '';
  pager.render($('gallery-pager'), {
    page,
    pages,
    onGo: (to) => { page = to; paint({ jump: true }); },
  });

  /* Landing at the top of a new page rather than wherever the old one's
     scroll position happened to leave you. Only on a page press: retyping in
     the search box must not yank the screen around. */
  if (opts.jump) {
    $('browse').scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
  }
}

/* Any narrowing puts you back on page one. Staying on page 4 of a result set
   that is now two pages long is how "the filter did nothing" happens. */
function narrow() {
  page = 1;
  paint();
}

function buildTools() {
  if ($('gallery-tools').children.length) return;

  const search = el('label', 'gallery-search');
  search.append(partIcon('search'));
  const input = el('input');
  input.type = 'search';
  input.placeholder = 'Make, model or year';
  input.setAttribute('aria-label', 'Search trucks by make, model or year');
  input.autocomplete = 'off';
  input.addEventListener('input', () => {
    query = input.value.trim().toLowerCase();
    facets.refresh();
    narrow();
  });
  search.append(input);

  const sorting = el('select', 'gallery-sort');
  sorting.setAttribute('aria-label', 'Sort trucks');
  SORTS.forEach(([value, label]) => {
    const opt = el('option', null, label);
    opt.value = value;
    sorting.append(opt);
  });
  sorting.addEventListener('change', () => { sort = sorting.value; narrow(); });

  $('gallery-tools').append(search, sorting);
}

function buildShelf() {
  const shelf = $('case-shelf');
  shelf.replaceChildren(...demos.map(nodeFor));
  $('shelf-block').hidden = demos.length === 0;
}

export function render(data, pick) {
  const cards = data.cards || [];
  demos = cards.filter((c) => c.demo);
  listings = cards.filter((c) => !c.demo);
  onPick = pick;

  buildTools();
  if (!facets) {
    facets = filters.create({
      bar: $('facet-bar'),
      pills: $('facet-pills'),
      onChange: narrow,
    });
    facets.setBase(inSearch);
  }
  facets.setCards(listings);
  facets.refresh();
  buildShelf();
  paint();
}

export function fail(message) {
  $('gallery-count').textContent = message;
  $('gallery').replaceChildren();
  $('shelf-block').hidden = true;
}
