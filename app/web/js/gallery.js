/* The truck gallery: one grid, every vehicle, filtered in the browser.

   Two hundred rows is small enough that filtering is a `filter()` over an
   array held in memory - no paging, no request per keystroke, and the count
   updates in the same frame as the chip. The covers are server-side
   thumbnails; the originals are 3-5 MP dealer photographs and 200 of those
   would take longer to paint than an appraisal takes to run.

   Nothing here is pressed-by-default. No chip selected means no filter, so the
   grid opens as the whole corpus and narrowing it is always something the
   person did. */

import { $, el, reduced } from './dom.js';

const KM_BANDS = [
  { id: 'low',  label: 'Under 300k', test: (km) => km != null && km < 300000 },
  { id: 'mid',  label: '300–500k',   test: (km) => km != null && km >= 300000 && km < 500000 },
  { id: 'high', label: 'Over 500k',  test: (km) => km != null && km >= 500000 },
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

const kkm = (km) => (km == null ? null : `${Math.round(km / 1000)}k km`);

function matches(card) {
  if (card.demo) return true;   // the rehearsed cases always stay reachable
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
      paint();
    }));
  });
  return wrap;
}

const anyPicked = () => Object.values(picked).some((s) => s.size);

function buildFilters(facets) {
  const row = $('filters');
  row.replaceChildren(
    group('Make', facets.makes.map((m) => ({ id: m.name, label: m.name, count: m.n })), 'make'),
    group('Year', YEAR_BANDS.map((b) => ({ id: b.id, label: b.label })), 'year'),
    group('Distance', KM_BANDS.map((b) => ({ id: b.id, label: b.label })), 'km'),
    group('Photos', CAPTURE.map((c) => ({ id: c.id, label: c.label })), 'capture'));

  const clear = el('button', 'filter-clear', 'Clear filters');
  clear.type = 'button';
  clear.id = 'filter-clear';
  clear.hidden = true;
  clear.addEventListener('click', () => {
    Object.values(picked).forEach((s) => s.clear());
    buildFilters(facets);
    paint();
  });
  row.append(clear);
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
  const spec = [card.year, kkm(card.km), `${card.n_photos} photos`]
    .filter(Boolean).join('   ');
  scrim.append(el('span', 'truck-spec', spec));
  b.append(scrim);

  b.title = card.demo ? card.blurb : `${card.n_photos} photos of this vehicle`;
  b.addEventListener('click', () => onPick(card));
  return b;
}

function paint() {
  const grid = $('gallery');
  const shown = cards.filter(matches);
  grid.replaceChildren(...shown.map(build));
  $('gallery-empty').hidden = shown.length > 0;

  const real = shown.filter((c) => !c.demo).length;
  $('gallery-count').textContent = anyPicked()
    ? `${real} of ${cards.filter((c) => !c.demo).length} trucks`
    : `${real} real listings, and ${shown.length - real} worth watching it get wrong`;

  /* One deal-in, on the first paint only. Re-running it on every filter click
     would turn a fast, quiet interaction into a light show. */
  if (!dealt && !reduced()) {
    dealt = true;
    [...grid.children].slice(0, 12).forEach((node, i) => {
      node.classList.add('deal');
      node.style.animationDelay = `${i * 28}ms`;
    });
  }
}

export function render(data, pick) {
  cards = data.cards || [];
  onPick = pick;
  buildFilters(data.facets || { makes: [] });
  paint();
}

export function fail(message) {
  $('gallery-count').textContent = message;
  $('gallery').replaceChildren();
}
