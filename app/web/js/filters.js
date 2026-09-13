/* The browse filters: one vocabulary, one control idiom.

   What this replaced was eight controls in two idioms - a segmented "Lot"
   switch, a second segmented All/Listings/Guided switch, four rows of chips
   and a sort menu - roughly twenty pressable things stacked above the trucks.
   The eye had to read all of them before it reached a photograph, and nothing
   on screen said what was currently applied.

   Five buttons now, each opening a checkbox list, and a row of removable pills
   underneath saying exactly what is on. That is the pattern every vehicle
   marketplace uses, which is the whole argument for it: the person looking at
   this screen has used one before.

   The counts are faceted. The number beside "Ford" is how many trucks you get
   if you add Ford to whatever is already selected - not how many Fords exist -
   so a count of zero means that box is a dead end, and it is dimmed. Counting
   any other way lets someone press three boxes in a row and land on an empty
   wall with no clue which one did it.

   Within a group the boxes are OR (Ford or MAN), across groups they are AND
   (a Ford, and 2020 or newer). That is what everyone expects and nobody says
   out loud, so it is not written on the screen either - but the counts make it
   observable, which is better than a legend. */

import { el } from './dom.js';

/* Bands rather than sliders. A used-truck buyer thinks in "under 300k", not in
   a continuous range, and three stops can carry a count where a slider cannot. */
const YEAR_BANDS = [
  { id: 'y20', label: '2020 and newer', test: (c) => c.year != null && c.year >= 2020 },
  { id: 'y15', label: '2015 – 2019', test: (c) => c.year != null && c.year >= 2015 && c.year < 2020 },
  { id: 'y10', label: '2010 – 2014', test: (c) => c.year != null && c.year >= 2010 && c.year < 2015 },
  { id: 'y00', label: 'Before 2010', test: (c) => c.year != null && c.year < 2010 },
];

const KM_BANDS = [
  { id: 'low', label: 'Under 300,000 km', test: (c) => c.km != null && c.km < 3e5 },
  { id: 'mid', label: '300,000 – 500,000 km', test: (c) => c.km != null && c.km >= 3e5 && c.km < 5e5 },
  { id: 'high', label: 'Over 500,000 km', test: (c) => c.km != null && c.km >= 5e5 },
];

/* `mixed` was in the data from the first harvest and was never given a chip,
   so 103 of 197 trucks could not be filtered to at all. */
const CAPTURE = [
  { id: 'dealer', label: 'Dealer photos', test: (c) => c.capture === 'dealer' },
  { id: 'phone', label: 'Phone photos', test: (c) => c.capture === 'phone' },
  { id: 'mixed', label: 'A mix of both', test: (c) => c.capture === 'mixed' },
];

/* Where the truck was advertised. A browse filter and nothing more: every
   appraisal still goes out as market=TR, because the price model is tr_only.
   The dropdown that let someone ask a Turkish fit for a number in dollars is
   the thing that stays gone. */
const COUNTRY = [
  { id: 'TR', label: 'Türkiye', test: (c) => c.market === 'TR' },
  { id: 'US', label: 'United States', test: (c) => c.market === 'US' },
];

const fixed = (options) => () => options;

/* The five groups, in the order a person narrows: what it is, how old, how
   far, where, then how it was photographed. */
export const GROUPS = [
  { id: 'make', label: 'Make', elementId: 'gallery-make', options: null },
  { id: 'year', label: 'Year', elementId: 'gallery-year', options: fixed(YEAR_BANDS) },
  { id: 'km', label: 'Mileage', elementId: 'gallery-km', options: fixed(KM_BANDS) },
  { id: 'market', label: 'Country', elementId: 'gallery-market', options: fixed(COUNTRY) },
  { id: 'capture', label: 'Photos', elementId: 'gallery-capture', options: fixed(CAPTURE) },
];

/* Makes come from the rows rather than a list in here, so a re-harvest that
   adds a manufacturer needs no edit in the browser. `gallery.pretty_make` has
   already collapsed FORD and Ford into one spelling. */
function makeOptions(cards) {
  const counts = {};
  for (const card of cards) {
    if (card.make === 'Unknown' || !card.make) continue;
    counts[card.make] = (counts[card.make] || 0) + 1;
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([name]) => ({ id: name, label: name, test: (c) => c.make === name }));
}

export function create({ bar, pills, onChange }) {
  const picked = new Map(GROUPS.map((g) => [g.id, new Set()]));
  const options = new Map();
  const menus = new Map();
  let pool = [];
  let base = () => true;
  let open = null;

  const optionsFor = (id) => options.get(id) || [];
  const testFor = (groupId, value) =>
    optionsFor(groupId).find((o) => o.id === value)?.test || (() => false);

  /* Every group except one. This is what a faceted count is measured against:
     the count beside Ford is what Ford would give you alongside the year and
     mileage already chosen, but not alongside the other makes - otherwise
     ticking a second make could only ever show zero. */
  function passes(card, skip) {
    for (const group of GROUPS) {
      if (group.id === skip) continue;
      const on = picked.get(group.id);
      if (!on.size) continue;
      if (![...on].some((value) => testFor(group.id, value)(card))) return false;
    }
    return true;
  }

  const matches = (card) => passes(card, null);

  function countFor(group, option) {
    let n = 0;
    for (const card of pool) {
      if (base(card) && passes(card, group.id) && option.test(card)) n += 1;
    }
    return n;
  }

  function close(focusButton) {
    if (!open) return;
    const { wrap, button, menu } = open;
    menu.hidden = true;
    wrap.classList.remove('on');
    button.setAttribute('aria-expanded', 'false');
    if (focusButton) button.focus();
    open = null;
  }

  /* Where the menu can hang without leaving the window. Measured rather than
     anchored in CSS, because the bar wraps: "the last two open to the right"
     is a rule about source order, and once Mileage is the rightmost button on
     a 390px screen it is the wrong rule - it put 9px of the menu off the edge
     and gave the whole page a horizontal scrollbar. Never pushed so far left
     that it leaves the other edge to save this one. */
  const EDGE = 12;

  function place({ menu, wrap }) {
    menu.style.left = '0px';
    const over = menu.getBoundingClientRect().right - (window.innerWidth - EDGE);
    if (over <= 0) return;
    const room = wrap.getBoundingClientRect().left - EDGE;
    menu.style.left = `${-Math.max(0, Math.min(over, room))}px`;
  }

  function openMenu(entry) {
    if (open && open.wrap === entry.wrap) { close(true); return; }
    close(false);
    entry.menu.hidden = false;
    entry.wrap.classList.add('on');
    entry.button.setAttribute('aria-expanded', 'true');
    open = entry;
    place(entry);
    entry.menu.querySelector('input:not([disabled])')?.focus();
  }

  function toggle(groupId, value, on) {
    const set = picked.get(groupId);
    on ? set.add(value) : set.delete(value);
    onChange();
    refresh();
  }

  function buildMenu(group) {
    const wrap = el('div', 'facet');
    const button = el('button', 'facet-btn');
    button.type = 'button';
    button.id = group.elementId;
    button.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-haspopup', 'true');
    button.append(el('span', 'facet-name', group.label),
                  el('b', 'facet-n'),
                  el('span', 'facet-caret', '▾'));

    const menu = el('div', 'facet-menu');
    menu.hidden = true;
    menu.setAttribute('role', 'group');
    menu.setAttribute('aria-label', group.label);
    button.setAttribute('aria-controls', `${group.elementId}-menu`);
    menu.id = `${group.elementId}-menu`;

    const entry = { wrap, button, menu, group };
    button.addEventListener('click', () => openMenu(entry));
    wrap.append(button, menu);
    menus.set(group.id, entry);
    return wrap;
  }

  /* Redrawn wholesale on every change rather than patched. Twelve to twenty
     checkboxes is nothing, and a diff here would be a place for a count to go
     stale without anyone noticing. */
  function paintMenu(group) {
    const { menu, button } = menus.get(group.id);
    const on = picked.get(group.id);
    menu.replaceChildren(...optionsFor(group.id).map((option) => {
      const n = countFor(group, option);
      const row = el('label', 'facet-row');
      const box = el('input');
      box.type = 'checkbox';
      /* Chrome restores the checked state of a form control across a reload
         and fires `change` when it does, which called toggle() and left the
         browser holding two Country filters nobody had pressed. The truth is
         `picked`; the boxes are a drawing of it. */
      box.autocomplete = 'off';
      box.checked = on.has(option.id);
      /* A dead end stays visible and goes flat. Removing it would make the
         list shuffle under the cursor every time a box is ticked. */
      box.disabled = n === 0 && !box.checked;
      row.classList.toggle('empty', box.disabled);
      box.addEventListener('change', () => toggle(group.id, option.id, box.checked));
      row.append(box, el('span', 'facet-label', option.label), el('b', null, String(n)));
      return row;
    }));
    if (!optionsFor(group.id).length) {
      menu.append(el('p', 'facet-none', 'Nothing to filter by here.'));
    }
    button.classList.toggle('on', on.size > 0);
    button.querySelector('.facet-n').textContent = on.size ? String(on.size) : '';
  }

  function paintPills() {
    const rows = [];
    for (const group of GROUPS) {
      for (const value of picked.get(group.id)) {
        const option = optionsFor(group.id).find((o) => o.id === value);
        const pill = el('button', 'pill');
        pill.type = 'button';
        pill.setAttribute('aria-label',
                          `Remove filter ${group.label}: ${option ? option.label : value}`);
        pill.append(el('span', 'pill-group', group.label),
                    el('span', 'pill-value', option ? option.label : value),
                    el('span', 'pill-x', '×'));
        pill.addEventListener('click', () => toggle(group.id, value, false));
        rows.push(pill);
      }
    }
    if (rows.length) {
      const clear = el('button', 'pill-clear', 'Clear all');
      clear.type = 'button';
      clear.addEventListener('click', () => { reset(); onChange(); refresh(); });
      rows.push(clear);
    }
    pills.replaceChildren(...rows);
    pills.hidden = rows.length === 0;
  }

  function refresh() {
    GROUPS.forEach(paintMenu);
    paintPills();
    /* A repaint can change the menu's width - a count going from 2 digits to
       3 - so an open one is measured again rather than left where it was. */
    if (open) place(open);
  }

  function reset() {
    picked.forEach((set) => set.clear());
  }

  /* A make that vanishes with the corpus underneath it - never in this build,
     but the pool is an argument - must not leave a pill nothing can remove. */
  function prune() {
    for (const group of GROUPS) {
      const valid = new Set(optionsFor(group.id).map((o) => o.id));
      for (const value of [...picked.get(group.id)]) {
        if (!valid.has(value)) picked.get(group.id).delete(value);
      }
    }
  }

  function setCards(cards) {
    pool = cards;
    options.set('make', makeOptions(cards));
    for (const group of GROUPS) {
      if (group.options) options.set(group.id, group.options());
    }
    prune();
  }

  bar.replaceChildren(...GROUPS.map(buildMenu));

  document.addEventListener('click', (e) => {
    if (open && !open.wrap.contains(e.target)) close(false);
  });
  window.addEventListener('resize', () => { if (open) place(open); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && open) { e.stopPropagation(); close(true); }
  });

  return {
    setCards,
    matches,
    refresh,
    setBase(fn) { base = fn; },
  };
}
