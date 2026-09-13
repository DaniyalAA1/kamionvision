/* Numbered pages under the wall.

   What this replaced was a "Show 24 more trucks" button, which has two
   problems on a corpus this size. It only goes one way - eight presses to
   reach the end and no way back short of reloading - and it hides how much
   there is: nine pages is a fact about the corpus, an ever-growing column is
   not. A page number is also somewhere you can be, which matters when the
   thing you are pointing at on stage is the fourth truck on page three.

   Ellipsis past seven pages so the row never wraps, and it always keeps the
   first, the last and the current page's neighbours - the four you actually
   press. */

import { el } from './dom.js';

export const PAGE_SIZE = 24;

export const pageCount = (n, size = PAGE_SIZE) => Math.max(1, Math.ceil(n / size));

/* The numbers to draw, with nulls where a gap goes. Kept separate from the
   drawing so it can be reasoned about as a list: [1, null, 4, 5, 6, null, 9]. */
export function pageList(page, pages, window = 1) {
  if (pages <= 7) return Array.from({ length: pages }, (_, i) => i + 1);
  const want = new Set([1, pages, page]);
  for (let d = 1; d <= window; d += 1) { want.add(page - d); want.add(page + d); }
  /* The ends keep their full run so the row does not change width as you walk
     through it - "1 2 3 … 9" and "1 … 4 5 6 … 9" are both seven cells wide. */
  if (page <= 3) [2, 3, 4].forEach((n) => want.add(n));
  if (page >= pages - 2) [pages - 1, pages - 2, pages - 3].forEach((n) => want.add(n));

  const kept = [...want].filter((n) => n >= 1 && n <= pages).sort((a, b) => a - b);
  const out = [];
  kept.forEach((n, i) => {
    if (i && n - kept[i - 1] > 1) out.push(null);
    out.push(n);
  });
  return out;
}

export function render(root, { page, pages, onGo }) {
  root.replaceChildren();
  root.hidden = pages <= 1;
  if (pages <= 1) return;

  const step = (label, to, enabled, aria) => {
    const b = el('button', 'page-step', label);
    b.type = 'button';
    b.disabled = !enabled;
    b.setAttribute('aria-label', aria);
    if (enabled) b.addEventListener('click', () => onGo(to));
    return b;
  };

  root.append(step('‹', page - 1, page > 1, 'Previous page'));
  for (const n of pageList(page, pages)) {
    if (n === null) {
      const gap = el('span', 'page-gap', '…');
      gap.setAttribute('aria-hidden', 'true');
      root.append(gap);
      continue;
    }
    const b = el('button', 'page-no', String(n));
    b.type = 'button';
    b.setAttribute('aria-label', `Page ${n} of ${pages}`);
    if (n === page) {
      b.classList.add('on');
      b.setAttribute('aria-current', 'page');
    }
    b.addEventListener('click', () => onGo(n));
    root.append(b);
  }
  root.append(step('›', page + 1, page < pages, 'Next page'));
}
