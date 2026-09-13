# The landing page's scroll story

Date: 2026-09-13
Status: approved, implementing

## Why

The landing page sells the product in three static sections: a hotspot "playground" with
three hardcoded findings, a three-step *how it works* strip, and an honesty note. None of
them show the thing that is actually interesting — that sixteen independent vision calls
read sixteen photographs, disagree, get merged, get cross-checked against an OCR read of
the dashboard, and only then become a number.

A visitor cannot tell from that page whether this is a wrapper. The scroll story is the
argument that it is not, made out of a run that actually happened.

## Decisions taken

| Question | Decision |
|---|---|
| Where do the verdicts and the price come from? | A real pipeline run on `demo/tr_clean`, frozen to JSON and committed. Nothing on the page is authored prose. |
| What survives the overhaul? | The hero, Kip and the pixel world stay. The playground and the three-step strip are replaced. |
| Scroll mechanic | Pinned stage, layered photo deck. Cards lift off a stack, are read, and file into a mosaic. |
| Dark or light? | Dark for the reading, light for the number. `run.css` already made this rule; the story obeys it. |
| No-JS / reduced motion | The photos and verdicts are written into `landing.html` as static markup at build time. JS upgrades it. The fallback is the source. |

## 1. The seven acts

One `<section id="story">`. Inside it a `position: sticky` stage, driven by scroll
progress through a tall track.

| Act | Stage | Track height |
|---|---|---|
| 0 · Handoff | Paper darkens to `--stage`. 21 photos scatter in, collapse into a fanned deck. | 90vh |
| 1 · Is this even a truck? | Detection boxes strobe the deck; one locks as subject, the rest dim. Counter ticks photos / vehicles / views. | 110vh |
| 2 · Reading the photos | Six beats. Card lifts, scales, its subject box draws, the verdict writes in beside it, the card files into the mosaic. | 6 × 75vh |
| 3 · One truck, not sixteen findings | Mosaic reflows; duplicate findings converge into one badged *also seen in N photos*. | 80vh |
| 4 · Cross-checks | The dashboard frame returns, OCR digits scrub to the read, reconciled against the seller's figure. | 80vh |
| 5 · The number | Lights up, back to paper. Mosaic compresses to a grid, the two bands draw apart, the figure counts up. | 110vh |
| 6 · What it won't say | `cannot_tell`, the re-ask, the CTA. | 70vh |

Acts 3 and 4 are the parts that make this not a wrapper, so they get a viewport each.

## 2. The build script

`scripts/freeze_story.py` turns one real appraisal into every asset the page needs.

```
Appraisal JSON ─┐
                ├─► app/web/assets/story/{000..005}.jpg   six hero photos, resized
demo/tr_clean/ ─┘   app/web/assets/story/story.json       trimmed to what the page reads
                    landing.html  (region between <!-- story:start --> / <!-- story:end -->)
```

Hero photo selection is deterministic: one per view, photos carrying issues first, capped
at six.

Writing the static markup at build time is what makes the no-JS path and the
reduced-motion path the same path, and it makes "every verdict on the page is in the
JSON" checkable by reading the HTML.

## 3. Module layout

```
app/web/js/story/scroll.js    one shared IntersectionObserver + rAF reader → progress 0‥1
app/web/js/story/deck.js      assemble / lift / draw the box / file into the mosaic
app/web/js/story/verdict.js   the per-photo card, rendered from story.json
app/web/js/story/reveal.js    the merge, the reconcile, the band, the count-up
app/web/js/story/index.js     reads the JSON, wires acts to progress
app/web/styles/story.css
```

No new dependency. `js/dom.js` and the vendored `vendor/motion.min.js` supply `reduced()`,
`tween()` and `animate()`, so the page still never reaches the network at run time.

`landing.js` becomes a module so it can import them.

## 4. Colour

`landing.css` carries its own `:root`, mirrored against `tokens.css` by `BrandPalette`.
The dark stage needs `--stage`, `--stage-text` and the four severity hues, which exist
only in `tokens.css`. `story.css` scopes them to `.story` rather than opening a second
`:root`, and `BrandPalette` is extended to cover them. Three copies and no test is how one
product becomes two.

## 5. Honesty invariants this page must hold

Everything in CLAUDE.md applies here, and three of them bite hardest on a front page:

- **The measured 80.3% belongs to the comparable-asking band.** It may not label the
  photo-adjusted one.
- **Asking prices, not sale prices**, said out loud.
- **No authored verdict prose.** If a sentence about the truck is on screen, it is in
  `story.json`. `tests/test_story.py` asserts this in both directions.

The section is stamped with what it is: a real run on a real listing, frozen on a date,
and the visitor's own photos run live.

## 6. Degradation

Under 820px, and under `prefers-reduced-motion: reduce`, the stage unpins and the acts
become a plain vertical sequence — every photo and verdict visible, no pinning, no scrub.
That is the build-time markup unchanged, so it is a cheaper path rather than a second
implementation.

## 7. Tests

`tests/test_landing.py` loses its `data-inspect` × 6 assertion with the playground.

`tests/test_story.py`:

- every `/static/` reference in the story region resolves
- every verdict sentence in the markup appears verbatim in `story.json`
- the price figures in the markup match `story.json`
- the region is exactly what `freeze_story.py` would regenerate (no hand drift)
- the measured coverage figure appears only next to the baseline band
- `prefers-reduced-motion: reduce` unpins the stage in `story.css`
- the story JS imports only modules that exist

`BrandPalette` grows to cover the stage and severity colours.
