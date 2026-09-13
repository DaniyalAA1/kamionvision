# Whole-vehicle vs component: 90 hand-labelled frames

## Why this exists

Nothing in this repository had ever measured whether the view classifier is
*correct*. The number that gets quoted for it - 0.758 - is **twin stability**,
not accuracy, and the trained view head in `app/perception/` is fitted on CLIP's
own zero-shot output, so it inherits whatever the teacher gets wrong. There was
no ground truth to check either against.

The distinction these labels carry is deliberately coarse: **is the whole
tractor in this frame, or one part of it?** That is the question the pipeline
actually acts on. Confusing `exterior_front` with `exterior_front_34` is
harmless - both are whole-vehicle views and their question banks are nearly the
same. Confusing a dashboard with `exterior_front` is not:

- the frame is handed the exterior checklist ("grille, bumper and valance...
  headlamp clouding... windscreen chips") and asked it about a steering wheel
- it counts toward `has_whole_vehicle`, so the gate believes the truck was
  photographed whole when it never was
- it is exempt from every part-view rule in `app/subject.py`, so a truck seen
  through the windscreen can be boxed and cropped as "the vehicle being sold"

## What was measured

| | accuracy | component close-ups miscalled whole |
|---|---|---|
| one template per class (before), dev / held-out | 83.3% / 86.7% | 18.9% / 15.8% |
| per-class ensemble, max-pooled (now) | 91.7% / **96.7%** | 10.8% / **0.0%** |

Over all 90: 83.3% -> 93.3% accuracy, 18.9% -> 7.1% on the dangerous error.

A margin requirement on top - "whole vehicle has to be earned", p >= 0.80 -
scored **better on dev (95.0%) and worse on held-out (93.3%)** while buying
nothing on the dangerous error, which was already 0. It was fitting 60 frames
and it was dropped. The dev/test split existed precisely to catch that.

## Provenance, stated plainly

Labelled by Claude Opus 5 in a single session, by eye, from 260px contact-sheet
thumbnails. **One rater, not a panel**, so there is no inter-rater agreement
figure to quote and none should be invented. The sample is stratified by the
*existing* CLIP tag and deliberately over-weights the exterior classes, so the
marginal class frequencies here are not the corpus's - only the conditional
error rates are meaningful.

Borderline calls were resolved by one rule, applied consistently: if the
vehicle's overall form is readable (cab together with chassis or wheels) it is
`whole`; if the frame is filled by one region of it, it is `part`. Frames 4 and
65 are both rear-of-cab shots and sit either side of that line.

`dev` is indices 0-59 and `test` is 60-89. The ensemble templates and the
pooling choice were selected on `dev` only.
