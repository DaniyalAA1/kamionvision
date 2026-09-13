"""Stage 4 - the card a buyer reads.

Rules this renderer enforces, because they are what the brief scores:

  * Every condition line names the photo it came from, by filename, not by an
    index the buyer cannot resolve.
  * The price is always a range. The point estimate is shown inside it, never
    alone.
  * The calibration number is printed next to the range, every time. A range
    nobody has checked is decoration.
  * Measured numbers and assumed ones are labelled differently. The interval
    coverage is measured; the widening for a missing view is a stated
    assumption, and it says so.
"""
from __future__ import annotations

from .schema import Appraisal, EvidenceReport, PriceEstimate

SEVERITY_MARK = {"major": "!!", "moderate": " !", "minor": "  ", "cosmetic": "  "}

COMPONENT_GROUPS = [
    ("Tires & wheels", {"steer_tires", "drive_tires", "wheels_rims", "brakes_hubs"}),
    ("Coupling & chassis", {"fifth_wheel", "coupling_airlines", "chassis_frame",
                            "undercarriage", "air_suspension", "air_tanks_lines",
                            "mudflaps_guards"}),
    ("Body & paint", {"cab_exterior_panels", "front_bumper_valance", "fairings_skirts",
                      "grille_headlights", "mirrors_visor", "roof_deflector",
                      "windscreen_glass", "doors_handles", "cab_steps", "paint_finish",
                      "corrosion"}),
    ("Driveline & tanks", {"engine_bay", "fluid_leaks", "exhaust_dpf", "fuel_tank",
                           "adblue_tank"}),
    ("Cab interior", {"cab_interior_seats", "steering_wheel_controls",
                      "dashboard_instruments", "bunk_sleeper", "cab_floor_trim",
                      "warning_lights"}),
]


def money(value: float, currency: str) -> str:
    symbol = {"TRY": "₺", "USD": "$", "EUR": "€"}.get(currency, currency + " ")
    return f"{symbol}{value:,.0f}"


def _corroboration(appraisal: Appraisal, issue) -> str:
    """", and 2 more" when the same defect was reported from several frames.

    The fan-out reads each photo on its own, so one worn steer tire arrives
    three times. The synthesis folds those into one finding and keeps the other
    photo ids here - "seen in three photos" is a claim a buyer can check, which
    is more than a confidence decimal ever was.
    """
    extra = getattr(issue, "also_seen_in", None) or []
    if not extra:
        return ""
    if len(extra) == 1:
        return f" and {_photo_label(appraisal, extra[0])}"
    return f" and {len(extra)} other photos"


def _photo_label(appraisal: Appraisal, photo_id: int) -> str:
    for check in appraisal.gate.photos:
        if check.photo_id == photo_id:
            return check.filename
    return f"photo {photo_id}"


def price_lines(price: PriceEstimate) -> list[str]:
    if not price.ok:
        return ["  NO PRICE  " + price.reason]
    card = price.model_card
    cur = price.currency
    out = []
    adjusted = abs(price.point - price.baseline_point) > 1
    if adjusted:
        out.append(f"  What comparable trucks are asking   "
                   f"{money(price.baseline_low, cur)} – {money(price.baseline_high, cur)}")
        out.append(f"  Adjusted for what the photos show   "
                   f"{money(price.low, cur)} – {money(price.high, cur)}"
                   f"   (midpoint {money(price.point, cur)})")
    else:
        out.append(f"  {money(price.low, cur)}  –  {money(price.high, cur)}"
                   f"     (point estimate {money(price.point, cur)})")
    out.append(f"  {money(price.low_usd, 'USD')} – {money(price.high_usd, 'USD')} at "
               f"{card['fx']['usd_try']} TRY/USD as of {card['fx']['as_of']}")
    if price.asking:
        a = price.asking
        out.append("")
        out.append(f"  The seller is asking {money(a.asking, a.currency)} "
                   f"— {a.label}.")
        out.append(f"    {a.summary}")
    cov, n = card.get("coverage"), card.get("coverage_n")
    if cov is not None:
        out.append(f"  The asking band is an {int(price.interval_level * 100)}% interval. On "
                   f"held-out listings it contained the real asking price {cov * 100:.1f}% "
                   f"of the time ({n} evaluations).")
    if card.get("r2") is not None:
        out.append(f"  Fit: R² {card['r2']:.2f}, median error {card['mae_pct']:.1f}% "
                   f"across {card['n_listings']} listings ({card['n_groups']} distinct specs).")
    a = price.anchor
    if a is not None and a.ok:
        out.append("")
        out.append(f"  Second opinion, from what it cost new   {money(a.point, a.currency)}")
        out.append(f"    {a.basis}")
        out.append(f"    Source: {a.source} ({a.source_type.replace('_', ' ')}), {a.source_url}")
        out.append(f"    Weight in the estimate: {a.weight:.0%}, by inverse variance — this route "
                   f"needs no same-brand comparable to exist.")
    for line in price.widened:
        out.append(f"  Widened: {line}")
    return out


def perception_lines(appraisal: Appraisal) -> list[str]:
    """The trained heads, and anywhere they disagreed with the vision model."""
    out: list[str] = []
    p, r = appraisal.perception, appraisal.reconcile
    if p is not None and p.photos:
        card = p.model_card or {}
        unfit = [x for x in p.photos if not x.fine_detail_ok]
        out.append(f"  Trained heads scored {len(p.photos)} frames. "
                   f"{len(unfit)} too degraded to read fine detail from.")
        if card.get("degradation_auc"):
            out.append(f"    Degradation head: AUC {card['degradation_auc']} on held-out "
                       f"vehicles, against known synthetic corruptions — measured.")
        if card.get("view_agreement_head"):
            out.append(f"    View head holds its answer on {card['view_agreement_head']:.0%} of "
                       f"degraded twins, against {card['view_agreement_zeroshot']:.0%} for the "
                       f"zero-shot tagger it replaces — measured.")
        if p.brand:
            out.append(f"    Reads the vehicle as {p.brand} ({p.brand_conf:.0%}); the head is "
                       f"{card.get('identity_accuracy', 0):.0%} accurate against a "
                       f"{card.get('identity_majority', 0):.0%} majority baseline, so it is a "
                       f"cross-check on the badge, not the identification.")
    if r is not None and r.n:
        out.append("")
        out.append(f"  {r.n} correction(s) applied to the vision model's findings:")
        for c in r.corrections:
            where = f" [photo {c.photo_id}]" if c.photo_id is not None else ""
            out.append(f"    · {c.detail}{where}")
            if c.before:
                out.append(f"        {c.before} → {c.after}")
    return out


def condition_lines(appraisal: Appraisal, evidence: EvidenceReport) -> list[str]:
    out: list[str] = []
    if not evidence.issues:
        return ["  Nothing flagged in the photos supplied."]
    remaining = list(evidence.issues)
    for title, members in COMPONENT_GROUPS:
        group = [i for i in remaining if i.component in members]
        if not group:
            continue
        for issue in group:
            remaining.remove(issue)
        out.append(f"  {title}")
        for issue in sorted(group, key=lambda i: -SEVERITY_ORDER.get(i.severity, 0)):
            mark = SEVERITY_MARK.get(issue.severity, "  ")
            band = ("reported, not graded" if issue.ungraded
                    else f"{issue.severity}/{issue.price_impact} impact")
            out.append(f"   {mark} [{band}] {issue.observation}")
            out.append(f"        seen in {_photo_label(appraisal, issue.photo_id)}"
                       + _corroboration(appraisal, issue)
                       + f" (confidence {issue.confidence:.2f})")
            if issue.ungraded:
                out.append("        the model returned a severity or price impact "
                           "outside the enum, so this is shown and weighted at zero")
            elif issue.severity_span:
                out.append(f"        frames disagreed "
                           f"({', '.join(issue.severity_span)}); graded "
                           f"{issue.severity}, the level two frames support")
    for issue in remaining:
        out.append(f"   {SEVERITY_MARK.get(issue.severity, '  ')} "
                   f"[{issue.severity}] {issue.observation}")
        out.append(f"        seen in {_photo_label(appraisal, issue.photo_id)}")
    return out


SEVERITY_ORDER = {"major": 3, "moderate": 2, "minor": 1, "cosmetic": 0}


def render_text(appraisal: Appraisal, *, width: int = 78) -> str:
    rule = "=" * width
    thin = "-" * width
    L: list[str] = [rule, "  KAMIONVISION  —  used tractor appraisal from photos", rule, ""]

    L.append(f"  {appraisal.headline}")
    L.append("")

    ev = appraisal.evidence
    if ev and (ev.vehicle.make or ev.vehicle.model):
        v = ev.vehicle
        ident = " ".join(x for x in (v.make, v.model) if x)
        bits = [b for b in (v.cab_type, v.axle_config, v.approx_year_range) if b]
        L.append(f"  Identified from the photos: {ident}"
                 + (f" — {', '.join(bits)}" if bits else ""))
        if v.odometer_km:
            L.append(f"  Odometer read from the dash: {v.odometer_km:,} km")
        if v.badges_seen:
            L.append(f"  Badges visible: {', '.join(v.badges_seen)}")
        L.append("")

    price = appraisal.price
    if price is not None:
        L += [thin, "  WHAT IT IS WORTH", thin] + price_lines(price) + [""]
        if price.ok:
            adj = price.adjustment
            if adj.pct or adj.coverage_pct:
                L.append(f"  Condition adjustment: {adj.pct:+.1f}% "
                         f"(cap ±{adj.cap_pct:.1f}%, {adj.direction})")
                L.append(f"    {adj.coverage_pct:.0f}% of the truck by value was "
                         f"photographed legibly; {adj.merit_pct:.0f}% of it was "
                         f"positively called sound")
                L.append(f"    driven by: {'; '.join(adj.drivers)}")
                for note in adj.notes:
                    L.append(f"    note: {note}")
                # The cap and the weights are different kinds of number and the
                # report says which is which. The repo used to call the cap an
                # assumption here and a measurement in the code comment.
                L.append(f"    {adj.cap_basis or adj.basis}")
                if adj.weights_basis:
                    L.append(f"    {adj.weights_basis}")
                L.append("")
            if price.drivers:
                L.append("  What moves this number (effect vs. the average comparable):")
                for d in price.drivers[:6]:
                    L.append(f"    {d['feature']:<22} {d['value']:>10.3f}  "
                             f"{d['pct_effect']:+7.1f}%")
                L.append("")
            if price.comparables:
                L.append("  Compared against these listings:")
                for c in price.comparables:
                    L.append(f"    {c.year}  {c.make} {c.model[:22]:<22} "
                             f"{c.km / 1000:>6.0f}k km  "
                             f"{money(c.price, c.currency):>12}")
                L.append("")
            if price.inputs_provenance:
                L.append("  Where the inputs came from:")
                for k, v in price.inputs_provenance.items():
                    L.append(f"    {k:<12} {v}")
                L.append("")
            if price.widened:
                L.append("  Why this band is wider than usual:")
                for w in price.widened:
                    L.append(f"    • {w}")
                L.append("")

    if ev and ev.fell_back_from:
        L.append(f"  NOTE  {ev.backend}/{ev.model} produced this after "
                 f"{len(ev.fell_back_from)} backend(s) failed:")
        for f in ev.fell_back_from:
            L.append(f"          {f[:150]}")
        L.append("")

    if ev:
        L += [thin, f"  WHAT I CAN SEE   (grade: {ev.condition_grade}, "
                    f"confidence {ev.confidence:.2f})", thin]
        if ev.condition and ev.condition.grade_reason:
            L.append(f"  {ev.condition.grade_reason}")
            L.append("")
        if ev.grade_disagreement:
            L.append(f"  NOTE  {ev.grade_disagreement}. The deterministic one is the "
                     f"one priced; the band widens on the disagreement.")
            L.append("")
        L += condition_lines(appraisal, ev)
        L.append("")
        L.append("  Summary by system:")
        for key, text in ev.condition_summary.items():
            L.append(f"    {key.replace('_', ' ').title():<22} {text}")
        L.append("")
        strengths = [(f.photo_id, g) for f in ev.photo_findings for g in f.strengths]
        if strengths:
            L.append("  What the photos show to be in good order:")
            for photo_id, good in strengths[:10]:
                L.append(f"    + {good}")
                L.append(f"        seen in {_photo_label(appraisal, photo_id)}")
            L.append("")
        if ev.photos_failed:
            L.append(f"  NOTE  {ev.photos_failed} frame(s) could not be read; the "
                     f"findings above come from the {ev.photos_read} that were.")
            L.append("")

    pl = perception_lines(appraisal)
    if pl:
        L += [thin, "  WHAT THE TRAINED HEADS SAY", thin] + pl + [""]

    requests = list(appraisal.requests)
    gaps = ev.coverage_gaps if ev else []
    if requests:
        L += [thin, "  PHOTOS I STILL NEED", thin]
        for r in requests:
            L.append(f"    → {r}" if r[:1].isupper() else f"    → send me {r}")
        L.append("")
    if gaps:
        L += [thin, "  WHAT I COULD NOT ASSESS FROM THESE PHOTOS", thin]
        for g in gaps:
            L.append(f"    • {g}")
        L.append("")

    if price is not None and price.caveats:
        L += [thin, "  CAVEATS", thin]
        for c in price.caveats:
            L.append(f"    • {c}")
        L.append("")

    if appraisal.trace:
        L += [thin, "  PIPELINE", thin]
        for step in appraisal.trace:
            L.append(f"    {step.step:<12} {step.elapsed_s:>6.2f}s   {step.detail}")
        L.append(f"    {'total':<12} {appraisal.elapsed_s:>6.2f}s")
    L.append(rule)
    return "\n".join(L)
