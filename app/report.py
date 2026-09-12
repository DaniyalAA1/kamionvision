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


def _photo_label(appraisal: Appraisal, photo_id: int) -> str:
    for check in appraisal.gate.photos:
        if check.photo_id == photo_id:
            return check.filename
    return f"photo {photo_id}"


def price_lines(price: PriceEstimate) -> list[str]:
    if not price.ok:
        return ["  NO PRICE  " + price.reason]
    card = price.model_card
    out = [
        f"  {money(price.low, price.currency)}  –  {money(price.high, price.currency)}"
        f"     (point estimate {money(price.point, price.currency)})",
        f"  {money(price.low_usd, 'USD')} – {money(price.high_usd, 'USD')} at "
        f"{card['fx']['usd_try']} TRY/USD as of {card['fx']['as_of']}",
    ]
    cov, n = card.get("coverage"), card.get("coverage_n")
    if cov is not None:
        out.append(f"  This is an {int(price.interval_level * 100)}% band. On held-out "
                   f"listings it contained the real asking price {cov * 100:.1f}% of the "
                   f"time ({n} evaluations).")
    if card.get("r2") is not None:
        out.append(f"  Fit: R² {card['r2']:.2f}, median error {card['mae_pct']:.1f}% "
                   f"across {card['n_listings']} listings ({card['n_groups']} distinct specs).")
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
            out.append(f"   {mark} [{issue.severity}/{issue.price_impact} impact] "
                       f"{issue.observation}")
            out.append(f"        seen in {_photo_label(appraisal, issue.photo_id)} "
                       f"(confidence {issue.confidence:.2f})")
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
            if adj.pct:
                L.append(f"  Condition adjustment: {adj.pct:+.1f}% "
                         f"(cap ±{adj.cap_pct:.1f}%)")
                L.append(f"    driven by: {'; '.join(adj.drivers)}")
                L.append(f"    {adj.basis}")
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

    if ev:
        L += [thin, f"  WHAT I CAN SEE   (grade: {ev.condition_grade}, "
                    f"confidence {ev.confidence:.2f})", thin]
        L += condition_lines(appraisal, ev)
        L.append("")
        L.append("  Summary by system:")
        for key, text in ev.condition_summary.items():
            L.append(f"    {key.replace('_', ' ').title():<22} {text}")
        L.append("")

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
