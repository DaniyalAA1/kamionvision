"""The vocabulary and the three prompts the evidence fan-out uses.

The closed `COMPONENTS` enum is unchanged and still load-bearing: a tool that
offers dent/scratch/paint is a car tool pointed at a truck, and "sensible to
someone who knows trucks" is a line in the brief's rubric.

What is new is `VIEW_QUESTIONS`. The single-call design asked one generic
"describe this truck" over fourteen photos and capped the answer at twelve
issues, so it skimmed - it had no budget to look at any one frame properly.
Here each photo gets its own call and its own checklist, and the checklist is
the difference between "tires look worn" and "outer shoulder of the near-side
steer is down to the wear bars while the centre ribs still carry depth, which
is an alignment fault rather than mileage".

What is newer still is everything under `SEVERITY_RUBRIC`. The checklist made
the model look properly; it did nothing to calibrate what it called what it
saw. `SEVERITIES` was four bare words with no threshold, no anchor and not one
worked example behind them, assigned by a model that sees one photograph and
has never been told how far the truck has run.

What is newest is that two of these prompts know WHICH truck they are looking
at, and a third deliberately does not. `app.modelspec` reads
`data/reference/models_tr.json` and hands the identity pass a closed model list
and a generation's visual markers, and the close-up pass that model's spec and
its known weak points; all of it degrades to nothing when the card is absent,
and a test pins that both prompts are then byte-identical to what they were
before the file existed. The weak points are the delicate half - a model told a
component is a known weak point will report it whether or not the frame shows
one, so they travel with the wording that makes them a prior about the model
rather than an observation about this truck. The third prompt is the badge
read, and it is told nothing: it is a second witness to make and model, and a
witness who has been handed the answer corroborates nothing.
"""
from __future__ import annotations

import json
import textwrap

from .. import modelspec

# Closed vocabulary. The model is told to use these ids and nothing else, so
# downstream severity weighting and the report's grouping are both stable.
COMPONENTS = [
    "steer_tires", "drive_tires", "wheels_rims", "brakes_hubs",
    "fifth_wheel", "coupling_airlines", "chassis_frame", "undercarriage",
    "air_suspension", "air_tanks_lines", "mudflaps_guards",
    "cab_exterior_panels", "front_bumper_valance", "fairings_skirts",
    "grille_headlights", "mirrors_visor", "roof_deflector", "windscreen_glass",
    "doors_handles", "cab_steps", "fuel_tank", "adblue_tank", "exhaust_dpf",
    "engine_bay", "fluid_leaks", "paint_finish", "corrosion",
    "cab_interior_seats", "steering_wheel_controls", "dashboard_instruments",
    "bunk_sleeper", "cab_floor_trim", "warning_lights",
]

SUMMARY_KEYS = ["tires", "wheels_brakes", "fifth_wheel_coupling", "chassis_corrosion",
                "body_paint", "cab_interior", "engine_driveline", "glass_lights"]

SEVERITIES = ["cosmetic", "minor", "moderate", "major"]
IMPACTS = ["none", "low", "medium", "high"]
GRADES = ["excellent", "good", "fair", "poor"]

# The absolute half of a finding. Every one of these is answerable from a single
# photograph without knowing what else is on the truck, which is exactly what
# `SEVERITIES` is not: a severity is a comparison, and the close-up call has
# nothing to compare against. So the close-up reports both - an ordinal it is
# told is provisional, and a magnitude that survives whatever the calibration
# pass does to the ordinal.
EXTENTS = ["spot", "local", "widespread", "whole_component"]
STATES = ["as_new", "worn_in_service", "end_of_life", "failed"]
BLOCKS_USE = ["no", "maybe", "yes", "cannot_tell"]

SYSTEM = """\
You are a heavy-vehicle appraiser for a Turkish freight marketplace. You value \
second-hand tractor units (çekici) - Ford Trucks F-MAX and F-LINE, Mercedes-Benz \
Actros, MAN TGX/TGS, Scania R/S, DAF XF, Volvo FH, Iveco S-Way, Renault T, and \
North American conventionals such as Freightliner Cascadia, Kenworth, Peterbilt.

You speak about trucks the way a fleet buyer does. Tires are steer, drive or tag, \
and you talk about tread depth, shoulder and centre wear, cupping, feathering, \
sidewall cracking, mismatched brands and retreads - not "the wheels look old". \
You know what a worn fifth wheel plate, a dry coupling, corroded frame rails, a \
sagging air bag, a cracked AdBlue tank, a delaminating roof deflector and a \
collapsed seat bolster look like.

You know the good version just as precisely: a fifth wheel plate carrying an \
even grease film with no step at the throat; frame rails with an even surface \
bloom and the paint still attached; tread standing well proud of the wear bars \
and matched across the axle; a seat that still holds its shape; a dry engine bay \
under ordinary road film; panel gaps that run parallel the length of the truck. \
Naming those is half the job, and it is the half that makes a buyer believe the \
other half.

Most of the trucks you are shown are working vehicles in ordinary condition for \
their age and distance. That is the base rate and your answers should reflect \
it: on a typical set most components are as they should be, a few are worn in \
proportion to the odometer, and one or two are genuinely worth money. A truck \
where everything is a problem is rare. A report that says so about an ordinary \
truck is wrong in the direction that costs a seller real money and costs this \
marketplace its credibility.

You are rigorous about what a photo can and cannot show. You never infer \
mechanical condition from a clean exterior, never report a component that is not \
visible in the photo you were given, and you say so plainly when something is not \
assessable. You are equally rigorous in the other direction: you do not treat \
wear as damage, you do not treat dirt as decay, and you do not treat a \
consumable that has done its job as a fault. Your notes go to a buyer who is \
deciding whether to drive six hours to see this truck - and to a seller whose \
truck is worth what you say it is."""


# --- the severity rubric ---------------------------------------------------
# `SEVERITIES` is four bare words, and until now nothing in this repo has told
# the model what any of them means. The weights they feed run 147x from end to
# end, so "moderate" against "major" on one worn tire moves the band further
# than any other single call the vision pass makes.
#
# Two rules shape everything below. Every anchor is a visually checkable STATE
# ("tread level with the wear bars"), never a measurement ("below 1.6 mm"): a
# photograph cannot support a millimetre, and a state claim survives the JPEG
# crush the degraded twins put it through in a way a measurement does not. And
# no anchor is written in money. The levels are anchored on repair EFFORT - a
# workshop morning, a component replacement - so that teaching the model what a
# severity means never puts a currency figure in front of it, and "the VLM
# never sees or emits a price" stays whole. `config.REPAIR_BANDS` is the lira
# reading of the same four words, it is for the README and the panel card, and
# a test here asserts it never reaches a prompt.
#
# None of this is injected into a prompt yet. The wiring lands with the
# close-up rebuild, after the baseline run against the current prompts.

SEVERITY_RUBRIC = """\
Severity is not how bad a thing looks. It is what it costs the next owner.
Work down this list and stop at the first level that fits.

major     The truck cannot be worked, or cannot be worked legally, until money
          is spent. Structural damage, anything that would fail a roadworthiness
          inspection, a safety part that is cracked or distorted, a leak that
          will not reach the next service. Replacing it is a component job, not
          a consumable: a fifth wheel, a frame repair, a cab panel, an engine or
          driveline part. A buyer walks away or re-prices the whole truck.

moderate  A real job, but a booked one, not an emergency. The truck works today
          and will keep working for weeks. A consumable at or past the end of
          its life; a leak that is staining but not running; corrosion that has
          lifted the paint and started to scale; a panel that needs replacing
          rather than polishing. A buyer subtracts the job and still buys.

minor     Worn, and worn ahead of the rest of this truck - but the buyer lives
          with it or fixes it at the next scheduled service out of petty cash.
          A consumable at half life on a truck that is otherwise newer. A small
          dent. Surface rust that has not lifted. A buyer mentions it and moves
          on.

cosmetic  Appearance only. Nothing on this truck works differently because of
          it. Scuffs, stone chips, kerbed rims, faded plastic, a torn mudflap,
          dirt, a scratched step, a neatly resprayed panel that is otherwise
          sound. A buyer does not raise it.

Before you assign any level, ask the question the other way round: is this
component worse than a truck of this age and distance would normally be? A
consumable that has done its job and is halfway through its life is not a
finding at any level. Name it in "strengths" as on schedule, or do not name it.
Wear that is exactly what the distance predicts is the most common thing you
will see, and it is not a defect.

When you are between two levels, take the LOWER one and put what would have to
be true for the higher one into the observation. Every level up multiplies what
this finding takes off the seller's asking figure, and "major" tells a buyer
this truck needs money spent before it earns. Reserve it for something you
could point at and name out loud standing next to the truck. A "moderate" you
are certain of is worth more to a buyer than a "major" you argued yourself
into."""


# --- the per-view severity anchors -----------------------------------------
# Written once per component family and composed per view, so a close-up call
# is shown the levels for the parts that are actually in its frame and nothing
# else. The families partition `COMPONENTS` exactly - a test asserts it - so no
# component can be reported without a rubric that covers it, and the ids in
# each block's first line are the enum subset that view should be reaching for.

_FAMILY_ANCHORS = {
    "tires": (
        "Tires and rims",
        ("steer_tires", "drive_tires", "wheels_rims", "brakes_hubs"),
        """\
Judge tread against the wear bars - the raised bridges inside the grooves -
because that is what a photograph can actually support.
  cosmetic  kerbed rim, missing balance weight, dirt, a scuffed sidewall with no
            crack in it. The tire itself is fine.
  minor     tread clearly used but the grooves still stand well proud of the
            wear bars across the full width; wear even; different brands across
            an axle with both tires in the same state.
  moderate  tread approaching the wear bars anywhere on the contact patch, OR a
            shoulder visibly lower than the centre, OR cupping or feathering.
            This is a booked replacement and it points at alignment or
            suspension - say which you mean.
  major     tread level with or below the wear bars, cord or belt showing, a
            sidewall bulge, a cut through to the casing, a flat spot. Nobody
            should drive on it.
Do not infer remaining tread from how dirty a tire is. Do not call a tire worn
because its tread pattern is aggressive. A retread on a drive axle is ordinary
commercial practice, not a defect.""",
    ),
    "fifth_wheel": (
        "The coupling",
        ("fifth_wheel", "coupling_airlines"),
        """\
  cosmetic  surface dirt, old grease, paint worn off the ramps.
  minor     the plate is dry where it should carry a grease film; light circular
            scoring with no visible step.
  moderate  deep circular grooving, a visible step at the jaw throat, a mounting
            bolt obviously missing or backed out, fretting rust around the
            mounting plate.
  major     a cracked or distorted plate, elongated mounting holes, a jaw or
            handle sitting where it cannot lock, impact damage through the
            throat. This is the coupling. It is a safety part and it is the one
            component where "probably fine" is not an answer.""",
    ),
    "chassis": (
        "Chassis and undercarriage",
        ("chassis_frame", "undercarriage", "air_suspension", "air_tanks_lines",
         "corrosion", "mudflaps_guards"),
        """\
  cosmetic  surface bloom on a crossmember or bracket, chipped chassis paint,
            rust on a mudflap bracket or a step.
  minor     even surface rust along a frame rail with the paint intact around
            it; rust staining running from a fastener.
  moderate  scaling rust - layered, lifting, paint coming away - or visible
            pitting, anywhere on the rails or crossmembers. A shock absorber wet
            down its body. An air bag perished or chafing.
  major     perforation, lost section, a crack, a weld or fish-plate repair, a
            rail that is visibly deformed.
A working Turkish tractor unit lives outdoors. Even surface rust under a
six-year-old truck is the expected state, not a finding.""",
    ),
    "body": (
        "Paint and panels",
        ("cab_exterior_panels", "front_bumper_valance", "fairings_skirts",
         "roof_deflector", "doors_handles", "cab_steps", "paint_finish"),
        """\
  cosmetic  stone chips, swirl marks, fade, a scuff, a scratch that has not gone
            through to primer, a panel resprayed neatly in a slightly different
            shade.
  minor     a dent you could push out or live with, a scratch through to primer
            or metal, a cracked fairing or skirt.
  moderate  a panel that needs replacing rather than repairing, a missing
            section of fairing, a cracked deflector, paint peeling in sheets, a
            door no longer sitting in its gap.
  major     accident damage - a folded panel, a door or A-pillar out of line, a
            repair that has left the shut lines wrong.
Paint and trim alone are never major. A truck is not unroadworthy because it is
scratched.""",
    ),
    "glass_lights": (
        "Glass, lamps and mirrors",
        ("windscreen_glass", "grille_headlights", "mirrors_visor"),
        """\
  cosmetic  a hazed or yellowed lens, a stone chip outside the swept area, a
            scuffed mirror back, wiper smear that wipes off.
  minor     a chip inside the swept area, a lens cracked but dry behind it, a
            mirror arm that has been bent back into line.
  moderate  water standing inside a lamp, a lamp unit that is clearly not
            lighting, a missing mirror glass, wiper arcs scored into the screen.
  major     a crack running across the driver's half of the swept area, a
            shattered or missing lamp unit, a screen that has gone.
This is the one part of the bodywork that fails an inspection, so judge it on
whether the truck can be driven at night and in rain, not on how it looks.""",
    ),
    "interior": (
        "The cab",
        ("cab_interior_seats", "steering_wheel_controls", "dashboard_instruments",
         "bunk_sleeper", "cab_floor_trim", "warning_lights"),
        """\
  cosmetic  dirt, litter, a stained mat, faded trim, a worn steering wheel rim
            on a truck that has earned it.
  minor     a torn seat cover, a cracked trim panel, a missing knob, worn pedal
            rubbers.
  moderate  a collapsed driver's bolster or a seat that has dropped, damp or
            mould staining, a cracked dash top, cut or taped wiring.
  major     a lit engine, brake, ABS/EBS or emissions lamp on a live cluster;
            standing water or active ingress; a dead cluster.
A cab worked in for half a million kilometres looks worked in. Wear in
proportion to the distance is "strengths", not a finding.""",
    ),
    "engine": (
        "The engine bay and what leaks into it",
        ("engine_bay", "fluid_leaks", "exhaust_dpf", "fuel_tank", "adblue_tank"),
        """\
  cosmetic  road film, dust, an old dry stain with no wet edge.
  minor     light oil misting at a joint, a weep with no drip and nothing fresh,
            a chafe mark that has not gone through.
  moderate  a wet leak with a run or a drip forming, a perished or swollen hose,
            a glazed or cracked belt, a corroded cooler core.
  major     coolant, fuel or oil pooling or running onto the chassis or the
            ground; a split charge pipe; a cut or spliced loom; oil standing in
            the charge pipework.
A steam-cleaned engine bay is not evidence of a leak. It is not evidence of no
leak either - say that in "cannot_tell".""",
    ),
}

# Which families each canonical view is shown. A view sees the parts that are
# in its frame: a rear shot is the coupling and the chassis, a damage close-up
# is a panel or rust and is judged as one.
VIEW_FAMILIES = {
    "exterior_front": ("body", "glass_lights"),
    "exterior_front_34": ("body", "tires"),
    "exterior_side": ("body", "chassis"),
    "exterior_rear": ("fifth_wheel", "chassis"),
    "interior_cab": ("interior",),
    "dashboard_odometer": ("interior",),
    "tire_wheel": ("tires",),
    "engine_bay": ("engine",),
    "chassis_undercarriage": ("chassis",),
    "fifth_wheel": ("fifth_wheel",),
    "damage_detail": ("body", "chassis"),
}

ANCHOR_COMPONENTS = {family: ids for family, (_, ids, _) in _FAMILY_ANCHORS.items()}


def view_components(view: str) -> tuple[str, ...]:
    """The component ids a photograph tagged `view` could actually show.

    The families partition `COMPONENTS`, so between them the views cover the
    whole enum - a test asserts both. This is what a model's known weak points
    are filtered through before they reach a close-up call: a tire close-up has
    no business being told about this model's AdBlue tank, because it cannot
    see one and a prior it cannot check is a prior it can only report on faith.
    """
    return tuple(c for f in VIEW_FAMILIES.get(view, ()) for c in ANCHOR_COMPONENTS[f])


def _anchor_block(family: str) -> str:
    title, ids, levels = _FAMILY_ANCHORS[family]
    head = textwrap.fill(f"{title} - report these as {', '.join(ids)}.", width=76)
    return f"{head}\n{levels}"


COMPONENT_ANCHORS = {
    view: "\n\n".join(_anchor_block(f) for f in families)
    for view, families in VIEW_FAMILIES.items()
}


# --- what this truck's distance already predicts ---------------------------
# The close-up call has never been told how far the truck has run, so it has
# been grading every consumable against a new one. These rows say what "on
# schedule" looks like at a distance, per family, so that half-worn tread at
# 600,000 km reads as maintenance rather than as a finding.
#
# STATED ASSUMPTIONS, domain-reasoned, not measured: the corpus carries no
# maintenance records. `config.KM_PER_YEAR_TR` is the measured part and it is
# measured on dealer stock offered for sale, which is not the population of
# trucks in service. Label both that way wherever they surface.

WEAR_BANDS = ((150_000, "low"), (400_000, "mid"), (800_000, "high"), (None, "very_high"))
WEAR_BAND_IDS = tuple(band for _, band in WEAR_BANDS)

_FAMILY_WEAR = {
    "tires": {
        "low": "still on its original tires, lightly and evenly worn, matched across "
               "each axle",
        "mid": "on its first or second set; a half-worn, evenly worn tread is on "
               "schedule",
        "high": "on its second or third set; different brands across axles are normal "
                "at this distance and are not by themselves a finding",
        "very_high": "on its third set or beyond; retreads on the drive axle are "
                     "ordinary commercial practice here",
    },
    "fifth_wheel": {
        "low": "a plate with an even grease film and no scoring you could catch a "
               "fingernail on",
        "mid": "an even grease film over light circular scoring; the jaws tight",
        "high": "visible circular wear in the plate and a jaw that has been adjusted or "
                "replaced once; this is maintenance, not damage",
        "very_high": "a plate that has been replaced or built up at least once",
    },
    "chassis": {
        "low": "paint holding on the rails, with at most a bloom of surface rust at the "
               "fasteners",
        "mid": "even surface rust along the rails and crossmembers with the paint still "
               "on them; shocks dry, air bags sitting square",
        "high": "surface rust over most of the underside and stone damage to the paint; "
                "a shock or an air bag replaced once is maintenance",
        "very_high": "rust over the whole underside and suspension parts that have been "
                     "changed; only lifting scale, pitting or a repaired section is a "
                     "finding",
    },
    "body": {
        "low": "original paint, matched panel to panel, stone chips on the leading "
               "edges",
        "mid": "stone chipping across the front, a scuff or two down the sides, "
               "fairings complete",
        "high": "chips, scuffs and at least one panel or fairing repaired or resprayed, "
                "with the shut lines still parallel",
        "very_high": "a truck patched up more than once; a mismatched panel and a "
                     "replaced fairing are expected, shut lines out of line are not",
    },
    "glass_lights": {
        "low": "clear lenses, an unchipped screen, mirror arms and glass undamaged",
        "mid": "stone chips in the screen outside the swept area and the first haze on "
               "the lamp lenses",
        "high": "hazed lenses, a repaired chip and wiper arcs in the screen",
        "very_high": "a replaced screen and clouded lamps; only a crack in the swept "
                     "area or water standing inside a lens is a finding",
    },
    "interior": {
        "low": "seat foam and bolsters holding their shape, trim and switches unmarked",
        "mid": "shine on the driver's bolster and the wheel rim, pedal rubbers starting "
               "to go",
        "high": "a polished wheel rim, worn pedal rubbers, a marked floor and a seat "
                "that has softened",
        "very_high": "a cab lived in for a decade; a seat cover, a re-trimmed wheel and "
                     "worn switchgear are on schedule, a collapsed bolster or a wet "
                     "floor is not",
    },
    "engine": {
        "low": "a dry bay under the road film, original hoses and clamps",
        "mid": "road film and dust, light misting at a joint or two, belts intact",
        "high": "staining around the joints and at least one hose or belt already "
                "replaced; a dry but stained bay is on schedule",
        "very_high": "hoses, pipework and belts replaced, and staining everywhere the "
                     "oil has ever been; only a wet leak with a run or a drip is a "
                     "finding",
    },
}

EXPECTED_WEAR = {
    view: {band: tuple(_FAMILY_WEAR[f][band] for f in families)
           for band in WEAR_BAND_IDS}
    for view, families in VIEW_FAMILIES.items()
}


def wear_band(km: int | None) -> str | None:
    """Which expected-wear row applies. None when the seller stated no distance."""
    if not km or km < 0:
        return None
    for ceiling, band in WEAR_BANDS:
        if ceiling is None or km < ceiling:
            return band
    return WEAR_BAND_IDS[-1]


# --- worked examples -------------------------------------------------------
# Six calls, and the pair that carries the block is 5 and 6: one component, one
# kind of photograph, two states, two answers. Two of the six are deliberately
# not findings at all, because the failure this rubric exists to fix is not a
# hallucinated defect - it is a real observation promoted a level to make it
# worth writing down.

WORKED_EXAMPLES = """\
Six calls, so that the levels mean the same thing twice:

1. "Near-side steer tire: tread even across the ribs and still standing well
   proud of the wear bars; brand matches the off-side."
   -> not an observation at all. This belongs in "strengths".

2. "Drive axle tires roughly half worn, evenly, on a truck showing 480,000 km."
   -> not an observation. A consumable at half life at that distance is on
   schedule. "strengths", named as on schedule.

3. "Outer shoulder of the near-side steer is down to the wear bars while the
   centre ribs still carry depth."
   -> moderate. It is a replacement AND it points at alignment, so both belong
   in the observation. It is not major: the truck is legal and drivable today.

4. "Sidewall of the near-side steer carries a bulge the size of a fist, below
   the shoulder."
   -> major. That is cord failure. Nobody drives on it.

5. "Surface rust, even and unlifted, along the visible length of the near-side
   frame rail on a 2018 truck."
   -> cosmetic, and "price_impact" none. That is what an eight-year-old working
   chassis looks like outdoors in this market.

6. "Rust on the near-side rail above the rear axle has lifted the paint and is
   scaling, with flakes standing off the metal."
   -> moderate. Same component, same kind of photograph, different state. The
   difference between 5 and 6 is the whole point of this list."""


# --- what to confirm sound -------------------------------------------------
# `strengths` is in the schema, is rendered on three surfaces, and until now
# was prompted for by nothing: 76 checklist items across eleven views, every
# one of them naming a failure mode. A model asked only what is wrong answers
# only what is wrong. These are the other half of `VIEW_QUESTIONS` - the same
# parts, asked whether they are sound - and an empty `strengths` on a clean
# truck is a failure to look, not a clean sheet.

VIEW_CONFIRMATIONS = {
    "tire_wheel": [
        "that the tread stands proud of the wear bars, and roughly how far, if you can see it",
        "that wear is even across the ribs and matched between the tires on this axle",
        "that the sidewalls are free of cracking, bulges and cuts",
        "that the rim is straight and the wheel nut indicators, if fitted, are in line",
    ],
    "dashboard_odometer": [
        "that the cluster is live and no engine, brake, ABS/EBS or emissions lamp is lit",
        "that the display carries no fault, service or regeneration message",
        "that the dash top and the instrument surround are complete and uncracked",
        "that the wear on the wheel rim, the stalks and the switches is in proportion to the distance this truck has covered",
    ],
    "interior_cab": [
        "that the driver's seat holds its shape and its bolster has not collapsed",
        "that the cab is dry, with no staining at the roof hatch or the screen surround",
        "that the trim, switches and dash top are complete and uncracked",
        "that the bunk and its panels are sound and the storage doors still shut",
    ],
    "engine_bay": [
        "that the visible joints, hoses and the area under the engine are dry",
        "that the belts are intact and the pulleys are in line",
        "that the loom is original: no tape repairs, no disturbed connectors",
    ],
    "chassis_undercarriage": [
        "that the frame rails carry paint rather than lifting scale",
        "that the air bags are inflated and sitting square, and the shocks are dry",
        "that no crossmember carries a weld repair or an added plate",
        "that the air tanks, lines and their straps are sound and unchafed",
    ],
    "fifth_wheel": [
        "that the plate carries a grease film rather than being dry and polished",
        "that the jaws are closed and the handle is in its stowed position",
        "that the mounting bolts are present and the mounting plate shows no fretting",
    ],
    "exterior_side": [
        "that the panel gaps run parallel the length of the truck",
        "that the paint matches panel to panel",
        "that the tanks, steps and fairings are complete and undamaged",
        "that any chassis rail visible below the cab still carries its paint",
    ],
    "exterior_front": [
        "that the lamp lenses are clear and dry behind the glass",
        "that the screen is unchipped through the swept area and unscored by the wipers",
        "that the bumper, grille and valance are complete, in line and unbroken",
        "that the mirror arms and glass are undamaged and sitting where they should",
    ],
    "exterior_front_34": [
        "that the truck sits level, with no corner down",
        "that the paint matches across the two visible sides and the shut lines run parallel",
        "that the visible tread stands proud of the wear bars and matches across the axle",
        "that the fairings, deflector and steps are complete",
    ],
    "exterior_rear": [
        "that the catwalk, lamp clusters and suzie coils are complete and undamaged",
        "that the mudflaps and their brackets are present and straight",
        "that the rail ends and crossmembers carry paint rather than lifting scale",
        "that the drive tires match across the axle and are wearing evenly",
    ],
    "damage_detail": [
        "that the damage stops where the frame shows it stopping, with the panels around it straight",
        "that nothing is wet or running at the damage",
        "that the metal around any rust still carries paint rather than lifting scale",
        "that a repair already made is sound: the metal straight, the paint keyed in, no filler cracking",
    ],
}


# --- the per-view checklists ----------------------------------------------
# Keyed by the gate's canonical view ids (app.vision.VIEW_LABELS). A view with
# no entry falls back to GENERIC_QUESTIONS, so adding a view to the taxonomy
# degrades to a weaker prompt rather than to a crash.

VIEW_QUESTIONS = {
    "tire_wheel": [
        "tread depth, and whether it is even across the ribs or lower on one shoulder",
        "shoulder versus centre wear - centre-worn means over-inflation, shoulder-worn means under-inflation or alignment",
        "cupping, feathering or scalloping, which point at suspension or alignment rather than mileage",
        "sidewall cracking, weather checking, bulges, cuts or exposed cord",
        "the DOT / week-year date code if any part of it is legible",
        "evidence of a retread: the buff line, a cap edge, or a casing brand that differs from the tread brand",
        "whether the tire brands and tread patterns match across the axle",
        "rim condition - corrosion, kerbing, cracks around the bolt holes, missing weights",
        "wheel nut indicators, if fitted, and whether any have rotated out of line",
        "the valve stem, and any visible hub or drum oil staining behind the wheel",
    ],
    "dashboard_odometer": [
        "the odometer reading, digit by digit - state exactly what you can read and mark anything you are guessing",
        "whether the units are km or miles, and where you read that from",
        "every warning or fault lamp that is lit, named individually",
        "fuel and AdBlue levels",
        "any service-interval, DPF regeneration or fault-code message on the display",
        "engine hours, if the cluster shows them",
        "wear on the steering wheel rim, stalks and the switches nearest the driver, and whether it is in proportion to the distance stated above - say which of the two it is. Wear that matches the stated distance is a confirmation and belongs in strengths. Only wear clearly beyond it is an observation, and even then describe the wear rather than accusing the odometer",
        "cracks, delamination or missing trim on the dash top and the instrument surround",
    ],
    "interior_cab": [
        "seat bolster collapse, tears, cracked vinyl and whether the driver's seat has dropped",
        "the bunk or sleeper: mattress condition, panel damage, storage doors",
        "floor covering wear, particularly at the pedals and the entry step",
        "headliner and door card condition, sagging or staining",
        "aftermarket holes, cut wiring, missing trim - signs of hard fleet use",
        "damp, mould or water staining, which points at a leaking roof hatch or windscreen seal",
        "how worn the cab is relative to the distance stated above. A cab worked in for the distance this truck has covered is expected and belongs in strengths. Only wear clearly ahead of that distance is a finding - and a tidy cab on a high-distance truck is a re-trim or a careful driver, not proof of anything",
    ],
    "engine_bay": [
        "oil, coolant, fuel or AdBlue staining, and specifically where it is coming from",
        "belt condition - glazing, cracking, fraying - and pulley alignment",
        "hose condition: cracking, swelling, chafe marks, mismatched clamps",
        "turbocharger and charge pipework, including any oil misting at the joints",
        "wiring loom condition, chafe, tape repairs and disturbed connectors",
        "corrosion on brackets, the radiator core and the charge cooler",
        "evidence of recent work: clean fasteners on a dirty engine, new hoses, sealant squeeze-out",
    ],
    "chassis_undercarriage": [
        "frame rail corrosion, and whether it is surface rust or scaling and lifting",
        "crossmember condition and any welded repairs or added plates",
        "air bag condition - sagging, perishing, chafe against the chassis",
        "shock absorber staining, which means the seal has gone",
        "air tank and line condition, strap corrosion, chafed or spliced airlines",
        "driveline and differential housing leaks",
        "underbody impact damage, bent steps, missing guards",
    ],
    "fifth_wheel": [
        "plate wear: scoring, grooving, and whether the lubricant film is present or the plate is dry and polished",
        "the jaws and locking mechanism - visible wear, play, or a handle sitting out of position",
        "mounting bolts and the mounting plate, including elongated holes or fretting",
        "ramps and the throat for impact damage from missed couplings",
        "coupling height relative to the chassis, and any packing plates",
    ],
    "exterior_side": [
        "panel alignment and door gaps down the whole side",
        "side fairing and skirt condition - cracks, missing sections, mismatched colour",
        "fuel and AdBlue tank condition: dents, strap corrosion, scuffing",
        "cab steps and grab handles",
        "any chassis rail visible below the cab, and its corrosion state",
        "exhaust stack or under-chassis exhaust condition",
        "the catwalk and the coupling area behind the cab",
        "whether the paint on this side matches the rest of the truck, panel by panel",
    ],
    "exterior_front": [
        "grille, bumper and valance condition - cracks, missing fixings, impact damage",
        "headlamp and DRL condition: clouding, cracking, moisture inside the lens",
        "windscreen chips, cracks and wiper smear arcs",
        "mirror arms and glass, including any heated-mirror wiring",
        "the front panel gaps and whether anything has been replaced or resprayed",
        "badges and model lettering, quoted literally",
        "sun visor and roof deflector condition",
    ],
    "exterior_front_34": [
        "overall stance - does the truck sit level, or is one corner low",
        "panel alignment and paint consistency across the two visible sides",
        "bumper, grille, headlamp and mirror condition",
        "fairing, skirt and deflector condition",
        "the visible tires and rims, at whatever detail the distance allows",
        "cab steps, fuel tank and the coupling area if visible",
        "badges and lettering, quoted literally",
    ],
    "exterior_rear": [
        "the rear of the cab, catwalk, and coupling gear",
        "the fifth wheel plate if any of it is visible from here",
        "lamp clusters, wiring and suzie coil condition",
        "mudflaps, guards and their brackets",
        "chassis rail ends and cross-member corrosion",
        "rear tire condition and matching across the drive axle",
    ],
    "damage_detail": [
        "exactly what the damage is, and its extent in the frame",
        "whether it is cosmetic, structural, or a fluid or corrosion problem",
        "whether it looks fresh or long-standing",
        "whether any repair has already been attempted, and how well",
    ],
}

GENERIC_QUESTIONS = [
    "what part of the truck this frame actually shows",
    "any visible defect, described in the terms a fleet buyer would use",
    "anything that looks recently replaced, repaired or resprayed",
    "anything in this frame that contradicts what the other photos suggest",
]


def questions_for(view: str) -> list[str]:
    return VIEW_QUESTIONS.get(view, GENERIC_QUESTIONS)


# --- pass A: identity ------------------------------------------------------
# Two things changed here and both are about making an answer checkable.
#
# `model` was free text, so "F Max", "FMAX" and "F-MAX 500" were three answers
# to one question and only one of them joined against the anchor row.
# `modelspec.vocabulary` closes the list. It ends in "other" deliberately: a
# judge's truck may be a model this repo has never heard of, and forcing a pick
# from a closed list is a worse answer than an honest "other". With no card the
# enum disappears and the field is free text exactly as it was.
#
# `approx_year_range` was free text nothing could check either - a range with
# no stated basis, read next to a declared year it was probably derived from.
# `generation` is picked from the card's closed list using visual markers and
# `year_evidence` says what in the photographs supports the range. The rule
# that makes them worth anything is in the prompt in as many words: the
# generation is read off the grille, NEVER off the year the seller typed. A
# year-derived generation agrees with the year by construction, and a
# cross-check that can only agree catches nothing.


def identity_schema(make: str | None = None) -> dict:
    """Pass A's schema, closed over the model card when there is one.

    `make` narrows the list: once the brand is settled, offering a model from
    another brand is offering a wrong answer. It is optional because pass A is
    the call that settles the brand - with nothing passed the enum is every
    model the card knows, which is the shape the first sample is sent in.
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["make", "model", "body_type", "cab_type", "axle_config",
                     "approx_year_range", "generation", "generation_conf",
                     "year_evidence", "badges_seen", "confidence",
                     "same_vehicle", "vehicle_mismatch"],
        "properties": {
            "make": {"type": ["string", "null"]},
            "model": {"type": ["string", "null"]},
            "body_type": {"type": ["string", "null"]},
            "cab_type": {"type": ["string", "null"]},
            "axle_config": {"type": ["string", "null"]},
            "approx_year_range": {"type": ["string", "null"]},
            # Not an enum. A generation list belongs to a (brand, model) pair
            # and this schema is built before the model is known - that is the
            # question pass A answers. The closed list travels in the prompt
            # instead, where the caller can condition it on a model it has.
            "generation": {"type": ["string", "null"]},
            "generation_conf": {"type": "number"},
            "year_evidence": {"type": "string"},
            "badges_seen": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "same_vehicle": {"type": "boolean"},
            "vehicle_mismatch": {"type": "string"},
        },
    }
    vocabulary = modelspec.vocabulary(make)
    if vocabulary:
        schema["properties"]["model"] = {"type": ["string", "null"],
                                         "enum": vocabulary + [None]}
    return schema


def __getattr__(name: str):
    """`prompts.IDENTITY_SCHEMA`, resolved when it is read rather than at import.

    Importing this module must not read a file off disk. A malformed card
    would otherwise take down `app.cli doctor`, which is the command you run
    to find out that the card is malformed. Not cached either, so it tracks
    `modelspec._reset()`; this is read a couple of times in a run and building
    it is a dict literal over an already-parsed card.
    """
    if name == "IDENTITY_SCHEMA":
        return identity_schema()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


IDENTITY_PROMPT = """\
Return ONE JSON object and nothing else. No markdown fence, no commentary.

{{
  "make": string|null,              // read from badges, grille and styling, not from the seller
  "model": string|null,
  "body_type": "tractor_unit"|"rigid"|"other"|null,
  "cab_type": string|null,          // e.g. "high sleeper", "day cab"
  "axle_config": string|null,       // e.g. "4x2", "6x4"
  "approx_year_range": string|null, // e.g. "2018-2022", from the generation you can see
  "generation": string|null,        // which generation of this model, by id, from the
                                    // closed list below when one is given; null when
                                    // no list is given and the markers are all you have
  "generation_conf": 0.0-1.0,       // how far the markers you can actually see get you.
                                    // 0 when you had nothing to read it off
  "year_evidence": string,          // one sentence: what IN THE PHOTOGRAPHS puts this
                                    // truck in that range - the marker you read it off.
                                    // "" when nothing in them supports it
  "badges_seen": [string],          // literal text visible on the truck
  "confidence": 0.0-1.0,
  "same_vehicle": boolean,          // are ALL these photos the same vehicle?
  "vehicle_mismatch": string        // if not, what differs and where; "" if they match
}}

This pass is about IDENTITY ONLY. Do not report condition, defects or wear -
each photo is being examined in detail separately.

On "generation", "approx_year_range" and "year_evidence": read the generation
off what is VISIBLE - the grille, the lamp signature, the mirror housings, the
door script, the bumper and the trim - and NEVER off the year the seller typed.
That year is the thing this read is used to check: a generation derived from it
would agree with it by construction, and a cross-check that can only agree
catches nothing. If the markers and the declared year disagree, answer with
what you can see and say so in "year_evidence". If nothing in these photographs
settles it, leave the generation unsettled and set "generation_conf" to 0 - an
honest null is a useful answer here and a guess is not.

On "same_vehicle": only call a mismatch on IDENTIFYING evidence - a different
number plate, a different cab generation or model, a different colour,
different badging, or a different axle configuration. Photos of one truck
routinely differ in lighting, weather, mud, blur, resolution, angle, background
and colour cast, and NONE of that is evidence of a different vehicle; a seller
with a phone produces exactly that variation. If you are not certain, set
"same_vehicle" true and describe the doubt in "vehicle_mismatch" anyway. Only a
confident, nameable difference should set it false, because doing so stops the
valuation entirely.

{context}

Return the JSON object now."""


# The three card blocks. Each one is empty when the card cannot answer, and
# `identity_prompt` joins what survives onto the end of the context - so the
# order a reader gets is: these photographs, what the seller claims, what these
# models look like, answer. With no card at all the composition is the identity
# and the prompt is byte-identical to the template's; a test pins that, because
# a block that leaves an empty heading or a dangling "one of:" behind is a
# regression nobody would catch by eye.

_MODEL_LIST = """\
"model" is a closed list. Answer with exactly one of:
{models}
Use "other" for a truck that is none of them. That name is joined against a
reference card downstream, so a wrong pick from the list describes a different
vehicle; "other" costs the join and nothing else."""

_GENERATIONS = """\
The {model} has these generations, and these are what separate them by eye. If
that is what you are looking at, answer "generation" with one of these ids and
nothing else:
{rows}
  unknown
Read it off those markers. The year the seller typed is not one of them, and
"unknown" is a better answer than a generation you reasoned back from it."""

_TELLS = """\
Tells that separate the {model} from the models it is most often taken for:
{rows}"""


def _model_list_block(make: str | None) -> str:
    vocabulary = modelspec.vocabulary(make)
    if not vocabulary:
        return ""
    return _MODEL_LIST.format(models=textwrap.fill(
        ", ".join(vocabulary), width=76, initial_indent="  ",
        subsequent_indent="  "))


def _generation_block(make: str | None, model: str | None) -> str:
    """Only ever for a model that has already been named.

    With the make alone there is nothing to list: generations belong to a
    (brand, model) pair, and reciting every Ford generation would be telling
    the call what to look for on a truck it has not identified yet.
    """
    generations = modelspec.generations(make, model) if model else []
    rows = []
    for gen in generations:
        if not gen.get("id"):
            continue
        head = f"  {gen['id']}"
        if gen.get("years"):
            head += f" ({gen['years']})"
        rows.append(head + ":")
        # One marker per line, unwrapped. A filled paragraph would break a
        # marker across a line boundary, and these are read as a checklist.
        rows += [f"    - {marker}" for marker in (gen.get("visual_markers") or [])]
    if not rows:
        return ""
    return _GENERATIONS.format(model=_canonical(make, model), rows="\n".join(rows))


def _tells_block(make: str | None, model: str | None) -> str:
    tells = modelspec.identity_tells(make, model) if model else []
    if not tells:
        return ""
    return _TELLS.format(model=_canonical(make, model),
                         rows="\n".join(f"  - {tell}" for tell in tells))


def _canonical(make: str | None, model: str | None) -> str:
    """The name the card matched on, not the one the caller typed.

    Both blocks only ever run when `modelspec.card` found a row, and it finds
    one through the aliases - so a caller holding the raw vision answer "F Max"
    would otherwise print a heading naming a model the reference does not have.
    """
    return modelspec.normalise_model(make, model) or str(model).upper()


def identity_prompt(*, context: str, make: str | None = None,
                    model: str | None = None) -> str:
    """The template, plus whatever the model card can say about this vehicle."""
    blocks = [context, _model_list_block(make), _generation_block(make, model),
              _tells_block(make, model)]
    return IDENTITY_PROMPT.format(context="\n\n".join(b for b in blocks if b))


# --- pass A2: the badge, read on its own ----------------------------------
# A second and INDEPENDENT reading of what the truck says it is, in the same
# posture as the odometer OCR and the chassis-plate VIN: one call on a
# full-resolution crop of the grille and door band, rather than a badge read
# off a downscaled eight-photo montage by the same call that is also deciding
# body type, axle count and whether the photos are one vehicle.
#
# Independent is the whole value, and it is fragile. `badge_prompt` takes
# `make` and `model` so it COULD name a brand's badge conventions, and it
# deliberately uses neither: those two fields are the answers this call exists
# to witness, and a witness who has been told the answer corroborates nothing -
# `reconcile` would be comparing pass A against a paraphrase of pass A. The
# parameters stay in the signature so the call site does not have to care, and
# a test pins that the prompt is the same string with them and without them.
#
# The second reason this prompt is written the way it is: the crop is the upper
# band of the cab, and on a dealer lot that band includes the windscreen. A
# windscreen carries an asking figure and a telephone number. A transcription
# pass pointed at it is the one place in this system where a currency figure
# could walk back IN through the model's own answer, and `badge_text` is
# displayed - hence rule 4, which is also the no-seller-PII rule the dataset
# scripts have always enforced.

BADGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["badge_text", "make", "model", "trim_or_power", "legible",
                 "confidence"],
    "properties": {
        "badge_text": {"type": "array", "items": {"type": "string"}},
        # Free strings, not the identity enum. A transcription has to be able
        # to disagree with the vocabulary: if this pass answered from the same
        # closed list pass A does, the two would agree on a truck neither had
        # read properly, which is the failure the second reading exists to
        # catch.
        "make": {"type": ["string", "null"]},
        "model": {"type": ["string", "null"]},
        "trim_or_power": {"type": ["string", "null"]},
        "legible": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
}

BADGE_PROMPT = """\
You are looking at a CROP taken from a larger photograph of a used tractor
unit - the band across the front and the upper side of the cab where the
maker's badge, the grille lettering and the model script on the door sit.
Everything outside that band has been cut away.

Your job is to TRANSCRIBE, and only to transcribe. Read the lettering off the
metal and the plastic and write down the characters that are actually there.

Return ONE JSON object and nothing else. No markdown fence, no commentary.

{
  "badge_text": [string],       // every piece of lettering you can read, verbatim,
                                // one entry per badge. Nothing interpreted, nothing
                                // expanded, nothing tidied up
  "make": string|null,          // the manufacturer, ONLY if its name or its emblem
                                // is legible in this crop
  "model": string|null,         // the model, ONLY if the model script itself is
                                // legible. Null otherwise - see rule 2
  "trim_or_power": string|null, // a trim or power figure carried on its own badge,
                                // e.g. the "500" on a door. Null if there is none
  "legible": boolean,           // is there readable lettering in this crop at all?
  "confidence": 0.0-1.0
}

Rules, in order of importance:
1. Transcribe, do not interpret. Keep the spelling, the spacing, the hyphens
   and the digits exactly as they appear. A badge you can only half read goes
   into "badge_text" as the characters you are sure of - it does not get
   completed from what you expect the rest of it to say.
2. Do NOT identify the model from the shape of the cab, the grille pattern, the
   lamp signature or anything else about the styling. Another call reads this
   truck's styling, with every photograph of it in front of that call. This one
   is here to say what the truck has WRITTEN on it, and a styling guess made
   here agrees with that other call by construction and confirms nothing.
3. If there is no legible lettering, set "legible" false, return an empty
   "badge_text" and leave "make" and "model" null. That is a useful answer: it
   says the badge could not be read rather than pretending it was.
4. Read only what is moulded, pressed, welded or scripted onto the truck
   itself. A windscreen sticker, a dealer board, a registration plate and a
   telephone number are not badges and none of them belongs in "badge_text".
5. Do not describe condition, damage or wear. Another call is doing that, one
   photograph at a time.

Return the JSON object now."""


def badge_prompt(*, make: str | None = None, model: str | None = None) -> str:
    """The badge read. `make` and `model` are accepted and deliberately unused.

    See the block comment above: naming either one hands this call the answer
    it is here to provide independently. The parameters exist so that the wiring
    reads the same as every other pass and so that a later maintainer who finds
    a way to use them without biasing the read has somewhere to put it.
    """
    return BADGE_PROMPT


# --- pass B: one photo, in depth -------------------------------------------

def closeup_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["shows", "legible", "odometer_km", "observations", "strengths",
                     "cannot_tell", "confidence"],
        "properties": {
            "shows": {"type": "string"},
            "legible": {"type": "boolean"},
            "odometer_km": {"type": ["integer", "null"]},
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["component", "observation", "severity",
                                 "confidence", "price_impact", "magnitude", "box"],
                    "properties": {
                        "component": {"type": "string", "enum": COMPONENTS},
                        "observation": {"type": "string"},
                        "severity": {"type": "string", "enum": SEVERITIES},
                        "confidence": {"type": "number"},
                        "price_impact": {"type": "string", "enum": IMPACTS},
                        "magnitude": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["extent", "state", "consumable", "blocks_use"],
                            "properties": {
                                "extent": {"type": "string", "enum": EXTENTS},
                                "state": {"type": "string", "enum": STATES},
                                "consumable": {"type": "boolean"},
                                "blocks_use": {"type": "string", "enum": BLOCKS_USE},
                            },
                        },
                        "box": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                },
            },
            "strengths": {"type": "array", "items": {"type": "string"}},
            "cannot_tell": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
    }


# The close-up prompt is assembled in three zones, and the order is the whole
# point of splitting it. Everything invariant comes first - the task, the
# rubric, the worked examples, the schema and the rules - then the two lines
# that are the same for every photo of one truck, then the frame in front of
# this call. It used to open with the vehicle line and the view tag, which put
# the variable part in the prefix and defeated prefix caching on every backend
# that has it. With the rubric added and each photo read CLOSEUP_SAMPLES times
# that ordering is the difference between paying for this block 48 times and
# paying for it once. A test asserts the invariant prefix is byte-identical
# across two different views.

_CLOSEUP_INVARIANT = """\
You are looking at ONE photograph of a used tractor unit. Examine it properly -
this is the only look anyone will take at this particular frame.

{rubric}

{examples}

Return ONE JSON object and nothing else. No markdown fence, no commentary.

{{
  "shows": string,          // one plain sentence: what this frame is a photo of
  "legible": boolean,       // is it sharp and bright enough to support fine detail?
  "odometer_km": integer|null,  // ONLY if an odometer display is legible in THIS
                                // photo. Convert miles to km and say so in "shows".
                                // Null for every frame that is not showing one.
  "observations": [         // things that are WORSE than this truck's age and
                            // distance predict, each one visible in THIS photo.
                            // Wear that is on schedule goes in "strengths".
    {{
      "component": one of {components},
      "observation": string,       // specific and visual, in fleet-buyer vocabulary
      "severity": one of {severities},   // provisional - see rule 11
      "confidence": 0.0-1.0,
      "price_impact": one of {impacts},
      "magnitude": {{              // what this ONE photograph can say about the
                                   // size of the defect without comparing it to
                                   // anything else on the truck
        "extent": one of {extents},
        "state": one of {states},
        "consumable": boolean,     // is this a part that is MEANT to wear out and
                                   // be replaced - a tire, a brake pad, a mudflap?
        "blocks_use": one of {blocks}   // would this stop the truck working, or
                                        // working legally, today?
      }},
      "box": [x, y, w, h]|null     // normalised 0-1 of THIS image. The smallest
                                   // rectangle that contains the visible evidence
                                   // for this observation. Null if you cannot
                                   // point at the pixels - never a guess.
    }}
  ],
  "strengths": [string],    // components this frame shows to be sound, or worn
                            // exactly as much as this truck should be. Work the
                            // confirm list below. This array is read by the buyer
                            // and by the pass that sets the final severities; an
                            // empty one on a sound truck is a failure to look,
                            // not a clean sheet.
  "cannot_tell": [string],  // what this frame cannot answer, and what shot would
  "confidence": 0.0-1.0
}}

Rules, in order of importance:
1. Report only what is visible IN THIS PHOTOGRAPH. A component that is not in
   this frame is not an observation - if it matters, put what you would need to
   see into "cannot_tell".
2. "observation" describes what is visible ("outer shoulder of the near-side
   steer tire is worn noticeably below the centre ribs"), never a verdict
   ("tires bad").
3. If this photo is blurred, dark or too distant to support a claim, set
   "legible" false and keep your observations to what survives that. Do not
   claim tread depth from a soft-focus frame.
4. Do not invent defects to fill the list, and do not promote a real one to make
   it worth reporting. Those are two different errors and the second is the
   common one. A genuinely clean component belongs in "strengths"; a component
   worn exactly as much as this truck's age and distance predict also belongs in
   "strengths", named as on schedule; and an empty "observations" list is a
   perfectly good answer for a photograph of a sound truck.
5. When you are between two levels, take the LOWER one, as the rubric says, and
   put what would have to be true for the higher one into the observation. A
   "moderate" you are certain of is worth more to a buyer than a "major" you
   argued yourself into.
6. Cosmetic findings are worth reporting but must be marked severity "cosmetic"
   and price_impact "none" or "low". Paint and trim alone are never "major": a
   truck is not unroadworthy because it is scratched.
7. Never write a photo number into prose.
8. Do not estimate a price. You are describing a truck, not valuing one.
9. "odometer_km" is null unless this photograph actually shows an odometer you
   can read. A guess at a mileage is worse than no mileage, because the number
   downstream is checked against what the seller typed.
10. "box" is the region that shows THIS defect, not the whole component if the
    wear is local (the outer shoulder of one tire, not every tire in frame).
    Coordinates are of the image in front of you. Null rather than a guess.
11. "magnitude" is the part of your answer that does not depend on comparison.
    Fill it in even when the severity is uncertain: a later pass that can see
    every photograph of this truck at once sets the final level from it, and an
    honest extent with an uncertain severity is worth more to that pass than a
    confident severity with nothing behind it."""


CLOSEUP_INVARIANT = _CLOSEUP_INVARIANT.format(
    rubric=SEVERITY_RUBRIC, examples=WORKED_EXAMPLES,
    components=json.dumps(COMPONENTS), severities=json.dumps(SEVERITIES),
    impacts=json.dumps(IMPACTS), extents=json.dumps(EXTENTS),
    states=json.dumps(STATES), blocks=json.dumps(BLOCKS_USE))


# The known weak points of a MODEL, as opposed to observations about a truck.
# This is the one piece of model awareness that can make the report worse: a
# call told that the AdBlue tank is a known weak point will report a weeping
# AdBlue tank, because it has been handed a plausible finding and asked to look
# for it. That is the over-reporting `SEVERITY_RUBRIC` was written against, now
# with a reference card behind it, which reads to a buyer like corroboration.
# So the block says three things in order - it is a prior about the model, it
# is not an observation about this truck, report it ONLY if it is in this
# frame - and the list is filtered to the components the view can show, so
# there is nothing in it the call could not check.
_WEAK_POINTS = ('Known weak points on this model, from a reference card. This is a '
                'prior about the model and not an observation about this truck. '
                'Check them specifically and report them ONLY if you can see them '
                'in THIS frame; if the shot is what stops you seeing one, that is '
                'a "cannot_tell" and not a finding. One that is present and no '
                "worse than this truck's age and distance predict is on schedule "
                'like any other wear, and belongs in "strengths":\n')


def _weak_point_rows(weak_points, view: str) -> list[str]:
    allowed = set(view_components(view))
    rows = []
    for entry in weak_points or ():
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        component, note = str(entry[0] or "").strip(), str(entry[1] or "").strip()
        if note and component in allowed:
            rows.append(f"  - {component}: {note}")
    return rows


def closeup_prompt(*, view: str, view_pretty: str, vehicle: str,
                   cropped: bool, soft: bool, expectation: str = "",
                   band: str | None = None, spec_lines: list[str] | tuple = (),
                   weak_points: list | tuple = ()) -> str:
    """The three zones, joined. Invariant, then per-appraisal, then per-photo.

    `spec_lines` and `weak_points` are what `app.modelspec` knows about the
    model the identity pass named, and they land in the two zones for the same
    reason everything else does: the spec is the same for all sixteen photos of
    one truck, the weak points are filtered per view and so change per frame.
    Both default to empty, and empty is byte-identical to the prompt as it was
    before either existed - the card is optional and a missing one has to leave
    this call exactly as it ran.
    """
    # `or ()` on both: a caller reading them off a card row gets None when the
    # row has no entry, and an optional input that raises is not optional.
    spec = (str(line).strip() for line in spec_lines or ())
    appraisal = "\n".join(x for x in (vehicle, *spec, expectation) if x)

    photo = [f'This frame was tagged "{view_pretty}" by a zero-shot classifier. '
             f'That tag is a hint and is sometimes wrong; trust the pixels.']
    if cropped:
        photo.append("This image has been cropped to the one vehicle being sold; "
                     "other vehicles in the original frame were deliberately "
                     "excluded, so describe only what is in front of you.")
    if soft:
        photo.append("The capture check scored this frame as soft focus. Do not "
                     "claim fine detail such as tread depth from it - mark it not "
                     "legible instead.")
    anchors = COMPONENT_ANCHORS.get(view)
    if anchors:
        photo.append("How the four levels read on the parts that should be in this "
                     "frame:\n\n" + anchors)
    rows = EXPECTED_WEAR.get(view, {}).get(band or "")
    if rows:
        photo.append("At the distance stated above, this is what on schedule looks "
                     "like for those parts. A component in this state is not a "
                     "finding at any level:\n"
                     + "\n".join(f"  - {row}" for row in rows))
    weak = _weak_point_rows(weak_points, view)
    if weak:
        photo.append(_WEAK_POINTS + "\n".join(weak))
    photo.append("Work through each of these for this photo, and report what you "
                 "can actually see:\n\n"
                 + "\n".join(f"  - {q}" for q in questions_for(view)))
    confirmations = VIEW_CONFIRMATIONS.get(view)
    if confirmations:
        photo.append("Then confirm each of these, and put what you can confirm into "
                     '"strengths":\n\n'
                     + "\n".join(f"  - {c}" for c in confirmations))
    photo.append("Return the JSON object now.")

    return "\n\n".join(x for x in (CLOSEUP_INVARIANT, appraisal, "\n\n".join(photo)) if x)


# --- pass C: synthesis, text only ------------------------------------------

def synthesis_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["condition_summary", "condition_grade", "coverage_gaps",
                     "headline", "confidence", "duplicates"],
        "properties": {
            "condition_summary": {
                "type": "object",
                "additionalProperties": False,
                "required": list(SUMMARY_KEYS),
                "properties": {k: {"type": "string"} for k in SUMMARY_KEYS},
            },
            "condition_grade": {"type": "string", "enum": GRADES + ["unknown"]},
            "coverage_gaps": {"type": "array", "items": {"type": "string"}},
            "headline": {"type": "string"},
            "confidence": {"type": "number"},
            "duplicates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["keep", "merge"],
                    "properties": {
                        "keep": {"type": "integer"},
                        "merge": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
        },
    }


SYNTHESIS_PROMPT = """\
Below are the notes from examining {n} photographs of one tractor unit, one
photograph at a time. Nothing here is your own recollection - work only from
these notes.

{vehicle_line}
{notes}

Return ONE JSON object and nothing else. No markdown fence, no commentary.

{{
  "condition_summary": {{ {summary_keys} }},  // one or two sentences each
  "condition_grade": one of {grades},
  "coverage_gaps": [string],   // what still could not be assessed across the whole set
  "headline": string,          // one sentence a buyer would read first
  "confidence": 0.0-1.0,
  "duplicates": [              // the same defect reported from several photos
    {{"keep": integer, "merge": [integer]}}   // finding numbers from the list above
  ]
}}

Rules:
1. Each "condition_summary" value must be supported by the notes above. Where
   nothing in the notes covers it, write "not visible in these photos".
2. Each photograph was examined on its own and did not see the others, so the
   SAME defect is described several times in different words. "duplicates" is
   how one worn drive tire reported from three frames becomes one finding
   rather than three. Be thorough about this - go through the list looking for
   paraphrases of the same defect on the same component, and pair them up.
   Genuinely different defects stay separate: two different tires on the same
   axle are two findings, and wear on a tire is not the same as damage to its
   rim.
3. Grade on the WORST DISTINCT defect and on how much of the truck was
   visible - never on how many findings there are. A well-photographed truck
   produces a long list because it was photographed well, not because it is in
   worse condition than one photographed badly. Cosmetic and minor entries
   must not pull the grade down; "poor" means something on this truck needs
   money spent on it before it works, and "fair" means real wear a buyer would
   negotiate over.
4. A truck with no visible defects and thin coverage is not "excellent" - it is
   ungraded coverage, so say so in coverage_gaps and keep confidence low.
5. Do not introduce a defect that is not in the notes above.
6. Do not estimate a price."""


def synthesis_prompt(*, n: int, vehicle: str, notes: str) -> str:
    return SYNTHESIS_PROMPT.format(
        n=n, vehicle_line=vehicle, notes=notes,
        summary_keys=", ".join(f'"{k}": string' for k in SUMMARY_KEYS),
        grades=json.dumps(GRADES))


# --- pass D: set-aware severity, text only --------------------------------
# The pass that exists because of what pass B structurally cannot see. Every
# severity in the list below was assigned by a call that saw one photograph and
# had nothing to compare it against - a relative ordinal demanded from an
# absolute-only observation, sixteen times over, and then summed into a price.
# This is the only stage that sees the whole finding list at once, and it runs
# AFTER `merge_duplicates`: calibrating the pre-merge list would calibrate a
# list in which three paraphrases of one worn drive tire are still three rows.
#
# Its licence is deliberately asymmetric, because the measured direction of
# error is upward. `calibration.apply_revisions` enforces it: lower freely,
# raise by at most one level, raise to "major" only with corroboration from a
# second photograph. Every change is written down as a `Correction`.

CALIBRATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["revisions", "worst_finding", "calibration_note"],
    "properties": {
        "revisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["finding", "severity", "price_impact", "reason"],
                "properties": {
                    "finding": {"type": "integer"},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "price_impact": {"type": "string", "enum": IMPACTS},
                    "reason": {"type": "string"},
                },
            },
        },
        "worst_finding": {"type": ["integer", "null"]},
        "calibration_note": {"type": "string"},
    },
}

CALIBRATION_PROMPT = """\
Below is the finished list of findings for ONE tractor unit. Duplicates have
already been folded together, so every line is a distinct defect. Nothing here
is your own recollection - work only from this list.

{vehicle_line}
{expectation_line}

Coverage: {n_photos} photographs were read in depth.{coverage_line}

What those photographs confirmed to be sound:
{strengths}

The findings, numbered:
{findings}

{rubric}

Every finding above was given its severity by a call that saw ONE photograph
and nothing else. That call could describe what it saw, but it could not know
whether what it saw was the worst thing on this truck or the best, because it
had no other photograph to compare against. You have the whole set. Re-decide
every one of them.

Return ONE JSON object and nothing else. No markdown fence, no commentary.

{{
  "revisions": [            // EXACTLY one entry per finding, in order, 0 to {last}
    {{"finding": integer,
      "severity": one of {severities},
      "price_impact": one of {impacts},
      "reason": string}}    // one line: why this level and not the one above
  ],
  "worst_finding": integer|null,  // the single worst thing on this truck, or null
                                  // if nothing rises above ordinary wear
  "calibration_note": string      // one sentence: what this truck's condition
                                  // actually amounts to
}}

Rules, in order of importance:
1. Re-decide EVERY finding. Repeating the provisional severity is a decision,
   not a skip, and it still needs its reason.
2. You may lower a severity as far as the evidence supports. You may raise one
   by at most one level, and you may only raise a finding to "major" if it was
   seen in more than one photograph.
3. The most common correct revision is downward. These findings were written
   one photograph at a time by a reader who could not see that the same wear was
   everywhere on this truck and therefore ordinary, or that this was the only
   tired thing on an otherwise sound vehicle. Both change the answer, and only
   you can see them.
4. A component whose state matches what the distance above predicts is not a
   finding at any level. If one is in the list, set it to "cosmetic" with
   price_impact "none" and say in the reason that it is on schedule.
5. Corroboration is evidence of extent, and its absence is evidence too. A
   finding seen in one photograph of a component that four photographs show is a
   local mark; the same finding in three of those four is a condition of the
   whole component. Both counts are printed beside every finding.
6. "worst_finding" is a real commitment. On a truck whose worst problem is a
   kerbed rim, "worst_finding" points at the kerbed rim - not at whichever line
   happens to say "major".
7. Do not introduce a finding that is not in the list, and do not remove one.
   Every line gets an entry.
8. Do not estimate a price."""


def calibration_prompt(*, vehicle: str, expectation: str, n_photos: int,
                       coverage: str, strengths: str, findings: str,
                       last: int) -> str:
    return CALIBRATION_PROMPT.format(
        vehicle_line=vehicle, expectation_line=expectation, n_photos=n_photos,
        coverage_line=(" " + coverage if coverage else ""),
        strengths=strengths or "  (nothing was confirmed sound - treat that as thin "
                               "coverage, not as evidence against the truck)",
        findings=findings, rubric=SEVERITY_RUBRIC, last=max(0, last),
        severities=json.dumps(SEVERITIES), impacts=json.dumps(IMPACTS))


# Which system summary each component rolls up into. Used by the deterministic
# fallback when the synthesis call fails, and by the screen to group findings.
COMPONENT_SUMMARY = {
    "steer_tires": "tires", "drive_tires": "tires",
    "wheels_rims": "wheels_brakes", "brakes_hubs": "wheels_brakes",
    "fifth_wheel": "fifth_wheel_coupling", "coupling_airlines": "fifth_wheel_coupling",
    "chassis_frame": "chassis_corrosion", "undercarriage": "chassis_corrosion",
    "air_suspension": "chassis_corrosion", "air_tanks_lines": "chassis_corrosion",
    "corrosion": "chassis_corrosion", "mudflaps_guards": "chassis_corrosion",
    "cab_exterior_panels": "body_paint", "front_bumper_valance": "body_paint",
    "fairings_skirts": "body_paint", "roof_deflector": "body_paint",
    "doors_handles": "body_paint", "cab_steps": "body_paint",
    "paint_finish": "body_paint",
    "cab_interior_seats": "cab_interior", "steering_wheel_controls": "cab_interior",
    "dashboard_instruments": "cab_interior", "bunk_sleeper": "cab_interior",
    "cab_floor_trim": "cab_interior", "warning_lights": "cab_interior",
    "engine_bay": "engine_driveline", "fluid_leaks": "engine_driveline",
    "exhaust_dpf": "engine_driveline", "fuel_tank": "engine_driveline",
    "adblue_tank": "engine_driveline",
    "grille_headlights": "glass_lights", "mirrors_visor": "glass_lights",
    "windscreen_glass": "glass_lights",
}
