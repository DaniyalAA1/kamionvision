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
"""
from __future__ import annotations

import json
import textwrap

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

You are rigorous about what a photo can and cannot show. You never infer \
mechanical condition from a clean exterior, never report a component that is not \
visible in the photo you were given, and you say so plainly when something is not \
assessable. Your notes go to a buyer who is deciding whether to drive six hours \
to see this truck."""


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
        "wear on the steering wheel rim, stalks and the switches nearest the driver - the honest cross-check on a low odometer",
        "cracks, delamination or missing trim on the dash top and the instrument surround",
    ],
    "interior_cab": [
        "seat bolster collapse, tears, cracked vinyl and whether the driver's seat has dropped",
        "the bunk or sleeper: mattress condition, panel damage, storage doors",
        "floor covering wear, particularly at the pedals and the entry step",
        "headliner and door card condition, sagging or staining",
        "aftermarket holes, cut wiring, missing trim - signs of hard fleet use",
        "damp, mould or water staining, which points at a leaking roof hatch or windscreen seal",
        "the general tidiness of the cab relative to the odometer you would expect",
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

IDENTITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["make", "model", "body_type", "cab_type", "axle_config",
                 "approx_year_range", "badges_seen", "confidence",
                 "same_vehicle", "vehicle_mismatch"],
    "properties": {
        "make": {"type": ["string", "null"]},
        "model": {"type": ["string", "null"]},
        "body_type": {"type": ["string", "null"]},
        "cab_type": {"type": ["string", "null"]},
        "axle_config": {"type": ["string", "null"]},
        "approx_year_range": {"type": ["string", "null"]},
        "badges_seen": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "same_vehicle": {"type": "boolean"},
        "vehicle_mismatch": {"type": "string"},
    },
}

IDENTITY_PROMPT = """\
Return ONE JSON object and nothing else. No markdown fence, no commentary.

{{
  "make": string|null,              // read from badges, grille and styling, not from the seller
  "model": string|null,
  "body_type": "tractor_unit"|"rigid"|"other"|null,
  "cab_type": string|null,          // e.g. "high sleeper", "day cab"
  "axle_config": string|null,       // e.g. "4x2", "6x4"
  "approx_year_range": string|null, // e.g. "2018-2022", from the generation you can see
  "badges_seen": [string],          // literal text visible on the truck
  "confidence": 0.0-1.0,
  "same_vehicle": boolean,          // are ALL these photos the same vehicle?
  "vehicle_mismatch": string        // if not, what differs and where; "" if they match
}}

This pass is about IDENTITY ONLY. Do not report condition, defects or wear -
each photo is being examined in detail separately.

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
                                 "confidence", "price_impact", "box"],
                    "properties": {
                        "component": {"type": "string", "enum": COMPONENTS},
                        "observation": {"type": "string"},
                        "severity": {"type": "string", "enum": SEVERITIES},
                        "confidence": {"type": "number"},
                        "price_impact": {"type": "string", "enum": IMPACTS},
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


CLOSEUP_PROMPT = """\
You are looking at ONE photograph of a used tractor unit. Examine it properly -
this is the only look anyone will take at this particular frame.

{vehicle_line}
A zero-shot classifier tagged this frame as "{view_pretty}". That tag is a hint
and is sometimes wrong; trust the pixels.
{crop_line}{soft_line}
Work through each of these for this photo, and report what you can actually see:

{checklist}

Return ONE JSON object and nothing else. No markdown fence, no commentary.

{{
  "shows": string,          // one plain sentence: what this frame is a photo of
  "legible": boolean,       // is it sharp and bright enough to support fine detail?
  "odometer_km": integer|null,  // ONLY if an odometer display is legible in THIS
                                // photo. Convert miles to km and say so in "shows".
                                // Null for every frame that is not showing one.
  "observations": [         // DEFECTS ONLY, each one a thing visible in THIS photo
    {{
      "component": one of {components},
      "observation": string,       // specific and visual, in fleet-buyer vocabulary
      "severity": one of {severities},
      "confidence": 0.0-1.0,
      "price_impact": one of {impacts},
      "box": [x, y, w, h]|null     // normalised 0-1 of THIS image. The smallest
                                   // rectangle that contains the visible evidence
                                   // for this observation. Null if you cannot
                                   // point at the pixels — never a guess.
    }}
  ],
  "strengths": [string],    // things this frame positively shows to be in good order
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
4. Do not invent defects to fill the list. A genuinely clean component belongs
   in "strengths", and an empty "observations" list is a perfectly good answer.
5. Cosmetic findings are worth reporting but must be marked severity "cosmetic"
   and price_impact "none" or "low".
6. Never write a photo number into prose.
7. Do not estimate a price. You are describing a truck, not valuing one.
8. "odometer_km" is null unless this photograph actually shows an odometer you
   can read. A guess at a mileage is worse than no mileage, because the number
   downstream is checked against what the seller typed.
9. "box" is the region that shows THIS defect, not the whole component if the
   wear is local (the outer shoulder of one tire, not every tire in frame).
   Coordinates are of the image in front of you. Null rather than a guess."""


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


def closeup_prompt(*, view: str, view_pretty: str, vehicle: str,
                   cropped: bool, soft: bool) -> str:
    checklist = "\n".join(f"  - {q}" for q in questions_for(view))
    crop_line = ""
    if cropped:
        crop_line = ("This image has been cropped to the one vehicle being sold; other "
                     "vehicles in the original frame were deliberately excluded, so "
                     "describe only what is in front of you.\n")
    soft_line = ""
    if soft:
        soft_line = ("The capture check scored this frame as soft focus. Do not claim "
                     "fine detail such as tread depth from it - mark it not legible "
                     "instead.\n")
    return CLOSEUP_PROMPT.format(
        vehicle_line=vehicle, view_pretty=view_pretty, crop_line=crop_line,
        soft_line=soft_line, checklist=checklist,
        components=json.dumps(COMPONENTS), severities=json.dumps(SEVERITIES),
        impacts=json.dumps(IMPACTS))


def synthesis_prompt(*, n: int, vehicle: str, notes: str) -> str:
    return SYNTHESIS_PROMPT.format(
        n=n, vehicle_line=vehicle, notes=notes,
        summary_keys=", ".join(f'"{k}": string' for k in SUMMARY_KEYS),
        grades=json.dumps(GRADES))


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
