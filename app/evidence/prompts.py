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
"""
from __future__ import annotations

import json

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
                                 "confidence", "price_impact"],
                    "properties": {
                        "component": {"type": "string", "enum": COMPONENTS},
                        "observation": {"type": "string"},
                        "severity": {"type": "string", "enum": SEVERITIES},
                        "confidence": {"type": "number"},
                        "price_impact": {"type": "string", "enum": IMPACTS},
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
      "price_impact": one of {impacts}
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
   downstream is checked against what the seller typed."""


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
