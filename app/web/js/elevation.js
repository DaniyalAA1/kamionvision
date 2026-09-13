/* The elevation as a state machine.

   Two inputs, in order. The gate's `views_present` says which parts of the
   truck were actually photographed, which promotes those zones from phantom
   to established. The vision model's findings then promote them again and set
   a severity, which changes stroke weight and adds a revision triangle - the
   mark a drawing uses for a changed area.

   The view vocabulary is VIEW_PROMPTS in app/vision.py; the component
   vocabulary is COMPONENTS in app/evidence.py. Both are closed sets, so this
   map is exhaustive rather than a guess. */

import { svg, svgText, titleise, reduced } from './dom.js';

const VIEW_ZONES = {
  exterior_front: ['grille_headlights', 'front_bumper_valance', 'windscreen_glass',
                   'mirrors_visor'],
  exterior_front_34: ['cab_exterior_panels', 'doors_handles', 'roof_deflector',
                      'steer_tires', 'grille_headlights', 'windscreen_glass'],
  exterior_side: ['cab_exterior_panels', 'doors_handles', 'fairings_skirts',
                  'fuel_tank', 'chassis_frame', 'steer_tires', 'drive_tires',
                  'cab_steps', 'roof_deflector'],
  exterior_rear: ['fifth_wheel', 'coupling_airlines', 'mudflaps_guards',
                  'chassis_frame', 'air_tanks_lines'],
  interior_cab: ['cab_interior_seats', 'bunk_sleeper', 'cab_floor_trim',
                 'steering_wheel_controls'],
  dashboard_odometer: ['dashboard_instruments', 'warning_lights',
                       'steering_wheel_controls'],
  tire_wheel: ['steer_tires', 'drive_tires', 'wheels_rims', 'brakes_hubs'],
  engine_bay: ['engine_bay'],
  chassis_undercarriage: ['chassis_frame', 'undercarriage', 'air_suspension',
                          'air_tanks_lines', 'adblue_tank', 'exhaust_dpf'],
  fifth_wheel: ['fifth_wheel', 'coupling_airlines'],
  /* A damage close-up establishes nothing on its own - it is a detail of
     whatever panel it was taken of, which the frame itself does not say. */
  damage_detail: [],
};

/* Three components are conditions rather than parts, and are observed on a
   panel that is already drawn. */
const ALIAS = {
  paint_finish: 'cab_exterior_panels',
  corrosion: 'chassis_frame',
  fluid_leaks: 'undercarriage',
};

const SEV_RANK = { cosmetic: 0, minor: 1, moderate: 2, major: 3 };

let source = null;

export async function load() {
  if (!source) {
    /* A frozen export embeds the drawing, because there is no server to fetch
       it from. */
    const text = window.KAMION_ELEVATION
      || await fetch('/static/assets/tractor-elevation.svg').then((r) => r.text());
    source = new DOMParser().parseFromString(text, 'image/svg+xml')
      .querySelector('svg');
  }
  return source;
}

/* Drop a fresh copy into a slot. Returns the <svg> so callers can address it. */
export async function mount(slot) {
  const src = await load();
  if (!src) return null;
  const copy = src.cloneNode(true);
  slot.replaceChildren(copy);
  return copy;
}

const zoneOf = (root, name) =>
  root.querySelector(`[data-component="${ALIAS[name] || name}"]`);

export function reset(root) {
  if (!root) return;
  root.querySelectorAll('.zone').forEach((z) => {
    delete z.dataset.state;
    delete z.dataset.sev;
    delete z.dataset.focus;
    z.classList.remove('lit');
  });
  root.querySelectorAll('.rev-mark').forEach((m) => m.remove());
}

/* Act 1: what the photographs actually cover. Staggered, because watching
   the drawing come up a zone at a time is the readable version of "eight
   distinct views". */
export function setCoverage(root, views, { stagger = 55 } = {}) {
  if (!root) return [];
  const zones = new Set();
  for (const v of views || []) for (const z of VIEW_ZONES[v] || []) zones.add(z);
  const nodes = [...zones].map((n) => zoneOf(root, n)).filter(Boolean);
  if (reduced()) {
    nodes.forEach((node) => { node.dataset.state = 'covered'; });
    return [...zones];
  }
  nodes.forEach((node, i) => {
    setTimeout(() => {
      node.dataset.state = 'covered';
      node.classList.add('lit');
      setTimeout(() => node.classList.remove('lit'), 750);
    }, i * stagger);
  });
  return [...zones];
}

/* Act 3: what the model found. Worst severity per zone wins, so one zone
   carrying a major and a cosmetic reads as major. */
export function setFindings(root, issues) {
  if (!root) return;
  const worst = new Map();
  for (const i of issues || []) {
    const key = ALIAS[i.component] || i.component;
    const prev = worst.get(key);
    if (!prev || (SEV_RANK[i.severity] ?? 0) > (SEV_RANK[prev.severity] ?? 0)) {
      worst.set(key, i);
    }
  }
  let n = 0;
  const marks = [];
  for (const [name, issue] of worst) {
    const node = zoneOf(root, name);
    if (!node) continue;
    node.dataset.state = 'read';
    node.dataset.sev = issue.severity;
    marks.push({ node, issue, n: ++n });
  }
  drawRevisionMarks(root, marks);
}

/* Every component the model wrote a summary line about is established even
   when it carried no finding - "checked, nothing flagged" is information. */
export function setSummarised(root, summary) {
  if (!root) return;
  for (const [key, text] of Object.entries(summary || {})) {
    if (!text) continue;
    /* condition_summary keys are the five report groups, not components, so
       promote every component in the group that is not already read. */
    for (const name of GROUP_ZONES[key] || []) {
      const node = zoneOf(root, name);
      if (node && node.dataset.state !== 'read') node.dataset.state = 'covered';
    }
  }
}

/* app/evidence.py builds condition_summary under these eight fixed keys. */
const GROUP_ZONES = {
  tires: ['steer_tires', 'drive_tires'],
  wheels_brakes: ['wheels_rims', 'brakes_hubs'],
  fifth_wheel_coupling: ['fifth_wheel', 'coupling_airlines'],
  chassis_corrosion: ['chassis_frame', 'undercarriage', 'air_suspension'],
  body_paint: ['cab_exterior_panels', 'front_bumper_valance', 'fairings_skirts',
               'doors_handles', 'roof_deflector'],
  cab_interior: ['cab_interior_seats', 'bunk_sleeper', 'cab_floor_trim',
                 'dashboard_instruments', 'steering_wheel_controls'],
  engine_driveline: ['engine_bay', 'exhaust_dpf', 'fuel_tank', 'adblue_tank'],
  glass_lights: ['windscreen_glass', 'grille_headlights', 'mirrors_visor'],
};

function drawRevisionMarks(root, marks) {
  root.querySelectorAll('.rev-mark').forEach((m) => m.remove());
  const placed = [];
  for (const { node, issue, n } of marks) {
    let box;
    try { box = node.getBBox(); } catch { continue; }
    /* A hidden section has no layout, so getBBox is all zeros - skip rather
       than pile every triangle on the origin. */
    if (!box || !box.width) continue;
    let x = box.x + box.width / 2;
    let y = box.y - 16;
    /* nudge up in rows until it clears the marks already down */
    let guard = 0;
    while (placed.some((p) => Math.abs(p.x - x) < 22 && Math.abs(p.y - y) < 18)
           && guard++ < 6) y -= 19;
    x = Math.max(14, Math.min(746, x));
    y = Math.max(14, y);
    placed.push({ x, y });

    const g = svg('g', { class: 'rev-mark in' });
    g.dataset.sev = issue.severity;
    g.style.animationDelay = `${0.4 + n * 0.08}s`;
    g.append(svg('path', { d: `M ${x} ${y - 9} L ${x + 9} ${y + 6} L ${x - 9} ${y + 6} Z` }));
    g.append(svgText({ x, y: y + 1 }, String(n)));
    const title = svg('title');
    title.textContent = `${titleise(issue.component)} — ${issue.severity}`;
    g.append(title);
    root.append(g);
  }
}

export function focus(root, component, on) {
  if (!root) return;
  const node = zoneOf(root, component);
  if (!node) return;
  if (on) node.dataset.focus = 'true';
  else delete node.dataset.focus;
}

export { VIEW_ZONES };
