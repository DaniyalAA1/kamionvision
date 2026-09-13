/* Shop-manual glyphs for the closed component vocabulary.

   The result screen used to be a wall of observations. Each finding and each
   system tile now carries a 32px technical drawing of the part, so you can
   scan "steer tires / fifth wheel / bunk" the way a workshop wall is scanned,
   and open the prose only when you want it. The drawings are local SVG, and
   the frozen export inlines this module with the rest of the graph. */

import { svg } from './dom.js';

/* One path per silhouette, 32×32, stroke only. Keep them distinct at 16px:
   a drive tire and a rim and a steering wheel are three different objects. */
export const PATHS = {
  truck: 'M3 23h2m6 0h10m6 0h2V13l-5-2-3-7H12v19M13 7h7l3 7H13zM5 20h6v7H5zM21 20h6v7h-6z',
  tire: 'M11 3h10l4 5v16l-4 5H11l-4-5V8zM13 7h6v18h-6zM8 10h4m-4 6h4m-4 6h4m8-12h4m-4 6h4m-4 6h4',
  wheel: 'M16 3a13 13 0 1 0 0 26 13 13 0 0 0 0-26M16 10a6 6 0 1 0 0 12 6 6 0 0 0 0-12M16 3v7m0 12v7M3 16h7m12 0h7',
  brake: 'M16 4a12 12 0 1 0 0 24 12 12 0 0 0 0-24M16 10a6 6 0 1 0 0 12 6 6 0 0 0 0-12M26 9l4 2v10l-4 2',
  cab: 'M7 28V7l4-4h13l3 6v19zM11 8h12v9H11zM12 21h5M5 28h24M21 22v6',
  glass: 'M6 5h20l3 20H3zM8 21l8-7m8 7-8-7M10 9h5m4 0h3',
  light: 'M5 9h9v14H5l-3-7zM18 9h10m-10 7h12m-12 7h10',
  mirror: 'M5 4h11v16H5zM16 12h6v16h5M8 7v9',
  tank: 'M7 6h18l3 4v14l-3 3H7l-3-3V10zM10 6v21M22 6v21M13 3h6v3',
  chassis: 'M5 3v26m22-26v26M5 8h22M5 16h22M5 24h22M2 6h6m16 0h6M2 26h6m16 0h6',
  coupling: 'M5 9l6-5h10l6 5v11l-7 7h-8l-7-7zM13 27V15h6v12M5 12h7m8 0h7',
  hose: 'M6 3v6c0 5 20 1 20 6s-20 1-20 6 20 1 20 6v2M3 3h6m14 26h6',
  seat: 'M8 3h9l2 15h7v7H8zM8 25v4m17-4v4M4 9v13h4',
  gauge: 'M4 25a14 14 0 1 1 24 0zM16 21l7-10M8 17H5m22 0h-3M16 7v4',
  engine: 'M8 8h13v4h5v-3h3v17h-3v-4h-5v4H7v-5H3V11h5zM11 4h8m-4 0v4M12 14l6 3-6 3',
  spring: 'M9 3h14L9 8l14 5-14 5 14 5-14 6h14M5 3h22M5 29h22',
  drop: 'M16 3S5 15 5 21a11 11 0 0 0 22 0C27 15 16 3 16 3zM10 20c0 4 2 6 5 6',
  camera: 'M3 9h7l3-5h8l3 5h5v19H3zM16 12a6 6 0 1 0 0 12 6 6 0 0 0 0-12',
  paint: 'M7 3h18v10H7zM25 8h4v10H16v4M13 22h6v7h-6z',
  check: 'M6 16l7 7L27 7',
  search: 'M13 3a10 10 0 1 0 0 20 10 10 0 0 0 0-20M21 21l8 8',
  bumper: 'M4 14h24l2 6H2zM8 20v5M24 20v5M12 8h8v6',
  fairing: 'M6 6c10 0 20 4 22 14v6H8c-4-4-4-12-2-20zM6 26h22',
  deflector: 'M4 22l6-14h16l2 6H12l-4 8zM4 22h24',
  door: 'M8 3h16v26H8zM11 8h10v10H11zM21 20h2',
  step: 'M6 26h20M10 20h16M14 14h12M6 26V20h4v6m0-6V14h4v6',
  exhaust: 'M14 4h4v18l-2 6-2-6zM10 10h12M11 16h10',
  rust: 'M6 8h8v6H6zM14 16h10v7H14zM8 22h6v5H8zM20 6h6v8h-6z',
  bunk: 'M4 18h24v8H4zM6 18V10h10v8M4 26h24',
  floor: 'M5 7h22M5 13h22M5 19h22M5 25h22M9 7v18M17 7v18M25 7v18',
  warning: 'M16 4L30 28H2zM16 13v8m0 3v2',
  mudflap: 'M10 4h12v6H10zM12 10h8v18l-4-3-4 3z',
  steer: 'M16 3a13 13 0 1 0 0 26 13 13 0 0 0 0-26M16 12a4 4 0 1 0 0 8 4 4 0 0 0 0-8M16 3v9M6 22l7-4M26 22l-7-4',
};

/* Closed `COMPONENTS` enum from app/evidence/prompts.py. A typo here is a
   finding that silently draws the generic truck instead of its part. */
export const COMPONENT_ICONS = {
  steer_tires: 'tire',
  drive_tires: 'tire',
  wheels_rims: 'wheel',
  brakes_hubs: 'brake',
  fifth_wheel: 'coupling',
  coupling_airlines: 'hose',
  chassis_frame: 'chassis',
  undercarriage: 'chassis',
  air_suspension: 'spring',
  air_tanks_lines: 'hose',
  mudflaps_guards: 'mudflap',
  cab_exterior_panels: 'cab',
  front_bumper_valance: 'bumper',
  fairings_skirts: 'fairing',
  grille_headlights: 'light',
  mirrors_visor: 'mirror',
  roof_deflector: 'deflector',
  windscreen_glass: 'glass',
  doors_handles: 'door',
  cab_steps: 'step',
  fuel_tank: 'tank',
  adblue_tank: 'tank',
  exhaust_dpf: 'exhaust',
  engine_bay: 'engine',
  fluid_leaks: 'drop',
  paint_finish: 'paint',
  corrosion: 'rust',
  cab_interior_seats: 'seat',
  steering_wheel_controls: 'steer',
  dashboard_instruments: 'gauge',
  bunk_sleeper: 'bunk',
  cab_floor_trim: 'floor',
  warning_lights: 'warning',
};

export const SUMMARY_ICONS = {
  tires: 'tire',
  wheels_brakes: 'brake',
  fifth_wheel_coupling: 'coupling',
  chassis_corrosion: 'chassis',
  body_paint: 'paint',
  cab_interior: 'seat',
  engine_driveline: 'engine',
  glass_lights: 'glass',
};

export function iconKind(part = '') {
  const p = String(part).toLowerCase();
  if (PATHS[p]) return p;
  if (COMPONENT_ICONS[p]) return COMPONENT_ICONS[p];
  if (SUMMARY_ICONS[p]) return SUMMARY_ICONS[p];
  const slug = p.replace(/[\s-]+/g, '_');
  if (COMPONENT_ICONS[slug]) return COMPONENT_ICONS[slug];
  if (SUMMARY_ICONS[slug]) return SUMMARY_ICONS[slug];
  return 'truck';
}

export function partIcon(part) {
  const kind = iconKind(part);
  const root = svg('svg', {
    viewBox: '0 0 32 32',
    fill: 'none',
    stroke: 'currentColor',
    'stroke-width': '1.5',
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    'aria-hidden': 'true',
    focusable: 'false',
    class: 'part-icon',
    'data-icon': kind,
  });
  root.append(svg('path', { d: PATHS[kind] }));
  return root;
}
