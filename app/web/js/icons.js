/* Bespoke 32px technical glyphs. Local SVG, also embedded by frozen exports. */
import { svg } from './dom.js';
const paths = {
 truck:'M3 23h2m6 0h10m6 0h2V13l-5-2-3-7H12v19M13 7h7l3 7H13z M5 20h6v7H5zM21 20h6v7h-6z',
 tire:'M11 3h10l4 5v16l-4 5H11l-4-5V8zM13 7h6v18h-6zM8 10h4m-4 6h4m-4 6h4m8-12h4m-4 6h4m-4 6h4',
 wheel:'M16 3a13 13 0 1 0 0 26 13 13 0 0 0 0-26M16 10a6 6 0 1 0 0 12 6 6 0 0 0 0-12M16 3v7m0 12v7M3 16h7m12 0h7',
 cab:'M7 28V7l4-4h13l3 6v19zM11 8h12v9H11zM12 21h5M5 28h24M21 22v6',
 glass:'M6 5h20l3 20H3zM8 21l8-7m8 7-8-7M10 9h5m4 0h3',
 light:'M5 9h9v14H5l-3-7zM18 9h10m-10 7h12m-12 7h10',
 mirror:'M5 4h11v16H5zM16 12h6v16h5M8 7v9',
 tank:'M7 6h18l3 4v14l-3 3H7l-3-3V10zM10 6v21M22 6v21M13 3h6v3',
 chassis:'M5 3v26m22-26v26M5 8h22M5 16h22M5 24h22M2 6h6m16 0h6M2 26h6m16 0h6',
 coupling:'M5 9l6-5h10l6 5v11l-7 7h-8l-7-7zM13 27V15h6v12M5 12h7m8 0h7',
 hose:'M6 3v6c0 5 20 1 20 6s-20 1-20 6 20 1 20 6v2M3 3h6m14 26h6',
 seat:'M8 3h9l2 15h7v7H8zM8 25v4m17-4v4M4 9v13h4',
 gauge:'M4 25a14 14 0 1 1 24 0zM16 21l7-10M8 17H5m22 0h-3M16 7v4',
 engine:'M8 8h13v4h5v-3h3v17h-3v-4h-5v4H7v-5H3V11h5zM11 4h8m-4 0v4M12 14l6 3-6 3',
 spring:'M9 3h14L9 8l14 5-14 5 14 5-14 6h14M5 3h22M5 29h22',
 drop:'M16 3S5 15 5 21a11 11 0 0 0 22 0C27 15 16 3 16 3zM10 20c0 4 2 6 5 6',
 camera:'M3 9h7l3-5h8l3 5h5v19H3zM16 12a6 6 0 1 0 0 12 6 6 0 0 0 0-12',
 paint:'M7 3h18v10H7zM25 8h4v10H16v4M13 22h6v7h-6z',
 check:'M6 16l7 7L27 7',
 search:'M13 3a10 10 0 1 0 0 20 10 10 0 0 0 0-20M21 21l8 8',
};
export function iconKind(part='') {
 const p=part.toLowerCase();
 if (/tire/.test(p)) return 'tire';
 if (/fifth|coupl/.test(p)) return /air|line/.test(p)?'hose':'coupling';
 if (/wheel|rim|brake|hub/.test(p)) return 'wheel';
 if (/glass|windscreen/.test(p)) return 'glass';
 if (/mirror|visor/.test(p)) return 'mirror';
 if (/light|grille/.test(p)) return 'light';
 if (/tank|adblue/.test(p)) return 'tank';
 if (/suspension/.test(p)) return 'spring';
 if (/chassis|under|corrosion/.test(p)) return 'chassis';
 if (/airline|air_tank|exhaust/.test(p)) return 'hose';
 if (/engine/.test(p)) return 'engine';
 if (/leak/.test(p)) return 'drop';
 if (/seat|bunk|interior|floor/.test(p)) return 'seat';
 if (/dashboard|odometer|instrument|control/.test(p)) return 'gauge';
 if (/paint/.test(p)) return 'paint';
 if (/cab|door|roof|bumper|fairing|mudflap|step|panel/.test(p)) return 'cab';
 return paths[p] ? p : 'truck';
}
export function partIcon(part) {
 const kind=iconKind(part);
 const root=svg('svg',{viewBox:'0 0 32 32',fill:'none',stroke:'currentColor','stroke-width':'1.5','stroke-linecap':'round','stroke-linejoin':'round','aria-hidden':'true',class:'part-icon','data-icon':kind});
 root.append(svg('path',{d:paths[kind]}));
 return root;
}
