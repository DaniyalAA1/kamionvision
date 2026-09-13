/* Event-driven inspection cards. No invented progress or diagnostic claims. */
import { $, el, viewName, reduced, titleise } from './dom.js';
import { urlFor, citePhoto } from './frames.js';
import { partIcon } from './icons.js';
const SEV_RANK = { major: 3, moderate: 2, minor: 1, cosmetic: 0 };
let expected = 0, landed = 0;
export function clear() { expected = 0; landed = 0; $('thoughts').replaceChildren(); }
export function begin(total) {
 clear(); expected = total;
 const pending = el('div', 'thought-pending');
 pending.append(partIcon('camera'), el('span', null, 'Reading the details in your photos…'));
 $('thoughts').append(pending);
 $('rail-title').textContent = 'Inspecting your truck';
 $('rail-sub').textContent = `${total} photos queued · results arrive as they finish`;
}
export function note(message) {
 $('thoughts').querySelectorAll('.thought-pending').forEach(n=>n.remove());
 $('thoughts').append(el('p', 'thought-quiet', message));
}
export function add(finding) {
 const rail = $('thoughts');
 const atTop = rail.scrollTop < 12;
 const previousHeight = rail.scrollHeight;
 rail.querySelectorAll('.thought-pending').forEach(n=>n.remove());
 const issues = [...(finding.issues || [])].sort((a,b)=>(SEV_RANK[b.severity]??0)-(SEV_RANK[a.severity]??0));
 const card = el('details', 'thought');
 card.dataset.worst = finding.error ? 'error' : issues[0]?.severity || 'clean';
 // Keep arriving work compact: reading an open explanation is never interrupted.
 const summary = el('summary', 'thought-summary');
 const thumb = el('img', 'thought-thumb'); thumb.src = urlFor(finding.photo_id); thumb.alt = ''; thumb.loading = 'lazy';
 const heading = el('span', 'thought-heading');
 heading.append(el('strong', null, viewName(finding.view)), el('span', 'thought-status', finding.error ? 'Could not read this photo' : issues.length ? `${issues.length} observation${issues.length===1?'':'s'} · ${issues[0].severity}` : 'No issues flagged in this photo'));
 summary.append(thumb, heading, el('span','thought-chevron','+'));
 card.append(summary);
 const body = el('div', 'thought-body');
 if (finding.shows) body.append(el('p','thought-shows',finding.shows));
 if (finding.cropped) body.append(el('p','thought-quiet','Cropped to the truck for inspection'));
 const list = el('ul','thought-list');
 issues.forEach(issue=>{
  const row=el('li'); row.dataset.sev=issue.severity;
  const copy=el('div'); copy.append(el('strong',null,titleise(issue.component)),el('p',null,issue.observation));
  row.append(partIcon(issue.component),copy); list.append(row);
 });
 (finding.strengths||[]).forEach(good=>{const row=el('li');row.dataset.sev='ok';row.append(partIcon('check'),el('p',null,good));list.append(row);});
 body.append(list);
 const source=el('button','thought-source','Open source photo ↗'); source.type='button';source.addEventListener('click',()=>citePhoto(finding.photo_id));body.append(source);card.append(body);
 // Small, distinct component tags are visible without expanding the prose.
 const tags=el('span','thought-parts');
 [...new Set(issues.map(i=>i.component))].slice(0,3).forEach(part=>{const tag=el('span','part-tag');tag.append(partIcon(part),el('span',null,titleise(part)));tags.append(tag);});
 if(tags.children.length) heading.append(tags);
 rail.prepend(card);
 if(!atTop) rail.scrollTop += rail.scrollHeight - previousHeight;
 if(!reduced()) card.classList.add('resolve');
 landed++;
 $('rail-sub').textContent=`${landed}${expected?` of ${expected}`:''} photos processed · expand any card`;
 return card;
}
export function done(evidence) {
 $('thoughts').querySelectorAll('.thought-pending').forEach(n=>n.remove());
 if(!evidence){$('rail-title').textContent='Photo checks complete';$('rail-sub').textContent='No visual appraisal was produced';return;}
 const count=(evidence.issues||[]).length;
 $('rail-title').textContent=count?`${count} observation${count===1?'':'s'} to review`:'Photo review complete';
 $('rail-sub').textContent=`${evidence.photos_read} photos read${evidence.photos_failed?` · ${evidence.photos_failed} unreadable`:''} · photo evidence, not a mechanical inspection`;
}
