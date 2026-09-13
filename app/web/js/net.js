/* Everything that talks to the server.

   The stream carries unnamed SSE frames discriminated by a `type` field:
   `gate` (the finished gate report, about a second in), `stage`, `photo` (one
   per photo, as its own vision call returns, out of order), `result` and
   `error`. */

export const getHealth  = () => fetch('/api/health').then((r) => r.json());
export const getGallery = () => fetch('/api/gallery').then((r) => r.json());

async function post(url, body) {
  const r = await fetch(url, { method: 'POST', body });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `upload failed (${r.status})`);
  return data;
}

export function uploadFiles(files) {
  const body = new FormData();
  [...files].forEach((f) => body.append('files', f));
  return post('/api/upload', body);
}

export function uploadSample(id) {
  const body = new FormData();
  body.set('case', id);
  return post('/api/upload-sample', body);
}

export function uploadTruck(id) {
  const body = new FormData();
  body.set('truck', id);
  return post('/api/upload-truck', body);
}

export function openStream(session, params, handlers) {
  const stream = new EventSource(`/api/appraise/${session}?${params}`);
  stream.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    const fn = handlers[msg.type];
    if (fn) fn(msg);
  };
  stream.onerror = () => stream.close();
  return stream;
}
