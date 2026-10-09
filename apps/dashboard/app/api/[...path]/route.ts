export const runtime = 'nodejs';
const MAX_BODY = 11 * 1024 * 1024;

async function forward(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  if (path.some(segment => segment === '.' || segment === '..' || segment.includes('\\'))) return Response.json({ detail: 'Invalid API path.' }, { status: 400 });
  const backend = (process.env.BACKEND_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const url = `${backend}/api/${path.map(encodeURIComponent).join('/')}${new URL(request.url).search}`;
  const headers = new Headers();
  for (const name of ['content-type', 'authorization', 'accept']) { const value = request.headers.get(name); if (value) headers.set(name, value); }
  const timeout = AbortSignal.timeout(path.at(-1) === 'train' ? 600000 : 120000);
  const signal = AbortSignal.any([request.signal, timeout]);
  try {
    // Explicit bytes give JSON, camera PNG and audio a consistent Content-Length
    // rather than the rewrite proxy's intermittent chunked request stalls.
    let body: Uint8Array<ArrayBuffer> | undefined;
    if (request.body && request.method !== 'GET' && request.method !== 'HEAD') {
      const reader = request.body.getReader(); const chunks: Uint8Array[] = []; let size = 0;
      while (true) {
        const chunk = await reader.read(); if (chunk.done) break;
        size += chunk.value.byteLength;
        if (size > MAX_BODY) { await reader.cancel(); return Response.json({ detail: 'Request exceeds the 11 MiB forwarding limit.' }, { status: 413 }); }
        chunks.push(chunk.value);
      }
      body = new Uint8Array(size); let offset = 0;
      for (const chunk of chunks) { body.set(chunk, offset); offset += chunk.byteLength; }
    }
    const response = await fetch(url, { method: request.method, headers, body, cache: 'no-store', redirect: 'manual', signal });
    const outgoing = new Headers({ 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    for (const name of ['content-type', 'content-disposition', 'x-request-id', 'www-authenticate']) { const value = response.headers.get(name); if (value) outgoing.set(name, value); }
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json') || contentType.includes('+json')) {
      const payload = await response.text();
      try { JSON.parse(payload); }
      catch { return Response.json({ detail: 'FastAPI returned an incomplete or invalid JSON response. Retry the request; if it continues, check the backend log.' }, { status: 502 }); }
      return new Response(payload, { status: response.status, headers: outgoing });
    }
    return new Response(response.body, { status: response.status, headers: outgoing });
  } catch {
    return Response.json({ detail: timeout.aborted ? 'FastAPI request timed out. Check the saved inspection before retrying a decision.' : 'Cannot reach FastAPI. Check the backend on port 8000.' }, { status: timeout.aborted ? 504 : 502 });
  }
}

export const GET = forward;
export const POST = forward;
export const HEAD = forward;
