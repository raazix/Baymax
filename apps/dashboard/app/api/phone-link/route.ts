import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

export const runtime = 'nodejs';
export async function GET() {
  try {
    const pairing = JSON.parse(await readFile(resolve(process.cwd(), '../../data/phone-demo.json'), 'utf8'));
    if (!/^https:\/\/[^/]+\.trycloudflare\.com$/.test(pairing.url) || Date.parse(pairing.expires_at) <= Date.now()) throw new Error();
    return Response.json({ ready: true, url: pairing.url, code: pairing.code, expires_at: pairing.expires_at }, { headers: { 'Cache-Control': 'no-store' } });
  } catch { return Response.json({ ready: false }, { headers: { 'Cache-Control': 'no-store' } }); }
}
