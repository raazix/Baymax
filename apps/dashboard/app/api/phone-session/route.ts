import { NextRequest, NextResponse } from 'next/server';
import { createPhoneSession, pairingCodeMatches, PHONE_COOKIE, phoneGateEnabled } from '../../../lib/phone-session';

export const runtime = 'nodejs';
const attempts = new Map<string, { count: number; until: number }>();

export async function POST(request: NextRequest) {
  if (!phoneGateEnabled()) return NextResponse.json({ detail: 'Phone pairing is not enabled on this server.' }, { status: 503 });
  if (request.headers.get('sec-fetch-site') === 'cross-site') return NextResponse.json({ detail: 'Open pairing from the workspace link.' }, { status: 403 });
  const address = request.headers.get('cf-connecting-ip') || request.headers.get('x-forwarded-for')?.split(',')[0] || 'local';
  const now = Date.now();
  const attempt = attempts.get(address);
  if (attempt && attempt.until > now && attempt.count >= 8) return NextResponse.json({ detail: 'Too many attempts. Wait one minute and retry.' }, { status: 429 });
  if (!attempt || attempt.until <= now) { if (attempts.size > 1000) attempts.clear(); attempts.set(address, { count: 0, until: now + 60000 }); }
  const entry = attempts.get(address)!; entry.count += 1;
  let payload: { code?: unknown };
  try { const text = await request.text(); if (text.length > 100) throw new Error(); payload = JSON.parse(text); }
  catch { return NextResponse.json({ detail: 'Enter the six-digit pairing code.' }, { status: 400 }); }
  if (!pairingCodeMatches(payload?.code)) return NextResponse.json({ detail: 'That code does not match. Check the laptop pairing panel.' }, { status: 401 });
  attempts.delete(address);
  const response = NextResponse.json({ paired: true }, { headers: { 'Cache-Control': 'no-store' } });
  response.cookies.set(PHONE_COOKIE, createPhoneSession(), { httpOnly: true, secure: true, sameSite: 'lax', path: '/', maxAge: 8 * 60 * 60 });
  return response;
}
