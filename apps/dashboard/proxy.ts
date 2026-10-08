import { NextRequest, NextResponse } from 'next/server';
import { PHONE_COOKIE, phoneGateEnabled, validPhoneSession } from './lib/phone-session';

export function proxy(request: NextRequest) {
  const path = request.nextUrl.pathname;
  if (!phoneGateEnabled() || path === '/phone-access' || path === '/api/phone-session' || validPhoneSession(request.cookies.get(PHONE_COOKIE)?.value)) return NextResponse.next();
  if (path.startsWith('/api/')) return NextResponse.json({ detail: 'Pair this phone with the workspace first.' }, { status: 401, headers: { 'Cache-Control': 'no-store' } });
  const destination = new URL('/phone-access', request.url);
  const publicHost = request.headers.get('x-forwarded-host')?.split(',')[0] || request.headers.get('host');
  if (publicHost && /^[a-z0-9-]+\.trycloudflare\.com$/.test(publicHost)) { destination.host = publicHost; destination.port = ''; destination.protocol = 'https:'; }
  return NextResponse.redirect(destination);
}
export const config = { matcher: ['/((?!_next/static|_next/image|icon.svg|favicon.ico).*)'] };
