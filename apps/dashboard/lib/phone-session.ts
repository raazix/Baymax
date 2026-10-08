import { createHmac, timingSafeEqual } from 'node:crypto';

export const PHONE_COOKIE = 'lineguard-phone-session';
const SESSION_SECONDS = 8 * 60 * 60;

export function phoneGateEnabled() { return Boolean(process.env.LINEGUARD_PHONE_ACCESS_CODE); }
export function pairingCodeMatches(code: unknown) {
  const expected = process.env.LINEGUARD_PHONE_ACCESS_CODE;
  return typeof code === 'string' && /^[0-9]{6}$/.test(code) && Boolean(expected) && code.length === expected!.length && timingSafeEqual(Buffer.from(code), Buffer.from(expected!));
}
function signature(value: string) {
  const secret = process.env.LINEGUARD_PHONE_SESSION_SECRET;
  if (!secret) throw new Error('Phone session secret is unavailable.');
  return createHmac('sha256', secret).update(value).digest('base64url');
}
export function createPhoneSession(now = Date.now()) {
  const expiry = String(Math.floor(now / 1000) + SESSION_SECONDS);
  return `${expiry}.${signature(expiry)}`;
}
export function validPhoneSession(value: string | undefined, now = Date.now()) {
  if (!value || !process.env.LINEGUARD_PHONE_SESSION_SECRET) return false;
  const [expiry, supplied, extra] = value.split('.');
  if (extra || !/^\d{10}$/.test(expiry) || !supplied || !/^[A-Za-z0-9_-]{43}$/.test(supplied) || Number(expiry) <= now / 1000 || Number(expiry) > now / 1000 + SESSION_SECONDS + 5) return false;
  const expected = signature(expiry);
  return supplied.length === expected.length && timingSafeEqual(Buffer.from(supplied), Buffer.from(expected));
}
