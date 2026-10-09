import type { NextConfig } from 'next';
const config: NextConfig = {
  devIndicators: false,
  // The phone tunnel and Vercel builds can coexist without replacing one
  // another's manifests or static asset hashes.
  distDir: process.env.LINEGUARD_PHONE_DIST === '1' ? '.next-phone' : '.next',
};
export default config;
