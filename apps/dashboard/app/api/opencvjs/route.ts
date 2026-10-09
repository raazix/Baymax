import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

export const runtime = 'nodejs';

export async function GET() {
  // The npm distribution omits video tracking algorithms. Use the official
  // OpenCV.js build so the AR feature tracker can use Lucas–Kanade optical flow.
  const source = await readFile(resolve(process.cwd(), 'public/vendor/opencv-4.13.0.js'));
  return new Response(source, {
    headers: {
      'Content-Type': 'text/javascript; charset=utf-8',
      'Cache-Control': 'public, max-age=31536000, immutable',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}
