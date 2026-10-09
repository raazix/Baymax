import type { PixelBox } from './bottleInference';
export type Point = { x: number; y: number };
export type Quad = [Point, Point, Point, Point];
export function boxQuad(box: PixelBox): Quad {
  return [{ x: box.x, y: box.y }, { x: box.x + box.width, y: box.y },
    { x: box.x + box.width, y: box.y + box.height }, { x: box.x, y: box.y + box.height }];
}
export function quadBounds(quad: Quad): PixelBox {
  const xs = quad.map(point => point.x), ys = quad.map(point => point.y);
  return { x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) };
}
export function overlap(a: PixelBox, b: PixelBox) {
  const intersection = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x))
    * Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return intersection / Math.max(a.width * a.height + b.width * b.height - intersection, 1);
}
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
// Robust 2D similarity from tracked feature pairs: translation, scale and in-plane rotation.
// Reject weak/implausible movement rather than attach old evidence to a different object.
export function featureMotion(before: Point[], after: Point[]) {
  if (before.length !== after.length || before.length < 8) return null;
  const dx = median(after.map((point, i) => point.x - before[i].x));
  const dy = median(after.map((point, i) => point.y - before[i].y));
  const deviations = after.map((point, i) => Math.hypot(point.x - before[i].x - dx, point.y - before[i].y - dy));
  const limit = Math.max(2, median(deviations) * 3);
  const keep = deviations.map((distance, i) => distance <= limit ? i : -1).filter(i => i >= 0);
  if (keep.length < 8 || keep.length / before.length < .6) return null;
  const mean = (points: Point[]) => ({ x: keep.reduce((sum, i) => sum + points[i].x, 0) / keep.length,
    y: keep.reduce((sum, i) => sum + points[i].y, 0) / keep.length });
  const p = mean(before), q = mean(after);
  let denominator = 0, dot = 0, cross = 0;
  for (const i of keep) {
    const px = before[i].x - p.x, py = before[i].y - p.y, qx = after[i].x - q.x, qy = after[i].y - q.y;
    denominator += px * px + py * py; dot += px * qx + py * qy; cross += px * qy - py * qx;
  }
  if (denominator < 16) return null;
  const a = dot / denominator, b = cross / denominator, scale = Math.hypot(a, b), angle = Math.atan2(b, a);
  if (scale < .85 || scale > 1.18 || Math.abs(angle) > .2 || !Number.isFinite(scale)) return null;
  const tx = q.x - a * p.x + b * p.y, ty = q.y - b * p.x - a * p.y;
  return { a, b, tx, ty, confidence: keep.length / before.length };
}
export function moveQuad(quad: Quad, motion: NonNullable<ReturnType<typeof featureMotion>>, sourceScale: number): Quad {
  return quad.map(point => ({ x: motion.a * point.x - motion.b * point.y + motion.tx * sourceScale,
    y: motion.b * point.x + motion.a * point.y + motion.ty * sourceScale })) as Quad;
}
