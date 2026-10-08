export type OverlayBox = { label: string; confidence: number; bbox_xyxy_px: number[] };
export type OverlayAnomaly = { grid: number[][]; threshold: number };
export type OverlayGeometry = { center_px: number[]; semi_axes_px: number[]; angle_deg: number };
export type OverlayMarker = { x: number; y: number; label: string };

// Fitted part outline, its centre, and a ray to each finding labelled with its polar position.
function drawGeometry(ctx: CanvasRenderingContext2D, width: number, geometry: OverlayGeometry, markers: OverlayMarker[]) {
  const [cx, cy] = geometry.center_px;
  const [a, b] = geometry.semi_axes_px;
  const line = Math.max(1.5, width / 400);
  const font = Math.max(11, Math.round(width / 46));
  ctx.save();
  ctx.lineWidth = line + 1.5; ctx.strokeStyle = 'rgba(255,255,255,.75)'; ctx.setLineDash([line * 6, line * 4]);
  ctx.beginPath(); ctx.ellipse(cx, cy, a, b, geometry.angle_deg * Math.PI / 180, 0, Math.PI * 2); ctx.stroke();
  ctx.lineWidth = line; ctx.strokeStyle = '#076d66'; ctx.stroke();
  ctx.setLineDash([]);
  const arm = Math.max(8, width / 60);
  ctx.lineWidth = line + 1.5; ctx.strokeStyle = 'rgba(255,255,255,.85)';
  ctx.beginPath(); ctx.moveTo(cx - arm, cy); ctx.lineTo(cx + arm, cy); ctx.moveTo(cx, cy - arm); ctx.lineTo(cx, cy + arm); ctx.stroke();
  ctx.lineWidth = line; ctx.strokeStyle = '#076d66'; ctx.stroke();
  ctx.font = `600 ${font}px sans-serif`; ctx.textBaseline = 'middle';
  for (const m of markers) {
    ctx.lineWidth = line + 1.5; ctx.strokeStyle = 'rgba(255,255,255,.85)';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(m.x, m.y); ctx.stroke();
    ctx.lineWidth = line; ctx.strokeStyle = '#203138'; ctx.stroke();
    ctx.fillStyle = '#203138'; ctx.beginPath(); ctx.arc(m.x, m.y, line * 2.2, 0, Math.PI * 2); ctx.fill();
    const tw = ctx.measureText(m.label).width + 10;
    const lx = Math.min(Math.max(m.x + 8, 2), width - tw - 2);
    const ly = Math.max(m.y - font - 6, font);
    ctx.fillStyle = 'rgba(32,49,56,.9)'; ctx.fillRect(lx, ly - font / 2 - 3, tw, font + 6);
    ctx.fillStyle = '#fff'; ctx.fillText(m.label, lx + 5, ly);
  }
  ctx.restore();
}

// Yellow near the threshold, through orange, to red at and above it.
function heatColor(t: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [[0, [250, 220, 70]], [.55, [245, 140, 40]], [1, [220, 40, 30]]];
  for (let i = 1; i < stops.length; i++) {
    const [p1, c1] = stops[i];
    const [p0, c0] = stops[i - 1];
    if (t <= p1) {
      const f = (t - p0) / (p1 - p0);
      return [0, 1, 2].map(k => Math.round(c0[k] + (c1[k] - c0[k]) * f)) as [number, number, number];
    }
  }
  return stops[stops.length - 1][1];
}

function drawHeatmap(ctx: CanvasRenderingContext2D, width: number, height: number, anomaly: OverlayAnomaly) {
  const rows = anomaly.grid.length;
  const cols = anomaly.grid[0]?.length ?? 0;
  if (!rows || !cols) return;
  const floor = anomaly.threshold * 0.9;
  const peakValue = Math.max(...anomaly.grid.flat());
  const top = Math.max(peakValue, anomaly.threshold * 1.15);
  // Paint one pixel per patch, then let the browser upsample smoothly.
  const small = document.createElement('canvas');
  small.width = cols; small.height = rows;
  const sctx = small.getContext('2d');
  if (!sctx) return;
  const pixels = sctx.createImageData(cols, rows);
  let peak = { value: -Infinity, x: 0, y: 0 };
  anomaly.grid.forEach((row, y) => row.forEach((value, x) => {
    if (value > peak.value) peak = { value, x, y };
    const i = (y * cols + x) * 4;
    if (value < floor) { pixels.data[i + 3] = 0; return; }
    const above = value > anomaly.threshold;
    const t = above ? 1 : (value - floor) / (anomaly.threshold - floor);
    const [r, g, b] = heatColor(Math.min(1, t));
    const strength = above ? .55 + .3 * Math.min(1, (value - anomaly.threshold) / Math.max(top - anomaly.threshold, 1e-6)) : .12 + .3 * t;
    pixels.data.set([r, g, b, Math.round(strength * 255)], i);
  }));
  sctx.putImageData(pixels, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, width, height);
  ctx.restore();
  // Ring the single most anomalous patch so the eye lands on it.
  const cx = (peak.x + .5) * width / cols;
  const cy = (peak.y + .5) * height / rows;
  const radius = Math.max(width / cols, height / rows) * 1.8;
  const line = Math.max(2, Math.round(width / 220));
  ctx.lineWidth = line + 2; ctx.strokeStyle = 'rgba(255,255,255,.9)';
  ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();
  ctx.lineWidth = line; ctx.strokeStyle = peak.value > anomaly.threshold ? '#c0281c' : '#b9770e';
  ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();
}

// Draws model output over the source image at its natural resolution (the canvas is scaled by CSS).
// With includeImage false the canvas holds only the model layer, so it can be animated over a separate image layer.
export function drawOverlay(canvas: HTMLCanvasElement, image: HTMLImageElement, boxes: OverlayBox[], anomaly?: OverlayAnomaly, includeImage = true, geometry?: OverlayGeometry | null, markers: OverlayMarker[] = []) {
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  if (includeImage) ctx.drawImage(image, 0, 0); else ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (anomaly) drawHeatmap(ctx, canvas.width, canvas.height, anomaly);
  if (geometry) drawGeometry(ctx, canvas.width, geometry, markers);
  const stroke = Math.max(2, Math.round(canvas.width / 200));
  const font = Math.max(12, Math.round(canvas.width / 40));
  ctx.lineWidth = stroke;
  ctx.font = `600 ${font}px sans-serif`;
  ctx.textBaseline = 'top';
  for (const d of boxes) {
    const [x1, y1, x2, y2] = d.bbox_xyxy_px;
    ctx.strokeStyle = '#e0452b';
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${d.label.replaceAll('_', ' ')} ${(d.confidence * 100).toFixed(0)}%`;
    const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = '#e0452b';
    ctx.fillRect(x1, Math.max(0, y1 - font - 4), w, font + 4);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, x1 + 4, Math.max(0, y1 - font - 4) + 2);
  }
}
