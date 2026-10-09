export type OverlayBox = { label: string; confidence: number; bbox_xyxy_px: number[]; unconfirmed?: boolean };
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

function drawHeatmap(ctx: CanvasRenderingContext2D, width: number, height: number, anomaly: OverlayAnomaly, geometry?: OverlayGeometry | null) {
  const rows = anomaly.grid.length;
  const cols = anomaly.grid[0]?.length ?? 0;
  if (!rows || !cols) return;
  // A normal-only PatchCore threshold can sit above every patch in a good image.
  // Scale the visualization to this image's robust range while reserving red for threshold breaches.
  // When part geometry is reliable, suppress the surrounding scene so background texture is not highlighted.
  const insidePart = (x: number, y: number) => {
    if (!geometry) return true;
    const px = (x + .5) * width / cols - geometry.center_px[0];
    const py = (y + .5) * height / rows - geometry.center_px[1];
    const angle = geometry.angle_deg * Math.PI / 180;
    const u = (px * Math.cos(angle) + py * Math.sin(angle)) / geometry.semi_axes_px[0];
    const v = (-px * Math.sin(angle) + py * Math.cos(angle)) / geometry.semi_axes_px[1];
    return u * u + v * v <= 1.1025;
  };
  const values = anomaly.grid.flatMap((row, y) => row.filter((value, x) => Number.isFinite(value) && insidePart(x, y))).sort((a, b) => a - b);
  if (!values.length) return;
  const quantile = (q: number) => values[Math.min(values.length - 1, Math.floor((values.length - 1) * q))];
  const floor = quantile(.1);
  const top = Math.max(quantile(.95), floor + 1e-6);
  // Paint one pixel per patch, then let the browser upsample smoothly.
  const small = document.createElement('canvas');
  small.width = cols; small.height = rows;
  const sctx = small.getContext('2d');
  if (!sctx) return;
  const pixels = sctx.createImageData(cols, rows);
  let peak = { value: -Infinity, x: 0, y: 0 };
  anomaly.grid.forEach((row, y) => row.forEach((value, x) => {
    const i = (y * cols + x) * 4;
    if (!insidePart(x, y) || !Number.isFinite(value)) { pixels.data[i + 3] = 0; return; }
    if (value > peak.value) peak = { value, x, y };
    const above = value > anomaly.threshold;
    const t = above ? 1 : Math.max(0, Math.min(1, (value - floor) / (top - floor)));
    const [r, g, b] = heatColor(Math.min(1, t));
    const strength = above ? .62 + .25 * Math.min(1, (value - anomaly.threshold) / Math.max(top - anomaly.threshold, 1e-6)) : .14 + .34 * t;
    pixels.data.set([r, g, b, Math.round(strength * 255)], i);
  }));
  sctx.putImageData(pixels, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, width, height);
  ctx.restore();
  // Only ring a threshold breach. A merely highest-scoring normal patch is not a defect finding.
  if (peak.value <= anomaly.threshold) return;
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
  if (anomaly) drawHeatmap(ctx, canvas.width, canvas.height, anomaly, geometry);
  if (geometry) drawGeometry(ctx, canvas.width, geometry, markers);
  const stroke = Math.max(2, Math.round(canvas.width / 200));
  const font = Math.max(9, Math.round(Math.max(canvas.width, canvas.height) / 30));   // small images are scaled up on screen
  ctx.lineWidth = stroke;
  ctx.font = `600 ${font}px sans-serif`;
  ctx.textBaseline = 'top';
  for (const d of boxes) {
    const [x1, y1, x2, y2] = d.bbox_xyxy_px;
    const colour = d.unconfirmed ? '#5d6f73' : '#e0452b';
    ctx.strokeStyle = colour;
    ctx.setLineDash(d.unconfirmed ? [stroke * 4, stroke * 3] : []);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    const text = `${d.unconfirmed ? 'YOLO? ' : ''}${d.label.replaceAll('_', ' ')} ${(d.confidence * 100).toFixed(0)}%`;
    const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = colour;
    ctx.fillRect(x1, Math.max(0, y1 - font - 4), w, font + 4);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, x1 + 4, Math.max(0, y1 - font - 4) + 2);
  }
}
