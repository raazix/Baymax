export type PixelBox = { x: number; y: number; width: number; height: number };
export type BottleAnomaly = {
  anomaly_grid: number[][]; threshold: number; anomaly_score: number;
  image_sha256: string; frame_id: string; patchcore_model: string;
};

export function clampCrop(box: PixelBox, width: number, height: number): PixelBox {
  const x = Math.max(0, Math.min(width - 1, Math.floor(box.x)));
  const y = Math.max(0, Math.min(height - 1, Math.floor(box.y)));
  const right = Math.max(x + 1, Math.min(width, Math.ceil(box.x + box.width)));
  const bottom = Math.max(y + 1, Math.min(height, Math.ceil(box.y + box.height)));
  return { x, y, width: right - x, height: bottom - y };
}

export function coverTransform(width: number, height: number, viewportWidth: number, viewportHeight: number) {
  const scale = Math.max(viewportWidth / width, viewportHeight / height);
  return { scale, x: (viewportWidth - width * scale) / 2, y: (viewportHeight - height * scale) / 2 };
}

export function validateAnomaly(result: BottleAnomaly, imageHash: string) {
  if (result.image_sha256 !== imageHash) throw new Error('Heatmap image hash does not match the captured crop.');
  const columns = result.anomaly_grid?.[0]?.length;
  if (!columns || !result.anomaly_grid.every(row => row.length === columns && row.every(Number.isFinite))
      || !Number.isFinite(result.threshold) || result.threshold <= 0 || !Number.isFinite(result.anomaly_score)) {
    throw new Error('The model returned an invalid anomaly grid.');
  }
  return result;
}

async function post(url: string, signal: AbortSignal, blob?: Blob) {
  const response = await fetch(url, { method: 'POST', signal, body: blob,
    headers: blob ? { 'Content-Type': blob.type } : undefined, cache: 'no-store' });
  const payload = await response.json();
  if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : `Inference request failed (${response.status}).`);
  return payload;
}

export async function inspectBottleCrop(crop: HTMLCanvasElement, model: string, signal: AbortSignal): Promise<BottleAnomaly> {
  const blob = await new Promise<Blob>((resolve, reject) => crop.toBlob(value => value ? resolve(value) : reject(new Error('Camera crop could not be encoded.')), 'image/png'));
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  const frame = await post('/api/frames?quality_profile=camera', signal, blob);
  if (frame.sha256 !== hash) throw new Error('Stored camera crop does not match the captured image.');
  if (!frame.quality?.passed) throw new Error('Camera crop failed the quality gate. Improve focus, lighting or reflections.');
  const result = await post(`/api/proxy/frames/${encodeURIComponent(frame.id)}/anomaly?patchcore_model=${encodeURIComponent(model)}`, signal);
  if (result.frame_id !== frame.id || result.patchcore_model !== model) throw new Error('Model result does not match the requested frame and model.');
  return validateAnomaly(result, hash);
}

export function anomalyLayer(result: BottleAnomaly): HTMLCanvasElement {
  const grid = result.anomaly_grid;
  const layer = document.createElement('canvas');
  layer.width = grid[0].length; layer.height = grid.length;
  const ctx = layer.getContext('2d');
  if (!ctx) throw new Error('Heatmap canvas is unavailable.');
  const pixels = ctx.createImageData(layer.width, layer.height);
  const range = Math.max(result.threshold * 0.3, 1e-6);
  grid.forEach((row, y) => row.forEach((value, x) => {
    // Below-threshold patches are transparent: no invented blue/amber findings.
    if (value <= result.threshold) return;
    const strength = Math.min(1, (value - result.threshold) / range);
    pixels.data.set([255, Math.round(145 * (1 - strength)), 40, Math.round(90 + strength * 90)], (y * layer.width + x) * 4);
  }));
  ctx.putImageData(pixels, 0, 0);
  return layer;
}
