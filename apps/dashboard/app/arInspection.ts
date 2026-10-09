import type { UploadedInspection } from './UploadedPanel';
export type ARInspection = UploadedInspection & {
  id: string; part_id: string; lot_id: string; machine_id: string; image_sha256: string;
  created_at: string; action: { status: string; text: string };
};
export async function inspectTrackedCrop(crop: HTMLCanvasElement, options: {
  model: 'casting' | 'neu'; patchcoreModel: string; machine: string;
}, signal: AbortSignal): Promise<{ inspection: ARInspection; image: HTMLImageElement }> {
  const blob = await new Promise<Blob>((resolve, reject) => crop.toBlob(value => value ? resolve(value) : reject(new Error('Camera crop could not be encoded.')), 'image/png'));
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())), byte => byte.toString(16).padStart(2, '0')).join('');
  const query = new URLSearchParams({ model: options.model, patchcore_model: options.patchcoreModel,
    machine_id: options.machine, input_source: 'camera', process_context: 'history' });
  const response = await fetch(`/api/inspections/upload?${query}`, { method: 'POST', body: blob,
    headers: { 'Content-Type': 'image/png' }, signal, cache: 'no-store' });
  const inspection = await response.json();
  if (!response.ok) throw new Error(typeof inspection.detail === 'string' ? inspection.detail : `Inspection failed (${response.status}).`);
  if (inspection.image_sha256 !== hash || inspection.context?.input_source !== 'camera' || inspection.context?.model !== options.model) {
    throw new Error('Inspection evidence does not match this camera crop and requested model.');
  }
  const image = new Image();
  const url = URL.createObjectURL(blob);
  try { image.src = url; await image.decode(); } finally { URL.revokeObjectURL(url); }
  return { inspection, image };
}
