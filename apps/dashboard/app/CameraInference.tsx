'use client';

import { useEffect, useRef, useState } from 'react';
import { drawOverlay } from './overlay';

type Model = 'neu' | 'casting';
type Detection = { label: string; confidence: number; bbox_xyxy_px: number[] };
type YoloResult = { detections: Detection[]; model_sha256: string; note: string; image_sha256: string };
type AnomalyResult = { label: string; anomaly_score: number; threshold: number; anomaly_grid: number[][]; artifact_sha256: string; domain_note: string; image_sha256: string };
type Outcome = { model: Model; image: HTMLImageElement; yolo?: YoloResult; anomaly?: AnomalyResult; ms: number };

const MODELS: Record<Model, { name: string; profile: string; path: string; scope: string }> = {
  neu: { name: 'Steel-surface defects (YOLO11n)', profile: 'neu_proxy', path: 'detect', scope: 'NEU steel proxy · 6 defect classes' },
  casting: { name: 'Casting anomaly (PatchCore)', profile: 'casting_proxy', path: 'anomaly', scope: 'Casting proxy · normal vs anomalous' },
};

async function postJson<T>(url: string, blob?: Blob): Promise<T> {
  const response = await fetch(url, { method: 'POST', body: blob, headers: blob ? { 'Content-Type': blob.type || 'image/jpeg' } : undefined, cache: 'no-store' });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}).`);
  }
  return response.json();
}

function loadImage(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not read that image.')); };
    img.src = url;
  });
}

function draw(canvas: HTMLCanvasElement, outcome: Outcome) {
  const { image, yolo, anomaly } = outcome;
  drawOverlay(canvas, image, yolo?.detections ?? [], anomaly ? { grid: anomaly.anomaly_grid, threshold: anomaly.threshold } : undefined);
}

export default function CameraInference({ grabFrame, cameraActive, patchcoreModel = 'default' }: { grabFrame: () => Promise<Blob>; cameraActive: boolean; patchcoreModel?: string }) {
  const [model, setModel] = useState<Model>('neu');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => { if (outcome && canvas.current) draw(canvas.current, outcome); }, [outcome]);

  async function run(blob: Blob, fromWebcam: boolean) {
    setBusy(true); setError('');
    const started = performance.now();
    try {
      const spec = MODELS[model];
      const profile = fromWebcam ? 'camera' : spec.profile;
      const image = await loadImage(blob);
      const frame = await postJson<{ id: string; quality: { passed: boolean } }>(`/api/frames?quality_profile=${profile}`, blob);
      if (!frame.quality.passed) throw new Error('Image quality gate failed (too blurry, dark or glare). Recapture and retry.');
      const result = await postJson<YoloResult | AnomalyResult>(`/api/proxy/frames/${frame.id}/${spec.path}${model === 'casting' ? `?patchcore_model=${encodeURIComponent(patchcoreModel)}` : ''}`);
      setOutcome({ model, image, ms: performance.now() - started, ...(model === 'neu' ? { yolo: result as YoloResult } : { anomaly: result as AnomalyResult }) });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Inference failed.');
    } finally {
      setBusy(false);
    }
  }

  async function onWebcam() {
    try { await run(await grabFrame(), true); } catch (e) { setError(e instanceof Error ? e.message : 'Capture failed.'); }
  }

  function onFile(files: FileList | null) {
    const file = files?.[0];
    if (file) void run(file, false);
  }

  const anomaly = outcome?.anomaly;
  const flagged = anomaly ? anomaly.anomaly_score > anomaly.threshold : false;

  return <div className="inference">
    <h2>Run a model</h2>
    <p>Analyse a webcam frame or an image file with a trained proxy model. Results are not brake-disc validated.</p>
    <div className="buttons">
      <label className="model-pick">Model
        <select value={model} onChange={e => { setModel(e.target.value as Model); setOutcome(null); setError(''); }}>
          {(Object.keys(MODELS) as Model[]).map(key => <option key={key} value={key}>{MODELS[key].name}</option>)}
        </select>
      </label>
      <button className="primary" onClick={onWebcam} disabled={busy || !cameraActive}>Analyse webcam frame</button>
      <label className={`file-button${busy ? ' disabled' : ''}`}>Analyse image file<input type="file" accept="image/jpeg,image/png" disabled={busy} onChange={e => { onFile(e.target.files); e.target.value = ''; }} /></label>
    </div>
    <p className="muted">{MODELS[model].scope}</p>
    {busy && <p role="status">Running model…</p>}
    {error && <p className="inference-error" role="alert">{error}</p>}
    {outcome && <div className="inference-result">
      <canvas ref={canvas} role="img" aria-label={outcome.yolo ? `${outcome.yolo.detections.length} detections drawn on the image` : `Anomaly heatmap, ${flagged ? 'anomalous' : 'within normal range'}`} />
      <div>
        {outcome.yolo && <>
          <h3>{outcome.yolo.detections.length ? `${outcome.yolo.detections.length} candidate defect${outcome.yolo.detections.length === 1 ? '' : 's'}` : 'No defects detected'}</h3>
          <ul className="defect-list">{outcome.yolo.detections.map((d, i) => <li key={i}><i style={{ background: '#e0452b' }} /><div><strong>{d.label.replaceAll('_', ' ')}</strong><span>confidence {(d.confidence * 100).toFixed(1)}% (uncalibrated)</span></div></li>)}</ul>
          <p className="footnote">{outcome.yolo.note}</p>
        </>}
        {anomaly && <>
          <h3 className={flagged ? 'flag' : ''}>{flagged ? 'Anomalous' : 'Within normal range'}</h3>
          <dl className="measurements"><div><dt>Anomaly score</dt><dd>{anomaly.anomaly_score.toFixed(3)}</dd></div><div><dt>Threshold</dt><dd>{anomaly.threshold.toFixed(3)}</dd></div></dl>
          <p className="footnote">Score is a feature distance, not a probability. Red cells exceed the threshold; amber cells are close. {anomaly.domain_note}</p>
        </>}
        <p className="footnote">Inference round trip {Math.round(outcome.ms)} ms including upload and quality gate. Image SHA-256 <span className="hash">{(outcome.yolo ?? anomaly)!.image_sha256.slice(0, 16)}…</span></p>
      </div>
    </div>}
  </div>;
}
