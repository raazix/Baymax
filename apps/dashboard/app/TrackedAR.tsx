'use client';
import { useEffect, useRef, useState } from 'react';
import { X, ScanLine, AlertTriangle, Layers, RotateCcw, Mic, MicOff, Camera } from 'lucide-react';
import { sourceCrop, coverTransform, type PixelBox } from './bottleInference';
import { boxQuad, quadBounds, overlap, type Quad } from './arTracking';
import { createAROverlay } from './arWebGL';
import { inspectTrackedCrop, type ARInspection } from './arInspection';
import { drawOverlay, type OverlayBox } from './overlay';
import { holdLevel } from './qualityHold';

type View = 'patchcore' | 'yolo' | 'corrosion';
type SpeechRecognitionAlternativeLike = { transcript: string };
type SpeechRecognitionResultLike = ArrayLike<SpeechRecognitionAlternativeLike> & { isFinal: boolean };
type SpeechRecognitionLike = {
  lang: string; continuous: boolean; interimResults: boolean; maxAlternatives: number;
  onresult: ((event: { results: ArrayLike<SpeechRecognitionResultLike> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null; onend: (() => void) | null;
  start(): void; stop(): void; abort(): void;
};
type SpeechWindow = Window & { SpeechRecognition?: new () => SpeechRecognitionLike; webkitSpeechRecognition?: new () => SpeechRecognitionLike };
function borderMedian(data: Uint8ClampedArray, width: number, height: number) {
  const values: number[] = [], band = Math.max(2, Math.round(Math.min(width, height) * .04));
  for (let y = 0; y < height; y += 2) for (let x = 0; x < width; x += 2) {
    if (x < band || y < band || x >= width - band || y >= height - band) {
      const i = (y * width + x) * 4; values.push(.299 * data[i] + .587 * data[i + 1] + .114 * data[i + 2]);
    }
  }
  values.sort((a, b) => a - b); return values[Math.floor(values.length / 2)] ?? 128;
}
function silhouetteBox(data: Uint8ClampedArray, width: number, height: number): PixelBox | null {
  const size = width * height, background = borderMedian(data, width, height), mask = new Uint8Array(size);
  const gray = new Uint8Array(size);
  for (let i = 0, p = 0; i < size; i++, p += 4) gray[i] = .299 * data[p] + .587 * data[p + 1] + .114 * data[p + 2];
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    const i = y * width + x, value = gray[i];
    if (Math.abs(value - background) > 22 || Math.abs(value - gray[i + 1]) > 34 || Math.abs(value - gray[i + width]) > 34) mask[i] = 1;
  }
  // One inexpensive close pass connects the silhouette edges without the
  // startup cost and per-frame work of a full OpenCV.js runtime on phones.
  const dilated = new Uint8Array(size), closed = new Uint8Array(size);
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    const i = y * width + x;
    if (mask[i] || mask[i - 1] || mask[i + 1] || mask[i - width] || mask[i + width]) dilated[i] = 1;
  }
  for (let y = 2; y < height - 2; y++) for (let x = 2; x < width - 2; x++) {
    const i = y * width + x;
    if (dilated[i] && dilated[i - 1] && dilated[i + 1] && dilated[i - width] && dilated[i + width]) closed[i] = 1;
  }
  const seen = new Uint8Array(size), queue = new Int32Array(size);
  let best: PixelBox | null = null, bestScore = 0;
  for (let start = 0; start < size; start++) {
    if (!closed[start] || seen[start]) continue;
    let head = 0, tail = 0, minX = width, minY = height, maxX = 0, maxY = 0, area = 0;
    queue[tail++] = start; seen[start] = 1;
    while (head < tail) {
      const i = queue[head++], x = i % width, y = Math.floor(i / width); area++;
      minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const nx = x + dx, ny = y + dy, ni = ny * width + nx;
        if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1 && closed[ni] && !seen[ni]) { seen[ni] = 1; queue[tail++] = ni; }
      }
    }
    const boxWidth = maxX - minX + 1, boxHeight = maxY - minY + 1, center = (minX + maxX) / 2 / width;
    const ratio = boxWidth / Math.max(boxHeight, 1);
    if (area < size * .025 || boxHeight < height * .22 || boxHeight > height * .995 || ratio < .12 || ratio > 1.8 || center < .12 || center > .88) continue;
    const score = area * (1 - Math.abs(center - .5));
    if (score > bestScore) { best = { x: minX, y: minY, width: boxWidth, height: boxHeight }; bestScore = score; }
  }
  return best;
}
function modelLayer(image: HTMLImageElement, inspection: ARInspection, view: View) {
  const layer = document.createElement('canvas'), context = inspection.context;
  const hints = (context?.detection_merge?.detections ?? []).map(detection => ({ ...detection, unconfirmed: detection.merged_into === null }));
  const boxes: OverlayBox[] = context?.model === 'casting' ? hints : (context?.yolo_view?.detections ?? inspection.defects
    .filter(defect => defect.bbox_xyxy_px && defect.confidence !== undefined)
    .map(defect => ({ label: defect.label, confidence: defect.confidence!, bbox_xyxy_px: defect.bbox_xyxy_px! })));
  const corrosion = context?.corrosion;
  const anomaly = view === 'patchcore' ? context?.anomaly : view === 'corrosion' && corrosion?.activation_map ? { grid: corrosion.activation_map, threshold: .5 } : null;
  drawOverlay(layer, image, view === 'yolo' ? boxes : [], anomaly ?? undefined, false, view === 'patchcore' ? context?.geometry : null);
  return layer;
}

export default function TrackedAR({ initialModel = 'mpdd_metal_plate', inspectionModel = 'casting', machineId = 'M-02', onInspection, onClose }: {
  initialModel?: string; inspectionModel?: 'casting' | 'neu'; machineId?: string;
  onInspection?: (inspection: ARInspection) => void | Promise<void>; onClose: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null), canvasRef = useRef<HTMLCanvasElement>(null), resultRef = useRef<HTMLCanvasElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null), callbackRef = useRef(onInspection), rescanRef = useRef<() => void>(() => {}), recaptureRef = useRef<() => void>(() => {});
  const speechRef = useRef<SpeechRecognitionLike | null>(null), voiceCommandRef = useRef<(spoken: string) => void>(() => {});
  const pauseRef = useRef<() => void>(() => {}), descriptionRequest = useRef<AbortController | null>(null);
  const [description, setDescription] = useState(''), [describing, setDescribing] = useState(false), [descriptionError, setDescriptionError] = useState('');
  const [liveVision, setLiveVision] = useState(false), [describedAt, setDescribedAt] = useState<number | null>(null);
  const [listening, setListening] = useState(false), [voiceStatus, setVoiceStatus] = useState('');
  const evidenceRef = useRef<{ inspection: ARInspection; image: HTMLImageElement; capturedAt: number } | null>(null);
  const [model, setModel] = useState(initialModel), [models, setModels] = useState([initialModel]);
  const [view, setView] = useState<View>(inspectionModel === 'neu' ? 'yolo' : 'patchcore');
  const viewRef = useRef(view); viewRef.current = view;
  const [status, setStatus] = useState('Starting camera and preparing tracking…');
  const [tracked, setTracked] = useState(false), [registered, setRegistered] = useState(false), [busy, setBusy] = useState(false);
  const [scanPaused, setScanPaused] = useState(false), [trackingMode, setTrackingMode] = useState('Preparing WebGL');
  const [evidence, setEvidence] = useState<{ inspection: ARInspection; image: HTMLImageElement; capturedAt: number } | null>(null);
  evidenceRef.current = evidence;
  useEffect(() => { callbackRef.current = onInspection; }, [onInspection]);
  useEffect(() => () => { descriptionRequest.current?.abort(); speechRef.current?.abort(); }, []);
  useEffect(() => {
    if (!liveVision) return;
    let lastId = '';
    const describeLatest = () => {
      const latest = evidenceRef.current;
      if (latest && latest.inspection.id !== lastId && !descriptionRequest.current) {
        lastId = latest.inspection.id; void describePart(true);
      }
    };
    describeLatest();
    const timer = window.setInterval(describeLatest, 1200);
    return () => { window.clearInterval(timer); descriptionRequest.current?.abort(); descriptionRequest.current = null; setDescribing(false); };
  }, [liveVision]);
  async function describePart(live = false) {
    const capture = evidenceRef.current;
    if (!capture || descriptionRequest.current) return;
    if (!live) pauseRef.current();
    const request = new AbortController(); descriptionRequest.current = request;
    setDescribing(true); setDescriptionError('');
    try {
      const response = await fetch(`/api/inspections/${encodeURIComponent(capture.inspection.id)}/vision-description`, { method: 'POST', signal: request.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Vision description failed. Try again.');
      if (data.inspection_id !== capture.inspection.id || data.image_sha256 !== capture.inspection.image_sha256) throw new Error('Description does not match this capture. Scan again.');
      setDescription(data.description); setDescribedAt(capture.capturedAt);
    } catch (error) { if (!request.signal.aborted) setDescriptionError(error instanceof Error ? error.message : 'Could not describe this part.'); }
    finally { descriptionRequest.current = null; if (!request.signal.aborted) setDescribing(false); }
  }
  function runVoiceCommand(spoken: string) {
    const command = spoken.toLocaleLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
    const context = evidenceRef.current?.inspection.context;
    if (/\b(heat ?map|patch ?core)\b/.test(command)) {
      if (context?.anomaly?.grid?.length) { setView('patchcore'); setVoiceStatus('Showing the PatchCore heatmap.'); }
      else setVoiceStatus('The latest scan has no PatchCore heatmap.');
    } else if (/\b(yolo|boxes|bounding boxes)\b/.test(command)) {
      if (context?.model === 'casting' ? context.detection_merge : context?.yolo_view || evidenceRef.current?.inspection.defects.some(defect => defect.bbox_xyxy_px)) { setView('yolo'); setVoiceStatus('Showing YOLO boxes.'); }
      else setVoiceStatus('The latest scan has no YOLO box layer.');
    } else if (/\b(corrosion|rust)\b/.test(command)) {
      if (context?.corrosion?.activation_map?.length) { setView('corrosion'); setVoiceStatus('Showing the corrosion map.'); }
      else setVoiceStatus('The latest scan has no corrosion map.');
    }
    else if (/\b(scan again|rescan|inspect again)\b/.test(command)) {
      if (!tracked || busy) setVoiceStatus('Center the part and wait for the current scan to finish before rescanning.');
      else { rescanRef.current(); setVoiceStatus('Scanning again. Hold the part steady.'); }
    } else if (/\b(describe|identify)\b/.test(command)) {
      if (!evidence || busy || describing) setVoiceStatus('Wait for an inspection capture before asking for a description.');
      else { setVoiceStatus('Describing the latest capture.'); void describePart(); }
    } else if (/\b(start|enable|turn on)\b.*\b(live vision|ai vision)\b/.test(command)) { setLiveVision(true); setVoiceStatus('Live AI vision is on.'); }
    else if (/\b(stop|disable|turn off)\b.*\b(live vision|ai vision)\b/.test(command)) { setLiveVision(false); setVoiceStatus('Live AI vision is off.'); }
    else if (/\b(close|exit)\b.*\b(ar|viewer|inspection)\b/.test(command)) { setVoiceStatus('Closing AR.'); onClose(); }
    else setVoiceStatus(`I heard “${spoken}”. Try “show heatmap”, “scan again”, or “describe part”.`);
  }
  voiceCommandRef.current = runVoiceCommand;
  function startVoiceInput() {
    if (listening) { speechRef.current?.stop(); return; }
    const Speech = (window as SpeechWindow).SpeechRecognition ?? (window as SpeechWindow).webkitSpeechRecognition;
    if (!Speech) { setVoiceStatus('Voice input is not supported in this browser. Try Chrome on Android or enter commands with the controls.'); return; }
    const recognition = new Speech(); speechRef.current = recognition;
    recognition.lang = 'en-US'; recognition.continuous = false; recognition.interimResults = false; recognition.maxAlternatives = 1;
    recognition.onresult = event => {
      const finalResult = Array.from(event.results).find(result => result.isFinal);
      const spoken = finalResult?.[0]?.transcript;
      if (spoken) voiceCommandRef.current(spoken);
    };
    recognition.onerror = event => setVoiceStatus(event.error === 'not-allowed' ? 'Microphone permission was blocked. Allow it in your browser settings and try again.' : `Voice input stopped (${event.error}). Tap the microphone to retry.`);
    recognition.onend = () => { setListening(false); speechRef.current = null; };
    setVoiceStatus('Listening for an AR command…');
    try { recognition.start(); setListening(true); }
    catch { speechRef.current = null; setListening(false); setVoiceStatus('Could not start the microphone. Check its permission and try again.'); }
  }
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null, overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden'; dialogRef.current?.querySelector<HTMLButtonElement>('button')?.focus();
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); onClose(); }
      if (event.key === 'Tab') {
        const controls = [...(dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), select:not(:disabled), [tabindex="0"]') ?? [])];
        const first = controls[0], last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener('keydown', keyboard);
    return () => { document.removeEventListener('keydown', keyboard); document.body.style.overflow = overflow; previous?.focus(); };
  }, [onClose]);
  useEffect(() => {
    const controller = new AbortController();
    void fetch('/api/patchcore/models', { signal: controller.signal, cache: 'no-store' }).then(response => response.ok ? response.json() : Promise.reject())
      .then(registry => setModels(Array.from(new Set<string>([initialModel, ...(registry.custom ?? []).map((item: { name: string }) => item.name),
        ...(registry.mpdd ?? []).map((item: { name: string }) => item.name)])))).catch(() => {});
    return () => controller.abort();
  }, [initialModel]);
  // The frozen result uses the same drawing function and model data as the dashboard.
  useEffect(() => {
    const canvas = resultRef.current, context = canvas?.getContext('2d');
    if (!canvas || !context || !evidence) return;
    canvas.width = evidence.image.naturalWidth; canvas.height = evidence.image.naturalHeight;
    context.drawImage(evidence.image, 0, 0); context.drawImage(modelLayer(evidence.image, evidence.inspection, view), 0, 0);
  }, [evidence, view]);

  useEffect(() => {
    let alive = true, raf = 0, stream: MediaStream | null = null;
    let gl: ReturnType<typeof createAROverlay> | null = null;
    const controller = new AbortController();
    const sourceCanvas = document.createElement('canvas'), trackingCanvas = document.createElement('canvas');
    const sourceCtx = sourceCanvas.getContext('2d'), trackCtx = trackingCanvas.getContext('2d', { willReadFrequently: true });
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas || !sourceCtx || !trackCtx) return;
    let inFlight = false, pause = false, lastTracking = 0, nextInference = 0, lost = 0, stable = 0;
    let smoothed: PixelBox | null = null, currentBox: PixelBox | null = null;
    type Anchor = { quad: Quad; valid: boolean; capturedAt: number };
    let pending: Anchor | null = null;
    let latest: (Anchor & { inspection: ARInspection; image: HTMLImageElement; renderedView: View }) | null = null;
    rescanRef.current = () => { pause = false; nextInference = 0; setScanPaused(false); };
    pauseRef.current = () => { pause = true; setScanPaused(true); };
    setEvidence(null); setTracked(false); setRegistered(false); setBusy(false); setScanPaused(false);

    const analyze = async (box: PixelBox) => {
      inFlight = true; setBusy(true);
      const anchor: Anchor = { quad: boxQuad(box), valid: true, capturedAt: Date.now() }; pending = anchor;
      const request = new AbortController();
      const abort = () => request.abort(); controller.signal.addEventListener('abort', abort, { once: true });
      const timeout = window.setTimeout(abort, 60000);
      try {
        const crop = document.createElement('canvas'); crop.width = box.width; crop.height = box.height;
        const context = crop.getContext('2d'); if (!context) throw new Error('Camera capture is unavailable.');
        context.drawImage(sourceCanvas, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
        const { inspection, image } = await inspectTrackedCrop(crop, { model: inspectionModel, patchcoreModel: model, machine: machineId }, request.signal);
        if (!alive) return;
        setEvidence({ inspection, image, capturedAt: anchor.capturedAt });
        if (inspection.quality.passed) {
          latest = { ...anchor, inspection, image, renderedView: viewRef.current };
          gl?.setLayer(modelLayer(image, inspection, viewRef.current));
          const level = holdLevel(inspection);
          if (level) { pause = true; setScanPaused(true); setStatus(`${level === 'critical' ? 'Critical' : 'High severity'} finding. Production simulation held; close AR to review.`); }
          else setStatus('Inspection saved. Hold the part in view for registered model highlights.');
        } else { latest = null; setStatus('Capture needs improvement. Hold still and reduce glare; automatic scanning will retry.'); }
        await callbackRef.current?.(inspection);
      } catch (error) {
        if (alive) { latest = null; setRegistered(false); setStatus(request.signal.aborted ? 'Inspection timed out. Check the backend, then scan again.' : error instanceof Error ? error.message : 'Inspection failed. Scan again.'); }
      } finally {
        window.clearTimeout(timeout); controller.signal.removeEventListener('abort', abort);
        if (pending === anchor) pending = null;
        inFlight = false; nextInference = performance.now() + 4000;
        if (alive) setBusy(false);
      }
    };
    recaptureRef.current = () => {
      if (inFlight) { setStatus('Wait for the current inspection to finish before recapturing.'); return; }
      if (!currentBox) { setStatus('Center the part in the camera before recapturing.'); return; }
      pause = false; nextInference = performance.now() + 4000; setScanPaused(false);
      void analyze(currentBox);
    };
    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost. Open the secure phone link.');
        gl = createAROverlay(canvas);
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 24, max: 30 } }, audio: false });
        if (!alive) { stream.getTracks().forEach(track => track.stop()); return; }
        video.srcObject = stream; await video.play();
        setStatus('Camera ready. Preparing lightweight tracking…');
        if (!video.videoWidth || !video.videoHeight) throw new Error('Camera started without a video frame. Check camera permission and try again.');
        const w = video.videoWidth, h = video.videoHeight;
        sourceCanvas.width = w; sourceCanvas.height = h;
        trackingCanvas.width = Math.min(240, w); trackingCanvas.height = Math.round(h * trackingCanvas.width / w);
        setTrackingMode(`Silhouette tracking · ${gl.mode}`);
        setStatus('Center the bottle or component against a plain background. Scanning starts when tracking settles.');
        let lastTracked = false, lastRegistered = false;
        const render = (now: number) => {
          if (!alive) return;
          raf = requestAnimationFrame(render);
          if (document.hidden || now - lastTracking < 180) return;
          lastTracking = now;
          try {
            sourceCtx.drawImage(video, 0, 0, w, h);
            const tw = trackingCanvas.width, th = trackingCanvas.height;
            trackCtx.drawImage(sourceCanvas, 0, 0, tw, th);
            const sample = trackCtx.getImageData(0, 0, tw, th);
            const candidate = silhouetteBox(sample.data, tw, th);
            const best = candidate ? sourceCrop(candidate, tw, th, w, h) : null;
            if (best) {
              currentBox = best;
              stable = smoothed && overlap(smoothed, best) > .85 ? stable + 1 : 0; lost = 0;
              smoothed = smoothed ? { x: smoothed.x * .5 + best.x * .5, y: smoothed.y * .5 + best.y * .5,
                width: smoothed.width * .5 + best.width * .5, height: smoothed.height * .5 + best.height * .5 } : best;
              if (latest?.valid && smoothed) latest.quad = boxQuad(smoothed);
              if (pending?.valid && smoothed) pending.quad = boxQuad(smoothed);
              if (!pause && !inFlight && stable >= 2 && now >= nextInference) void analyze(best);
            } else {
              stable = 0;
              if (++lost > 3) { currentBox = null; smoothed = null; if (latest) latest.valid = false; if (pending) pending.valid = false; }
            }
            const isTracked = Boolean(smoothed);
            if (isTracked !== lastTracked) { lastTracked = isTracked; setTracked(isTracked); }
            const fit = coverTransform(w, h, canvas.clientWidth, canvas.clientHeight);
            const toScreen = (quad: Quad) => quad.map(point => ({ x: fit.x + point.x * fit.scale, y: fit.y + point.y * fit.scale })) as Quad;
            const registration = latest && smoothed ? overlap(quadBounds(latest.quad), smoothed) : 0;
            const visible = Boolean(latest?.valid && smoothed && registration >= .75 && Date.now() - latest.capturedAt < 12000);
            if (visible !== lastRegistered) { lastRegistered = visible; setRegistered(visible); }
            if (latest && latest.renderedView !== viewRef.current) {
              latest.renderedView = viewRef.current; gl?.setLayer(modelLayer(latest.image, latest.inspection, viewRef.current));
            }
            gl?.render(visible && latest ? toScreen(latest.quad) : null, smoothed ? toScreen(boxQuad(smoothed)) : null, visible ? 1 : 0);
          } catch (error) { cancelAnimationFrame(raf); controller.abort(); pause = true; setRegistered(false); gl?.render(null, null, 0); setStatus(`Tracking interrupted: ${error instanceof Error ? error.message : String(error)}. Close AR and reopen to retry.`); }
        };
        raf = requestAnimationFrame(render);
      } catch (error) { if (alive) setStatus(error instanceof Error ? error.message : 'Camera or WebGL could not start. Check permissions and try again.'); }
    };
    void start();
    return () => { alive = false; cancelAnimationFrame(raf); controller.abort(); stream?.getTracks().forEach(track => track.stop());
      gl?.dispose(); video.srcObject = null; rescanRef.current = () => {}; recaptureRef.current = () => {}; };
  }, [model, inspectionModel, machineId]);

  const level = evidence ? holdLevel(evidence.inspection) : null;
  const capturedContext = evidence?.inspection.context;
  const viewReady: Record<View, boolean> = {
    patchcore: Boolean(capturedContext?.anomaly?.grid?.length),
    yolo: capturedContext?.model === 'casting' ? Boolean(capturedContext.detection_merge) : Boolean(capturedContext?.yolo_view || evidence?.inspection.defects.some(defect => defect.bbox_xyxy_px)),
    corrosion: Boolean(capturedContext?.corrosion?.activation_map?.length),
  };
  useEffect(() => {
    if (viewReady[view]) return;
    const available = (['patchcore', 'yolo', 'corrosion'] as const).find(candidate => viewReady[candidate]);
    if (available) setView(available);
  }, [evidence, view, viewReady.patchcore, viewReady.yolo, viewReady.corrosion]);
  return <div ref={dialogRef} className="marker-ar tracked-ar" role="dialog" aria-modal="true" aria-labelledby="ar-title" aria-describedby="ar-description">
    <video ref={videoRef} className="bottle-ar-video" playsInline muted />
    <canvas ref={canvasRef} className="bottle-ar-canvas" aria-label="WebGL model highlights registered to the tracked component" />
    <header className="marker-ar-head"><div><strong id="ar-title">Live AR inspection</strong><span>{tracked ? registered ? 'Model highlights anchored' : 'Part tracked · waiting for a fresh result' : 'Center a part to begin'}</span></div><button aria-label="Close AR inspection" onClick={onClose}><X size={22} /></button></header>
    <aside className="marker-ar-panel"><div className="panel-heading"><h2><ScanLine size={22} /> Live findings</h2><span>{trackingMode}</span></div>
      <div className="ar-model-controls"><label>Inspection model<select value={model} disabled={busy} onChange={event => setModel(event.target.value)}>{models.map(name => <option key={name} value={name}>{name.replaceAll('_', ' ')}</option>)}</select></label><span>Machine {machineId} · {inspectionModel === 'neu' ? 'YOLO primary' : 'PatchCore primary'}</span></div>
      <div className="ar-view-switch" role="group" aria-label="AR model overlay">{(['patchcore', 'yolo', 'corrosion'] as const).map(value => <button key={value} type="button" aria-pressed={view === value} disabled={!viewReady[value]} title={viewReady[value] ? `Show ${value === 'patchcore' ? 'PatchCore heatmap' : value === 'yolo' ? 'YOLO boxes' : 'corrosion map'}` : 'This layer was not returned by the latest inspection'} onClick={() => setView(value)}>{value === 'patchcore' ? 'Heatmap' : value === 'yolo' ? 'YOLO boxes' : 'Corrosion'}</button>)}</div>
      <p role="status">{busy ? 'Running the inspection pipeline… ' : ''}{status}</p>
      {level && <div className="ar-quality-alert" role="alert"><AlertTriangle size={22} /><strong>{level === 'critical' ? 'Critical' : 'High severity'} · production simulation held</strong></div>}
      <details className="ar-capture" open={scanPaused || undefined}><summary><Layers size={18} /> Captured evidence {evidence ? new Date(evidence.capturedAt).toLocaleTimeString() : ''}</summary><canvas ref={resultRef} hidden={!evidence} className="bottle-result-canvas" aria-label="Model output on the exact inspected camera crop" />{evidence && <p>{evidence.inspection.defects.length} finding{evidence.inspection.defects.length === 1 ? '' : 's'} · inspection {evidence.inspection.id.slice(0, 8)}</p>}</details>
      <button className="primary ar-rescan" type="button" disabled={busy || !tracked} onClick={() => { recaptureRef.current(); setVoiceStatus(''); }}><Camera size={18} /> {busy ? 'Capturing…' : 'Recapture'}</button>
      <button type="button" disabled={busy || !tracked || !scanPaused} onClick={() => { rescanRef.current(); setVoiceStatus(''); }}><RotateCcw size={18} /> Resume automatic scan</button>
      <button type="button" disabled={!evidence || busy || describing} onClick={() => void describePart()}>{describing ? 'Describing captured part…' : 'Describe part with AI'}</button>
      <button type="button" aria-pressed={liveVision} onClick={() => setLiveVision(value => !value)}>{liveVision ? 'Stop live AI vision' : 'Start live AI vision'}</button>
      <button type="button" className="ar-voice-button" aria-pressed={listening} onClick={startVoiceInput} disabled={busy && !listening}>{listening ? <MicOff size={18} /> : <Mic size={18} />}{listening ? 'Stop listening' : 'Voice commands'}</button>
      {voiceStatus && <p className="ar-voice-status" role="status" aria-live="polite">{voiceStatus}</p>}
      {liveVision && <p role="status">Live AI describes new inspection captures when available. Descriptions may take a few seconds.</p>}
      {descriptionError && <p role="alert">{descriptionError}</p>}
      {description && <div className="ar-description"><strong>AI vision · {describedAt ? new Date(describedAt).toLocaleTimeString() : ''}</strong><p>{description}</p><span>Tentative description of that capture. Does not change severity or approval.</span></div>}
      <p id="ar-description" className="marker-ar-caution">Same models and severity rules as the dashboard. Highlights follow the tracked silhouette and hide if tracking is lost. This is 2D tracked AR; bottle accuracy and 3D pose are unvalidated.</p>
    </aside>
  </div>;
}
