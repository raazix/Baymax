'use client';

import { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { anomalyLayer, clampCrop, coverTransform, inspectBottleCrop, type BottleAnomaly } from './bottleInference';

type Box = { x: number; y: number; width: number; height: number };
type Cv = any;
let openCvLoading: Promise<Cv> | null = null;

function loadOpenCv(): Promise<Cv> {
  const existing = (window as any).cv;
  if (existing?.Mat) return Promise.resolve(existing);
  if (openCvLoading) return openCvLoading;

  openCvLoading = new Promise<Cv>((resolve, reject) => {
    let settled = false;
    let awaitingModule = false;
    let poll = 0;
    const scriptId = 'lineguard-opencv-runtime';
    const finish = (error?: Error, runtime?: Cv) => {
      if (settled) return;
      settled = true;
      window.clearInterval(poll);
      window.clearTimeout(timeout);
      document.removeEventListener('error', onError, true);
      if (error) document.getElementById(scriptId)?.remove();
      if (error) reject(error);
      else resolve(runtime ?? (window as any).cv);
    };
    const check = () => {
      const candidate = (window as any).cv;
      if (candidate?.Mat) finish(undefined, candidate);
      else if (candidate && typeof candidate.then === 'function' && !awaitingModule) {
        awaitingModule = true;
        Promise.resolve(candidate).then(runtime => {
          if (!runtime?.Mat) { finish(new Error('OpenCV resolved without its Mat API.')); return; }
          (window as any).cv = runtime;
          finish(undefined, runtime);
        }, error => finish(new Error(`OpenCV WebAssembly initialization failed: ${String(error)}`)));
      }
    };
    const timeout = window.setTimeout(() => {
      const candidate = (window as any).cv;
      finish(new Error(candidate
        ? `OpenCV.js loaded but its WebAssembly runtime did not become ready (calledRun=${Boolean(candidate.calledRun)}).`
        : 'OpenCV.js loaded without exposing its browser runtime.'));
    }, 60000);

    const onError = (event: Event) => {
      const target = event.target;
      if (target instanceof HTMLScriptElement && target.id === scriptId) finish(new Error('Could not load the OpenCV.js runtime script.'));
    };
    document.addEventListener('error', onError, true);
    const script = document.getElementById(scriptId) as HTMLScriptElement | null ?? document.createElement('script');
    script.id = scriptId;
    script.src = '/api/opencvjs';
    script.async = true;
    script.onload = check;
    script.onerror = () => finish(new Error('Could not load the OpenCV.js runtime script.'));
    poll = window.setInterval(check, 100);
    if (!script.isConnected) document.head.appendChild(script);
  }).catch(error => {
    openCvLoading = null;
    throw error;
  });
  return openCvLoading;
}

export default function MarkerARViewer({ initialModel = 'default', onClose }: { initialModel?: string; onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState(initialModel);
  const [models, setModels] = useState<string[]>([initialModel]);
  const [status, setStatus] = useState('Loading OpenCV and starting the camera...');
  const [tracked, setTracked] = useState(false);
  const [busy, setBusy] = useState(false);
  const [evidence, setEvidence] = useState<{ result: BottleAnomaly; capturedAt: number } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetch('/api/patchcore/models', { signal: controller.signal, cache: 'no-store' }).then(response => {
      if (!response.ok) throw new Error('Could not load the model registry.');
      return response.json();
    }).then(registry => {
      const names = Array.from(new Set<string>(['default', initialModel,
        ...(registry.custom ?? []).map((item: { name: string }) => item.name),
        ...(registry.mpdd ?? []).map((item: { name: string }) => item.name)]));
      setModels(names);
      if (names.includes('bottle')) setModel('bottle');
    }).catch(() => {});
    return () => controller.abort();
  }, [initialModel]);

  useEffect(() => {
    let alive = true, raf = 0;
    let stream: MediaStream | null = null;
    let cv: Cv, capture: Cv, frameImage: Cv, gray: Cv, blurred: Cv, edges: Cv, hierarchy: Cv, contours: Cv;
    let smoothed: Box | null = null;
    let lostFrames = 0, inFlight = false, nextInference = 0, lastTracked = false;
    let analyzed: { snapshot: HTMLCanvasElement; layer: HTMLCanvasElement; box: Box } | null = null;
    const controller = new AbortController();
    const video = videoRef.current, canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!video || !canvas || !ctx) return;
    setEvidence(null); setTracked(false); setBusy(false);

    const analyze = async (box: Box, width: number, height: number) => {
      inFlight = true; setBusy(true);
      const capturedAt = Date.now();
      const requestController = new AbortController();
      const abortRequest = () => requestController.abort();
      controller.signal.addEventListener('abort', abortRequest, { once: true });
      let timedOut = false;
      const timeout = window.setTimeout(() => { timedOut = true; requestController.abort(); }, 45000);
      try {
        // Copy the exact OpenCV frame used for contour detection before the next video frame arrives.
        const snapshot = document.createElement('canvas');
        cv.imshow(snapshot, frameImage);
        const crop = document.createElement('canvas');
        crop.width = box.width; crop.height = box.height;
        const cropCtx = crop.getContext('2d');
        if (!cropCtx) throw new Error('Camera crop canvas is unavailable.');
        cropCtx.drawImage(snapshot, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
        const result = await inspectBottleCrop(crop, model, requestController.signal);
        if (!alive) return;
        if (snapshot.width !== width || snapshot.height !== height) throw new Error('Captured frame dimensions changed.');
        analyzed = { snapshot, layer: anomalyLayer(result), box };
        setEvidence({ result, capturedAt });
        setStatus('Showing the exact analyzed frame. Captures refresh automatically while the bottle is visible.');
      } catch (error) {
        if (!alive) return;
        analyzed = null; setEvidence(null);
        setStatus(timedOut ? 'Camera inference timed out. Check the backend connection.' : error instanceof Error ? error.message : 'Camera inference failed.');
      } finally {
        window.clearTimeout(timeout);
        controller.signal.removeEventListener('abort', abortRequest);
        inFlight = false; nextInference = performance.now() + 2500;
        if (alive) setBusy(false);
      }
    };

    const start = async () => {
      try {
        cv = await loadOpenCv();
        if (!alive) return;
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 960 }, height: { ideal: 720 } }, audio: false });
        if (!alive) { stream.getTracks().forEach(track => track.stop()); return; }
        video.srcObject = stream;
        await video.play();
        if (!alive) return;
        video.width = video.videoWidth; video.height = video.videoHeight;
        capture = new cv.VideoCapture(video);
        frameImage = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        gray = new cv.Mat(); blurred = new cv.Mat(); edges = new cv.Mat();
        hierarchy = new cv.Mat(); contours = new cv.MatVector();
        setStatus('Center the full bottle upright against a plain background. Live capture analysis starts automatically.');

        const render = () => {
          if (!alive) return;
          try {
            const w = video.videoWidth, h = video.videoHeight;
            if (!w || !h) { raf = requestAnimationFrame(render); return; }
            const cw = Math.round(canvas.clientWidth * devicePixelRatio), ch = Math.round(canvas.clientHeight * devicePixelRatio);
            if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
            ctx.clearRect(0, 0, cw, ch);
            capture.read(frameImage);
            cv.cvtColor(frameImage, gray, cv.COLOR_RGBA2GRAY);
            cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
            cv.Canny(blurred, edges, 45, 125);
            const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7));
            try { cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel); } finally { kernel.delete(); }
            cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            let best: Box | null = null, bestScore = 0;
            for (let i = 0; i < contours.size(); i++) {
              const contour = contours.get(i);
              try {
                const area = cv.contourArea(contour), rect = cv.boundingRect(contour);
                const aspect = rect.width / Math.max(rect.height, 1), centerX = (rect.x + rect.width / 2) / w;
                if (area < w * h * 0.012 || rect.height < h * 0.24 || aspect < 0.12 || aspect > 1.05 || centerX < 0.12 || centerX > 0.88) continue;
                const score = area * (1 - Math.abs(centerX - 0.5));
                if (score > bestScore) { bestScore = score; best = clampCrop(rect, w, h); }
              } finally { contour.delete(); }
            }
            if (best) {
              lostFrames = 0;
              smoothed = smoothed ? {
                x: smoothed.x * 0.65 + best.x * 0.35, y: smoothed.y * 0.65 + best.y * 0.35,
                width: smoothed.width * 0.65 + best.width * 0.35, height: smoothed.height * 0.65 + best.height * 0.35,
              } : best;
              if (!inFlight && performance.now() >= nextInference) void analyze(best, w, h);
            } else if (++lostFrames > 8) { smoothed = null; }
            const isTracked = smoothed !== null;
            if (isTracked !== lastTracked) { lastTracked = isTracked; setTracked(isTracked); }

            const transform = coverTransform(w, h, cw, ch);
            const { scale, x: ox, y: oy } = transform;
            // A delayed model result is always paired with its captured pixels, never the moving live video.
            if (analyzed) {
              ctx.drawImage(analyzed.snapshot, ox, oy, w * scale, h * scale);
              const box = analyzed.box;
              ctx.save(); ctx.imageSmoothingEnabled = false;
              ctx.drawImage(analyzed.layer, ox + box.x * scale, oy + box.y * scale, box.width * scale, box.height * scale);
              ctx.restore();
            }
            const outline = analyzed?.box ?? smoothed;
            if (outline) {
              ctx.strokeStyle = '#61f0d0'; ctx.lineWidth = 2 * devicePixelRatio;
              ctx.strokeRect(ox + outline.x * scale, oy + outline.y * scale, outline.width * scale, outline.height * scale);
            }
            raf = requestAnimationFrame(render);
          } catch (error) {
            controller.abort();
            analyzed = null; ctx.clearRect(0, 0, canvas.width, canvas.height);
            setEvidence(null); setStatus(`Camera tracking stopped: ${error instanceof Error ? error.message : String(error)}`);
          }
        };
        render();
      } catch (error) {
        if (alive) setStatus(error instanceof Error ? `Could not start bottle tracking: ${error.message}` : 'Could not start bottle tracking. Check camera permission.');
      }
    };
    void start();
    return () => {
      alive = false; cancelAnimationFrame(raf); controller.abort();
      stream?.getTracks().forEach(track => track.stop());
      for (const item of [capture, frameImage, gray, blurred, edges, hierarchy, contours]) item?.delete?.();
    };
  }, [model]);

  return <div className="marker-ar bottle-ar" role="dialog" aria-modal="true" aria-label="Bottle camera analysis">
    <video ref={videoRef} className="bottle-ar-video" playsInline muted />
    <canvas ref={canvasRef} className="bottle-ar-canvas" />
    <header className="marker-ar-head"><div><strong>LINEGUARD / CAMERA ANALYSIS</strong><span className={tracked ? 'tracking-live' : ''}>{evidence ? 'ANALYZED FRAME' : tracked ? 'BOTTLE IN VIEW' : 'SEARCHING FOR BOTTLE'}</span></div><button className="icon-button" aria-label="Close camera analysis" onClick={onClose}><X size={18} /></button></header>
    <aside className="marker-ar-panel"><h2>Camera heatmap</h2>
      <label>Experimental model <select value={model} onChange={event => setModel(event.target.value)}>{models.map(name => <option value={name} key={name}>{name.replaceAll('_', ' ')}</option>)}</select></label>
      <p role="status">{busy ? 'Analyzing a new camera crop... ' : ''}{status}</p>
      {evidence && <p>Captured {new Date(evidence.capturedAt).toLocaleTimeString()}<br />Crop score {evidence.result.anomaly_score.toFixed(3)} / threshold {evidence.result.threshold.toFixed(3)}<br />Image {evidence.result.image_sha256.slice(0, 12)}</p>}
      {evidence && !tracked && <p>Bottle out of live view; showing the last analyzed frame.</p>}
      <p className="marker-ar-caution">Colors come from this camera crop. Only patches above the model threshold are colored. The view holds the analyzed frame until the next result arrives. PatchCore grids are coarse; reflections and background may trigger scores. Accuracy on your steel bottle is unvalidated.</p>
    </aside>
  </div>;
}
