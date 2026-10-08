'use client';

import { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { anomalyLayer, sourceCrop, coverTransform, inspectBottleCrop, type BottleAnomaly } from './bottleInference';

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
  const resultCanvasRef = useRef<HTMLCanvasElement>(null);
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
    let cv: Cv, frameImage: Cv, gray: Cv, blurred: Cv, edges: Cv, hierarchy: Cv, contours: Cv, kernel: Cv;
    let smoothed: Box | null = null;
    let lostFrames = 0, inFlight = false, nextInference = 0, lastTracked = false;
    let lastTracking = 0;
    const sourceCanvas = document.createElement('canvas');
    const trackingCanvas = document.createElement('canvas');
    const sourceCtx = sourceCanvas.getContext('2d');
    const trackingCtx = trackingCanvas.getContext('2d', { willReadFrequently: true });
    const controller = new AbortController();
    const video = videoRef.current, canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!video || !canvas || !ctx || !sourceCtx || !trackingCtx) return;
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
        // Preserve the full-resolution source used to produce the small tracking frame.
        const snapshot = document.createElement('canvas');
        snapshot.width = width; snapshot.height = height;
        const snapshotCtx = snapshot.getContext('2d');
        if (!snapshotCtx) throw new Error('Snapshot canvas is unavailable.');
        snapshotCtx.drawImage(sourceCanvas, 0, 0);
        const crop = document.createElement('canvas');
        crop.width = box.width; crop.height = box.height;
        const cropCtx = crop.getContext('2d');
        if (!cropCtx) throw new Error('Camera crop canvas is unavailable.');
        cropCtx.drawImage(snapshot, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
        const result = await inspectBottleCrop(crop, model, requestController.signal);
        if (!alive) return;
        if (snapshot.width !== width || snapshot.height !== height) throw new Error('Captured frame dimensions changed.');
        const resultCanvas = resultCanvasRef.current;
        const resultCtx = resultCanvas?.getContext('2d');
        if (!resultCanvas || !resultCtx) return;
        resultCanvas.width = width; resultCanvas.height = height;
        resultCtx.drawImage(snapshot, 0, 0);
        resultCtx.imageSmoothingEnabled = false;
        resultCtx.drawImage(anomalyLayer(result), box.x, box.y, box.width, box.height);
        resultCtx.strokeStyle = '#61f0d0'; resultCtx.lineWidth = 2;
        resultCtx.strokeRect(box.x, box.y, box.width, box.height);
        setEvidence({ result, capturedAt });
        setStatus('Live camera is running. The heatmap capture below refreshes automatically.');
      } catch (error) {
        if (!alive) return;
        setEvidence(null);
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
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 960 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } }, audio: false });
        if (!alive) { stream.getTracks().forEach(track => track.stop()); return; }
        video.srcObject = stream;
        await video.play();
        if (!alive) return;
        video.width = video.videoWidth; video.height = video.videoHeight;
        sourceCanvas.width = video.width; sourceCanvas.height = video.height;
        const ratio = Math.min(1, 320 / video.width);
        trackingCanvas.width = Math.max(1, Math.round(video.width * ratio));
        trackingCanvas.height = Math.max(1, Math.round(video.height * ratio));
        frameImage = new cv.Mat(trackingCanvas.height, trackingCanvas.width, cv.CV_8UC4);
        gray = new cv.Mat(); blurred = new cv.Mat(); edges = new cv.Mat();
        hierarchy = new cv.Mat(); contours = new cv.MatVector();
        kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        setStatus('Center the full bottle upright against a plain background. Live capture analysis starts automatically.');

        const render = () => {
          if (!alive) return;
          try {
            const now = performance.now();
            if (document.hidden || now - lastTracking < 100) { raf = requestAnimationFrame(render); return; }
            lastTracking = now;
            const w = video.videoWidth, h = video.videoHeight;
            if (!w || !h) { raf = requestAnimationFrame(render); return; }
            const pixelRatio = Math.min(devicePixelRatio, 1.5);
            const cw = Math.round(canvas.clientWidth * pixelRatio), ch = Math.round(canvas.clientHeight * pixelRatio);
            if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
            ctx.clearRect(0, 0, cw, ch);
            if (sourceCanvas.width !== w || sourceCanvas.height !== h) { sourceCanvas.width = w; sourceCanvas.height = h; }
            sourceCtx.drawImage(video, 0, 0, w, h);
            const targetWidth = Math.min(320, w), targetHeight = Math.max(1, Math.round(h * targetWidth / w));
            if (trackingCanvas.width !== targetWidth || trackingCanvas.height !== targetHeight) {
              trackingCanvas.width = targetWidth; trackingCanvas.height = targetHeight;
              frameImage.delete(); frameImage = new cv.Mat(targetHeight, targetWidth, cv.CV_8UC4);
            }
            const tw = trackingCanvas.width, th = trackingCanvas.height;
            trackingCtx.drawImage(sourceCanvas, 0, 0, tw, th);
            frameImage.data.set(trackingCtx.getImageData(0, 0, tw, th).data);
            cv.cvtColor(frameImage, gray, cv.COLOR_RGBA2GRAY);
            cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
            cv.Canny(blurred, edges, 45, 125);
            cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel);
            cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            let best: Box | null = null, bestScore = 0;
            for (let i = 0; i < contours.size(); i++) {
              const contour = contours.get(i);
              try {
                const area = cv.contourArea(contour), rect = cv.boundingRect(contour);
                const aspect = rect.width / Math.max(rect.height, 1), centerX = (rect.x + rect.width / 2) / tw;
                if (area < tw * th * 0.012 || rect.height < th * 0.24 || aspect < 0.12 || aspect > 1.05 || centerX < 0.12 || centerX > 0.88) continue;
                const score = area * (1 - Math.abs(centerX - 0.5));
                if (score > bestScore) { bestScore = score; best = sourceCrop(rect, tw, th, w, h); }
              } finally { contour.delete(); }
            }
            if (best) {
              lostFrames = 0;
              smoothed = smoothed ? {
                x: smoothed.x * 0.65 + best.x * 0.35, y: smoothed.y * 0.65 + best.y * 0.35,
                width: smoothed.width * 0.65 + best.width * 0.35, height: smoothed.height * 0.65 + best.height * 0.35,
              } : best;
              if (!inFlight && performance.now() >= nextInference) void analyze(best, w, h);
            } else if (++lostFrames > 4) { smoothed = null; }
            const isTracked = smoothed !== null;
            if (isTracked !== lastTracked) { lastTracked = isTracked; setTracked(isTracked); }

            const transform = coverTransform(w, h, cw, ch);
            const { scale, x: ox, y: oy } = transform;
            // The full-screen canvas holds only a lightweight outline; native video stays live.
            const outline = smoothed;
            if (outline) {
              ctx.strokeStyle = '#61f0d0'; ctx.lineWidth = 2 * pixelRatio;
              ctx.strokeRect(ox + outline.x * scale, oy + outline.y * scale, outline.width * scale, outline.height * scale);
            }
            raf = requestAnimationFrame(render);
          } catch (error) {
            controller.abort();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
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
      for (const item of [frameImage, gray, blurred, edges, hierarchy, contours, kernel]) item?.delete?.();
    };
  }, [model]);

  return <div className="marker-ar bottle-ar" role="dialog" aria-modal="true" aria-label="Bottle camera analysis">
    <video ref={videoRef} className="bottle-ar-video" playsInline muted />
    <canvas ref={canvasRef} className="bottle-ar-canvas" />
    <header className="marker-ar-head"><div><strong>LINEGUARD / LIVE CAMERA</strong><span className={tracked ? 'tracking-live' : ''}>{tracked ? 'BOTTLE IN VIEW' : 'SEARCHING FOR BOTTLE'}</span></div><button className="icon-button" aria-label="Close camera analysis" onClick={onClose}><X size={18} /></button></header>
    <aside className="marker-ar-panel"><h2>Camera heatmap</h2>
      <label>Experimental model <select value={model} onChange={event => setModel(event.target.value)}>{models.map(name => <option value={name} key={name}>{name.replaceAll('_', ' ')}</option>)}</select></label>
      <p role="status">{busy ? 'Analyzing a new camera crop... ' : ''}{status}</p>
      <canvas ref={resultCanvasRef} className="bottle-result-canvas" hidden={!evidence} aria-label="Heatmap on the exact analyzed camera capture" />
      {evidence && <p>Captured {new Date(evidence.capturedAt).toLocaleTimeString()}<br />Crop score {evidence.result.anomaly_score.toFixed(3)} / threshold {evidence.result.threshold.toFixed(3)}<br />Image {evidence.result.image_sha256.slice(0, 12)}</p>}
      {evidence && !tracked && <p>Bottle out of live view; the result panel retains the last analyzed capture.</p>}
      <p className="marker-ar-caution">The main camera stays live. The result panel pairs each heatmap with its analyzed capture. Only above-threshold patches are colored. Accuracy on your steel bottle is unvalidated.</p>
    </aside>
  </div>;
}
