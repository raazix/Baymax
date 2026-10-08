'use client';

import { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import type { RotorHeatmap } from './RotorViewer';

type Box = { x: number; y: number; width: number; height: number };
type Cv = any;

async function loadOpenCv(): Promise<Cv> {
  const existing = (window as any).cv;
  if (existing?.Mat) return existing;
  return await new Promise<Cv>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = '/api/opencvjs';
    script.async = true;
    const timeout = window.setTimeout(() => reject(new Error('OpenCV.js did not initialize.')), 30000);
    script.onload = () => {
      const candidate = (window as any).cv;
      if (!candidate) { window.clearTimeout(timeout); reject(new Error('OpenCV.js runtime is unavailable.')); return; }
      if (candidate.Mat) { window.clearTimeout(timeout); resolve(candidate); }
      else candidate.onRuntimeInitialized = () => { window.clearTimeout(timeout); resolve(candidate); };
    };
    script.onerror = () => { window.clearTimeout(timeout); reject(new Error('Could not load the OpenCV.js runtime.')); };
    document.head.appendChild(script);
  });
}

export default function MarkerARViewer({ heatmap, onClose }: { heatmap: RotorHeatmap; onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState('Loading OpenCV and starting the camera…');
  const [tracked, setTracked] = useState(false);
  const trackedRef = useRef(false);

  useEffect(() => {
    let alive = true;
    let raf = 0;
    let stream: MediaStream | null = null;
    let cv: Cv;
    let capture: Cv;
    let frameImage: Cv;
    let gray: Cv;
    let blurred: Cv;
    let edges: Cv;
    let hierarchy: Cv;
    let contours: Cv;
    let smoothed: Box | null = null;
    let lostFrames = 0;
    let lastUi = 0;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!video || !canvas || !ctx) return;

    const start = async () => {
      try {
        cv = await loadOpenCv();
        if (!alive) return;
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 960 }, height: { ideal: 720 } }, audio: false });
        if (!alive) { stream.getTracks().forEach(track => track.stop()); return; }
        video.srcObject = stream;
        await video.play();
        capture = new cv.VideoCapture(video);
        frameImage = new cv.Mat(); gray = new cv.Mat(); blurred = new cv.Mat(); edges = new cv.Mat();
        hierarchy = new cv.Mat(); contours = new cv.MatVector();
        setStatus('Show the full bottle upright against a plain background. Keep the camera steady.');

        const render = () => {
          if (!alive || !video.videoWidth) return;
          raf = requestAnimationFrame(render);
          const w = video.videoWidth, h = video.videoHeight;
          if (canvas.width !== canvas.clientWidth * devicePixelRatio || canvas.height !== canvas.clientHeight * devicePixelRatio) {
            canvas.width = Math.round(canvas.clientWidth * devicePixelRatio);
            canvas.height = Math.round(canvas.clientHeight * devicePixelRatio);
          }
          const cw = canvas.width, ch = canvas.height;
          ctx.clearRect(0, 0, cw, ch);
          capture.read(frameImage);
          cv.cvtColor(frameImage, gray, cv.COLOR_RGBA2GRAY);
          cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
          cv.Canny(blurred, edges, 45, 125);
          const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7));
          cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel);
          kernel.delete();
          cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

          let best: Box | null = null;
          let bestScore = 0;
          for (let i = 0; i < contours.size(); i++) {
            const contour = contours.get(i);
            const area = cv.contourArea(contour);
            if (area < w * h * 0.012) { contour.delete(); continue; }
            const rect = cv.boundingRect(contour);
            const aspect = rect.width / Math.max(rect.height, 1);
            const centerX = (rect.x + rect.width / 2) / w;
            if (rect.height < h * 0.24 || aspect < 0.12 || aspect > 1.05 || centerX < 0.12 || centerX > 0.88) { contour.delete(); continue; }
            const centerPenalty = 1 - Math.abs(centerX - 0.5);
            const score = area * centerPenalty;
            if (score > bestScore) { bestScore = score; best = { x: rect.x, y: rect.y, width: rect.width, height: rect.height }; }
            contour.delete();
          }
          if (best) {
            lostFrames = 0;
            smoothed = smoothed ? {
              x: smoothed.x * 0.65 + best.x * 0.35, y: smoothed.y * 0.65 + best.y * 0.35,
              width: smoothed.width * 0.65 + best.width * 0.35, height: smoothed.height * 0.65 + best.height * 0.35,
            } : best;
          } else if (++lostFrames > 8) smoothed = null;

          const isTracked = smoothed !== null;
          if (isTracked !== trackedRef.current && performance.now() - lastUi > 700) {
            trackedRef.current = isTracked; setTracked(isTracked); lastUi = performance.now();
            setStatus(isTracked ? 'Bottle outline tracked. Hold it steady for a cleaner overlay.' : 'Bottle not found. Center it against a plain background.');
          }
          if (smoothed) {
            // Match the object-fit:cover camera crop to the overlay coordinate space.
            const scale = Math.max(cw / w, ch / h);
            const ox = (cw - w * scale) / 2, oy = (ch - h * scale) / 2;
            const x = ox + smoothed.x * scale, y = oy + smoothed.y * scale;
            const bw = smoothed.width * scale, bh = smoothed.height * scale;
            const insetX = bw * 0.16, insetY = bh * 0.1;
            ctx.save();
            ctx.beginPath(); ctx.roundRect(x + insetX, y + insetY, bw - insetX * 2, bh - insetY * 2, Math.min(bw * 0.22, 40 * devicePixelRatio)); ctx.clip();
            const rows = heatmap.grid.length, cols = heatmap.grid[0]?.length ?? 0;
            for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
              const value = heatmap.grid[row][col];
              const flagged = value > heatmap.threshold;
              ctx.fillStyle = flagged ? `rgba(255, ${Math.max(75, 165 - Math.round((value - heatmap.threshold) * 70))}, 54, 0.3)` : 'rgba(46, 154, 214, 0.12)';
              const cellX = x + insetX + col * (bw - insetX * 2) / cols;
              const cellY = y + insetY + row * (bh - insetY * 2) / rows;
              ctx.fillRect(cellX, cellY, (bw - insetX * 2) / cols + 1, (bh - insetY * 2) / rows + 1);
            }
            ctx.restore();
            ctx.strokeStyle = '#61f0d0'; ctx.lineWidth = 3 * devicePixelRatio;
            ctx.beginPath(); ctx.roundRect(x, y, bw, bh, Math.min(bw * 0.25, 52 * devicePixelRatio)); ctx.stroke();
            ctx.fillStyle = 'rgba(8, 31, 35, .78)'; ctx.beginPath(); ctx.roundRect(x, Math.max(0, y - 32 * devicePixelRatio), 170 * devicePixelRatio, 27 * devicePixelRatio, 10 * devicePixelRatio); ctx.fill();
            ctx.fillStyle = '#eafff8'; ctx.font = `${12 * devicePixelRatio}px sans-serif`; ctx.fillText('BOTTLE TRACKED · DEMO HEATMAP', x + 9 * devicePixelRatio, Math.max(18, y - 14 * devicePixelRatio));
          }
        };
        render();
      } catch (error) {
        if (alive) setStatus(error instanceof Error ? `Could not start bottle tracking: ${error.message}` : 'Could not start bottle tracking. Check camera permission.');
      }
    };
    void start();
    return () => {
      alive = false; cancelAnimationFrame(raf);
      stream?.getTracks().forEach(track => track.stop());
      for (const item of [capture, frameImage, gray, blurred, edges, hierarchy, contours]) item?.delete?.();
    };
  }, [heatmap.grid, heatmap.threshold]);

  return <div className="marker-ar bottle-ar" role="dialog" aria-modal="true" aria-label="Markerless bottle tracking demo">
    <video ref={videoRef} className="bottle-ar-video" playsInline muted />
    <canvas ref={canvasRef} className="bottle-ar-canvas" />
    <header className="marker-ar-head"><div><strong>LINEGUARD · BOTTLE TRACKING</strong><span className={tracked ? 'tracking-live' : ''}>{tracked ? 'OBJECT LOCKED' : 'SEARCHING FOR BOTTLE'}</span></div><button className="icon-button" aria-label="Close AR" onClick={onClose}><X size={18} /></button></header>
    <aside className="marker-ar-panel"><h2>Markerless demo</h2><p>{status}</p><p className="marker-ar-caution">OpenCV tracks the bottle outline in the camera image. The colored grid is an illustrative proxy heatmap; it is not attached to the metal surface in 3D and is not a defect measurement.</p></aside>
  </div>;
}
