'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { drawOverlay, type OverlayAnomaly, type OverlayBox, type OverlayGeometry, type OverlayMarker } from './overlay';

type UploadedDefect = { label: string; confidence?: number; bbox_xyxy_px?: number[] };

export default function InspectionImage({ src, defects, anomaly, geometry, markers = [], alt }: { src: string; defects: UploadedDefect[]; anomaly?: OverlayAnomaly | null; geometry?: OverlayGeometry | null; markers?: OverlayMarker[]; alt: string }) {
  const base = useRef<HTMLCanvasElement>(null);
  const layer = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [failed, setFailed] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const reduce = useReducedMotion();

  useEffect(() => {
    let cancelled = false;
    setImage(null); setFailed(false);
    const img = new Image();
    img.onload = () => { if (!cancelled) setImage(img); };
    img.onerror = () => { if (!cancelled) setFailed(true); };
    img.src = src;
    return () => { cancelled = true; };
  }, [src]);

  useEffect(() => {
    if (!image || !base.current || !layer.current) return;
    drawOverlay(base.current, image, []);
    const boxes: OverlayBox[] = defects.filter(d => d.bbox_xyxy_px && d.confidence !== undefined).map(d => ({ label: d.label, confidence: d.confidence!, bbox_xyxy_px: d.bbox_xyxy_px! }));
    drawOverlay(layer.current, image, boxes, anomaly ?? undefined, false, geometry, markers);
  }, [image, defects, anomaly, geometry, markers]);

  const hasModelLayer = Boolean(anomaly) || Boolean(geometry) || defects.some(d => d.bbox_xyxy_px);
  return <div className="inspection-image">
    {failed ? <p className="inference-error" role="alert">The uploaded image could not be loaded.</p> : <div className="image-stack">
      <canvas ref={base} role="img" aria-label={alt} />
      {/* The model layer scans in left to right when a result arrives: the moment the model's answer lands on the part. */}
      <motion.canvas ref={layer} key={src} className="model-layer" aria-hidden="true"
        initial={reduce ? false : { clipPath: 'inset(0 100% 0 0)' }}
        animate={{ clipPath: image ? 'inset(0 0% 0 0)' : 'inset(0 100% 0 0)', opacity: overlay ? 1 : 0 }}
        transition={{ clipPath: { duration: reduce ? 0 : .6, ease: [0.16, 1, 0.3, 1], delay: reduce ? 0 : .15 }, opacity: { duration: .18 } }} />
    </div>}
    <div className="overlay-row">
      {hasModelLayer && <label className="overlay-toggle"><input type="checkbox" checked={overlay} onChange={e => setOverlay(e.target.checked)} /> Show model overlay</label>}
      {anomaly && <span className="heat-legend" aria-hidden="true"><i className="heat-scale" />near threshold → above threshold · ring = most anomalous patch</span>}
      {geometry && <span className="heat-legend" aria-hidden="true"><i className="rim-key" />fitted part outline · + centre · ray = finding position</span>}
    </div>
  </div>;
}
