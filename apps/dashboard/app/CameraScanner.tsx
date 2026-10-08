'use client';

import { useEffect, useRef, useState } from 'react';
import { Camera, RefreshCw, X } from 'lucide-react';

export default function CameraScanner({ modelLabel, onScan, onClose }: {
  modelLabel: string; onScan: (file: File, signal: AbortSignal) => Promise<void>; onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const request = useRef<AbortController | null>(null);
  const submitting = useRef(false);
  const [facing, setFacing] = useState<'environment' | 'user'>('environment');
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null;
    dialog.current?.focus();
    return () => { request.current?.abort(); previousFocus?.focus(); };
  }, []);

  useEffect(() => {
    let alive = true;
    let stream: MediaStream | null = null;
    setReady(false); setError('');
    void (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost.');
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: facing },
          width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } }, audio: false });
        if (!alive) { stream.getTracks().forEach(track => track.stop()); return; }
        const preview = video.current;
        if (!preview) return;
        preview.srcObject = stream;
        await preview.play();
        if (alive) setReady(true);
      } catch (error) {
        stream?.getTracks().forEach(track => track.stop());
        if (alive) setError(error instanceof Error ? `${error.message} Allow camera access and close other camera apps if needed.` : 'Could not start the camera.');
      }
    })();
    return () => { alive = false; stream?.getTracks().forEach(track => track.stop()); };
  }, [facing]);

  async function scan() {
    if (submitting.current) return;
    submitting.current = true; setBusy(true); setError('');
    const controller = new AbortController(); request.current = controller;
    try {
      const preview = video.current;
      if (!preview?.videoWidth || !preview.videoHeight) throw new Error('Wait for the camera preview before scanning.');
      const capture = document.createElement('canvas');
      capture.width = preview.videoWidth; capture.height = preview.videoHeight;
      const context = capture.getContext('2d');
      if (!context) throw new Error('Could not capture the camera frame.');
      context.drawImage(preview, 0, 0);
      const blob = await new Promise<Blob>((resolve, reject) => capture.toBlob(value => value ? resolve(value) : reject(new Error('Could not encode the camera image.')), 'image/png'));
      if (controller.signal.aborted) return;
      await onScan(new File([blob], `camera-${Date.now()}.png`, { type: 'image/png' }), controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) setError(error instanceof Error ? error.message : 'Camera inspection failed.');
    } finally {
      submitting.current = false;
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return <div className="camera-scan-backdrop">
    <div className="camera-scan-dialog" ref={dialog} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="camera-scan-title" onKeyDown={event => {
      if (event.key === 'Escape') { event.preventDefault(); onClose(); }
      if (event.key !== 'Tab') return;
      const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')];
      const first = buttons[0], last = buttons[buttons.length - 1];
      if (!first) return;
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }}>
      <header><div><h2 id="camera-scan-title">Scan with camera</h2><p>{modelLabel}</p></div><button className="icon-button" onClick={onClose} aria-label="Close camera scanner"><X size={20} /></button></header>
      <div className="camera-scan-preview"><video ref={video} autoPlay muted playsInline aria-label="Live camera scanning preview" />{!ready && !error && <p role="status">Starting camera...</p>}</div>
      <p className="muted">Keep the entire part in view with even lighting. Capture runs the inspection pipeline and saves the result to history.</p>
      {error && <p className="inference-error" role="alert">{error}</p>}
      <footer><button onClick={() => setFacing(value => value === 'environment' ? 'user' : 'environment')} disabled={busy}><RefreshCw size={16} /> Switch camera</button><button className="primary" disabled={!ready || busy} onClick={() => void scan()}><Camera size={17} />{busy ? 'Inspecting capture...' : 'Capture and inspect'}</button></footer>
    </div>
  </div>;
}
