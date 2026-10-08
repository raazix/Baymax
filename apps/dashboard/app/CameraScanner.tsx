'use client';

import { useEffect, useRef, useState } from 'react';
import { Camera, RefreshCw, X } from 'lucide-react';
import * as Dialog from '@radix-ui/react-dialog';

export default function CameraScanner({ modelLabel, onScan, onClose }: {
  modelLabel: string; onScan: (file: File, signal: AbortSignal) => Promise<void>; onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const request = useRef<AbortController | null>(null);
  const submitting = useRef(false);
  const [facing, setFacing] = useState<'environment' | 'user'>('environment');
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [aspectRatio, setAspectRatio] = useState(16 / 9);

  useEffect(() => {
    return () => { request.current?.abort(); };
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

  async function nativeScan(file: File | undefined) {
    if (!file || submitting.current) return;
    submitting.current = true; setBusy(true); setError('');
    const controller = new AbortController(); request.current = controller;
    try { await onScan(file, controller.signal); }
    catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : 'Camera inspection failed.'); }
    finally { submitting.current = false; if (!controller.signal.aborted) setBusy(false); }
  }

  return <Dialog.Root open onOpenChange={open => { if (!open) onClose(); }}><Dialog.Portal>
    <Dialog.Overlay className="workspace-overlay camera-dialog-overlay" />
    <Dialog.Content className="camera-scan-dialog" onOpenAutoFocus={() => { previousFocus.current = document.activeElement as HTMLElement | null; }} onCloseAutoFocus={event => { event.preventDefault(); previousFocus.current?.focus(); }}>
      <header><div><Dialog.Title id="camera-scan-title">Scan with camera</Dialog.Title><p>{modelLabel}</p></div><Dialog.Close asChild><button className="icon-button" aria-label="Close camera scanner"><X size={20} /></button></Dialog.Close></header>
      <div className="camera-scan-preview" style={{ aspectRatio }}><video ref={video} autoPlay muted playsInline onLoadedMetadata={event => { const preview = event.currentTarget; if (preview.videoWidth && preview.videoHeight) setAspectRatio(preview.videoWidth / preview.videoHeight); }} style={{ transform: facing === 'user' ? 'scaleX(-1)' : undefined }} aria-label="Live camera scanning preview" />{!ready && !error && <p role="status">Starting camera...</p>}<div className="capture-guides" aria-hidden="true"><i /><i /><i /><i /></div><span className="camera-live-tag">{busy ? 'Inspecting' : ready ? 'Live preview' : 'Camera setup'}</span></div>
      <Dialog.Description className="muted">Keep the whole part inside the frame, with even lighting. Your capture runs the inspection pipeline and saves its evidence.</Dialog.Description>
      {error && <p className="inference-error" role="alert">{error}</p>}
      <footer><button onClick={() => setFacing(value => value === 'environment' ? 'user' : 'environment')} disabled={busy}><RefreshCw size={16} /> Switch camera</button><button className="primary" disabled={!ready || busy} onClick={() => void scan()}><Camera size={17} />{busy ? 'Inspecting capture...' : 'Capture and inspect'}</button></footer>
      <label className={`native-camera file-button${busy ? ' disabled' : ''}`}><Camera size={16} /> Use the phone camera app<input type="file" accept="image/jpeg,image/png" capture="environment" disabled={busy} onChange={event => { void nativeScan(event.target.files?.[0]); event.target.value = ''; }} /></label>
    </Dialog.Content>
  </Dialog.Portal></Dialog.Root>;
}
