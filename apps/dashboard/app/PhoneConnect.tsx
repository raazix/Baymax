'use client';

import { useEffect, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { QRCodeSVG } from 'qrcode.react';
import { Copy, Smartphone, X, Check, RefreshCw } from 'lucide-react';

type Link = { ready: boolean; url?: string; code?: string };
export default function PhoneConnect({ onClose }: { onClose: () => void }) {
  const [link, setLink] = useState<Link | null>(null); const [copied, setCopied] = useState(false); const [error, setError] = useState('');
  async function load(signal?: AbortSignal) {
    setError('');
    try { const response = await fetch('/api/phone-link', { cache: 'no-store', signal }); if (!response.ok) throw new Error('Could not load the phone link.'); setLink(await response.json()); }
    catch (e) { if (!signal?.aborted) setError(e instanceof Error ? e.message : 'Could not load pairing.'); }
  }
  useEffect(() => { const controller = new AbortController(); void load(controller.signal); return () => controller.abort(); }, []);
  return <Dialog.Root open onOpenChange={open => { if (!open) onClose(); }}><Dialog.Portal><Dialog.Overlay className="workspace-overlay" /><Dialog.Content className="workspace-dialog phone-connect">
    <div className="dialog-heading"><Dialog.Title><Smartphone size={21} /> Connect your phone</Dialog.Title><Dialog.Close asChild><button className="icon-button" aria-label="Close phone pairing"><X size={20} /></button></Dialog.Close></div>
    <Dialog.Description>Use the same inspection workspace, camera and production demo from your phone.</Dialog.Description>
    {link?.ready && link.url ? <><div className="phone-qr"><QRCodeSVG value={link.url} size={176} level="M" marginSize={4} title="Open the LineGuard phone workspace" /></div><p className="pair-code"><span>Pairing code</span><strong>{link.code}</strong></p><a className="phone-url" href={link.url} target="_blank" rel="noreferrer">{link.url}</a><button className="phone-copy" onClick={() => { void navigator.clipboard.writeText(link.url!).then(() => setCopied(true)).catch(() => setError('Copy the link above manually.')); }}>{copied ? <Check size={17} /> : <Copy size={17} />}{copied ? 'Link copied' : 'Copy phone link'}</button><p className="muted">Scan the QR, enter the code, then allow camera access. Keep the laptop and demo tunnel running.</p></> : <div className="phone-setup"><p>{link ? 'Start the secure phone demo on your laptop:' : 'Loading pairing details…'}</p>{link && <code>powershell -File scripts/start_phone_demo.ps1</code>}<p className="muted">This starts a temporary HTTPS link and displays your private pairing code.</p><button onClick={() => void load()}><RefreshCw size={16} /> Refresh link</button></div>}
    {error && <p role="alert" className="inference-error">{error}</p>}
  </Dialog.Content></Dialog.Portal></Dialog.Root>;
}
