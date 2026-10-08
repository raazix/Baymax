'use client';

import { useState } from 'react';
import { Smartphone, ArrowRight, ShieldCheck, LoaderCircle } from 'lucide-react';

export default function PhoneAccess() {
  const [code, setCode] = useState(''); const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  async function pair(event: React.FormEvent) {
    event.preventDefault(); if (busy) return;
    setBusy(true); setError('');
    try {
      const response = await fetch('/api/phone-session', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Could not pair this phone.');
      window.location.assign('/');
    } catch (e) { setError(e instanceof Error ? e.message : 'Pairing failed. Try again.'); setBusy(false); }
  }
  return <main className="phone-access"><section className="phone-access-panel">
    <div className="pairing-symbol"><Smartphone size={30} /></div><h1>LineGuard, on your phone.</h1>
    <p>Enter the pairing code shown on your laptop to open the inspection workspace.</p>
    <form onSubmit={event => void pair(event)}><label htmlFor="pairing-code">Pairing code</label><input autoFocus id="pairing-code" inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6} value={code} onChange={event => setCode(event.target.value.replace(/\D/g, ''))} placeholder="000000" required />
      {error && <p className="inference-error" role="alert">{error}</p>}<button className="primary" disabled={busy || code.length !== 6}>{busy ? <LoaderCircle size={18} className="is-spinning" /> : <ArrowRight size={18} />}{busy ? 'Opening workspace…' : 'Open workspace'}</button>
    </form><p className="pairing-note"><ShieldCheck size={16} /> Private session · camera permission stays in your control</p>
  </section></main>;
}
