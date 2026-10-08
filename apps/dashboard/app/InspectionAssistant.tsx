'use client';

import { useEffect, useRef, useState } from 'react';
import { Mic, Square, Volume2, MessageSquare } from 'lucide-react';

type Answer = { id: string; answer: string; model: string; citations: { id: string; url: string; pointer: string }[] };
async function checked(response: Response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : 'Assistant request failed. Try again.');
  }
  return response;
}

export default function InspectionAssistant({ inspectionId }: { inspectionId: string }) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');
  const [voiceReady, setVoiceReady] = useState(false);
  const [ready, setReady] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alive = useRef(true);
  const voiceQuestion = useRef(false);

  useEffect(() => {
    alive.current = true;
    const controller = new AbortController();
    fetch('/api/assistant/status', { signal: controller.signal }).then(checked).then(r => r.json()).then(s => {
      setReady(s.llm_configured); setVoiceReady(s.voice_configured);
    }).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => {
      alive.current = false; controller.abort();
      if (timer.current) clearTimeout(timer.current);
      if (recorder.current?.state === 'recording') recorder.current.stop();
      stream.current?.getTracks().forEach(track => track.stop());
    };
  }, []);
  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  async function ask() {
    setBusy('Thinking...'); setError(''); setAudioUrl('');
    try {
      const response = await checked(await fetch(`/api/inspections/${inspectionId}/assistant`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() || 'Explain this inspection and the recommended next action.' }),
      }));
      const result: Answer = await response.json();
      if (alive.current) {
        setAnswer(result);
        if (voiceQuestion.current && voiceReady) { voiceQuestion.current = false; await speak(result); }
      }
    } catch (e) { if (alive.current) setError(e instanceof Error ? e.message : 'Unable to explain this inspection.'); }
    finally { if (alive.current) setBusy(''); }
  }

  async function speak(result: Answer | null = answer) {
    if (!result) return;
    setBusy('Preparing audio...'); setError('');
    try {
      const response = await checked(await fetch(`/api/assistant/responses/${result.id}/speech`, { method: 'POST' }));
      const blob = await response.blob();
      if (alive.current) setAudioUrl(URL.createObjectURL(blob));
    } catch (e) { if (alive.current) setError(e instanceof Error ? e.message : 'Unable to generate audio.'); }
    finally { if (alive.current) setBusy(''); }
  }

  async function record() {
    setError(''); setAudioUrl('');
    try {
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') throw new Error('Microphone recording requires a supported browser on localhost or HTTPS.');
      const capture = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!alive.current) { capture.getTracks().forEach(t => t.stop()); return; }
      stream.current = capture;
      const mimeType = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4'].find(t => MediaRecorder.isTypeSupported(t));
      const media = new MediaRecorder(capture, mimeType ? { mimeType } : undefined);
      recorder.current = media;
      const chunks: Blob[] = [];
      media.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      media.onstop = async () => {
        capture.getTracks().forEach(t => t.stop());
        if (timer.current) clearTimeout(timer.current);
        if (!alive.current) return;
        setRecording(false); setBusy('Transcribing...');
        try {
          const clip = new Blob(chunks, { type: media.mimeType || 'audio/webm' });
          const response = await checked(await fetch('/api/assistant/transcribe', { method: 'POST', headers: { 'Content-Type': clip.type }, body: clip }));
          const result = await response.json();
          if (alive.current) { setQuestion(result.text); voiceQuestion.current = true; }
        } catch (e) { if (alive.current) setError(e instanceof Error ? e.message : 'Transcription failed.'); }
        finally { if (alive.current) setBusy(''); }
      };
      media.start(); setRecording(true);
      timer.current = setTimeout(() => { if (media.state === 'recording') media.stop(); }, 30000);
    } catch (e) {
      stream.current?.getTracks().forEach(t => t.stop());
      setError(e instanceof Error ? e.message : 'Microphone access failed.');
    }
  }

  return <section className="panel inspection-assistant" aria-label="Inspection assistant">
    <div className="panel-heading"><h2><MessageSquare size={17} aria-hidden="true" /> Inspection assistant</h2><span>AI explanation</span></div>
    <form onSubmit={e => { e.preventDefault(); void ask(); }}>
      <label htmlFor="assistant-question">Ask about this inspection</label>
      <div className="assistant-input"><input id="assistant-question" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Why was this part flagged?" maxLength={1200} disabled={Boolean(busy) || recording} />
        <button type="button" onClick={() => recording ? recorder.current?.stop() : void record()} disabled={Boolean(busy) || !voiceReady} aria-label={recording ? 'Stop recording and transcribe' : 'Record a question'}>{recording ? <Square size={18} /> : <Mic size={18} />}</button>
        <button className="primary" disabled={Boolean(busy) || recording || !ready}>{busy || 'Explain'}</button>
      </div>
    </form>
    <p className="muted" role="status">{recording ? 'Recording. Stop when finished (30-second limit).' : 'Check the transcript before sending. AI explanations require engineer review.'}</p>
    {error && <p className="inference-error" role="alert">{error}</p>}
    {answer && <div className="assistant-answer"><p>{answer.answer}</p><div className="assistant-sources">{answer.citations.map(c => <a key={c.id} href={c.url} target="_blank" rel="noreferrer">{c.id}</a>)}<button type="button" onClick={() => void speak()} disabled={Boolean(busy) || recording || !voiceReady}><Volume2 size={16} aria-hidden="true" /> Listen</button></div></div>}
    {audioUrl && <audio controls autoPlay src={audioUrl} aria-label="Spoken inspection explanation" />}
  </section>;
}
