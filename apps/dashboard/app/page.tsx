'use client';

import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { ScanSearch, FileCheck2, Camera, BrainCircuit, SlidersHorizontal, Activity, Smartphone, Upload, ShieldCheck, LoaderCircle, AlertTriangle, Volume2 } from 'lucide-react';
import { toast } from 'sonner';
import MobileNavigation from './MobileNavigation';
import AnimatedValue from './AnimatedValue';
import DeferredRotorViewer from './DeferredRotorViewer';
import PipelineTrace, { type PipelineModule } from './PipelineTrace';
import ModelStatus from './ModelStatus';
import dynamic from 'next/dynamic';
import CameraInference from './CameraInference';
import UploadedPanel, { type UploadedInspection } from './UploadedPanel';
import TrainPatchCore from './TrainPatchCore';
import ProcessLab from './ProcessLab';
import InspectionAssistant from './InspectionAssistant';
import PresenterBar, { type PresenterStep } from './PresenterBar';
import HistorySensors from './HistorySensors';

const MarkerARViewer = dynamic(() => import('./MarkerARViewer'), { ssr: false });
const CameraScanner = dynamic(() => import('./CameraScanner'), { ssr: false });
const ProductionLine = dynamic(() => import('./ProductionLine'), { ssr: false });
const PhoneConnect = dynamic(() => import('./PhoneConnect'), { ssr: false });

type Defect = { label: string; confidence?: number; length_mm?: number; equivalent_diameter_mm?: number; r_mm?: number; theta_deg?: number; zone: string; severity: { level: string; reason: string }; bbox_xyxy_px?: number[]; length_px?: number; anomaly_score?: number; threshold?: number; anomaly_grid?: number[][]; cells_over_threshold?: number };
type Inspection = {
  id: string; part_id: string; lot_id: string; machine_id: string; created_at: string;
  scenario: string; source: string; image_url: string; image_sha256: string;
  context?: { model: 'neu' | 'casting'; input_source?: 'upload' | 'camera'; process_context: string; telemetry_source: string; patchcore_model?: string | null; anomaly?: { score: number; threshold: number; flagged: boolean; grid: number[][] } | null; inference?: { mode: 'full' | 'sliced'; engine?: string; tile_size_px?: number; overlap_ratio?: number; tile_count?: number; inference_ms?: number } | null };
  disposition: string; quality: { passed: boolean; rejection_reasons?: string[]; profile?: string; profile_note?: string }; defects: Defect[];
  telemetry: Record<string, number>;
  analytics: null | { rca: { hypothesis: string; confidence: number | null; method: string; feature_contributions: { feature: string; heuristic_risk_contribution?: number; contribution?: number }[] }; forecast: { risk: number; method: string }; uncertainty: { interval_95: number[]; breach_probability: number; simulations: number } };
  action: { text: string; status: string; required: boolean };
  verification?: { status: string; parts: number; defective: number };
  audit: { at: string; event: string; engineer?: string }[];
};
type WorkspaceTab = 'inspection' | 'audit' | 'camera' | 'train' | 'lab' | 'history';
const workspaceLabels: Record<WorkspaceTab, string> = { inspection: 'Inspection console', audit: 'Evidence & audit', camera: 'Camera station', train: 'Train PatchCore', history: 'Sensor history', lab: 'Process what-if' };
const labels: Record<string, string> = { upload_neu: 'Upload · steel (YOLO)', upload_casting: 'Upload · casting (PatchCore)', thermal_drift: 'Thermal drift · critical crack', cosmetic_scratch: 'Cosmetic scratch', normal: 'Normal component', unknown_anomaly: 'Unclassified anomaly', blurred: 'Rejected capture' };
const human = (value: string) => value.replaceAll('_', ' ');
const FEATURES: Record<string, [string, string]> = { temperature_c: ['Temperature', '°C'], pressure_bar: ['Pressure', 'bar'], vibration_mm_s: ['Vibration', 'mm/s'], machine_speed_rpm: ['Spindle speed', 'rpm'] };
const featureName = (key: string) => FEATURES[key]?.[0] ?? human(key);
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, { method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store' });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}). Check FastAPI is running on port 8000.`);
  }
  return response.json();
}

export default function Dashboard() {
  const [records, setRecords] = useState<Inspection[]>([]);
  const [current, setCurrent] = useState<Inspection | null>(null);
  const [model, setModel] = useState<'casting' | 'neu'>('casting');
  const [inferenceMode, setInferenceMode] = useState<'full' | 'sliced'>('full');
  const [tileSize, setTileSize] = useState(512);
  const [tileOverlap, setTileOverlap] = useState(0.2);
  const [processContext, setProcessContext] = useState<'nominal' | 'thermal_drift'>('nominal');
  const [progress, setProgress] = useState('');
  const [customModels, setCustomModels] = useState<{ name: string; train_images: number }[]>([]);
  const [mpddModels, setMpddModels] = useState<{ name: string; category: string }[]>([]);
  const [castingModel, setCastingModel] = useState('mpdd_metal_plate');
  const [presenter, setPresenter] = useState(false);
  const [arOpen, setArOpen] = useState(false);
  const [scannerOpen, setScannerOpen] = useState(false);
  const [phoneOpen, setPhoneOpen] = useState(false);
  const [partDiameter, setPartDiameter] = useState('');
  const [demo, setDemo] = useState<{ lot: string; target?: string; done: string[] }>({ lot: '', done: [] });
  const [demoStatus, setDemoStatus] = useState('');
  async function loadModels(select?: string) {
    try { const data = await api<{ custom: { name: string; train_images: number }[]; mpdd?: { name: string; category: string }[] }>('patchcore/models'); setCustomModels(data.custom); setMpddModels(data.mpdd ?? []); if (select) setCastingModel(select); } catch { /* selector simply stays on the default model */ }
  }
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [engineer, setEngineer] = useState('');
  const [note, setNote] = useState('');
  const [tab, setTab] = useState<WorkspaceTab>('inspection');
  const [cameraStatus, setCameraStatus] = useState('');
  const [quality, setQuality] = useState<{ passed: boolean; laplacian_variance: number; specular_fraction: number } | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [alertAudio, setAlertAudio] = useState('');
  const [voiceStatus, setVoiceStatus] = useState('');
  const [criticalHold, setCriticalHold] = useState<Inspection | null>(null);
  const releasedHolds = useRef(new Set<string>());
  const observedCritical = useRef(new Map<string, Inspection>());
  const spokenAlerts = useRef(new Set<string>());
  const voiceRequest = useRef<AbortController | null>(null);
  const alertPlayer = useRef<HTMLAudioElement>(null);
  useEffect(() => {
    if (current?.quality.passed && current.defects.some(d => d.severity.level === 'critical') && !releasedHolds.current.has(current.id)) {
      observedCritical.current.set(current.id, current);
      setCriticalHold(previous => previous?.id === current.id ? current : previous ?? current);
      void autoSpeakAlert(current);
    }
  }, [current]);
  useEffect(() => {
    if (!alertAudio) return;
    void alertPlayer.current?.play().catch(() => setVoiceStatus('Browser blocked automatic audio. Press Play to hear the ElevenLabs alert.'));
    return () => { URL.revokeObjectURL(alertAudio); };
  }, [alertAudio]);
  useEffect(() => () => { voiceRequest.current?.abort(); }, []);
  useEffect(() => {
    if (busy) toast.loading(progress || 'Working with inspection evidence…', { id: 'inspection-operation' });
    else toast.dismiss('inspection-operation');
  }, [busy, progress]);
  useEffect(() => {
    if (criticalHold) toast.error('Critical finding: production simulation held', { id: `critical-${criticalHold.id}`, description: `Lot ${criticalHold.lot_id} requires engineer review.`, duration: 6000 });
  }, [criticalHold?.id]);

  useEffect(() => { void loadModels(); }, []);
  useEffect(() => {
    let cancelled = false;
    api<Inspection[]>('inspections').then(data => { if (!cancelled) { setRecords(data); setCurrent(data[0] || null); } }).catch(e => { if (!cancelled) setError(e.message); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; stream.current?.getTracks().forEach(track => track.stop()); };
  }, []);
  useEffect(() => {
    if (tab === 'camera' && video.current && stream.current) video.current.srcObject = stream.current;
    if (tab !== 'camera') { stream.current?.getTracks().forEach(track => track.stop()); stream.current = null; setCameraActive(false); }
  }, [tab]);

  async function operation(task: () => Promise<void>) {
    setBusy(true); setError('');
    try { await task(); } catch (e) { const message = e instanceof Error ? e.message : 'Unable to complete request. Try again.'; setError(message); toast.error(message); }
    finally { setBusy(false); }
  }
  async function refresh(result: Inspection) { setCurrent(result); setRecords(await api<Inspection[]>('inspections')); }
  async function autoSpeakAlert(result: Inspection) {
    if (!result.quality.passed || !result.defects.some(item => ['critical', 'high'].includes(item.severity.level)) || spokenAlerts.current.has(result.id)) return;
    spokenAlerts.current.add(result.id);
    voiceRequest.current?.abort();
    const controller = new AbortController(); voiceRequest.current = controller;
    setVoiceStatus('Generating ElevenLabs quality alert...'); setAlertAudio('');
    try {
      const response = await fetch(`/api/inspections/${result.id}/alert-speech`, { method: 'POST', cache: 'no-store', signal: controller.signal });
      if (!response.ok) throw new Error('ElevenLabs alert unavailable. Check voice credentials or provider quota.');
      const blob = await response.blob();
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(blob);
      setAlertAudio(url);
      setVoiceStatus('ElevenLabs quality alert');
    } catch (e) { if (!controller.signal.aborted) { spokenAlerts.current.delete(result.id); setVoiceStatus(e instanceof Error ? e.message : 'Voice alert unavailable. Visual hold remains active.'); } }
  }
  function criticalDemo() {
    void operation(async () => {
      const result = await api<Inspection>('replay', { scenario: 'thermal_drift', seed: 42, analytics_mode: 'inline' });
      await refresh(result); setTab('inspection'); void autoSpeakAlert(result);
    });
  }
  function resumeSimulation() {
    if (!criticalHold || criticalHold.action.status !== 'approved') return;
    releasedHolds.current.add(criticalHold.id);
    setCriticalHold([...observedCritical.current.values()].find(record => !releasedHolds.current.has(record.id)) ?? null);
  }
  async function inspectFile(file: File, options: { model: 'casting' | 'neu'; processContext: string; lot?: string; machine?: string; patchcoreModel?: string; inferenceMode?: 'full' | 'sliced'; tileSize?: number; overlap?: number; inputSource?: 'upload' | 'camera'; signal?: AbortSignal }): Promise<Inspection> {
    const params = new URLSearchParams({ model: options.model, process_context: options.processContext });
    if (options.inputSource) params.set('input_source', options.inputSource);
    if (options.model === 'neu') {
      params.set('inference_mode', options.inferenceMode ?? 'full');
      if (options.inferenceMode === 'sliced') { params.set('tile_size', String(options.tileSize ?? 512)); params.set('overlap', String(options.overlap ?? 0.2)); }
    }
    if (options.model === 'casting' && options.patchcoreModel) params.set('patchcore_model', options.patchcoreModel);
    const diameter = Number(partDiameter);
    if (partDiameter.trim() && Number.isFinite(diameter) && diameter > 0) params.set('part_diameter_mm', String(diameter));
    if (options.lot) params.set('lot_id', options.lot);
    if (options.machine) params.set('machine_id', options.machine);
    const response = await fetch(`/api/inspections/upload?${params}`, { method: 'POST', body: file, headers: { 'Content-Type': file.type || 'image/jpeg' }, cache: 'no-store', signal: options.signal });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(typeof payload.detail === 'string' ? payload.detail : `Upload failed (${response.status}). Check FastAPI is running on port 8000.`);
    }
    return response.json();
  }
  function upload(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    void operation(async () => { const result = await inspectFile(file, { model, processContext, patchcoreModel: castingModel, inferenceMode, tileSize, overlap: tileOverlap }); await refresh(result); setTab('inspection'); toast.success(result.quality.passed ? 'Inspection saved' : 'Capture saved for review', { description: result.part_id }); void autoSpeakAlert(result); });
  }
  function openScanner() {
    stream.current?.getTracks().forEach(track => track.stop()); stream.current = null;
    setCameraActive(false); setArOpen(false); setScannerOpen(true);
  }
  async function scanCamera(file: File, signal: AbortSignal) {
    setBusy(true); setError('');
    try {
      const result = await inspectFile(file, { model, processContext, patchcoreModel: castingModel,
        inferenceMode, tileSize, overlap: tileOverlap, inputSource: 'camera', signal });
      if (signal.aborted) return;
      await refresh(result);
      if (signal.aborted) return;
      setScannerOpen(false); setTab('inspection');
      toast.success(result.quality.passed ? 'Camera inspection saved' : 'Recapture needed', { description: result.part_id });
      void autoSpeakAlert(result);
    } finally { setBusy(false); }
  }
  function decision(value: string) { if (!current) return; void operation(async () => { await refresh(await api<Inspection>(`inspections/${current.id}/decision`, { decision: value, engineer: engineer.trim(), note })); }); }
  function verifyFiles(files: FileList | null) {
    if (!current?.context || !files) return;
    const list = [...files];
    if (list.length < 20) { setError('Verification needs at least 20 distinct follow-up images. Select more files.'); return; }
    const target = current;
    void operation(async () => {
      const ids: string[] = [];
      for (const [index, file] of list.entries()) {
        setProgress(`Inspecting follow-up image ${index + 1} of ${list.length}…`);
        const result = await inspectFile(file, { model: target.context!.model, processContext: 'nominal', lot: target.lot_id, machine: target.machine_id, patchcoreModel: target.context!.patchcore_model ?? 'default', inferenceMode: target.context!.inference?.mode ?? 'full', tileSize: target.context!.inference?.tile_size_px ?? 512, overlap: target.context!.inference?.overlap_ratio ?? 0.2 });
        if (!result.quality.passed) throw new Error(`Follow-up image ${index + 1} (${file.name}) failed the quality gate. Remove or recapture it and retry.`);
        ids.push(result.id);
      }
      setProgress('Recording verification…');
      await refresh(await api<Inspection>(`inspections/${target.id}/verification`, { inspection_ids: ids }));
      setProgress('');
    });
  }
  async function demoFile(set: string, index = 0): Promise<File> {
    const catalog = await api<Record<string, string[]>>('demo/catalog');
    const name = catalog[set]?.[index];
    if (!name) throw new Error(`No demo images found for "${set}". Check the demo_images folder.`);
    const response = await fetch(`/api/demo/file/${set}/${encodeURIComponent(name)}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Could not load demo image ${name}.`);
    const blob = await response.blob();
    return new File([blob], name, { type: blob.type || 'image/jpeg' });
  }
  function startPresenter() {
    const stamp = new Date().toISOString().replace(/\D/g, '').slice(4, 12);
    setDemo({ lot: `DEMO-${stamp}`, done: [] }); setDemoStatus(''); setPresenter(true);
  }
  function markDone(step: string, target?: string) { setDemo(d => ({ ...d, target: target ?? d.target, done: [...new Set([...d.done, step])] })); }
  function demoInspect(step: string, set: string, demoModel: 'casting' | 'neu', context: 'nominal' | 'thermal_drift', becomesTarget: boolean, index = 0) {
    void operation(async () => {
      setDemoStatus('Running the real pipeline: quality gate, model, severity, root cause, forecast…');
      const file = await demoFile(set, index);
      const result = await inspectFile(file, { model: demoModel, processContext: context, lot: demo.lot, machine: 'M-DEMO', patchcoreModel: 'default' });
      await refresh(result); setTab('inspection'); setModel(demoModel); setProcessContext(context);
      markDone(step, becomesTarget ? result.id : undefined); setDemoStatus(`Inspected ${file.name}.`);
    });
  }
  function demoApprove() {
    if (!demo.target) return;
    void operation(async () => {
      const name = engineer.trim().length >= 2 ? engineer.trim() : 'Demo Engineer';
      await refresh(await api<Inspection>(`inspections/${demo.target}/decision`, { decision: 'approve', engineer: name, note: 'Approved during live demo' }));
      setTab('inspection'); markDone('approve'); setDemoStatus(`Approved by ${name}. Nothing is sent to the machine; the engineer stays in control.`);
    });
  }
  function demoVerify() {
    if (!demo.target) return;
    const target = demo.target;
    void operation(async () => {
      const catalog = await api<Record<string, string[]>>('demo/catalog');
      const count = catalog.verification?.length ?? 0;
      if (count < 20) throw new Error('The verification folder needs 20 images.');
      const ids: string[] = [];
      for (let i = 0; i < count; i++) {
        setDemoStatus(`Re-inspecting follow-up part ${i + 1} of ${count}…`);
        const result = await inspectFile(await demoFile('verification', i), { model: 'casting', processContext: 'nominal', lot: demo.lot, machine: 'M-DEMO', patchcoreModel: 'default' });
        ids.push(result.id);
      }
      await refresh(await api<Inspection>(`inspections/${target}/verification`, { inspection_ids: ids }));
      setTab('inspection'); markDone('verify'); setDemoStatus(`${count} follow-up parts inspected and recorded as verification evidence.`);
    });
  }
  function demoAudit() {
    if (!demo.target) return;
    void operation(async () => { await refresh(await api<Inspection>(`inspections/${demo.target}`)); setTab('audit'); markDone('audit'); setDemoStatus('Every step above is in this hash-chained audit trail.'); });
  }
  const presenterSteps: PresenterStep[] = [
    { title: 'Defective casting', say: 'Upload a defective casting under a thermal-drift process. PatchCore flags it and the heatmap shows where.', run: () => demoInspect('defect', 'defect', 'casting', 'thermal_drift', true, 3), done: demo.done.includes('defect') },
    { title: 'Approve action', say: 'XGBoost points to thermal drift; an engineer approves the recommended action.', run: demoApprove, done: demo.done.includes('approve'), disabled: !demo.target },
    { title: 'Verify 20 parts', say: 'Re-inspect 20 follow-up parts to close the loop.', run: demoVerify, done: demo.done.includes('verify'), disabled: !demo.done.includes('approve') },
    { title: 'Audit trail', say: 'Show the tamper-evident record of every step.', run: demoAudit, done: demo.done.includes('audit'), disabled: !demo.target },
    { title: 'Normal casting', say: 'Contrast: a good part stays below the threshold, and the heatmap is mostly clear.', run: () => demoInspect('normal', 'normal', 'casting', 'nominal', false), done: demo.done.includes('normal') },
    { title: 'Steel defects', say: 'Second model: YOLO finds and labels steel-surface defects.', run: () => demoInspect('steel', 'steel', 'neu', 'nominal', false), done: demo.done.includes('steel') },
    { title: 'What-if lab', say: 'Move process readings and watch XGBoost, TreeSHAP, PLSR and Monte Carlo react live.', run: () => { setTab('lab'); markDone('lab'); setDemoStatus('Click the presets: Nominal, then Thermal drift.'); }, done: demo.done.includes('lab') },
  ];
  const reduceMotion = useReducedMotion();
  function jumpTo(module: PipelineModule) {
    const target = document.querySelector<HTMLElement>(`[data-module~="${module}"]`);
    if (!target) return;
    target.querySelectorAll<HTMLDetailsElement>('details').forEach(detail => { detail.open = true; });
    target.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
    if (!reduceMotion) target.animate([{ boxShadow: '0 0 0 0 rgba(7,109,102,0)' }, { boxShadow: '0 0 0 4px rgba(7,109,102,.38)' }, { boxShadow: '0 0 0 0 rgba(7,109,102,0)' }], { duration: 1200, easing: 'cubic-bezier(.16,1,.3,1)', delay: 250 });
  }
  async function startCamera() {
    setCameraStatus('Requesting camera access…');
    try {
      stream.current?.getTracks().forEach(track => track.stop());
      stream.current = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      if (video.current) video.current.srcObject = stream.current;
      setCameraActive(true); setCameraStatus('Camera ready. Place the component in consistent lighting.');
    } catch { setCameraStatus('Camera unavailable. Allow browser access and close other camera apps, then retry.'); }
  }
  async function grabFrame(): Promise<Blob> {
    if (!video.current?.videoWidth) throw new Error('Start the camera and wait for the video.');
    const canvas = document.createElement('canvas'); canvas.width = video.current.videoWidth; canvas.height = video.current.videoHeight;
    canvas.getContext('2d')!.drawImage(video.current, 0, 0);
    const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', .9));
    if (!blob) throw new Error('Capture failed. Retry.');
    return blob;
  }
  function capture() { void operation(async () => {
    const blob = await grabFrame();
    const response = await fetch('/api/camera/quality', { method: 'POST', body: blob, headers: { 'Content-Type': 'image/jpeg' } });
    if (!response.ok) throw new Error('Quality check failed. Verify the backend and retry.');
    setQuality(await response.json());
  }); }

  const severity = !current?.quality.passed ? 'recapture' : ['critical', 'high', 'review_required', 'medium', 'low'].find(level => current.defects.some(item => item.severity.level === level)) || 'pass';
  const defect = current?.defects.find(item => item.severity.level === severity) || current?.defects[0];
  const castingModelLabel = castingModel === 'mvtec_metal_nut' ? 'MVTec AD · metal nut benchmark'
    : castingModel.startsWith('mpdd_') ? `MPDD · ${human(castingModel.slice(5))} proxy`
      : `${castingModel} custom model`;
  function navigate(item: typeof tab) { setTab(item); window.scrollTo({ top: 0, behavior: 'auto' }); }
  return <div className="shell">
    <aside className="sidebar">
      <a className="brand" href="/" aria-label="LineGuard home"><span className="brand-mark"><ScanSearch size={22} /></span><span className="brand-word">LineGuard<small>Quality intelligence</small></span></a>
      <nav aria-label="Workspace">{(['inspection', 'audit', 'camera', 'train', 'lab', 'history'] as const).map(item => { const Icon = { inspection: ScanSearch, audit: FileCheck2, camera: Camera, train: BrainCircuit, lab: SlidersHorizontal, history: Activity }[item]; return <button key={item} aria-label={workspaceLabels[item]} title={workspaceLabels[item]} className={tab === item ? 'nav active' : 'nav'} aria-current={tab === item ? 'page' : undefined} onClick={() => navigate(item)}>{tab === item && <motion.span layoutId="nav-active" className="nav-indicator" transition={{ duration: reduceMotion ? 0 : .28, ease: [0.16, 1, 0.3, 1] }} aria-hidden="true" />}<Icon size={17} strokeWidth={1.9} aria-hidden="true" /><span>{workspaceLabels[item]}</span></button>; })}</nav>
      <div className="sidebar-bottom"><span className="connection-dot" /> Local demo workspace<p>Track 3 · Singularity 2026</p></div><details className="sidebar-details"><summary>Model status</summary><ModelStatus /></details>
    </aside>
    <main>
      <header className="topbar"><div><h1>{tab === 'inspection' ? 'Inspection console' : tab === 'audit' ? 'Evidence & audit' : tab === 'camera' ? 'Camera station' : tab === 'train' ? 'Train PatchCore' : tab === 'history' ? 'Historical sensor data' : 'Process what-if lab'}</h1><p>Find the flaw. Find the cause. Stop the next one.</p></div><div className="topbar-actions"><button className="phone-connect-action" onClick={() => setPhoneOpen(true)}><Smartphone size={17} /> Connect phone</button><span className="demo-tag">Proxy model demo</span>{!presenter && <button className="primary" onClick={startPresenter}>Presenter mode</button>}</div></header>
      {presenter && <PresenterBar steps={presenterSteps} busy={busy} status={busy && progress ? progress : demoStatus} onClose={() => setPresenter(false)} />}
      {(tab === 'inspection' || tab === 'audit') && <section className="upload-workspace" aria-label="Image inspection">
        <div className="upload-controls">
          <div className="capture-model"><span className="capture-symbol"><ScanSearch size={25} /></span><div><strong>Inspect a component</strong><span>{castingModelLabel}</span></div></div>
          <button className="scan-camera-action" disabled={busy || loading} onClick={openScanner}><Camera size={17} /> Scan with camera</button>
          <label className={`file-button primary-file${busy || loading ? ' disabled' : ''}`}><Upload size={17} />{busy ? 'Processing image…' : 'Upload image'}<input type="file" accept="image/jpeg,image/png" disabled={busy || loading} onChange={e => { upload(e.target.files); e.target.value = ''; }} /></label>
        </div>
        <div className="capture-context"><span><ShieldCheck size={14} /> Industrial-part proxy</span><span>Simulated process inputs</span><details><summary>Scope &amp; model limits</summary><p>Brake discs are the target; current models are industrial-part proxies. Imported sensor history is shown separately and is not joined to these forecasts. No available model is brake-disc validated.</p></details></div>
        <details className="model-options"><summary>Advanced · choose a PatchCore model</summary><label>Model for this image<select value={castingModel} onChange={e => setCastingModel(e.target.value)}><optgroup label="MPDD component proxies">{mpddModels.map(item => <option key={item.name} value={item.name}>MPDD · {human(item.category)}</option>)}</optgroup><optgroup label="Custom / benchmark models">{customModels.map(item => <option key={item.name} value={item.name}>{item.name === 'mvtec_metal_nut' ? 'MVTec AD · metal nut' : item.name}</option>)}</optgroup></select></label><p>{castingModel === 'mvtec_metal_nut' ? 'MVTec metal-nut test: 81/93 defect images detected (87.1% recall), 12 missed, 0/22 normal false alarms; image AUROC 0.989. This result applies to MVTec nuts, not brake discs.' : castingModel === 'mpdd_metal_plate' ? 'MPDD metal-plate proxy: test recall 38.0% (27/71 defects detected; 44 missed). It is not validated on brake discs.' : 'Choose a model matching the inspected component. No available model is validated on brake discs.'}</p></details>
      </section>}
      {error && <div className="error" role="alert">{error}<button onClick={() => void operation(async () => { const data = await api<Inspection[]>('inspections'); setRecords(data); setCurrent(data[0] || null); })} disabled={busy}>Reconnect</button></div>}
      {criticalHold && <motion.section layout className="production-hold" role="alert" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><AlertTriangle size={24} aria-hidden="true" /><div><strong>Critical defect — simulated production stopped</strong><p>{human(criticalHold.defects.find(d => d.severity.level === 'critical')?.label ?? 'critical defect')} · Part {criticalHold.part_id} · Lot {criticalHold.lot_id} · Machine {criticalHold.machine_id}</p><span>Engineer approval and explicit resume are required. No physical machine command is sent.</span></div><div className="buttons"><button onClick={() => { setCurrent(criticalHold); setTab('inspection'); }}>Review finding</button><button disabled={busy || criticalHold.action.status !== 'approved'} onClick={resumeSimulation}>Resume simulation after approval</button><button disabled={busy} onClick={() => { spokenAlerts.current.delete(criticalHold.id); void autoSpeakAlert(criticalHold); }}>Replay voice alert</button></div></motion.section>}
      {voiceStatus && <section className="voice-alert panel" aria-label="Spoken quality inspection alert"><Volume2 size={19} aria-hidden="true" /><strong>Voice alert</strong><span role="status">{voiceStatus}</span>{alertAudio && <audio ref={alertPlayer} controls src={alertAudio} aria-label="ElevenLabs inspection alert" />}</section>}
      {busy && <div className="processing-banner" role="status"><LoaderCircle size={19} className="is-spinning" /><span>{progress || 'Working with inspection evidence…'}</span></div>}
      {tab === 'inspection' && <ProductionLine stopped={!!criticalHold} onCriticalDemo={criticalDemo} busy={busy || loading} />}
      <AnimatePresence mode="wait" initial={false}><motion.div key={tab} className="tab-body" initial={{ opacity: 0, y: reduceMotion ? 0 : 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, transition: { duration: .1 } }} transition={{ duration: reduceMotion ? 0 : .2, ease: [0.16, 1, 0.3, 1] }}>{loading && tab !== 'history' ? <div className="empty" role="status">Loading inspection history…</div> : tab === 'history' ? <HistorySensors /> : tab === 'camera' ? <section className="camera panel"><h2>Capture quality check</h2><p>Check blur and bright reflections with your webcam, then run a trained proxy model below. Marker calibration is not connected yet.</p><video ref={video} autoPlay playsInline muted aria-label="Webcam preview" /><div className="buttons"><button className="primary" onClick={startCamera}>Start camera</button><button onClick={capture} disabled={busy || !cameraActive}>Check frame quality</button><button disabled={!cameraActive} onClick={() => { stream.current?.getTracks().forEach(t => t.stop()); stream.current = null; setCameraActive(false); setCameraStatus('Camera stopped.'); }}>Stop camera</button></div><p role="status">{cameraStatus}</p>{quality && <dl className="measurements"><div><dt>Quality gate</dt><dd>{quality.passed ? 'Pass' : 'Recapture'}</dd></div><div><dt>Laplacian variance</dt><dd>{quality.laplacian_variance}</dd></div><div><dt>Bright pixel fraction</dt><dd>{percent(quality.specular_fraction)}</dd></div></dl>}<p className="footnote">Thresholds are provisional and require validation for your camera and lighting.</p><CameraInference grabFrame={grabFrame} cameraActive={cameraActive} patchcoreModel={castingModel} /></section> : tab === 'lab' ? <ProcessLab /> : tab === 'train' ? <TrainPatchCore onTrained={name => void loadModels(name)} /> : !current ? <section className="empty"><h2>Ready for the first inspection</h2><p>Choose a model above, then upload an image. It runs through the quality gate, the model, severity rules, root-cause and risk analytics, and an engineer decision.</p></section> : tab === 'audit' ? <section className="panel audit"><h2>Inspection evidence</h2><dl><dt>Inspection ID</dt><dd>{current.id}</dd><dt>Image SHA-256</dt><dd className="hash">{current.image_sha256}</dd><dt>Source</dt><dd>{current.source === 'uploaded_image' ? (current.context?.input_source === 'camera' ? 'Camera capture with real proxy-model output; process telemetry is a simulated preset' : 'Uploaded image with real proxy-model output; process telemetry is a simulated preset') : 'Synthetic geometry and process telemetry'}</dd></dl><h2>Decision history</h2><table><thead><tr><th>Time</th><th>Event</th><th>Engineer</th></tr></thead><tbody>{current.audit.map((event, i) => <tr key={i}><td>{new Date(event.at).toLocaleString()}</td><td>{human(event.event)}</td><td>{event.engineer || (current.source === 'uploaded_image' ? 'Upload pipeline' : 'Replay system')}</td></tr>)}</tbody></table><a className="download" href={`/api/inspections/${current.id}`} target="_blank" rel="noreferrer">Open full evidence JSON</a></section> : <>
        <div className="part-strip"><div><span>Part</span><strong>{current.part_id}</strong></div><div><span>Lot</span><strong>{current.lot_id}</strong></div><div><span>Machine</span><strong>{current.machine_id}</strong></div><div><span>Inspected</span><strong>{new Date(current.created_at).toLocaleTimeString()}</strong></div>{current.context?.inference?.mode === 'sliced' && <div><span>YOLO inference</span><strong>SAHI · {current.context.inference.tile_count} tiles</strong></div>}<span className={`status ${severity}`}>{human(severity)}</span></div>
        <details className="evidence-disclosure pipeline-disclosure"><summary>Pipeline evidence <span>9 inspection stages</span></summary><PipelineTrace inspection={current as unknown as Parameters<typeof PipelineTrace>[0]['inspection']} onJump={jumpTo} /></details>
        <div className="inspection-grid">
          <section className="panel component" data-module="quality model measure severity">{current.source === 'uploaded_image' ? <UploadedPanel inspection={current as unknown as UploadedInspection} /> : <><div className="panel-heading"><h2>Component inspection</h2><span>Rotor fixture</span></div><img src={current.image_url} width={512} height={512} alt={`Synthetic rotor ${defect ? `with highlighted ${human(defect.label)}` : 'without highlighted defects'}`} /><p className="caption">Illustrative geometry · detections supplied by replay</p><div className="finding"><h3>{defect ? human(defect.label) : current.quality.passed ? 'No defect in replay' : 'Image quality rejected'}</h3><p>{defect?.severity.reason || (current.quality.passed ? 'This synthetic fixture contains no defect.' : 'Recapture before passing an image to the models.')}</p></div>{defect && defect.length_mm !== undefined && <dl className="measurements"><div><dt>Length</dt><dd>{defect.length_mm.toFixed(2)} mm</dd></div><div><dt>Radial zone</dt><dd>{human(defect.zone)}</dd></div><div><dt>Radius</dt><dd>{defect.r_mm} mm</dd></div><div><dt>Angle</dt><dd>{defect.theta_deg}°</dd></div></dl>}<p className="footnote">Demo severity rules and synthetic scale; engineering validation pending.</p></>}</section>
          <div className="evidence-stack"><section className="panel evidence" data-module="cause"><h2>Probable cause</h2><h3 className="hypothesis">{current.analytics ? human(current.analytics.rca.hypothesis) : 'Awaiting valid capture'}</h3><p className="muted">Model hypothesis / synthetic process data</p>{current.analytics?.rca.confidence != null && <p className="muted">Classifier confidence: {percent(current.analytics.rca.confidence)} (uncalibrated)</p>}<details className="evidence-disclosure"><summary>Process & explanation</summary><dl className="telemetry">{Object.entries(current.telemetry).map(([key, value]) => <div key={key}><dt>{featureName(key)}</dt><dd>{value}{FEATURES[key] ? ` ${FEATURES[key][1]}` : ''}</dd></div>)}</dl>{current.context && <p className="sim-note">{current.context.telemetry_source}</p>}<h3>Feature contributions</h3>{current.analytics?.rca.feature_contributions.map(item => <div className="contribution" key={item.feature}><div><span>{featureName(item.feature)}</span><strong>{item.contribution !== undefined ? `${item.contribution >= 0 ? '+' : ''}${item.contribution.toFixed(3)} margin` : `+${((item.heuristic_risk_contribution ?? 0) * 100).toFixed(1)} pp`}</strong></div><div className="bar"><i style={{ width: `${Math.min(100, Math.abs(item.contribution ?? (item.heuristic_risk_contribution ?? 0) * 2) * 20)}%` }} /></div></div>)}<p className="footnote">{current.analytics?.rca.method}. Confidence is uncalibrated. Contributions explain the prediction; they do not establish causality.</p></details></section>
          <section className="panel decision" data-module="forecast risk"><h2>Next-lot risk</h2>{current.analytics ? <><div className="risk"><strong><AnimatedValue value={current.analytics.forecast.risk} /></strong><span>simulated expected defect fraction</span></div><details className="evidence-disclosure"><summary>Forecast details</summary><dl className="risk-details"><div><dt>95% simulated interval</dt><dd>{current.analytics.uncertainty.interval_95.map(percent).join(' – ')}</dd></div><div><dt>Risk exceeds 50%</dt><dd>{percent(current.analytics.uncertainty.breach_probability)}</dd></div></dl><p className="footnote">{current.analytics.forecast.method} / {current.analytics.uncertainty.simulations.toLocaleString()} simulations</p></details></> : <p>Forecast blocked by capture quality.</p>}<div className="action" data-module="action verify"><h2>Recommended action</h2><p>{current.action.text.split(/(?<=\.)\s+/)[0]}</p><details className="evidence-disclosure"><summary>Full recommendation</summary><p>{current.action.text}</p></details><span className={`status ${current.action.status}`}>{human(current.action.status)}</span></div>{current.action.status === 'pending' && <form onSubmit={e => { e.preventDefault(); decision('approve'); }}><label htmlFor="engineer">Engineer name</label><input id="engineer" autoComplete="name" value={engineer} onChange={e => setEngineer(e.target.value)} placeholder="Enter your name" minLength={2} maxLength={100} required /><label htmlFor="note">Decision note</label><textarea id="note" value={note} onChange={e => setNote(e.target.value)} placeholder="Reason or follow-up" maxLength={1000} /><div className="buttons"><button className="primary" disabled={busy || engineer.trim().length < 2}>Approve</button><button type="button" onClick={() => decision('reject')} disabled={busy || engineer.trim().length < 2}>Reject</button><button type="button" onClick={() => decision('escalate')} disabled={busy || engineer.trim().length < 2}>Escalate</button></div></form>}{current.action.status === 'approved' && current.source === 'uploaded_image' && <><p>Approval recorded. Upload at least 20 distinct follow-up images of the same kind (after the corrective action) to verify.</p><label className={`file-button primary-file${busy ? ' disabled' : ''}`}>{busy && progress ? progress : 'Upload 20+ follow-up images'}<input type="file" multiple accept="image/jpeg,image/png" disabled={busy} onChange={e => { verifyFiles(e.target.files); e.target.value = ''; }} /></label></>}{current.verification && <p role="status">{current.source === 'uploaded_image' ? 'Verification on uploaded images' : 'Synthetic verification'}: {current.verification.defective}/{current.verification.parts} defective. {human(current.verification.status)}. Production fix remains unverified.</p>}<p className="footnote">Engineer-controlled recommendation. No machine command is sent.</p></section>
          </div>
        </div>
        {current.source === 'synthetic_replay' && <section className="panel rotor-panel"><div className="panel-heading"><h2>3D rotor view</h2><span>Procedural model · replay measurements</span></div><DeferredRotorViewer defects={current.defects.filter((d): d is Defect & { r_mm: number; theta_deg: number; equivalent_diameter_mm: number } => d.r_mm !== undefined && d.theta_deg !== undefined && d.equivalent_diameter_mm !== undefined)} /></section>}
        {current.source === 'uploaded_image' && current.context?.model === 'casting' && current.context.patchcore_model !== 'mvtec_metal_nut' && (current.context.anomaly?.grid?.length ?? 0) > 0 && <section className="panel rotor-panel"><div className="panel-heading"><div><h2>Inspection heatmap preview</h2><span>PatchCore proxy output</span></div><button className="primary" onClick={() => setArOpen(true)}>Bottle camera heatmap</button></div><p className="muted">Blue is at or below the PatchCore threshold; amber/red is above it. Grid values are model features, not pixel-level defect boundaries.</p><DeferredRotorViewer defects={[]} heatmap={{ grid: current.context.anomaly!.grid, threshold: current.context.anomaly!.threshold }} />{arOpen && <MarkerARViewer initialModel={castingModel} onClose={() => setArOpen(false)} />}</section>}
        <InspectionAssistant key={current.id} inspectionId={current.id} />
        <details className="history panel evidence-disclosure"><summary>Recent inspections <span>{Math.min(records.length, 8)} records</span></summary><div className="table-scroll"><table><thead><tr><th>Part</th><th>Scenario</th><th>Lot / machine</th><th>Disposition</th><th>Action</th><th>Evidence</th></tr></thead><tbody>{records.slice(0, 8).map(record => <tr key={record.id}><td>{record.part_id}</td><td>{record.context?.input_source === 'camera' ? 'Camera scan' : labels[record.scenario] ?? human(record.scenario)}</td><td>{record.lot_id} / {record.machine_id}</td><td>{human(record.disposition)}</td><td>{human(record.action.status)}</td><td><button className="text-button" disabled={busy} onClick={() => setCurrent(record)}>Inspect</button></td></tr>)}</tbody></table></div></details>
      </>}</motion.div></AnimatePresence>
      <footer className="workspace-footer">LineGuard · Evidence-linked quality control <a href="/api/health" target="_blank" rel="noreferrer">System health</a></footer>
    </main>
    <MobileNavigation tab={tab} onSelect={navigate} onScan={openScanner} onPhone={() => setPhoneOpen(true)} onPresenter={startPresenter} busy={busy || loading} />
    {phoneOpen && <PhoneConnect onClose={() => setPhoneOpen(false)} />}
    {scannerOpen && <CameraScanner modelLabel={model === 'casting' ? castingModelLabel : 'YOLO11n / steel surface proxy'} onScan={scanCamera} onClose={() => setScannerOpen(false)} />}
  </div>;
}
