'use client';

import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { ScanSearch, FileCheck2, Camera, BrainCircuit, SlidersHorizontal, Activity, Gauge, Smartphone, Upload, ShieldCheck, LoaderCircle, AlertTriangle, Volume2, Bell, BellRing } from 'lucide-react';
import { holdLevel } from './qualityHold';
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
import MachineContext from './MachineContext';
import ProcessPanel from './ProcessPanel';
import LineRisk from './LineRisk';

const MarkerARViewer = dynamic(() => import('./MarkerARViewer'), { ssr: false });
const CameraScanner = dynamic(() => import('./CameraScanner'), { ssr: false });
const ProductionLine = dynamic(() => import('./ProductionLine'), { ssr: false });
const PhoneConnect = dynamic(() => import('./PhoneConnect'), { ssr: false });

type Defect = { label: string; confidence?: number; length_mm?: number; equivalent_diameter_mm?: number; r_mm?: number; theta_deg?: number; zone: string; severity: { level: string; reason: string }; bbox_xyxy_px?: number[]; length_px?: number; anomaly_score?: number; threshold?: number; anomaly_grid?: number[][]; cells_over_threshold?: number };
type Inspection = {
  id: string; part_id: string; lot_id: string; machine_id: string; created_at: string;
  scenario: string; source: string; image_url: string; image_sha256: string;
  context?: { model: 'neu' | 'casting'; input_source?: 'upload' | 'camera'; process_context: string; telemetry_source: string; patchcore_model?: string | null; lot_history?: Parameters<typeof ProcessPanel>[0]['history']; anomaly?: { score: number; threshold: number; flagged: boolean; grid: number[][] } | null; inference?: { mode: 'full' | 'sliced'; engine?: string; tile_size_px?: number; overlap_ratio?: number; tile_count?: number; inference_ms?: number } | null };
  disposition: string; quality: { passed: boolean; rejection_reasons?: string[]; profile?: string; profile_note?: string }; defects: Defect[];
  telemetry: Record<string, number>;
  analytics: null | { rca: { hypothesis: string; confidence: number | null; method: string; feature_contributions: { feature: string; heuristic_risk_contribution?: number; contribution?: number }[] }; forecast: { risk: number; method: string }; uncertainty: { interval_95: number[]; breach_probability: number; simulations: number } };
  action: { text: string; status: string; required: boolean };
  verification?: { status: string; parts: number; defective: number };
  audit: { at: string; event: string; engineer?: string }[];
};
type WorkspaceTab = 'inspection' | 'risk' | 'audit' | 'camera' | 'train' | 'lab' | 'history';
const workspaceLabels: Record<WorkspaceTab, string> = { inspection: 'Inspections', risk: 'Production risk', audit: 'Evidence & audit', camera: 'Camera station', train: 'Model training', history: 'Sensor history', lab: 'Process lab' };
const labels: Record<string, string> = { upload_neu: 'Upload · steel (YOLO)', upload_casting: 'Upload · casting (PatchCore)', thermal_drift: 'Thermal drift · critical crack', cosmetic_scratch: 'Cosmetic scratch', normal: 'Normal component', unknown_anomaly: 'Unclassified anomaly', blurred: 'Rejected capture' };
const human = (value: string) => value.replaceAll('_', ' ');
const FEATURES: Record<string, [string, string]> = { temperature_c: ['Temperature', '°C'], pressure_bar: ['Pressure', 'bar'], vibration_mm_s: ['Vibration', 'mm/s'], machine_speed_rpm: ['Spindle speed', 'rpm'] };
const featureName = (key: string) => FEATURES[key]?.[0] ?? human(key);
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

const RECENT_INSPECTION_LIMIT = 25;
async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, { method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store' });
  const text = await response.text();
  let payload: unknown;
  try { payload = text ? JSON.parse(text) : {}; }
  catch {
    if (response.ok) throw new Error('The server response was incomplete. Check the connection and try again.');
    throw new Error(`Request failed (${response.status}). Check the connection and try again.`);
  }
  if (!response.ok) {
    throw new Error(typeof payload === 'object' && payload !== null && 'detail' in payload && typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}). Check FastAPI is running on port 8000.`);
  }
  return payload as T;
}

export default function Dashboard() {
  const [records, setRecords] = useState<Inspection[]>([]);
  const [current, setCurrent] = useState<Inspection | null>(null);
  const [model, setModel] = useState<'casting' | 'neu'>('casting');
  const [inferenceMode, setInferenceMode] = useState<'full' | 'sliced'>('full');
  const [tileSize, setTileSize] = useState(512);
  const [tileOverlap, setTileOverlap] = useState(0.2);
  const [processContext, setProcessContext] = useState<'history' | 'nominal' | 'thermal_drift'>('history');
  const [machine, setMachine] = useState('M-02');
  const [lineOpen, setLineOpen] = useState(false);
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
  const [qualityHold, setQualityHold] = useState<Inspection | null>(null);
  const releasedHolds = useRef(new Set<string>());
  const observedHolds = useRef(new Map<string, Inspection>());
  const notifiedHolds = useRef(new Set<string>());
  const [notificationPermission, setNotificationPermission] = useState<NotificationPermission | 'unsupported'>('default');
  useEffect(() => { setNotificationPermission('Notification' in window ? Notification.permission : 'unsupported'); }, []);
  async function enableNotifications() {
    if (!('Notification' in window)) return;
    setNotificationPermission(await Notification.requestPermission());
  }
  const spokenAlerts = useRef(new Set<string>());
  const voiceRequest = useRef<AbortController | null>(null);
  const alertPlayer = useRef<HTMLAudioElement>(null);
  useEffect(() => {
    if (current && holdLevel(current) && !releasedHolds.current.has(current.id)) {
      const newlyObserved = !observedHolds.current.has(current.id);
      observedHolds.current.set(current.id, current);
      setQualityHold(previous => previous?.id === current.id || !previous || (holdLevel(current) === 'critical' && holdLevel(previous) !== 'critical') ? current : previous);
      setLineOpen(true);
      if (newlyObserved) toast.error(`${holdLevel(current) === 'critical' ? 'Critical' : 'High severity'} finding — line stopped`, {
        id: `hold-${current.id}`, description: `Machine ${current.machine_id} · lot ${current.lot_id}. Engineer review required.`, duration: 10000,
      });
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
    if (!current || !holdLevel(current) || notificationPermission !== 'granted' || notifiedHolds.current.has(current.id)) return;
    try {
      const notification = new Notification('LineGuard · production simulation stopped', {
        body: `${holdLevel(current) === 'critical' ? 'Critical' : 'High severity'} finding on ${current.part_id}. Machine ${current.machine_id}, lot ${current.lot_id}.`,
        icon: '/icon.svg', tag: `quality-${current.id}`,
      });
      notification.onclick = () => { window.focus(); setTab('inspection'); notification.close(); };
      notifiedHolds.current.add(current.id);
    } catch { /* In-app notifications remain available when native notifications are unavailable. */ }
  }, [current, notificationPermission]);

  useEffect(() => { void loadModels(); }, []);
  useEffect(() => {
    let cancelled = false;
    api<Inspection[]>(`inspections?limit=${RECENT_INSPECTION_LIMIT}`).then(data => { if (!cancelled) { setRecords(data); setCurrent(data[0] || null); } }).catch(e => { if (!cancelled) setError(e.message); }).finally(() => { if (!cancelled) setLoading(false); });
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
  async function refresh(result: Inspection) { setCurrent(result); setRecords(await api<Inspection[]>(`inspections?limit=${RECENT_INSPECTION_LIMIT}`)); }
  async function autoSpeakAlert(result: Inspection) {
    if (!result.quality.passed || !result.defects.some(item => ['critical', 'high'].includes(item.severity.level)) || spokenAlerts.current.has(result.id)) return;
    spokenAlerts.current.add(result.id);
    voiceRequest.current?.abort();
    const controller = new AbortController(); voiceRequest.current = controller;
    setVoiceStatus('Generating ElevenLabs quality alert...'); setAlertAudio('');
    try {
      const response = await fetch(`/api/inspections/${result.id}/alert-speech`, { method: 'POST', cache: 'no-store', signal: controller.signal });
      if (!response.ok) {
        let detail = '';
        try { detail = (await response.json()).detail || ''; } catch { /* Use a safe fallback for non-JSON proxy errors. */ }
        throw new Error(detail || `ElevenLabs voice alert failed (HTTP ${response.status}). Review the hold and retry.`);
      }
      if (!response.headers.get('content-type')?.startsWith('audio/')) throw new Error('ElevenLabs returned an invalid audio response. Check the backend logs and retry.');
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
    if (!qualityHold || qualityHold.action.status !== 'approved') return;
    releasedHolds.current.add(qualityHold.id);
    setQualityHold([...observedHolds.current.values()].find(record => !releasedHolds.current.has(record.id)) ?? null);
  }
  async function inspectFile(file: File, options: { model: 'casting' | 'neu'; processContext: string; lot?: string; machine?: string; patchcoreModel?: string; inferenceMode?: 'full' | 'sliced'; tileSize?: number; overlap?: number; inputSource?: 'upload' | 'camera'; signal?: AbortSignal }): Promise<Inspection> {
    const params = new URLSearchParams({ model: options.model, process_context: options.processContext });
    if (options.inputSource) params.set('input_source', options.inputSource);
    if (options.model === 'neu') {
      params.set('inference_mode', options.inferenceMode ?? 'full');
      if (options.inferenceMode === 'sliced') { params.set('tile_size', String(options.tileSize ?? 512)); params.set('overlap', String(options.overlap ?? 0.2)); }
    }
    if (options.patchcoreModel) params.set('patchcore_model', options.patchcoreModel);   // YOLO mode also shows a PatchCore scan
    const diameter = Number(partDiameter);
    if (partDiameter.trim() && Number.isFinite(diameter) && diameter > 0) params.set('part_diameter_mm', String(diameter));
    if (options.lot) params.set('lot_id', options.lot);
    params.set('machine_id', options.machine ?? machine);
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
  function demoInspect(step: string, set: string, demoModel: 'casting' | 'neu', demoMachine: string, becomesTarget: boolean, index = 0) {
    void operation(async () => {
      setDemoStatus('Running the real pipeline: quality gate, model, severity, root cause, forecast…');
      const file = await demoFile(set, index);
      const result = await inspectFile(file, { model: demoModel, processContext: 'history', lot: demo.lot, machine: demoMachine, patchcoreModel: 'default' });
      await refresh(result); setTab('inspection'); setModel(demoModel); setMachine(demoMachine);
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
        const result = await inspectFile(await demoFile('verification', i), { model: 'casting', processContext: 'history', lot: demo.lot, machine: 'M-02', patchcoreModel: 'default' });
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
    { title: 'Defective casting', say: 'Inspect a defective casting from machine M-02, whose temperature has drifted over the last week. PatchCore flags it, severity rules rate it, and the lot history explains the likely cause.', run: () => demoInspect('defect', 'defect', 'casting', 'M-02', true, 3), done: demo.done.includes('defect') },
    { title: 'Approve action', say: 'XGBoost points to thermal drift from the lot history; an engineer approves, and the approval is remembered in Supermemory.', run: demoApprove, done: demo.done.includes('approve'), disabled: !demo.target },
    { title: 'Verify 20 parts', say: 'Re-inspect 20 follow-up parts to close the loop.', run: demoVerify, done: demo.done.includes('verify'), disabled: !demo.done.includes('approve') },
    { title: 'Audit trail', say: 'Show the tamper-evident record of every step.', run: demoAudit, done: demo.done.includes('audit'), disabled: !demo.target },
    { title: 'Normal casting', say: 'Contrast: a good part stays below the threshold, and the heatmap is mostly clear.', run: () => demoInspect('normal', 'normal', 'casting', 'M-01', false), done: demo.done.includes('normal') },
    { title: 'Steel defects', say: 'Second model: YOLO finds and labels steel-surface defects.', run: () => demoInspect('steel', 'steel', 'neu', 'M-03', false), done: demo.done.includes('steel') },
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
      <nav aria-label="Workspace">{(['inspection', 'risk', 'audit', 'camera', 'train', 'lab', 'history'] as const).map(item => { const Icon = { inspection: ScanSearch, risk: Gauge, audit: FileCheck2, camera: Camera, train: BrainCircuit, lab: SlidersHorizontal, history: Activity }[item]; return <button key={item} aria-label={workspaceLabels[item]} title={workspaceLabels[item]} className={tab === item ? 'nav active' : 'nav'} aria-current={tab === item ? 'page' : undefined} onClick={() => navigate(item)}>{tab === item && <motion.span layoutId="nav-active" className="nav-indicator" transition={{ duration: reduceMotion ? 0 : .28, ease: [0.16, 1, 0.3, 1] }} aria-hidden="true" />}<Icon size={17} strokeWidth={1.9} aria-hidden="true" /><span>{workspaceLabels[item]}</span></button>; })}</nav>
      <div className="sidebar-bottom"><span className="connection-dot" /> Local demo workspace<p>Track 3 · Singularity 2026</p></div><details className="sidebar-details"><summary>Model status</summary><ModelStatus /></details>
    </aside>
    <main>
      <header className="topbar"><div><h1>{workspaceLabels[tab]}</h1><p>Inspect components and review quality decisions.</p></div><div className="topbar-actions"><button className="notification-action" disabled={notificationPermission === 'unsupported' || notificationPermission === 'denied'} title={notificationPermission === 'denied' ? 'Allow notifications in your browser settings. In-app alerts remain active.' : 'Enable system notifications for high and critical findings'} onClick={enableNotifications}>{notificationPermission === 'granted' ? <BellRing size={19} /> : <Bell size={19} />}{notificationPermission === 'granted' ? 'Alerts enabled' : 'Enable alerts'}</button><button className="phone-connect-action" onClick={() => setPhoneOpen(true)}><Smartphone size={19} /> Connect phone</button><span className="demo-tag">Proxy models</span>{!presenter && <button className="primary" onClick={startPresenter}>Presenter mode</button>}</div></header>
      {presenter && <PresenterBar steps={presenterSteps} busy={busy} status={busy && progress ? progress : demoStatus} onClose={() => setPresenter(false)} />}
      {(tab === 'inspection' || tab === 'audit') && <section className="upload-workspace" aria-label="Image inspection">
        <div className="upload-controls">
          <div className="capture-model"><span className="capture-symbol"><ScanSearch size={25} /></span><div><strong>Inspect a component</strong><span>{model === 'casting' ? `${castingModelLabel} · YOLO11n hints` : `YOLO11n steel-surface detector${inferenceMode === 'sliced' ? ` · SAHI ${tileSize}px tiles` : ''}`}</span></div></div>
          <div className="detector-switch" role="radiogroup" aria-label="Inspection model for the next upload"><button type="button" role="radio" aria-checked={model === 'casting'} className={model === 'casting' ? 'on' : ''} disabled={busy} onClick={() => setModel('casting')}>PatchCore heatmap<small>Anomalies + YOLO hints</small></button><button type="button" role="radio" aria-checked={model === 'neu'} className={model === 'neu' ? 'on' : ''} disabled={busy} onClick={() => setModel('neu')}>YOLO detection boxes<small>YOLO11n · 6 classes</small></button></div>
          <button className="scan-camera-action" disabled={busy || loading} onClick={openScanner}><Camera size={17} /> Scan with camera</button>
          <label className={`file-button primary-file${busy || loading ? ' disabled' : ''}`}><Upload size={17} />{busy ? 'Processing image…' : 'Upload image'}<input type="file" accept="image/jpeg,image/png" disabled={busy || loading} onChange={e => { upload(e.target.files); e.target.value = ''; }} /></label>
        </div>
        <div className="capture-context"><MachineContext machine={machine} onChange={setMachine} disabled={busy} /><span className="proxy-chip"><ShieldCheck size={14} /> Industrial-part proxy</span><details><summary>Scope &amp; model limits</summary><p>Brake discs are the target; current models are industrial-part proxies, and none is brake-disc validated. The selected machine's latest lot from imported sensor history feeds the root-cause model and the next-lot forecast; the demo history is synthetic and labelled as such.</p></details></div>
        <details className="model-options"><summary>Advanced · {model === 'casting' ? 'choose a PatchCore model' : 'YOLO tiling (SAHI)'}</summary>{model === 'neu' && <div className="yolo-options"><label>Inference<select value={inferenceMode} onChange={e => setInferenceMode(e.target.value as 'full' | 'sliced')}><option value="full">Full image</option><option value="sliced">SAHI sliced tiles</option></select></label>{inferenceMode === 'sliced' && <><label>Tile size<select value={tileSize} onChange={e => setTileSize(Number(e.target.value))}>{[256, 384, 512, 768, 1024].map(v => <option key={v} value={v}>{v} px</option>)}</select></label><label>Overlap<select value={tileOverlap} onChange={e => setTileOverlap(Number(e.target.value))}>{[.1, .2, .3, .4, .5].map(v => <option key={v} value={v}>{Math.round(v * 100)}%</option>)}</select></label></>}<p>YOLO11n was trained on NEU steel-surface patches (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches); test mAP50 72.4%. Sliced inference helps small defects on large images.</p></div>}{model === 'casting' && <><label>Model for this image<select value={castingModel} onChange={e => setCastingModel(e.target.value)}><optgroup label="MPDD component proxies">{mpddModels.map(item => <option key={item.name} value={item.name}>MPDD · {human(item.category)}</option>)}</optgroup><optgroup label="Custom / benchmark models">{customModels.map(item => <option key={item.name} value={item.name}>{item.name === 'mvtec_metal_nut' ? 'MVTec AD · metal nut' : item.name}</option>)}</optgroup></select></label><p>{castingModel === 'mvtec_metal_nut' ? 'MVTec metal-nut test: 81/93 defect images detected (87.1% recall), 12 missed, 0/22 normal false alarms; image AUROC 0.989. This result applies to MVTec nuts, not brake discs.' : castingModel === 'mpdd_metal_plate' ? 'MPDD metal-plate proxy: test recall 38.0% (27/71 defects detected; 44 missed). It is not validated on brake discs.' : 'Choose a model matching the inspected component. No available model is validated on brake discs.'}</p></>}</details>
      </section>}
      {error && <div className="error" role="alert">{error}<button onClick={() => void operation(async () => { const data = await api<Inspection[]>(`inspections?limit=${RECENT_INSPECTION_LIMIT}`); setRecords(data); setCurrent(data[0] || null); })} disabled={busy}>Reconnect</button></div>}
      {qualityHold && <motion.section layout className="production-hold" role="alert" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><AlertTriangle size={24} aria-hidden="true" /><div className="hold-text"><strong>Production simulation stopped · {holdLevel(qualityHold) === 'critical' ? 'critical' : 'high severity'} finding</strong><span>Part {qualityHold.part_id} · lot {qualityHold.lot_id} · machine {qualityHold.machine_id}. Review and approve the action before resuming.</span></div><div className="buttons">{current?.id !== qualityHold.id && <button onClick={() => { setCurrent(qualityHold); setTab('inspection'); }}>Review part</button>}<button onClick={() => { setLineOpen(true); setTimeout(() => document.getElementById('production-line')?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' }), 50); }}>View line</button><button disabled={busy || qualityHold.action.status !== 'approved'} onClick={resumeSimulation}>Resume after approval</button></div></motion.section>}
      {voiceStatus && <section className="voice-alert" aria-label="Spoken quality alert"><Volume2 size={17} aria-hidden="true" /><span role="status">{voiceStatus}</span>{alertAudio && <audio ref={alertPlayer} controls src={alertAudio} aria-label="ElevenLabs inspection alert" />}{qualityHold && <button className="text-button" disabled={busy || voiceStatus.startsWith('Generating')} onClick={() => { spokenAlerts.current.delete(qualityHold.id); void autoSpeakAlert(qualityHold); }}>{alertAudio ? 'Replay' : 'Retry voice alert'}</button>}</section>}
      {busy && <div className="processing-banner" role="status"><LoaderCircle size={19} className="is-spinning" /><span>{progress || 'Working with inspection evidence…'}</span></div>}
      <AnimatePresence mode="wait" initial={false}><motion.div key={tab} className="tab-body" initial={{ opacity: 0, y: reduceMotion ? 0 : 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, transition: { duration: .1 } }} transition={{ duration: reduceMotion ? 0 : .2, ease: [0.16, 1, 0.3, 1] }}>{loading && tab !== 'history' && tab !== 'risk' ? <div className="empty" role="status">Loading inspection history…</div> : tab === 'history' ? <HistorySensors /> : tab === 'risk' ? <LineRisk records={records} onInspect={m => { setMachine(m); setTab('inspection'); window.scrollTo({ top: 0, behavior: 'auto' }); }} /> : tab === 'camera' ? <section className="camera panel"><h2>Capture quality check</h2><p>Check blur and bright reflections with your webcam, then run a trained proxy model below. Marker calibration is not connected yet.</p><video ref={video} autoPlay playsInline muted aria-label="Webcam preview" /><div className="buttons"><button className="primary" onClick={startCamera}>Start camera</button><button onClick={capture} disabled={busy || !cameraActive}>Check frame quality</button><button disabled={!cameraActive} onClick={() => { stream.current?.getTracks().forEach(t => t.stop()); stream.current = null; setCameraActive(false); setCameraStatus('Camera stopped.'); }}>Stop camera</button></div><p role="status">{cameraStatus}</p>{quality && <dl className="measurements"><div><dt>Quality gate</dt><dd>{quality.passed ? 'Pass' : 'Recapture'}</dd></div><div><dt>Laplacian variance</dt><dd>{quality.laplacian_variance}</dd></div><div><dt>Bright pixel fraction</dt><dd>{percent(quality.specular_fraction)}</dd></div></dl>}<p className="footnote">Thresholds are provisional and require validation for your camera and lighting.</p><CameraInference grabFrame={grabFrame} cameraActive={cameraActive} patchcoreModel={castingModel} /></section> : tab === 'lab' ? <ProcessLab /> : tab === 'train' ? <TrainPatchCore onTrained={name => void loadModels(name)} /> : !current ? <section className="empty"><h2>Ready for the first inspection</h2><p>Choose a model above, then upload an image. It runs through the quality gate, the model, severity rules, root-cause and risk analytics, and an engineer decision.</p></section> : tab === 'audit' ? <section className="panel audit"><h2>Inspection evidence</h2><dl><dt>Inspection ID</dt><dd>{current.id}</dd><dt>Image SHA-256</dt><dd className="hash">{current.image_sha256}</dd><dt>Source</dt><dd>{current.source === 'uploaded_image' ? `${current.context?.input_source === 'camera' ? 'Camera capture' : 'Uploaded image'} with real proxy-model output. Process data: ${current.context?.telemetry_source ?? 'not recorded'}` : 'Synthetic geometry and process telemetry'}</dd></dl><h2>Decision history</h2><table><thead><tr><th>Time</th><th>Event</th><th>Engineer</th></tr></thead><tbody>{current.audit.map((event, i) => <tr key={i}><td>{new Date(event.at).toLocaleString()}</td><td>{human(event.event)}</td><td>{event.engineer || (event.event === 'memory_write' ? 'Supermemory' : current.source === 'uploaded_image' ? 'Upload pipeline' : 'Replay system')}</td></tr>)}</tbody></table><a className="download" href={`/api/inspections/${current.id}`} target="_blank" rel="noreferrer">Open full evidence JSON</a></section> : <>
        <div className="part-strip"><div><span>Part</span><strong>{current.part_id}</strong></div><div><span>Lot</span><strong>{current.lot_id}</strong></div><div><span>Machine</span><strong>{current.machine_id}</strong></div><div><span>Inspected</span><strong>{new Date(current.created_at).toLocaleTimeString()}</strong></div>{current.context?.inference?.mode === 'sliced' && <div><span>YOLO inference</span><strong>SAHI · {current.context.inference.tile_count} tiles</strong></div>}<span className={`status ${severity}`}>{human(severity)}</span></div>
        <details className="evidence-disclosure pipeline-disclosure"><summary>Pipeline evidence <span>9 inspection stages</span></summary><PipelineTrace inspection={current as unknown as Parameters<typeof PipelineTrace>[0]['inspection']} onJump={jumpTo} /></details>
        <div className="inspection-grid">
          <section className="panel component" data-module="quality model measure severity">{current.source === 'uploaded_image' ? <UploadedPanel inspection={current as unknown as UploadedInspection} onOpenAR={() => setArOpen(true)} /> : <><div className="panel-heading"><h2>Component inspection</h2><span>Rotor fixture</span></div><img src={current.image_url} width={512} height={512} alt={`Synthetic rotor ${defect ? `with highlighted ${human(defect.label)}` : 'without highlighted defects'}`} /><p className="caption">Illustrative geometry · detections supplied by replay</p><div className="finding"><h3>{defect ? human(defect.label) : current.quality.passed ? 'No defect in replay' : 'Image quality rejected'}</h3><p>{defect?.severity.reason || (current.quality.passed ? 'This synthetic fixture contains no defect.' : 'Recapture before passing an image to the models.')}</p></div>{defect && defect.length_mm !== undefined && <dl className="measurements"><div><dt>Length</dt><dd>{defect.length_mm.toFixed(2)} mm</dd></div><div><dt>Radial zone</dt><dd>{human(defect.zone)}</dd></div><div><dt>Radius</dt><dd>{defect.r_mm} mm</dd></div><div><dt>Angle</dt><dd>{defect.theta_deg}°</dd></div></dl>}<p className="footnote">Demo severity rules and synthetic scale; engineering validation pending.</p></>}</section>
          <div className="evidence-stack"><ProcessPanel machine={current.machine_id} lot={current.lot_id} telemetry={current.telemetry} telemetrySource={current.context?.telemetry_source} history={current.context?.lot_history ?? null} rca={current.analytics?.rca ?? null} defects={current.defects.map(d => d.label)} />
          <section className="panel decision" data-module="forecast risk"><h2>Next-lot risk</h2>{current.analytics ? <><div className="risk"><strong><AnimatedValue value={current.analytics.forecast.risk} /></strong><span>simulated expected defect fraction</span></div><details className="evidence-disclosure"><summary>Forecast details</summary><dl className="risk-details"><div><dt>95% simulated interval</dt><dd>{current.analytics.uncertainty.interval_95.map(percent).join(' – ')}</dd></div><div><dt>Risk exceeds 50%</dt><dd>{percent(current.analytics.uncertainty.breach_probability)}</dd></div></dl><p className="footnote">{current.analytics.forecast.method} / {current.analytics.uncertainty.simulations.toLocaleString()} simulations</p></details></> : <p>Forecast blocked by capture quality.</p>}<div className="action" data-module="action verify"><h2>Recommended action</h2><p>{current.action.text.split(/(?<=\.)\s+/)[0]}</p><details className="evidence-disclosure"><summary>Full recommendation</summary><p>{current.action.text}</p></details><span className={`status ${current.action.status}`}>{human(current.action.status)}</span></div>{current.action.status === 'pending' && <form onSubmit={e => { e.preventDefault(); decision('approve'); }}><label htmlFor="engineer">Engineer name</label><input id="engineer" autoComplete="name" value={engineer} onChange={e => setEngineer(e.target.value)} placeholder="Enter your name" minLength={2} maxLength={100} required /><label htmlFor="note">Decision note</label><textarea id="note" value={note} onChange={e => setNote(e.target.value)} placeholder="Reason or follow-up" maxLength={1000} /><div className="buttons"><button className="primary" disabled={busy || engineer.trim().length < 2}>Approve</button><button type="button" onClick={() => decision('reject')} disabled={busy || engineer.trim().length < 2}>Reject</button><button type="button" onClick={() => decision('escalate')} disabled={busy || engineer.trim().length < 2}>Escalate</button></div></form>}{current.action.status === 'approved' && current.source === 'uploaded_image' && <><p>Approval recorded. Upload at least 20 distinct follow-up images of the same kind (after the corrective action) to verify.</p><label className={`file-button primary-file${busy ? ' disabled' : ''}`}>{busy && progress ? progress : 'Upload 20+ follow-up images'}<input type="file" multiple accept="image/jpeg,image/png" disabled={busy} onChange={e => { verifyFiles(e.target.files); e.target.value = ''; }} /></label></>}{current.verification && <p role="status">{current.source === 'uploaded_image' ? 'Verification on uploaded images' : 'Synthetic verification'}: {current.verification.defective}/{current.verification.parts} defective. {human(current.verification.status)}. Production fix remains unverified.</p>}<p className="footnote">Engineer-controlled recommendation. No machine command is sent.</p></section>
          </div>
        </div>
        {current.source === 'synthetic_replay' && <section className="panel rotor-panel"><div className="panel-heading"><h2>3D rotor view</h2><span>Procedural model · replay measurements</span></div><DeferredRotorViewer defects={current.defects.filter((d): d is Defect & { r_mm: number; theta_deg: number; equivalent_diameter_mm: number } => d.r_mm !== undefined && d.theta_deg !== undefined && d.equivalent_diameter_mm !== undefined)} /></section>}
        <InspectionAssistant key={current.id} inspectionId={current.id} />
        <section className="tools-row" aria-label="More views">
          {current.source === 'uploaded_image' && current.context?.model === 'casting' && current.context.patchcore_model !== 'mvtec_metal_nut' && (current.context.anomaly?.grid?.length ?? 0) > 0 && <details className="panel evidence-disclosure"><summary>3D heatmap preview <span>PatchCore grid on a procedural rotor</span></summary><p className="muted">Blue is at or below the PatchCore threshold; amber/red is above it. Grid values are model features, not pixel-level defect boundaries.</p><DeferredRotorViewer defects={[]} heatmap={{ grid: current.context.anomaly!.grid, threshold: current.context.anomaly!.threshold }} /></details>}
          <details id="production-line" className="panel evidence-disclosure" open={lineOpen || undefined} onToggle={e => setLineOpen((e.target as HTMLDetailsElement).open)}><summary>Production line simulation <span>{qualityHold ? 'held for review' : 'running'}</span></summary>{lineOpen && <ProductionLine stopped={!!qualityHold} onCriticalDemo={criticalDemo} busy={busy || loading} />}</details>
        </section>
        <details className="history panel evidence-disclosure"><summary>Recent inspections <span>{Math.min(records.length, 8)} records</span></summary><div className="table-scroll"><table><thead><tr><th>Part</th><th>Scenario</th><th>Lot / machine</th><th>Disposition</th><th>Action</th><th>Evidence</th></tr></thead><tbody>{records.slice(0, 8).map(record => <tr key={record.id}><td>{record.part_id}</td><td>{record.context?.input_source === 'camera' ? 'Camera scan' : labels[record.scenario] ?? human(record.scenario)}</td><td>{record.lot_id} / {record.machine_id}</td><td>{human(record.disposition)}</td><td>{human(record.action.status)}</td><td><button className="text-button" disabled={busy} onClick={() => setCurrent(record)}>Inspect</button></td></tr>)}</tbody></table></div></details>
      </>}</motion.div></AnimatePresence>
      <footer className="workspace-footer">LineGuard · Evidence-linked quality control <a href="/api/health" target="_blank" rel="noreferrer">System health</a></footer>
    </main>
    <MobileNavigation tab={tab} onSelect={navigate} onScan={openScanner} onPhone={() => setPhoneOpen(true)} onPresenter={startPresenter} busy={busy || loading} />
    {phoneOpen && <PhoneConnect onClose={() => setPhoneOpen(false)} />}
    {arOpen && <MarkerARViewer initialModel={current?.context?.patchcore_model ?? castingModel} inspectionModel={current?.context?.model ?? model} machineId={current?.machine_id ?? machine} onInspection={result => refresh(result as unknown as Inspection)} onClose={() => setArOpen(false)} />}
    {scannerOpen && <CameraScanner modelLabel={model === 'casting' ? castingModelLabel : 'YOLO11n / steel surface proxy'} onScan={scanCamera} onClose={() => setScannerOpen(false)} />}
  </div>;
}
