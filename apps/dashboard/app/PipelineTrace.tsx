'use client';

import { motion, useReducedMotion } from 'motion/react';
import { ScanSearch, Radar, Ruler, Scale, GitBranch, TrendingUp, Dices, ClipboardCheck, ShieldCheck, type LucideIcon } from 'lucide-react';

type Tone = 'ok' | 'flag' | 'wait' | 'off';
export type PipelineModule = 'quality' | 'model' | 'measure' | 'severity' | 'cause' | 'forecast' | 'risk' | 'action' | 'verify';
type Node = { id: PipelineModule; icon: LucideIcon; name: string; tool: string; value: string; detail?: string; tone: Tone };

type TraceInspection = {
  id: string;
  source: string;
  quality: { passed: boolean; waived_checks?: string[] };
  defects: { label: string; severity: { level: string }; r_fraction?: number; theta_deg?: number; length_px?: number; length_mm?: number; zone?: string }[];
  calibration?: { mm_per_px?: number | null };
  analytics: null | {
    rca: { hypothesis: string; class_probabilities?: Record<string, number>; feature_contributions: { feature: string; contribution?: number }[] };
    forecast: { risk: number };
    uncertainty: { breach_probability: number; simulations: number; interval_95: number[] };
  };
  action: { status: string };
  verification?: { parts: number; defective: number };
  audit: unknown[];
  context?: { model: 'neu' | 'casting'; patchcore_model?: string | null; anomaly?: { score: number; threshold: number; flagged: boolean } | null; geometry?: object | null };
};

const human = (value: string) => value.replaceAll('_', ' ');
const cap = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);
const FEATURES: Record<string, string> = { temperature_c: 'temperature', pressure_bar: 'pressure', vibration_mm_s: 'vibration', machine_speed_rpm: 'spindle speed' };
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const EASE = [0.16, 1, 0.3, 1] as const;

function buildNodes(r: TraceInspection): Node[] {
  const passed = r.quality.passed;
  const casting = r.context?.model === 'casting';
  const anomaly = r.context?.anomaly;
  const top = r.defects[0];
  const a = r.analytics;
  const driver = a?.rca.feature_contributions[0];
  const probability = a?.rca.class_probabilities?.[a.rca.hypothesis];

  let model: Node;
  if (!passed) model = { id: 'model', icon: Radar, name: 'Defect model', tool: casting ? 'PatchCore' : r.source === 'uploaded_image' ? 'YOLO11n' : 'Replay fixture', value: 'Skipped', detail: 'image rejected first', tone: 'off' };
  else if (r.source !== 'uploaded_image') model = { id: 'model', icon: Radar, name: 'Defect model', tool: 'Replay fixture', value: top ? human(top.label) : 'No defect', detail: 'synthetic detections', tone: top ? 'flag' : 'ok' };
  else if (casting && anomaly) model = { id: 'model', icon: Radar, name: 'Defect model', tool: `PatchCore · ResNet18${r.context?.patchcore_model && r.context.patchcore_model !== 'default' ? ` (${r.context.patchcore_model})` : ''}`, value: `${anomaly.score.toFixed(2)} ${anomaly.flagged ? '>' : '≤'} ${anomaly.threshold.toFixed(2)}`, detail: anomaly.flagged ? 'anomaly flagged' : 'within normal range', tone: anomaly.flagged ? 'flag' : 'ok' };
  else model = { id: 'model', icon: Radar, name: 'Defect model', tool: 'YOLO11n · 6 classes', value: top ? `${r.defects.length} finding${r.defects.length === 1 ? '' : 's'}` : 'No findings', detail: top ? human(top.label) : 'above confidence cut-off', tone: top ? 'flag' : 'ok' };

  const level = top?.severity.level;
  const located = Boolean(r.context?.geometry);
  const scaled = Boolean(r.calibration?.mm_per_px);
  let measure: Node;
  if (!passed) measure = { id: 'measure', icon: Ruler, name: 'Metrology', tool: 'Polar part map', value: 'Skipped', detail: 'image rejected first', tone: 'off' };
  else if (r.source !== 'uploaded_image') measure = { id: 'measure', icon: Ruler, name: 'Metrology', tool: 'Fixture scale', value: top ? 'Measured' : 'Nothing to measure', detail: 'synthetic geometry', tone: 'ok' };
  else if (top?.r_fraction !== undefined) measure = { id: 'measure', icon: Ruler, name: 'Metrology', tool: 'Polar part map', value: `${top.r_fraction.toFixed(2)} R · ${top.theta_deg?.toFixed(0)}°`, detail: `${top.zone}${top.length_mm !== undefined ? ` · ${top.length_mm.toFixed(1)} mm` : top.length_px !== undefined ? ` · ${top.length_px.toFixed(0)} px` : ''}${scaled ? '' : ' · no scale'}`, tone: 'ok' };
  else if (located) measure = { id: 'measure', icon: Ruler, name: 'Metrology', tool: 'Polar part map', value: 'Part located', detail: top ? 'finding has no centre' : 'nothing flagged to place', tone: 'ok' };
  else measure = { id: 'measure', icon: Ruler, name: 'Metrology', tool: 'Polar part map', value: 'No outline', detail: top?.length_px !== undefined ? `size ${top.length_mm !== undefined ? `${top.length_mm.toFixed(1)} mm` : `${top.length_px.toFixed(0)} px`} only` : 'flat or unclear part', tone: 'off' };
  return [
    { id: 'quality', icon: ScanSearch, name: 'Quality gate', tool: 'OpenCV · normalised', value: passed ? 'Pass' : 'Rejected', detail: passed ? (r.quality.waived_checks?.length ? 'dataset limits waived' : 'sharp, exposed, not blank') : 'recapture required', tone: passed ? 'ok' : 'flag' },
    model,
    measure,
    { id: 'severity', icon: Scale, name: 'Severity', tool: 'Deterministic rules', value: !passed ? 'Skipped' : level ? human(level) : 'None', detail: level ? 'by defect class, not confidence' : 'no finding to rate', tone: !passed ? 'off' : level === 'critical' || level === 'high' ? 'flag' : level ? 'wait' : 'ok' },
    { id: 'cause', icon: GitBranch, name: 'Root cause', tool: 'XGBoost + TreeSHAP', value: a ? human(a.rca.hypothesis) : 'Skipped', detail: a ? `${probability !== undefined ? `${pct(probability)} · ` : ''}driver: ${driver ? FEATURES[driver.feature] ?? human(driver.feature) : 'n/a'}` : 'needs a valid capture', tone: !a ? 'off' : a.rca.hypothesis === 'nominal_process' ? 'ok' : 'flag' },
    { id: 'forecast', icon: TrendingUp, name: 'Next-lot forecast', tool: 'PLSR', value: a ? pct(a.forecast.risk) : 'Skipped', detail: a ? 'predicted defect fraction' : 'needs a valid capture', tone: !a ? 'off' : a.forecast.risk >= .35 ? 'flag' : 'ok' },
    { id: 'risk', icon: Dices, name: 'Risk simulation', tool: `Monte Carlo × ${a ? (a.uncertainty.simulations / 1000).toFixed(0) : 10}k`, value: a ? `${pct(a.uncertainty.breach_probability)} over` : 'Skipped', detail: a ? `95%: ${a.uncertainty.interval_95.map(pct).join('–')}` : 'needs a valid capture', tone: !a ? 'off' : a.uncertainty.breach_probability > .05 ? 'flag' : 'ok' },
    { id: 'action', icon: ClipboardCheck, name: 'Action', tool: 'SOP engine + engineer', value: human(r.action.status), detail: r.action.status === 'pending' ? 'awaiting engineer decision' : r.action.status === 'blocked' ? 'recapture first' : 'no machine command sent', tone: r.action.status === 'pending' ? 'wait' : r.action.status === 'blocked' ? 'off' : 'ok' },
    { id: 'verify', icon: ShieldCheck, name: 'Verify & audit', tool: 'Re-inspection · SHA‑256', value: r.verification ? `${r.verification.defective}/${r.verification.parts} defective` : `${r.audit.length} audit event${r.audit.length === 1 ? '' : 's'}`, detail: r.verification ? 'follow-up parts recorded' : 'hash-chained evidence', tone: r.verification ? (r.verification.defective ? 'flag' : 'ok') : 'wait' },
  ];
}

export default function PipelineTrace({ inspection, onJump }: { inspection: TraceInspection; onJump: (module: PipelineModule) => void }) {
  const reduce = useReducedMotion();
  const nodes = buildNodes(inspection);
  const step = reduce ? 0 : 0.075;
  return <section className="pipeline panel" aria-label="Inspection pipeline">
    <div className="pipeline-head">
      <h2>Inspection pipeline</h2>
      <span>Each stage, the model that ran it, and its result for this part. Select a stage to see its evidence.</span>
    </div>
    <ol className="pipeline-track" key={inspection.id}>
      <motion.span className="pipeline-wire" aria-hidden="true" initial={{ scaleX: reduce ? 1 : 0 }} animate={{ scaleX: 1 }} transition={{ duration: reduce ? 0 : nodes.length * step + .2, ease: EASE }} />
      {nodes.map((node, i) => {
        const Icon = node.icon;
        return <motion.li key={node.id} className={`stage ${node.tone}`} initial={reduce ? false : { opacity: 0.25, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * step, duration: reduce ? 0 : .32, ease: EASE }}>
          <button onClick={() => onJump(node.id)} aria-label={`${node.name}, ${node.tool}: ${node.value}. Show evidence.`}>
            <motion.span className="stage-icon" initial={reduce ? false : { scale: .6 }} animate={{ scale: 1 }} transition={{ delay: i * step + .05, duration: reduce ? 0 : .3, ease: EASE }}><Icon size={16} strokeWidth={2} aria-hidden="true" /></motion.span>
            <span className="stage-name">{node.name}</span>
            <span className="stage-tool">{node.tool}</span>
            <strong className="stage-value">{cap(node.value)}</strong>
            {node.detail && <span className="stage-detail">{node.detail}</span>}
          </button>
        </motion.li>;
      })}
    </ol>
  </section>;
}
