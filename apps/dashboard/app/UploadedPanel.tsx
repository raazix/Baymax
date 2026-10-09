'use client';

import { useEffect, useMemo, useState } from 'react';
import { Glasses } from 'lucide-react';
import InspectionImage from './InspectionImage';

type Defect = {
  label: string; confidence?: number; bbox_xyxy_px?: number[]; centroid_px?: number[];
  length_px?: number; width_px?: number; length_mm?: number; width_mm?: number;
  r_px?: number; r_fraction?: number; r_mm?: number; theta_deg?: number; zone?: string;
  severity: { level: string; reason: string; factors?: { factor: string; value: string; effect: string; source?: string }[]; rule_version?: string };
  calibrated_confidence?: number | null; class_hint?: { label: string; confidence: number; inside_region: number };
};
type Merge = { merged: number; unconfirmed: number; rule: string; detections: { label: string; confidence: number; bbox_xyxy_px: number[]; merged_into: number | null }[] };
type Anomaly = { score: number; threshold: number; flagged: boolean; grid: number[][] };
type Geometry = { center_px: number[]; semi_axes_px: number[]; angle_deg: number; rim_radius_px: number };
export type UploadedInspection = {
  image_url: string;
  defects: Defect[];
  quality: { passed: boolean; rejection_reasons?: string[]; profile_note?: string };
  calibration?: { mm_per_px?: number | null; source?: string; part_outline?: string };
  context?: { model: 'neu' | 'casting'; input_source?: 'upload' | 'camera'; part_identity?: { part_type: 'brake_disc' | 'unclassified'; source: string; note: string; model_prediction: boolean }; patchcore_model?: string | null; anomaly?: Anomaly | null; geometry?: Geometry | null; part_shape_hint?: string; routing_note?: string; detection_merge?: Merge | null;
    anomaly_role?: 'primary' | 'view_only' | null; yolo_view?: { role: string; detections: { label: string; confidence: number; calibrated_confidence?: number | null; bbox_xyxy_px: number[] }[] | null } | null;
    corrosion?: { skipped?: boolean; reason?: string; activation_map?: number[][] | null; corrosion_probability: number | null; severe_probability: number | null; calibrated_corrosion_probability?: number | null; calibrated_severe_probability?: number | null; probability_calibration_note?: string; corrosion: boolean; severe: boolean; thresholds: { corrosion: number; severe: number | null; domain?: string } | null } | null };
};

const human = (value: string) => value.replaceAll('_', ' ');
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const LABELS: Record<string, string> = { anomaly_unclassified: 'Anomaly detected', corrosion: 'Corrosion', severe_corrosion: 'Severe corrosion' };
const cap = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

export default function UploadedPanel({ inspection, onOpenAR }: { inspection: UploadedInspection; onOpenAR?: () => void }) {
  const { defects, quality, context, calibration } = inspection;
  const anomaly = context?.anomaly ?? null;
  const geometry = context?.geometry ?? null;
  const casting = context?.model === 'casting';
  const top = defects[0];
  const scaled = Boolean(calibration?.mm_per_px);
  const over = anomaly ? anomaly.grid.flat().filter(v => v > anomaly.threshold).length : 0;
  const cells = anomaly ? anomaly.grid.length * (anomaly.grid[0]?.length ?? 0) : 0;
  const modelName = casting ? `PatchCore · ${context?.patchcore_model && context.patchcore_model !== 'default' ? context.patchcore_model : 'default casting'}` : 'YOLO11n · steel proxy';
  const markers = useMemo(() => geometry ? defects.filter(d => d.centroid_px && d.r_fraction !== undefined).slice(0, 4).map(d => ({
    x: d.centroid_px![0], y: d.centroid_px![1], label: `${d.r_fraction!.toFixed(2)} R · ${d.theta_deg!.toFixed(0)}°`,
  })) : [], [defects, geometry]);

  // Three scans of the same image: PatchCore heatmap, YOLO boxes, corrosion activation map.
  const yoloBoxes = useMemo(() => casting
    ? (context?.detection_merge?.detections ?? []).map(d => ({ label: d.label, confidence: d.confidence, bbox_xyxy_px: d.bbox_xyxy_px, unconfirmed: d.merged_into === null }))
    : (context?.yolo_view?.detections ?? defects.filter(d => d.bbox_xyxy_px && d.confidence !== undefined).map(d => ({ label: d.label, confidence: d.confidence!, calibrated_confidence: d.calibrated_confidence, bbox_xyxy_px: d.bbox_xyxy_px! }))),
  [casting, context?.detection_merge, context?.yolo_view, defects]);
  const corrosion = context?.corrosion ?? null;
  const corrosionMap = corrosion && !corrosion.skipped && corrosion.activation_map ? { grid: corrosion.activation_map, threshold: .5 } : null;
  const views = [
    { id: 'patchcore', name: 'PatchCore', status: !anomaly ? 'not run' : anomaly.flagged ? `flagged ${anomaly.score.toFixed(2)} > ${anomaly.threshold.toFixed(2)}` : `normal ${anomaly.score.toFixed(2)} ≤ ${anomaly.threshold.toFixed(2)}`, tone: anomaly?.flagged ? 'bad' : anomaly ? 'ok' : 'off', ready: Boolean(anomaly), role: context?.anomaly_role === 'view_only' ? 'view only' : 'decides findings' },
    { id: 'yolo', name: 'YOLO boxes', status: !casting || context?.detection_merge ? `${yoloBoxes.length} box${yoloBoxes.length === 1 ? '' : 'es'}${casting && yoloBoxes.length ? ', hints only' : ''}` : 'not run', tone: yoloBoxes.length ? (casting ? 'warn' : 'bad') : 'ok', ready: !casting || Boolean(context?.detection_merge), role: casting ? 'class hints' : 'decides findings' },
    { id: 'corrosion', name: 'Corrosion', status: !corrosion ? 'not run' : corrosion.skipped ? 'skipped: greyscale' : corrosion.severe ? 'severe rust' : corrosion.corrosion ? 'rust found' : 'no rust', tone: corrosion?.corrosion ? 'bad' : corrosion && !corrosion.skipped ? 'ok' : 'off', ready: Boolean(corrosionMap), role: 'adds corrosion findings' },
  ] as const;
  const [view, setView] = useState<'patchcore' | 'yolo' | 'corrosion'>(casting ? 'patchcore' : 'yolo');
  useEffect(() => { setView(casting ? 'patchcore' : 'yolo'); }, [inspection.image_url, casting]);   // each inspection opens on its own detector's view

  let heading: string;
  let explanation: string;
  if (!quality.passed) {
    heading = 'Image quality rejected';
    explanation = `Rejected by the quality gate${quality.rejection_reasons?.length ? `: ${quality.rejection_reasons.map(human).join(', ')}` : ''}. Recapture or use a sharper image.`;
  } else if (top) {
    heading = `${LABELS[top.label] ?? cap(human(top.label))}${defects.length > 1 ? ` + ${defects.length - 1} more` : ''}`;
    explanation = top.severity.reason;
  } else if (casting) {
    heading = 'Within normal range';
    explanation = 'The anomaly score is below the threshold, so nothing was flagged. The heatmap still shows where the part looks least like the normal training parts. This is not proof the part is defect-free.';
  } else {
    heading = 'No defect found';
    explanation = 'The detector found nothing above its confidence cut-off. This is not proof the part is defect-free.';
  }

  const size = top?.length_px !== undefined
    ? scaled && top.length_mm !== undefined ? `${top.length_mm.toFixed(1)} × ${top.width_mm?.toFixed(1)} mm` : `${top.length_px.toFixed(0)} × ${top.width_px?.toFixed(0)} px`
    : null;
  const position = top?.r_fraction !== undefined
    ? `${scaled && top.r_mm !== undefined ? `${top.r_mm.toFixed(1)} mm` : `${top.r_px?.toFixed(0)} px`} · ${top.r_fraction.toFixed(2)} R · ${top.theta_deg?.toFixed(0)}°`
    : null;

  return <>
    <p className="muted" title={context?.part_identity?.note}><strong>{context?.part_identity?.part_type === 'brake_disc' ? 'Brake disc' : 'Unclassified part'}</strong>{context?.part_identity?.part_type === 'brake_disc' ? ' · reference image, operator-declared part type' : ' · part type not identified'}</p>
    <div className="panel-heading"><h2>{context?.input_source === 'camera' ? 'Camera inspection' : 'Component inspection'}</h2><div className="panel-tools"><span>{casting ? 'PatchCore + YOLO11n' : 'YOLO11n'}</span>{onOpenAR && <button type="button" className="ar-button" onClick={onOpenAR}><Glasses size={15} aria-hidden="true" /> Live AR</button>}</div></div>
    <p className="muted">Shape hint: {context?.part_shape_hint === 'disc_like_round_outline' ? 'round-disc-like outline' : 'flat surface or unresolved outline'} / geometric cue only</p>
    <div className="scan-views" role="tablist" aria-label="Model scans of this image">{views.map(v => <button key={v.id} type="button" role="tab" aria-selected={view === v.id} className={`${view === v.id ? 'on' : ''} ${v.tone}`} disabled={!v.ready} onClick={() => setView(v.id)}><strong>{v.name}</strong><span>{v.status}</span><small>{v.role}</small></button>)}</div>
    <InspectionImage key={view} src={inspection.image_url} defects={view === 'yolo' && !casting ? defects : []} anomaly={view === 'patchcore' ? anomaly : view === 'corrosion' ? corrosionMap : null} geometry={view === 'patchcore' ? geometry : null} markers={view === 'patchcore' ? markers : []} hints={view === 'yolo' && casting ? yoloBoxes : []} legend={view === 'corrosion' ? 'corrosion evidence (activation map)' : undefined} alt={top ? `Uploaded image with ${defects.length} model finding${defects.length === 1 ? '' : 's'}` : 'Uploaded image with no model findings'} />
    {view === 'patchcore' && anomaly && <p className="footnote">Heat colors show relative patch distances for this image; red means above the model threshold. This is a coarse map, not a defect boundary. {geometry ? 'Pixels outside the detected disc outline are hidden.' : 'The disc outline was not located, so scene background may affect highlighted areas.'}</p>}
    <div className={`finding${anomaly?.flagged || top ? ' flagged' : ''}`}><h3>{heading}{top?.class_hint && <span className="class-hint"> · YOLO hint: {human(top.class_hint.label)}</span>}</h3>{top && <span className={`status ${top.severity.level}`}>{human(top.severity.level)}</span>}</div>
    {top?.severity.factors && <div className="severity-breakdown" aria-label="How severity was set">
      <ul>{top.severity.factors.map(f => <li key={f.factor} className={f.effect === 'none' ? 'neutral' : ''}><span>{cap(f.factor)}</span><strong>{f.value}</strong><em>{f.effect}</em></li>)}</ul>
      <p>{top.severity.reason.includes('capped') ? top.severity.reason.slice(top.severity.reason.indexOf('capped')).replace(/\.$/, '') + ' · ' : ''}Deterministic {top.severity.rule_version ?? 'rules'}; model confidence and anomaly score are not inputs.</p>
    </div>}
    {defects.length > 1 && <ul className="finding-list" aria-label="All findings on this part">{defects.map((d, i) => <li key={i}>
      <span className={`status ${d.severity.level}`}>{human(d.severity.level)}</span>
      <div><strong>{LABELS[d.label] ?? cap(human(d.label))}{d.class_hint ? ` · YOLO hint: ${human(d.class_hint.label)}` : ''}</strong><span>{d.severity.reason}</span></div>
    </li>)}</ul>}
    {casting && context?.detection_merge && <p className="yolo-note">YOLO11n ran on this image: {context.detection_merge.merged} detection{context.detection_merge.merged === 1 ? '' : 's'} matched a flagged region{context.detection_merge.unconfirmed ? `, ${context.detection_merge.unconfirmed} unconfirmed outside flagged areas (not counted)` : ''}. Its steel-surface classes are hints only on this part type.</p>}
    {casting && anomaly && cells > 0 && over / cells > .5 && <p className="mismatch-note" role="note">{Math.round(over / cells * 100)}% of the image is above the threshold. The selected model ({context?.patchcore_model?.replaceAll('_', ' ') ?? 'default'}) probably does not match this part type; choose a matching model under Advanced before relying on this result.</p>}
    {quality.passed && context?.corrosion && (context.corrosion.skipped || context.corrosion.corrosion_probability === null || !context.corrosion.thresholds
      ? <p className="corrosion-note">Corrosion check skipped: {context.corrosion.reason ?? 'not assessed'}.</p>
      : <p className={`corrosion-note${context.corrosion.corrosion ? ' found' : ''}`}>Corrosion classifier: {context.corrosion.severe ? 'severe corrosion' : context.corrosion.corrosion ? 'corrosion present' : 'no corrosion found'} (score {context.corrosion.corrosion_probability.toFixed(3)}{context.corrosion.calibrated_corrosion_probability != null ? `; calibrated ${context.corrosion.calibrated_corrosion_probability.toFixed(2)}` : ''} vs {context.corrosion.thresholds.domain ? `${context.corrosion.thresholds.domain.replaceAll('_', ' ')} ` : ''}threshold {context.corrosion.thresholds.corrosion.toFixed(3)}{context.corrosion.corrosion && context.corrosion.thresholds.severe !== null && context.corrosion.severe_probability !== null ? `; severe ${context.corrosion.severe_probability.toFixed(2)} vs ${context.corrosion.thresholds.severe.toFixed(2)}` : ''}).{context.corrosion.thresholds.domain ? ' Plate threshold set from normal training plates; severity grade not assessed on plates.' : ' Trained on general rust photos, not brake discs.'}</p>)}
    <p className="muted">Proxy result / not brake-disc validated</p><details className="evidence-disclosure"><summary>Measurement & model details</summary><p>{explanation}</p><p className="muted">{modelName}</p>
    {quality.passed && (top || anomaly) && <dl className="measurements">
      {casting && anomaly && <>
        <div><dt>Anomaly score</dt><dd className={anomaly.flagged ? 'over' : 'under'}>{anomaly.score.toFixed(3)} <span className="dd-sub">threshold {anomaly.threshold.toFixed(3)}</span></dd></div>
        <div><dt>Patches over threshold</dt><dd>{over} of {cells}</dd></div>
      </>}
      {!casting && top && <>
        <div><dt>Findings</dt><dd>{defects.length}</dd></div>
        <div><dt>Top confidence</dt><dd>{top?.calibrated_confidence != null ? percent(top.calibrated_confidence) : percent(Math.max(...defects.map(d => d.confidence ?? 0)))} <span className="dd-sub">{top?.calibrated_confidence != null ? 'NEU validation calibrated · steel proxy' : 'uncalibrated'}</span></dd></div>
      </>}
      {size && <div><dt>{casting ? 'Anomalous region' : 'Largest finding'}</dt><dd>{size}{!scaled && <span className="dd-sub">no scale entered</span>}</dd></div>}
      {top && <div><dt>Radial position</dt><dd>{position ?? 'Not measured'}<span className="dd-sub">{position ? `zone: ${top.zone}` : !casting ? 'flat surface patch, no centre' : geometry ? 'no finding centre' : 'part outline not located'}</span></dd></div>}
    </dl>}
    <p className="footnote">{geometry ? 'The dashed outline is fitted from the image; radial position uses its detected centre. ' : casting ? 'The part outline was not located reliably, so no radial position is given. ' : ''}{scaled ? `Millimetres use ${calibration?.source}. ` : 'No scale supplied; lengths remain in pixels. '}{context?.routing_note || 'Proxy output is not brake-disc validated. Image-relative triage rules do not set production severity.'}{quality.profile_note ? ` ${quality.profile_note}` : ''}</p></details>
  </>;
}
