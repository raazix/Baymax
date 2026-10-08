'use client';

import { useMemo } from 'react';
import InspectionImage from './InspectionImage';

type Defect = {
  label: string; confidence?: number; bbox_xyxy_px?: number[]; centroid_px?: number[];
  length_px?: number; width_px?: number; length_mm?: number; width_mm?: number;
  r_px?: number; r_fraction?: number; r_mm?: number; theta_deg?: number; zone?: string;
  severity: { level: string; reason: string };
};
type Anomaly = { score: number; threshold: number; flagged: boolean; grid: number[][] };
type Geometry = { center_px: number[]; semi_axes_px: number[]; angle_deg: number; rim_radius_px: number };
export type UploadedInspection = {
  image_url: string;
  defects: Defect[];
  quality: { passed: boolean; rejection_reasons?: string[]; profile_note?: string };
  calibration?: { mm_per_px?: number | null; source?: string; part_outline?: string };
  context?: { model: 'neu' | 'casting'; patchcore_model?: string | null; anomaly?: Anomaly | null; geometry?: Geometry | null; part_shape_hint?: string; routing_note?: string };
};

const human = (value: string) => value.replaceAll('_', ' ');
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const LABELS: Record<string, string> = { anomaly_unclassified: 'Anomaly detected' };
const cap = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

export default function UploadedPanel({ inspection }: { inspection: UploadedInspection }) {
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
    <div className="panel-heading"><h2>Component inspection</h2><span>{casting ? 'PatchCore' : 'YOLO11n'}</span></div>
    <p className="muted">Shape hint: {context?.part_shape_hint === 'disc_like_round_outline' ? 'round-disc-like outline' : 'flat surface or unresolved outline'} ? geometric cue only</p>
    <InspectionImage src={inspection.image_url} defects={defects} anomaly={anomaly} geometry={geometry} markers={markers} alt={top ? `Uploaded image with ${defects.length} model finding${defects.length === 1 ? '' : 's'}` : 'Uploaded image with no model findings'} />
    <div className={`finding${anomaly?.flagged || top ? ' flagged' : ''}`}><h3>{heading}</h3>{top && <span className={`status ${top.severity.level}`}>{human(top.severity.level)}</span>}</div><p className="muted">Proxy result / not brake-disc validated</p><details className="evidence-disclosure"><summary>Measurement & model details</summary><p>{explanation}</p><p className="muted">{modelName}</p>
    {quality.passed && (top || anomaly) && <dl className="measurements">
      {casting && anomaly && <>
        <div><dt>Anomaly score</dt><dd className={anomaly.flagged ? 'over' : 'under'}>{anomaly.score.toFixed(3)} <span className="dd-sub">threshold {anomaly.threshold.toFixed(3)}</span></dd></div>
        <div><dt>Patches over threshold</dt><dd>{over} of {cells}</dd></div>
      </>}
      {!casting && top && <>
        <div><dt>Findings</dt><dd>{defects.length}</dd></div>
        <div><dt>Top confidence</dt><dd>{percent(Math.max(...defects.map(d => d.confidence ?? 0)))} <span className="dd-sub">uncalibrated</span></dd></div>
      </>}
      {size && <div><dt>{casting ? 'Anomalous region' : 'Largest finding'}</dt><dd>{size}{!scaled && <span className="dd-sub">no scale entered</span>}</dd></div>}
      {top && <div><dt>Radial position</dt><dd>{position ?? 'Not measured'}<span className="dd-sub">{position ? `zone: ${top.zone}` : !casting ? 'flat surface patch, no centre' : geometry ? 'no finding centre' : 'part outline not located'}</span></dd></div>}
    </dl>}
    <p className="footnote">{geometry ? 'The dashed outline is fitted from the image; radial position uses its detected centre. ' : casting ? 'The part outline was not located reliably, so no radial position is given. ' : ''}{scaled ? `Millimetres use ${calibration?.source}. ` : 'No scale supplied; lengths remain in pixels. '}{context?.routing_note || 'Proxy output is not brake-disc validated. Image-relative triage rules do not set production severity.'}{quality.profile_note ? ` ${quality.profile_note}` : ''}</p></details>
  </>;
}
