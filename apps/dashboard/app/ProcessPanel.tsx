'use client';

import { SENSOR_NAMES, driftLabel, type Drift } from './MachineContext';

type Contribution = { feature: string; contribution?: number; heuristic_risk_contribution?: number };
type LotHistory = { lot_window?: string[]; prior_lots?: string[]; drift?: Drift[]; trend?: ({ lot_id: string } & Record<string, number | string>)[]; synthetic?: boolean; readings?: number; units?: Record<string, string | null>; baseline_lots?: number } | null;
type Props = {
  machine: string; lot: string; telemetry: Record<string, number>; telemetrySource?: string; history?: LotHistory;
  rca?: { hypothesis: string; confidence: number | null; method: string; feature_contributions: Contribution[] } | null;
  defects?: string[];
};

const UNITS: Record<string, string> = { temperature_c: '°C', pressure_bar: 'bar', vibration_mm_s: 'mm/s', machine_speed_rpm: 'rpm' };
const human = (value: string) => value.replaceAll('_', ' ');

// Trend of one sensor across recent lots: the last point is the inspected lot.
function Trend({ points, label }: { points: number[]; label: string }) {
  if (points.length < 3) return null;
  const min = Math.min(...points), max = Math.max(...points), span = max - min || 1;
  const x = (i: number) => 2 + i * 196 / (points.length - 1);
  const y = (v: number) => 34 - (v - min) / span * 30;
  const path = points.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  return <svg className="trend" viewBox="0 0 200 38" role="img" aria-label={`${label} across the last ${points.length} lots, from ${points[0].toFixed(1)} to ${points.at(-1)!.toFixed(1)}`}>
    <path d={path} />
    <circle cx={x(points.length - 1)} cy={y(points.at(-1)!)} r="3" />
  </svg>;
}

export default function ProcessPanel({ machine, lot, telemetry, telemetrySource, history, rca, defects = [] }: Props) {
  const when = (iso?: string) => iso ? new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
  const drift = history?.drift ?? [];
  const focus = drift[0]?.sensor ?? 'temperature_c';
  const trend = (history?.trend ?? []).map(point => Number(point[focus])).filter(Number.isFinite);
  const contributions = (rca?.feature_contributions ?? []).slice(0, 3);
  const scale = Math.max(...contributions.map(c => Math.abs(c.contribution ?? 0)), .01);
  return <section className="panel evidence process-panel" data-module="cause">
    <div className="panel-heading"><h2>Process &amp; probable cause</h2><span>{history ? (history.synthetic ? 'Synthetic lot history' : 'Imported lot history') : 'Simulated preset'}</span></div>
    <p className="process-where">Machine <strong>{machine}</strong> · lot <strong>{lot}</strong>{history?.lot_window?.length === 2 && <> · produced {when(history.lot_window[0])} – {when(history.lot_window[1])}</>}{history?.prior_lots?.length ? <> · compared with {history.prior_lots.join(', ')}</> : null}</p>
    {defects.length > 0 && rca && <p className="defect-link">This part's {[...new Set(defects)].map(d => d.replaceAll('_', ' ')).join(', ')} {defects.length > 1 ? 'were' : 'was'} produced in this batch; the process data below points to <strong>{human(rca.hypothesis)}</strong>{rca.confidence != null ? ` (${(rca.confidence * 100).toFixed(0)}% classifier confidence, uncalibrated)` : ''}.</p>}
    {history ? <div className="drift-block">
      {drift.length ? <ul className="drift-list">{drift.map(d => <li key={d.sensor} className="drift-chip strong">{driftLabel(d)} <span>vs baseline · z {d.z_score}</span></li>)}</ul>
        : <p className="context-note">No sensor is outside its baseline range for this machine.</p>}
      <div className="trend-row"><span>{SENSOR_NAMES[focus] ?? focus}, last {trend.length} lots</span><Trend points={trend} label={SENSOR_NAMES[focus] ?? focus} /></div>
    </div> : <p className="sim-note">{telemetrySource}</p>}
    <dl className="telemetry compact">{Object.entries(telemetry).map(([key, value]) => <div key={key}><dt>{SENSOR_NAMES[key] ?? human(key)}</dt><dd>{typeof value === 'number' ? (Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2)) : value} {UNITS[key] ?? ''}</dd></div>)}</dl>
    {rca ? <div className="cause">
      <h3 className="hypothesis">{human(rca.hypothesis)}</h3>
      <p className="muted">XGBoost hypothesis{rca.confidence != null ? ` · ${(rca.confidence * 100).toFixed(0)}% classifier confidence (uncalibrated)` : ''} · trained on synthetic process data</p>
      {contributions.length > 0 && <div className="drivers" aria-label="Top drivers from TreeSHAP">
        {contributions.map(c => { const value = c.contribution ?? 0; return <div className="driver" key={c.feature}>
          <span>{SENSOR_NAMES[c.feature] ?? human(c.feature)}</span>
          <span className="driver-bar" aria-hidden="true"><i className={value >= 0 ? 'pos' : 'neg'} style={{ width: `${Math.abs(value) / scale * 100}%` }} /></span>
          <strong>{value >= 0 ? '+' : ''}{value.toFixed(2)}</strong>
        </div>; })}
      </div>}
      <p className="footnote">TreeSHAP margins show what pushed this hypothesis; they do not prove causality.</p>
    </div> : <p className="context-note">Root cause is not analysed until the image passes the quality gate.</p>}
  </section>;
}
