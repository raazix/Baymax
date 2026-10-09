'use client';

import { useEffect, useMemo, useState } from 'react';
import { LoaderCircle, RefreshCw } from 'lucide-react';
import { BatchTrend, CountBars, Donut, RiskBars, STATUS } from './Charts';
import { SENSOR_NAMES, driftLabel, type Drift } from './MachineContext';

type Machine = {
  rank: number; machine_id: string; current_lot: string; produced: string[]; process: Record<string, number>; units: Record<string, string | null>;
  drift: Drift[]; predicted_defect_fraction: number; interval_95: number[]; probability_over_tolerance: number; tolerance: number;
  root_cause: string; root_cause_confidence: number | null; top_drivers: { feature: string; shap_margin: number }[]; recommended_action: string;
  batches: { lot_id: string; predicted_defect_fraction_next: number; root_cause: string }[];
  observed_inspections: { inspected: number; with_findings: number; worst_severity: string | null }; synthetic_history: boolean;
};
type Ranking = { ranking: Machine[]; method: string; limitations: string; generated_at?: string; cached_seconds_ago?: number };
type InspectionRecord = { quality: { passed: boolean }; defects: { label: string; severity: { level: string } }[] };

const human = (value: string) => value.replaceAll('_', ' ');
const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
const LABELS: { [k: string]: string } = { anomaly_unclassified: 'Unclassified anomaly', severe_corrosion: 'Severe corrosion', 'rolled-in_scale': 'Rolled-in scale', pitted_surface: 'Pitted surface', surface_crack: 'Surface crack' };
const label = (k: string) => LABELS[k] ?? (k.charAt(0).toUpperCase() + human(k).slice(1));
const when = (iso?: string) => iso ? new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';

export default function LineRisk({ records, onInspect }: { records: InspectionRecord[]; onInspect: (machine: string) => void }) {
  const [data, setData] = useState<Ranking | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  async function load(refresh = false) {
    setLoading(true); setError('');
    try {
      const response = await fetch(`/api/history/risk-ranking${refresh ? '?refresh=true' : ''}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Risk ranking failed (${response.status}). Check that sensor history is imported and the API is running.`);
      setData(await response.json());
    } catch (e) { setError(e instanceof Error ? e.message : 'Risk ranking failed.'); }
    finally { setLoading(false); }
  }
  useEffect(() => { void load(); }, []);

  // Severity mix: the worst finding per recent inspection; status colours, always with labels.
  const severityMix = useMemo(() => {
    const counts = { pass: 0, minor: 0, high: 0, critical: 0, none: 0 };
    for (const r of records) {
      if (!r.quality.passed) { counts.none++; continue; }
      const levels = r.defects.map(d => d.severity.level);
      if (levels.includes('critical')) counts.critical++;
      else if (levels.some(l => l === 'high' || l === 'review_required')) counts.high++;
      else if (levels.length) counts.minor++;
      else counts.pass++;
    }
    return counts;
  }, [records]);
  const classes = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of records) for (const d of r.defects) counts.set(label(d.label), (counts.get(label(d.label)) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 7).map(([k, v]) => ({ label: k, value: v }));
  }, [records]);

  const top = data?.ranking[0];
  return <section className="line-risk">
    <div className="panel risk-summary">
      <div>
        <h2>Which machine is likely to produce defects next</h2>
        {loading && !data ? <p className="context-note"><LoaderCircle size={14} className="is-spinning" /> Scoring every machine and batch from production history (about 30 s the first time)…</p>
          : error ? <p className="inference-error" role="alert">{error}</p>
          : top && <p className="risk-lead"><strong>{top.machine_id}</strong> has the highest predicted defect rate for its next batch: <strong>{pct(top.predicted_defect_fraction)}</strong> (simulated 95% range {pct(top.interval_95[0])}–{pct(top.interval_95[1])}). Probable cause: <strong>{human(top.root_cause)}</strong>{top.root_cause_confidence !== null && <> at {pct(top.root_cause_confidence)} classifier confidence (uncalibrated)</>}{top.drift[0] && <>, with {driftLabel(top.drift[0])} versus its baseline</>}. Recommended: {top.recommended_action}</p>}
      </div>
      <button type="button" className="icon-text" onClick={() => void load(true)} disabled={loading}><RefreshCw size={15} aria-hidden="true" /> Re-score</button>
    </div>

    {data && <>
      <div className="risk-grid">
        <section className="panel"><div className="panel-heading"><h2>Next-batch defect risk by machine</h2><span>PLSR + Monte Carlo</span></div>
          <RiskBars tolerance={data.ranking[0]?.tolerance ?? .5} rows={data.ranking.map(m => ({ id: m.machine_id, value: m.predicted_defect_fraction, low: m.interval_95[0], high: m.interval_95[1], note: human(m.root_cause) }))} />
          <p className="footnote">Bars: predicted defect fraction of each machine's next lot. Whiskers: simulated 95% range. Red line: demo tolerance.</p>
        </section>
        <section className="panel"><div className="panel-heading"><h2>Severity of recent inspections</h2><span>{records.length} parts</span></div>
          <Donut label="inspections" total={records.length} parts={[
            { key: 'critical', label: 'Critical', value: severityMix.critical, color: STATUS.critical },
            { key: 'high', label: 'High', value: severityMix.high, color: STATUS.high },
            { key: 'minor', label: 'Low / medium', value: severityMix.minor, color: STATUS.minor },
            { key: 'pass', label: 'No finding', value: severityMix.pass, color: STATUS.pass },
            { key: 'none', label: 'Recapture', value: severityMix.none, color: STATUS.none }]} />
          <p className="footnote">Worst finding per inspection, from severity rules v3.</p>
        </section>
      </div>
      <div className="risk-grid">
        <section className="panel"><div className="panel-heading"><h2>Risk trend over recent batches</h2><span>each point scores one batch</span></div>
          <BatchTrend series={data.ranking.map(m => ({ id: m.machine_id, points: m.batches.map(b => ({ label: b.lot_id.split('-L').pop() ?? b.lot_id, value: b.predicted_defect_fraction_next })) }))} />
        </section>
        <section className="panel"><div className="panel-heading"><h2>Defect classes found</h2><span>recent inspections</span></div>
          {classes.length ? <CountBars rows={classes} unit="findings" /> : <p className="context-note">No findings recorded yet.</p>}
        </section>
      </div>
      <section className="panel machine-table"><div className="panel-heading"><h2>Machines, batches and probable causes</h2><span>{data.ranking.some(m => m.synthetic_history) ? 'synthetic production history' : 'imported production history'}</span></div>
        <div className="table-scroll"><table>
          <thead><tr><th>#</th><th>Machine / batch</th><th>Process (batch mean)</th><th>Probable cause</th><th>Next batch</th><th>Inspected parts</th><th>Recommended action</th><th></th></tr></thead>
          <tbody>{data.ranking.map(m => <tr key={m.machine_id}>
            <td data-label="Rank">{m.rank}</td>
            <td data-label="Machine / batch"><strong>{m.machine_id}</strong><span className="cell-sub">{m.current_lot}</span><span className="cell-sub">{when(m.produced[0])} – {when(m.produced[1])}</span></td>
            <td data-label="Process (batch mean)"><ul className="process-cells">{Object.entries(m.process).map(([k, v]) => { const d = m.drift.find(x => x.sensor === k); return <li key={k} className={d ? 'drifting' : ''}>{SENSOR_NAMES[k] ?? k} <strong>{Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)} {m.units[k] ?? ''}</strong>{d && <em title={`${d.change > 0 ? '+' : ''}${d.change} versus the machine's baseline lots (z ${d.z_score})`}>{d.change > 0 ? '+' : ''}{Math.abs(d.change) >= 10 ? d.change.toFixed(0) : d.change.toFixed(1)}</em>}</li>; })}</ul></td>
            <td data-label="Probable cause"><strong className="cause">{human(m.root_cause)}</strong><span className="cell-sub">{m.root_cause_confidence !== null ? `${pct(m.root_cause_confidence)} classifier confidence` : ''}</span><span className="cell-sub">drivers: {m.top_drivers.map(d => SENSOR_NAMES[d.feature] ?? d.feature).join(', ')}</span></td>
            <td data-label="Next batch"><strong>{pct(m.predicted_defect_fraction)}</strong><span className="cell-sub">{pct(m.interval_95[0])}–{pct(m.interval_95[1])}</span><span className="cell-sub">{pct(m.probability_over_tolerance)} over tolerance</span></td>
            <td data-label="Inspected parts"><strong>{m.observed_inspections.with_findings}/{m.observed_inspections.inspected}</strong><span className="cell-sub">with findings{m.observed_inspections.worst_severity ? `, worst ${human(m.observed_inspections.worst_severity)}` : ''}</span></td>
            <td data-label="Recommended action" className="action-cell">{m.recommended_action}</td>
            <td><button type="button" onClick={() => onInspect(m.machine_id)}>Inspect a part</button></td>
          </tr>)}</tbody>
        </table></div>
        <p className="footnote">{data.method} {data.limitations}</p>
      </section>
    </>}
  </section>;
}
