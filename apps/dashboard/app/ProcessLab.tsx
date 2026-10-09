'use client';

import { useEffect, useRef, useState } from 'react';

type Readings = { temperature_c: number; pressure_bar: number; vibration_mm_s: number; machine_speed_rpm: number };
type Result = {
  rca: { hypothesis: string; class_probabilities: Record<string, number>; method: string; base_value: number; prediction_margin: number; out_of_training_range: boolean | string[]; feature_contributions: { feature: string; contribution: number }[] };
  forecast: { predicted_defect_fraction: number; method: string; held_out_mae: number; outside_training_range: boolean; history_assumption: string };
  uncertainty: { interval_95: number[]; breach_probability: number; tolerance: number; simulations: number; calibration_residual_count: number; mean: number };
  histogram: { edges: number[]; counts: number[]; simulations: number };
  training_feature_ranges: Record<string, number[]>;
};

const PRESETS: { name: string; values: Readings }[] = [
  { name: 'Nominal', values: { temperature_c: 705, pressure_bar: 100, vibration_mm_s: 2.0, machine_speed_rpm: 1200 } },
  { name: 'Thermal drift', values: { temperature_c: 753, pressure_bar: 103, vibration_mm_s: 2.3, machine_speed_rpm: 1200 } },
  { name: 'Pressure instability', values: { temperature_c: 709, pressure_bar: 78, vibration_mm_s: 2.2, machine_speed_rpm: 1200 } },
  { name: 'Tooling vibration', values: { temperature_c: 705, pressure_bar: 102, vibration_mm_s: 5.4, machine_speed_rpm: 1230 } },
  { name: 'Speed drift', values: { temperature_c: 713, pressure_bar: 100, vibration_mm_s: 2.4, machine_speed_rpm: 1480 } },
];
const FIELDS: { key: keyof Readings; label: string; unit: string; step: number }[] = [
  { key: 'temperature_c', label: 'Temperature', unit: '°C', step: 1 },
  { key: 'pressure_bar', label: 'Pressure', unit: 'bar', step: 1 },
  { key: 'vibration_mm_s', label: 'Vibration', unit: 'mm/s', step: 0.1 },
  { key: 'machine_speed_rpm', label: 'Spindle speed', unit: 'rpm', step: 5 },
];
const human = (value: string) => value.replaceAll('_', ' ');
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function ProcessLab() {
  const [readings, setReadings] = useState<Readings>(PRESETS[0].values);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState('');
  const latest = useRef(0);

  useEffect(() => {
    const ticket = ++latest.current;
    const timer = setTimeout(async () => {
      try {
        const response = await fetch('/api/analytics/what-if', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(readings), cache: 'no-store' });
        const payload = await response.json();
        if (ticket !== latest.current) return;
        if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}).`);
        setResult(payload); setError('');
      } catch (e) { if (ticket === latest.current) setError(e instanceof Error ? e.message : 'Analytics unavailable.'); }
    }, 150);
    return () => clearTimeout(timer);
  }, [readings]);

  const ranges = result?.training_feature_ranges;
  const maxCount = result ? Math.max(...result.histogram.counts, 1) : 1;
  const maxContribution = result ? Math.max(...result.rca.feature_contributions.map(c => Math.abs(c.contribution)), 0.01) : 1;
  const probabilities = result ? Object.entries(result.rca.class_probabilities).sort((a, b) => b[1] - a[1]) : [];

  return <section className="lab">
    <div className="panel lab-inputs">
      <h2>Process what-if lab</h2>
      <div className="buttons" role="group" aria-label="Scenario presets">{PRESETS.map(p => <button key={p.name} onClick={() => setReadings(p.values)}>{p.name}</button>)}</div>
      {FIELDS.map(f => {
        const [low, high] = ranges?.[f.key] ?? [readings[f.key] - 50, readings[f.key] + 50];
        return <div className="slider" key={f.key}>
          <label htmlFor={f.key}><span>{f.label}</span><strong>{readings[f.key]} {f.unit}</strong></label>
          <input id={f.key} type="range" min={Math.floor(low)} max={Math.ceil(high)} step={f.step} value={readings[f.key]} onChange={e => setReadings({ ...readings, [f.key]: Number(e.target.value) })} />
          <span className="muted">training range {low.toFixed(f.step < 1 ? 1 : 0)}–{high.toFixed(f.step < 1 ? 1 : 0)}</span>
        </div>;
      })}
    </div>
    {error && <div className="error" role="alert">{error}</div>}
    {result && <div className="lab-outputs">
      <div className="panel">
        <h2>1 · Probable cause <span className="tag">XGBoost</span></h2>
        <h3 className="hypothesis">{human(result.rca.hypothesis)}</h3>
        {probabilities.map(([name, value]) => <div className="contribution" key={name}><div><span>{human(name)}</span><strong>{percent(value)}</strong></div><div className="bar"><i style={{ width: `${value * 100}%` }} /></div></div>)}
      </div>
      <div className="panel">
        <h2>2 · Why <span className="tag">exact TreeSHAP</span></h2>
        {result.rca.feature_contributions.map(item => <div className="contribution" key={item.feature}><div><span>{human(item.feature)}</span><strong>{item.contribution >= 0 ? '+' : ''}{item.contribution.toFixed(2)}</strong></div><div className="bar diverging"><i className={item.contribution >= 0 ? 'pos' : 'neg'} style={{ width: `${(Math.abs(item.contribution) / maxContribution) * 50}%`, [item.contribution >= 0 ? 'left' : 'right']: '50%' }} /></div></div>)}
      </div>
      <div className="panel wide">
        <h2>3 · Next-lot forecast and risk <span className="tag">PLSR + Monte Carlo</span></h2>
        <div className="risk"><strong>{percent(result.forecast.predicted_defect_fraction)}</strong><span>predicted defect fraction (held-out MAE {percent(result.forecast.held_out_mae)})</span></div>
        <svg className="histogram" viewBox="0 0 400 140" role="img" aria-label={`Distribution of ${result.uncertainty.simulations} simulated outcomes; ${percent(result.uncertainty.breach_probability)} exceed the ${percent(result.uncertainty.tolerance)} tolerance`}>
          {result.histogram.counts.map((count, i) => {
            const x0 = result.histogram.edges[i] * 400, w = (result.histogram.edges[i + 1] - result.histogram.edges[i]) * 400;
            const h = (count / maxCount) * 100, mid = (result.histogram.edges[i] + result.histogram.edges[i + 1]) / 2;
            return <rect key={i} x={x0 + 1} y={110 - h} width={Math.max(w - 2, 1)} height={h} className={mid > result.uncertainty.tolerance ? 'over' : 'under'} />;
          })}
          <line x1={result.uncertainty.tolerance * 400} x2={result.uncertainty.tolerance * 400} y1="4" y2="112" className="tol" />
          <text x={result.uncertainty.tolerance * 400 + 4} y="14" className="svg-label">tolerance {percent(result.uncertainty.tolerance)}</text>
          <line x1={result.uncertainty.interval_95[0] * 400} x2={result.uncertainty.interval_95[1] * 400} y1="122" y2="122" className="ci" />
          <text x={result.uncertainty.interval_95[0] * 400} y="137" className="svg-label">95% range {percent(result.uncertainty.interval_95[0])} – {percent(result.uncertainty.interval_95[1])}</text>
        </svg>
        <dl className="risk-details"><div><dt>Chance a simulated lot exceeds the tolerance</dt><dd>{percent(result.uncertainty.breach_probability)}</dd></div><div><dt>Simulations</dt><dd>{result.uncertainty.simulations.toLocaleString()} (seeded, reproducible)</dd></div><div><dt>Calibration residuals resampled</dt><dd>{result.uncertainty.calibration_residual_count}</dd></div></dl>
      </div>
    </div>}
  </section>;
}
