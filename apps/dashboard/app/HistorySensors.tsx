'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, Database, FileUp, RefreshCw, TrendingDown, TrendingUp } from 'lucide-react';

type Catalog = { machines: string[]; sensors: string[]; reading_count: number; dataset_count: number; source: string };
type Series = { machine_id: string; sensor: string; days: number; current_window: { start: string; end: string; points: { observed_at: string; value: number; unit: string; lot_id: string | null }[]; count: number; mean: number | null; min: number | null; max: number | null; stddev: number | null; unit: string | null }; previous_window: { mean: number | null; count: number; unit: string | null }; mean_change_fraction: number | null; source: string };
async function checked(response: Response) { if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Request failed (${response.status}).`); } return response; }
const number = (value: number | null, unit = '') => value === null ? '—' : `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${unit ? ` ${unit}` : ''}`;

export default function HistorySensors() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [machine, setMachine] = useState('');
  const [sensor, setSensor] = useState('');
  const [days, setDays] = useState(30);
  const [series, setSeries] = useState<Series | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [sourceLabel, setSourceLabel] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const loadCatalog = useCallback(async () => {
    const response = await checked(await fetch('/api/history/sensors/catalog', { cache: 'no-store' }));
    const result: Catalog = await response.json();
    setCatalog(result);
    setMachine(current => result.machines.includes(current) ? current : result.machines[0] ?? '');
    setSensor(current => result.sensors.includes(current) ? current : result.sensors[0] ?? '');
  }, []);
  useEffect(() => { void loadCatalog().catch(e => setError(e.message)); }, [loadCatalog]);
  useEffect(() => {
    if (!machine || !sensor) { setSeries(null); return; }
    const controller = new AbortController();
    const params = new URLSearchParams({ machine_id: machine, sensor, days: String(days) });
    fetch(`/api/history/sensors/series?${params}`, { cache: 'no-store', signal: controller.signal }).then(checked).then(r => r.json()).then(setSeries).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, [machine, sensor, days, catalog?.reading_count]);

  async function upload() {
    if (!file) return;
    setBusy(true); setError(''); setMessage('');
    try {
      const params = new URLSearchParams({ filename: file.name, source_label: sourceLabel || 'source not verified' });
      const response = await checked(await fetch(`/api/history/sensor-datasets?${params}`, { method: 'POST', headers: { 'Content-Type': 'text/csv' }, body: file, cache: 'no-store' }));
      const result = await response.json();
      setMessage(`${result.rows_inserted.toLocaleString()} new readings saved; ${result.rows_duplicate.toLocaleString()} duplicates skipped.`);
      setFile(null); await loadCatalog();
    } catch (e) { setError(e instanceof Error ? e.message : 'CSV import failed.'); }
    finally { setBusy(false); }
  }

  const points = series?.current_window.points ?? [];
  const chart = useMemo(() => {
    if (!points.length) return null;
    const values = points.map(p => p.value), min = Math.min(...values), max = Math.max(...values);
    const pad = (max - min || Math.abs(max) * .05 || 1) * .12;
    const low = min - pad, high = max + pad;
    const xy = values.map((value, i) => `${(i * 800 / Math.max(1, values.length - 1)).toFixed(1)},${(220 - (value - low) * 200 / (high - low)).toFixed(1)}`).join(' ');
    return { xy, min: low, max: high };
  }, [points]);
  const change = series?.mean_change_fraction;

  return <div className="history-workspace">
    <section className="panel history-import"><div className="panel-heading"><h2><Database size={18} aria-hidden="true" /> Historical sensor data</h2><span>Neon PostgreSQL</span></div>
      <p>Import timestamped machine readings, then compare each sensor with the previous period. CSV labels describe the source but are not independently verified.</p>
      <div className="history-upload-row"><label className="file-button"><FileUp size={16} aria-hidden="true" /> {file ? file.name : 'Choose sensor CSV'}<input type="file" accept=".csv,text/csv" disabled={busy} onChange={e => setFile(e.target.files?.[0] ?? null)} /></label>
        <label className="history-source">Source label<input value={sourceLabel} maxLength={180} placeholder="e.g. historian export" onChange={e => setSourceLabel(e.target.value)} disabled={busy} /></label>
        <button className="primary" onClick={() => void upload()} disabled={busy || !file}>{busy ? 'Importing...' : 'Import history'}</button></div>
      <p className="muted">UTF-8 CSV: timestamp (ISO-8601 with timezone), machine_id, sensor, value, unit; optional lot_id. Up to 5 MiB and 50,000 rows per file. Duplicate observations are skipped.</p>
      {message && <p className="history-success" role="status">{message}</p>}{error && <p className="inference-error" role="alert">{error}</p>}
      <div className="history-counts"><span>{catalog?.reading_count.toLocaleString() ?? '—'} readings</span><span>{catalog?.machines.length ?? 0} machines</span><span>{catalog?.dataset_count ?? 0} imported files</span></div>
    </section>
    <section className="panel sensor-trends"><div className="panel-heading"><h2><Activity size={18} aria-hidden="true" /> Sensor trends</h2><button aria-label="Refresh sensor history" onClick={() => { void loadCatalog().catch(e => setError(e.message)); }}><RefreshCw size={16} /></button></div>
      {catalog?.machines.length ? <div className="trend-controls"><label>Machine<select value={machine} onChange={e => setMachine(e.target.value)}>{catalog.machines.map(item => <option key={item}>{item}</option>)}</select></label><label>Sensor<select value={sensor} onChange={e => setSensor(e.target.value)}>{catalog.sensors.map(item => <option key={item}>{item}</option>)}</select></label><label>Window<select value={days} onChange={e => setDays(Number(e.target.value))}><option value={7}>7 days</option><option value={30}>30 days</option><option value={90}>90 days</option></select></label></div> : <div className="history-empty"><h3>Import your historian export to start</h3><p>CSV template: <code>timestamp,machine_id,sensor,value,unit,lot_id</code></p><p>Include timezone-aware ISO dates. Repeated uploads are safely deduplicated.</p></div>}
      {series && <><div className="trend-summary"><div><span>Current mean</span><strong>{number(series.current_window.mean, series.current_window.unit ?? '')}</strong></div><div><span>Previous {days}-day mean</span><strong>{number(series.previous_window.mean, series.current_window.unit ?? '')}</strong></div><div><span>Mean change</span><strong className={change == null ? '' : change > 0 ? 'trend-up' : change < 0 ? 'trend-down' : ''}>{change == null ? 'Not enough history' : <>{change > 0 ? <TrendingUp size={16} /> : change < 0 ? <TrendingDown size={16} /> : null}{(change * 100).toFixed(1)}%</>}</strong></div><div><span>Readings in window</span><strong>{series.current_window.count.toLocaleString()}</strong></div></div>
        {chart ? <div className="sensor-chart-wrap"><div className="sensor-chart-labels"><span>{number(chart.max, series.current_window.unit ?? '')}</span><span>{number(chart.min, series.current_window.unit ?? '')}</span></div><svg className="sensor-chart" viewBox="0 0 800 240" role="img" aria-label={`${sensor} readings for ${machine} over the last ${days} days`} preserveAspectRatio="none"><line x1="0" y1="10" x2="800" y2="10"/><line x1="0" y1="220" x2="800" y2="220"/><polyline points={chart.xy}/></svg><div className="sensor-chart-dates"><span>{new Date(series.current_window.start).toLocaleDateString()}</span><span>Now</span></div></div> : <p className="muted">No readings in the selected recent window. Try a wider window or import newer history.</p>}
        <p className="muted">{series.source} · comparison is descriptive historical change, not a fault diagnosis or forecast.</p></>}
    </section>
  </div>;
}
