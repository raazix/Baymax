'use client';

import { useEffect, useState } from 'react';

type Status = Record<string, { configured?: boolean; loaded?: boolean; mode?: string } | undefined>;
type Row = { label: string; role: string; state: 'ready' | 'standby' | 'missing' };

// Live readiness of every model the pipeline uses. "Standby" means configured but loaded lazily on first use.
export default function ModelStatus() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/api/models', { cache: 'no-store' });
        if (!response.ok) throw new Error();
        const s: Status = await response.json();
        const state = (key: string, lazy = true): Row['state'] => !s[key]?.configured ? 'missing' : lazy && !s[key]?.loaded ? 'standby' : 'ready';
        if (!cancelled) {
          setOffline(false);
          setRows([
            { label: 'PatchCore', role: 'casting anomaly', state: state('patchcore') },
            { label: 'YOLO11n', role: 'steel defects', state: state('proxy_detection') },
            { label: 'XGBoost', role: 'root cause + SHAP', state: state('xgboost_rca', false) },
            { label: 'PLSR', role: 'next-lot forecast', state: state('plsr', false) },
            { label: 'Monte Carlo', role: '10k risk draws', state: state('plsr', false) },
          ]);
        }
      } catch { if (!cancelled) setOffline(true); }
    };
    void load();
    const timer = setInterval(load, 15000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  return <div className="model-status" aria-label="Model status">
    <p className="model-status-title">{offline ? 'Backend offline' : 'Models'}</p>
    {offline ? <p className="model-status-off">Start FastAPI on port 8000, then this list refreshes.</p> :
      <ul>{(rows ?? []).map(row => <li key={row.label} title={row.state === 'standby' ? 'Configured; loads on first use' : row.state}>
        <i className={`dot ${row.state}`} aria-hidden="true" /><span><strong>{row.label}</strong>{row.role}</span><em className="sr-only">{row.state}</em>
      </li>)}</ul>}
  </div>;
}
