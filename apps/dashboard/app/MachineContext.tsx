'use client';

import { useEffect, useState } from 'react';
import { Factory, TrendingUp } from 'lucide-react';

export type Drift = { sensor: string; current: number; baseline_mean: number; change: number; z_score: number; unit?: string | null };
export type LotContext = {
  machine_id: string; lot_id: string; lot_window: string[]; readings: number; telemetry: Record<string, number>;
  units: Record<string, string | null>; prior_lots: string[]; drift: Drift[]; synthetic: boolean; source_labels: string[];
  trend: ({ lot_id: string; end: string } & Record<string, number | string>)[];
};

export const SENSOR_NAMES: Record<string, string> = { temperature_c: 'Temperature', pressure_bar: 'Pressure', vibration_mm_s: 'Vibration', machine_speed_rpm: 'Spindle speed' };

export function driftLabel(d: Drift) {
  return `${SENSOR_NAMES[d.sensor] ?? d.sensor} ${d.change > 0 ? '+' : ''}${Math.abs(d.change) >= 10 ? d.change.toFixed(0) : d.change.toFixed(1)} ${d.unit ?? ''}`.trim();
}

// Machine selector for the inspect card: the selected machine's current lot from imported history becomes the
// process context of the next inspection (RCA input and forecast history).
export default function MachineContext({ machine, onChange, disabled }: { machine: string; onChange: (machine: string) => void; disabled?: boolean }) {
  const [machines, setMachines] = useState<string[]>([]);
  const [context, setContext] = useState<LotContext | null>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'none' | 'offline'>('loading');

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/history/sensors/catalog', { signal: controller.signal, cache: 'no-store' })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(catalog => {
        const list: string[] = catalog.machines ?? [];
        setMachines(list);
        if (list.length && !list.includes(machine)) onChange(list.includes('M-02') ? 'M-02' : list[0]);
        if (!list.length) setState('none');
      }).catch(() => { if (!controller.signal.aborted) setState('offline'); });
    return () => controller.abort();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!machine || !machines.includes(machine)) return;
    const controller = new AbortController();
    setState('loading');
    fetch(`/api/history/lot-context?machine_id=${encodeURIComponent(machine)}`, { signal: controller.signal, cache: 'no-store' })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(result => { setContext(result); setState('ready'); })
      .catch(() => { if (!controller.signal.aborted) { setContext(null); setState('none'); } });
    return () => controller.abort();
  }, [machine, machines]);

  if (state === 'offline' || (state === 'none' && !machines.length)) {
    return <div className="machine-context muted-context"><Factory size={16} aria-hidden="true" /><span>No sensor history imported. Inspections use a labelled simulated process preset.</span></div>;
  }
  return <div className="machine-context">
    <label className="machine-select"><Factory size={16} aria-hidden="true" /><span className="sr-only">Machine</span>
      <select value={machine} onChange={e => onChange(e.target.value)} disabled={disabled} aria-label="Machine for the next inspection">
        {machines.map(m => <option key={m} value={m}>{m}</option>)}
      </select>
    </label>
    {state === 'loading' && <span className="context-note">Reading lot history…</span>}
    {state === 'ready' && context && <>
      <span className="context-lot">Lot <strong>{context.lot_id}</strong></span>
      {context.drift.length ? context.drift.slice(0, 2).map(d => <span key={d.sensor} className="drift-chip" title={`z = ${d.z_score} versus ${context.machine_id} baseline lots`}><TrendingUp size={13} aria-hidden="true" />{driftLabel(d)}</span>)
        : <span className="context-note">No sensor drift versus baseline</span>}
      {context.synthetic && <span className="synthetic-tag" title={context.source_labels.join('; ')}>Synthetic history</span>}
    </>}
  </div>;
}
