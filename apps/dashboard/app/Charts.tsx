'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';

// Lay charts out at their real pixel width so text stays a fixed, readable size on phones instead of being scaled down.
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(300, Math.round(entry.contentRect.width))));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

// Categorical slots 1-4 (validated: CVD dE >= 9.1 adjacent, normal >= 22.9 on #fff). Aqua/yellow are < 3:1 on white, so
// every series also carries a direct label and the page offers a table view. Status colours are reserved for severity.
export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];
export const STATUS: Record<string, string> = { pass: '#0ca30c', minor: '#fab219', high: '#ec835a', critical: '#d03b3b', none: '#a5b1b0' };
const GRID = '#e3e9e7';
const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

type Tip = { x: number; y: number; body: ReactNode } | null;
function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  return <div className="chart-tip" style={{ left: tip.x, top: tip.y }} role="status">{tip.body}</div>;
}

/** Horizontal bars: predicted next-lot defect fraction per machine, with the 95% simulated range and the tolerance line. */
export function RiskBars({ rows, tolerance }: { rows: { id: string; value: number; low: number; high: number; note: string }[]; tolerance: number }) {
  const [tip, setTip] = useState<Tip>(null);
  const [ref, W] = useWidth(560);
  const narrow = W < 480;
  const label = 52, right = narrow ? 64 : 150, rowH = 44, top = 18, bar = 20;
  const H = top + rows.length * rowH + 22;
  const max = Math.max(tolerance * 1.15, ...rows.map(r => r.high), .1);
  const x = (v: number) => label + (v / max) * (W - label - right);
  const ticks = [0, .1, .2, .3, .4, .5, .6].filter(t => t <= max);
  return <div className="chart" ref={ref} onMouseLeave={() => setTip(null)}>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Predicted next-lot defect fraction by machine: ${rows.map(r => `${r.id} ${pct(r.value)}`).join(', ')}`}>
      {ticks.map(t => <g key={t}><line x1={x(t)} x2={x(t)} y1={top - 6} y2={H - 22} stroke={GRID} /><text x={x(t)} y={H - 6} className="axis" textAnchor="middle">{Math.round(t * 100)}%</text></g>)}
      <line x1={x(tolerance)} x2={x(tolerance)} y1={top - 10} y2={H - 22} stroke="#b33928" strokeWidth={1.5} />
      <text x={x(tolerance) + 4} y={top - 2} className="axis strong">tolerance {Math.round(tolerance * 100)}%</text>
      {rows.map((r, i) => {
        const y = top + i * rowH + (rowH - bar) / 2;
        const hover = (e: React.MouseEvent) => setTip({ x: e.nativeEvent.offsetX + 14, y: e.nativeEvent.offsetY - 10, body: <><strong>{r.id}</strong><span>Next lot {pct(r.value)}</span><span>95% range {pct(r.low)}–{pct(r.high)}</span><span>{r.note}</span></> });
        return <g key={r.id} onMouseMove={hover} className="hit">
          <rect x={0} y={y - 10} width={W} height={bar + 20} fill="transparent" />
          <text x={label - 10} y={y + bar / 2 + 4} className="axis-label" textAnchor="end">{r.id}</text>
          <path d={`M${x(0)},${y} H${x(r.value) - 4} a4,4 0 0 1 4,4 V${y + bar - 4} a4,4 0 0 1 -4,4 H${x(0)} Z`} fill={i === 0 ? 'var(--accent)' : '#8fb5ae'} />
          <line x1={x(r.low)} x2={x(r.high)} y1={y + bar / 2} y2={y + bar / 2} stroke="#203434" strokeWidth={1.5} />
          <line x1={x(r.low)} x2={x(r.low)} y1={y + 5} y2={y + bar - 5} stroke="#203434" strokeWidth={1.5} />
          <line x1={x(r.high)} x2={x(r.high)} y1={y + 5} y2={y + bar - 5} stroke="#203434" strokeWidth={1.5} />
          <text x={x(r.high) + 8} y={y + bar / 2 + 4} className="value">{pct(r.value)}{!narrow && <tspan className="axis"> {r.note}</tspan>}</text>
        </g>;
      })}
    </svg>
    <Tooltip tip={tip} />
  </div>;
}

/** Multi-line: predicted next-lot fraction across each machine's recent batches. */
export function BatchTrend({ series }: { series: { id: string; points: { label: string; value: number }[] }[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const [ref, W] = useWidth(560);
  const H = 230, left = 40, right = 56, top = 14, bottom = 30;
  const n = Math.max(...series.map(s => s.points.length), 2);
  const values = series.flatMap(s => s.points.map(p => p.value));
  const min = Math.max(0, Math.floor((Math.min(...values) - .02) * 20) / 20), max = Math.ceil((Math.max(...values) + .02) * 20) / 20;
  const x = (i: number) => left + i * (W - left - right) / (n - 1);
  const y = (v: number) => top + (1 - (v - min) / (max - min || 1)) * (H - top - bottom);
  const ticks = Array.from({ length: Math.round((max - min) / .05) + 1 }, (_, i) => min + i * .05);
  return <div className="chart" ref={ref} onMouseLeave={() => setHover(null)}>
    <ul className="chart-legend">{series.map((s, i) => <li key={s.id}><i style={{ background: SERIES[i] }} />{s.id}</li>)}</ul>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Predicted next-lot defect fraction across the last batches of each machine"
      onMouseMove={e => { const box = (e.currentTarget as SVGSVGElement).getBoundingClientRect(); const i = Math.round(((e.clientX - box.left) * W / box.width - left) / ((W - left - right) / (n - 1))); setHover(i >= 0 && i < n ? i : null); }}>
      {ticks.map(t => <g key={t}><line x1={left} x2={W - right} y1={y(t)} y2={y(t)} stroke={GRID} /><text x={left - 8} y={y(t) + 4} className="axis" textAnchor="end">{Math.round(t * 100)}%</text></g>)}
      {series[0]?.points.map((p, i) => <text key={i} x={x(i)} y={H - 8} className="axis" textAnchor="middle">{p.label}</text>)}
      {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={top} y2={H - bottom} stroke="#9fb3b0" />}
      {(() => { const ends = series.map((s, si) => ({ si, y: y(s.points.at(-1)!.value) })).sort((a, b) => a.y - b.y);
        for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;   // keep end labels apart
        return ends.map(e => <text key={`l${e.si}`} x={x(series[e.si].points.length - 1) + 9} y={e.y + 4} className="value">{series[e.si].id}</text>); })()}
      {series.map((s, si) => <g key={s.id}>
        <path d={s.points.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.value)}`).join(' ')} fill="none" stroke={SERIES[si]} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={x(s.points.length - 1)} cy={y(s.points.at(-1)!.value)} r={4.5} fill={SERIES[si]} stroke="#fff" strokeWidth={2} />
        {hover !== null && s.points[hover] && <circle cx={x(hover)} cy={y(s.points[hover].value)} r={4} fill={SERIES[si]} stroke="#fff" strokeWidth={2} />}
      </g>)}
    </svg>
    {hover !== null && <div className="chart-tip fixed-tip" role="status"><strong>Batch {series[0]?.points[hover]?.label}</strong>{series.map((s, i) => s.points[hover] && <span key={s.id}><i style={{ background: SERIES[i] }} />{s.id} {pct(s.points[hover].value)}</span>)}</div>}
  </div>;
}

/** Donut for a part-to-whole with few, distinct categories (severity mix). Centre carries the total. */
export function Donut({ parts, total, label }: { parts: { key: string; label: string; value: number; color: string }[]; total: number; label: string }) {
  const [active, setActive] = useState<string | null>(null);
  const R = 62, r = 40, C = 80, gap = .025;
  let angle = -Math.PI / 2;
  const arcs = parts.filter(p => p.value > 0).map(p => {
    const sweep = (p.value / Math.max(total, 1)) * Math.PI * 2;
    const a0 = angle + gap / 2, a1 = angle + Math.max(sweep - gap / 2, gap / 2);
    angle += sweep;
    const pt = (rad: number, a: number) => `${C + rad * Math.cos(a)},${C + rad * Math.sin(a)}`;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    return { ...p, d: `M${pt(R, a0)} A${R},${R} 0 ${large} 1 ${pt(R, a1)} L${pt(r, a1)} A${r},${r} 0 ${large} 0 ${pt(r, a0)} Z` };
  });
  const focus = parts.find(p => p.key === active);
  return <div className="donut">
    <svg viewBox="0 0 160 160" role="img" aria-label={`${label}: ${parts.map(p => `${p.label} ${p.value}`).join(', ')}`}>
      {arcs.map(a => <path key={a.key} d={a.d} fill={a.color} opacity={active && active !== a.key ? .35 : 1} onMouseEnter={() => setActive(a.key)} onMouseLeave={() => setActive(null)} />)}
      <text x={C} y={C - 2} textAnchor="middle" className="donut-total">{focus ? focus.value : total}</text>
      <text x={C} y={C + 15} textAnchor="middle" className="axis">{focus ? focus.label : label}</text>
    </svg>
    <ul className="chart-legend vertical">{parts.map(p => <li key={p.key} onMouseEnter={() => setActive(p.key)} onMouseLeave={() => setActive(null)}><i style={{ background: p.color }} />{p.label}<strong>{p.value}</strong><span>{total ? pct(p.value / total, 0) : '–'}</span></li>)}</ul>
  </div>;
}

/** Horizontal count bars, one series (no legend; the title names it). */
export function CountBars({ rows, unit }: { rows: { label: string; value: number }[]; unit: string }) {
  const [tip, setTip] = useState<Tip>(null);
  const [ref, W] = useWidth(420);
  const label = Math.min(150, Math.round(W * .38)), rowH = 30, bar = 16;
  const H = rows.length * rowH + 6;
  const max = Math.max(...rows.map(r => r.value), 1);
  return <div className="chart" ref={ref} onMouseLeave={() => setTip(null)}>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={rows.map(r => `${r.label} ${r.value} ${unit}`).join(', ')}>
      {rows.map((r, i) => { const y = i * rowH + (rowH - bar) / 2; const w = Math.max(4, (r.value / max) * (W - label - 50));
        return <g key={r.label} className="hit" onMouseMove={e => setTip({ x: e.nativeEvent.offsetX + 14, y: e.nativeEvent.offsetY - 10, body: <><strong>{r.label}</strong><span>{r.value} {unit}</span></> })}>
          <rect x={0} y={y - 6} width={W} height={bar + 12} fill="transparent" />
          <text x={label - 10} y={y + bar / 2 + 4} className="axis-label" textAnchor="end">{r.label}</text>
          <path d={`M${label},${y} H${label + w - 4} a4,4 0 0 1 4,4 V${y + bar - 4} a4,4 0 0 1 -4,4 H${label} Z`} fill="var(--accent)" />
          <text x={label + w + 8} y={y + bar / 2 + 4} className="value">{r.value}</text>
        </g>; })}
    </svg>
    <Tooltip tip={tip} />
  </div>;
}
