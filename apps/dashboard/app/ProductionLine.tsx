'use client';

import { useEffect, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import dynamic from 'next/dynamic';
import { Workflow, Play, Pause, ChevronDown, CircleAlert } from 'lucide-react';

const ProductionScene = dynamic(() => import('./ProductionScene'), { ssr: false, loading: () => <div className="scene-loading" role="status">Preparing production scene…</div> });
const stages = ['Casting', 'Machining', 'Inspection', 'Engineer review', 'Dispatch'];

export default function ProductionLine({ stopped, onCriticalDemo, busy }: { stopped: boolean; onCriticalDemo: () => void; busy: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const [manualPause, setManualPause] = useState(false);
  const reduced = useReducedMotion();
  useEffect(() => { if (stopped) setExpanded(true); }, [stopped]);
  return <section className="panel production-line" aria-label="Manufacturing lifecycle simulation">
    <div className="production-heading"><Workflow size={22} className="production-icon" /><div><h2>Production lifecycle</h2><p>Inspection-driven manufacturing simulation</p></div><span className={`status ${stopped ? 'critical' : manualPause ? 'pending' : 'pass'}`}>{stopped ? 'Quality hold' : manualPause ? 'Paused' : 'Running'}</span>
      <div className="production-actions"><button disabled={busy} onClick={() => { setExpanded(true); onCriticalDemo(); }}><CircleAlert size={16} /> Demonstrate critical defect</button><button className="production-toggle" aria-expanded={expanded} aria-controls="production-visual" onClick={() => setExpanded(value => !value)}>{expanded ? 'Hide simulation' : 'Show live line'}<ChevronDown size={16} style={{ transform: expanded ? 'rotate(180deg)' : undefined }} /></button></div>
    </div>
    <AnimatePresence initial={false}>{expanded && <motion.div id="production-visual" className="production-visual" initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: reduced ? 0 : .28 }}>
      <ProductionScene stopped={stopped} manualPause={manualPause} />
      <div className="production-stages">{stages.map((stage, index) => <span key={stage} className={stopped && index === 2 ? 'stage-held' : ''}>{stage}</span>)}</div>
      <div className="production-foot"><button disabled={stopped} onClick={() => setManualPause(value => !value)}>{manualPause ? <Play size={15} /> : <Pause size={15} />}{manualPause ? 'Play animation' : 'Pause animation'}</button><span>Illustrative process. Critical findings require approval and explicit resume.</span></div>
    </motion.div>}</AnimatePresence>
  </section>;
}
