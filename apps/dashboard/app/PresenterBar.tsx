'use client';

import { motion, useReducedMotion } from 'motion/react';
import { Check, X } from 'lucide-react';

export type PresenterStep = { title: string; say: string; run: () => void; done: boolean; disabled?: boolean };

export default function PresenterBar({ steps, busy, status, onClose }: { steps: PresenterStep[]; busy: boolean; status: string; onClose: () => void }) {
  const reduce = useReducedMotion();
  const next = steps.findIndex(step => !step.done);
  const current = steps[next === -1 ? steps.length - 1 : next];
  return <motion.section className="presenter" aria-label="Presenter steps" initial={reduce ? false : { opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .25, ease: [0.16, 1, 0.3, 1] }}>
    <div className="presenter-head">
      <strong>Presenter mode</strong>
      <span aria-live="polite">{next === -1 ? 'All steps done. Restart any step, or close presenter mode.' : `Next, step ${next + 1}: ${current.say}`}</span>
      <button className="icon-button" onClick={onClose} aria-label="Close presenter mode"><X size={16} aria-hidden="true" /></button>
    </div>
    <ol className="presenter-steps">
      {steps.map((step, i) => <li key={step.title}>
        <button className={`${step.done ? 'done' : ''}${i === next ? ' next' : ''}`} onClick={step.run} disabled={busy || step.disabled} title={step.say}>
          {i === next && <motion.span layoutId="presenter-next" className="next-glow" transition={{ duration: reduce ? 0 : .35, ease: [0.16, 1, 0.3, 1] }} aria-hidden="true" />}
          <span className="step-num">{step.done
            ? <motion.span initial={reduce ? false : { scale: .4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: .25, ease: [0.16, 1, 0.3, 1] }} style={{ display: 'grid' }}><Check size={13} strokeWidth={3} aria-hidden="true" /></motion.span>
            : i + 1}</span>
          <span className="step-title">{step.title}</span>
        </button>
      </li>)}
    </ol>
    {status && <p className="presenter-status" role="status">{status}</p>}
  </motion.section>;
}
