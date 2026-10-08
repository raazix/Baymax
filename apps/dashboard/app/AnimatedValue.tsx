'use client';

import { useEffect } from 'react';
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from 'motion/react';

export default function AnimatedValue({ value }: { value: number }) {
  const reduced = useReducedMotion();
  const number = useMotionValue(value);
  const display = useTransform(number, n => `${(n * 100).toFixed(1)}%`);
  useEffect(() => {
    const control = animate(number, value, { duration: reduced ? 0 : .45, ease: [.16, 1, .3, 1] });
    return () => control.stop();
  }, [number, value, reduced]);
  return <><span className="sr-only">{(value * 100).toFixed(1)}%</span><motion.span aria-hidden="true">{display}</motion.span></>;
}
