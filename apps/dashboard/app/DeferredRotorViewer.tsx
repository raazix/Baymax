'use client';

import { useState, type ComponentProps } from 'react';
import dynamic from 'next/dynamic';
import type RotorViewerType from './RotorViewer';

const RotorViewer = dynamic(() => import('./RotorViewer'), { ssr: false, loading: () => <p className="muted" role="status">Preparing 3D view…</p> });
export default function DeferredRotorViewer(props: ComponentProps<typeof RotorViewerType>) {
  const [open, setOpen] = useState(false);
  return <details className="evidence-disclosure visual-disclosure" onToggle={event => setOpen(event.currentTarget.open)}><summary>Open 3D inspection view</summary>{open && <RotorViewer {...props} />}</details>;
}
