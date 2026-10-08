'use client';

import { MotionConfig } from 'motion/react';
import { Toaster } from 'sonner';

export default function WorkspaceProviders({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user" transition={{ duration: .22, ease: [.16, 1, .3, 1] }}>
    {children}
    <Toaster richColors closeButton position="top-right" visibleToasts={2} toastOptions={{ className: 'workspace-toast', duration: 4500 }} />
  </MotionConfig>;
}
