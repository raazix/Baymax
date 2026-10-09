'use client';

import { useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { motion } from 'motion/react';
import { Activity, Gauge, Camera, ScanSearch, FileCheck2, Ellipsis, BrainCircuit, SlidersHorizontal, Smartphone, X, Presentation } from 'lucide-react';

type Tab = 'inspection' | 'risk' | 'audit' | 'camera' | 'train' | 'lab' | 'history';
export default function MobileNavigation({ tab, onSelect, onScan, onPhone, onPresenter, busy }: {
  tab: Tab; onSelect: (tab: Tab) => void; onScan: () => void; onPhone: () => void; onPresenter: () => void; busy: boolean;
}) {
  const [open, setOpen] = useState(false);
  const items = [{ id: 'inspection' as const, label: 'Inspect', Icon: ScanSearch }, { id: 'risk' as const, label: 'Risk', Icon: Gauge }, { id: 'audit' as const, label: 'Evidence', Icon: FileCheck2 }];
  return <>
    <nav className="mobile-navigation" aria-label="Phone workspace">
      {items.slice(0, 1).map(({ id, label, Icon }) => <motion.button whileTap={{ scale: .94 }} key={id} onClick={() => onSelect(id)} aria-current={tab === id ? 'page' : undefined} className={tab === id ? 'active' : ''}>{tab === id && <motion.span layoutId="phone-navigation" className="mobile-nav-indicator" aria-hidden="true" />}<Icon size={21} /><span>{label}</span></motion.button>)}
      <motion.button whileTap={{ scale: .94 }} className="mobile-scan" onClick={onScan} disabled={busy} aria-label="Scan part"><Camera size={22} /><span>Scan</span></motion.button>
      {items.slice(1).map(({ id, label, Icon }) => <motion.button whileTap={{ scale: .94 }} key={id} onClick={() => onSelect(id)} aria-current={tab === id ? 'page' : undefined} className={tab === id ? 'active' : ''}>{tab === id && <motion.span layoutId="phone-navigation" className="mobile-nav-indicator" aria-hidden="true" />}<Icon size={21} /><span>{label}</span></motion.button>)}
      <Dialog.Root open={open} onOpenChange={setOpen}>
        <Dialog.Trigger asChild><button aria-label="More tools"><Ellipsis size={22} /><span>More</span></button></Dialog.Trigger>
        <Dialog.Portal><Dialog.Overlay className="workspace-overlay" /><Dialog.Content className="workspace-sheet">
          <div className="dialog-heading"><Dialog.Title>Workspace tools</Dialog.Title><Dialog.Close asChild><button className="icon-button" aria-label="Close tools"><X size={20} /></button></Dialog.Close></div>
          <Dialog.Description className="muted">Capture, explain and review your inspection evidence.</Dialog.Description>
          <div className="mobile-tool-list">{[{ id: 'camera' as const, label: 'Camera quality station', Icon: Camera }, { id: 'train' as const, label: 'Train PatchCore', Icon: BrainCircuit }, { id: 'lab' as const, label: 'Process what-if', Icon: SlidersHorizontal }].map(({ id, label, Icon }) => <button key={id} onClick={() => { setOpen(false); onSelect(id); }}><Icon size={19} />{label}</button>)}
            <button onClick={() => { setOpen(false); onPresenter(); }}><Presentation size={19} />Presenter walkthrough</button>
            <button onClick={() => { setOpen(false); onPhone(); }}><Smartphone size={19} />Connect another device</button>
          </div>
        </Dialog.Content></Dialog.Portal>
      </Dialog.Root>
    </nav>
  </>;
}
