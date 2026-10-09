import type { Metadata, Viewport } from 'next';
import '@fontsource-variable/bricolage-grotesque';
import './globals.css';
import './workspace.css';
import './material.css';
import WorkspaceProviders from './WorkspaceProviders';

export const metadata: Metadata = { title: 'LineGuard · Quality intelligence', description: 'Evidence-linked component inspection and quality decisions' };
export const viewport: Viewport = { width: 'device-width', initialScale: 1, viewportFit: 'cover', themeColor: '#67509b' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><span hidden dangerouslySetInnerHTML={{ __html: '<!-- THESIS: A quality inspection desk where evidence and the next action stay together. OWN-WORLD: Material 3 purple surfaces, Bricolage Grotesque, expressive rounded controls, legible 16px body text. STORY: Capture a part, inspect real model evidence, review a hold, approve a response. FIRST VIEWPORT: Navigation drawer, upload workspace, then a large inspection image with a supporting evidence column. FORM: User-pinned Material 3 operative workspace; seed key material3-user-brief. FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md -->' }} /><WorkspaceProviders>{children}</WorkspaceProviders></body></html>;
}
