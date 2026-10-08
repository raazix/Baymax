import type { Metadata, Viewport } from 'next';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-sans/600.css';
import './globals.css';
import './workspace.css';
import WorkspaceProviders from './WorkspaceProviders';

export const metadata: Metadata = { title: 'LineGuard · Quality intelligence', description: 'Evidence-linked component inspection and quality decisions' };
export const viewport: Viewport = { width: 'device-width', initialScale: 1, viewportFit: 'cover', themeColor: '#183b3b' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><WorkspaceProviders>{children}</WorkspaceProviders></body></html>;
}
