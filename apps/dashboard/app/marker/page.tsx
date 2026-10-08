import Image from 'next/image';

export default function MarkerPrintPage() {
  return <main className="marker-print">
    <div className="marker-print-instructions"><h1>LineGuard tracking marker</h1><p>Print this page at 100% scale with all fit-to-page options turned off. The marker’s outer black square should measure 40 mm across.</p><p>Place it flat at the center of the rotor hub. Keep it still while using tracked AR.</p><p>Use your browser’s Print command.</p></div>
    <Image className="marker-print-image" src="/ar/hiro.png" alt="AR.js Hiro tracking marker" width={1600} height={1600} priority />
  </main>;
}
