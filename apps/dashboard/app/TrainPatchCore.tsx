'use client';

import { useState } from 'react';

type Report = { name: string; train_images: number; heldout_images: number; bank_patches: number; threshold: number; heldout_score_max: number; heldout_score_mean: number; threshold_margin: number; seconds: number; artifact_sha256: string };

async function send<T>(url: string, body?: File): Promise<T> {
  const response = await fetch(url, { method: 'POST', body, headers: body ? { 'Content-Type': body.type || 'image/jpeg' } : undefined, cache: 'no-store' });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}).`);
  return payload;
}

export default function TrainPatchCore({ onTrained }: { onTrained: (name: string) => void }) {
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState('');
  const [error, setError] = useState('');
  const [report, setReport] = useState<Report | null>(null);
  const [skipped, setSkipped] = useState<string[]>([]);
  const nameOk = /^[a-z0-9][a-z0-9_-]{1,39}$/.test(name);

  async function train(files: FileList | null) {
    if (!files || !nameOk) return;
    const list = [...files];
    setBusy(true); setError(''); setReport(null); setSkipped([]);
    const rejected: string[] = [];
    try {
      for (const [index, file] of list.entries()) {
        setProgress(`Checking and storing image ${index + 1} of ${list.length}…`);
        try { await send(`/api/patchcore/custom/${name}/images`, file); }
        catch (e) { rejected.push(`${file.name}: ${e instanceof Error ? e.message : 'rejected'}`); }
      }
      setSkipped(rejected);
      setProgress('Building the memory bank and calibrating the threshold…');
      const result = await send<Report>(`/api/patchcore/custom/${name}/train`);
      setReport(result);
      onTrained(name);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Training failed.');
    } finally {
      setBusy(false); setProgress('');
    }
  }

  return <section className="panel train">
    <h2>Train PatchCore on your casting</h2>
    <p>PatchCore learns what a <strong>good</strong> part looks like, so it only needs normal images of your casting. Upload at least 20 clear photos of good parts (more is better, up to 400). The model is trained in seconds and then appears in the casting-model selector.</p>
    <div className="buttons">
      <label className="model-pick">Model name
        <input value={name} onChange={e => setName(e.target.value.toLowerCase())} placeholder="e.g. engine-block-a" maxLength={40} disabled={busy} aria-invalid={name !== '' && !nameOk} />
      </label>
      <label className={`file-button primary-file${busy || !nameOk ? ' disabled' : ''}`}>{busy ? 'Training…' : 'Choose normal images and train'}
        <input type="file" multiple accept="image/jpeg,image/png" disabled={busy || !nameOk} onChange={e => { void train(e.target.files); e.target.value = ''; }} />
      </label>
    </div>
    {name !== '' && !nameOk && <p className="muted">Use 2–40 lowercase letters, digits, hyphens or underscores.</p>}
    {progress && <p role="status">{progress}</p>}
    {error && <p className="inference-error" role="alert">{error}</p>}
    {skipped.length > 0 && <details className="skipped"><summary>{skipped.length} image{skipped.length === 1 ? '' : 's'} skipped</summary><ul>{skipped.map(item => <li key={item}>{item}</li>)}</ul></details>}
    {report && <div className="train-result">
      <h3>Model “{report.name}” trained in {report.seconds}s</h3>
      <dl className="measurements">
        <div><dt>Memory-bank images</dt><dd>{report.train_images}</dd></div>
        <div><dt>Held-out normal images</dt><dd>{report.heldout_images}</dd></div>
        <div><dt>Decision threshold</dt><dd>{report.threshold.toFixed(3)}</dd></div>
        <div><dt>Held-out normal scores</dt><dd>mean {report.heldout_score_mean.toFixed(2)} · max {report.heldout_score_max.toFixed(2)}</dd></div>
      </dl>
      <p className="footnote">Threshold = highest held-out normal score × {report.threshold_margin}. No defect images were used, so the false-alarm rate is estimated from only {report.heldout_images} images and detection of real defects on this part is unmeasured. On the public casting test set, models trained this way on 25–60 normals reached 0–3% false alarms but caught roughly 20–40% of defects; more normal images improve that. Select it in the casting-model list on the Inspection console.</p>
    </div>}
  </section>;
}
