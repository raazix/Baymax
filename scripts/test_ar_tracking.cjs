// Transformation and evidence-identity tests; these do not establish physical camera accuracy.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { webcrypto, createHash } = require('node:crypto');
const ts = require('../apps/dashboard/node_modules/typescript');
function load(name, globals = {}) {
  const exports = {};
  vm.runInNewContext(ts.transpileModule(fs.readFileSync(`apps/dashboard/app/${name}.ts`, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText, { exports, console, ...globals });
  return exports;
}
const tracking = load('arTracking');
const before = Array.from({ length: 16 }, (_, i) => ({ x: 20 + (i % 4) * 20, y: 30 + Math.floor(i / 4) * 20 }));
const angle = .08, scale = 1.04, a = scale * Math.cos(angle), b = scale * Math.sin(angle);
const after = before.map(p => ({ x: a * p.x - b * p.y + 6, y: b * p.x + a * p.y - 3 }));
const motion = tracking.featureMotion(before, after);
assert(motion); assert(Math.abs(motion.a - a) < 1e-6); assert(Math.abs(motion.b - b) < 1e-6);
const moved = tracking.moveQuad(tracking.boxQuad({ x: 80, y: 120, width: 160, height: 320 }), motion, 2);
assert(Math.abs(moved[0].x - (a * 80 - b * 120 + 12)) < 1e-6);
assert.equal(tracking.featureMotion(before.slice(0, 4), after.slice(0, 4)), null);
assert.equal(tracking.featureMotion(before, before.map(p => ({ x: p.x * 2, y: p.y * 2 }))), null);
const corrupt = after.map(p => ({ ...p })); corrupt[0] = { x: 2000, y: -2000 };
assert(tracking.featureMotion(before, corrupt), 'A single outlier must not move the overlay to the background');
const { holdLevel } = load('qualityHold');
const inspection = level => ({ quality: { passed: true }, defects: [{ label: 'anomaly_unclassified', severity: { level } }] });
assert.equal(holdLevel(inspection('high')), 'high'); assert.equal(holdLevel(inspection('critical')), 'critical');
assert.equal(holdLevel(inspection('medium')), null); assert.equal(holdLevel({ ...inspection('critical'), quality: { passed: false } }), null);

const bytes = Buffer.from('unchanged crop bytes'), hash = createHash('sha256').update(bytes).digest('hex');
let wrongHash = false, calls = [];
const { inspectTrackedCrop } = load('arInspection', { crypto: webcrypto, URLSearchParams, URL: {
  createObjectURL: () => 'blob:fixture', revokeObjectURL: () => {},
}, Image: class { async decode() {} }, fetch: async (url, init) => {
  calls.push({ url, init }); return { ok: true, json: async () => ({ id: 'ar-test', image_sha256: wrongHash ? 'wrong' : hash,
    context: { input_source: 'camera', model: 'casting' }, quality: { passed: true }, defects: [] }) };
} });
(async () => {
  const crop = { toBlob: callback => callback(new Blob([bytes], { type: 'image/png' })) };
  await inspectTrackedCrop(crop, { model: 'casting', patchcoreModel: 'brake_disc', machine: 'M-02' }, new AbortController().signal);
  assert(calls[0].url.startsWith('/api/inspections/upload?'));
  assert(calls[0].url.includes('input_source=camera')); assert(calls[0].url.includes('process_context=history'));
  assert(calls[0].url.includes('machine_id=M-02')); assert(calls[0].url.includes('patchcore_model=brake_disc'));
  wrongHash = true;
  await assert.rejects(inspectTrackedCrop(crop, { model: 'casting', patchcoreModel: 'brake_disc', machine: 'M-02' }, new AbortController().signal), /does not match/);
  console.log('PASS: 2D rotation/scale/translation, outlier rejection, weak-tracking rejection, high/critical holds, camera full-pipeline routing and image identity.');
})().catch(error => { console.error(error); process.exitCode = 1; });
