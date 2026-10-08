// Regression checks for camera crop registration and inference/image identity.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto, createHash } = require('node:crypto');
const ts = require('../apps/dashboard/node_modules/typescript');
const source = fs.readFileSync(path.join(__dirname, '../apps/dashboard/app/bottleInference.ts'), 'utf8');
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
const exportsObject = {};
let pixels, requests = [], qualityPassed = true, returnWrongHash = false;
const bytes = Buffer.from('captured-crop');
const hash = createHash('sha256').update(bytes).digest('hex');
const result = { anomaly_grid: [[1, 2, 3]], threshold: 2, anomaly_score: 3,
  image_sha256: hash, frame_id: 'camera-frame', patchcore_model: 'bottle' };
const context = { exports: exportsObject, crypto: webcrypto, Blob, Uint8Array, Array, Number, Math, Error,
  document: { createElement: () => ({ getContext: () => ({
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: image => { pixels = image.data; },
  }) }) },
  fetch: async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => url.startsWith('/api/frames?')
      ? { id: 'camera-frame', sha256: hash, quality: { passed: qualityPassed } }
      : { ...result, image_sha256: returnWrongHash ? 'unrelated-frame' : hash } };
  },
};
vm.runInNewContext(js, context);
const { clampCrop, sourceCrop, coverTransform, validateAnomaly, anomalyLayer, inspectBottleCrop } = exportsObject;

async function main() {
  assert.equal(JSON.stringify(clampCrop({ x: -2, y: 8.4, width: 15, height: 20 }, 10, 12)),
    JSON.stringify({ x: 0, y: 8, width: 10, height: 4 }));
  const transform = coverTransform(640, 480, 300, 600);
  assert.equal(transform.scale, 1.25);
  assert.equal(transform.x, -250); assert.equal(transform.y, 0);
  assert.equal(200 * transform.scale + transform.x, 0); // Source crop x maps through the same cover crop as video.
  assert.equal(JSON.stringify(sourceCrop({ x: 50, y: 20, width: 75, height: 180 }, 320, 240, 960, 720)),
    JSON.stringify({ x: 150, y: 60, width: 225, height: 540 })); // Small-frame tracking preserves native inference pixels.
  assert.throws(() => validateAnomaly(result, 'wrong-frame'), /does not match/);
  assert.throws(() => validateAnomaly({ ...result, anomaly_grid: [[1], [2, 3]] }, hash), /invalid/);
  assert.throws(() => validateAnomaly({ ...result, anomaly_grid: [[NaN]] }, hash), /invalid/);
  anomalyLayer(result);
  assert.equal(pixels[3], 0); assert.equal(pixels[7], 0); assert.ok(pixels[11] > 0);
  const signal = new AbortController().signal;
  const crop = { toBlob: callback => callback(new Blob([bytes], { type: 'image/png' })) };
  const output = await inspectBottleCrop(crop, 'bottle', signal);
  assert.equal(output.image_sha256, hash); assert.equal(requests.length, 2);
  assert.equal(requests[0].options.signal, signal);
  assert.equal(requests[0].options.headers['Content-Type'], 'image/png');
  assert.match(requests[1].url, /camera-frame\/anomaly\?patchcore_model=bottle$/);
  qualityPassed = false; requests = [];
  await assert.rejects(inspectBottleCrop(crop, 'bottle', signal), /quality gate/);
  assert.equal(requests.length, 1); // Failed images never reach the model.
  qualityPassed = true; returnWrongHash = true;
  await assert.rejects(inspectBottleCrop(crop, 'bottle', signal), /does not match/);
  console.log('PASS: crop bounds, cover mapping, threshold transparency, invalid grids, image identity, quality gate and request contract.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
