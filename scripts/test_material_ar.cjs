// Browser regression with explicitly synthetic API fixtures and a synthetic moving camera target.
// Exercises UI, holds, notification dispatch, silhouette registration and full inspection routing.
// No provider calls, database writes or physical-camera accuracy claims.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { chromium, devices } = require('../data/browser-tools/node_modules/playwright');
const base = process.env.DASHBOARD_URL || 'http://127.0.0.1:3000';
const output = path.resolve('data/browser-check/material3'); fs.mkdirSync(output, { recursive: true });
const videoPath = path.join(output, 'tracked-target.y4m');
const frames = [], width = 640, height = 480;
for (let frame = 0; frame < 48; frame++) {
  const pixels = Buffer.alloc(width * height * 3 / 2, 128); pixels.fill(174, 0, width * height);
  const offset = Math.round(Math.sin(frame / 48 * Math.PI * 2) * 9);
  for (let y = 84; y < 412; y++) for (let x = 242 + offset; x < 396 + offset; x++) {
    const localX = x - offset;
    pixels[y * width + x] = ((Math.floor(localX / 11) + Math.floor(y / 11)) % 2) ? 62 : 108;
  }
  frames.push(Buffer.from('FRAME\n'), pixels);
}
fs.writeFileSync(videoPath, Buffer.concat([Buffer.from(`YUV4MPEG2 W640 H480 F24:1 Ip A1:1 C420jpeg\n`), ...frames]));
const reference = fs.readFileSync('C:/Users/AYUSH NARAYAN/Downloads/bottle/dents/correct/145.jpg');
let sequence = 0;
function fixture(level = 'medium', model = 'casting', body = reference, input = 'upload') {
  const id = `UI-FIXTURE-${++sequence}`;
  const w = body.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10])) ? body.readUInt32BE(16) : 640;
  const h = body.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10])) ? body.readUInt32BE(20) : 480;
  const box = [w * .22, h * .3, w * .55, h * .62];
  return { id, part_id: id, lot_id: 'SYNTHETIC-UI-LOT', machine_id: 'M-02', created_at: new Date().toISOString(),
    source: 'uploaded_image', scenario: `upload_${model}`, seed: 42, disposition: 'review',
    image_url: `/api/fixture-image/${id}`, image_sha256: createHash('sha256').update(body).digest('hex'),
    quality: { passed: true, profile_note: 'Synthetic browser fixture, not model validation.' }, calibration: {},
    defects: [{ label: model === 'neu' ? 'scratches' : 'anomaly_unclassified', confidence: .72,
      bbox_xyxy_px: box, length_px: 60, width_px: 16,
      severity: { level, reason: 'Synthetic severity fixture for interface regression.' } }],
    telemetry: { temperature_c: 718, pressure_bar: 102, vibration_mm_s: 2.4, machine_speed_rpm: 1210 },
    spatial_fingerprint: {}, model_versions: { vision: 'synthetic-ui-fixture' }, audit: [{ at: new Date().toISOString(), event: 'inspection_created' }],
    analytics: { status: 'synthetic_demonstration', rca: { hypothesis: 'thermal_process_drift', confidence: .74, method: 'Synthetic UI fixture',
      feature_contributions: [{ feature: 'temperature_c', contribution: .84 }, { feature: 'pressure_bar', contribution: .24 }] },
      forecast: { risk: .28, method: 'Synthetic UI fixture' }, uncertainty: { interval_95: [.18, .4], breach_probability: .06, simulations: 10000 } },
    action: { status: 'pending', required: true, text: 'Review the highlighted region and confirm the finding. Inspect the process settings before approving the response.' },
    context: { model, input_source: input, patchcore_model: 'mpdd_metal_plate', process_context: 'history',
      telemetry_source: 'SYNTHETIC UI test telemetry, not measured process data.', part_shape_hint: 'flat_or_unresolved_surface',
      part_identity: { part_type: input === 'upload' ? 'brake_disc' : 'unclassified', source: input === 'upload' ? 'curated_reference_image' : 'not_identified', model_prediction: false, note: 'Synthetic UI reference label' },
      anomaly_role: model === 'casting' ? 'primary' : 'view_only', anomaly: { grid: [[1,1.1,1.2,1],[1,2.4,2.8,1],[1,1.5,2.1,1],[1,1,1,1]], threshold: 2, score: 2.8, flagged: true },
      yolo_view: { role: model === 'neu' ? 'primary' : 'hints', detections: [{ label: 'scratches', confidence: .72, bbox_xyxy_px: box }] },
      detection_merge: { merged: 0, unconfirmed: 1, detections: [{ label: 'scratches', confidence: .72, bbox_xyxy_px: box, merged_into: null }] },
    },
  };
}
async function installFixtures(context) {
  let current = fixture(); const images = new Map([[current.id, reference]]), requests = [], cameraStreams = [];
  let uploadLevel = 'high', cameraLevel = 'medium';
  await context.addInitScript(() => {
    window.__qualityNotifications = [];
    class TestNotification { static permission = 'granted'; static async requestPermission() { return 'granted'; }
      constructor(title, options) { window.__qualityNotifications.push({ title, ...options }); } close() {} }
    window.Notification = TestNotification;
    window.__cameraStreams = [];
    const acquire = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (...args) => { const stream = await acquire(...args); window.__cameraStreams.push(stream); return stream; };
  });
  await context.route('**/api/fixture-image/*', route => {
    const id = new URL(route.request().url()).pathname.split('/').pop();
    const body = images.get(id) || reference;
    return route.fulfill({ contentType: body[0] === 137 ? 'image/png' : 'image/jpeg', body });
  });
  await context.route('**/api/inspections**', async route => {
    const request = route.request(), url = new URL(request.url()), pathname = url.pathname;
    if (pathname.endsWith('/alert-speech')) return route.fulfill({ status: 409, json: { detail: 'Voice mocked for browser test; visual hold remains active.' } });
    if (pathname === '/api/inspections/upload') {
      const body = request.postDataBuffer(), input = url.searchParams.get('input_source') || 'upload';
      current = fixture(input === 'camera' ? cameraLevel : uploadLevel, url.searchParams.get('model'), body, input);
      images.set(current.id, body); requests.push({ source: input, model: current.context.model, sha: current.image_sha256 });
      return route.fulfill({ status: 201, json: current });
    }
    if (pathname.endsWith('/decision')) { current.action.status = 'approved'; return route.fulfill({ json: current }); }
    if (pathname === '/api/inspections') return route.fulfill({ json: [current] });
    if (pathname.endsWith('/assistant')) return route.fulfill({ json: { answer: 'Synthetic UI fixture.', citations: [], model: 'fixture' } });
    return route.fulfill({ json: current });
  });
  await context.route('**/api/replay', route => { current = fixture('critical'); images.set(current.id, reference); return route.fulfill({ status: 201, json: current }); });
  return { requests, setUploadLevel: value => { uploadLevel = value; }, setCameraLevel: value => { cameraLevel = value; } };
}
async function fits(page, label) {
  const shape = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth,
    font: getComputedStyle(document.body).fontFamily, bodySize: getComputedStyle(document.body).fontSize }));
  assert(shape.scroll <= shape.width + 1, `${label} horizontal overflow: ${JSON.stringify(shape)}`);
  assert(shape.font.includes('Bricolage')); assert.equal(shape.bodySize, '16px');
}
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_EXECUTABLE || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-video-capture=${videoPath}`] });
  const errors = [];
  try {
    const desktop = await browser.newContext({ viewport: { width: 1440, height: 1000 }, permissions: ['camera'] });
    const desktopFixtures = await installFixtures(desktop), page = await desktop.newPage(); page.setDefaultTimeout(60000);
    page.on('pageerror', error => errors.push(error.message)); await page.goto(base);
    await page.getByRole('radio', { name: /YOLO detection boxes/ }).waitFor();
    await page.locator('.inspection-image canvas').first().waitFor(); await page.evaluate(() => document.fonts.ready);
    await fits(page, 'Desktop');
    await page.screenshot({ path: path.join(output, 'desktop-inspection.png'), fullPage: true });
    await page.getByRole('radio', { name: /YOLO detection boxes/ }).click();
    await page.locator('.primary-file input[type=file]').first().setInputFiles({ name: 'synthetic-interface-image.jpg', mimeType: 'image/jpeg', buffer: reference });
    await page.locator('.production-hold').waitFor(); await page.locator('.production-scene canvas').waitFor();
    assert((await page.locator('.production-hold').innerText()).includes('high severity'));
    assert.equal(await page.locator('.production-line .status').innerText(), 'Quality Hold');
    assert(await page.getByRole('button', { name: 'Resume after approval', exact: true }).isDisabled());
    await page.waitForFunction(() => window.__qualityNotifications.length === 1);
    await page.locator('.voice-alert').waitFor();
    await page.screenshot({ path: path.join(output, 'desktop-high-hold.png'), fullPage: true });
    await page.getByLabel('Engineer name').fill('Synthetic UI Test Engineer'); await page.getByRole('button', { name: 'Approve', exact: true }).click();
    await page.waitForFunction(() => [...document.querySelectorAll('button')].some(button => button.textContent === 'Resume after approval' && !button.disabled));
    assert.equal(await page.locator('.production-line .status').innerText(), 'Quality Hold', 'Approval alone must not restart the conveyor');
    await page.getByRole('button', { name: 'Resume after approval', exact: true }).click();
    await page.waitForFunction(() => !document.querySelector('.production-hold'));
    assert.equal(await page.locator('.production-line .status').innerText(), 'Running');
    desktopFixtures.setUploadLevel('critical');
    await page.locator('.primary-file input[type=file]').first().setInputFiles({ name: 'critical-interface-fixture.jpg', mimeType: 'image/jpeg', buffer: reference });
    await page.locator('.production-hold').waitFor(); assert((await page.locator('.production-hold').innerText()).includes('critical'));
    assert.equal(await page.locator('.production-line .status').innerText(), 'Quality Hold');
    await fits(page, 'Desktop hold');

    const phone = await browser.newContext({ ...devices['Pixel 5'], deviceScaleFactor: 1, permissions: ['camera'] });
    const phoneFixtures = await installFixtures(phone), mobile = await phone.newPage(); mobile.setDefaultTimeout(90000);
    mobile.on('pageerror', error => errors.push(error.message)); await mobile.goto(base);
    await mobile.getByRole('button', { name: 'Scan part', exact: true }).waitFor();
    await mobile.locator('.inspection-image canvas').first().waitFor(); await mobile.evaluate(() => document.fonts.ready);
    await fits(mobile, 'Phone'); await mobile.screenshot({ path: path.join(output, 'phone-inspection.png'), fullPage: true });
    await mobile.getByRole('button', { name: 'Live AR', exact: true }).click();
    await mobile.waitForFunction(() => document.querySelector('.marker-ar-head')?.textContent.includes('Model highlights anchored'), null, { timeout: 90000 }).catch(async error => { console.error(await mobile.locator('.marker-ar-panel').innerText()); await mobile.screenshot({path:path.join(output,'phone-ar-error.png')}); throw error; });
    assert(phoneFixtures.requests.some(request => request.source === 'camera' && request.model === 'casting'));
    console.log('AR mode:', (await mobile.locator('.marker-ar-panel').innerText()).match(/Silhouette tracking/)?[0]:'unknown');
    await mobile.getByRole('button', { name: 'YOLO boxes', exact: true }).click();
    await mobile.getByRole('button', { name: 'Heatmap', exact: true }).click();
    await mobile.screenshot({ path: path.join(output, 'phone-ar.png') });
    await mobile.evaluate(() => { window.__cameraStreams.at(-1).getVideoTracks()[0].enabled = false; });
    await mobile.waitForFunction(() => document.querySelector('.marker-ar-head')?.textContent.includes('Center a part'));
    assert(await mobile.locator('.ar-capture canvas').count() === 1, 'Captured evidence survives loss of live tracking');
    await mobile.evaluate(() => { window.__cameraStreams.at(-1).getVideoTracks()[0].enabled = true; });
    phoneFixtures.setCameraLevel('high');
    await mobile.waitForFunction(() => !document.querySelector('.ar-rescan')?.disabled);
    await mobile.getByRole('button', { name: 'Recapture', exact: true }).click();
    await mobile.locator('.ar-quality-alert').waitFor();
    await mobile.getByRole('button', { name: 'Close AR inspection' }).click();
    await mobile.locator('.production-hold').waitFor();
    assert.equal(await mobile.locator('.production-line .status').innerText(), 'Quality Hold', 'High AR findings must stop the same production simulation');
    assert(await mobile.evaluate(() => window.__cameraStreams.every(stream => stream.getTracks().every(track => track.readyState === 'ended'))), 'Camera tracks cleaned up after closing AR');
    await fits(mobile, 'Phone high hold');
    await mobile.screenshot({ path: path.join(output, 'phone-high-hold.png'), fullPage: true });
    assert.deepEqual(errors, [], 'No browser runtime errors');
    console.log('PASS: desktop/phone Material 3 controls and layout; high and critical holds; native + in-app notifications; approval/resume interlock; tracked-silhouette AR with Three.js WebGL/fallback; full camera pipeline + hash identity; stale overlay hiding; recapture; camera cleanup. Synthetic camera and model fixtures only.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
