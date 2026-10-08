// Real local backend; phone camera frames are generated test imagery, not hardware validation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, devices } = require(process.env.PLAYWRIGHT_MODULE || '../data/browser-tools/node_modules/playwright');
const base = process.env.DASHBOARD_URL || 'http://127.0.0.1:3001';
// Deterministic textured grayscale fixture, with neutral chroma and no specular pixels.
fs.mkdirSync('data/browser-check', { recursive: true });
const fixturePath = 'data/browser-check/phone-camera.y4m';
if (!fs.existsSync(fixturePath)) {
  const width = 640, height = 480;
  const frame = Buffer.alloc(width * height * 3 / 2, 128);
  let seed = 42;
  for (let i = 0; i < width * height; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    frame[i] = 60 + (seed >>> 24) % 130;
  }
  fs.writeFileSync(fixturePath, Buffer.concat([
    Buffer.from(`YUV4MPEG2 W${width} H${height} F30:1 Ip A1:1 C420jpeg\n`),
    ...Array.from({ length: 8 }, () => Buffer.concat([Buffer.from('FRAME\n'), frame])),
  ]));
}
async function normal() {
  const response = await fetch('http://127.0.0.1:8000/api/replay', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario: 'normal', seed: 42 }) });
  assert(response.ok, 'Normal synthetic replay available');
}
async function fits(page, label) {
  const result = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, viewport: innerWidth }));
  assert(result.scroll <= result.viewport + 1, `${label}: horizontal overflow ${result.scroll}/${result.viewport}`);
}
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_EXECUTABLE, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-video-capture=${path.resolve('data/browser-check/phone-camera.y4m')}`] });
  const errors = [];
  try {
    await normal();
    const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    desktop.on('pageerror', error => errors.push(error.message));
    await desktop.goto(base); await desktop.getByRole('button', { name: 'Scan with camera', exact: true }).waitFor();
    await desktop.waitForFunction(() => !document.querySelector('.scan-camera-action')?.disabled);
    await desktop.waitForFunction(() => document.querySelector('.component img')?.naturalWidth > 0);
    await fits(desktop, 'desktop');
    assert.equal(await desktop.locator('.production-scene canvas').count(), 0, '3D work deferred until requested');
    await desktop.screenshot({ path: 'data/browser-check/workspace-desktop.png', fullPage: true });

    const mobile = await browser.newContext({ ...devices['Pixel 5'], deviceScaleFactor: 1, permissions: ['camera', 'microphone'] });
    const phone = await mobile.newPage(); phone.setDefaultTimeout(60000); phone.on('pageerror', error => errors.push(error.message));
    await phone.goto(base); await phone.getByRole('button', { name: 'Scan part', exact: true }).waitFor();
    await phone.waitForFunction(() => !document.querySelector('.mobile-scan')?.disabled);
    await phone.waitForFunction(() => document.querySelector('.component img')?.naturalWidth > 0);
    await fits(phone, 'Android viewport');
    await phone.screenshot({ path: 'data/browser-check/workspace-phone.png', fullPage: true });
    await phone.getByRole('button', { name: 'Scan part', exact: true }).click();
    const capture = phone.getByRole('button', { name: 'Capture and inspect', exact: true });
    await phone.waitForFunction(() => [...document.querySelectorAll('button')].some(button => button.textContent === 'Capture and inspect' && !button.disabled));
    assert.equal(await phone.locator('input[capture]').getAttribute('capture'), 'environment', 'Native rear-camera fallback exists');
    await fits(phone, 'phone camera dialog');
    await phone.screenshot({ path: 'data/browser-check/workspace-phone-camera.png' });
    const pending = phone.waitForResponse(response => response.url().includes('/api/inspections/upload') && response.request().method() === 'POST');
    await capture.click(); const response = await pending; assert.equal(response.status(), 201);
    const record = await response.json();
    assert.equal(record.context.input_source, 'camera'); assert(record.quality.passed, 'Generated sharp, non-specular camera frame passes gate');
    assert(record.analytics && record.action, 'Actual image model and process analytics completed');
    assert(record.image_sha256 && record.audit.some(event => event.actor === 'camera_scan'), 'Evidence persisted');
    await phone.getByRole('dialog').waitFor({ state: 'hidden' });
    await phone.getByRole('heading', { name: 'Camera inspection', exact: true }).waitFor();
    await fits(phone, 'camera inspection result');
    await phone.getByRole('button', { name: 'Sensors', exact: true }).click();
    await phone.getByRole('heading', { name: 'Historical sensor data', exact: true }).waitFor(); await fits(phone, 'sensor history');
    await phone.getByRole('button', { name: 'More tools' }).click();
    await phone.getByRole('button', { name: 'Process what-if', exact: true }).click();
    await phone.getByRole('heading', { name: 'Process what-if lab', exact: true }).waitFor(); await fits(phone, 'process lab');
    await phone.getByRole('button', { name: 'Inspect', exact: true }).click();
    await phone.getByRole('button', { name: 'Show live line' }).click(); await phone.locator('.production-scene canvas').waitFor();
    const critical = phone.waitForResponse(response => response.url().endsWith('/api/replay') && response.request().method() === 'POST');
    await phone.getByRole('button', { name: 'Demonstrate critical defect' }).click(); await critical;
    await phone.getByText('Critical defect — simulated production stopped').waitFor();
    const resume = phone.getByRole('button', { name: 'Resume simulation after approval' }); assert(await resume.isDisabled());
    await phone.getByLabel('Engineer name').fill('Phone Demo Engineer'); await phone.getByRole('button', { name: 'Approve', exact: true }).click();
    await phone.waitForFunction(() => ![...document.querySelectorAll('button')].find(button => button.textContent === 'Resume simulation after approval')?.disabled);
    assert.equal(await phone.locator('.production-line .status').textContent(), 'Quality hold'); await resume.click();
    await phone.waitForFunction(() => document.querySelector('.production-line .status')?.textContent === 'Running'); await fits(phone, 'mobile production lifecycle');
    await mobile.close();

    await normal();
    const iphone = await browser.newPage({ ...devices['iPhone 13'], deviceScaleFactor: 1, reducedMotion: 'reduce' });
    iphone.on('pageerror', error => errors.push(error.message));
    await iphone.goto(base); await iphone.getByRole('button', { name: 'More tools' }).waitFor(); await fits(iphone, 'iPhone viewport');
    await iphone.waitForFunction(() => document.querySelector('.component img')?.naturalWidth > 0);
    await iphone.screenshot({ path: 'data/browser-check/workspace-iphone.png', fullPage: true });
    await iphone.setViewportSize({ width: 320, height: 568 }); await fits(iphone, 'narrow 320px phone');
    assert.deepEqual(errors, [], 'No browser runtime errors');
    fs.writeFileSync('data/browser-check/mobile-verification.json', JSON.stringify({ passed: true, camera_inspection_id: record.id, viewport_checks: ['1440 desktop', 'Pixel 5', 'iPhone 13 dimensions', '320px phone'], camera: 'generated WebRTC frame; physical phone not tested', runtime_errors: errors }, null, 2));
    console.log('PASS: desktop/mobile layouts, deferred 3D, phone camera through actual model pipeline, sensors/lab navigation, critical approval/resume, iPhone dimensions and narrow phone; no runtime errors.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
