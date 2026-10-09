const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('../data/browser-tools/node_modules/playwright');
const pairing = JSON.parse(fs.readFileSync('data/phone-demo.json', 'utf8'));
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_EXECUTABLE || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 393, height: 852 }, permissions: ['camera'] });
    const page = await context.newPage(); page.setDefaultTimeout(20000);
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    const response = await page.goto(`${pairing.url}/phone-access`, { waitUntil: 'domcontentloaded', timeout: 45000 });
    assert.equal(response.status(), 200, 'public phone route opens');
    const preAuth = await page.evaluate(async () => (await fetch('/api/assistant/status')).status);
    assert.equal(preAuth, 401, 'unpaired API is private');
    await page.getByLabel('Pairing code').fill(pairing.code);
    await page.getByRole('button', { name: 'Open workspace' }).click();
    await page.waitForURL(url => url.pathname === '/', { timeout: 30000 });
    await page.getByRole('heading', { name: 'Inspections' }).waitFor({ timeout: 30000 }).catch(async () => {
      if (!(await page.locator('body').innerText()).includes('Inspect components')) throw new Error('Paired workspace did not render.');
    });
    const cookie = (await context.cookies()).find(item => item.name === 'lineguard-phone-session');
    assert(cookie && cookie.secure && cookie.httpOnly, 'secure signed session cookie is set');
    const postAuth = await page.evaluate(async () => (await fetch('/api/assistant/status')).status);
    assert.equal(postAuth, 200, 'paired phone can reach protected API through tunnel');
    const camera = await page.evaluate(() => ({ secure: isSecureContext, available: Boolean(navigator.mediaDevices?.getUserMedia) }));
    assert(camera.secure && camera.available, 'phone route has secure camera APIs');
    assert.deepEqual(errors, [], 'no client runtime errors');
    console.log('PASS: HTTPS tunnel, pairing, secure session, protected API, camera context; no runtime errors.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
