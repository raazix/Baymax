// Runs against local Next.js/FastAPI; creates explicitly synthetic demo records.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '../data/browser-tools/node_modules/playwright');
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_EXECUTABLE, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  try {
    const normal = await fetch('http://127.0.0.1:8000/api/replay', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario: 'normal', seed: 42 }) });
    assert(normal.ok, 'Seed an explicitly synthetic normal record');
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.setDefaultTimeout(60000);
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.DASHBOARD_URL || 'http://127.0.0.1:3001');
    await page.getByRole('button', { name: 'Demonstrate critical defect' }).waitFor();
    await page.getByRole('button', { name: 'Show live line' }).click();
    await page.locator('.production-scene canvas').waitFor();
    await page.getByRole('button', { name: 'Demonstrate critical defect' }).click();
    await page.getByText('Critical defect — simulated production stopped').waitFor();
    const resume = page.getByRole('button', { name: 'Resume simulation after approval' });
    assert(await resume.isDisabled(), 'Pending action cannot resume');
    await page.locator('.voice-alert').waitFor();
    assert.equal(await page.locator('.production-line .status').textContent(), 'Quality hold');
    if (process.env.SAVE_SCREENSHOT) await page.screenshot({ path: 'data/production-critical-demo.png', fullPage: true });
    await page.getByLabel('Engineer name').fill('Production Demo Engineer');
    await page.getByRole('button', { name: 'Approve', exact: true }).click();
    await page.waitForFunction(() => ![...document.querySelectorAll('button')].find(b => b.textContent === 'Resume simulation after approval')?.disabled);
    assert.equal(await page.locator('.production-line .status').textContent(), 'Quality hold', 'Approval alone must not move conveyor');
    await resume.click();
    await page.waitForFunction(() => document.querySelector('.production-line .status')?.textContent === 'Running');
    assert.equal(await page.locator('.production-hold').count(), 0);
    await page.getByRole('button', { name: 'Pause animation' }).click();
    assert.equal(await page.locator('.production-line .status').textContent(), 'Paused');
    await page.getByRole('button', { name: 'Play animation' }).click();
    assert.equal(await page.locator('.production-line .status').textContent(), 'Running');
    assert.deepEqual(errors, [], 'No browser runtime errors');
    console.log('PASS: real API, WebGL scene, critical hold, voice notification, approval interlock, explicit resume, pause/play; no browser errors.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
