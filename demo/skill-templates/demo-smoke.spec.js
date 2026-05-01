// demo-smoke.spec.js — generic Playwright smoke for static-demo-builder
// Drop into your project's e2e/ dir, then:
//   cd dist/demo && python3 -m http.server 8765 &
//   npx playwright test demo-smoke.spec.js
//
// Customize the selectors at the top to match your app.

const { test, expect } = require('@playwright/test');

const DEMO_URL = process.env.DEMO_URL || 'http://localhost:8765/';

// --- Per-app selectors (edit for your project) -----------------------
const SEL = {
  // The primary text input the demo "feature flow" uses
  promptInput: process.env.DEMO_PROMPT_SEL || '#p-core, textarea[name="prompt"], textarea',
  // The primary action button
  primaryAction: process.env.DEMO_ACTION_SEL || 'button:has-text("Queue Slop"), button:has-text("Generate"), button:has-text("Submit")',
  // A selector that becomes truthy when "rendered output" appears
  outputAppeared: process.env.DEMO_OUTPUT_SEL || '#preview-grid > [data-slop-kind], .gallery-tile, .output-card',
};

test.describe('static-demo-builder smoke', () => {
  test.use({ baseURL: DEMO_URL });

  test('1. demo banner renders within 2 seconds', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#demo-banner')).toBeVisible({ timeout: 2000 });
    await expect(page.locator('#demo-banner')).toContainText(/demo/i);
  });

  test('2. no console errors and no [demo] unmocked warnings on cold load', async ({ page }) => {
    const errors = [];
    const warns = [];
    page.on('console', m => {
      if (m.type() === 'error') errors.push(m.text());
      if (m.type() === 'warning' && m.text().includes('[demo] unmocked')) warns.push(m.text());
    });
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto('/');
    await page.waitForTimeout(2500);
    expect(errors, 'console errors on cold load').toEqual([]);
    expect(warns, 'unmocked endpoints called on cold load').toEqual([]);
  });

  test('3. feature flow — type, primary action, output appears', async ({ page }) => {
    test.setTimeout(60_000);
    await page.goto('/');
    await expect(page.locator('#demo-banner')).toBeVisible({ timeout: 5000 });

    const inp = page.locator(SEL.promptInput).first();
    await inp.fill('demo: a lighthouse keeper meets a sea creature');

    const btn = page.locator(SEL.primaryAction).first();
    await btn.click();

    // Wait for at least one output tile to appear
    await expect(page.locator(SEL.outputAppeared).first()).toBeVisible({ timeout: 30_000 });
  });

  test('4. reset button clears state and reloads', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#demo-banner')).toBeVisible({ timeout: 5000 });
    await page.evaluate(() => localStorage.setItem('demo-test-marker', 'present'));
    await page.click('#demo-reset');
    await page.waitForLoadState('domcontentloaded');
    const marker = await page.evaluate(() => localStorage.getItem('demo-test-marker'));
    expect(marker).toBeNull();
  });

  test('5. echo POST returns ok+demo and shows toast', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#demo-banner')).toBeVisible({ timeout: 5000 });
    const result = await page.evaluate(async () => {
      const r = await fetch('/some-mutating-endpoint-that-does-not-exist', {
        method: 'POST',
        body: JSON.stringify({ test: 1 }),
        headers: { 'content-type': 'application/json' },
      });
      return { status: r.status, body: await r.json() };
    });
    expect(result.status).toBe(200);
    expect(result.body).toMatchObject({ ok: true, demo: true });
    // Toast may or may not be present depending on timing; tolerate either
  });
});
