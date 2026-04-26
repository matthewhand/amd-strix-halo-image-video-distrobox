// Visual screenshot capture for all 7 layouts. Each layout is loaded via
// ?layout= URL param (so the dashboard wires it on first paint without
// having to programmatically click View dropdown items). Outputs PNGs to
// /tmp/layout-<mode>.png — useful for design QA and regression diffing.
//
// Not an assertion suite; it'll pass even if a layout looks broken. Pair
// with layouts.spec.js for actual contract verification.

const { test } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

const LAYOUTS = ['default', 'subjects', 'queue', 'gallery', 'subj-slop', 'queue-slop', 'subj-queue'];

for (const layout of LAYOUTS) {
    test(`layout=${layout}: full-page screenshot`, async ({ page }) => {
        await page.addInitScript(() => {
            try { localStorage.clear(); } catch (_) {}
        });
        await page.goto(`${BASE}/?layout=${layout}`, { waitUntil: 'domcontentloaded' });
        // Wait for splash fade — same trick as layouts.spec.js.
        await page.waitForFunction(() => {
            const splash = document.getElementById('splash-overlay');
            const main = document.querySelector('main');
            const mainOpacity = main ? parseFloat(main.style.opacity || '1') : 1;
            return !splash && mainOpacity >= 1;
        }, null, { timeout: 5000 });
        // Small settle for any deferred renders (queue badges, slop grid).
        await page.waitForTimeout(400);
        await page.screenshot({
            path: `/tmp/layout-${layout}.png`,
            fullPage: false,
        });
    });
}
