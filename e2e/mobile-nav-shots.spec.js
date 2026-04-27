// Mobile-only nav bar screenshots. Captures the three single-card
// layouts (prompt / queue / slop) at a phone viewport with the bottom
// nav bar visible, so the user can verify left/right arrows + labels
// + the auto-redirect from multi-pane layouts works.

const { test } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const MOBILE_VIEWPORT = { width: 390, height: 844 };  // iPhone 12-ish
const LAYOUTS = [
    { layout: 'subjects', label: 'prompt' },
    { layout: 'queue',    label: 'queue'  },
    { layout: 'gallery',  label: 'slop'   },
];

test.use({ viewport: MOBILE_VIEWPORT });

for (const { layout, label } of LAYOUTS) {
    test(`mobile ${label} layout with bottom nav`, async ({ page }) => {
        await page.addInitScript(() => {
            try { localStorage.clear(); } catch (_) {}
        });
        await page.goto(`${BASE}/?layout=${layout}`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => {
            const splash = document.getElementById('splash-overlay');
            const main = document.querySelector('main');
            const mainOpacity = main ? parseFloat(main.style.opacity || '1') : 1;
            return !splash && mainOpacity >= 1;
        }, null, { timeout: 5000 });
        await page.waitForTimeout(400);
        await page.screenshot({
            path: `/tmp/mobile-${label}.png`,
            fullPage: false,
        });

        // Sanity: bottom nav bar is visible.
        const nav = await page.locator('#mobile-nav-bar').boundingBox();
        if (!nav) console.warn(`[mobile-${label}] mobile-nav-bar has no bounding box — CSS @media may not have matched`);
        else if (nav.width < 200) console.warn(`[mobile-${label}] nav bar width=${nav.width} — too narrow`);
    });
}

// Bonus: verify mobile auto-redirect from multi-pane → subjects.
test('mobile redirects multi-pane to prompt', async ({ page }) => {
    await page.addInitScript(() => {
        try { localStorage.clear(); } catch (_) {}
    });
    // Go to default (no layout query) — desktop default is "all 3 panes",
    // mobile should redirect to prompt-only.
    await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        return !splash;
    }, null, { timeout: 5000 });
    await page.waitForTimeout(500);
    const layout = await page.evaluate(() => document.body.dataset.layout || '');
    if (layout !== 'subjects') {
        console.warn(`[mobile-redirect] expected subjects, got "${layout}"`);
    }
    await page.screenshot({ path: '/tmp/mobile-default-redirect.png', fullPage: false });
});
