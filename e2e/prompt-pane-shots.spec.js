// Phase-1 per-mode pane screenshots. Switches the Prompt card through
// each of the four modes (simple / raw / endless / chat) by pressing
// the matching mode-pill button + screenshots both:
//   /tmp/pane-<mode>-card.png      — just the Prompt card (cropped)
//   /tmp/pane-<mode>-full.png      — full viewport (default layout)
// Plus a focused-layout shot per mode:
//   /tmp/pane-<mode>-focused.png   — body[data-layout="subjects"]
//
// Lets the user eyeball pane copy + per-mode visual consistency in
// one go without manually clicking each tab. Not an assertion suite —
// pair with layouts.spec.js for contract checks.

const { test } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const MODES = ['simple', 'raw', 'endless', 'chat'];

for (const mode of MODES) {
    test(`pane=${mode}: prompt-card screenshot`, async ({ page }) => {
        await page.addInitScript(() => {
            try {
                localStorage.clear();
                // Force a generous upper-pane height (700 px) so the whole
                // Prompt card content is visible — the default 200 px min
                // crops everything below the first row in raw / endless.
                localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            } catch (_) {}
        });
        await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });

        // Splash + first paint settle — same idiom as layout-shots.spec.js.
        await page.waitForFunction(() => {
            const splash = document.getElementById('splash-overlay');
            const main = document.querySelector('main');
            const mainOpacity = main ? parseFloat(main.style.opacity || '1') : 1;
            return !splash && mainOpacity >= 1;
        }, null, { timeout: 5000 });

        // Click the matching mode-pill button. The pill lives at the top
        // of the Prompt card; data-subj-mode carries the mode key.
        await page.click(`.subjects-mode-pill button[data-subj-mode="${mode}"]`);
        await page.waitForTimeout(400); // pane swap + heading paint

        // Clear the seed textarea so the per-mode placeholder shows
        // through (otherwise carry-over config.infinity_themes content
        // hides the hint we want to capture).
        await page.evaluate(() => {
            const ta = document.getElementById('p-core');
            if (ta) {
                ta.value = '';
                ta.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });
        await page.waitForTimeout(120);

        // Sanity: the matching .subjects-pane should be visible, others not.
        const paneVisible = await page.evaluate((m) => {
            const pane = document.querySelector(`.subjects-pane[data-pane-mode="${m}"]`);
            if (!pane) return null;
            const cs = window.getComputedStyle(pane);
            return cs.display !== 'none';
        }, mode);
        if (!paneVisible) {
            console.warn(`[pane=${mode}] pane not visible after mode click — screenshot may be misleading`);
        }

        // Whole viewport — gives surrounding context (queue, slop bar).
        await page.screenshot({
            path: `/tmp/pane-${mode}-full.png`,
            fullPage: false,
        });

        // Just the Prompt card body — cropped to the card so it's easier
        // to compare per-pane copy + heading without all the other chrome.
        // #split-left wraps the entire Prompt card in default layout.
        const card = await page.$('#split-left');
        if (card) {
            await card.screenshot({ path: `/tmp/pane-${mode}-card.png` });
        }
    });

    test(`pane=${mode}: prompt-focused layout`, async ({ page }) => {
        await page.addInitScript(() => {
            try { localStorage.clear(); } catch (_) {}
        });
        // ?layout=subjects activates the prompt-focus single-pane mode.
        await page.goto(`${BASE}/?layout=subjects`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => {
            const splash = document.getElementById('splash-overlay');
            const main = document.querySelector('main');
            const mainOpacity = main ? parseFloat(main.style.opacity || '1') : 1;
            return !splash && mainOpacity >= 1;
        }, null, { timeout: 5000 });
        await page.click(`.subjects-mode-pill button[data-subj-mode="${mode}"]`);
        await page.waitForTimeout(400);
        // Clear the seed textarea so the placeholder shows.
        await page.evaluate(() => {
            const ta = document.getElementById('p-core');
            if (ta) {
                ta.value = '';
                ta.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });
        await page.waitForTimeout(120);
        await page.screenshot({
            path: `/tmp/pane-${mode}-focused.png`,
            fullPage: false,
        });
    });
}
