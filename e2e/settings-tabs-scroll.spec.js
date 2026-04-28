// v303 — scroll-arrow controls on the settings tab strip.
//
// Verifies:
//   1. tabs strip is wrapped in .tabs-scroll-wrap
//   2. .tabs-scroll-prev + .tabs-scroll-next chevron buttons render
//   3. wrap initially has .at-start (we open at scrollLeft = 0)
//   4. clicking .tabs-scroll-next scrolls the strip right (scrollLeft > 0)
//   5. window._settingsTabsScroll is exposed
//
// Markup: slopfinity/templates/index.html:2142
// Wiring: slopfinity/static/app.js:7199

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.describe('settings tab-strip scroll arrows (v303)', () => {
    test('arrows render, exposed helper works, scroll moves strip', async ({ page }) => {
        // Force a smaller viewport so the 9-tab strip overflows and the
        // scroll arrows have something to do.
        await page.setViewportSize({ width: 900, height: 800 });
        await page.goto(BASE + '/');
        await page.waitForLoadState('domcontentloaded');

        // Open the settings drawer so the tabs strip is hydrated/measurable.
        await page.click('#btn-settings');
        await page.waitForTimeout(400);

        // Both arrows render inside the wrap.
        const wrap = page.locator('.tabs-scroll-wrap');
        await expect(wrap).toHaveCount(1);
        await expect(wrap.locator('.tabs-scroll-arrow.tabs-scroll-prev')).toHaveCount(1);
        await expect(wrap.locator('.tabs-scroll-arrow.tabs-scroll-next')).toHaveCount(1);

        // The strip is overflow-x-auto.
        const overflowX = await page.evaluate(() => {
            const el = document.getElementById('settings-tab-strip');
            return el && getComputedStyle(el).overflowX;
        });
        expect(['auto', 'scroll']).toContain(overflowX);

        // window._settingsTabsScroll is exposed.
        const helperType = await page.evaluate(() => typeof window._settingsTabsScroll);
        expect(helperType).toBe('function');

        // Initial state: wrap has .at-start (scrollLeft is 0).
        await expect(wrap).toHaveClass(/at-start/);

        // Click the next chevron → scrollLeft moves right.
        const startScroll = await page.evaluate(() => document.getElementById('settings-tab-strip').scrollLeft);
        expect(startScroll).toBe(0);

        await page.click('.tabs-scroll-next');
        // smooth scroll + 240px target — give it a beat.
        await page.waitForTimeout(600);

        const afterScroll = await page.evaluate(() => document.getElementById('settings-tab-strip').scrollLeft);
        expect(afterScroll).toBeGreaterThan(0);

        // After scrolling away from the left edge, .at-start should drop.
        await expect(wrap).not.toHaveClass(/at-start/);
    });
});
