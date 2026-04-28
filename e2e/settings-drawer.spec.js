// v303 — settings dialog → side-sliding daisyUI drawer.
//
// Verifies:
//   1. clicking the gear icon (#btn-settings) opens the drawer (toggle = true,
//      drawer-side panel visible)
//   2. clicking the .drawer-overlay closes the drawer (toggle = false)
//   3. the legacy #settings-modal id is preserved on the inner panel so all
//      existing selectors keep matching
//
// Drawer markup lives at slopfinity/templates/index.html:2111
// openSettings() / _setSettingsOpen() wiring lives at slopfinity/static/app.js:7186

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.describe('settings drawer (v303)', () => {
    test('gear icon opens drawer; overlay click closes it', async ({ page }) => {
        await page.goto(BASE + '/');
        await page.waitForLoadState('domcontentloaded');

        // Drawer markup must be present on initial render.
        await expect(page.locator('#settings-drawer.drawer.drawer-end')).toHaveCount(1);
        const toggle = page.locator('#settings-drawer-toggle');
        await expect(toggle).toHaveCount(1);
        // Inner panel keeps the legacy #settings-modal id.
        await expect(page.locator('#settings-modal')).toHaveCount(1);

        // Initial state: closed.
        await expect(toggle).not.toBeChecked();

        // Click the gear → drawer opens.
        await page.click('#btn-settings');
        // openSettings() is async (hydrates form fields) — give it a beat.
        await page.waitForTimeout(250);
        await expect(toggle).toBeChecked();

        // The drawer-side panel must be visible.
        const panel = page.locator('#settings-modal');
        await expect(panel).toBeVisible();
        const box = await panel.boundingBox();
        expect(box).not.toBeNull();
        expect(box.width).toBeGreaterThan(200);

        // Click the overlay → drawer closes. The overlay is a <label>
        // bound to the toggle; click the panel-adjacent edge so we don't
        // hit the inner panel itself.
        // Use position to ensure we land on the overlay (left edge) and
        // not on the side-panel that overlays the right portion.
        const overlay = page.locator('#settings-drawer .drawer-overlay');
        await overlay.click({ position: { x: 10, y: 200 } });
        await page.waitForTimeout(300);
        await expect(toggle).not.toBeChecked();
    });
});
