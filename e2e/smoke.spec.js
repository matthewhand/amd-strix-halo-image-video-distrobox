// Slopfinity dashboard smoke test.
//
// Checks the page loads without JS parse errors, every primary card
// (Subjects / Queue / Slop) renders with non-zero size, and a small
// set of expected elements are present in the DOM. Screenshots the
// failure case so we can eyeball what regressed.
//
// Run: npx playwright test e2e/smoke.spec.js
//   (assumes slopfinity is reachable at http://localhost:9099)

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.describe('slopfinity dashboard smoke', () => {
    test('loads with no JS errors and key cards render', async ({ page }) => {
        // Fail fast if JS errors fire — these silently kill everything
        // (the stray '}' bug from earlier today wouldn't have shown
        // anything broken without this listener).
        const errors = [];
        page.on('pageerror', err => errors.push(err.message));
        const consoleErrors = [];
        page.on('console', msg => {
            if (msg.type() === 'error') consoleErrors.push(msg.text());
        });

        await page.goto(BASE, { waitUntil: 'domcontentloaded' });
        // Give the WS handshake + initial render a moment.
        await page.waitForTimeout(800);

        // Capture a screenshot regardless so the artefact is on disk
        // for review when something fails further down.
        await page.screenshot({
            path: 'e2e/artifacts/smoke.png',
            fullPage: false,
        });

        expect(errors, `pageerror: ${errors.join('\n')}`).toEqual([]);

        // Helper: assert an element exists AND has non-zero rendered size.
        const expectVisible = async (selector, label) => {
            const el = page.locator(selector);
            await expect(el, `${label} (${selector}) should exist`).toHaveCount(1);
            const box = await el.boundingBox();
            expect(box, `${label} should have a bounding box`).not.toBeNull();
            expect(box.width, `${label} width`).toBeGreaterThan(0);
            expect(box.height, `${label} height`).toBeGreaterThan(0);
        };

        await expectVisible('#split-left', 'Subjects card');
        await expectVisible('#split-right', 'Queue card');
        await expectVisible('#output-section', 'Slop output');

        // Subjects card internals — if these are missing, the mode pill
        // refactor regressed something.
        await expect(page.locator('#p-core'), 'seed textarea').toBeVisible();
        await expect(
            page.locator('.subjects-mode-pill button[data-subj-mode]'),
            'mode pill segments'
        ).toHaveCount(4);
        // Either the ↻ refresh button (#subjects-suggest-btn) OR the +
        // add button (#subjects-suggest-add-btn) should be visible at
        // any given time — never both. The badge swaps between them
        // based on mode + state (simple pre-first-batch shows +;
        // simple post-batch shows ↻; endless always shows +; raw/chat
        // hide both). On a fresh load with no cached suggestions the
        // simple mode is active and + shows.
        const refreshOrAdd = page.locator(
            '#subjects-suggest-btn:not(.hidden), #subjects-suggest-add-btn:not(.hidden)'
        );
        await expect(refreshOrAdd, 'Suggest refresh OR add button').toHaveCount(1);
        await expect(refreshOrAdd.first(), 'Suggest refresh/add visible').toBeVisible();

        // Top-bar nav: settings cog + view dropdown should be reachable.
        await expect(page.locator('#btn-settings'), 'settings cog button').toBeVisible();

        // Console errors are advisory — log but don't fail (some browser
        // warnings are unavoidable: unbound media element, etc).
        if (consoleErrors.length) {
            console.warn('console errors during load:', consoleErrors);
        }
    });

    test('Settings modal opens and shows tabs', async ({ page }) => {
        await page.goto(BASE);
        await page.waitForTimeout(400);
        // openSettings is the canonical entry; click the gear button.
        await page.click('#btn-settings');
        const modal = page.locator('#settings-modal');
        await expect(modal).toBeVisible();
        // Tabs strip should have multiple tabs.
        const tabs = modal.locator('input[name="settings_tabs"]');
        const count = await tabs.count();
        expect(count, 'settings tab count').toBeGreaterThan(3);
        await page.screenshot({
            path: 'e2e/artifacts/settings-open.png',
            fullPage: false,
        });
    });
});
