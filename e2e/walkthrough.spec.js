// Happy-path walkthrough — one screenshot per meaningful state.
// Output: e2e/artifacts/walkthrough/NN_label.png
// Run:    npx playwright test e2e/walkthrough.spec.js --reporter=list
const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const OUT = 'e2e/artifacts/walkthrough';

const shot = (page, name) => page.screenshot({ path: `${OUT}/${name}.png`, fullPage: false });

const dismissSplash = async (page) => {
    // Splash overlay (#splash-overlay) auto-dismisses via _hideSplash() in
    // slopfinity/static/app.js: fades to opacity:0, then calls el.remove()
    // 600ms later, then fades <main> in over 400ms. Trigger paths:
    //   1. First WS state tick (handleTick)
    //   2. window.load + 2.5s timeout (belt-and-braces)
    // Worst case ~3.1s after `load` before the node detaches, plus a 400ms
    // main fade-in. Wait for detachment (the canonical end-state) then for
    // <main> to reach opacity:1 — otherwise screenshots can catch either
    // the splash itself or the half-faded handoff.
    const splash = page.locator('#splash-overlay');
    try {
        await splash.waitFor({ state: 'detached', timeout: 10000 });
    } catch (_) {
        // Fall back: force-trigger the hide and wait hidden.
        await page.evaluate(() => { try { window._hideSplash && window._hideSplash(); } catch (_) { } });
        await splash.waitFor({ state: 'detached', timeout: 5000 }).catch(() => { });
    }
    // Wait for the dashboard fade-in (<main> opacity transitions 0 → 1).
    await page.waitForFunction(() => {
        const m = document.querySelector('main');
        if (!m) return false;
        return parseFloat(getComputedStyle(m).opacity) > 0.95;
    }, null, { timeout: 5000 }).catch(() => { });
};

test.describe.configure({ mode: 'serial' });

test.describe('slopfinity happy-path walkthrough', () => {
    let page;
    const issues = [];

    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
        page.on('pageerror', e => issues.push(`pageerror: ${e.message}`));
        page.on('console', m => { if (m.type() === 'error') issues.push(`console.error: ${m.text()}`); });
        // Pre-navigate so individual tests can be run via --grep without needing test 01.
        await page.goto(BASE, { waitUntil: 'domcontentloaded' });
        await dismissSplash(page);
    });

    test('01 landing', async () => {
        // beforeAll already navigated + dismissed splash. Re-assert dismissal
        // (cheap when already detached) so this test stays runnable alone via
        // --grep without depending on beforeAll's pre-navigation order.
        await dismissSplash(page);
        await shot(page, '01_landing');
        await expect(page.locator('#split-left')).toBeVisible();
        await expect(page.locator('#split-right')).toBeVisible();
        await expect(page.locator('#output-section')).toBeVisible();
    });

    test('02 mode pill: endless / simple / raw / chat', async () => {
        for (const mode of ['endless', 'simple', 'raw', 'chat']) {
            const btn = page.locator(`button[data-subj-mode="${mode}"]`).first();
            if (await btn.count() === 0) { issues.push(`mode ${mode} button missing`); continue; }
            await btn.click();
            await page.waitForTimeout(400);
            await shot(page, `02_mode_${mode}`);
        }
    });

    test('03 prompt entry', async () => {
        // Back to simple mode for the prompt flow.
        await page.locator('button[data-subj-mode="simple"]').first().click();
        await page.waitForTimeout(300);
        await shot(page, '03a_simple_empty');
        const ta = page.locator('#p-core');
        await ta.fill('a neon cat surfing on a glowing wave at sunset');
        await page.waitForTimeout(200);
        await shot(page, '03b_prompt_typed');
    });

    test('04 queue submission', async () => {
        const startBtn = page.locator('#btn-start-stop-inline');
        const exists = await startBtn.count();
        if (!exists) { issues.push('btn-start-stop-inline missing'); return; }
        await startBtn.click();
        await page.waitForTimeout(800);
        await shot(page, '04_queue_submitted');
    });

    test('05 settings drawer', async () => {
        await page.locator('#btn-settings').click();
        const modal = page.locator('#settings-modal');
        await expect(modal).toBeVisible({ timeout: 5000 });
        await page.waitForTimeout(300);
        await shot(page, '05a_settings_open');

        // Settings uses DaisyUI tabs-lifted: the radio inputs themselves are
        // the visible tabs (CSS :checked + .tab-content swaps panels). Clicking
        // the <input> via Playwright was unreliable because the radios live in
        // a horizontally-scrolling strip and may be off-screen; force a state
        // change via .check() + dispatch a `change` event so app.js's listener
        // also runs (auto-scrolls active tab into view).
        const TAB_LABELS = ['General', 'LLM', 'Pipeline', 'Speech', 'Prompts', 'Triggers'];
        for (let i = 0; i < TAB_LABELS.length; i++) {
            const label = TAB_LABELS[i];
            const radio = modal.locator(`input[name="settings_tabs"][aria-label="${label}"]`);
            // Scroll the radio's tab pill into view inside the overflow strip
            // before checking, then mark checked and dispatch change so the
            // CSS :checked rule swaps the visible tabpanel content.
            await radio.evaluate((el) => {
                el.scrollIntoView({ inline: 'center', block: 'nearest' });
                el.checked = true;
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }).catch(() => {});
            // Brief wait for any layout/scroll, then capture.
            await page.waitForTimeout(400);
            await shot(page, `05b_settings_tab_${i + 1}`);
        }
        // Close drawer — uncheck the DaisyUI drawer toggle checkbox.
        // Escape does NOT dismiss a DaisyUI drawer; the .drawer-overlay label
        // would otherwise intercept all subsequent clicks.
        await page.locator('#settings-drawer-toggle').uncheck({ force: true }).catch(() => {});
        await page.waitForTimeout(300);
        // Belt-and-braces: if still open, click the overlay label.
        const toggle = page.locator('#settings-drawer-toggle');
        if (await toggle.isChecked().catch(() => false)) {
            await page.locator('label.drawer-overlay[for="settings-drawer-toggle"]').click({ force: true }).catch(() => {});
            await page.waitForTimeout(300);
        }
        await expect(page.locator('label.drawer-overlay[for="settings-drawer-toggle"]')).toBeHidden({ timeout: 3000 }).catch(() => {});
    });

    test('06 chat panel', async () => {
        await page.locator('button[data-subj-mode="chat"]').first().click();
        await page.waitForTimeout(500);
        await shot(page, '06a_chat_empty');
        // Try to find a chat input — fall back to whatever exists.
        const chatInput = page.locator('#subjects-chat-pane textarea, #subjects-chat-pane input[type="text"]').first();
        if (await chatInput.count()) {
            await chatInput.fill('hello there').catch(() => {});
            await page.waitForTimeout(200);
            await shot(page, '06b_chat_typed');
        } else {
            issues.push('chat input not located');
        }
    });

    test('07 story log', async () => {
        await page.locator('button[data-subj-mode="endless"]').first().click();
        await page.waitForTimeout(400);
        await shot(page, '07_story_endless');
    });

    test('08 slop outputs section', async () => {
        // Scroll the outputs section into view.
        await page.locator('#output-section').scrollIntoViewIfNeeded();
        await page.waitForTimeout(300);
        await shot(page, '08_outputs');
    });

    test.afterAll(async () => {
        // Dump issues alongside screenshots.
        const fs = require('fs');
        fs.writeFileSync(`${OUT}/_issues.json`, JSON.stringify(issues, null, 2));
        await page.close();
    });
});
