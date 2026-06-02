// Endless mode RUNNING state. Historically this exercised a separate
// "Start Story" gesture (type a seed in #p-core, click the big button)
// and asserted a story-pane footer rendered BELOW the log. Both have
// changed:
//
//   * v316+: there is no separate Start Story state — ENTERING endless
//     mode auto-starts the story (body.endless-running flips on). The
//     shared #p-core seed textarea is HIDDEN in endless (the story-pane
//     owns the per-beat inputs).
//   * The story-pane controls (Copy / Reset / Auto-Stitch) now live in a
//     HEADER row ABOVE #subjects-story-log, not a footer below it.
//
// This spec asserts the current contract:
//   1. switching to endless flips body.endless-running on (no Start click)
//   2. the + badge button is enabled
//   3. the story pane is visible (no .hidden)
//   4. the story log exists and the header controls sit ABOVE it

// Backend-gated: needs a live LLM (see e2e/_fixtures.js). Skipped in CI.
const { test, expect } = require('./_fixtures');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.use({ viewport: { width: 1440, height: 900 } });

test('endless auto-starts on mode switch — running flag, + enabled, pane visible', async ({ page }) => {
    // Stub /subjects/suggest so any row render is deterministic + fast.
    await page.route('**/subjects/suggest**', (route) => {
        const arr = ['beat-1', 'beat-2', 'beat-3', 'beat-4', 'beat-5', 'beat-6'];
        return route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: arr, simple: arr, chat: arr } }),
        });
    });
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            // Generous upper-pane height so the full Prompt card body
            // fits — without this, the story-pane gets cropped.
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
        } catch (_) {}
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const mainOpacity = main ? parseFloat(main.style.opacity || '1') : 1;
        return !splash && mainOpacity >= 1;
    }, null, { timeout: 5000 });

    // Switch to endless mode — this IS starting the story now.
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');

    // body.endless-running flips on immediately (no Start Story click).
    await page.waitForFunction(() => document.body.classList.contains('endless-running'), null, { timeout: 4000 });
    const isRunning = await page.evaluate(() => document.body.classList.contains('endless-running'));
    expect(isRunning).toBe(true);

    // + button is enabled (the user asked for it to never look greyed out).
    const addBtn = page.locator('#subjects-suggest-add-btn');
    const addDisabled = await addBtn.evaluate(el => el.disabled);
    expect(addDisabled).toBe(false);

    // Story pane is visible (no .hidden class) for the whole endless session.
    const storyPane = page.locator('#subjects-story-pane');
    const storyPaneHidden = await storyPane.evaluate(el => el.classList.contains('hidden'));
    expect(storyPaneHidden).toBe(false);

    // Story-pane structure: the controls header (Copy / Reset) sits BEFORE
    // the story-log in document order (header-above-log layout).
    const order = await page.evaluate(() => {
        const pane = document.getElementById('subjects-story-pane');
        if (!pane) return null;
        const log = pane.querySelector('#subjects-story-log');
        const reset = pane.querySelector('#subjects-story-reset');
        if (!log || !reset) return null;
        // DOCUMENT_POSITION_FOLLOWING (4) means log FOLLOWS reset.
        return (reset.compareDocumentPosition(log) & Node.DOCUMENT_POSITION_FOLLOWING) ? 'header-above-log' : 'log-above-header';
    });
    expect(order).toBe('header-above-log');

    await page.screenshot({ path: '/tmp/endless-running.png', fullPage: false });

    // A "+" click renders an endless suggestion row with a lead cluster
    // (dropdown + refresh + minus chip group).
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(() => {
        return !!document.querySelector('#subject-chips-stack-endless .suggest-marquee-row [data-endless-row-lead]');
    }, null, { timeout: 6000 });
    const hasLead = await page.evaluate(() => {
        return !!document.querySelector('#subject-chips-stack-endless .suggest-marquee-row [data-endless-row-lead]');
    });
    expect(hasLead).toBe(true);
    await page.screenshot({ path: '/tmp/endless-running-with-rows.png', fullPage: false });
});
