// Endless mode RUNNING state — the user's reported "+ button locked"
// and "submit/reset above story" issues only show up after a story has
// started. This spec drives the actual flow:
//   1. Switch to endless mode
//   2. Type a seed in the textarea
//   3. Click Start Story
//   4. Verify + button is enabled (was reading window._endlessRunning,
//      always undefined → button stayed locked)
//   5. Verify story-pane is visible
//   6. Verify story-pane footer (Copy/Submit/Reset/Stitch) is BELOW
//      the story log, not in a header row alongside "Story so far"
//   7. Screenshot for visual verification

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.use({ viewport: { width: 1440, height: 900 } });

test('endless story running — + enabled, story pane visible, buttons below log', async ({ page }) => {
    // Stub /subjects/suggest so the bonus row-rendering check doesn't
    // wait on the live LLM (~minutes per call).
    await page.route('**/subjects/suggest**', (route) => {
        const arr = ['lonely lighthouse', 'cyberpunk dragon', 'hermit crab lawyer'];
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
            // Seed Suggestions visible — _applySuggestionsHiddenState
            // inline-style-hides the +/regen badges on a cold browser
            // (default state is "hidden") which would defeat the +
            // enabled-state assertion. Same fix as smoke spec c159d69.
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) {}
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const mainOpacity = main ? parseFloat(main.style.opacity || '1') : 1;
        return !splash && mainOpacity >= 1;
    }, null, { timeout: 12000 });

    // Switch to endless mode. Note: as of the Story-mode redesign,
    // entering endless mode IS starting the story — there is no
    // separate "Start Story" button anymore. _setSubjectsMode flips
    // `_endlessRunning=true` and adds `body.endless-running`
    // synchronously (see app.js ~line 2596). The big inline button
    // is now the standard "Queue Slop" — it does NOT start endless.
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    await page.waitForTimeout(400);

    const addBtn = page.locator('#subjects-suggest-add-btn');
    await page.screenshot({ path: '/tmp/endless-prestart.png', fullPage: false });

    // Pre-seed via JS — #subjects-input-row is CSS-hidden in
    // body.subj-mode-endless.endless-running, so page.fill('#p-core')
    // times out on visibility. Setting .value + dispatching 'input'
    // is the post-redesign equivalent.
    await page.evaluate(() => {
        const ta = document.getElementById('p-core');
        if (ta) {
            ta.value = 'A lonely lighthouse keeper discovers a stranded sea creature.';
            ta.dispatchEvent(new Event('input', { bubbles: true }));
        }
    });
    await page.waitForTimeout(150);

    // Verify _endlessRunning is true module-side via the body class
    // (set synchronously in _setSubjectsMode for endless).
    const isRunning = await page.evaluate(() => {
        return document.body.classList.contains('endless-running');
    });
    expect(isRunning).toBe(true);

    // + button should be enabled in endless mode (always — see
    // _refreshSuggestBadge "isEndless: allow=true" branch).
    const postStartDisabled = await addBtn.evaluate(el => el.disabled);
    expect(postStartDisabled).toBe(false);

    // Story pane should be visible (no .hidden class).
    const storyPane = page.locator('#subjects-story-pane');
    const storyPaneHidden = await storyPane.evaluate(el => el.classList.contains('hidden'));
    expect(storyPaneHidden).toBe(false);

    // Story-pane structure check: footer (with Submit + Reset + Stitch
    // buttons) should appear AFTER the story-log <pre>, not BEFORE.
    const order = await page.evaluate(() => {
        const pane = document.getElementById('subjects-story-pane');
        if (!pane) return null;
        const log = pane.querySelector('#subjects-story-log');
        const submit = pane.querySelector('#subjects-story-submit');
        if (!log || !submit) return null;
        // Compare DOM positions — Node.DOCUMENT_POSITION_FOLLOWING (4)
        // means submit FOLLOWS log in document order.
        return (log.compareDocumentPosition(submit) & Node.DOCUMENT_POSITION_FOLLOWING) ? 'after' : 'before';
    });
    expect(order).toBe('after');

    await page.screenshot({ path: '/tmp/endless-running.png', fullPage: false });

    // Bonus: verify a suggestion row is rendering with a lead cluster
    // (the [data-endless-row-lead] container — that's the dropdown +
    // refresh + minus chip group). The Start-Story-triggered fetch
    // is gone too, so we kick it via the public regenSuggestions API.
    await page.evaluate(() => window.regenSuggestions && window.regenSuggestions());
    await page.waitForTimeout(1500);
    const hasLead = await page.evaluate(() => {
        return !!document.querySelector('.subject-chips-stack:not(.hidden) .suggest-marquee-row [data-endless-row-lead]');
    });
    if (!hasLead) {
        console.warn('[endless-lead] no [data-endless-row-lead] cluster on any rendered row — endless rows are missing the dropdown');
    }
    await page.screenshot({ path: '/tmp/endless-running-with-rows.png', fullPage: false });
});
