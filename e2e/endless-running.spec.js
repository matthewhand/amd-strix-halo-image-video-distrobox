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

    // Switch to endless mode.
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    await page.waitForTimeout(300);

    // Snapshot pre-start — + button SHOULD be disabled here.
    const addBtn = page.locator('#subjects-suggest-add-btn');
    const preStartDisabled = await addBtn.evaluate(el => el.disabled);
    if (!preStartDisabled) {
        console.warn('[endless-prestart] + button should be disabled before Start Story, but isn\'t');
    }
    await page.screenshot({ path: '/tmp/endless-prestart.png', fullPage: false });

    // Type a seed + click Start Story.
    await page.fill('#p-core', 'A lonely lighthouse keeper discovers a stranded sea creature.');
    await page.waitForTimeout(150);
    await page.click('#btn-start-stop-inline');

    // Give the start handler time to flip _endlessRunning + repaint.
    await page.waitForTimeout(600);

    // Verify _endlessRunning is true module-side.
    const isRunning = await page.evaluate(() => {
        // _endlessRunning is module-scoped; we expose it indirectly via
        // body.endless-running class set in _startEndlessStory.
        return document.body.classList.contains('endless-running');
    });
    if (!isRunning) {
        console.warn('[endless-start] body.endless-running class missing after Start Story click');
    }
    expect(isRunning).toBe(true);

    // + button should now be enabled.
    const postStartDisabled = await addBtn.evaluate(el => el.disabled);
    expect(postStartDisabled).toBe(false);
    if (postStartDisabled) {
        console.warn('[endless-poststart] + button STILL disabled after Start Story — gating bug');
    }

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
    // refresh + minus chip group). Wait for the cycle to inject one;
    // the first row should have promptId+rowIdx → lead cluster present.
    await page.waitForTimeout(2500);
    const hasLead = await page.evaluate(() => {
        return !!document.querySelector('#subject-chips-stack .suggest-marquee-row [data-endless-row-lead]');
    });
    if (!hasLead) {
        console.warn('[endless-lead] no [data-endless-row-lead] cluster on any rendered row — endless rows are missing the dropdown');
    }
    await page.screenshot({ path: '/tmp/endless-running-with-rows.png', fullPage: false });
});
