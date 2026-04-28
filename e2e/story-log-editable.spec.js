// Endless story log — editable rows in #subjects-story-log instead of
// the legacy <pre> blob. Each row is a contenteditable .endless-row-text
// + an .endless-row-del × button. Edits and deletes both persist into
// localStorage `slopfinity-endless-story-log` as a JSON array. Legacy
// `\n`-joined string storage migrates to the array shape on first read.
//
// Verifies:
//   1. story pane in endless mode renders editable rows (contenteditable
//      spans), not <pre>
//   2. each row has a × delete button
//   3. editing row text persists into localStorage as a JSON array
//   4. legacy `\n`-joined string in storage migrates to array on first read
//   5. × removes the row + persists the new array

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.use({ viewport: { width: 1440, height: 900 } });

const STORY_KEY = 'slopfinity-endless-story-log';

async function bootEndless(page, opts) {
    const { storySeed, legacyString } = opts || {};
    await page.route('**/subjects/suggest**', (route) => {
        const arr = ['chip-a', 'chip-b', 'chip-c'];
        return route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: arr, simple: arr, chat: arr } }),
        });
    });
    await page.addInitScript((init) => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            if (init.storySeed) {
                localStorage.setItem('slopfinity-endless-story-log',
                    JSON.stringify(init.storySeed));
            } else if (init.legacyString) {
                // Plain legacy `\n`-joined string — _loadEndlessLogLines
                // must parse it without choking.
                localStorage.setItem('slopfinity-endless-story-log', init.legacyString);
            }
        } catch (_) { }
    }, { storySeed, legacyString });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    await page.waitForTimeout(250);
}

test.describe('endless story log — editable rows', () => {
    test('story pane renders contenteditable rows, not <pre>', async ({ page }) => {
        await bootEndless(page, { storySeed: ['First beat.', 'Second beat.', 'Third beat.'] });
        // Ensure the story log paints — _setSubjectsMode hydrates from
        // saved lines when entering endless mode.
        await page.waitForFunction(() => {
            const log = document.getElementById('subjects-story-log');
            return log && log.querySelectorAll('.endless-row').length >= 1;
        }, null, { timeout: 4000 });

        // No <pre> child inside the story log.
        const preCount = await page.locator('#subjects-story-log pre').count();
        expect(preCount).toBe(0);

        // Each row's text span is contenteditable.
        const editableCount = await page.locator('#subjects-story-log .endless-row-text[contenteditable="true"]').count();
        expect(editableCount).toBeGreaterThanOrEqual(3);

        // Each row has a × delete button.
        const delCount = await page.locator('#subjects-story-log .endless-row-del').count();
        expect(delCount).toBeGreaterThanOrEqual(3);
    });

    test('editing a row text persists into localStorage JSON array', async ({ page }) => {
        await bootEndless(page, { storySeed: ['alpha', 'bravo'] });
        await page.waitForSelector('#subjects-story-log .endless-row-text[contenteditable="true"]', { timeout: 4000, state: 'attached' });

        // Edit the first row via direct DOM mutation + dispatching input
        // (simpler than typing into a contenteditable span).
        await page.evaluate(() => {
            const span = document.querySelector('#subjects-story-log .endless-row-text[data-row-idx="0"]');
            if (span) {
                span.textContent = 'alpha-edited';
                span.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });
        await page.waitForTimeout(150);

        const stored = await page.evaluate(() => {
            try {
                const raw = localStorage.getItem('slopfinity-endless-story-log');
                return raw ? JSON.parse(raw) : null;
            } catch (_) { return null; }
        });
        expect(Array.isArray(stored)).toBe(true);
        expect(stored[0]).toBe('alpha-edited');
        expect(stored[1]).toBe('bravo');
    });

    test('legacy `\\n`-joined string in storage migrates to array on first read', async ({ page }) => {
        // Seed a legacy plaintext blob — exactly what older storage looked
        // like before the editable-rows refactor.
        await bootEndless(page, { legacyString: 'one\ntwo\nthree' });
        // The renderer should split it into 3 .endless-row entries.
        await page.waitForFunction(() => {
            const log = document.getElementById('subjects-story-log');
            return log && log.querySelectorAll('.endless-row').length === 3;
        }, null, { timeout: 4000 });

        const rows = await page.locator('#subjects-story-log .endless-row-text').count();
        expect(rows).toBe(3);
        const texts = await page.evaluate(() => Array.from(
            document.querySelectorAll('#subjects-story-log .endless-row-text')
        ).map(s => s.textContent));
        expect(texts).toEqual(['one', 'two', 'three']);
    });

    test('× removes the row and persists the shrunk array', async ({ page }) => {
        await bootEndless(page, { storySeed: ['alpha', 'bravo', 'charlie'] });
        await page.waitForSelector('#subjects-story-log .endless-row', { timeout: 4000, state: 'attached' });
        // Click the × on the second row. Use evaluate so the click works
        // even if the parent pane is visually hidden (we're testing the
        // wiring, not the user-clickability).
        await page.evaluate(() => {
            const btn = document.querySelector('#subjects-story-log .endless-row[data-row-idx="1"] .endless-row-del');
            if (btn) btn.click();
        });
        await page.waitForFunction(() => {
            const log = document.getElementById('subjects-story-log');
            return log && log.querySelectorAll('.endless-row').length === 2;
        }, null, { timeout: 3000 });

        const stored = await page.evaluate(() => {
            const raw = localStorage.getItem('slopfinity-endless-story-log');
            return raw ? JSON.parse(raw) : null;
        });
        expect(Array.isArray(stored)).toBe(true);
        expect(stored.length).toBe(2);
        expect(stored).toEqual(['alpha', 'charlie']);
    });
});
