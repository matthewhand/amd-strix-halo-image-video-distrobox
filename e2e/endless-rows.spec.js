// Endless mode add/remove/swap row contracts. Every test mocks
// /subjects/suggest with deterministic chip text per prompt_id so we
// never wait for the live LLM (~3 min per call) AND assertions don't
// depend on the LLM's mood. Each chip's text encodes the prompt_id
// it came from so we can verify per-row prompt routing.

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const VIEWPORT = { width: 1440, height: 900 };

test.use({ viewport: VIEWPORT });

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

async function bootstrap(page) {
    // Stub /subjects/suggest BEFORE any page load. Returns 6 chips
    // tagged with the prompt_id so we can assert which prompt produced
    // each row's chips.
    await page.route('**/subjects/suggest**', (route) => {
        const url = new URL(route.request().url());
        const promptId = url.searchParams.get('prompt_id') || 'default';
        const opener = url.searchParams.get('opener');
        // Server response shape (post-23ec1b1) is a per-mode dict, not
        // a flat array. _fetchSuggestBatch picks dict.story for endless
        // and dict.simple for simple/raw. Opener path is handled by
        // _startEndlessStory which still reads d.suggestions[0]; we keep
        // a flat list for that case (server returns the dict either way
        // but _startEndlessStory only looks at the first entry).
        if (opener === '1') {
            return route.fulfill({
                status: 200, contentType: 'application/json',
                body: JSON.stringify({ suggestions: [`opener-from-${promptId}`] }),
            });
        }
        const arr = Array.from({ length: 6 }, (_, i) => `chip-${promptId}-${i + 1}`);
        return route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: arr, simple: arr, chat: arr } }),
        });
    });
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            // Generous upper-pane height so the full Prompt card body
            // fits without splitter-cropping the chip stack.
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
        } catch (_) { }
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const opacity = main ? parseFloat(main.style.opacity || '1') : 1;
        return !splash && opacity >= 1;
    }, null, { timeout: 5000 });
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    await page.waitForTimeout(200);
}

async function startStory(page, seed = 'A lighthouse keeper meets a sea creature.') {
    await page.fill('#p-core', seed);
    await page.click('#btn-start-stop-inline');
    // Wait for the initial endless render to land.
    await page.waitForSelector('#subject-chips-stack .suggest-marquee-row', { timeout: 5000 });
    await page.waitForTimeout(200);
}

async function rowCount(page) {
    return await page.locator('#subject-chips-stack .suggest-marquee-row').count();
}

// Returns [{rowIdx, promptLabel, firstChipText}, ...] for every row.
async function rowSnapshot(page) {
    return await page.evaluate(() => {
        const rows = document.querySelectorAll('#subject-chips-stack .suggest-marquee-row');
        return Array.from(rows).map((r, i) => {
            const lead = r.querySelector('[data-endless-row-lead]');
            const promptBtn = r.querySelector('[data-row-prompt-btn]');
            const label = promptBtn ? promptBtn.textContent.trim() : null;
            // Marquee duplicates chips for wraparound; first chip is the
            // canonical one. dataset.suggest carries the raw text.
            const firstChip = r.querySelector('.btn[data-suggest]');
            return {
                rowIdx: lead ? Number(lead.getAttribute('data-endless-row-lead')) : -1,
                promptLabel: label,
                firstChip: firstChip ? firstChip.dataset.suggest : null,
            };
        });
    });
}

// ---------------------------------------------------------------------------
// CONTRACT: Start Story → exactly ONE row using current default prompt.
// ---------------------------------------------------------------------------

test('endless-rows: Start Story creates exactly 1 row with default prompt', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    const rows = await rowCount(page);
    expect(rows).toBe(1);
    const snap = await rowSnapshot(page);
    expect(snap[0].rowIdx).toBe(0);
    // Default prompt id is 'yes-and' in fresh localStorage state.
    expect(snap[0].firstChip).toMatch(/^chip-yes-and-/);
});

// ---------------------------------------------------------------------------
// CONTRACT: + button appends ONE row using whatever the dropdown is
// currently set to (not the row that's already there).
// ---------------------------------------------------------------------------

test('endless-rows: + click adds 1 row with current default', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    expect(await rowCount(page)).toBe(1);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(() => {
        return document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 2;
    }, null, { timeout: 5000 });
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(2);
    // Both rows came from the default prompt → same chip-text family.
    expect(snap[0].firstChip).toMatch(/^chip-yes-and-/);
    expect(snap[1].firstChip).toMatch(/^chip-yes-and-/);
    // rowIdx values are sequential 0, 1.
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1]);
});

// ---------------------------------------------------------------------------
// CONTRACT: + click N times adds N rows, all with sequential rowIdx.
// ---------------------------------------------------------------------------

test('endless-rows: 3 sequential + clicks → 4 total rows, indices 0..3', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    for (let i = 0; i < 3; i++) {
        await page.click('#subjects-suggest-add-btn');
        await page.waitForFunction((expected) => {
            return document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === expected;
        }, i + 2, { timeout: 5000 });
    }
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(4);
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1, 2, 3]);
});

// ---------------------------------------------------------------------------
// CONTRACT: clicking the − button on a row REMOVES it, surviving rows
// reindex 0..N-1, persisted prompt array shrinks.
// ---------------------------------------------------------------------------

test('endless-rows: − click removes row + reindexes survivors', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    // Add 3 more so we have 4 total.
    for (let i = 0; i < 3; i++) {
        await page.click('#subjects-suggest-add-btn');
        await page.waitForFunction((n) => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === n, i + 2);
    }
    expect(await rowCount(page)).toBe(4);

    // Remove the row at index 1 (second from top) via its − button.
    await page.click('#subject-chips-stack .suggest-marquee-row:nth-child(2) [data-row-remove]');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 3);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(3);
    // Surviving rows must have indices 0,1,2 (REINDEXED). The bug being
    // pinned: if remove leaves indices as 0,2,3 then subsequent + adds
    // and onclick handlers go to the wrong rowIdx.
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1, 2]);

    // localStorage should reflect the shrunk array.
    const saved = await page.evaluate(() => {
        const raw = localStorage.getItem('slopfinity-endless-row-prompts');
        return raw ? JSON.parse(raw) : null;
    });
    expect(saved).toHaveLength(3);
});

// ---------------------------------------------------------------------------
// CONTRACT: − removes the LAST row cleanly without leaving DOM artifacts.
// ---------------------------------------------------------------------------

test('endless-rows: − click on last row leaves N-1 clean rows', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 2);
    // Click − on the LAST row.
    const remBtns = page.locator('#subject-chips-stack [data-row-remove]');
    await remBtns.nth(1).click();
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 1);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(1);
    expect(snap[0].rowIdx).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: − click on the ONLY row removes it. Stack ends empty.
// (Edge case — user removes everything, then needs a way back to 1
// row via +. Tested separately below.)
// ---------------------------------------------------------------------------

test('endless-rows: − click on only row → 0 rows', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    expect(await rowCount(page)).toBe(1);
    await page.click('#subject-chips-stack [data-row-remove]');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 0);
    expect(await rowCount(page)).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: After removing all rows, + click adds a fresh row at idx=0.
// Regression case — empty rowPrompts array shouldn't break the +
// handler.
// ---------------------------------------------------------------------------

test('endless-rows: + after empty stack adds row at idx=0', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subject-chips-stack [data-row-remove]');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 0);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 1);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(1);
    expect(snap[0].rowIdx).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: Per-row ↻ refetches IN PLACE — row stays at its original
// position in the stack instead of moving to the bottom (the
// insertAtIdx bug we fixed earlier).
// ---------------------------------------------------------------------------

test('endless-rows: ↻ refresh stays at original index', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subjects-suggest-add-btn');
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 3);
    // Refresh row 1 (middle).
    await page.click('#subject-chips-stack .suggest-marquee-row:nth-child(2) [data-row-refresh]');
    // Wait for the row content to swap (any chip is OK; we just want
    // the loading class to clear).
    await page.waitForFunction(() => {
        const row = document.querySelectorAll('#subject-chips-stack .suggest-marquee-row')[1];
        return row && !row.querySelector('.suggest-marquee-mask.row-loading');
    }, null, { timeout: 5000 });
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(3);
    // Indices stay 0,1,2 — the refreshed row didn't move.
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1, 2]);
});

// ---------------------------------------------------------------------------
// CONTRACT: When a story is RUNNING, the cycle does NOT add naked rows.
// All rows have a [data-endless-row-lead] cluster. (regression: cycle
// used to call _appendSuggestBatchRow with no opts → naked rows.)
// ---------------------------------------------------------------------------

test('endless-rows: every row has a lead cluster', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subjects-suggest-add-btn');
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 3);
    const allHaveLead = await page.evaluate(() => {
        const rows = document.querySelectorAll('#subject-chips-stack .suggest-marquee-row');
        return Array.from(rows).every(r => r.querySelector('[data-endless-row-lead]'));
    });
    expect(allHaveLead).toBe(true);
});

// ---------------------------------------------------------------------------
// CONTRACT: After Submit, the story ends — _endlessRunning flips false,
// body.endless-running is removed, and the seed textarea is cleared.
// (Prior versions of this test asserted that the + button re-disables
// and body got an `endless-pill-locked` class. Both behaviours were
// REMOVED in v316/v317 — the user asked for the + and prompt-name to
// never look greyed out. See app.js _refreshSuggestBadge ~line 1414 +
// the "isEndless: allow=true" branch ~line 1472.)
// ---------------------------------------------------------------------------

test('endless-rows: Submit ends story (clears running flag + seed)', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 2);
    // Submit ends the story.
    await page.click('#subjects-story-submit');
    await page.waitForTimeout(400);
    // body.endless-running must be cleared.
    const isRunning = await page.evaluate(() => document.body.classList.contains('endless-running'));
    expect(isRunning).toBe(false);
    // Seed textarea was emptied (per _submitEndlessStory).
    const seed = await page.locator('#p-core').inputValue();
    expect(seed).toBe('');
    // + button stays enabled in endless mode (see contract update).
    const addDisabled = await page.locator('#subjects-suggest-add-btn').evaluate(el => el.disabled);
    expect(addDisabled).toBe(false);
});

// ---------------------------------------------------------------------------
// CONTRACT: rapid double-click on + does NOT add 2 rows when the first
// fetch is still in flight (re-entrancy guard via _addEndlessRowInflight).
// ---------------------------------------------------------------------------

test('endless-rows: rapid double-+ click adds only 1 row (re-entrancy guard)', async ({ page }) => {
    // Slow down /subjects/suggest so the first + click is still pending
    // when the second fires.
    await page.route('**/subjects/suggest**', async (route) => {
        const url = new URL(route.request().url());
        const promptId = url.searchParams.get('prompt_id') || 'default';
        const opener = url.searchParams.get('opener');
        await new Promise(r => setTimeout(r, 800));
        if (opener === '1') {
            return route.fulfill({
                status: 200, contentType: 'application/json',
                body: JSON.stringify({ suggestions: [`opener-from-${promptId}`] }),
            });
        }
        const arr = Array.from({ length: 6 }, (_, i) => `chip-${promptId}-${i + 1}`);
        return route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: arr, simple: arr, chat: arr } }),
        });
    });
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
        } catch (_) { }
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'));
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    await page.fill('#p-core', 'seed');
    await page.click('#btn-start-stop-inline');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 1, null, { timeout: 5000 });

    // Double-click + with no delay between.
    await page.click('#subjects-suggest-add-btn');
    await page.click('#subjects-suggest-add-btn');
    // Wait long enough for both fetches to land.
    await page.waitForTimeout(2000);
    const rows = await rowCount(page);
    // Should be 1 (initial) + 1 (only one + landed) = 2, NOT 3.
    expect(rows).toBe(2);
});

// ---------------------------------------------------------------------------
// Screenshot: 3-row endless state for visual verification.
// ---------------------------------------------------------------------------

test('endless-rows: 3-row endless screenshot', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subjects-suggest-add-btn');
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack .suggest-marquee-row').length === 3);
    await page.waitForTimeout(300);
    await page.screenshot({ path: '/tmp/endless-rows-3.png', fullPage: false });
});
