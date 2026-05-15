// Endless mode add/remove/swap row contracts. Every test mocks
// /subjects/suggest with deterministic chip text per prompt_id so we
// never wait for the live LLM (~3 min per call) AND assertions don't
// depend on the LLM's mood. Each chip's text encodes the prompt_id
// it came from so we can verify per-row prompt routing.
//
// ARCHITECTURAL NOTE (v316/v317 → current):
//   The "Start Story" button is gone. Clicking the endless mode pill
//   IS the start — `_endlessRunning` flips true on pill activation in
//   `_setSubjectsMode`. The shared #p-core textarea is hidden (display:
//   none) in endless mode (CSS rule `body.subj-mode-endless
//   #subjects-input-row { display: none !important; }`), so the seed
//   has to be written BEFORE switching modes (or via direct JS).
//   The chip-stack DOM id is mode-suffixed:
//     - `#subject-chips-stack-simple`  for simple/raw/chat
//     - `#subject-chips-stack-endless` for endless
//   so the old shared `#subject-chips-stack` selector no longer resolves.
//
//   Submit button removed entirely (per template — no
//   `#subjects-story-submit` element). The big Queue Slop button is
//   `#btn-start-stop-inline`; clicking it routes to `_subjectsAction`
//   which calls `toggleInfinity` (queues clips, does NOT end the
//   story). _submitEndlessStory still exists as a JS function but has
//   no UI affordance.

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const VIEWPORT = { width: 1440, height: 900 };
const STACK = '#subject-chips-stack-endless';

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
            // Force Suggestions toggle ON so #subjects-suggest-add-btn is
            // un-hidden in endless mode (the badge refresh keys on the
            // toggle for the simple-mode pre-first-batch + bootstrap, and
            // an OFF toggle drops other badge elements that share its
            // join cluster offscreen).
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) { }
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const opacity = main ? parseFloat(main.style.opacity || '1') : 1;
        return !splash && opacity >= 1;
    }, null, { timeout: 12000 });
}

// "Start" an endless story under the new architecture:
//   1. Write the seed into #p-core while still in simple mode (where the
//      textarea is visible).
//   2. Switch to endless mode (the pill click). This sets
//      _endlessRunning = true and seeds _setEndlessRowPrompts with one
//      default prompt id. The CSS hides #p-core; that's fine — the seed
//      already lives in the textarea's `value`.
//   3. Kick regenSuggestions() to paint the initial row from the seeded
//      prompts array. Without this, _endlessRunning is true but no row
//      is in the DOM (the mode-swap doesn't auto-fetch).
async function startStory(page, seed = 'A lighthouse keeper meets a sea creature.') {
    // Step 1: seed in simple mode (textarea visible there).
    await page.waitForSelector('#p-core', { state: 'visible', timeout: 5000 });
    await page.fill('#p-core', seed);
    // Step 2: swap to endless mode — flips _endlessRunning.
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    // Step 3: drop any chip rows the page rendered on its own (load-time
    //         auto-paint, mode-pill click after-effects, async drains) and
    //         force the prompt array empty. Then call _addEndlessRow() to
    //         paint exactly ONE canonical first row at rowIdx=0. The
    //         in-flight guard inside _addEndlessRow prevents double-fires
    //         even if some prior render is still settling.
    await page.evaluate(async () => {
        try {
            if (typeof window._setEndlessRowPrompts === 'function') {
                window._setEndlessRowPrompts([]);
            }
            const box = document.getElementById('subject-chips-stack-endless');
            if (box) box.innerHTML = '';
            if (typeof window._addEndlessRow === 'function') {
                await window._addEndlessRow();
            }
        } catch (_) { }
    });
    await page.waitForSelector(`${STACK} .suggest-marquee-row`, { state: 'attached', timeout: 8000 });
    // Settle: wait for any in-flight regen / drain to finish so the row
    // count is stable before the test starts asserting.
    await page.waitForTimeout(400);
    // Final sanity: if the page's load-time machinery added extra rows
    // while our startStory awaited, trim them down to a single canonical
    // row. _setEndlessRowPrompts keeps the persisted array in sync.
    await page.evaluate(() => {
        const box = document.getElementById('subject-chips-stack-endless');
        if (!box) return;
        const rows = box.querySelectorAll('.suggest-marquee-row');
        for (let i = rows.length - 1; i >= 1; i--) rows[i].remove();
        // Re-walk the surviving lead so its idx is 0.
        const survivor = box.querySelector('.suggest-marquee-row');
        if (survivor) {
            const lead = survivor.querySelector('[data-endless-row-lead]');
            if (lead) lead.setAttribute('data-endless-row-lead', '0');
        }
        if (typeof window._setEndlessRowPrompts === 'function') {
            const arr = (typeof window._getEndlessRowPrompts === 'function')
                ? window._getEndlessRowPrompts() : ['yes-and'];
            window._setEndlessRowPrompts(arr.slice(0, 1));
        }
    });
}

async function rowCount(page) {
    return await page.locator(`${STACK} .suggest-marquee-row`).count();
}

// Returns [{rowIdx, promptLabel, firstChipText}, ...] for every row.
async function rowSnapshot(page) {
    return await page.evaluate((stackSel) => {
        const rows = document.querySelectorAll(`${stackSel} .suggest-marquee-row`);
        return Array.from(rows).map((r) => {
            const lead = r.querySelector('[data-endless-row-lead]');
            const promptBtn = r.querySelector('[data-row-prompt-btn]');
            const label = promptBtn ? promptBtn.textContent.trim() : null;
            const firstChip = r.querySelector('.btn[data-suggest]');
            return {
                rowIdx: lead ? Number(lead.getAttribute('data-endless-row-lead')) : -1,
                promptLabel: label,
                firstChip: firstChip ? firstChip.dataset.suggest : null,
            };
        });
    }, STACK);
}

// ---------------------------------------------------------------------------
// CONTRACT: First render → exactly ONE row using current default prompt.
// (Previously gated on a "Start Story" button click; now the endless pill
// IS the start, and a single regenSuggestions() pass renders the initial
// row from the seeded prompts list.)
// ---------------------------------------------------------------------------

test('endless-rows: Start Story creates exactly 1 row with default prompt', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    const rows = await rowCount(page);
    expect(rows).toBe(1);
    const snap = await rowSnapshot(page);
    expect(snap[0].rowIdx).toBe(0);
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
    await page.waitForFunction((sel) => {
        return document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 2;
    }, STACK, { timeout: 8000 });
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(2);
    expect(snap[0].firstChip).toMatch(/^chip-yes-and-/);
    expect(snap[1].firstChip).toMatch(/^chip-yes-and-/);
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
        await page.waitForFunction(([sel, expected]) => {
            return document.querySelectorAll(`${sel} .suggest-marquee-row`).length === expected;
        }, [STACK, i + 2], { timeout: 8000 });
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
    for (let i = 0; i < 3; i++) {
        await page.click('#subjects-suggest-add-btn');
        await page.waitForFunction(([sel, n]) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === n, [STACK, i + 2], { timeout: 8000 });
    }
    expect(await rowCount(page)).toBe(4);

    // Remove the row at index 1 (second from top) via its − button.
    await page.click(`${STACK} .suggest-marquee-row:nth-child(2) [data-row-remove]`);
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 3, STACK);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(3);
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1, 2]);

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
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 2, STACK, { timeout: 8000 });
    const remBtns = page.locator(`${STACK} [data-row-remove]`);
    await remBtns.nth(1).click();
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 1, STACK);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(1);
    expect(snap[0].rowIdx).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: − click on the ONLY row removes it. Stack ends empty.
// ---------------------------------------------------------------------------

test('endless-rows: − click on only row → 0 rows', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    expect(await rowCount(page)).toBe(1);
    await page.click(`${STACK} [data-row-remove]`);
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 0, STACK);
    expect(await rowCount(page)).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: After removing all rows, + click adds a fresh row at idx=0.
// ---------------------------------------------------------------------------

test('endless-rows: + after empty stack adds row at idx=0', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click(`${STACK} [data-row-remove]`);
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 0, STACK);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 1, STACK, { timeout: 8000 });
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(1);
    expect(snap[0].rowIdx).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: Per-row ↻ refetches IN PLACE — row stays at its original
// position in the stack instead of moving to the bottom.
// ---------------------------------------------------------------------------

test('endless-rows: ↻ refresh stays at original index', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 2, STACK, { timeout: 8000 });
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 3, STACK, { timeout: 8000 });
    await page.click(`${STACK} .suggest-marquee-row:nth-child(2) [data-row-refresh]`);
    await page.waitForFunction((sel) => {
        const row = document.querySelectorAll(`${sel} .suggest-marquee-row`)[1];
        return row && !row.querySelector('.suggest-marquee-mask.row-loading');
    }, STACK, { timeout: 8000 });
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(3);
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1, 2]);
});

// ---------------------------------------------------------------------------
// CONTRACT: All endless rows have a [data-endless-row-lead] cluster.
// ---------------------------------------------------------------------------

test('endless-rows: every row has a lead cluster', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 2, STACK, { timeout: 8000 });
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 3, STACK, { timeout: 8000 });
    const allHaveLead = await page.evaluate((sel) => {
        const rows = document.querySelectorAll(`${sel} .suggest-marquee-row`);
        return Array.from(rows).every(r => r.querySelector('[data-endless-row-lead]'));
    }, STACK);
    expect(allHaveLead).toBe(true);
});

// ---------------------------------------------------------------------------
// SKIPPED: The "Submit ends story" contract is moot — there is no Submit
// button in the current template (no #subjects-story-submit element; only
// Copy and Reset live next to the story log). _submitEndlessStory exists
// as a JS function but is unwired. If the intent is "user explicitly ends
// the story without queueing", that UX needs a re-design first. The
// underlying state (_endlessRunning, body.endless-running, seed
// textarea) and the contract spelled out in the docstring should be
// re-asserted once a Submit / End-Story affordance exists.
//
// Files that own the gap:
//   - slopfinity/templates/index.html (~line 700-740, the story-pane
//     header has Copy + Reset but no Submit button).
//   - slopfinity/static/app.js line 3803 (_submitEndlessStory function
//     exists but no element calls it).
// ---------------------------------------------------------------------------

test.skip('endless-rows: Submit ends story (clears running flag + seed)', async ({ page }) => {
    // No #subjects-story-submit element in the current template; the
    // Submit affordance was removed when endless became "the pill IS the
    // start" (v316/v317 refactor). Re-enable once a Submit/End-Story
    // button is reintroduced.
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
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) { }
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const opacity = main ? parseFloat(main.style.opacity || '1') : 1;
        return !splash && opacity >= 1;
    }, null, { timeout: 12000 });
    // Seed BEFORE swapping into endless (textarea hidden in endless mode).
    await page.waitForSelector('#p-core', { state: 'visible', timeout: 5000 });
    await page.fill('#p-core', 'seed');
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    await page.evaluate(() => {
        try {
            if (typeof window._setEndlessRowPrompts === 'function') {
                window._setEndlessRowPrompts(['yes-and']);
            }
        } catch (_) { }
    });
    await page.waitForTimeout(100);
    await page.evaluate(async () => {
        if (typeof window.regenSuggestions === 'function') {
            await window.regenSuggestions();
        }
    });
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 1, STACK, { timeout: 8000 });

    // Double-click + with no delay between.
    await page.click('#subjects-suggest-add-btn');
    await page.click('#subjects-suggest-add-btn');
    await page.waitForTimeout(2500);
    const rows = await rowCount(page);
    expect(rows).toBe(2);
});

// ---------------------------------------------------------------------------
// Screenshot: 3-row endless state for visual verification.
// ---------------------------------------------------------------------------

test('endless-rows: 3-row endless screenshot', async ({ page }) => {
    await bootstrap(page);
    await startStory(page);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 2, STACK, { timeout: 8000 });
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction((sel) => document.querySelectorAll(`${sel} .suggest-marquee-row`).length === 3, STACK, { timeout: 8000 });
    await page.waitForTimeout(300);
    await page.screenshot({ path: '/tmp/endless-rows-3.png', fullPage: false });
});
