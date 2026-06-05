// Endless mode add/remove/swap row contracts. Every test mocks
// /subjects/suggest with deterministic chip text per prompt_id so we
// never wait for the live LLM (~3 min per call) AND assertions don't
// depend on the LLM's mood. Each chip's text encodes the prompt_id
// it came from so we can verify per-row prompt routing.
//
// PRODUCT MODEL (v316/v317+): endless mode no longer has a separate
// "Start Story" gesture. Clicking the endless mode pill IS the start —
// `_setSubjectsMode` flips `_endlessRunning` true, adds
// body.endless-running, and seeds the row-prompt array. The shared
// #p-core seed textarea is HIDDEN in endless (CSS rule
// `body.subj-mode-endless #subjects-input-row { display: none
// !important; }` — the story-pane owns the per-beat inputs), so the seed
// has to be written BEFORE switching modes (or via direct JS). Tests
// drive row creation through the "+" badge button
// (#subjects-suggest-add-btn) and the per-row − / ↻ controls.
//
// The chip-stack DOM id is mode-suffixed:
//   - `#subject-chips-stack-simple`  for simple/raw/chat
//   - `#subject-chips-stack-endless` for endless
// so the old shared `#subject-chips-stack` selector no longer resolves;
// both carry the .subject-chips-stack class.
//
// SYNCED BASELINE: the per-row indices (data-endless-row-lead, the −/↻/
// picker onclick args) are positions into the persisted row-prompt array
// `slopfinity-endless-row-prompts`. _removeEndlessRow / _regenEndlessRow
// look survivors up by that index, so the array and the rendered rows must
// stay 1:1.
//
// Submit button removed entirely (per template — no
// `#subjects-story-submit` element). The big Queue Slop button is
// `#btn-start-stop-inline`; clicking it routes to `_subjectsAction`
// which calls `toggleInfinity` (queues clips, does NOT end the story).
// _submitEndlessStory still exists as a JS function but has no UI
// affordance.

// Backend-gated: needs a live LLM (see e2e/_fixtures.js). Skipped in CI.
const { test, expect } = require('./_fixtures');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const VIEWPORT = { width: 1440, height: 900 };
const STACK = '#subject-chips-stack-endless';

test.use({ viewport: VIEWPORT });

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

// Boot endless mode with `seedRows` saved beat-prompts rendered as rows.
// Default: one row. Pass 0 for an empty stack.
async function bootstrap(page, seedRows = 1) {
    // Stub /subjects/suggest BEFORE any page load. Returns 6 chips
    // tagged with the prompt_id so we can assert which prompt produced
    // each row's chips.
    await page.route('**/subjects/suggest**', (route) => {
        const url = new URL(route.request().url());
        const promptId = url.searchParams.get('prompt_id') || 'default';
        const opener = url.searchParams.get('opener');
        // Server response shape is a per-mode dict. _fetchSuggestBatch picks
        // dict.story for endless and dict.simple for simple/raw.
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
    await page.addInitScript((n) => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            // Seed exactly `n` beat-prompts (all the default 'yes-and') so
            // the rendered rows and the saved array start 1:1.
            localStorage.setItem('slopfinity-endless-row-prompts',
                JSON.stringify(Array.from({ length: n }, () => 'yes-and')));
            // Force Suggestions toggle ON so #subjects-suggest-add-btn is
            // un-hidden in endless mode (the badge refresh keys on the
            // toggle for the simple-mode pre-first-batch + bootstrap, and
            // an OFF toggle drops other badge elements that share its
            // join cluster offscreen).
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) { }
    }, seedRows);
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
    }, null, { timeout: 12000 });
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    // Entering endless auto-starts the story (body.endless-running).
    await page.waitForFunction(() => document.body.classList.contains('endless-running'), null, { timeout: 4000 });
    // Let the dashboard settle (first WS tick / any one-shot reload) before
    // driving the render — otherwise the evaluate can race a navigation and
    // throw "Execution context was destroyed". Retry the render once.
    await page.waitForLoadState('networkidle').catch(() => {});
    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            // Paint one chip row per saved beat-prompt (the in-product
            // "story running" state) so prompt-array indices and DOM rows
            // stay aligned.
            await page.evaluate(() => window._renderEndlessRows(6));
            await page.waitForFunction(
                (n) => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length === n,
                seedRows,
                { timeout: 8000 },
            );
            return;
        } catch (e) {
            if (attempt === 1) throw e;
            await page.waitForTimeout(500);
        }
    }
}

// Add one row via the "+" badge and wait for it to render.
async function addRow(page) {
    const before = await rowCount(page);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(
        (n) => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length === n,
        before + 1,
        { timeout: 5000 },
    );
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
// CONTRACT: the seeded baseline renders exactly ONE row from the current
// default prompt; chips route from that prompt_id. (Previously gated on a
// "Start Story" button click; now the endless pill IS the start, and the
// synced render paints one row per saved beat-prompt.)
// ---------------------------------------------------------------------------

test('endless-rows: baseline renders one row with default prompt', async ({ page }) => {
    await bootstrap(page, 1);
    expect(await rowCount(page)).toBe(1);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(1);
    expect(snap[0].rowIdx).toBe(0);
    // Default prompt id is 'yes-and' → chips routed from that prompt.
    expect(snap[0].firstChip).toMatch(/^chip-yes-and-/);
});

// ---------------------------------------------------------------------------
// CONTRACT: each + appends ONE row using the current default prompt.
// ---------------------------------------------------------------------------

test('endless-rows: + click adds 1 row with current default', async ({ page }) => {
    await bootstrap(page, 1);
    expect(await rowCount(page)).toBe(1);
    await addRow(page);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(2);
    expect(snap[0].firstChip).toMatch(/^chip-yes-and-/);
    expect(snap[1].firstChip).toMatch(/^chip-yes-and-/);
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1]);
});

// ---------------------------------------------------------------------------
// CONTRACT: N + clicks add N rows with sequential indices.
// ---------------------------------------------------------------------------

test('endless-rows: 3 sequential + clicks → 4 total rows, indices 0..3', async ({ page }) => {
    await bootstrap(page, 1);
    for (let i = 0; i < 3; i++) await addRow(page);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(4);
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1, 2, 3]);
});

// ---------------------------------------------------------------------------
// CONTRACT: clicking the − button on a row REMOVES it, surviving rows
// reindex 0..N-1, persisted prompt array shrinks.
// ---------------------------------------------------------------------------

test('endless-rows: − click removes row + reindexes survivors', async ({ page }) => {
    // Seed 4 rows directly so the array and DOM start aligned at 4.
    await bootstrap(page, 4);
    expect(await rowCount(page)).toBe(4);

    const promptsBefore = await page.evaluate(() => {
        const raw = localStorage.getItem('slopfinity-endless-row-prompts');
        return raw ? JSON.parse(raw) : null;
    });

    // Remove the row at index 1 (second from top) via its − button.
    await page.click(`${STACK} .suggest-marquee-row:nth-child(2) [data-row-remove]`);
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length === 3, null, { timeout: 5000 });
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(3);
    // Surviving rows must have contiguous indices 0,1,2 (REINDEXED). The
    // bug being pinned: if remove leaves a gap then subsequent + adds and
    // onclick handlers go to the wrong rowIdx.
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1, 2]);

    // localStorage prompt array shrank by exactly one entry.
    const promptsAfter = await page.evaluate(() => {
        const raw = localStorage.getItem('slopfinity-endless-row-prompts');
        return raw ? JSON.parse(raw) : null;
    });
    expect(promptsAfter.length).toBe(promptsBefore.length - 1);
});

// ---------------------------------------------------------------------------
// CONTRACT: − removes the LAST row cleanly without leaving DOM artifacts.
// ---------------------------------------------------------------------------

test('endless-rows: − click on last row leaves N-1 clean rows', async ({ page }) => {
    await bootstrap(page, 2);
    expect(await rowCount(page)).toBe(2);
    // Click − on the LAST row.
    const remBtns = page.locator(`${STACK} [data-row-remove]`);
    await remBtns.nth(1).click();
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length === 1, null, { timeout: 5000 });
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(1);
    expect(snap[0].rowIdx).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: − click on the ONLY row removes it. Stack ends empty.
// ---------------------------------------------------------------------------

test('endless-rows: − click on only row → 0 rows', async ({ page }) => {
    await bootstrap(page, 1);
    expect(await rowCount(page)).toBe(1);
    await page.click(`${STACK} [data-row-remove]`);
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length === 0, null, { timeout: 5000 });
    expect(await rowCount(page)).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: After removing all rows, + click adds a fresh row at idx=0.
// ---------------------------------------------------------------------------

test('endless-rows: + after empty stack adds row at idx=0', async ({ page }) => {
    await bootstrap(page, 1);
    await page.click(`${STACK} [data-row-remove]`);
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length === 0, null, { timeout: 5000 });
    await addRow(page);
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(1);
    expect(snap[0].rowIdx).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: Per-row ↻ refetches IN PLACE — row stays at its original
// position in the stack instead of moving to the bottom.
// ---------------------------------------------------------------------------

test('endless-rows: ↻ refresh stays at original index', async ({ page }) => {
    await bootstrap(page, 3);
    expect(await rowCount(page)).toBe(3);
    // Refresh row 1 (middle).
    await page.click(`${STACK} .suggest-marquee-row:nth-child(2) [data-row-refresh]`);
    // Wait for the row content to swap (loading class clears).
    await page.waitForFunction(() => {
        const row = document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row')[1];
        return row && !row.querySelector('.suggest-marquee-mask.row-loading');
    }, STACK, { timeout: 8000 });
    const snap = await rowSnapshot(page);
    expect(snap).toHaveLength(3);
    // Indices stay 0,1,2 — the refreshed row didn't move to the bottom.
    expect(snap.map(r => r.rowIdx)).toEqual([0, 1, 2]);
});

// ---------------------------------------------------------------------------
// CONTRACT: every rendered row has a lead cluster ([data-endless-row-lead]
// = the dropdown + refresh + minus chip group). (regression: cycle used
// to call _appendSuggestBatchRow with no opts → naked rows.)
// ---------------------------------------------------------------------------

test('endless-rows: every row has a lead cluster', async ({ page }) => {
    await bootstrap(page, 3);
    const allHaveLead = await page.evaluate(() => {
        const rows = document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row');
        return Array.from(rows).length > 0 && Array.from(rows).every(r => r.querySelector('[data-endless-row-lead]'));
    });
    expect(allHaveLead).toBe(true);
});

// ---------------------------------------------------------------------------
// CONTRACT: endless mode keeps body.endless-running for the whole session
// and the + button stays enabled (the user asked for the + to never look
// greyed out — see app.js _refreshSuggestBadge "isEndless: allow=true").
// ---------------------------------------------------------------------------

test('endless-rows: + stays enabled and story stays running', async ({ page }) => {
    await bootstrap(page, 2);
    // body.endless-running persists across the session.
    const isRunning = await page.evaluate(() => document.body.classList.contains('endless-running'));
    expect(isRunning).toBe(true);
    // + button stays enabled in endless mode.
    const addDisabled = await page.locator('#subjects-suggest-add-btn').evaluate(el => el.disabled);
    expect(addDisabled).toBe(false);
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
        await new Promise(r => setTimeout(r, 800));
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
            // Start from one synced row so the array/DOM are aligned.
            localStorage.setItem('slopfinity-endless-row-prompts', JSON.stringify(['yes-and']));
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) { }
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 12000 });
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    await page.waitForFunction(() => document.body.classList.contains('endless-running'), null, { timeout: 4000 });
    await page.waitForLoadState('networkidle').catch(() => {});
    await page.evaluate(() => window._renderEndlessRows(6));
    await page.waitForFunction(() => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length === 1, null, { timeout: 8000 });

    // Double-click + with no delay between.
    await page.click('#subjects-suggest-add-btn');
    await page.click('#subjects-suggest-add-btn');
    await page.waitForTimeout(2500);
    const rows = await rowCount(page);
    // Should be 1 (baseline) + 1 (only one + landed) = 2, NOT 3.
    expect(rows).toBe(2);
});

// ---------------------------------------------------------------------------
// Screenshot: 3-row endless state for visual verification.
// ---------------------------------------------------------------------------

test('endless-rows: 3-row endless screenshot', async ({ page }) => {
    await bootstrap(page, 3);
    await page.waitForTimeout(300);
    await page.screenshot({ path: '/tmp/endless-rows-3.png', fullPage: false });
});
