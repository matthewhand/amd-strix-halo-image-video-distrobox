// Contract tests for the suggestion system. Each test pins a single
// invariant from the spec — when one breaks we get a precise pointer
// to which contract failed instead of a generic visual diff.
//
// Tests use the *real* /subjects/suggest endpoint when reasonable; when
// we need deterministic chip content (junk filter, error rendering)
// we stub fetch in the page context so the assertions don't depend on
// the live LLM's mood.

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const VIEWPORT = { width: 1440, height: 900 };

test.use({ viewport: VIEWPORT });

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function bootstrap(page, layout = 'default') {
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            // Seed Suggestions toggle as VISIBLE — new browsers default to
            // `slopfinity_suggestions_hidden=null` which the app treats as
            // hidden (intentional UX — new users opt in). Without this
            // seed, the +/regen badges get inline style.display='none'
            // applied by _applySuggestionsHiddenState() at DOMContentLoaded,
            // and any test that clicks them times out on "not visible".
            // Same fix as smoke.spec.js (commit c159d69).
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) {}
    });
    await page.goto(`${BASE}/?layout=${layout}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const opacity = main ? parseFloat(main.style.opacity || '1') : 1;
        return !splash && opacity >= 1;
    }, null, { timeout: 12000 });
}

async function setMode(page, mode) {
    await page.click(`.subjects-mode-pill button[data-subj-mode="${mode}"]`);
    await page.waitForTimeout(250);
}

async function stubSuggestionResponse(page, suggestions) {
    // Replace fetch for /subjects/suggest with a deterministic response.
    // Other endpoints fall through to the real network.
    // Server response shape is now a per-mode dict: {story, simple, chat}.
    // _fetchSuggestBatch picks the slot based on subjects mode. We mirror
    // the same chip set into all three slots so tests stay mode-agnostic.
    await page.addInitScript((stubChips) => {
        const realFetch = window.fetch.bind(window);
        window.fetch = async function (input, init) {
            const url = typeof input === 'string' ? input : input.url;
            if (url && url.includes('/subjects/suggest')) {
                const body = JSON.stringify({
                    suggestions: { story: stubChips, simple: stubChips, chat: stubChips },
                });
                return new Response(body, {
                    status: 200, headers: { 'Content-Type': 'application/json' },
                });
            }
            return realFetch(input, init);
        };
    }, suggestions);
}

// Chip stacks are mode-suffixed (#subject-chips-stack-simple /
// -endless / -chat); they share class `.subject-chips-stack`. The
// HTML keeps a stale `.hidden` class on every non-default stack
// (it's only the bare initial state — body.subj-mode-<mode> CSS
// rules override with `display: flex !important`). Filtering on
// `:not(.hidden)` therefore excludes the active mode's stack too —
// use computed style instead so we count rows from the actually-
// visible stack only.
async function countRows(page) {
    return await page.evaluate(() => {
        const stacks = Array.from(document.querySelectorAll('.subject-chips-stack'));
        return stacks
            .filter(s => getComputedStyle(s).display !== 'none')
            .reduce((n, s) => n + s.querySelectorAll('.suggest-marquee-row').length, 0);
    });
}

async function rowsHaveLead(page) {
    return await page.evaluate(() => {
        const stacks = Array.from(document.querySelectorAll('.subject-chips-stack'))
            .filter(s => getComputedStyle(s).display !== 'none');
        const rows = stacks.flatMap(s => Array.from(s.querySelectorAll('.suggest-marquee-row')));
        if (!rows.length) return null; // n/a
        return rows.every(r => r.querySelector('[data-endless-row-lead]'));
    });
}

// ---------------------------------------------------------------------------
// CONTRACT: in endless mode, EVERY rendered row has the lead cluster
// (subject-prompt chip + per-row refresh + remove). Naked rows are the
// "simple-mode leak" symptom we fought through 4+ commits.
// ---------------------------------------------------------------------------

test('endless: every row has [data-endless-row-lead]', async ({ page }) => {
    await stubSuggestionResponse(page, ['lonely lighthouse keeper', 'cyberpunk dragon', 'hermit crab lawyer', 'neon jellyfish', 'clay robot rebellion', 'symbiotic mushroom city']);
    await bootstrap(page);
    // Pre-seed the textarea via JS — entering endless mode hides
    // #subjects-input-row via CSS (body.subj-mode-endless.endless-running)
    // so we can't .fill() it post-switch. Setting the value before mode
    // change avoids the visibility race. The seed is informational only —
    // setMode('endless') already flips _endlessRunning=true and seeds the
    // row-prompts (no separate Start Story button anymore; see
    // _setSubjectsMode in app.js line ~2589).
    await page.evaluate(() => {
        const ta = document.getElementById('p-core');
        if (ta) { ta.value = 'A lighthouse keeper meets a sea creature.'; ta.dispatchEvent(new Event('input', { bubbles: true })); }
    });
    await setMode(page, 'endless');
    // setMode already started the story; trigger row rendering via the
    // public regenSuggestions API (which calls _renderEndlessRows for
    // endless mode). _startEndlessStory used to call regenSuggestions
    // itself but the Start-Story gate was removed.
    await page.evaluate(() => window.regenSuggestions && window.regenSuggestions());
    // Wait for the _renderEndlessRows fetch to land + paint.
    await page.waitForSelector('.subject-chips-stack .suggest-marquee-row', { timeout: 5000 });
    await page.waitForTimeout(400);
    const allHaveLead = await rowsHaveLead(page);
    expect(allHaveLead).toBe(true);
});

// ---------------------------------------------------------------------------
// CONTRACT: + button in endless renders the row scaffold IMMEDIATELY but
// without any visible chip placeholders. The 'two empty chips' bug came
// from passing [' '] which the marquee duplicator turned into [' ', ' '].
// ---------------------------------------------------------------------------

test('endless: + click renders empty mask, no blank-chip placeholders', async ({ page }) => {
    // Stub returns a slow promise so we capture the placeholder state.
    await page.addInitScript(() => {
        const realFetch = window.fetch.bind(window);
        window.fetch = async function (input, init) {
            const url = typeof input === 'string' ? input : input.url;
            if (url && url.includes('/subjects/suggest')) {
                await new Promise(r => setTimeout(r, 800)); // simulate slow LLM
                const arr = ['x', 'y'];
                return new Response(JSON.stringify({
                    suggestions: { story: arr, simple: arr, chat: arr },
                }), {
                    status: 200, headers: { 'Content-Type': 'application/json' },
                });
            }
            return realFetch(input, init);
        };
    });
    await bootstrap(page);
    // Seed before mode switch — see endless mode CSS hides input-row.
    await page.evaluate(() => {
        const ta = document.getElementById('p-core');
        if (ta) { ta.value = 'seed'; ta.dispatchEvent(new Event('input', { bubbles: true })); }
    });
    await setMode(page, 'endless');
    // Kick the initial row fetch (no Start-Story button anymore).
    await page.evaluate(() => window.regenSuggestions && window.regenSuggestions());
    await page.waitForSelector('.subject-chips-stack .suggest-marquee-row', { timeout: 5000 });
    await page.waitForTimeout(200); // let paint settle but stay before the next fetch resolves
    const rowsBefore = await countRows(page);
    await page.click('#subjects-suggest-add-btn');
    // Capture state DURING the slow fetch — should have N+1 rows but
    // the new row's mask should have ZERO chip buttons.
    await page.waitForTimeout(200);
    const rowsAfter = await countRows(page);
    expect(rowsAfter).toBe(rowsBefore + 1);
    const newRowChipCount = await page.evaluate(() => {
        const rows = document.querySelectorAll('.subject-chips-stack .suggest-marquee-row');
        const last = rows[rows.length - 1];
        if (!last) return -1;
        // Chips inside the marquee mask, not the lead cluster.
        return last.querySelectorAll('.suggest-marquee-mask .btn[data-suggest]').length;
    });
    expect(newRowChipCount).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: + button stays ENABLED in endless mode regardless of story state.
// (Was: disabled pre-start, enabled post-start. Product moved in
// _refreshSuggestBadge — see app.js ~line 1472, "isEndless: allow=true".
// The user asked for the + to never look greyed out. _addEndlessRow is
// no-op-safe pre-Start; clicking it just appends to the saved row
// prompts and renders when the story actually starts.)
// ---------------------------------------------------------------------------

test('endless: + always enabled in endless mode (pre + post Start)', async ({ page }) => {
    await stubSuggestionResponse(page, ['a', 'b']);
    await bootstrap(page);
    // setMode('endless') *is* the Start Story trigger now (Story-mode
    // redesign removed the separate Start-Story button — see
    // _setSubjectsMode in app.js ~line 2589). Test asserts the +
    // button stays enabled both at entry and after the user has
    // interacted with the seed/textarea + queue button.
    await setMode(page, 'endless');
    const preDisabled = await page.locator('#subjects-suggest-add-btn').evaluate(el => el.disabled);
    expect(preDisabled).toBe(false);
    // Seed via JS (input-row is CSS-hidden in endless mode) + click
    // Queue Slop (the inline button is now Queue Slop in endless, not
    // Start Story — but the + enable-state should be unchanged).
    await page.evaluate(() => {
        const ta = document.getElementById('p-core');
        if (ta) { ta.value = 'seed'; ta.dispatchEvent(new Event('input', { bubbles: true })); }
    });
    await page.waitForTimeout(300);
    const postDisabled = await page.locator('#subjects-suggest-add-btn').evaluate(el => el.disabled);
    expect(postDisabled).toBe(false);
});

// ---------------------------------------------------------------------------
// CONTRACT: the prefetch system is SIMPLE-MODE ONLY. Typing into #p-core
// while in endless mode should NOT fire /subjects/suggest from the
// prefetch idle timer (the user's complaint: 'starts adding non-endless
// suggestions when I type a seed').
// ---------------------------------------------------------------------------

test('endless: prefetch does not fire when typing seed', async ({ page }) => {
    let suggestCalls = 0;
    await page.route('**/subjects/suggest**', (route) => {
        suggestCalls += 1;
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: ['stub'], simple: ['stub'], chat: ['stub'] } }),
        });
    });
    await bootstrap(page);
    await setMode(page, 'endless');
    await page.waitForTimeout(400); // let any setMode-time fetches settle
    const callsAfterModeSwitch = suggestCalls;
    // Simulate slow per-character typing into the seed textarea. We
    // can't use page.keyboard.type because #subjects-input-row (the
    // parent of #p-core) is CSS-hidden in endless-running mode, so
    // page.locator('#p-core').click() times out on visibility. We
    // dispatch 'input' events directly — that's the same event the
    // prefetch handler listens for, so this still exercises the
    // prefetch-leak path.
    const text = 'A lighthouse keeper';
    for (let i = 1; i <= text.length; i++) {
        await page.evaluate((v) => {
            const ta = document.getElementById('p-core');
            if (!ta) return;
            ta.value = v;
            ta.dispatchEvent(new Event('input', { bubbles: true }));
        }, text.slice(0, i));
        await page.waitForTimeout(40);
    }
    await page.waitForTimeout(9000); // > _PREFETCH_IDLE_TRIGGER_MS (8s)
    const callsAfterTyping = suggestCalls;
    // Allow at most the same number of calls (no prefetch should fire).
    // If the prefetch leaked, this would be callsAfterModeSwitch + 1+.
    expect(callsAfterTyping).toBeLessThanOrEqual(callsAfterModeSwitch);
});

// ---------------------------------------------------------------------------
// CONTRACT: looksLikeJunk filter drops markdown headers, error strings,
// and LLM scaffolding leaks. None of these should ever appear as a chip.
// ---------------------------------------------------------------------------

test('suggest: junk filter drops markdown / errors / scaffolding', async ({ page }) => {
    const junkAndChips = [
        'lonely lighthouse',                         // ✓ keep
        '**Constraint check**: Each idea must be ≤8 words',  // ✗ drop
        '**Generated Concepts:**',                   // ✗ drop
        'Need 6 short visual subject ideas for AI video fleet concepts',  // ✗ drop
        'Plain text only with no formatting elements',  // ✗ drop
        'Error: timed out',                          // ✗ drop
        'HTTP 500 internal server error',            // ✗ drop
        'Here are 6 ideas:',                         // ✗ drop
        '# Heading',                                 // ✗ drop
        '1. **bold numbered item**',                 // ✗ drop
        'cyberpunk dragon',                          // ✓ keep
    ];
    await stubSuggestionResponse(page, junkAndChips);
    await bootstrap(page);
    await setMode(page, 'simple');
    // Trigger fetch via the + bootstrap button.
    await page.click('#subjects-suggest-add-btn');
    await page.waitForSelector('.subject-chips-stack .suggest-marquee-row', { timeout: 5000 });
    await page.waitForTimeout(300);
    const chipTexts = await page.evaluate(() => {
        // Filter to the VISIBLE stack via computed style — class
        // `.hidden` is on every non-default stack as stale baseline
        // markup but CSS overrides it for the active mode.
        const stacks = Array.from(document.querySelectorAll('.subject-chips-stack'))
            .filter(s => getComputedStyle(s).display !== 'none');
        const buttons = stacks.flatMap(s => Array.from(s.querySelectorAll('.btn[data-suggest]')));
        // Marquee duplicates chips for the wraparound — dedupe via Set.
        return Array.from(new Set(buttons.map(b => (b.dataset.suggest || '').trim())));
    });
    expect(chipTexts).toContain('lonely lighthouse');
    expect(chipTexts).toContain('cyberpunk dragon');
    // Junk should be filtered out:
    expect(chipTexts.some(t => t.startsWith('**'))).toBe(false);
    expect(chipTexts.some(t => /^error/i.test(t))).toBe(false);
    expect(chipTexts.some(t => /^http\s*5/i.test(t))).toBe(false);
    expect(chipTexts.some(t => /constraint/i.test(t))).toBe(false);
    expect(chipTexts.some(t => /generated\s+concepts/i.test(t))).toBe(false);
    expect(chipTexts.some(t => /^need\s+\d/i.test(t))).toBe(false);
    expect(chipTexts.some(t => /^plain\s+text/i.test(t))).toBe(false);
    expect(chipTexts.some(t => /^here\s+are/i.test(t))).toBe(false);
    expect(chipTexts.some(t => /^#/.test(t))).toBe(false);
});

// ---------------------------------------------------------------------------
// CONTRACT: simple-mode swap → endless → simple does NOT carry simple's
// chips into endless or vice versa. Mode swap must clear / repaint.
// ---------------------------------------------------------------------------

test('mode swap: chips do not leak across modes', async ({ page }) => {
    await stubSuggestionResponse(page, ['simple-a', 'simple-b', 'simple-c']);
    await bootstrap(page);
    await setMode(page, 'simple');
    await page.click('#subjects-suggest-add-btn');
    await page.waitForSelector('.subject-chips-stack .suggest-marquee-row', { timeout: 5000 });
    const simpleRowCount = await countRows(page);
    expect(simpleRowCount).toBeGreaterThan(0);
    // Swap to endless — the endless stack starts empty until
    // _renderEndlessRows / regenSuggestions is fired. CSS hides the
    // simple stack via display:none, so it can't leak into the
    // endless view. Verify the VISIBLE (endless) stack has zero rows.
    await setMode(page, 'endless');
    await page.waitForTimeout(300);
    const endlessRowCount = await countRows(page);
    expect(endlessRowCount).toBe(0);
    // The endless stack has an idle-state placeholder span
    // (#subject-chips-empty-endless) with the "click ↻ Regenerate
    // for suggestions" hint baked into the DOM. The real contract is
    // that NO simple-mode chips leaked in — i.e. no marquee rows AND
    // no buttons with data-suggest in the visible stack.
    const leakedChips = await page.evaluate(() => {
        const stack = document.getElementById('subject-chips-stack-endless');
        if (!stack) return -1;
        return stack.querySelectorAll('.btn[data-suggest]').length;
    });
    expect(leakedChips).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: endless-cycle does NOT fire when no story is running. The
// localStorage 'slopfinity-endless-story' flag from a prior session must
// NOT resurrect the cycle on page load. (regression test)
// ---------------------------------------------------------------------------

test('endless: stale localStorage flag does not auto-add rows', async ({ page }) => {
    await stubSuggestionResponse(page, ['leak-a', 'leak-b']);
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            // Simulate a prior session where the user started a story.
            localStorage.setItem('slopfinity-endless-story', '1');
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
        } catch (_) {}
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
    await setMode(page, 'endless');
    // Wait past the cycle interval — even though the toggle is checked
    // (from localStorage), _endlessRunning is false, so the cycle's
    // hard-gate should bail and no rows should be appended.
    await page.waitForTimeout(13_000);
    const rows = await countRows(page);
    expect(rows).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: simple cache hydration on second page load — no LLM call,
// chips appear immediately from localStorage.
// ---------------------------------------------------------------------------

test('simple: cached chips hydrate without LLM call on reload', async ({ page }) => {
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            // Pre-seed the simple-mode chip cache.
            localStorage.setItem('slopfinity_suggestions_v1',
                JSON.stringify(['cached-1', 'cached-2', 'cached-3']));
            // Seed Suggestions visible — without this the chip stack is
            // inline-style display:none-d by _applySuggestionsHiddenState
            // and no hydration happens (or it happens but the stack is
            // never made visible). Same fix as smoke.spec.js c159d69.
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) {}
    });
    let suggestCalls = 0;
    await page.route('**/subjects/suggest**', (route) => {
        suggestCalls += 1;
        return route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: ['live-llm-result'], simple: ['live-llm-result'], chat: ['live-llm-result'] } }),
        });
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 12000 });
    await page.waitForTimeout(800); // give the cache hydrate a moment
    // Cached chips should appear without any /subjects/suggest call.
    // Stacks are mode-suffixed (#subject-chips-stack-simple). The
    // initial markup leaves `.hidden` on every non-default stack as
    // stale baseline; body.subj-mode-<mode> CSS overrides it for the
    // active mode, so filter on computed display instead of class.
    const chipTexts = await page.evaluate(() => {
        const stacks = Array.from(document.querySelectorAll('.subject-chips-stack'))
            .filter(s => getComputedStyle(s).display !== 'none');
        const buttons = stacks.flatMap(s => Array.from(s.querySelectorAll('.btn[data-suggest]')));
        return Array.from(new Set(buttons.map(b => (b.dataset.suggest || '').trim())));
    });
    expect(chipTexts).toContain('cached-1');
    expect(suggestCalls).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: endless mode — body.endless-pill-locked class was REMOVED in
// the v316/v317 cleanup (see app.js _refreshSuggestBadge line ~1414:
// "the body.endless-pill-locked dimmer was REMOVED — earlier iterations
// dimmed the whole row pre-Start-Story, but the user explicitly asked
// for the prompt-name and + button to NEVER look greyed out."). The
// suggestions pill remains interactive at every stage; the + button
// gating now keys ONLY on storyRunning visually via repaint, but
// `disabled` stays false in endless. This test now asserts the absence
// of the lock class across all 3 phases (pre/running/post-Submit).
// ---------------------------------------------------------------------------

test('endless: pill is never locked (class removed in v316/v317)', async ({ page }) => {
    await stubSuggestionResponse(page, ['a', 'b']);
    await bootstrap(page);
    // The Start-Story button + lock class were both removed in v316/v317.
    // Entering endless mode IS the start; submit/reset end the story but
    // never reinstate the dimmer. Verify across all reachable phases:
    //   (1) just entered endless mode (story running)
    //   (2) after Submit (ends the auto-cycle)
    await setMode(page, 'endless');
    let locked = await page.evaluate(() => document.body.classList.contains('endless-pill-locked'));
    expect(locked).toBe(false);
    // Pointer-events should NOT be blocked anywhere on the row.
    const pillPointerEvents = await page.evaluate(() => {
        const el = document.getElementById('subjects-need-ideas-row');
        return el ? getComputedStyle(el).pointerEvents : '';
    });
    expect(pillPointerEvents).not.toBe('none');

    // Submit ends the story — still no lock.
    await page.click('#subjects-story-submit');
    await page.waitForTimeout(300);
    locked = await page.evaluate(() => document.body.classList.contains('endless-pill-locked'));
    expect(locked).toBe(false);
});

// ---------------------------------------------------------------------------
// CONTRACT: in raw mode, the suggestion cluster + chip stack are hidden.
// No fetch should ever fire from a raw-mode interaction.
// ---------------------------------------------------------------------------

test('raw: no chip stack, no suggestion fetches', async ({ page }) => {
    let suggestCalls = 0;
    await page.route('**/subjects/suggest**', (route) => {
        suggestCalls += 1;
        return route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: ['stub'], simple: ['stub'], chat: ['stub'] } }),
        });
    });
    await bootstrap(page);
    await setMode(page, 'raw');
    // Click into the per-stage panel + type — none of this should
    // trigger a suggestion fetch.
    await page.locator('#p-image').click();
    await page.keyboard.type('lighting test prompt');
    await page.waitForTimeout(2000);
    // Suggestions cluster should not be visible.
    const clusterVisible = await page.evaluate(() => {
        const el = document.getElementById('subjects-need-ideas-row');
        if (!el) return false;
        return getComputedStyle(el).display !== 'none';
    });
    expect(clusterVisible).toBe(false);
    expect(suggestCalls).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: chat replies are NOT in #subject-chips-stack — they live
// in #subjects-chat-replies inside the chat pane. No marquee chips.
// ---------------------------------------------------------------------------

test('chat: replies render in #subjects-chat-replies, not chip stack', async ({ page }) => {
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            // Pre-seed an assistant turn so chat replies can fire.
            localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify([
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: 'hello! what would you like to make?' },
            ]));
        } catch (_) {}
    });
    await stubSuggestionResponse(page, ['try queueing a dragon', 'list recent outputs', 'cancel job 4', 'pause the queue']);
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
    await setMode(page, 'chat');
    await page.waitForTimeout(800);
    // Replies should be in #subjects-chat-replies, NOT in chip stack.
    const stackRows = await countRows(page);
    expect(stackRows).toBe(0);
    const replyCount = await page.locator('#subjects-chat-replies button').count();
    expect(replyCount).toBeGreaterThan(0);
});
