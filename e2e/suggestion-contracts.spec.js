// Contract tests for the suggestion system. Each test pins a single
// invariant from the spec — when one breaks we get a precise pointer
// to which contract failed instead of a generic visual diff.
//
// Tests use the *real* /subjects/suggest endpoint when reasonable; when
// we need deterministic chip content (junk filter, error rendering)
// we stub fetch in the page context so the assertions don't depend on
// the live LLM's mood.

// Backend-gated: needs a live LLM (see e2e/_fixtures.js). Skipped in CI.
const { test, expect } = require('./_fixtures');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const VIEWPORT = { width: 1440, height: 900 };

test.use({ viewport: VIEWPORT });

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function bootstrap(page, layout = 'default', endlessRowSeed = null) {
    await page.addInitScript((seed) => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            if (seed !== null) {
                localStorage.setItem('slopfinity-endless-row-prompts', JSON.stringify(seed));
            }
            // Seed Suggestions toggle as VISIBLE — new browsers default to
            // `slopfinity_suggestions_hidden=null` which the app treats as
            // hidden (intentional UX — new users opt in). Without this
            // seed, the +/regen badges get inline style.display='none'
            // applied by _applySuggestionsHiddenState() at DOMContentLoaded,
            // and any test that clicks them times out on "not visible".
            // Same fix as smoke.spec.js (commit c159d69).
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) {}
    }, endlessRowSeed);
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
// -endless / -chat); they share class `.subject-chips-stack`. Endless and
// simple now have separate stacks, so pass the mode-suffixed selector when
// you want one specific stack. The default `.subject-chips-stack` class
// matches whichever is active (the HTML keeps a stale `.hidden` class on
// non-default stacks, but body.subj-mode-<mode> CSS overrides it).
async function countRows(page, sel = '.subject-chips-stack') {
    return await page.locator(`${sel} .suggest-marquee-row`).count();
}

async function rowsHaveLead(page, sel = '.subject-chips-stack') {
    return await page.evaluate((s) => {
        const rows = document.querySelectorAll(`${s} .suggest-marquee-row`);
        if (!rows.length) return null; // n/a
        return Array.from(rows).every(r => r.querySelector('[data-endless-row-lead]'));
    }, sel);
}

// Enter endless mode and paint `n` synced chip rows (one per saved
// beat-prompt). Mirrors the in-product "story running with N beats" state;
// keeps the prompt-array and DOM rows 1:1 so per-row −/↻ work. Seed the
// prompt array via addInitScript BEFORE calling bootstrap.
async function enterEndlessSynced(page, n) {
    await page.click('.subjects-mode-pill button[data-subj-mode="endless"]');
    await page.waitForFunction(() => document.body.classList.contains('endless-running'), null, { timeout: 4000 });
    // Let the dashboard settle (the first WS tick / any one-shot reload)
    // before driving the render, otherwise the evaluate can race a
    // navigation and throw "Execution context was destroyed". Retry the
    // render once if that happens.
    await page.waitForLoadState('networkidle').catch(() => {});
    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            await page.evaluate(() => window._renderEndlessRows(6));
            await page.waitForFunction(
                (count) => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length === count,
                n,
                { timeout: 8000 },
            );
            return;
        } catch (e) {
            if (attempt === 1) throw e;
            await page.waitForTimeout(500);
        }
    }
}

// ---------------------------------------------------------------------------
// CONTRACT: in endless mode, EVERY rendered row has the lead cluster
// (subject-prompt chip + per-row refresh + remove). Naked rows are the
// "simple-mode leak" symptom we fought through 4+ commits.
// ---------------------------------------------------------------------------

test('endless: every row has [data-endless-row-lead]', async ({ page }) => {
    await stubSuggestionResponse(page, ['lonely lighthouse keeper', 'cyberpunk dragon', 'hermit crab lawyer', 'neon jellyfish', 'clay robot rebellion', 'symbiotic mushroom city']);
    // Seed 2 beat-prompts so the synced render paints 2 endless rows.
    // (setMode/enterEndlessSynced flips _endlessRunning=true and seeds the
    // row-prompts — no separate Start Story button anymore; see
    // _setSubjectsMode in app.js line ~2589.)
    await bootstrap(page, 'default', ['yes-and', 'plot-twist']);
    await enterEndlessSynced(page, 2);
    const allHaveLead = await rowsHaveLead(page, '#subject-chips-stack-endless');
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
    // Start from one synced row, then + (during the slow fetch the new
    // row's mask should be empty — no blank-chip placeholders).
    await bootstrap(page, 'default', ['yes-and']);
    await enterEndlessSynced(page, 1);
    const rowsBefore = await countRows(page, '#subject-chips-stack-endless');
    await page.click('#subjects-suggest-add-btn');
    // Capture state DURING the slow fetch — should have N+1 rows but
    // the new row's mask should have ZERO chip buttons.
    await page.waitForTimeout(200);
    const rowsAfter = await countRows(page, '#subject-chips-stack-endless');
    expect(rowsAfter).toBe(rowsBefore + 1);
    const newRowChipCount = await page.evaluate(() => {
        const rows = document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row');
        const last = rows[rows.length - 1];
        if (!last) return -1;
        // Chips inside the marquee mask, not the lead cluster.
        return last.querySelectorAll('.suggest-marquee-mask .btn[data-suggest]').length;
    });
    expect(newRowChipCount).toBe(0);
});

// ---------------------------------------------------------------------------
// CONTRACT: + button stays ENABLED in endless mode the whole session.
// (The user asked for the + to never look greyed out — see app.js
// _refreshSuggestBadge "isEndless: allow=true". Entering endless auto-
// starts the story now, so the + is enabled immediately and stays so as
// rows are added.)
// ---------------------------------------------------------------------------

test('endless: + always enabled in endless mode', async ({ page }) => {
    await stubSuggestionResponse(page, ['a', 'b']);
    // setMode('endless') *is* the Start Story trigger now (Story-mode
    // redesign removed the separate Start-Story button — see
    // _setSubjectsMode in app.js ~line 2589). Test asserts the +
    // button stays enabled both at entry and after a row is painted/added.
    await bootstrap(page, 'default', ['yes-and']);
    await setMode(page, 'endless');
    // Enabled immediately on entering endless (auto-started).
    const preDisabled = await page.locator('#subjects-suggest-add-btn').evaluate(el => el.disabled);
    expect(preDisabled).toBe(false);
    // Still enabled after painting a row + adding one.
    await enterEndlessSynced(page, 1);
    await page.click('#subjects-suggest-add-btn');
    await page.waitForTimeout(500);
    const postDisabled = await page.locator('#subjects-suggest-add-btn').evaluate(el => el.disabled);
    expect(postDisabled).toBe(false);
});

// ---------------------------------------------------------------------------
// CONTRACT: the prefetch system is SIMPLE-MODE ONLY. Typing into #p-core
// while in endless mode should NOT fire /subjects/suggest from the
// prefetch idle timer (the user's complaint: 'starts adding non-endless
// suggestions when I type a seed').
// ---------------------------------------------------------------------------

test('endless: idle prefetch does not fire', async ({ page }) => {
    // The prefetch idle-timer system is SIMPLE-MODE only. Entering endless
    // and sitting idle must NOT auto-fire /subjects/suggest (the user's
    // complaint: 'starts adding non-endless suggestions'). The shared
    // #p-core seed textarea is hidden in endless now, so there's nothing to
    // type into; we just assert no prefetch fires while idle in endless.
    let suggestCalls = 0;
    await page.route('**/subjects/suggest**', (route) => {
        suggestCalls += 1;
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: ['stub'], simple: ['stub'], chat: ['stub'] } }),
        });
    });
    // Enter endless with NO seeded rows so nothing is painted automatically.
    await bootstrap(page, 'default', []);
    await setMode(page, 'endless');
    await page.waitForTimeout(400); // let any setMode-time fetches settle
    const callsAfterModeSwitch = suggestCalls;
    // Sit idle past the prefetch idle trigger (8s). The shared #p-core seed
    // textarea is hidden in endless now, so there's nothing to type into;
    // we just assert no prefetch fires while idle. If the prefetch leaked,
    // this would be callsAfterModeSwitch + 1+.
    await page.waitForTimeout(9000); // > _PREFETCH_IDLE_TRIGGER_MS (8s)
    const callsAfterIdle = suggestCalls;
    expect(callsAfterIdle).toBeLessThanOrEqual(callsAfterModeSwitch);
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
    await page.waitForSelector('#subject-chips-stack-simple .suggest-marquee-row', { timeout: 5000 });
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
    // No seeded endless rows — endless must start empty after the swap.
    await bootstrap(page, 'default', []);
    await setMode(page, 'simple');
    await page.click('#subjects-suggest-add-btn');
    await page.waitForSelector('#subject-chips-stack-simple .suggest-marquee-row', { timeout: 5000 });
    const simpleRowCount = await countRows(page, '#subject-chips-stack-simple');
    expect(simpleRowCount).toBeGreaterThan(0);
    // Swap to endless — endless owns its OWN stack, so the simple chips
    // can't leak in. The endless stack starts empty (no rows painted until
    // the first +). CSS hides the simple stack via display:none, so it
    // can't leak into the endless view. Each mode's stack is independent.
    await setMode(page, 'endless');
    await page.waitForTimeout(300);
    const endlessRowCount = await countRows(page, '#subject-chips-stack-endless');
    expect(endlessRowCount).toBe(0);
    // The endless stack has no chip rows and none of simple's chips leaked
    // into it (it may show an idle-state placeholder hint string —
    // #subject-chips-empty-endless — but no clickable chips).
    const endlessChips = await page.evaluate(() => Array.from(
        document.querySelectorAll('#subject-chips-stack-endless .btn[data-suggest]')
    ).map(b => (b.dataset.suggest || '').trim()));
    expect(endlessChips).not.toContain('simple-a');
    expect(endlessChips.length).toBe(0);
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
    // Cached chips should appear (in the simple stack — the default mode on
    // a cold load) without any /subjects/suggest call.
    const chipTexts = await page.evaluate(() => Array.from(new Set(
        Array.from(document.querySelectorAll('#subject-chips-stack-simple .btn[data-suggest]'))
            .map(b => (b.dataset.suggest || '').trim())
    )));
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
    // The Start-Story button + lock class were both removed in v316/v317.
    // Entering endless mode IS the start; it never reinstates the dimmer.
    await bootstrap(page, 'default', ['yes-and']);
    // Enter endless + paint one synced row.
    await enterEndlessSynced(page, 1);
    // On entering endless: no lock class.
    let locked = await page.evaluate(() => document.body.classList.contains('endless-pill-locked'));
    expect(locked).toBe(false);
    // Pointer-events should NOT be blocked anywhere on the row.
    const pillPointerEvents = await page.evaluate(() => {
        const el = document.getElementById('subjects-need-ideas-row');
        return el ? getComputedStyle(el).pointerEvents : '';
    });
    expect(pillPointerEvents).not.toBe('none');

    // After adding another row — still no lock. (The Submit affordance was
    // removed in v316/v317, so we exercise the + path instead.)
    await page.click('#subjects-suggest-add-btn');
    await page.waitForFunction(
        () => document.querySelectorAll('#subject-chips-stack-endless .suggest-marquee-row').length >= 2,
        null, { timeout: 6000 },
    );
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
