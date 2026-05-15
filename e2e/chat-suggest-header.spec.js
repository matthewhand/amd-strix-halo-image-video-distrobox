// Chat-mode suggestions header — the pill cluster + count stepper that
// drives reply suggestions. Also covers the thinking-bubble run states
// (active vs done) which gate the cog animation.
//
// Verifies:
//   1. #chat-suggest-prompt-pills populated from suggest_prompts (≥3 buttons)
//   2. #chat-suggest-count displays the current count (default 3)
//   3. − decrements (min 1) and + increments (max 6); tally + chip count
//      update in lockstep
//   4. localStorage `slopfinity-chat-suggest-count` persists across reload
//   5. in-flight thinking run has .chat-thought-active .chat-cog
//   6. resolved thinking run has .chat-thought-done (no animation gating)

const { test, expect } = require('./_fixtures');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.use({ viewport: { width: 1440, height: 900 } });

async function bootChat(page, history) {
    // Stub /subjects/suggest with the per-mode dict shape so chat replies
    // render deterministically.
    await page.route('**/subjects/suggest**', (route) => {
        const arr = ['reply-1', 'reply-2', 'reply-3', 'reply-4', 'reply-5', 'reply-6'];
        return route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ suggestions: { story: arr, simple: arr, chat: arr } }),
        });
    });
    await page.addInitScript((hist) => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            if (hist) {
                localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify(hist));
            }
        } catch (_) { }
    }, history || null);
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 12000 });
    await page.click('.subjects-mode-pill button[data-subj-mode="chat"]');
    await page.waitForTimeout(400);
}

test.describe('chat suggestions header (pill cluster + count stepper)', () => {
    test('pill cluster populated and count badge defaults to 3', async ({ page }) => {
        await bootChat(page);
        // Pills hydrate from suggest_prompts via _renderChatSuggestPromptPills.
        await page.waitForFunction(() => {
            const host = document.getElementById('chat-suggest-prompt-pills');
            return host && host.querySelectorAll('button').length >= 3;
        }, null, { timeout: 5000 });
        const pillCount = await page.locator('#chat-suggest-prompt-pills button').count();
        expect(pillCount).toBeGreaterThanOrEqual(3);

        // v324: per-mode count badge was REMOVED from the chat-suggest
        // cluster per user request "under chat mode the tally counter
        // should be removed". The default count (3) lives on in the
        // localStorage key + _getChatSuggestCount() — no visible badge.
        await expect(page.locator('#chat-suggest-count')).toHaveCount(0);
        const defaultCount = await page.evaluate(() => (typeof window._getChatSuggestCount === 'function' ? window._getChatSuggestCount() : null));
        expect(defaultCount).toBe(3);
    });

    test.skip('+/- step badge in lockstep, clamped to 1..6', async ({ page }) => {
        // v324: the +/- stepper buttons + count badge were REMOVED from
        // the chat-suggest cluster (user request "under chat mode the
        // tally counter should be removed"). Chat replies now render
        // whatever the server returns (capped server-side via
        // suggest_max_len_chat) without a client-side display-count
        // gate. The clamp logic lives in _setChatSuggestCount() — see
        // js-tests/ for that unit coverage.
    });

    test.skip('count persists across reload via localStorage', async ({ page }) => {
        // v324: the count badge was REMOVED from the chat-suggest
        // cluster (user request "under chat mode the tally counter
        // should be removed"). The localStorage key
        // slopfinity-chat-suggest-count still exists for
        // _getChatSuggestCount() but there is no longer a visible
        // badge to read it back into.
    });

    test('in-flight thinking run is .chat-thought-active; resolved is .chat-thought-done', async ({ page }) => {
        // Seed a chat history that ends with a thinking run (assistant
        // tool_calls + tool result) so it renders ACTIVE — nothing follows.
        const inflightHistory = [
            { role: 'user', content: 'do a thing' },
            { role: 'assistant', tool_calls: [
                { id: 'c1', function: { name: 'queue_clip', arguments: '{"prompt":"a dragon"}' } },
            ] },
            { role: 'tool', name: 'queue_clip', content: 'queued ts=123' },
        ];
        await bootChat(page, inflightHistory);
        // Wait for the chat log to render the thought bubble.
        await page.waitForFunction(() => {
            return document.querySelector('.chat-thought');
        }, null, { timeout: 4000 });

        const activeCount = await page.locator('.chat-thought.chat-thought-active').count();
        const cogCount = await page.locator('.chat-thought.chat-thought-active .chat-cog').count();
        expect(activeCount).toBeGreaterThanOrEqual(1);
        expect(cogCount).toBeGreaterThanOrEqual(1);

        // Now seed a history with a thinking run FOLLOWED by an assistant
        // text turn — that flips the run to done.
        const doneHistory = [
            { role: 'user', content: 'do a thing' },
            { role: 'assistant', tool_calls: [
                { id: 'c1', function: { name: 'queue_clip', arguments: '{"prompt":"a dragon"}' } },
            ] },
            { role: 'tool', name: 'queue_clip', content: 'queued ts=123' },
            { role: 'assistant', content: 'Queued the dragon clip.' },
        ];
        await page.evaluate((hist) => {
            localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify(hist));
            // Trigger a re-render via the global helper.
            if (typeof window._renderChatLog === 'function') window._renderChatLog();
        }, doneHistory);
        await page.waitForTimeout(300);

        const doneCount = await page.locator('.chat-thought.chat-thought-done').count();
        expect(doneCount).toBeGreaterThanOrEqual(1);
        // No active bubbles when the run resolved.
        const stillActive = await page.locator('.chat-thought.chat-thought-active').count();
        expect(stillActive).toBe(0);
    });
});
