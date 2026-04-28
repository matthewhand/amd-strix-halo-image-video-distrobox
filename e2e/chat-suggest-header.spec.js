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

const { test, expect } = require('@playwright/test');

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
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
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

        // Count badge is at #chat-suggest-count, default 3.
        const badge = page.locator('#chat-suggest-count');
        await expect(badge).toHaveCount(1);
        const txt = (await badge.textContent() || '').trim();
        expect(txt).toBe('3');
        const dataCount = await badge.getAttribute('data-count');
        expect(dataCount).toBe('3');
    });

    test('+/- step badge in lockstep, clamped to 1..6', async ({ page }) => {
        await bootChat(page);
        await page.waitForSelector('#chat-suggest-count', { timeout: 5000 });
        // The − button is the join-item btn before the badge; the + is after.
        const minus = page.locator('.chat-suggest-step').first();
        const plus = page.locator('.chat-suggest-step').last();
        const badge = page.locator('#chat-suggest-count');

        // From 3, click − twice → expect 1 (floor at 1).
        await minus.click();
        await page.waitForFunction(() => document.getElementById('chat-suggest-count').textContent.trim() === '2');
        await minus.click();
        await page.waitForFunction(() => document.getElementById('chat-suggest-count').textContent.trim() === '1');

        // Another − should be a no-op (clamped at min 1).
        await minus.click();
        await page.waitForTimeout(200);
        expect((await badge.textContent() || '').trim()).toBe('1');

        // Now ramp + up to 6 then beyond — expect cap at 6.
        for (let i = 0; i < 6; i++) {
            await plus.click();
            await page.waitForTimeout(80);
        }
        const final = (await badge.textContent() || '').trim();
        expect(final).toBe('6');
        const finalAttr = await badge.getAttribute('data-count');
        expect(finalAttr).toBe('6');
    });

    test('count persists across reload via localStorage', async ({ page }) => {
        // Pre-seed the localStorage value DIRECTLY via addInitScript and
        // navigate. addInitScript fires on every navigation including
        // reload, so we encode the seeded value into the script itself
        // (which means no clear() between page visits).
        await page.route('**/subjects/suggest**', (route) => {
            const arr = ['reply-1', 'reply-2', 'reply-3'];
            return route.fulfill({
                status: 200, contentType: 'application/json',
                body: JSON.stringify({ suggestions: { story: arr, simple: arr, chat: arr } }),
            });
        });
        await page.addInitScript(() => {
            try {
                localStorage.setItem('slopfinity-chat-suggest-count', '5');
                localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            } catch (_) { }
        });
        await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
        await page.click('.subjects-mode-pill button[data-subj-mode="chat"]');
        await page.waitForSelector('#chat-suggest-count', { timeout: 5000 });

        // _initChatSuggestCluster reads the persisted count and paints
        // the badge — should read "5" instead of the default "3".
        const restored = (await page.locator('#chat-suggest-count').textContent() || '').trim();
        expect(restored).toBe('5');
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
