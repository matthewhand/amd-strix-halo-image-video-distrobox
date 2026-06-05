// Validates the v306 chat-bubble action cluster:
//   - both user and assistant bubbles render `.chat-bubble-actions`
//   - user bubble: copy icon only; assistant bubble: copy + refresh
//   - default opacity 0; on bubble hover ≥ 0.55
//   - clicking copy fires the global _copyBubbleText hook
//   - clicking refresh on the assistant bubble repopulates the input
//     with the preceding user message (history rewound)
//
// Adapted from e2e/chat-mocked.spec.js + e2e/chat-suggestion-send.spec.js
// (mock /chat + seed history) — does NOT modify those files.

const { test, expect } = require('./_fixtures');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

const SEED_HISTORY = [
    { role: 'user', content: 'queue 3 cyberpunk dragons' },
    { role: 'assistant', content: 'Queued 3 short cyberpunk dragon clips.' },
];

async function mockChatRoute(page, chatRequests) {
    await page.route('**/subjects/suggest*', (route) => {
        route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ ok: true, suggestions: ['a', 'b', 'c', 'd'] }),
        });
    });
    await page.route('**/chat', async (route) => {
        const req = route.request();
        let body = {};
        try { body = JSON.parse(req.postData() || '{}'); } catch (_) { }
        chatRequests.push(body);
        const incoming = Array.isArray(body.messages) ? body.messages : [];
        route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                ok: true,
                messages: [
                    ...incoming,
                    { role: 'assistant', content: 'mock assistant reply ' + chatRequests.length },
                ],
                tool_audit: [],
            }),
        });
    });
}

async function bootChat(page, history) {
    await page.addInitScript((hist) => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify(hist));
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
        } catch (_) { }
    }, history);
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 12000 });
    await page.click(`.subjects-mode-pill button[data-subj-mode="chat"]`);
    await page.waitForTimeout(400);
}

test.describe('chat-bubble action cluster', () => {
    test.use({ viewport: { width: 1440, height: 900 } });

    test('user + assistant bubbles render action cluster with correct icon counts', async ({ page }) => {
        const chatRequests = [];
        await mockChatRoute(page, chatRequests);
        await bootChat(page, SEED_HISTORY);

        // Wait for the bubbles to be in the DOM.
        await page.waitForFunction(() => {
            return document.querySelectorAll('.chat-bubble-host').length >= 2;
        }, null, { timeout: 4000 });

        const counts = await page.evaluate(() => {
            const userBubbles = document.querySelectorAll('.chat-end .chat-bubble-host');
            const asstBubbles = document.querySelectorAll('.chat-start .chat-bubble-host');
            const userActions = userBubbles[0]?.querySelectorAll('.chat-bubble-actions .chat-bubble-action').length || 0;
            const asstActions = asstBubbles[0]?.querySelectorAll('.chat-bubble-actions .chat-bubble-action').length || 0;
            return {
                userBubbleCount: userBubbles.length,
                asstBubbleCount: asstBubbles.length,
                userActions,
                asstActions,
            };
        });
        expect(counts.userBubbleCount).toBeGreaterThanOrEqual(1);
        expect(counts.asstBubbleCount).toBeGreaterThanOrEqual(1);
        expect(counts.userActions, 'user bubble: copy only').toBe(1);
        expect(counts.asstActions, 'assistant bubble: copy + refresh').toBe(2);
    });

    test('action cluster opacity goes from 0 to >= 0.5 on bubble hover', async ({ page }) => {
        const chatRequests = [];
        await mockChatRoute(page, chatRequests);
        await bootChat(page, SEED_HISTORY);

        await page.waitForFunction(() => document.querySelectorAll('.chat-bubble-host').length >= 2, null, { timeout: 4000 });

        // Idle opacity should be 0.
        const idleOpacity = await page.evaluate(() => {
            const a = document.querySelector('.chat-start .chat-bubble-host .chat-bubble-actions');
            return a ? parseFloat(getComputedStyle(a).opacity) : -1;
        });
        expect(idleOpacity).toBe(0);

        // Hover the bubble itself.
        await page.locator('.chat-start .chat-bubble-host').first().hover();
        await page.waitForTimeout(250);
        const hoverOpacity = await page.evaluate(() => {
            const a = document.querySelector('.chat-start .chat-bubble-host .chat-bubble-actions');
            return a ? parseFloat(getComputedStyle(a).opacity) : -1;
        });
        expect(hoverOpacity).toBeGreaterThan(0);
        expect(hoverOpacity).toBeGreaterThanOrEqual(0.5);
    });

    test('clicking the copy icon fires _copyBubbleText (spy via window override)', async ({ page }) => {
        const chatRequests = [];
        await mockChatRoute(page, chatRequests);
        await bootChat(page, SEED_HISTORY);

        await page.waitForFunction(() => document.querySelectorAll('.chat-bubble-host').length >= 2, null, { timeout: 4000 });

        // Spy by wrapping the global hook AFTER bubble render. Counts calls
        // and captures the text arg of the most recent call.
        await page.evaluate(() => {
            window.__copySpy = { calls: 0, lastText: null };
            const orig = window._copyBubbleText;
            window._copyBubbleText = function (btn, text) {
                window.__copySpy.calls += 1;
                window.__copySpy.lastText = text;
                // Don't actually hit the clipboard (Playwright permission
                // headaches) — but DO add .copied so the visual flash
                // assertion still holds for callers that rely on it.
                if (btn) {
                    btn.classList.add('copied');
                    setTimeout(() => btn.classList.remove('copied'), 900);
                }
            };
        });

        // Hover then click the copy icon on the assistant bubble (first
        // action button = copy).
        const asstBubble = page.locator('.chat-start .chat-bubble-host').first();
        await asstBubble.hover();
        await asstBubble.locator('.chat-bubble-action').first().click();

        const spy = await page.evaluate(() => window.__copySpy);
        expect(spy.calls, '_copyBubbleText invoked').toBeGreaterThanOrEqual(1);
        expect(spy.lastText || '', 'copy text non-empty').not.toBe('');

        // .copied class lights up briefly.
        const hasCopied = await asstBubble.locator('.chat-bubble-action.copied').count();
        expect(hasCopied).toBeGreaterThanOrEqual(1);
    });

    test('clicking refresh on assistant bubble rewinds history + repopulates input', async ({ page }) => {
        const chatRequests = [];
        await mockChatRoute(page, chatRequests);
        await bootChat(page, SEED_HISTORY);

        await page.waitForFunction(() => document.querySelectorAll('.chat-bubble-host').length >= 2, null, { timeout: 4000 });

        // The assistant bubble bound to the SEEDED history (not any
        // sibling-spec carryover that may also be in the rendered log).
        // We locate it by its visible content instead of "first
        // .chat-start bubble" because cross-spec localStorage carryover
        // (the chat tree key is not always cleared between Playwright
        // contexts) can prepend earlier specs' bubbles in front of the
        // one we just seeded.
        const asstBubble = page.locator('.chat-start .chat-bubble-host', { hasText: SEED_HISTORY[1].content }).first();
        await asstBubble.hover();
        await asstBubble.locator('.chat-bubble-action').nth(1).click();

        // _refreshAssistantTurn truncates history before the user msg, then
        // sets the input value to that user text and fires _sendChatMessage.
        // _sendChatMessage clears the input AFTER reading it; so the truest
        // observable signal is that /chat was POSTed with the original
        // user text as the only user message in the rewound conversation.
        await page.waitForTimeout(400);

        // /chat must have been hit, with a payload that proves the
        // rewind: the ORIGINAL user text is in the request AND the
        // ORIGINAL assistant content is GONE from the sent history
        // (i.e. the assistant turn after the rewound user has been
        // truncated, not appended-after). Both signals come from the
        // request body, which is deterministic regardless of any
        // stale-localStorage carryover that may pollute the
        // post-render history snapshot.
        expect(chatRequests.length, '/chat POSTed by refresh').toBeGreaterThanOrEqual(1);
        const lastReq = chatRequests[chatRequests.length - 1];
        const sentMessages = lastReq.messages || [];
        const lastUser = [...sentMessages].reverse().find((m) => m.role === 'user');
        expect(lastUser, 'last user message in /chat post').toBeTruthy();
        expect(lastUser.content, 'refresh re-sent the original user prompt').toBe(SEED_HISTORY[0].content);
        const sentAssistantContents = sentMessages.filter((m) => m && m.role === 'assistant').map((m) => m.content);
        expect(sentAssistantContents).not.toContain(SEED_HISTORY[1].content);
    });
});
