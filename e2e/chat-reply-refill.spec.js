// v303 — chat reply chips: 4 → 3 + consume-and-refill.
//
// _renderChatReplies fetches n=6 from /subjects/suggest, displays slice(0,3),
// stashes slice(3) on host._chatReplyBuffer. Clicking a chip:
//   1. submits text via _sendChatMessage
//   2. fades out via .chip-disappear
//   3. ~700 ms later, swaps in a fresh chip from the buffer (.chip-arriving)
//
// _renderChatReplies + _consumeChatReply at slopfinity/static/app.js:8661

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

const MOCK_SUGGESTIONS = [
    'reply-A: yes and the moon is jealous',
    'reply-B: tell me about the dragons',
    'reply-C: queue 2 short clips of neon koi',
    'reply-D: what is currently rendering',
    'reply-E: pause the queue please',
    'reply-F: how much disk is free',
];

test.describe('chat reply chips refill (v303)', () => {
    test('renders 3 chips, buffers 3 spares, consumes-and-refills on click', async ({ page }) => {
        // Mock /subjects/suggest BEFORE navigation so the first call inside
        // _renderChatReplies (triggered when we switch to chat mode) hits
        // the stub, not the live LLM.
        await page.route('**/subjects/suggest*', async (route) => {
            // Server response shape (post-23ec1b1) is a per-mode dict, not
            // a flat array. _renderChatReplies reads dict.chat. Mirror
            // the same suggestions across all 3 slots to stay mode-agnostic.
            const arr = MOCK_SUGGESTIONS.slice();
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ suggestions: { story: arr, simple: arr, chat: arr } }),
            });
        });
        // Mock the chat send endpoint so clicking a chip doesn't actually
        // hammer the LLM — we only care about the chip animation here.
        await page.route('**/chat/send*', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ ok: true, reply: 'mocked' }),
            });
        });

        // Pre-seed chat history with an assistant turn — _renderChatReplies
        // only auto-fires on mode-switch when the history already contains
        // an assistant message (otherwise we'd burn LLM cycles on chips
        // the user didn't ask for). See app.js:2425-2429.
        await page.addInitScript(() => {
            try {
                localStorage.clear();
                localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify([
                    { role: 'user', content: 'queue 3 dragons' },
                    { role: 'assistant', content: 'Sure, queueing now.' },
                ]));
            } catch (_) {}
        });
        await page.goto(BASE + '/');
        await page.waitForLoadState('domcontentloaded');

        // Switch to chat mode → triggers _renderChatReplies (because history
        // already has an assistant turn).
        await page.click('.subjects-mode-pill button[data-subj-mode="chat"]');

        // Wait for chips to render.
        await page.waitForSelector('#subjects-chat-replies .chat-reply-chip', { timeout: 5000 });

        // Exactly 3 chips visible.
        const chipCount = await page.locator('#subjects-chat-replies .chat-reply-chip').count();
        expect(chipCount).toBe(3);

        // Buffer has the remaining 3.
        const bufLen = await page.evaluate(() => {
            const host = document.getElementById('subjects-chat-replies');
            return (host && host._chatReplyBuffer && host._chatReplyBuffer.length) || 0;
        });
        expect(bufLen).toBe(3);

        // Capture the original first-chip text so we can confirm it gets
        // replaced by a NEW chip after the click.
        const firstText = (await page.locator('#subjects-chat-replies .chat-reply-chip').first().textContent() || '').trim();

        // Click the first chip → animation kicks off. Use force:true
        // because the chip has an active CSS transition (transform +
        // opacity) on hover/render that Playwright's stability check
        // can flake on; the test is about the click handler firing,
        // not whether the chip is animation-quiescent.
        await page.locator('#subjects-chat-replies .chat-reply-chip').first().click({ force: true });

        // During the ~700ms fade window the original chip should still be
        // in the DOM with .chip-disappear.
        await expect(page.locator('#subjects-chat-replies .chip-disappear')).toHaveCount(1, { timeout: 600 });

        // Wait past the 700ms swap-in.
        await page.waitForTimeout(900);

        // Still 3 chips — fresh one slid into the vacated slot.
        const postCount = await page.locator('#subjects-chat-replies .chat-reply-chip').count();
        expect(postCount).toBe(3);

        // Buffer now has 2.
        const bufLenAfter = await page.evaluate(() => {
            const host = document.getElementById('subjects-chat-replies');
            return (host && host._chatReplyBuffer && host._chatReplyBuffer.length) || 0;
        });
        expect(bufLenAfter).toBe(2);

        // The new first chip should NOT be the original (it was either
        // replaced by a buffer entry, or the buffer-entry slid into the
        // vacated slot while the rest shifted).
        const allTexts = await page.locator('#subjects-chat-replies .chat-reply-chip').allTextContents();
        const trimmed = allTexts.map(s => s.trim());
        // The original chip text must no longer be present.
        expect(trimmed.includes(firstText)).toBe(false);
        // And one of the buffered entries (D/E/F) must now appear.
        const buffered = MOCK_SUGGESTIONS.slice(3);
        expect(trimmed.some(t => buffered.includes(t))).toBe(true);
    });
});
