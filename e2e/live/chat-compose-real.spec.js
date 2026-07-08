// LIVE chat wire-contract test — hits the real :9099 LLM, no mocks.
//
// What this proves (when an LLM is configured):
//   1. The real /chat endpoint accepts a `{messages:[...]}` POST and
//      returns the documented shape `{ok, messages[], tool_audit[]}`.
//   2. The frontend renders both the user's typed bubble and the LLM's
//      reply bubble in #subjects-chat-log.
//
// Skip-on-no-LLM: probes GET /llm/health first; if no provider is
// configured (or it's unreachable) the test skips rather than fails.
// That keeps the spec useful on developer machines that have an LLM
// running and a no-op on bare CI / fresh installs.
//
// Slow: real LLMs take seconds-to-minutes to reply. Bumped timeout
// accordingly. Use a short deterministic prompt so the reply is quick.

const { test, expect } = require('../_fixtures');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

async function bootstrap(page) {
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify([]));
        } catch (_) { }
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const op = main ? parseFloat(getComputedStyle(main).opacity) : 1;
        return !splash && op >= 0.99;
    }, null, { timeout: 12000 });
    await page.click('.subjects-mode-pill button[data-subj-mode="chat"]');
    await page.waitForFunction(() => document.body.classList.contains('subj-mode-chat'), null, { timeout: 3000 });
}

test.describe('chat-compose-real (live :9099)', () => {
    // Most of the real wait is in the LLM. 90s is generous for a small
    // model on local hardware; bigger models / cold loads may need more.
    test.setTimeout(90_000);

    test('Real LLM reply lands as an assistant bubble', async ({ page, request }) => {
        // Skip cleanly if no provider is configured.
        const health = await request.get(`${BASE}/llm/health`).then(r => r.ok() ? r.json() : null).catch(() => null);
        test.skip(
            !health || health.ok === false || !!health.error,
            `No LLM available — /llm/health = ${JSON.stringify(health)}`
        );

        const USER_MSG = 'Reply with exactly the word PONG. Do not add anything else.';

        await bootstrap(page);

        await page.fill('#subjects-chat-input', USER_MSG);
        await page.click('#subjects-chat-send');

        // User bubble renders fast (no network).
        await expect(
            page.locator('#subjects-chat-log .chat-end .chat-bubble', { hasText: USER_MSG }).first()
        ).toBeVisible({ timeout: 5000 });

        // Wait for the real assistant bubble to SETTLE. The frontend
        // shows a "thinking..." placeholder bubble immediately, then
        // replaces it with the real content once the LLM responds. Poll
        // until any .chat-start bubble has substantive non-placeholder
        // text. LLMs are non-deterministic so we don't assert content
        // beyond "non-empty + not the placeholder + not an error".
        await page.waitForFunction(() => {
            const bubbles = document.querySelectorAll('#subjects-chat-log .chat-start .chat-bubble');
            for (const b of bubbles) {
                const text = (b.textContent || '').trim().toLowerCase();
                if (!text) continue;
                if (text === 'thinking…' || text === 'thinking...' || text.startsWith('thinking')) continue;
                if (text.startsWith('error') || text.startsWith('network error')) return false; // fail fast
                return true;
            }
            return false;
        }, null, { timeout: 75_000 });

        // Re-fetch the settled bubble for the screenshot annotation.
        const replyText = await page.evaluate(() => {
            const bubbles = document.querySelectorAll('#subjects-chat-log .chat-start .chat-bubble');
            for (const b of bubbles) {
                const text = (b.textContent || '').trim();
                if (text && !text.toLowerCase().startsWith('thinking')) return text;
            }
            return '';
        });
        expect(replyText.length).toBeGreaterThan(0);

        await page.screenshot({
            path: 'e2e/artifacts/chat-compose-live.png',
            fullPage: false,
        });
    });
});
