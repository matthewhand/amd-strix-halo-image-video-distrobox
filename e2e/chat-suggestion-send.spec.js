// QA: chat-mode reply-suggestion chips and direct send.
//
// Reproduces the user complaint "clicking a chat-suggestion chip
// doesn't actually send". Mocks /subjects/suggest so the suggestion
// strip renders 4 deterministic chips, mocks /chat so the round-trip
// completes locally, then:
//
//   1. clicks the first suggestion chip and asserts that
//      (a) the input got the chip's text,
//      (b) /chat was POSTed to,
//      (c) the assistant reply rendered in the log.
//
//   2. types a message + clicks Send and asserts the same path.
//
//   3. inspects the chat log for a tool-call assistant turn and
//      asserts the .chat-thought class is present + the dashed
//      border / pseudo-element dots are computed (thought-bubble
//      visual).
//
//   4. snapshots the rendered onclick attribute of the first chip
//      so a future regression (e.g. JSON.stringify producing
//      attribute-breaking quotes) is caught.

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

// Suggestions intentionally include an apostrophe — that's the exact
// shape that previously broke the inline-onclick attribute when it was
// double-quoted (JSON.stringify produces "what's running?" → the
// embedded " bytes break the onclick="..." attribute).
const MOCK_SUGGESTIONS = [
    "queue 3 short clips of cyberpunk dragons",
    "what's running right now?",
    'list "recent" outputs',
    'pause the queue',
];

const MOCK_TOOL_REPLY = {
    ok: true,
    messages: [
        // user echo (whatever the client sent us — we'll merge in the
        // route handler below so this is just a placeholder shape)
        { role: 'user', content: '__placeholder__' },
        {
            role: 'assistant',
            content: '',
            tool_calls: [
                {
                    id: 'call_1',
                    function: {
                        name: 'queue_clip',
                        arguments: JSON.stringify({ subject: 'dragons', count: 3 }),
                    },
                },
            ],
        },
        {
            role: 'tool',
            tool_call_id: 'call_1',
            name: 'queue_clip',
            content: JSON.stringify({ ok: true, queued: 3 }),
        },
        { role: 'assistant', content: 'Queued 3 dragon clips.' },
    ],
    tool_audit: [],
};

async function mockChatRoutes(page, chatRequests) {
    await page.route('**/subjects/suggest*', (route) => {
        route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ ok: true, suggestions: MOCK_SUGGESTIONS }),
        });
    });
    await page.route('**/chat', async (route) => {
        const req = route.request();
        let body = {};
        try { body = JSON.parse(req.postData() || '{}'); } catch (_) {}
        chatRequests.push(body);
        // Echo last user message back, then append assistant reply.
        const incoming = Array.isArray(body.messages) ? body.messages : [];
        const reply = {
            ok: true,
            messages: [
                ...incoming,
                { role: 'assistant', content: 'Got it: "' + (incoming[incoming.length - 1]?.content || '') + '"' },
            ],
            tool_audit: [],
        };
        route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(reply),
        });
    });
}

test.describe('chat-mode suggestion chips + direct send', () => {
    test.use({ viewport: { width: 1440, height: 900 } });

    test('clicking a suggestion chip POSTs to /chat and renders reply', async ({ page }) => {
        const chatRequests = [];
        await mockChatRoutes(page, chatRequests);
        await page.addInitScript(() => {
            try {
                localStorage.clear();
                localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            } catch (_) {}
        });
        const consoleErrors = [];
        page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));
        page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(`console: ${m.text()}`); });

        await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
        await page.click(`.subjects-mode-pill button[data-subj-mode="chat"]`);
        // Wait for replies host to be populated by _renderChatReplies.
        await page.waitForFunction(() => {
            const host = document.getElementById('subjects-chat-replies');
            return host && host.querySelectorAll('button').length >= 4;
        }, null, { timeout: 5000 });

        // Snapshot the first chip's onclick attribute — this is the
        // exact code that used to break when the suggestion contained a
        // double-quote (JSON.stringify embeds " into a "..." attribute).
        const firstChipOnclick = await page.evaluate(() => {
            const btns = document.querySelectorAll('#subjects-chat-replies button');
            return btns[0] ? btns[0].getAttribute('onclick') : null;
        });
        console.log('[chip0 onclick]', firstChipOnclick);
        expect(firstChipOnclick).toBeTruthy();

        // Take a screenshot of the rendered chips — useful for the
        // visual review of "did the chips render at all".
        await page.screenshot({ path: '/tmp/pane-chat-suggestion-chips.png', fullPage: false });

        // Click the chip whose text contains an apostrophe — this is
        // the most likely shape to break the attribute parsing.
        const chips = page.locator('#subjects-chat-replies button');
        const chipCount = await chips.count();
        expect(chipCount).toBeGreaterThanOrEqual(4);
        let targetIdx = 0;
        for (let i = 0; i < chipCount; i++) {
            const t = (await chips.nth(i).textContent() || '').trim();
            if (t.includes("'")) { targetIdx = i; break; }
        }
        const expectedText = (await chips.nth(targetIdx).textContent() || '').trim();
        await chips.nth(targetIdx).click();

        // /chat must be hit AND last user message must equal the chip text.
        await page.waitForFunction(
            (n) => window.__chatPosts === undefined || true,
            null,
            { timeout: 100 },
        );
        // Give the route handler a beat to record + the renderer to redraw.
        await page.waitForTimeout(800);

        if (chatRequests.length === 0) {
            console.error('[FAIL] /chat was never hit — suggestion-click did not fire _sendChatMessage');
            console.error('[debug] page errors:', consoleErrors);
        }
        expect(chatRequests.length).toBeGreaterThanOrEqual(1);
        const lastSent = chatRequests[chatRequests.length - 1];
        const sentMessages = lastSent.messages || [];
        const lastUser = [...sentMessages].reverse().find((m) => m.role === 'user');
        expect(lastUser, 'last user message in /chat POST').toBeTruthy();
        expect(lastUser.content).toBe(expectedText);

        // Reply should be rendered in the chat log.
        const logText = await page.locator('#subjects-chat-log').innerText();
        expect(logText).toContain('Got it');
        expect(logText).toContain(expectedText);
    });

    test('typing a message and pressing Send POSTs to /chat', async ({ page }) => {
        const chatRequests = [];
        await mockChatRoutes(page, chatRequests);
        await page.addInitScript(() => { try { localStorage.clear(); } catch (_) {} });
        await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
        await page.click(`.subjects-mode-pill button[data-subj-mode="chat"]`);
        await page.waitForSelector('#subjects-chat-input');
        await page.fill('#subjects-chat-input', 'hello world from typing');
        await page.click('#subjects-chat-send');
        await page.waitForTimeout(800);
        expect(chatRequests.length).toBeGreaterThanOrEqual(1);
        const sent = chatRequests[chatRequests.length - 1].messages || [];
        const lastUser = [...sent].reverse().find((m) => m.role === 'user');
        expect(lastUser.content).toBe('hello world from typing');
        const logText = await page.locator('#subjects-chat-log').innerText();
        expect(logText).toContain('hello world from typing');
        expect(logText).toContain('Got it');
    });

    test('thought-bubble class + computed style sanity', async ({ page }) => {
        await mockChatRoutes(page, []);
        const HISTORY = [
            { role: 'user', content: 'queue 3 dragons' },
            {
                role: 'assistant',
                content: '',
                tool_calls: [
                    { id: 'c1', function: { name: 'queue_clip', arguments: '{"subject":"dragons","count":3}' } },
                ],
            },
            { role: 'tool', tool_call_id: 'c1', name: 'queue_clip', content: '{"ok":true,"queued":3}' },
            { role: 'assistant', content: 'Queued 3 dragon clips.' },
        ];
        await page.addInitScript((hist) => {
            try {
                localStorage.clear();
                localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify(hist));
                localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            } catch (_) {}
        }, HISTORY);
        await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
        await page.click(`.subjects-mode-pill button[data-subj-mode="chat"]`);
        await page.waitForTimeout(500);
        await page.screenshot({ path: '/tmp/pane-chat-thought-bubbles.png', fullPage: false });

        const stats = await page.evaluate(() => {
            const log = document.getElementById('subjects-chat-log');
            if (!log) return { error: 'no log' };
            const thoughts = log.querySelectorAll('.chat-thought');
            const bubbles = log.querySelectorAll('.chat-bubble');
            const first = thoughts[0];
            let cs = null;
            let pseudoBefore = null;
            let pseudoAfter = null;
            let matchedRules = [];
            if (first) {
                const c = getComputedStyle(first);
                cs = {
                    border: c.border,
                    borderStyle: c.borderStyle,
                    borderWidth: c.borderWidth,
                    borderColor: c.borderColor,
                    borderRadius: c.borderRadius,
                    background: c.backgroundColor,
                    position: c.position,
                };
                const b = getComputedStyle(first, '::before');
                pseudoBefore = { content: b.content, width: b.width, height: b.height, borderStyle: b.borderStyle };
                const a = getComputedStyle(first, '::after');
                pseudoAfter = { content: a.content, width: a.width, height: a.height, borderStyle: a.borderStyle };
                // Walk every stylesheet and collect rules that match this element.
                for (const sheet of document.styleSheets) {
                    let rules;
                    try { rules = sheet.cssRules; } catch (_) { continue; }
                    if (!rules) continue;
                    for (const rule of rules) {
                        if (!rule.selectorText) continue;
                        try {
                            if (first.matches(rule.selectorText)) {
                                matchedRules.push({
                                    sel: rule.selectorText,
                                    cssText: rule.cssText.slice(0, 300),
                                    href: sheet.href || 'inline',
                                });
                            }
                        } catch (_) {}
                    }
                }
            }
            return {
                thoughtCount: thoughts.length,
                bubbleCount: bubbles.length,
                firstHTML: first ? first.outerHTML.slice(0, 200) : null,
                cs,
                pseudoBefore,
                pseudoAfter,
                matchedRules,
            };
        });
        console.log('[chat-thought stats]', JSON.stringify(stats, null, 2));
        expect(stats.thoughtCount, 'expected at least one .chat-thought (assistant tool_calls + tool result)').toBeGreaterThanOrEqual(2);
        expect(stats.cs.borderStyle).toContain('dashed');
        // Pseudo-element dots should have non-empty content.
        expect(stats.pseudoBefore.content).not.toBe('none');
        expect(stats.pseudoAfter.content).not.toBe('none');
    });

    test('tool-call gear-cog animation indicator', async ({ page }) => {
        await mockChatRoutes(page, []);
        await page.addInitScript(() => { try { localStorage.clear(); } catch (_) {} });
        await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
        await page.click(`.subjects-mode-pill button[data-subj-mode="chat"]`);
        await page.waitForSelector('#subjects-chat-input');
        await page.fill('#subjects-chat-input', 'queue 3 dragons');
        // Slow-roll the /chat response so we can inspect the in-flight UI.
        await page.unroute('**/chat');
        await page.route('**/chat', async (route) => {
            await new Promise((r) => setTimeout(r, 1500));
            const body = JSON.parse(route.request().postData() || '{}');
            route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    ok: true,
                    messages: [...(body.messages || []), { role: 'assistant', content: 'done' }],
                    tool_audit: [],
                }),
            });
        });
        await page.click('#subjects-chat-send');
        await page.waitForTimeout(400);
        // Look for any gear/cog/spin indicator while in-flight.
        const probe = await page.evaluate(() => {
            const root = document.querySelector('.subjects-pane[data-pane-mode="chat"]') || document.body;
            const html = root.outerHTML;
            return {
                hasAnimateSpin: !!root.querySelector('.animate-spin'),
                hasCogClass: /\bcog\b|gear/i.test(html),
                hasGearSvg: !!root.querySelector('svg[data-icon="gear"], svg[data-icon="cog"]'),
                sendBtnText: document.getElementById('subjects-chat-send')?.textContent || null,
                sendBtnDisabled: document.getElementById('subjects-chat-send')?.disabled,
            };
        });
        console.log('[tool-call indicator probe]', JSON.stringify(probe, null, 2));
        // Don't fail the test — this is investigative. Just record what's there.
    });
});
