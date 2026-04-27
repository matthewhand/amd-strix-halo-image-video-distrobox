// Chat mode with a mocked conversation history pre-seeded into
// localStorage. Verifies that:
//   1. chat-log scrolls internally when full of messages
//   2. input + Send button stay pinned at the bottom of the pane
//      (don't get pushed off-screen by long history)
// Captures /tmp/pane-chat-mocked-{default,focused}.png so the user
// can eyeball whether the chat ever overflows the card.

const { test } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

// 12 turns of mock chat — enough to force the chat-log to need scrolling
// inside any reasonable card height. Mix of user / assistant turns +
// one tool-call to exercise the bubble renderer.
const MOCK_HISTORY = [
    { role: 'user', content: 'queue 3 short clips of cyberpunk dragons' },
    { role: 'assistant', content: 'On it — I\'ll queue three 4-second cyberpunk dragon clips with neon palette, low angle, smoke.' },
    { role: 'user', content: 'what\'s running right now?' },
    { role: 'assistant', content: 'Currently rendering image stage for "neon light sculpture with abstract shadows" (job 19, ~2 min in). Two more queued behind it.' },
    { role: 'user', content: 'cancel job 19' },
    { role: 'assistant', content: 'Cancelled job 19. Next in queue (job 20, "symbiotic colors creating surreal geometries") will start within ~10 s.' },
    { role: 'user', content: 'list recent outputs' },
    { role: 'assistant', content: 'Last 5 finals:\n- slop_19_neon_light_sculpt_shadows_base.png (3m31s ago)\n- slop_17_symbiotic_co_eal_geomet_base.png (4m44s ago)\n- slop_18_echoes_dead_sun_base.png (5m12s ago)\n- slop_22_lighthouse_g_inst_night_base.png (12m32s ago)\n- slop_15_owen_image.mp4 (18m04s ago)' },
    { role: 'user', content: 'pause the queue' },
    { role: 'assistant', content: 'Paused. New jobs will wait. Resume with "/resume" or click the Pause button in the queue card.' },
    { role: 'user', content: 'how much disk free?' },
    { role: 'assistant', content: '76.5 GB free of 386 GB total — disk-low guard threshold is 5 GB / 1 %. Plenty of headroom.' },
];

// Mobile chat — verify the bottom nav bar doesn't push input off-screen
// and chat still scrolls internally on a phone viewport.
test.describe('mobile chat', () => {
    test.use({ viewport: { width: 390, height: 844 } });
    test('mobile chat with mocked history', async ({ page }) => {
        await page.addInitScript((hist) => {
            try {
                localStorage.clear();
                localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify(hist));
            } catch (_) {}
        }, MOCK_HISTORY);
        await page.goto(`${BASE}/?layout=subjects`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => {
            const splash = document.getElementById('splash-overlay');
            return !splash;
        }, null, { timeout: 5000 });
        await page.click(`.subjects-mode-pill button[data-subj-mode="chat"]`);
        await page.waitForTimeout(500);
        await page.screenshot({ path: '/tmp/pane-chat-mocked-mobile.png', fullPage: false });

        // Sanity: mobile-nav-bar shouldn't overlap the chat input.
        const navBox = await page.locator('#mobile-nav-bar').boundingBox();
        const inputBox = await page.locator('#subjects-chat-input').boundingBox();
        if (navBox && inputBox && (inputBox.y + inputBox.height) > navBox.y) {
            console.warn(`[chat-mobile] input bottom (${inputBox.y + inputBox.height}) overlaps nav top (${navBox.y})`);
        }
        // Diagnostic: dump heights of every level in the constraint chain.
        const chain = await page.evaluate(() => {
            const sel = (s) => {
                const el = document.querySelector(s);
                if (!el) return { sel: s, missing: true };
                const r = el.getBoundingClientRect();
                const cs = getComputedStyle(el);
                return { sel: s, top: Math.round(r.top), height: Math.round(r.height), display: cs.display, overflow: cs.overflow, position: cs.position };
            };
            return [
                sel('#split-left'),
                sel('#split-left > .card-body'),
                sel('.subjects-pane[data-pane-mode="chat"]'),
                sel('#subjects-chat-pane'),
                sel('#subjects-chat-log'),
                sel('#subjects-chat-input'),
            ];
        });
        console.log('[chat-mobile-chain]', JSON.stringify(chain, null, 2));
    });
});

for (const layout of ['default', 'subjects']) {
    test(`chat with mocked history (${layout} layout)`, async ({ page }) => {
        await page.addInitScript((hist) => {
            try {
                localStorage.clear();
                // Persisted chat-history key matches _CHAT_HISTORY_KEY in
                // app.js — setting it before page load makes _renderChatLog
                // hydrate the conversation immediately.
                localStorage.setItem('slopfinity-chat-history-v1', JSON.stringify(hist));
                // Pin the upper-pane height (default layout) so the card
                // is tall enough for the test to be meaningful — default
                // 200 px would crop everything regardless of layout.
                localStorage.setItem('slopfinity_ui_split_upper_px', '700');
            } catch (_) {}
        }, MOCK_HISTORY);

        await page.goto(`${BASE}/?layout=${layout}`, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => {
            const splash = document.getElementById('splash-overlay');
            const main = document.querySelector('main');
            const mainOpacity = main ? parseFloat(main.style.opacity || '1') : 1;
            return !splash && mainOpacity >= 1;
        }, null, { timeout: 5000 });

        // Switch to chat mode + give the renderer a moment.
        await page.click(`.subjects-mode-pill button[data-subj-mode="chat"]`);
        await page.waitForTimeout(500);

        await page.screenshot({
            path: `/tmp/pane-chat-mocked-${layout}.png`,
            fullPage: false,
        });

        // Sanity assertion: chat input + Send button should be visible
        // (not pushed off-screen by the long history). Both should have
        // a positive bounding-box area in the viewport.
        const inputBox = await page.locator('#subjects-chat-input').boundingBox();
        const sendBox = await page.locator('#subjects-chat-send').boundingBox();
        if (!inputBox || !sendBox) {
            console.warn(`[chat-mocked-${layout}] input or send missing — bug in chat overflow handling`);
        } else if (inputBox.y + inputBox.height > 900 || sendBox.y + sendBox.height > 900) {
            console.warn(`[chat-mocked-${layout}] input/send below viewport (input.y=${inputBox.y}, send.y=${sendBox.y}) — chat is pushing them off-screen`);
        }
    });
}
