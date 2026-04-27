// Layout-mode coverage. body[data-layout] has 7 valid values:
//   default · subjects · queue · gallery · subj-slop · queue-slop · subj-queue
//
// For each mode this suite asserts:
//   • the cards expected to be visible actually render with non-zero size
//   • the cards expected to be hidden are NOT visible
//   • the inter-card splitters are present only when both sides of a
//     splitter are on screen (no dangling dividers)
//   • the focus-mode FAB shows / hides per the layout's contract
//
// Driven by the URL ?layout=<mode> param so each test gets a fresh page
// (avoids cross-contamination from localStorage).

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

// helpers -----------------------------------------------------------------
// Visible = in the DOM AND has a non-zero bounding box AND every
// opacity walks up to a non-zero effective opacity. boundingBox alone
// misses opacity:0 traps (splash overlay holds <main> at opacity:0
// until its fade completes — cards behind have layout but are
// invisible). Walk parents up to body checking effective opacity.
const isVisible = async (page, selector) => {
    return await page.evaluate((sel) => {
        const el = document.querySelector(sel);
        if (!el) return false;
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return false;
        let cur = el;
        while (cur && cur !== document.documentElement) {
            const cs = getComputedStyle(cur);
            if (cs.display === 'none' || cs.visibility === 'hidden') return false;
            const o = parseFloat(cs.opacity || '1');
            if (!isFinite(o) || o <= 0.01) return false;
            cur = cur.parentElement;
        }
        return true;
    }, selector);
};

// Wait for the splash to finish its fade-out and main to reach
// opacity:1. 3.5s upper bound covers the 2.5s splash fallback + 600ms
// fade + 400ms main fade-in + slack.
const waitForSplashGone = async (page) => {
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const mainOpacity = main ? parseFloat(main.style.opacity || '1') : 1;
        return !splash && mainOpacity >= 1;
    }, null, { timeout: 5000 });
};

const assertVisible = async (page, selector, label) => {
    expect(await isVisible(page, selector), `${label} (${selector}) should be visible`).toBe(true);
};

const assertHidden = async (page, selector, label) => {
    expect(await isVisible(page, selector), `${label} (${selector}) should be hidden`).toBe(false);
};

// Each entry: layout → { visible:[ids], hidden:[ids],
//                        showSplitDivider:bool, showHorizontalSplitter:bool,
//                        showFocusFab:bool }
//
// split-divider sits BETWEEN Subjects + Queue in the upper pane —
// only meaningful when BOTH are visible.
// ui-split-handle sits BETWEEN the upper pane and the Slop output —
// only meaningful when at least one upper card AND Slop are visible.
// focus-slop-fab is the floating quick-jump cluster shown in the
// single/dual-card focused modes (NOT default, NOT subj-queue per
// the filmstrip rules, NOT gallery which has its own gallery-fab).
const SCENARIOS = {
    default: {
        visible: ['#split-left', '#split-right', '#output-section'],
        hidden: [],
        showSplitDivider: true,
        showHorizontalSplitter: true,
        showFocusFab: false,
    },
    subjects: {
        visible: ['#split-left'],
        hidden: ['#split-right', '#output-section'],
        showSplitDivider: false,
        showHorizontalSplitter: false,
        showFocusFab: true,
    },
    queue: {
        visible: ['#split-right'],
        hidden: ['#split-left', '#output-section'],
        showSplitDivider: false,
        showHorizontalSplitter: false,
        showFocusFab: true,
    },
    gallery: {
        visible: ['#output-section'],
        hidden: ['#split-left', '#split-right'],
        showSplitDivider: false,
        showHorizontalSplitter: false,
        showFocusFab: false, // gallery uses its OWN #gallery-fab
    },
    'subj-slop': {
        visible: ['#split-left', '#output-section'],
        hidden: ['#split-right'],
        showSplitDivider: false,
        showHorizontalSplitter: true,
        showFocusFab: true,
    },
    'queue-slop': {
        visible: ['#split-right', '#output-section'],
        hidden: ['#split-left'],
        showSplitDivider: false,
        showHorizontalSplitter: true,
        showFocusFab: true,
    },
    'subj-queue': {
        // Slop is visible but as a horizontal filmstrip — still has a
        // bounding box, just compressed. Filmstrip mode hides the FAB
        // (the strip is the layout's defining feature).
        visible: ['#split-left', '#split-right', '#output-section'],
        hidden: [],
        showSplitDivider: true,
        showHorizontalSplitter: false,
        showFocusFab: false,
    },
};

for (const [layout, expected] of Object.entries(SCENARIOS)) {
    test(`layout=${layout}: cards + dividers render per spec`, async ({ page }) => {
        const errors = [];
        page.on('pageerror', err => errors.push(err.message));

        // Clear localStorage on every page load so previous tests'
        // layout / closed-card state never bleeds in. Init-script runs
        // before any page JS, so the URL ?layout= param wins cleanly.
        await page.addInitScript(() => {
            try { localStorage.clear(); } catch (_) {}
        });
        await page.goto(`${BASE}/?layout=${layout}`, { waitUntil: 'domcontentloaded' });
        // Wait for the splash overlay to fully fade out and <main> to
        // reach opacity:1 (otherwise cards have layout but are invisible).
        await waitForSplashGone(page);

        // _applyLayoutView intentionally REMOVES the body data-attribute
        // for the 'default' layout (no rule needed — that's the natural
        // page state). Every other layout sets it.
        const bodyLayout = await page.locator('body').getAttribute('data-layout');
        if (layout === 'default') {
            expect(bodyLayout, 'body[data-layout] for default').toBeNull();
        } else {
            expect(bodyLayout, 'body[data-layout]').toBe(layout);
        }

        for (const sel of expected.visible) {
            await assertVisible(page, sel, `${layout}:${sel}`);
        }
        for (const sel of expected.hidden) {
            await assertHidden(page, sel, `${layout}:${sel}`);
        }

        // Dividers ----------------------------------------------------
        // The split-divider element only exists at xl+ viewports
        // (hidden xl:flex), and the test runs at 1440 wide so it
        // qualifies. If the layout shouldn't show it, it's display:none.
        const splitDividerVisible = await isVisible(page, '#split-divider');
        expect(splitDividerVisible, `${layout}: vertical split-divider`)
            .toBe(expected.showSplitDivider);

        const horizontalSplitterVisible = await isVisible(page, '#ui-split-handle');
        expect(horizontalSplitterVisible, `${layout}: horizontal ui-split-handle`)
            .toBe(expected.showHorizontalSplitter);

        // Focus-mode flanking nav FABs (#focus-fab-prev / #focus-fab-next)
        // replaced the old #focus-slop-fab cluster. The spec's
        // showFocusFab boolean now means "is the user IN a single-card
        // focus layout where flanking-nav makes sense" — true for
        // subjects / queue / gallery (one of the prev/next buttons
        // shows depending on linear position), false for default,
        // subj-slop, queue-slop, subj-queue (multi-pane layouts).
        const fabPrevVisible = await isVisible(page, '#focus-fab-prev');
        const fabNextVisible = await isVisible(page, '#focus-fab-next');
        const focusFabActive = fabPrevVisible || fabNextVisible;
        const expectedFabActive = ['subjects', 'queue', 'gallery'].includes(layout);
        expect(focusFabActive, `${layout}: focus-fab visibility`)
            .toBe(expectedFabActive);

        // No JS pageerror during layout switch.
        expect(errors, `${layout}: pageerror — ${errors.join('|')}`).toEqual([]);

        await page.screenshot({
            path: `e2e/artifacts/layout-${layout}.png`,
            fullPage: false,
        });
    });
}
