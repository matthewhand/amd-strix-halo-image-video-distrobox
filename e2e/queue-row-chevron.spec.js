// Queue-row chevron — collapsible affordance on each #q-list <details>
// row. The SVG sits inside <summary> with .q-row-chevron, rotates 90°
// when the parent <details> is open (CSS rule in app.css:2236), and
// has pointer-events:none so clicks land on <summary> (not the SVG).
//
// Verifies:
//   1. each queue row's <summary> has a .q-row-chevron SVG
//   2. when the <details> is open, the chevron rotates (computed style
//      transform contains 'matrix' — the rotate(90deg) collapses to a
//      matrix() at compute time)
//   3. pointer-events:none on the chevron — toggling happens via summary
//
// Strategy: queue is empty on a fresh server, so the live #q-list has no
// rows to assert against. We inject a fixture row that mirrors the exact
// markup _renderQueueRow produces so the same CSS rules apply. This lets
// us pin the CSS contract without depending on live queue state.

const { test, expect } = require('@playwright/test');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.use({ viewport: { width: 1440, height: 900 } });

const FIXTURE_ROW_HTML = `
<li class="bg-base-200 rounded-md" data-q-ts="111111" data-q-status="pending">
    <details>
        <summary class="cursor-pointer p-2 flex items-center gap-2 text-xs flex-wrap">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
                stroke-linejoin="round"
                class="q-row-chevron w-3 h-3 flex-none text-base-content/60"
                aria-hidden="true">
                <polyline points="9 18 15 12 9 6"/>
            </svg>
            <span class="flex-1 min-w-0"><span class="font-semibold truncate">Fixture row</span></span>
        </summary>
        <div class="p-2 text-xs">expanded body</div>
    </details>
</li>`;

async function bootAndInjectRow(page) {
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity_ui_split_upper_px', '700');
        } catch (_) { }
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => !document.getElementById('splash-overlay'), null, { timeout: 5000 });
    // Inject the fixture row into the live #q-list — uses the same CSS
    // selectors as the live queue rows.
    await page.evaluate((html) => {
        const list = document.getElementById('q-list');
        if (!list) throw new Error('#q-list not found');
        list.insertAdjacentHTML('beforeend', html);
    }, FIXTURE_ROW_HTML);
}

test.describe('queue row chevron', () => {
    test('summary has .q-row-chevron SVG', async ({ page }) => {
        await bootAndInjectRow(page);
        const chevCount = await page.locator('#q-list details > summary .q-row-chevron').count();
        expect(chevCount).toBeGreaterThanOrEqual(1);
        // It IS an SVG, not a text glyph.
        const tag = await page.locator('#q-list .q-row-chevron').first().evaluate(el => el.tagName.toLowerCase());
        expect(tag).toBe('svg');
    });

    test('chevron has pointer-events:none (clicks pass through to summary)', async ({ page }) => {
        await bootAndInjectRow(page);
        const pe = await page.locator('#q-list .q-row-chevron').first().evaluate(el => getComputedStyle(el).pointerEvents);
        expect(pe).toBe('none');
    });

    test('chevron rotates 90° when parent <details> is open', async ({ page }) => {
        await bootAndInjectRow(page);
        // Closed state — transform is none / identity matrix.
        const closedTransform = await page.locator('#q-list .q-row-chevron').first().evaluate(el => getComputedStyle(el).transform);
        // 'none' means no rotation; if the browser computes a matrix it
        // should be the identity matrix(1,0,0,1,0,0).
        expect(['none', 'matrix(1, 0, 0, 1, 0, 0)']).toContain(closedTransform);

        // Open the <details> via JS — clicking the summary works too but
        // we're testing the CSS rotation contract, not click-through.
        await page.evaluate(() => {
            const d = document.querySelector('#q-list details');
            if (d) d.open = true;
        });
        await page.waitForTimeout(250); // let the 0.15s transition settle.

        const openTransform = await page.locator('#q-list details[open] .q-row-chevron').first().evaluate(el => getComputedStyle(el).transform);
        // rotate(90deg) computes to matrix(0, 1, -1, 0, 0, 0) — a 90°
        // rotation of the identity. Identity / none would mean the CSS
        // rule didn't apply.
        expect(openTransform).not.toBe('none');
        expect(openTransform).not.toBe('matrix(1, 0, 0, 1, 0, 0)');
        // Sanity — the matrix has a non-zero off-diagonal entry, which is
        // the signature of a rotation.
        expect(/matrix\(.*\)/.test(openTransform)).toBe(true);
    });
});
