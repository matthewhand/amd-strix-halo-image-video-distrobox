// LIVE wire-contract test — hits the real :9099 backend, no mocks.
//
// What this proves (with the real server, AI workers optional):
//   1. Filling the prompt + clicking Queue Slop reaches the real
//      POST /inject endpoint with the FormData shape the server expects.
//   2. The new row actually lands in the persisted queue, identifiable by
//      a UUID prefix the test stamps into the prompt.
//   3. POST /queue/cancel removes the row by ts.
//
// Each test stamps a UUID prefix (e2e-<uuid>) into the prompt and uses it
// as a marker for poll + cleanup. Even if the test crashes mid-run the
// only residue is one cancelled row tagged with the prefix, easy to grep.
//
// Companion: e2e/queue-submit-roundtrip.spec.js (mocked CI version).

const { test, expect } = require('../_fixtures');
const crypto = require('crypto');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

async function bootstrap(page) {
    await page.addInitScript(() => {
        try {
            localStorage.clear();
            localStorage.setItem('slopfinity_suggestions_hidden', '0');
        } catch (_) { }
    });
    await page.goto(`${BASE}/?layout=default`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const op = main ? parseFloat(getComputedStyle(main).opacity) : 1;
        return !splash && op >= 0.99;
    }, null, { timeout: 12000 });
}

// Poll /queue/paginated until a row whose `prompt` contains `marker` shows
// up. Returns the row (with its `ts`) or throws on timeout.
async function findRowByMarker(request, marker, { timeoutMs = 8000 } = {}) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const res = await request.get(`${BASE}/queue/paginated?filter=pending&limit=100`);
        if (res.ok()) {
            const body = await res.json();
            const hit = (body.items || []).find(it => (it.prompt || '').includes(marker));
            if (hit) return hit;
        }
        await new Promise(r => setTimeout(r, 200));
    }
    throw new Error(`Row with marker "${marker}" did not appear in /queue/paginated within ${timeoutMs}ms`);
}

async function cancelByTs(request, ts) {
    const res = await request.post(`${BASE}/queue/cancel`, {
        data: { ts },
        headers: { 'Content-Type': 'application/json' },
    });
    // 404 is acceptable — the row may already have been cancelled by a
    // teardown retry. Anything else is a real failure.
    if (!res.ok() && res.status() !== 404) {
        throw new Error(`/queue/cancel failed: ${res.status()} ${await res.text()}`);
    }
}

test.describe('queue-submit-real (live :9099)', () => {
    test('Submit lands a real row in the persisted queue; cancel removes it', async ({ page, request }) => {
        const uuid = crypto.randomUUID();
        const marker = `e2e-${uuid}`;
        const PROMPT = `[${marker}] live wire-contract probe`;

        let row;
        try {
            await bootstrap(page);
            await page.fill('#p-core', PROMPT);
            await page.click('#btn-start-stop-inline');

            // Wait for the toast as a fast confirmation the client-side
            // submit completed. Server-side persistence is a separate
            // poll below.
            await expect(page.locator('.toast .alert').first()).toBeVisible({ timeout: 3000 });

            // Now hit the real /queue/paginated endpoint and confirm a
            // row tagged with our marker exists.
            row = await findRowByMarker(request, marker);
            expect(row.prompt).toContain(marker);
            expect(typeof row.ts).toBe('number');
            expect(row.status === null || row.status === 'pending' || row.status === 'working').toBe(true);

            // Visual proof — open "View all" so the persisted row is
            // visible in the queue drawer, then screenshot. This is the
            // strongest demonstration that the real server accepted and
            // stored the row (vs the toast which only proves client-side
            // optimism).
            await page.click('text=View all');
            const drawerRow = page.locator(`#queue-drawer-body :text("${marker}")`).first();
            await expect(drawerRow).toBeVisible({ timeout: 5000 });
            await page.screenshot({
                path: 'e2e/artifacts/queue-submit-live-drawer.png',
                fullPage: false,
            });
        } finally {
            // Cleanup. Always attempt cancel even if the test failed
            // mid-way — if /inject did land a row we don't want it
            // polluting the queue.
            if (row && row.ts !== undefined) {
                await cancelByTs(request, row.ts);
            }
        }

        // Verify cleanup actually removed (or status-flipped) the row from
        // the pending filter. The cancelled row stays in `all` but should
        // no longer appear in `pending`.
        const after = await request.get(`${BASE}/queue/paginated?filter=pending&limit=100`);
        const body = await after.json();
        const stillPending = (body.items || []).find(it => (it.prompt || '').includes(marker));
        expect(stillPending).toBeUndefined();
    });
});
