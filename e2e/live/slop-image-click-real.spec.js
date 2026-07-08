// LIVE wire-contract test for the slop card → metadata modal flow.
//
// Hits the real :9099 backend. Picks whatever real file is already in the
// SSR-rendered grid (first child of #preview-grid) and asserts:
//   1. /asset/<that-file> returns ok:true with non-empty File/Kind/Created
//      rows.
//   2. The modal renders an <img>/<video>/<audio> sourced from /files/<file>.
//
// This proves the real server agrees with the frontend's expectations on
// the /asset endpoint shape — the wire contract the mocked spec asserts.
//
// Skip-on-empty: if the grid has no cards (fresh server, no slop yet),
// the test is skipped rather than failed — the contract is correct but
// there's nothing to probe. This means the spec is informative on
// populated servers and a no-op on empty ones.

const { test, expect } = require('../_fixtures');

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

// Same 404 noise opt-out as the mocked spec — SSR cards trigger
// /vae_grid + .md sidecar fetches that 404 for older / cleaned-up files.
test.use({ ignoreErrors: { console: [/Failed to load resource.*404/i] } });

test.describe('slop-image-click-real (live :9099)', () => {
    test('Click a real slop card; modal opens with real /asset metadata', async ({ page }) => {
        await bootstrap(page);

        // Find the first card the SSR baked into the page. Skip if the
        // server has no slop yet.
        const firstCard = page.locator('#preview-grid [data-slop-kind][data-slop-file]').first();
        const cardCount = await page.locator('#preview-grid [data-slop-kind]').count();
        test.skip(cardCount === 0, 'No slop on this server yet — nothing to click.');

        const filename = await firstCard.getAttribute('data-slop-file');
        expect(filename).toBeTruthy();

        await firstCard.click();

        const modal = page.locator('#asset-info-modal');
        await expect(modal).toBeVisible({ timeout: 5000 });

        const body = page.locator('#asset-info-body');
        // The real server should populate at least these rows. Don't
        // assert specific values — they vary per file — just that each
        // label has an accompanying non-skeleton value.
        await expect(body).toContainText('File');
        await expect(body).toContainText(filename);
        await expect(body).toContainText('Kind');
        await expect(body).toContainText('Created');
        await expect(body.locator('.badge').first()).toBeVisible();
        // Skeleton placeholders should be gone — the metadata grid
        // renders OVER the skeleton div, replacing it.
        await expect(body.locator('.skeleton')).toHaveCount(0);

        // Media slot was hydrated with a real source pointing back at
        // /files/<filename>. Match img / video / audio.
        const media = page.locator('#asset-info-media :is(img, video source, audio source)').first();
        const src = await media.getAttribute('src');
        expect(src).toContain(`/files/`);
        expect(decodeURIComponent(src)).toContain(filename);

        await page.screenshot({
            path: 'e2e/artifacts/slop-image-click-live.png',
            fullPage: false,
        });
    });
});
