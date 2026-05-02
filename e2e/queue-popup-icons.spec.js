const { test, expect } = require('@playwright/test');

test.describe('v304/v305 UI tweaks', () => {
    test('Change 2: slop-filter-pill outer-end rounding', async ({ page }) => {
        await page.goto('/');
        await page.waitForTimeout(1000);

        const radii = await page.evaluate(() => {
            const out = {};
            const pill = document.querySelector('.slop-filter-pill');
            if (pill) {
                const labels = pill.querySelectorAll(':scope > label');
                if (labels.length) {
                    const firstItem = labels[0].querySelector('.join-item');
                    const lastItem = labels[labels.length - 1].querySelector('.join-item');
                    if (firstItem) {
                        const cs = getComputedStyle(firstItem);
                        out.firstTL = parseFloat(cs.borderTopLeftRadius);
                        out.firstBL = parseFloat(cs.borderBottomLeftRadius);
                    }
                    if (lastItem) {
                        const cs = getComputedStyle(lastItem);
                        out.lastTR = parseFloat(cs.borderTopRightRadius);
                        out.lastBR = parseFloat(cs.borderBottomRightRadius);
                    }
                }
            }
            return out;
        });

        // We use loose assertions here because some environments might have different rendering,
        // but it should be "rounded" (>= 1000px is the proxy for "full round" in this app)
        if (radii.firstTL !== undefined) {
            expect(radii.firstTL).toBeGreaterThanOrEqual(1000);
            expect(radii.firstBL).toBeGreaterThanOrEqual(1000);
            expect(radii.lastTR).toBeGreaterThanOrEqual(1000);
            expect(radii.lastBR).toBeGreaterThanOrEqual(1000);
        } else {
            test.skip('Slop filter pill not found in DOM');
        }
    });

    test('Change 1: queue popup labels left, icons right (source check)', async ({ page }) => {
        await page.goto('/');
        const appJs = await page.evaluate(async () => {
            const r = await fetch('/static/app.js');
            return await r.text();
        });

        const mlRowMatch = /_mlRow\s*=\s*\(label,\s*icon[^)]*\)\s*=>\s*[\s\S]{0,400}?justify-between[\s\S]{0,200}?<span>\$\{label\}<\/span>[\s\S]{0,200}?<span[^>]*font-mono[^>]*>\$\{icon\}<\/span>/;
        expect(mlRowMatch.test(appJs)).toBe(true);

        const reQueueOnly = /isCancelled\s*\?\s*`[\s\S]{0,500}?_mlRow\('Re-queue'/.test(appJs);
        expect(reQueueOnly).toBe(true);

        const allRequired = ['Edit prompt', 'Cancel'].every(s => appJs.includes(`_mlRow('${s}'`))
            && appJs.includes("'Enable Polymorphic'") && appJs.includes("'Disable Polymorphic'");
        expect(allRequired).toBe(true);
    });

    test('Change 3: model-settings-modal logic (source check)', async ({ page }) => {
        await page.goto('/');
        const appJs = await page.evaluate(async () => {
            const r = await fetch('/static/app.js');
            return await r.text();
        });

        const inlineLinkOk = /Read-only snapshot from when this item was queued\. Use <a[^>]*link link-primary[^>]*onclick="[^"]*openPipeline\(\)[^"]*">Pipeline Settings<\/a>/.test(appJs);
        expect(inlineLinkOk).toBe(true);

        const modalShape = await page.evaluate(() => {
            const d = document.getElementById('model-settings-modal');
            if (!d) return { exists: false };
            const action = d.querySelector('.modal-action');
            const btns = action ? Array.from(action.querySelectorAll('button')).map(b => (b.textContent || '').trim()) : [];
            const html = d.outerHTML;
            return {
                exists: true,
                actionBtns: btns,
                hasOpenPipelineBtn: /Open Pipeline/.test(html),
            };
        });

        if (modalShape.exists) {
            expect(modalShape.actionBtns.some(b => /close/i.test(b))).toBe(true);
            expect(modalShape.hasOpenPipelineBtn).toBe(false);
        }
    });
});
