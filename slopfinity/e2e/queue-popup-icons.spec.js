// Standalone Playwright validation for v304/v305 UI tweaks.
// Run via:  node slopfinity/e2e/queue-popup-icons.spec.js
// Asserts the three changes documented in the v304/v305 notes.

const { chromium } = require('playwright');

const BASE = process.env.SLOP_URL || 'http://localhost:9099';

(async () => {
    const browser = await chromium.launch();
    const ctx = await browser.newContext();
    const page = await ctx.newPage();

    const results = [];
    const log = (name, ok, detail) => {
        results.push({ name, ok, detail });
        console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  -- ' + detail : ''}`);
    };

    try {
        await page.goto(BASE, { waitUntil: 'networkidle', timeout: 30000 });
        await page.waitForTimeout(1500);

        // ------------------------------------------------------------------
        // Change 2 — slop-filter-pill outer-end rounding (>= 1000px proxy)
        // ------------------------------------------------------------------
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
            const fs = document.querySelector('fieldset[aria-label="Intermediate assets filter"] .join');
            if (fs) {
                const labels = fs.querySelectorAll(':scope > label');
                if (labels.length) {
                    const f = labels[0].querySelector('.join-item');
                    const l = labels[labels.length - 1].querySelector('.join-item');
                    if (f) out.fsFirstTL = parseFloat(getComputedStyle(f).borderTopLeftRadius);
                    if (l) out.fsLastTR = parseFloat(getComputedStyle(l).borderTopRightRadius);
                }
            }
            return out;
        });

        const pillOk =
            radii.firstTL >= 1000 && radii.firstBL >= 1000 &&
            radii.lastTR >= 1000 && radii.lastBR >= 1000;
        log('Change2: slop-filter-pill outer ends rounded', pillOk, JSON.stringify(radii));

        const fsOk = (radii.fsFirstTL || 0) >= 1000 && (radii.fsLastTR || 0) >= 1000;
        log('Change2: intermediates fieldset outer ends rounded',
            fsOk || (radii.fsFirstTL === undefined),
            radii.fsFirstTL === undefined
                ? 'fieldset not in DOM at load (ok if conditional)'
                : `fsFirstTL=${radii.fsFirstTL} fsLastTR=${radii.fsLastTR}`);

        // ------------------------------------------------------------------
        // Change 1 — queue popup labels left, icons right
        // We probe the JS template directly via fetch+grep of app.js so we
        // don't depend on a live queue item being present.
        // ------------------------------------------------------------------
        const appJs = await page.evaluate(async () => {
            const r = await fetch('/static/app.js');
            return await r.text();
        });
        const mlRowMatch = /_mlRow\s*=\s*\(label,\s*icon[^)]*\)\s*=>\s*[\s\S]{0,400}?justify-between[\s\S]{0,200}?<span>\$\{label\}<\/span>[\s\S]{0,200}?<span[^>]*font-mono[^>]*>\$\{icon\}<\/span>/;
        log('Change1: _mlRow renders justify-between with <span>label</span><span font-mono>icon</span>',
            mlRowMatch.test(appJs),
            'pattern from app.js source');

        const reQueueOnly = /isCancelled\s*\?\s*`[\s\S]{0,500}?_mlRow\('Re-queue'/.test(appJs);
        log('Change1: cancelled items show only Re-queue', reQueueOnly, '');

        const allFour = ['Edit prompt', 'Cancel'].every(s => appJs.includes(`_mlRow('${s}'`))
            && appJs.includes("'Enable Polymorphic'") && appJs.includes("'Disable Polymorphic'");
        log('Change1: Edit prompt / Polymorphic / Cancel rows present', allFour, '');

        // Try to actually open a queue dropdown if any queue items exist.
        const hasQueueItem = await page.locator('.dropdown .btn:has-text("⋯")').count();
        if (hasQueueItem > 0) {
            await page.locator('.dropdown .btn:has-text("⋯")').first().click({ force: true });
            await page.waitForTimeout(300);
            const liveCheck = await page.evaluate(() => {
                const a = document.querySelector('.dropdown-content a.flex.items-center.justify-between');
                if (!a) return { ok: false, reason: 'no .dropdown-content a found' };
                const spans = a.querySelectorAll('span');
                return {
                    ok: spans.length === 2 && /font-mono/.test(spans[1].className),
                    label: spans[0]?.textContent,
                    icon: spans[1]?.textContent,
                };
            });
            log('Change1: live dropdown row markup', liveCheck.ok,
                `label="${liveCheck.label}" icon="${liveCheck.icon}"`);
        } else {
            log('Change1: live dropdown click', true, 'skipped - no queue items present (source-pattern check sufficed)');
        }

        // ------------------------------------------------------------------
        // Change 3 — model-settings-modal: no Open Pipeline btn, inline link
        // ------------------------------------------------------------------
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
        log('Change3: modal exists', modalShape.exists, '');
        log('Change3: modal-action has only Close', modalShape.actionBtns.length === 1 && /close/i.test(modalShape.actionBtns[0]),
            JSON.stringify(modalShape.actionBtns));
        log('Change3: NO "Open Pipeline" button in modal (initial DOM)', !modalShape.hasOpenPipelineBtn, '');

        // Confirm the inline-link template lives in app.js.
        const inlineLinkOk = /Read-only snapshot from when this item was queued\. Use <a[^>]*link link-primary[^>]*onclick="[^"]*openPipeline\(\)[^"]*">Pipeline Settings<\/a>/.test(appJs);
        log('Change3: inline Pipeline Settings link template in app.js', inlineLinkOk, '');

        // If a queue item exists, open the modal and confirm the rendered link.
        const badgeCount = await page.locator('.queue-strip [data-stage-badge], [onclick*="openModelSettingsPopup"]').count();
        if (badgeCount > 0) {
            await page.locator('[onclick*="openModelSettingsPopup"]').first().click({ force: true });
            await page.waitForTimeout(300);
            const live = await page.evaluate(() => {
                const d = document.getElementById('model-settings-modal');
                if (!d || !d.open) return { open: false };
                const html = d.innerHTML;
                const link = d.querySelector('a.link.link-primary');
                return {
                    open: true,
                    hasOpenPipelineBtn: /Open Pipeline/.test(html),
                    inlineLinkText: link?.textContent || null,
                };
            });
            log('Change3: live modal has inline link, no Open Pipeline btn',
                live.open && !live.hasOpenPipelineBtn && live.inlineLinkText === 'Pipeline Settings',
                JSON.stringify(live));
        } else {
            log('Change3: live modal click', true, 'skipped - no stage badges present');
        }
    } catch (e) {
        log('runner', false, e.message);
    } finally {
        await browser.close();
    }

    const failed = results.filter(r => !r.ok);
    console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
    process.exit(failed.length ? 1 : 0);
})();
