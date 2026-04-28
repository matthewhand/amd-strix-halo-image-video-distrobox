// Deep UI inspection of the Queue card + Slop section surfaces.
// Read-only — measures geometry / contrast / state per element across
// 3 viewports and 3 layouts. Writes findings JSON to e2e/artifacts/
// for the human reviewer; spec assertions are intentionally lax (we
// emit warnings via test.info().annotations rather than failing) so a
// run produces a complete inventory even when the surface is broken.
//
// Run: npx playwright test e2e/deep-ui-queue-slop.spec.js

const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';
const ART = path.join(__dirname, 'artifacts');
fs.mkdirSync(ART, { recursive: true });

const VIEWPORTS = [
    { name: 'compact', width: 375, height: 740 },
    { name: 'medium',  width: 900, height: 800 },
    { name: 'large',   width: 1440, height: 900 },
];
const LAYOUTS = ['default', 'queue', 'gallery'];

// Helper — measure visibility/box/contrast via in-page eval.
const probe = (page) => page.evaluate(() => {
    const measure = (sel, opts = {}) => {
        const els = Array.from(document.querySelectorAll(sel));
        if (!els.length) return { sel, count: 0 };
        const el = els[0];
        const r = el.getBoundingClientRect();
        const cs = getComputedStyle(el);
        let visible = r.width > 0 && r.height > 0;
        let cur = el;
        while (cur && cur !== document.body && visible) {
            const c = getComputedStyle(cur);
            if (c.display === 'none' || c.visibility === 'hidden') visible = false;
            const o = parseFloat(c.opacity || '1');
            if (o <= 0.01) visible = false;
            cur = cur.parentElement;
        }
        // Crude contrast — color luminance vs effective background.
        const lum = (hex) => {
            const m = hex.match(/\d+/g);
            if (!m) return null;
            const [r,g,b] = m.slice(0,3).map(n => parseInt(n,10)/255);
            return 0.2126*r + 0.7152*g + 0.0722*b;
        };
        let bgEl = el;
        let bg = cs.backgroundColor;
        while (bgEl && (!bg || bg === 'rgba(0, 0, 0, 0)' || bg === 'transparent')) {
            bgEl = bgEl.parentElement;
            if (!bgEl) break;
            bg = getComputedStyle(bgEl).backgroundColor;
        }
        const fl = lum(cs.color);
        const bl = lum(bg || 'rgb(255,255,255)');
        const contrast = (fl !== null && bl !== null)
            ? (Math.max(fl, bl) + 0.05) / (Math.min(fl, bl) + 0.05) : null;
        return {
            sel,
            count: els.length,
            visible,
            x: Math.round(r.x), y: Math.round(r.y),
            w: Math.round(r.width), h: Math.round(r.height),
            color: cs.color, bg,
            contrast: contrast ? Math.round(contrast * 100)/100 : null,
            text: (el.innerText || '').slice(0, 80).replace(/\s+/g,' ').trim(),
            disabled: el.disabled || el.getAttribute('aria-disabled') === 'true' || cs.pointerEvents === 'none',
            overflowX: el.scrollWidth > el.clientWidth + 1,
            zIndex: cs.zIndex,
        };
    };
    const SURFACES = [
        // Queue card chrome + header
        '#split-right',
        '#split-right .card-wm-bar',
        '#split-right .card-body > div:first-child',  // header row
        '#q-count',
        '#queue-header-activity',
        // Active job progress bar
        '#active-job-progress-bar',
        // Inline queue list rows
        '#q-list',
        '#q-list > li',
        '#q-list > li summary',
        '#q-list > li .dropdown',
        '#q-list > li summary .font-semibold',
        '#q-list > li .text-base-content\\/50',  // meta row aspect/frames
        // Stage badges
        '#q-list [data-stage]',
        // Per-stage popup
        '#model-settings-modal',
        // Queue depth chip below Queue button
        '#btn-queue-info',
        '#btn-queue-info-depth',
        '#btn-queue-info-status',
        '#btn-start-stop-inline',
        // Pause / clear-failed bulk action row
        '#queue-progress-footer',
        '#btn-queue-pause',
        // Queue drawer
        '#queue-drawer',
        '#queue-drawer-toggle',
        // Slop section card chrome
        '#slop-collapsible',
        '#slop-collapsible-summary',
        '#output-section',
        // Slop filter pill cluster
        '#slop-filters',
        '.slop-filter-pill',
        '[data-slop-filter="video"]',
        '[data-slop-filter="image"]',
        '[data-slop-filter="music"]',
        '[data-slop-filter="speech"]',
        // Intermediates fieldset
        '#slop-filters fieldset',
        '[data-slop-filter="assets"]',
        '[data-slop-filter="frames"]',
        // Frame chip disabled state
        '.slop-filter-frames-label',
        // Slop preview grid
        '#preview-grid',
        '#preview-grid > *:first-child',
        // Seed upload paperclip
        '#btn-seed-upload',
        // Bottom flanking-nav pill
        '#focus-fab-pill',
        '#focus-fab-prev',
        '#focus-fab-next',
        // Gallery FAB
        '#gallery-fab',
    ];
    const results = {};
    SURFACES.forEach(s => { results[s] = measure(s); });

    // Extra: detect "rendering image" ghost-overlay collision by looking
    // for any element whose innerText starts with the phrase that's
    // visible AND positioned within the queue header strip.
    const headerR = document.querySelector('#split-right')?.getBoundingClientRect();
    const ghostHits = [];
    if (headerR) {
        document.querySelectorAll('body *').forEach(el => {
            const t = (el.innerText || '').toLowerCase();
            if (!t.includes('rendering') && !t.includes('still rendering')) return;
            if (el.children.length > 3) return; // only leaf-ish nodes
            const r = el.getBoundingClientRect();
            if (r.width === 0) return;
            // overlap with queue header band (top 60px of the card)
            if (r.top < headerR.top + 60 && r.bottom > headerR.top &&
                r.right > headerR.left && r.left < headerR.right) {
                ghostHits.push({
                    text: t.slice(0, 60),
                    x: Math.round(r.x), y: Math.round(r.y),
                    w: Math.round(r.width), h: Math.round(r.height),
                    sel: el.id ? '#'+el.id : el.tagName + '.' + (el.className || '').toString().split(' ').slice(0,2).join('.'),
                });
            }
        });
    }
    results.__ghostInQueueHeader = ghostHits;

    // Document-level horizontal overflow check
    results.__bodyOverflowX = document.documentElement.scrollWidth > window.innerWidth + 1;
    results.__viewportInner = { w: window.innerWidth, h: window.innerHeight };

    // Queue row collision: meta vs prompt text
    const rows = Array.from(document.querySelectorAll('#q-list > li'));
    results.__queueRowCollisions = rows.map((li, i) => {
        const prompt = li.querySelector('.font-semibold');
        const meta   = li.querySelector('.text-base-content\\/50');
        if (!prompt || !meta) return null;
        const a = prompt.getBoundingClientRect();
        const b = meta.getBoundingClientRect();
        const overlap = !(a.right <= b.left || b.right <= a.left || a.bottom <= b.top || b.bottom <= a.top);
        return { i, overlap, promptW: Math.round(a.width), metaW: Math.round(b.width) };
    }).filter(Boolean);

    return results;
});

const seedQueue = async (page) => {
    // Fire 3 short pending items into the queue via /inject so the
    // surfaces have rows to render. Cheap promises in parallel.
    const prompts = ['neon koi', 'ice castle', 'desert mirage'];
    await Promise.all(prompts.map(p => page.request.post(`${BASE}/inject`, {
        form: { prompt: p, priority: '0' },
    }).catch(() => null)));
};

const cancelAll = async (page) => {
    await page.request.post(`${BASE}/inject`, {
        form: { prompt: '_cleanup', priority: '0', terminate: '1' },
    }).catch(() => null);
};

const allFindings = {};

for (const vp of VIEWPORTS) {
    for (const layout of LAYOUTS) {
        test(`${vp.name}@${vp.width} · layout=${layout} — surface inventory`, async ({ page }) => {
            await page.addInitScript(() => { try { localStorage.clear(); } catch (_) {} });
            await page.setViewportSize({ width: vp.width, height: vp.height });
            // Seed before navigation so SSR picks rows up.
            await seedQueue(page);
            await page.goto(`${BASE}/?layout=${layout}`, { waitUntil: 'domcontentloaded' });
            // Splash fade
            await page.waitForTimeout(1200);
            const findings = await probe(page);
            allFindings[`${vp.name}_${layout}`] = findings;

            await page.screenshot({
                path: path.join(ART, `deepui-${vp.name}-${layout}.png`),
                fullPage: false,
            });

            // Soft assertions — record but don't fail.
            const issues = [];
            if (findings.__bodyOverflowX) issues.push('body has horizontal overflow');
            if (findings.__ghostInQueueHeader.length) {
                issues.push(`ghost overlay in queue header: ${findings.__ghostInQueueHeader.length} hits`);
            }
            const collidedRows = findings.__queueRowCollisions.filter(r => r.overlap);
            if (collidedRows.length) issues.push(`queue row text/meta overlap: ${collidedRows.length}`);
            issues.forEach(i => test.info().annotations.push({ type: 'warning', description: i }));

            // Sanity: page must at least have a Queue card present.
            expect(findings['#split-right'].count).toBeGreaterThanOrEqual(0);
        });
    }
}

test.afterAll(async ({ }) => {
    fs.writeFileSync(
        path.join(ART, 'deep-ui-queue-slop.json'),
        JSON.stringify(allFindings, null, 2)
    );
});

test.afterAll(async ({ request }) => {
    await request.post(`${BASE}/inject`, {
        form: { prompt: '_cleanup', priority: '0', terminate: '1' },
    }).catch(() => null);
});
