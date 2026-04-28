// deep-ui-chrome-nav.spec.js
//
// Read-only deep inspection of Slopfinity's GLOBAL CHROME + NAVIGATION
// + SETTINGS surfaces. Captures visibility, computed styles, bounding
// boxes, transitions, and per-breakpoint behaviour for the surfaces
// listed in the task brief. Output is dumped to
// /tmp/deep-ui-chrome-nav.json and a handful of screenshots to
// /tmp/deep-ui-*.png so the parent agent can score per-element.

const { test, expect } = require('@playwright/test');
const fs = require('fs');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

const VPS = [
    { name: 'compact', width: 375, height: 800 },
    { name: 'medium', width: 900, height: 800 },
    { name: 'large', width: 1440, height: 900 },
];

const LAYOUTS = [
    'default', 'subjects', 'queue', 'gallery',
    'subj-slop', 'queue-slop', 'subj-queue',
];

// shared mutable report — written at the very end.
const REPORT = { meta: { base: BASE, runAt: new Date().toISOString() }, surfaces: {} };

function rec(surface, vp, payload) {
    REPORT.surfaces[surface] = REPORT.surfaces[surface] || {};
    REPORT.surfaces[surface][vp] = payload;
}

async function clearStorage(page) {
    await page.addInitScript(() => {
        try { localStorage.clear(); } catch (_) {}
        try { sessionStorage.clear(); } catch (_) {}
    });
}

async function waitReady(page, timeout = 6000) {
    try {
        await page.waitForFunction(() => {
            const splash = document.getElementById('splash-overlay');
            const main = document.querySelector('main');
            const mainOpacity = main ? parseFloat(getComputedStyle(main).opacity || '1') : 1;
            return !splash && mainOpacity >= 0.99;
        }, null, { timeout });
    } catch (_) { /* still continue — capture what we can */ }
    await page.waitForTimeout(200);
}

async function inspect(page, selector) {
    return page.evaluate((sel) => {
        const el = document.querySelector(sel);
        if (!el) return { exists: false };
        const cs = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        const visible = !!(r.width || r.height) && cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0.01;
        return {
            exists: true,
            visible,
            box: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) },
            display: cs.display,
            visibility: cs.visibility,
            opacity: cs.opacity,
            color: cs.color,
            background: cs.backgroundColor,
            position: cs.position,
            zIndex: cs.zIndex,
            transition: cs.transition,
            transform: cs.transform,
            ariaLabel: el.getAttribute('aria-label'),
            ariaHidden: el.getAttribute('aria-hidden'),
            role: el.getAttribute('role'),
            tabIndex: el.getAttribute('tabindex'),
            text: (el.innerText || '').slice(0, 80),
        };
    }, selector);
}

async function inspectAll(page, selector) {
    return page.evaluate((sel) => {
        const els = Array.from(document.querySelectorAll(sel));
        return els.map((el) => {
            const cs = getComputedStyle(el);
            const r = el.getBoundingClientRect();
            const visible = !!(r.width || r.height) && cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0.01;
            return {
                visible,
                box: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) },
                display: cs.display,
                opacity: cs.opacity,
                color: cs.color,
                bg: cs.backgroundColor,
                text: (el.innerText || '').slice(0, 60),
                ariaLabel: el.getAttribute('aria-label'),
                cls: el.className,
            };
        });
    }, selector);
}

for (const vp of VPS) {
    test.describe(`deep chrome+nav @ ${vp.name} (${vp.width}x${vp.height})`, () => {
        test.use({ viewport: { width: vp.width, height: vp.height } });

        test(`navbar / ticker / mobile nav / fab @ ${vp.name}`, async ({ page }) => {
            await clearStorage(page);
            await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
            await waitReady(page);

            // Top stats navbar
            rec('topNavbar', vp.name, {
                container: await inspect(page, '.navbar, header.navbar, #top-navbar, header'),
                brand: await inspect(page, '.brand, [data-brand], a[href="/"], .navbar a'),
                gear: await inspect(page, '#btn-settings'),
            });

            // 4 metric pills + ticker (sparkline + abbrev)
            const tickerPills = await inspectAll(page, '[data-metric], .stat-pill, .metric-pill, .ticker-pill, .stats .stat');
            const sparklines = await inspectAll(page, '[data-spark], .sparkline, svg.sparkline, .ticker svg');
            // Common abbreviation labels
            const abbrevText = await page.evaluate(() => {
                const labels = ['Dsk', 'RAM', 'GPU', 'Load'];
                const found = {};
                for (const l of labels) {
                    const els = Array.from(document.querySelectorAll('*')).filter(e => e.children.length === 0 && (e.innerText || '').trim() === l);
                    found[l] = els.length;
                }
                return found;
            });
            // SVG icon swap in tight tier — look for an icon inside a metric pill
            const tickerIcons = await inspectAll(page, '.ticker svg, .metric-pill svg, .stat-pill svg, [data-metric] svg');

            rec('ticker', vp.name, {
                pillCount: tickerPills.length,
                pills: tickerPills.slice(0, 8),
                sparklineCount: sparklines.length,
                sparklinesVisible: sparklines.filter(s => s.visible).length,
                abbrevTextCount: abbrevText,
                iconCount: tickerIcons.length,
                iconsVisible: tickerIcons.filter(i => i.visible).length,
            });

            // Mobile nav-bar (only <768)
            const mobileNav = await inspect(page, '#mobile-nav-bar');
            const navPrev = await inspect(page, '#mobile-nav-bar [data-nav="prev"], #mobile-nav-prev, #mobile-nav-bar .prev');
            const navNext = await inspect(page, '#mobile-nav-bar [data-nav="next"], #mobile-nav-next, #mobile-nav-bar .next');
            const navIndicator = await inspect(page, '#mobile-nav-bar .indicator, #mobile-nav-bar .cycle, #mobile-nav-bar .dots');
            const navBadges = await inspectAll(page, '#mobile-nav-bar .badge, #mobile-nav-bar [class*="badge"]');
            // Verify badge "glued" to its sibling circle by measuring distance to nearest btn-circle/avatar
            const badgeGlued = await page.evaluate(() => {
                const nav = document.querySelector('#mobile-nav-bar');
                if (!nav) return null;
                const badges = Array.from(nav.querySelectorAll('.badge'));
                return badges.map((b) => {
                    const br = b.getBoundingClientRect();
                    const circles = Array.from(nav.querySelectorAll('.btn-circle, .avatar, .rounded-full'));
                    let bestDist = Infinity;
                    for (const c of circles) {
                        const cr = c.getBoundingClientRect();
                        const dx = (br.left + br.width / 2) - (cr.left + cr.width / 2);
                        const dy = (br.top + br.height / 2) - (cr.top + cr.height / 2);
                        const d = Math.hypot(dx, dy);
                        if (d < bestDist) bestDist = d;
                    }
                    return {
                        text: (b.innerText || '').trim().slice(0, 12),
                        size: { w: Math.round(br.width), h: Math.round(br.height) },
                        nearestCircleDist: Math.round(bestDist),
                    };
                });
            });
            rec('mobileNav', vp.name, { container: mobileNav, prev: navPrev, next: navNext, indicator: navIndicator, badges: navBadges, badgeGlued });

            // Focus-fab pill
            const fab = await inspect(page, '#focus-fab, .focus-fab, [data-focus-fab]');
            const fabKids = await inspectAll(page, '#focus-fab .btn-circle, #focus-fab button, .focus-fab button');
            rec('focusFab', vp.name, { container: fab, kids: fabKids });

            await page.screenshot({ path: `/tmp/deep-ui-${vp.name}-chrome.png`, fullPage: false });
        });

        test(`settings drawer + tabs @ ${vp.name}`, async ({ page }) => {
            await clearStorage(page);
            await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
            await waitReady(page);

            const beforeOpen = await inspect(page, '#settings-modal');
            // Open
            await page.click('#btn-settings').catch(() => {});
            await page.waitForTimeout(450);
            const afterOpen = await inspect(page, '#settings-modal');
            const drawer = await inspect(page, '#settings-drawer');
            const overlay = await inspect(page, '#settings-drawer .drawer-overlay');
            const stickySave = await inspect(page, '#settings-modal .sticky, #settings-modal [class*="sticky"], #settings-save, #settings-modal footer');

            // Tab strip
            const tabs = await inspectAll(page, '#settings-modal [role="tab"], #settings-modal .tab, #settings-modal .tabs button');
            const arrowL = await inspect(page, '#settings-modal .tabs-arrow-left, #settings-modal [data-tabs-arrow="left"], #settings-modal .arrow-left');
            const arrowR = await inspect(page, '#settings-modal .tabs-arrow-right, #settings-modal [data-tabs-arrow="right"], #settings-modal .arrow-right');

            // Sample form controls
            const controls = await page.evaluate(() => {
                const root = document.querySelector('#settings-modal');
                if (!root) return null;
                const ranges = Array.from(root.querySelectorAll('input[type=range]')).slice(0, 5);
                const toggles = Array.from(root.querySelectorAll('input[type=checkbox], .toggle')).slice(0, 5);
                const selects = Array.from(root.querySelectorAll('select')).slice(0, 5);
                const labels = Array.from(root.querySelectorAll('label')).slice(0, 6);
                const labelledControls = labels.map((l) => ({
                    text: (l.innerText || '').trim().slice(0, 40),
                    forAttr: l.getAttribute('for'),
                    hasMatchingControl: !!(l.getAttribute('for') && root.querySelector(`#${CSS.escape(l.getAttribute('for'))}`)),
                }));
                const summarise = (el) => {
                    const cs = getComputedStyle(el);
                    const r = el.getBoundingClientRect();
                    return { tag: el.tagName, type: el.type, w: Math.round(r.width), h: Math.round(r.height), display: cs.display, name: el.name || el.id || '' };
                };
                return {
                    ranges: ranges.map(summarise),
                    toggles: toggles.map(summarise),
                    selects: selects.map(summarise),
                    labels: labelledControls,
                };
            });

            rec('settingsDrawer', vp.name, { beforeOpen, afterOpen, drawer, overlay, stickySave });
            rec('settingsTabs', vp.name, { tabCount: tabs.length, tabs: tabs.slice(0, 12), arrowL, arrowR });
            rec('settingsForm', vp.name, controls);

            await page.screenshot({ path: `/tmp/deep-ui-${vp.name}-settings.png`, fullPage: false });

            // Theme selector visibility (known issue: should hide compact)
            const themeSel = await inspect(page, '[data-theme-controller], .theme-controller, #theme-select, [name="theme"]');
            rec('themeSelector', vp.name, themeSel);

            // Layout dropdown (View)
            const layoutDD = await inspect(page, '#layout-select, [data-layout-dropdown], select[name="layout"]');
            rec('layoutDropdown', vp.name, layoutDD);

            // Close drawer. At compact (375x800) the overlay click can hang
            // (locator auto-wait when overlay is partially obscured) or tear
            // down the context entirely. Cap the click attempt + ignore any
            // close-side error — the test's assertions (rec'd above) are
            // already complete.
            await page.locator('#settings-drawer .drawer-overlay').click({ position: { x: 8, y: 200 }, timeout: 1500 }).catch(() => {});
            await page.waitForTimeout(300).catch(() => {});
        });

        test(`layouts + slide animation @ ${vp.name}`, async ({ page }) => {
            await clearStorage(page);
            await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
            await waitReady(page);

            const layoutResults = {};
            for (const layout of LAYOUTS) {
                await page.evaluate((l) => {
                    try {
                        document.body.dataset.layout = l;
                        if (window.setLayout) window.setLayout(l);
                    } catch (_) {}
                }, layout);
                // Try the URL form too (more reliable than poking dataset)
                await page.goto(`${BASE}/?layout=${layout}`, { waitUntil: 'domcontentloaded' }).catch(() => {});
                await waitReady(page, 3000);
                const body = await page.evaluate(() => ({
                    layout: document.body.dataset.layout || '',
                    transition: getComputedStyle(document.body).transition,
                    main: (() => {
                        const m = document.querySelector('main');
                        if (!m) return null;
                        const r = m.getBoundingClientRect();
                        return { w: Math.round(r.width), h: Math.round(r.height), transform: getComputedStyle(m).transform };
                    })(),
                    panelsVisible: Array.from(document.querySelectorAll('main > *, main section, .panel'))
                        .filter(e => {
                            const cs = getComputedStyle(e);
                            const r = e.getBoundingClientRect();
                            return r.width > 50 && cs.display !== 'none';
                        }).length,
                }));
                layoutResults[layout] = body;
            }
            rec('layouts', vp.name, layoutResults);
        });

        test(`modals + drawers @ ${vp.name}`, async ({ page }) => {
            await clearStorage(page);
            await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' });
            await waitReady(page);

            const inv = {
                pipelineModal: await inspect(page, '#pipeline-modal'),
                modelSettingsModal: await inspect(page, '#model-settings-modal'),
                seedsPickerModal: await inspect(page, '#seeds-picker-modal'),
                queueDrawer: await inspect(page, '#queue-drawer'),
            };

            // model-settings-modal should NOT have an "Open Pipeline" button
            const hasOpenPipelineBtn = await page.evaluate(() => {
                const m = document.querySelector('#model-settings-modal');
                if (!m) return null;
                const btns = Array.from(m.querySelectorAll('button, a'));
                return btns.some(b => /open pipeline/i.test((b.innerText || '').trim()));
            });
            inv.modelSettingsHasOpenPipeline = hasOpenPipelineBtn;

            // Toast container
            inv.toastContainer = await inspect(page, '.toast, .toast-container, #toast, [class*="toast-themed"]');
            inv.toastsVisible = await inspectAll(page, '.toast-themed-primary, .alert.toast, .toast .alert');

            // Splash overlay (should now be removed)
            inv.splash = await inspect(page, '#splash-overlay');

            rec('modals', vp.name, inv);
        });
    });
}

test.afterAll(async () => {
    try {
        fs.writeFileSync('/tmp/deep-ui-chrome-nav.json', JSON.stringify(REPORT, null, 2));
        console.log('[deep-ui] wrote /tmp/deep-ui-chrome-nav.json');
    } catch (e) {
        console.warn('[deep-ui] failed to write report:', e.message);
    }
});
