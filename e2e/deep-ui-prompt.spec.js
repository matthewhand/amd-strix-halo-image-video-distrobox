// Deep read-only inspection of the Slopfinity Prompt card + Suggestions
// surfaces. Captures visibility, computed styles, bounding-boxes, and
// click-handler presence at three viewports (compact / medium / large)
// for every UI element listed in the inspection scope.
//
// All findings are dumped as JSON to /tmp/deep-ui-prompt-<vp>.json so
// the human reviewer can grade each surface against the score table.
//
// Pure inspection — does NOT mutate the page beyond clicking mode-pill
// tabs to surface mode-specific markup. localStorage is wiped per test
// for reproducibility.

const { test } = require('@playwright/test');
const fs = require('fs');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

const VIEWPORTS = [
    { tier: 'compact', width: 375, height: 740 },
    { tier: 'medium', width: 900, height: 800 },
    { tier: 'large', width: 1440, height: 900 },
];

// Selectors keyed by surface label. Each entry returns one or many DOM
// snapshots (visible/styles/box). Supports both single-element and
// multi-element capture via `all: true`.
const SURFACES = [
    { label: 'prompt-card-root', selector: '#split-left' },
    { label: 'mode-pill-container', selector: '.subjects-mode-pill' },
    { label: 'mode-pill-endless', selector: '.subjects-mode-pill button[data-subj-mode="endless"]' },
    { label: 'mode-pill-simple', selector: '.subjects-mode-pill button[data-subj-mode="simple"]' },
    { label: 'mode-pill-raw', selector: '.subjects-mode-pill button[data-subj-mode="raw"]' },
    { label: 'mode-pill-chat', selector: '.subjects-mode-pill button[data-subj-mode="chat"]' },
    { label: 'mode-icon-active', selector: '.subjects-mode-pill button.btn-primary:not(.btn-outline) .subj-mode-icon' },
    { label: 'subjects-textarea', selector: '#p-core' },
    { label: 'subjects-input-row', selector: '#subjects-input-row' },
    { label: 'btn-start-stop-inline', selector: '#btn-start-stop-inline' },
    { label: 'queue-info-chip', selector: '#btn-queue-info' },
    { label: 'suggest-cluster', selector: '#subjects-suggest-cluster' },
    { label: 'suggest-toggle', selector: '#subjects-suggestions-toggle' },
    { label: 'suggest-prompt-name-btn', selector: '#subjects-suggest-prompt-name' },
    { label: 'suggest-refresh-btn', selector: '#subjects-suggest-btn' },
    { label: 'suggest-add-btn', selector: '#subjects-suggest-add-btn' },
    { label: 'simple-rowctl-add', selector: '#subjects-simple-add-row' },
    { label: 'simple-rowctl-remove', selector: '#subjects-simple-remove-row' },
    { label: 'subject-chips-stack', selector: '#subject-chips-stack' },
    { label: 'subject-chips-empty', selector: '#subject-chips-empty' },
    { label: 'gen-mode-pill', selector: '#gen-mode-pill' },
    { label: 'pipeline-config-button', selector: '#pipeline-config-button' },
    { label: 'subjects-pane-simple', selector: '.subjects-pane[data-pane-mode="simple"]' },
    { label: 'subjects-pane-raw', selector: '.subjects-pane[data-pane-mode="raw"]' },
    { label: 'subjects-pane-endless', selector: '.subjects-pane[data-pane-mode="endless"]' },
    { label: 'subjects-pane-chat', selector: '.subjects-pane[data-pane-mode="chat"]' },
    // raw mode
    { label: 'raw-stage-image', selector: '#p-image' },
    { label: 'raw-stage-video', selector: '#p-video' },
    { label: 'raw-stage-music', selector: '#p-music' },
    { label: 'raw-stage-tts', selector: '#p-tts' },
    { label: 'raw-rewrite-btn', selector: '.raw-rewrite-btn', all: true },
    { label: 'raw-queue-btn', selector: '#btn-start-stop-raw' },
    // chat mode
    { label: 'chat-pane-root', selector: '#subjects-chat-pane' },
    { label: 'chat-suggestions-box', selector: '.chat-suggestions-box' },
    { label: 'chat-replies-container', selector: '#subjects-chat-replies' },
    { label: 'chat-reply-chip', selector: '#subjects-chat-replies > *', all: true },
    { label: 'chat-input', selector: '#subjects-chat-input' },
    { label: 'chat-send-btn', selector: '#subjects-chat-send' },
    { label: 'chat-reset-btn', selector: '#subjects-chat-reset' },
    { label: 'chat-log', selector: '#subjects-chat-log' },
    { label: 'chat-bubble-actions', selector: '#subjects-chat-log .chat-bubble-actions, #subjects-chat-log button[title*="opy"]', all: true },
    { label: 'chat-thought-bubble', selector: '.thought-bubble, [data-role="thought"]', all: true },
    // endless story
    { label: 'story-pane', selector: '#subjects-story-pane' },
    { label: 'story-log', selector: '#subjects-story-log' },
    { label: 'story-stitch-btn', selector: '#subjects-story-stitch' },
    { label: 'story-copy-btn', selector: '#subjects-story-copy' },
    { label: 'story-submit-btn', selector: '#subjects-story-submit' },
    { label: 'story-reset-btn', selector: '#subjects-story-reset' },
    // settings drawer + tabs
    { label: 'settings-drawer-root', selector: '#settings-drawer' },
    { label: 'settings-tab-strip', selector: '#settings-tab-strip' },
];

const MODES = ['endless', 'simple', 'raw', 'chat'];

function inspectSurface(page, sel, all) {
    return page.evaluate(({ sel, all }) => {
        const fmt = (el) => {
            if (!el) return null;
            const cs = window.getComputedStyle(el);
            const r = el.getBoundingClientRect();
            const taSrc = (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') ? el : null;
            // Walk up to find an interactive ancestor for click-handler check
            let clickable = false;
            let cur = el;
            for (let i = 0; i < 4 && cur; i++, cur = cur.parentElement) {
                if (cur.onclick || cur.tagName === 'BUTTON' || cur.tagName === 'A' || cur.getAttribute('role') === 'button') {
                    clickable = true; break;
                }
            }
            return {
                tag: el.tagName,
                id: el.id || null,
                cls: (el.className && typeof el.className === 'string') ? el.className.slice(0, 160) : null,
                visible: cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0.05 && r.width > 0 && r.height > 0,
                display: cs.display,
                visibility: cs.visibility,
                opacity: cs.opacity,
                color: cs.color,
                bg: cs.backgroundColor,
                border: `${cs.borderTopWidth} ${cs.borderTopStyle} ${cs.borderTopColor}`,
                fontSize: cs.fontSize,
                pointerEvents: cs.pointerEvents,
                box: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) },
                placeholder: taSrc ? taSrc.placeholder : null,
                value: taSrc ? (taSrc.value || '').slice(0, 60) : null,
                clickable,
                ariaLabel: el.getAttribute('aria-label'),
                role: el.getAttribute('role'),
                title: el.getAttribute('title'),
                tabIndex: el.tabIndex,
                disabled: el.disabled || el.getAttribute('aria-disabled') === 'true',
            };
        };
        if (all) {
            const els = Array.from(document.querySelectorAll(sel));
            return { count: els.length, items: els.slice(0, 6).map(fmt) };
        }
        const el = document.querySelector(sel);
        return { count: el ? 1 : 0, items: el ? [fmt(el)] : [] };
    }, { sel, all: !!all });
}

async function gotoAndSettle(page, url) {
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
        const splash = document.getElementById('splash-overlay');
        const main = document.querySelector('main');
        const op = main ? parseFloat(main.style.opacity || '1') : 1;
        return !splash && op >= 1;
    }, null, { timeout: 8000 }).catch(() => {});
}

async function harvestForMode(page, mode) {
    await page.click(`.subjects-mode-pill button[data-subj-mode="${mode}"]`).catch(() => {});
    await page.waitForTimeout(450);
    const out = {};
    for (const s of SURFACES) {
        out[s.label] = await inspectSurface(page, s.selector, s.all);
    }
    // Enable suggestions to reveal prompt-name + refresh in this mode
    await page.evaluate(() => {
        const t = document.getElementById('subjects-suggestions-toggle-input');
        if (t && !t.checked) { t.checked = true; t.dispatchEvent(new Event('change', { bubbles: true })); }
    });
    await page.waitForTimeout(200);
    out['__suggestions_enabled'] = {
        suggest_prompt_name_btn_visible: await inspectSurface(page, '#subjects-suggest-prompt-name', false),
        suggest_refresh_btn_visible: await inspectSurface(page, '#subjects-suggest-btn', false),
        suggest_add_btn_visible: await inspectSurface(page, '#subjects-suggest-add-btn', false),
        chips_stack: await inspectSurface(page, '#subject-chips-stack', false),
        chip_rows: await inspectSurface(page, '#subject-chips-stack > *', true),
    };
    return out;
}

for (const vp of VIEWPORTS) {
    test(`deep-ui prompt+suggestions @ ${vp.tier} ${vp.width}x${vp.height}`, async ({ page }) => {
        test.setTimeout(120000);
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await page.addInitScript(() => {
            try {
                localStorage.clear();
                localStorage.setItem('slopfinity_ui_split_upper_px', '720');
            } catch (_) {}
        });
        await gotoAndSettle(page, `${BASE}/?layout=default`);

        const report = { tier: vp.tier, viewport: vp, modes: {} };
        for (const mode of MODES) {
            report.modes[mode] = await harvestForMode(page, mode);
        }

        // Drawer open snapshot — settings tabs scroll arrows etc.
        await page.evaluate(() => {
            const dt = document.getElementById('settings-drawer-toggle');
            if (dt) { dt.checked = true; dt.dispatchEvent(new Event('change', { bubbles: true })); }
            // Some builds use openSettings() helper
            try { if (typeof openSettings === 'function') openSettings(); } catch (_) {}
        });
        await page.waitForTimeout(400);
        report.settings_drawer_opened = {
            drawer_root: await inspectSurface(page, '#settings-drawer', false),
            tab_strip: await inspectSurface(page, '#settings-tab-strip', false),
            tabs: await inspectSurface(page, '#settings-tab-strip > *', true),
            scroll_overflow_x: await page.evaluate(() => {
                const s = document.getElementById('settings-tab-strip');
                if (!s) return null;
                return { scrollWidth: s.scrollWidth, clientWidth: s.clientWidth, overflowX: window.getComputedStyle(s).overflowX };
            }),
        };

        const path = `/tmp/deep-ui-prompt-${vp.tier}.json`;
        fs.writeFileSync(path, JSON.stringify(report, null, 2));
        console.log(`[deep-ui ${vp.tier}] wrote ${path}`);
    });
}
