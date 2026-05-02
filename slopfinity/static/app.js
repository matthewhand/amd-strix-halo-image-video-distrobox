// Slopfinity dashboard client.

// Theme persistence — apply any previously-chosen theme as early as possible
// to avoid a flash of unstyled/default theme on load.
(function () {
    const saved = localStorage.getItem('slopfinity-theme');
    if (saved) document.documentElement.dataset.theme = saved;
    // Visuals density — also apply early so border/padding rules using
    // calc(... * var(--slop-density)) settle before first paint. Default
    // 1.0; user-tunable in Settings → General → Visuals.
    const dens = parseFloat(localStorage.getItem('slopfinity-visuals-density') || '1');
    if (isFinite(dens) && dens > 0) {
        document.documentElement.style.setProperty('--slop-density',
            String(Math.max(0.5, Math.min(1.5, dens))));
    }
})();

function applyTheme(name) {
    if (!name) return;
    document.documentElement.dataset.theme = name;
    localStorage.setItem('slopfinity-theme', name);
}
window.applyTheme = applyTheme;

// ---------------------------------------------------------------------------
// 100% celebration animations — when a stat ticker (GPU/RAM/Load/Disk)
// hits 100, its percentage label gets a random animation class for a
// little visual reward. The user can pin a specific style (or keep
// 'random' for the rotate-on-each-arrival behaviour) via Settings →
// Display. Persisted in localStorage; default 'random'.
// ---------------------------------------------------------------------------
const CELEBRATE_STYLES = ['celebrate-bounce', 'celebrate-wobble',
    'celebrate-pulse', 'celebrate-jump', 'celebrate-wave'];
const _CELEBRATE_KEY = 'slopfinity-celebrate-style';
function _getCelebrateChoice() {
    try {
        const v = localStorage.getItem(_CELEBRATE_KEY);
        if (v && (v === 'random' || CELEBRATE_STYLES.includes(v))) return v;
    } catch (_) { }
    return 'random';
}
function _pickCelebrateClass() {
    const choice = _getCelebrateChoice();
    if (choice !== 'random') return choice;
    return CELEBRATE_STYLES[Math.floor(Math.random() * CELEBRATE_STYLES.length)];
}
function setCelebrateChoice(v) {
    try { localStorage.setItem(_CELEBRATE_KEY, v || 'random'); } catch (_) { }
}
window._pickCelebrateClass = _pickCelebrateClass;
window._getCelebrateChoice = _getCelebrateChoice;
window.setCelebrateChoice = setCelebrateChoice;
window.CELEBRATE_STYLES = CELEBRATE_STYLES;

// ---------------------------------------------------------------------------
// Loading splash — pick a random tip on first paint, fade out once the
// dashboard is live (first WS state tick, or 2.5s timeout). Self-removing.
// ---------------------------------------------------------------------------
const _SPLASH_TIPS = [
    "Hover the right edge of any suggestion row to reveal a 🗑 — drops just that batch.",
    "Click 🎲 Suggest for an instant batch; flip ↻ Auto to keep them flowing.",
    "Endless mode locks the seed and grows a story log — copy it before image/video stages rewrite each line.",
    "Drag the splitter pill between Prompt/Queue and the Slop gallery to re-balance vertical room.",
    "S / M / L pill in the Slop card flips preview density — small for skimming, large for scrubbing.",
    "Click any progress-bar segment to jump straight to that stage's settings.",
    "Pin a wall display to one layout: append ?layout=queue (or subj-slop, gallery, …) to the URL.",
    "CPU offload (Settings → Scheduler) lets the LLM + TTS ride CPU while the iGPU stays free for image/video.",
    "Slop video thumbnails cycle first → middle → last frame as a GIF preview. Toggle off in Diagnostics.",
    "Tap 'I'm Feeling Lucky' in Endless mode (with empty seed) and the LLM picks a story opener for you.",
];

(function _splashController() {
    const splash = () => document.getElementById('splash-overlay');
    // Pick a tip immediately so the user sees it on first paint, even
    // before any JS event fires.
    document.addEventListener('DOMContentLoaded', () => {
        const tip = _SPLASH_TIPS[Math.floor(Math.random() * _SPLASH_TIPS.length)];
        const el = document.getElementById('splash-tip');
        if (el) el.textContent = tip;
    });
    let _splashHidden = false;
    function hideSplash() {
        if (_splashHidden) return;
        _splashHidden = true;
        const el = splash();
        if (!el) return;
        el.style.opacity = '0';
        el.style.pointerEvents = 'none';
        // Two-phase reveal: fade splash for 500ms, remove the node, THEN
        // fade main in over 400ms. Main stays at opacity:0 (set on the
        // <main> tag) so the dashboard never bleeds through the fading
        // splash — the user gets a clean handoff with no overlap.
        setTimeout(() => {
            try { el.remove(); } catch (_) { }
            const main = document.querySelector('main');
            if (main) {
                // requestAnimationFrame so the browser commits opacity:0
                // before we kick the transition; without this the same-
                // frame style change occasionally short-circuits the
                // transition and main pops in instantly.
                requestAnimationFrame(() => { main.style.opacity = '1'; });
            }
        }, 600);
    }
    window._hideSplash = hideSplash;
    // Belt-and-braces hide: 2.5s after page load, regardless of WS state.
    // The WS handler in handleTick() also calls _hideSplash on first tick.
    window.addEventListener('load', () => setTimeout(hideSplash, 2500));
})();

// Slop preview-size pill — sets body[data-slop-size] so the CSS rules
// in app.css resize the preview-grid columns + per-card figure height.
// Default 'm' matches the legacy 2/3/4-col grid; 's' packs more thumbs
// per row for skimming, 'l' shows fewer/bigger for scrubbing video.
const _SLOP_SIZE_KEY = 'slopfinity-slop-size';
function _getSlopSize() {
    try {
        const v = localStorage.getItem(_SLOP_SIZE_KEY);
        return (v === 's' || v === 'l') ? v : 'm';
    } catch (_) { return 'm'; }
}
function _setSlopSize(size) {
    if (size !== 's' && size !== 'm' && size !== 'l') size = 'm';
    try { localStorage.setItem(_SLOP_SIZE_KEY, size); } catch (_) { }
    document.body.dataset.slopSize = size;
    document.querySelectorAll('button[data-slop-size]').forEach(b => {
        b.classList.toggle('subj-mode-active', b.getAttribute('data-slop-size') === size);
    });
}
window._setSlopSize = _setSlopSize;
document.addEventListener('DOMContentLoaded', () => _setSlopSize(_getSlopSize()));

// ---------------------------------------------------------------------------
// Seed-asset uploads — drag/drop anywhere on the page, paste from clipboard,
// or click the ⊕ button in the slop gallery header. Files POST to /upload
// and land in EXP_DIR with a `seed_` prefix so they surface in the gallery
// via the existing /assets endpoint.
// ---------------------------------------------------------------------------
function _seedToast(msg, kind) {
    // success / warning / error stay on daisyUI semantic alert tokens —
    // those are intentionally semantic (a "success" alert reading as
    // theme-primary would lose its meaning). info → primary so the
    // default toast pops with the dashboard's accent colour rather
    // than the generic info-blue. All four are theme-aware via the
    // daisyUI CSS variable layer (--p, --su, --wa, --er) so they
    // re-skin when the user changes themes.
    const cls = ({
        success: 'alert-success',
        warning: 'alert-warning',
        error: 'alert-error',
    })[kind] || 'toast-themed-primary';
    const t = document.createElement('div');
    t.className = 'toast toast-end z-50';
    t.innerHTML = `<div class="alert ${cls} shadow-lg max-w-sm"><span class="text-sm">${msg.replace(/[<>&]/g, s => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[s]))}</span></div>`;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3500);
}
// Generic alias — _seedToast started life as the seed-upload notifier
// but the body is fully mode-agnostic. Expose as window._toast so any
// caller (queue submit, chat send confirm, etc.) can surface a quick
// "we did the thing" message without having to know the seed origin.
window._toast = _seedToast;

// ---------------------------------------------------------------------------
// `apiFetch(url, opts, label)` — wrapper around fetch() that auto-toasts
// on non-2xx + network errors. The audit found ~30 silent failure sites
// across the dashboard (settings save, queue cancel/edit/requeue, prompts
// save, asset delete, etc.) that swallowed errors with `console.warn`.
// User-perceived effect: clicked a button, nothing happened, no signal.
//
// Use:
//   try {
//     const data = await apiFetch('/queue/cancel', {method:'POST', body: ...}, 'Cancel job');
//     // success path
//   } catch (e) {
//     // toast already fired; only catch if you need to react beyond the toast
//   }
//
// Behavior:
// - Fires the underlying fetch unchanged.
// - Network error → toast `${label}: network error` and re-throw.
// - HTTP non-2xx → toast `${label}: HTTP <code>` (or server-supplied
//   `error` field) and throw with the parsed body for callers that
//   want to react. The CSRF middleware (#142) returns 403 with
//   `{ok:false, error:"csrf: ..."}` which surfaces as a clear toast.
// - 2xx → return parsed JSON (or null for empty body).
//
// ---------------------------------------------------------------------------
async function apiFetch(url, opts, label) {
    label = label || 'Request';
    let r;
    try {
        r = await fetch(url, opts || {});
    } catch (e) {
        if (window._toast) window._toast(`${label}: network error — ${e.message || e}`, 'error');
        throw e;
    }
    let body = null;
    const ct = r.headers.get('content-type') || '';
    if (ct.includes('json')) {
        try { body = await r.json(); } catch (e) { body = null; }
    } else {
        try { body = await r.text(); } catch (e) { body = null; }
    }
    if (!r.ok) {
        const errMsg = (body && typeof body === 'object' && body.error)
            ? body.error
            : `HTTP ${r.status}`;
        if (window._toast) window._toast(`${label}: ${errMsg}`, 'error');
        const err = new Error(`${label}: ${errMsg}`);
        err.status = r.status;
        err.body = body;
        throw err;
    }
    return body;
}
window.apiFetch = apiFetch;

async function _uploadSeedFiles(fileList) {
    const files = Array.from(fileList || []).filter(f => f && f.type && f.type.startsWith('image/'));
    if (!files.length) return { saved: [], skipped: [] };
    const fd = new FormData();
    files.forEach(f => fd.append('files', f, f.name || 'upload.png'));
    try {
        const r = await fetch('/upload', { method: 'POST', body: fd });
        const j = await r.json();
        const saved = (j && j.saved) || [];
        const skipped = (j && j.skipped) || [];
        if (saved.length) {
            // VISIBILITY: previously the upload silently landed on disk
            // and the user saw only a toast. Three things hide newly-
            // uploaded seeds from the user's eye:
            //   1. Non-png seeds (jpg/jpeg/webp/gif) never get a WS
            //      `new_file` broadcast (the watcher filters .png/.mp4).
            //   2. Even when broadcast, the slop preview grid hides any
            //      card with data-slop-final=0 unless the 'assets' chip
            //      is on (default off).
            //   3. The seeds-picker modal only refreshes on next open.
            // Fix on three fronts: prepend the new file as a slop card
            // tagged data-slop-seed=1 (the filter exempts seeds), auto-
            // stage them so the staged-seeds strip updates immediately,
            // and refresh the picker if it's open.
            const grid = document.getElementById('preview-grid');
            if (grid && typeof _buildSlopCard === 'function') {
                saved.forEach(file => {
                    const card = _buildSlopCard(file, { pulse: true, autoplay: false });
                    if (card) {
                        card.dataset.slopSeed = '1';
                        grid.insertBefore(card, grid.firstChild);
                    }
                });
                if (typeof _applySlopFilters === 'function') _applySlopFilters();
            }
            // Auto-stage the new uploads so they land in the staged-seeds
            // strip immediately. User intent on uploading a seed is almost
            // always "I want to use this in the next generation."
            if (typeof _setStagedSeeds === 'function' && typeof _getStagedSeeds === 'function') {
                const cur = _getStagedSeeds();
                saved.forEach(f => { if (cur.indexOf(f) === -1) cur.push(f); });
                _setStagedSeeds(cur);
            }
            // If the picker modal is currently open, repopulate its grid
            // so the user sees the new uploads inline rather than having
            // to close + reopen.
            const pickerModal = document.getElementById('seeds-picker-modal');
            if (pickerModal && pickerModal.open && typeof _openSeedsPicker === 'function') {
                _openSeedsPicker();
            }
            _seedToast(`Uploaded ${saved.length} seed${saved.length === 1 ? '' : 's'} — auto-staged`, 'success');
        }
        if (skipped.length) _seedToast(`Skipped ${skipped.length} (${skipped[0].reason})`, 'warning');
        return { saved, skipped };
    } catch (e) {
        _seedToast(`Upload failed: ${e.message}`, 'error');
        return { saved: [], skipped: [] };
    }
}
window._uploadSeedFiles = _uploadSeedFiles;

(function _wireSeedUpload() {
    let dragDepth = 0;
    function isFileDrag(e) {
        const types = e.dataTransfer && e.dataTransfer.types;
        if (!types) return false;
        for (let i = 0; i < types.length; i++) if (types[i] === 'Files') return true;
        return false;
    }
    document.addEventListener('dragenter', (e) => {
        if (!isFileDrag(e)) return;
        dragDepth++;
        document.body.classList.add('seed-drag-active');
    });
    document.addEventListener('dragleave', (e) => {
        if (!isFileDrag(e)) return;
        dragDepth = Math.max(0, dragDepth - 1);
        if (dragDepth === 0) document.body.classList.remove('seed-drag-active');
    });
    document.addEventListener('dragover', (e) => {
        if (!isFileDrag(e)) return;
        e.preventDefault();
    });
    document.addEventListener('drop', (e) => {
        if (!isFileDrag(e)) return;
        e.preventDefault();
        dragDepth = 0;
        document.body.classList.remove('seed-drag-active');
        const files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length) _uploadSeedFiles(files);
    });
    // Clipboard paste — only fire when no input/textarea is focused, so we
    // don't hijack normal text paste.
    document.addEventListener('paste', (e) => {
        const ae = document.activeElement;
        if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.isContentEditable)) return;
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        const files = [];
        for (let i = 0; i < items.length; i++) {
            const it = items[i];
            if (it.kind === 'file') {
                const f = it.getAsFile();
                if (f && f.type && f.type.startsWith('image/')) files.push(f);
            }
        }
        if (files.length) _uploadSeedFiles(files);
    });
    // Click handler for the ⊕ button + hidden file input in the gallery header.
    document.addEventListener('DOMContentLoaded', () => {
        const btn = document.getElementById('btn-seed-upload');
        const input = document.getElementById('seed-upload-input');
        if (btn && input) {
            btn.addEventListener('click', () => input.click());
            input.addEventListener('change', () => {
                if (input.files && input.files.length) {
                    _uploadSeedFiles(input.files).then(() => { input.value = ''; });
                }
            });
        }
    });
})();

// ---------------------------------------------------------------------------
// Seed staging — picker, mode toggle, persistence. Reads /seeds/list to
// populate a thumbnail grid in the modal; selection persists in localStorage
// so the user can stage seeds, navigate around, and queue later. inject()
// reads _getStagedSeeds() / _getSeedsMode() and forwards as form fields.
// ---------------------------------------------------------------------------
const _SEEDS_KEY = 'slopfinity_staged_seeds_v1';
const _SEEDS_MODE_KEY = 'slopfinity_seeds_mode_v1';

function _getStagedSeeds() {
    try {
        const raw = localStorage.getItem(_SEEDS_KEY);
        const arr = raw ? JSON.parse(raw) : [];
        return Array.isArray(arr) ? arr.filter(s => typeof s === 'string' && s.startsWith('seed_')) : [];
    } catch (_) { return []; }
}
window._getStagedSeeds = _getStagedSeeds;

function _setStagedSeeds(arr) {
    const clean = (Array.isArray(arr) ? arr : []).filter(s => typeof s === 'string' && s.startsWith('seed_'));
    try { localStorage.setItem(_SEEDS_KEY, JSON.stringify(clean)); } catch (_) { }
    _refreshSeedsStrip();
}

function _getSeedsMode() {
    try {
        const v = localStorage.getItem(_SEEDS_MODE_KEY);
        return (v === 'per-chain') ? 'per-chain' : 'per-task';
    } catch (_) { return 'per-task'; }
}
window._getSeedsMode = _getSeedsMode;

function _setSeedsMode(mode) {
    const m = (mode === 'per-chain') ? 'per-chain' : 'per-task';
    try { localStorage.setItem(_SEEDS_MODE_KEY, m); } catch (_) { }
    _refreshSeedsStrip();
}
window._setSeedsMode = _setSeedsMode;

function _clearSeeds() {
    _setStagedSeeds([]);
}
window._clearSeeds = _clearSeeds;

function _refreshSeedsStrip() {
    const seeds = _getStagedSeeds();
    const mode = _getSeedsMode();
    const lbl = document.getElementById('seeds-pick-label');
    const pill = document.getElementById('seeds-mode-pill');
    const clear = document.getElementById('btn-seeds-clear');
    if (lbl) lbl.textContent = seeds.length ? `🌱 ${seeds.length} seed${seeds.length === 1 ? '' : 's'} · pick` : 'Pick seeds…';
    if (pill) pill.classList.toggle('hidden', seeds.length === 0);
    if (clear) clear.classList.toggle('hidden', seeds.length === 0);
    if (pill) {
        pill.querySelectorAll('button[data-seeds-mode]').forEach(b => {
            const isActive = b.getAttribute('data-seeds-mode') === mode;
            b.classList.toggle('subj-mode-active', isActive);
            // per-chain needs >=2 seeds — disable visually when not eligible
            if (b.getAttribute('data-seeds-mode') === 'per-chain') {
                b.disabled = seeds.length < 2;
                b.classList.toggle('opacity-40', seeds.length < 2);
            }
        });
    }
    // Update the picker grid checkboxes if the modal is open
    document.querySelectorAll('#seeds-picker-grid [data-seed-file]').forEach(card => {
        const f = card.getAttribute('data-seed-file');
        const cb = card.querySelector('input[type=checkbox]');
        if (cb) cb.checked = seeds.indexOf(f) !== -1;
        card.classList.toggle('ring-2', seeds.indexOf(f) !== -1);
        card.classList.toggle('ring-primary', seeds.indexOf(f) !== -1);
    });
    const cnt = document.getElementById('seeds-picker-count');
    if (cnt) cnt.textContent = `${seeds.length} selected`;
}
window._refreshSeedsStrip = _refreshSeedsStrip;

async function _openSeedsPicker() {
    const modal = document.getElementById('seeds-picker-modal');
    const grid = document.getElementById('seeds-picker-grid');
    const empty = document.getElementById('seeds-picker-empty');
    if (!modal || !grid) return;
    grid.innerHTML = '<div class="col-span-full text-center text-xs opacity-60 py-6">Loading…</div>';
    if (empty) empty.classList.add('hidden');
    if (typeof modal.showModal === 'function') modal.showModal();
    let items = [];
    try {
        const r = await fetch('/seeds/list');
        const j = await r.json();
        items = (j && j.items) || [];
    } catch (_) { items = []; }
    if (!items.length) {
        grid.innerHTML = '';
        if (empty) empty.classList.remove('hidden');
        _refreshSeedsStrip();
        return;
    }
    if (empty) empty.classList.add('hidden');
    const staged = new Set(_getStagedSeeds());
    grid.innerHTML = items.map(it => {
        const f = it.file;
        const checked = staged.has(f);
        return `
            <label data-seed-file="${f}"
                class="cursor-pointer relative block rounded-md overflow-hidden border border-base-300 bg-base-300/40 hover:border-primary transition-colors${checked ? ' ring-2 ring-primary' : ''}">
                <img src="/files/${encodeURIComponent(f)}" alt="${f}" loading="lazy"
                    class="w-full aspect-square object-cover" />
                <input type="checkbox" class="absolute top-1 right-1 checkbox checkbox-sm checkbox-primary bg-base-200"${checked ? ' checked' : ''} />
                <span class="absolute bottom-0 inset-x-0 bg-base-100/85 text-[10px] truncate px-1.5 py-0.5">${f.replace(/^seed_\d+_\d+_/, '')}</span>
            </label>`;
    }).join('');
    grid.querySelectorAll('[data-seed-file]').forEach(card => {
        const f = card.getAttribute('data-seed-file');
        card.addEventListener('click', (e) => {
            // Toggle. Stop checkbox's own click from double-firing.
            e.preventDefault();
            const cur = _getStagedSeeds();
            const i = cur.indexOf(f);
            if (i === -1) cur.push(f);
            else cur.splice(i, 1);
            _setStagedSeeds(cur);
        });
    });
    _refreshSeedsStrip();
}
window._openSeedsPicker = _openSeedsPicker;

document.addEventListener('DOMContentLoaded', _refreshSeedsStrip);

// ---------------------------------------------------------------------------
// PriorityLoader: Limits concurrent media (img/video/audio) transfers.
// Prevents browser slamming when many previews enter the viewport at once.
// ---------------------------------------------------------------------------
const PriorityLoader = (function () {
    const MAX_CONCURRENT = 3;
    let activeCount = 0;
    const queue = [];

    function process() {
        while (activeCount < MAX_CONCURRENT && queue.length > 0) {
            const el = queue.shift();
            activeCount++;
            _doLoad(el);
        }
    }

    function _doLoad(el) {
        const src = el.dataset.src;
        if (!src) { activeCount--; process(); return; }

        // Check if it's a <source> inside <video>/<audio> or just <img>
        const target = el.tagName === 'SOURCE' ? el.parentElement : el;

        const onDone = () => {
            target.removeEventListener('load', onDone);
            target.removeEventListener('canplaythrough', onDone);
            target.removeEventListener('error', onDone);
            activeCount--;
            process();
        };

        target.addEventListener('load', onDone);
        target.addEventListener('canplaythrough', onDone);
        target.addEventListener('error', onDone);

        const isSource = el.tagName === 'SOURCE';

        if (isSource) {
            el.src = src;
            target.load();
        } else {
            el.src = src;
        }
    }

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const targetEl = entry.target;
                // If the container itself has a data-src, or any children do.
                if (targetEl.dataset.src && !targetEl.src) {
                    queue.push(targetEl);
                }
                targetEl.querySelectorAll('[data-src]').forEach(el => {
                    if (!el.src) queue.push(el);
                });
                process();
                observer.unobserve(targetEl);
            }
        });
    }, { rootMargin: '200px' });

    return {
        register: (container) => observer.observe(container)
    };
})();

// UI surface toggles (Settings → Diagnostics → UI surfaces). Default ON.
const _UI_TOGGLE_KEYS = {
    topbar: 'slopfinity-ui-topbar',
    queueBar: 'slopfinity-ui-queue-bar',
    outputThumbs: 'slopfinity-ui-output-thumbs',
};
function _isUiToggleOn(name) {
    try { const v = localStorage.getItem(_UI_TOGGLE_KEYS[name]); return v === null ? true : v === '1'; }
    catch (_) { return true; }
}
function _applyUiToggle(name, on) {
    try { localStorage.setItem(_UI_TOGGLE_KEYS[name], on ? '1' : '0'); } catch (_) { }
    if (name === 'topbar') {
        const navHw = document.querySelector('header .flex-1.flex.items-center.gap-x-6');
        if (navHw) navHw.style.display = on ? '' : 'none';
    } else if (name === 'queueBar') {
        const bar = document.getElementById('active-job-progress-bar');
        if (bar) bar.style.display = on ? '' : 'none';
    }
    // outputThumbs has no DOM mutation here — the cycle interval reads the
    // pref each tick. Toggling off freezes the thumbs on whatever frame
    // they were last seeked to (cheap, no flash).
}
window._applyUiToggle = _applyUiToggle;
document.addEventListener('DOMContentLoaded', () => {
    ['topbar', 'queueBar', 'outputThumbs'].forEach(name => {
        const on = _isUiToggleOn(name);
        _applyUiToggle(name, on);
        const elId = name === 'topbar' ? 'ui-show-topbar'
            : name === 'queueBar' ? 'ui-show-queue-bar'
                : 'ui-output-thumbs';
        const el = document.getElementById(elId);
        if (el) el.checked = on;
    });
});

// ---------------------------------------------------------------------------
// Layout view (Settings → Layout). Two views:
//   - default : Subjects + Queue cards on top, Slop output below.
//   - gallery : Slop fills the viewport. Subjects + Queue collapse
//               out of view; two FABs open them in modals.
//
// The DOM-move pattern below is critical: when a FAB is clicked we MOVE
// the existing #split-left or #split-right node into the dialog mount,
// open the dialog, and on close MOVE it back to its original parent at
// its original position. This keeps every existing event handler /
// observer / live-update binding alive — no markup duplication.
// ---------------------------------------------------------------------------
const _LAYOUT_VIEW_KEY = 'slopfinity-layout-view';
function _applyLayoutView(view) {
    try { localStorage.setItem(_LAYOUT_VIEW_KEY, view); } catch (_) { }
    // Layout modes (drives body[data-layout="..."]; CSS in app.css gates
    // visibility):
    //   default     — Subjects + Queue + Slop (full dashboard)
    //   subjects    — Subjects only (Queue + Slop hidden; FAB → Queue + Slop)
    //   queue       — Queue only (Subjects + Slop hidden; FAB → Subjects + Slop)
    //   gallery     — Slop only (Subjects + Queue hidden; FAB → Subjects + Queue)
    //
    // 2-card layouts (subj-slop / queue-slop / subj-queue) REMOVED — the
    // user found them unnecessary now that single-card focus modes have
    // the side-rail nav + the default multi-pane shows everything at
    // once on large viewports. Stale `?layout=subj-slop` URLs fall
    // through to default.
    const valid = new Set([
        'gallery', 'queue', 'subjects',
    ]);
    if (valid.has(view)) document.body.dataset.layout = view;
    else delete document.body.dataset.layout;
    // Slop section is no longer a <details> — it's a plain <div> that
    // always renders open. Nothing to force here. data-layout still
    // drives which split-pane is visible, but the gallery itself never
    // collapses any more.
    // Mobile nav bar reflects current layout — repaint arrows + label
    // any time the layout changes.
    if (typeof _mobileNavRefresh === 'function') _mobileNavRefresh();
}
window._applyLayoutView = _applyLayoutView;

// ---------------------------------------------------------------------------
// Mobile bottom-nav bar — linear left/right navigation through the three
// single-card layouts. Visible only on mobile widths (CSS @media); hidden
// on desktop where the FAB cluster + View dropdown serve. Order:
//   prompt (subjects)  ←→  queue  ←→  slop (gallery)
// ---------------------------------------------------------------------------
const _MOBILE_NAV_ORDER = [
    { layout: 'subjects', label: 'Prompt' },
    { layout: 'queue', label: 'Queue' },
    { layout: 'gallery', label: 'Slop' },
];
function _mobileNavCurrentIdx() {
    const cur = document.body.dataset.layout || '';
    const i = _MOBILE_NAV_ORDER.findIndex(s => s.layout === cur);
    return i >= 0 ? i : 0;  // default to prompt when desktop layout active
}
function _mobileNavGo(idx) {
    const clamped = Math.max(0, Math.min(_MOBILE_NAV_ORDER.length - 1, idx));
    const target = _MOBILE_NAV_ORDER[clamped];
    if (typeof selectLayoutView === 'function') {
        selectLayoutView(target.layout);
    } else {
        _applyLayoutView(target.layout);
    }
    _mobileNavRefresh();
}
function _mobileNavStep(delta) {
    _mobileNavGo(_mobileNavCurrentIdx() + delta);
}
function _mobileNavRefresh() {
    const i = _mobileNavCurrentIdx();
    const cur = _MOBILE_NAV_ORDER[i];
    const prev = _MOBILE_NAV_ORDER[i - 1];
    const next = _MOBILE_NAV_ORDER[i + 1];
    // Sync the desktop flanking-FAB labels to the prev/next destination
    // names. Used by `_focusFabNav`'s circular buttons so the user reads
    // "PROMPT" / "QUEUE" / "SLOP" inside the circle instead of just an
    // arrow. Independent of mobile-nav-bar visibility (FABs are desktop-
    // only via @media). Safe to call even when the nav bar isn't on
    // screen yet — the FABs use the same _MOBILE_NAV_ORDER cycle.
    const fabPrevLabel = document.querySelector('#focus-fab-prev .fab-label');
    const fabNextLabel = document.querySelector('#focus-fab-next .fab-label');
    if (fabPrevLabel) fabPrevLabel.textContent = prev ? prev.label : '';
    if (fabNextLabel) fabNextLabel.textContent = next ? next.label : '';
    const nav = document.getElementById('mobile-nav-bar');
    if (!nav) return;
    const prevBtn = document.getElementById('mobile-nav-prev');
    const nextBtn = document.getElementById('mobile-nav-next');
    const prevLbl = document.getElementById('mobile-nav-prev-label');
    const nextLbl = document.getElementById('mobile-nav-next-label');
    const curLbl = document.getElementById('mobile-nav-current');
    // Center: render the full cycle (Prompt · Queue · Slop) with the
    // active step highlighted. Gives the user a constant sense of where
    // they are in the linear sequence rather than just naming the
    // current card. _MOBILE_NAV_ORDER is the source of truth for the
    // labels so adding/removing layouts updates this automatically.
    if (curLbl) {
        curLbl.innerHTML = _MOBILE_NAV_ORDER.map((s, j) => {
            const cls = (j === i) ? 'is-current' : '';
            return `<span class="${cls}">${s.label}</span>`;
        }).join('<span class="sep">·</span>');
    }
    if (prevBtn) {
        prevBtn.disabled = !prev;
        prevBtn.classList.toggle('invisible', !prev);
    }
    if (nextBtn) {
        nextBtn.disabled = !next;
        nextBtn.classList.toggle('invisible', !next);
    }
    // Symmetric: each arrow names where it WILL take you. Empty string
    // when at an end so the pill collapses cleanly via .invisible.
    if (prevLbl) prevLbl.textContent = prev ? prev.label : '';
    if (nextLbl) nextLbl.textContent = next ? next.label : '';
}
window._mobileNavStep = _mobileNavStep;
window._mobileNavRefresh = _mobileNavRefresh;

// Focus-mode flanking FABs — same linear order as the mobile nav
// (prompt ↔ queue ↔ slop). delta=-1 → prev, +1 → next. Triggers a
// 220 ms slide-out on <main> in the right direction, swaps the
// layout, then lets the new layout fade in via the same transition.
function _focusFabNav(delta) {
    const idx = _mobileNavCurrentIdx();
    const target = _MOBILE_NAV_ORDER[idx + delta];
    if (!target) return;
    const main = document.querySelector('main');
    if (!main) {
        _mobileNavGo(idx + delta);
        return;
    }
    const cls = delta < 0 ? 'layout-sliding-right' : 'layout-sliding-left';
    main.classList.add(cls);
    // Wait for the slide-out to land, then swap layout + clear class.
    // The new layout's <main> starts off-screen (the same translateX
    // the dying view ended at) and transitions back to identity for
    // a directional 'slide-in' feel.
    setTimeout(() => {
        if (typeof selectLayoutView === 'function') selectLayoutView(target.layout);
        else _applyLayoutView(target.layout);
        // Wipe class on next frame so the new layout's main slides
        // back in from the same direction.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => main.classList.remove(cls));
        });
        _mobileNavRefresh();
    }, 220);
}
window._focusFabNav = _focusFabNav;
document.addEventListener('DOMContentLoaded', () => {
    // On viewports too narrow for the desktop multi-pane layout, redirect
    // to the prompt-only single-card layout. The side-by-side Prompt+Queue
    // split needs ~1280 px to be usable; below that the cards collapse to
    // unreadable widths and the layout looks broken. 1024 px (tablet
    // portrait) is the threshold — below that, force single-card and let
    // the user navigate via the bottom nav. The 767 px @media still
    // controls when the mobile-nav-bar SHOWS, so tablets in this 768-1023
    // band fall through to the same redirect without the bottom nav
    // (their viewport is wide enough that the mode pill + cards work
    // standalone). NOTE: only redirect if we landed on a multi-pane
    // layout — explicit ?layout=subj-queue from a desktop user wins
    // unless they're sub-1024.
    const isNarrow = window.matchMedia && window.matchMedia('(max-width: 1023px)').matches;
    if (isNarrow) {
        const cur = document.body.dataset.layout || '';
        const singleCardLayouts = new Set(['subjects', 'queue', 'gallery']);
        if (!singleCardLayouts.has(cur)) {
            // Use _applyLayoutView directly so we don't go through
            // selectLayoutView's lock check — this redirect should
            // always win on narrow screens.
            _applyLayoutView('subjects');
        }
    }
    _mobileNavRefresh();
});

// ---------------------------------------------------------------------------
// Card maximize — maps each card to the layout that gives it the most space.
// Subjects → 'subjects', Queue → 'queue', Slop → 'gallery'.
// Calling again while already in that mode restores 'default' (toggle).
// ---------------------------------------------------------------------------
function _cardMaximizeLayout(which) {
    if (_isLayoutLocked()) return;
    const focusMap = { subjects: 'subjects', queue: 'queue', slop: 'gallery' };
    const target = focusMap[which] || 'default';
    const current = document.body.dataset.layout || 'default';
    selectLayoutView(current === target ? 'default' : target);
}
window._cardMaximizeLayout = _cardMaximizeLayout;

// ---------------------------------------------------------------------------
// Layout lock — persists in localStorage; gates selectLayoutView calls.
// ---------------------------------------------------------------------------
const _LAYOUT_LOCK_KEY = 'slopfinity-layout-locked';

function _isLayoutLocked() {
    try { return localStorage.getItem(_LAYOUT_LOCK_KEY) === '1'; } catch (_) { return false; }
}

function _setLayoutLocked(locked) {
    try { localStorage.setItem(_LAYOUT_LOCK_KEY, locked ? '1' : '0'); } catch (_) { }
    document.body.classList.toggle('layout-locked', locked);
    ['layout-lock-toggle', 'settings-layout-lock-toggle'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.checked = !locked;
    });
}
window._setLayoutLocked = _setLayoutLocked;

// Gate selectLayoutView so locked state prevents all layout changes.
const _selectLayoutViewOrig = typeof selectLayoutView === 'function' ? selectLayoutView : null;
// (selectLayoutView is defined later in this file; we monkey-patch it after
//  DOMContentLoaded so we catch the real definition.)
document.addEventListener('DOMContentLoaded', () => {
    // Restore lock state on page load. The toggle now lives only in
    // Settings → General (the View dropdown copy was a chicken-and-egg —
    // the lock gates the dropdown, so the lock can't sit inside it).
    if (_isLayoutLocked()) {
        document.body.classList.add('layout-locked');
        ['layout-lock-toggle', 'settings-layout-lock-toggle'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.checked = false;
        });
    }

    // Register initial SSR items for throttled loading.
    // Gallery cards
    document.querySelectorAll('#preview-grid > [data-slop-kind]').forEach(card => {
        PriorityLoader.register(card);
    });
    // Queue items (thumbnails)
    document.querySelectorAll('#q-list > li').forEach(li => {
        PriorityLoader.register(li);
    });

    // Patch selectLayoutView to respect lock.
    const orig = window.selectLayoutView;
    if (orig) {
        window.selectLayoutView = function (v) {
            if (_isLayoutLocked()) return;
            orig(v);
        };
    }
}, { once: true });


// openGalleryFabDialog + #gallery-fab pair removed — flanking focus-fab
// pill + bottom mobile-nav circles now cover prompt/queue access in
// gallery layout. The split-left/split-right cards stay where they are.

// Sibling helper for the Slop FAB. In Subjects-focused / Queue-focused
// view modes the inline Slop card is hidden by CSS; the FAB lets the
// user pop it open in a modal. We reuse the move-card-into-mount
// pattern so all live handlers + WS bindings keep firing.
const _slopFocusReturn = { slop: null };
function openSlopFabDialog() {
    const card = document.getElementById('output-section');
    const mount = document.getElementById('focus-slop-mount');
    const dialog = document.getElementById('focus-slop-modal');
    if (!card || !mount || !dialog) return;
    if (!_slopFocusReturn.slop) {
        _slopFocusReturn.slop = { parent: card.parentNode, next: card.nextSibling };
    }
    mount.appendChild(card);
    // CSS hides #output-section under subjects/queue layouts via
    // `display:none !important`. Inside the modal mount we want it to
    // show, so override the hide by setting an inline display.
    card.style.display = 'block';
    const onClose = () => {
        const r = _slopFocusReturn.slop;
        if (r && r.parent && card.parentNode === mount) {
            r.parent.insertBefore(card, r.next || null);
        }
        // Drop the inline override so the CSS rule resumes control.
        card.style.display = '';
        dialog.removeEventListener('close', onClose);
    };
    dialog.addEventListener('close', onClose);
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
}
window.openSlopFabDialog = openSlopFabDialog;

// `selectLayoutView` is the click target for the new navbar View dropdown.
// It syncs the hidden Settings → Layout radio (single source of truth) and
// dispatches a `change` event so the existing layout-view handler runs —
// no duplicate handler logic, just a relocated trigger.
function selectLayoutView(v) {
    const r = document.querySelector(`input[name="layout-view"][value="${v}"]`);
    if (r) {
        r.checked = true;
        r.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
        _applyLayoutView(v);
    }
    _refreshLayoutViewIndicator(v);
}
window.selectLayoutView = selectLayoutView;

let _lastSingleView = 'subjects';
function toggleSingleDashboard() {
    const cur = document.body.dataset.layout || 'default';
    if (cur !== 'default') {
        _lastSingleView = cur;
    }
    selectLayoutView(_lastSingleView);
}
window.toggleSingleDashboard = toggleSingleDashboard;

function _refreshLayoutViewIndicator(v) {
    document.querySelectorAll('[data-view-check]').forEach(el => {
        const check = el.getAttribute('data-view-check');
        let active = (check === v);
        if (check === 'single') {
            active = (v === 'subjects' || v === 'queue' || v === 'gallery');
        }
        el.classList.toggle('hidden', !active);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    // URL ?layout=<mode> wins over localStorage so a kiosk / wall-display
    // can pin a layout via its bookmark/PWA shortcut without depending on
    // per-browser-profile state. The valid modes are listed in
    // _applyLayoutView's `valid` set: default | subjects | queue | gallery
    // | subj-slop | queue-slop | subj-queue. The URL choice is also
    // persisted to localStorage so subsequent visits without the param
    // remember it (use ?layout=default to reset).
    const urlLayout = new URLSearchParams(location.search).get('layout');
    // Mobile auto-default: when the viewport is below the desktop breakpoint
    // (Tailwind md = 768 px) AND the user hasn't explicitly picked a layout
    // before, default to 'gallery' (slop-only) — the gallery is the most
    // useful single view on a phone, and Subjects/Queue are reachable via
    // edge-swipe gestures (see _wireMobileEdgeSwipes). On desktop we keep
    // the existing 'default' (3-pane) behaviour.
    const isMobile = window.matchMedia && window.matchMedia('(max-width: 767px)').matches;
    const storedRaw = (() => { try { return localStorage.getItem(_LAYOUT_VIEW_KEY); } catch (_) { return null; } })();
    const stored = storedRaw || (isMobile ? 'gallery' : 'default');
    const v = urlLayout || stored;
    _applyLayoutView(v);
    const r = document.querySelector(`input[name="layout-view"][value="${v}"]`);
    if (r) r.checked = true;
    document.querySelectorAll('input[name="layout-view"]').forEach(el => {
        el.addEventListener('change', e => {
            _applyLayoutView(e.target.value);
            _refreshLayoutViewIndicator(e.target.value);
            if (typeof _refreshCardVisibility === 'function') _refreshCardVisibility();
        });
    });
    _refreshLayoutViewIndicator(v);
    if (isMobile) _wireMobileEdgeSwipes();
});

// Mobile edge-swipe gestures — when the user is in gallery layout on a
// phone, a swipe from the LEFT edge slides the Subjects card in as a
// sheet; right edge slides Queue. Tap-outside or swipe-back dismisses.
// PointerEvent-based; no library dependency. Only wired on mobile to
// keep desktop drag-behaviour unchanged.
function _wireMobileEdgeSwipes() {
    const EDGE = 28;          // px from edge that counts as a swipe-start
    const TRIGGER = 60;       // px of horizontal travel before the sheet opens
    let startX = 0, startY = 0, startedFrom = null, currentSheet = null;
    const SHEETS = {
        left: { sel: '#split-left', cls: 'mobile-sheet-left' },
        right: { sel: '#split-right', cls: 'mobile-sheet-right' },
    };
    const closeAnyOpenSheet = () => {
        document.body.classList.remove('mobile-sheet-open-left', 'mobile-sheet-open-right');
        currentSheet = null;
    };
    document.addEventListener('pointerdown', (e) => {
        if (e.pointerType !== 'touch') return;
        // Outside-click closes an open sheet.
        if (currentSheet) {
            const cur = document.querySelector(SHEETS[currentSheet].sel);
            if (cur && !cur.contains(e.target)) {
                closeAnyOpenSheet();
                return;
            }
        }
        if (e.clientX <= EDGE) startedFrom = 'left';
        else if (e.clientX >= window.innerWidth - EDGE) startedFrom = 'right';
        else { startedFrom = null; return; }
        startX = e.clientX;
        startY = e.clientY;
    });
    document.addEventListener('pointermove', (e) => {
        if (!startedFrom || e.pointerType !== 'touch') return;
        const dx = e.clientX - startX;
        const dy = Math.abs(e.clientY - startY);
        if (dy > 40) { startedFrom = null; return; } // vertical drag — ignore
        const open = (startedFrom === 'left' && dx > TRIGGER)
            || (startedFrom === 'right' && -dx > TRIGGER);
        if (open) {
            currentSheet = startedFrom;
            document.body.classList.add('mobile-sheet-open-' + startedFrom);
            startedFrom = null; // single-shot
        }
    });
    document.addEventListener('pointerup', () => { startedFrom = null; });
    // Esc key dismisses too.
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && currentSheet) closeAnyOpenSheet();
    });
}
window._wireMobileEdgeSwipes = _wireMobileEdgeSwipes;

// ---------------------------------------------------------------------------
// Card close/restore — Subjects (#split-left), Queue (#split-right), and
// Slop output (#output-section) each get a (×) close button that hides the
// card and persists the choice in localStorage. When ALL THREE are hidden
// we surface a placeholder with three "restore" buttons. The placeholder
// is gallery-mode aware: `body[data-layout="gallery"]` already CSS-hides
// Subjects + Queue, but the user can still reach them via FABs — so the
// placeholder only flips on when the Slop card is closed too.
// ---------------------------------------------------------------------------
const _CARD_KEYS = {
    subjects: { dom: 'split-left', storage: 'slopfinity_card_subjects_hidden' },
    queue: { dom: 'split-right', storage: 'slopfinity_card_queue_hidden' },
    slop: { dom: 'output-section', storage: 'slopfinity_card_slop_hidden' },
};

function _isCardHidden(which) {
    try { return localStorage.getItem(_CARD_KEYS[which].storage) === '1'; }
    catch (_) { return false; }
}

function _setCardHidden(which, hidden) {
    try { localStorage.setItem(_CARD_KEYS[which].storage, hidden ? '1' : '0'); }
    catch (_) { }
    const el = document.getElementById(_CARD_KEYS[which].dom);
    if (el) {
        if (hidden) {
            el.style.display = 'none';
        } else {
            // The Slop output-section ships with an inline `display:block|none`
            // baked in by the server template (visibility depends on whether
            // there are assets). Force `block` here so a manual restore wins
            // — otherwise `removeProperty` could fall back to `display:none`
            // from the inline style and the user's click would appear to do
            // nothing. The card body will still render its own empty state.
            el.style.display = (which === 'slop') ? 'block' : '';
            if (which !== 'slop') el.style.removeProperty('display');
        }
    }
    _refreshCardVisibility();
}

function closeCard(which) { _setCardHidden(which, true); }
function restoreCard(which) { _setCardHidden(which, false); }

// ---------------------------------------------------------------------------
// Window-control close ✕ now drives LAYOUT switching instead of inline
// hide. "Hide subjects from this view" → flip to the layout that has
// the OTHER cards from the current view, sans subjects. Cleaner mental
// model + the user can always recover via the View dropdown OR by
// clicking close on a different card. The legacy closeCard()/restoreCard()
// pair is kept in place for any external callers but no template uses
// them anymore.
// ---------------------------------------------------------------------------
const _LAYOUT_TO_CARDS = {
    default: ['S', 'Q', 'Slop'],
    subjects: ['S'],
    queue: ['Q'],
    gallery: ['Slop'],
    'subj-slop': ['S', 'Slop'],
    'queue-slop': ['Q', 'Slop'],
    'subj-queue': ['S', 'Q'],
};
const _CARD_SET_TO_LAYOUT = {
    'S': 'subjects',
    'Q': 'queue',
    'Slop': 'gallery',
    'SQ': 'subj-queue',
    'SSlop': 'subj-slop',
    'QSlop': 'queue-slop',
    'SQSlop': 'default',
};
function _closeCardLayout(which) {
    const cardKey = which === 'subjects' ? 'S'
        : which === 'queue' ? 'Q'
            : which === 'slop' ? 'Slop'
                : null;
    if (!cardKey) return;
    const current = document.body.dataset.layout || 'default';
    const cards = _LAYOUT_TO_CARDS[current] || ['S', 'Q', 'Slop'];
    const remaining = cards.filter(c => c !== cardKey);
    // Closing the only visible card is a no-op — don't strand the user
    // on an empty layout. They should use View dropdown to switch away.
    if (!remaining.length) return;
    const target = _CARD_SET_TO_LAYOUT[remaining.join('')] || 'default';
    if (typeof selectLayoutView === 'function') selectLayoutView(target);
}
window._closeCardLayout = _closeCardLayout;

// Restore-FAB visibility toggle (Settings → General). Default ON.
// Persisted in localStorage; CSS rule .no-card-fabs hides every FAB.
const _SHOW_CARD_FABS_KEY = 'slopfinity-show-card-fabs';
function _isShowCardFabsOn() {
    try { const v = localStorage.getItem(_SHOW_CARD_FABS_KEY); return v === null ? true : v === '1'; }
    catch (_) { return true; }
}
function _setShowCardFabs(on) {
    try { localStorage.setItem(_SHOW_CARD_FABS_KEY, on ? '1' : '0'); } catch (_) { }
    document.body.classList.toggle('no-card-fabs', !on);
    const el = document.getElementById('settings-show-card-fabs');
    if (el && el.checked !== on) el.checked = on;
}
window._setShowCardFabs = _setShowCardFabs;
document.addEventListener('DOMContentLoaded', () => {
    _setShowCardFabs(_isShowCardFabsOn());
});

// Card window-controls (hover overlay close/restore/max chrome). User
// found these annoying — default OFF. Persisted in localStorage; CSS
// rule body.no-card-wm hides every .card-wm-bar.
const _CARD_WM_KEY = 'slopfinity-card-wm-enabled';
function _isCardWmOn() {
    try { const v = localStorage.getItem(_CARD_WM_KEY); return v === null ? false : v === '1'; }
    catch (_) { return false; }
}
function _setCardWm(on) {
    try { localStorage.setItem(_CARD_WM_KEY, on ? '1' : '0'); } catch (_) { }
    document.body.classList.toggle('no-card-wm', !on);
    const el = document.getElementById('settings-card-wm-toggle');
    if (el && el.checked !== on) el.checked = on;
}
window._setCardWm = _setCardWm;
document.addEventListener('DOMContentLoaded', () => {
    _setCardWm(_isCardWmOn());
    // Hydrate the celebrate-style dropdown (Settings → General) with the
    // saved choice so the user sees their previous selection on open.
    const cel = document.getElementById('settings-celebrate-style');
    if (cel && typeof _getCelebrateChoice === 'function') {
        cel.value = _getCelebrateChoice();
    }
});

// One-click "show every card" — clears all three hidden flags at once.
// Wired into Settings so users don't have to drop into DevTools when they
// accidentally close ✕ a card and forget how to bring it back.
function restoreAllCards() {
    Object.keys(_CARD_KEYS).forEach(which => _setCardHidden(which, false));
}
window.restoreAllCards = restoreAllCards;
window.closeCard = closeCard;
window.restoreCard = restoreCard;

function _refreshCardVisibility() {
    const ph = document.getElementById('cards-all-hidden-placeholder');
    const subjectsHidden = _isCardHidden('subjects');
    let queueHidden = _isCardHidden('queue');
    const slopHidden = _isCardHidden('slop');
    const galleryMode = document.body.dataset.layout === 'gallery';
    const isDefaultLayout = !document.body.dataset.layout || document.body.dataset.layout === 'default';

    // "default mode layout hides queue, but shows queue if queue exists"
    if (isDefaultLayout) {
        const qList = document.getElementById('q-list');
        const hasItems = qList && qList.querySelector('li[data-q-ts]');
        if (!hasItems) queueHidden = true;
    }
    if (ph) {
        // In gallery mode, Subjects + Queue are still reachable via FABs, so
        // the placeholder is only meaningful when Slop is also closed.
        const allGone = galleryMode ? slopHidden : (subjectsHidden && queueHidden && slopHidden);
        ph.classList.toggle('hidden', !allGone);
    }
    // Hide the vertical divider between Subjects and Queue when only one of
    // them is visible — there's nothing to drag-resize, and the visible
    // card should fill the upper pane horizontally instead of sitting in a
    // half-width column with empty space beside it.
    const vDivider = document.getElementById('split-divider');
    if (vDivider) {
        const oneSide = subjectsHidden !== queueHidden; // exactly one visible
        vDivider.classList.toggle('hidden', oneSide || (subjectsHidden && queueHidden));
        const left = document.getElementById('split-left');
        const right = document.getElementById('split-right');
        // When one side is hidden, drop the basis-1/2 cap on the other so it
        // can grow to fill the row. When both visible, restore equal split.
        if (left && right) {
            if (subjectsHidden && !queueHidden) {
                right.style.flexBasis = '100%';
                right.style.maxWidth = '100%';
            } else if (queueHidden && !subjectsHidden) {
                left.style.flexBasis = '100%';
                left.style.maxWidth = '100%';
            } else {
                left.style.flexBasis = '';
                left.style.maxWidth = '';
                right.style.flexBasis = '';
                right.style.maxWidth = '';
            }
        }
    }
    // Hide the horizontal divider (between upper [Subjects+Queue] and lower
    // [Slop]) when one side is empty — same logic, different axis.
    const hHandle = document.getElementById('ui-split-handle');
    if (hHandle) {
        const upperEmpty = subjectsHidden && queueHidden;
        const oneSide = upperEmpty !== slopHidden; // exactly one half-pane has cards
        hHandle.classList.toggle('hidden', oneSide || (upperEmpty && slopHidden));
    }
}
window._refreshCardVisibility = _refreshCardVisibility;

// Migration (post-Apr-26): the legacy slopfinity_card_*_hidden flags
// are obsolete — close ✕ now switches layouts via _closeCardLayout()
// rather than inline-hiding cards. Old flags from prior sessions would
// otherwise resurrect themselves on every page load and leave the user
// staring at an empty layout. Clear them on first load and skip the
// inline-hide application path entirely.
document.addEventListener('DOMContentLoaded', () => {
    Object.keys(_CARD_KEYS).forEach(which => {
        try { localStorage.removeItem(_CARD_KEYS[which].storage); } catch (_) { }
        const el = document.getElementById(_CARD_KEYS[which].dom);
        // Wipe any inline display:none a previous session may have stuck on.
        if (el && el.style.display === 'none') {
            if (which === 'slop') el.style.display = 'block';
            else el.style.removeProperty('display');
        }
    });
    _refreshCardVisibility();
});

// ---------------------------------------------------------------------------
// RAM-tight guard — wrapper that shows a confirmation modal before any
// manual AI button (🎲 Suggest, ✨ Enhance, fan-out, TTS preview) fires
// when GET /system/ram reports tight=true. Returns true iff the action
// should proceed. Fail-open if the endpoint is unreachable so we never
// strand the user without their manual buttons.
//
// This is a SOFT guard — it adds a confirm step but never auto-blocks.
// The deeper protection (queueing behind fleet GPU stages, suspending
// LM Studio etc.) lives in the server-side acquire_gpu wrap.
// ---------------------------------------------------------------------------
async function _ramGuardCheck() {
    let r;
    try {
        const resp = await fetch('/system/ram');
        r = await resp.json();
    } catch (_) {
        return true; // fail-open
    }
    if (!r || !r.tight) return true;
    return new Promise(resolve => {
        const modal = document.getElementById('ram-tight-modal');
        if (!modal) return resolve(true);
        const availEl = document.getElementById('ram-tight-available');
        const safeEl = document.getElementById('ram-tight-safety');
        if (availEl) availEl.textContent = Number(r.available_gb || 0).toFixed(1);
        if (safeEl) safeEl.textContent = Number(r.safety_gb || 0).toFixed(0);
        const btn = document.getElementById('ram-tight-proceed');
        let resolved = false;
        const finish = (val) => {
            if (resolved) return;
            resolved = true;
            btn && btn.removeEventListener('click', onProceed);
            modal.removeEventListener('close', onClose);
            resolve(val);
        };
        const onProceed = () => { try { modal.close(); } catch (_) { } finish(true); };
        const onClose = () => finish(false);
        btn && btn.addEventListener('click', onProceed, { once: true });
        modal.addEventListener('close', onClose, { once: true });
        try { modal.showModal(); } catch (_) { finish(true); }
    });
}
window._ramGuardCheck = _ramGuardCheck;

// ---------------------------------------------------------------------------
// Suggestions hidden-state — when the user clicks the × next to "Need ideas?"
// the chip area collapses, the 🎲 button hides, and every auto-fetch entry
// (tryAutoSuggest, _maybePrefetch, carousel right-overlay fresh-fetch) bails
// out early. The "Need ideas?" label becomes a clickable link to reveal.
// State persists in localStorage across reloads.
// ---------------------------------------------------------------------------
const _SUGGEST_HIDDEN_KEY = 'slopfinity_suggestions_hidden';

function _isSuggestionsHidden() {
    try { return localStorage.getItem(_SUGGEST_HIDDEN_KEY) === '1'; }
    catch (_) { return false; }
}

function toggleSuggestionsHidden(hide) {
    try { localStorage.setItem(_SUGGEST_HIDDEN_KEY, hide ? '1' : '0'); }
    catch (_) { }
    _applySuggestionsHiddenState();
}
window.toggleSuggestionsHidden = toggleSuggestionsHidden;

function _applySuggestionsHiddenState() {
    const hidden = _isSuggestionsHidden();
    const area = document.getElementById('subjects-suggestions-area');
    const closeBtn = document.getElementById('subjects-suggestions-close');
    const suggestBtn = document.getElementById('subjects-suggest-btn');
    const promptBtn = document.getElementById('subjects-suggestion-prompt-link');
    const toggleBtn = document.getElementById('subjects-suggestions-toggle');
    const toggleInput = document.getElementById('subjects-suggestions-toggle-input');
    const endlessBtn = document.getElementById('subjects-endless-story');
    if (area) area.style.display = hidden ? 'none' : '';
    if (closeBtn) closeBtn.style.display = hidden ? 'none' : '';
    // Regenerate + Suggest-Prompt buttons follow visibility of the suggestion
    // controls. Endless toggle now stays VISIBLE alongside Suggestions but
    // becomes disabled when Suggestions is off (handled by _updateEndlessEnabled).
    if (suggestBtn) suggestBtn.style.display = hidden ? 'none' : '';
    if (promptBtn) promptBtn.style.display = hidden ? 'none' : '';
    if (endlessBtn) endlessBtn.style.display = '';
    // Mirror state on the new switch input + the host's aria-pressed.
    if (toggleInput) toggleInput.checked = !hidden;
    if (toggleBtn) toggleBtn.setAttribute('aria-pressed', String(!hidden));
    if (typeof _updateEndlessEnabled === 'function') _updateEndlessEnabled();
}
document.addEventListener('DOMContentLoaded', _applySuggestionsHiddenState);

// Endless is a sub-toggle of Suggestions — meaningless when Suggestions
// is hidden because no marquee rows render at all. When Suggestions is
// off we disable the input, force-uncheck it, and dim the host so the
// dependency reads visually. Mirrors _updateTermEnabled's pattern.
function _updateEndlessEnabled() {
    const sug = document.getElementById('subjects-suggestions-toggle-input');
    const ends = document.getElementById('endless-story-toggle');
    const host = document.getElementById('subjects-endless-story');
    const enabled = !!(sug && sug.checked);
    if (ends) {
        ends.disabled = !enabled;
        if (!enabled && ends.checked) {
            ends.checked = false;
            ends.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
    if (host) {
        host.classList.toggle('opacity-50', !enabled);
        host.classList.toggle('pointer-events-none', !enabled);
    }
    // Mode-pill Endless button stays ENABLED regardless of Suggestions
    // toggle — picking endless is fine; what's gated is the actual
    // Start Story click. _updateSubjectsActionLabel handles the disable
    // on the big queue button (with the explanatory tooltip there).
    const endlessBtn = document.querySelector('.subjects-mode-pill button[data-subj-mode="endless"]');
    if (endlessBtn) {
        endlessBtn.disabled = false;
        endlessBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        endlessBtn.title = "Endless — seed a story (or use 'I'm Feeling Lucky') and the LLM auto-cycles continuation prompts.";
    }
    // If user is currently in endless mode AND a story is already running,
    // turning off suggestions kills the cycle (since there's nothing to
    // continue). Surface that by ending the story; mode stays endless.
    if (!enabled && _endlessRunning && typeof _endEndlessStory === 'function') {
        _endEndlessStory();
    }
    if (typeof _updateSubjectsActionLabel === 'function') _updateSubjectsActionLabel();
    // Suggestions toggle drives the + button's enabled state too — repaint
    // the badge so the disabled/opaque/tooltip flips immediately rather
    // than lingering until the next mode swap or refresh.
    if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
}
window._updateEndlessEnabled = _updateEndlessEnabled;
document.addEventListener('DOMContentLoaded', _updateEndlessEnabled);

// ---------------------------------------------------------------------------
// Named suggestion prompts — the registry powering the unified Suggestions
// badge + per-mode rendering. Loaded from /settings.suggest_prompts on
// DOMContentLoaded; cached on window so subsequent reads are sync.
// ---------------------------------------------------------------------------
const _DEFAULT_SUGGEST_PROMPT_ID_KEY = 'slopfinity-suggest-default-prompt-id';
const _ENDLESS_ROW_PROMPTS_KEY = 'slopfinity-endless-row-prompts';
// Hardcoded mirror of slopfinity/config.py DEFAULT_SUGGEST_PROMPTS — used as a
// fallback when /settings is served by a pre-bounce server that doesn't yet
// expose suggest_prompts. Without this the dropdown would render empty even
// though the user has the registry on disk.
const _SUGGEST_PROMPTS_FALLBACK = [
    { id: 'yes-and', title: 'Yes, and…', active: true, builtin: true },
    { id: 'plot-twist', title: 'Plot Twist', active: true, builtin: true },
    { id: 'concrete-detail', title: 'Concrete Detail', active: true, builtin: true },
    { id: 'cynic', title: "Cynic's Take", active: false, builtin: true },
    { id: 'wonder', title: 'Childlike Wonder', active: false, builtin: true },
];
let _suggestPromptsCache = null;
async function _loadSuggestPrompts() {
    if (_suggestPromptsCache && _suggestPromptsCache.length) return _suggestPromptsCache;
    try {
        const r = await fetch('/settings');
        const d = await r.json();
        if (Array.isArray(d.suggest_prompts) && d.suggest_prompts.length) {
            _suggestPromptsCache = d.suggest_prompts;
        } else {
            _suggestPromptsCache = _SUGGEST_PROMPTS_FALLBACK;
        }
    } catch (_) {
        _suggestPromptsCache = _SUGGEST_PROMPTS_FALLBACK;
    }
    return _suggestPromptsCache;
}
window._loadSuggestPrompts = _loadSuggestPrompts;
function _getActivePrompts() {
    return (_suggestPromptsCache || []).filter(p => p && p.active);
}
function _getPromptById(id) {
    return (_suggestPromptsCache || []).find(p => p && p.id === id) || null;
}
function _getDefaultPromptId() {
    try {
        const v = localStorage.getItem(_DEFAULT_SUGGEST_PROMPT_ID_KEY);
        if (v && _getPromptById(v)) return v;
    } catch (_) { }
    // First-active fallback so a fresh session doesn't render an empty badge.
    const first = _getActivePrompts()[0];
    return first ? first.id : 'yes-and';
}
function _setDefaultPromptId(id) {
    const prev = _getDefaultPromptId();
    try { localStorage.setItem(_DEFAULT_SUGGEST_PROMPT_ID_KEY, id); } catch (_) { }
    // Prompt swap → every prefetched batch was generated under the OLD
    // prompt and is now stale. Dump the buffer so a subsequent + click
    // doesn't surprise the user with off-topic suggestions, then kick
    // a fresh prefetch under the new prompt so the NEW + click can
    // still hit the instant-render fast path.
    if (id !== prev) {
        if (typeof window._dropPrefetchedBatches === 'function') window._dropPrefetchedBatches();
        if (typeof window._maybePrefetch === 'function') window._maybePrefetch();
    }
    _refreshSuggestBadge();
    // Endless mode treats the dropdown as "the prompt the NEXT added row
    // will use" — existing rows have their own per-row prompts and should
    // NOT be wiped. Other modes (simple/chat) regenerate to reflect the
    // new default immediately.
    const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
    if (mode === 'endless') return;
    if (typeof _spinRefreshBriefly === 'function') _spinRefreshBriefly(1500);
    if (typeof regenSuggestions === 'function') regenSuggestions().catch(() => { });
}
window._getDefaultPromptId = _getDefaultPromptId;
window._setDefaultPromptId = _setDefaultPromptId;

// Endless mode tracks ONE prompt_id per row so each marquee line can
// represent a different angle on the seed. Persisted in localStorage as
// an array of ids; first run defaults to all-active prompts.
function _getEndlessRowPrompts() {
    try {
        const raw = localStorage.getItem(_ENDLESS_ROW_PROMPTS_KEY);
        // Key set at all = user has explicit state we should respect
        // (including the empty-array case after − removed every row).
        // Only fall through to defaults when the key has never been
        // touched. Filter to resolvable ids but otherwise trust the
        // saved value. The previous "diversity check" (require
        // unique.size > 1) silently swapped the saved array for the
        // 4-prompt default whenever it collapsed to a single repeated
        // id — which broke Start-Story-saves-[defaultId], + click-
        // saves-[same,same], and the explicit-empty-after-remove case.
        if (raw !== null) {
            const arr = JSON.parse(raw);
            if (Array.isArray(arr)) {
                return arr.filter(id => typeof id === 'string' && _getPromptById(id));
            }
        }
    } catch (_) { }
    // Build the default endless-row sequence: the user's currently-selected
    // default prompt FIRST (so what the unified badge shows is what the
    // first row renders with), followed by the other active prompts.
    const defaultId = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : null;
    const orderActive = (list) => {
        if (!defaultId) return list;
        const head = list.filter(p => p.id === defaultId);
        const tail = list.filter(p => p.id !== defaultId);
        return head.concat(tail);
    };
    const active = orderActive(_getActivePrompts());
    if (active.length) return active.slice(0, Math.min(4, active.length)).map(p => p.id);
    // No cache loaded yet → fall through to the client-side fallback list
    // so endless mode never renders "all same prompt" on first paint.
    return orderActive(_SUGGEST_PROMPTS_FALLBACK.filter(p => p.active)).slice(0, 3).map(p => p.id);
}
function _setEndlessRowPrompts(arr) {
    const clean = (Array.isArray(arr) ? arr : []).filter(id => typeof id === 'string' && _getPromptById(id));
    try { localStorage.setItem(_ENDLESS_ROW_PROMPTS_KEY, JSON.stringify(clean)); } catch (_) { }
}
window._getEndlessRowPrompts = _getEndlessRowPrompts;
window._setEndlessRowPrompts = _setEndlessRowPrompts;

function _refreshSuggestBadge() {
    const sugInput = document.getElementById('subjects-suggestions-toggle-input');
    const isOn = !!(sugInput && sugInput.checked);
    const nameBtn = document.getElementById('subjects-suggest-prompt-name');
    const refreshBtn = document.getElementById('subjects-suggest-btn');
    const lbl = document.getElementById('subjects-suggest-prompt-name-label');
    // The prompt-name button hides when Suggestions toggle is OFF — the
    // dropdown picker has no purpose without a chip stack to feed.
    // Settings → Prompts is still the canonical edit surface for the
    // registry, so users who want to manage prompts pre-toggle-on go
    // there directly. body.suggest-cluster-collapsed lets CSS round
    // the toggle's right edge so it visually stands alone.
    const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
    const isEndless = mode === 'endless';
    if (nameBtn) {
        // Endless mode: keep the prompt-name dropdown visible regardless
        // of the Suggestions toggle state — the user wants to see/swap
        // the prompt for upcoming rows even when auto-suggest is paused.
        // Other modes still hide the dropdown when the toggle is off
        // (no chip stack to feed = no point in showing the picker).
        nameBtn.classList.toggle('hidden', !isOn && !isEndless);
        nameBtn.title = "Active suggestion prompt — click to swap";
    }
    // suggest-cluster-collapsed marks the toggle as standalone (rounds
    // both edges via CSS). In endless mode the prompt-name button stays
    // visible regardless, so the toggle is NOT standalone — skip the class.
    document.body.classList.toggle('suggest-cluster-collapsed', !isOn && !isEndless);
    // The body.endless-pill-locked dimmer was REMOVED — earlier iterations
    // dimmed the whole row pre-Start-Story, but the user explicitly asked
    // for the prompt-name and + button to NEVER look greyed out. The +
    // button's enabled-state below now keys ONLY on storyRunning in
    // endless mode (not on the Suggestions toggle), so the visual
    // affordance stays clean. Class removal is idempotent — safe to call
    // even when the class isn't present.
    document.body.classList.remove('endless-pill-locked');
    // Swap which action button is visible based on mode. Two distinct
    // buttons — refresh (#subjects-suggest-btn) and add (#subjects-suggest-add-btn)
    // — share the joined badge slot so the refresh-tap-spin handler only
    // ever matches the actual refresh control.
    const addBtn = document.getElementById('subjects-suggest-add-btn');
    // (mode + isEndless captured above for the prompt-name visibility
    // decision — reuse them rather than redeclare.)
    // Simple mode UX: pre-first-batch, hide ↻ refresh and show + add
    // (which fires regenSuggestions). Once any chip rows exist, swap
    // back to ↻ refresh (regenSuggestions overwrites the stack). User
    // never sees both at once. Endless owns the + permanently. Other
    // modes (chat/raw) don't apply.
    const { stack } = _getSuggestStack();
    const stackHasRows = !!(stack && stack.querySelector('.suggest-marquee-row'));
    const simpleNeedsAdd = (mode === 'simple') && !stackHasRows;
    // Simple mode +/- row-stack pill — visible only in simple mode WITH
    // existing rows (you can't − a row that doesn't exist; the
    // first-batch + bootstrap is the dedicated badge button).
    const rowCtl = document.getElementById('subjects-simple-rowctl');
    if (rowCtl) {
        const showRowCtl = (mode === 'simple') && stackHasRows && isOn;
        rowCtl.classList.toggle('hidden', !showRowCtl);
        const removeBtn = document.getElementById('subjects-simple-remove-row');
        if (removeBtn) {
            removeBtn.disabled = !stackHasRows;
            removeBtn.classList.toggle('opacity-40', !stackHasRows);
        }
        const countEl = document.getElementById('subjects-simple-row-count');
        if (countEl && stack) {
            countEl.innerText = stack.querySelectorAll('.suggest-marquee-row').length;
        }
    }
    if (refreshBtn) {
        // Hide refresh in: endless (+ owns the slot), simple-pre-first-batch.
        refreshBtn.classList.toggle('hidden', isEndless || simpleNeedsAdd);
        refreshBtn.title = isOn
            ? 'Regenerate suggestions for the active prompt'
            : 'Suggestions are off — click to enable + fetch a fresh batch';
    }
    if (addBtn) {
        // Show + in: endless, simple-pre-first-batch. Hide otherwise.
        addBtn.classList.toggle('hidden', !(isEndless || simpleNeedsAdd));
        // Enable rules:
        //   endless: ALWAYS enabled — the user wants the + to never look
        //            disappeared/dimmed while endless mode is selected.
        //            _addEndlessRow is no-op-safe before Start Story
        //            (just appends to the saved row prompts; renders when
        //            the story actually starts).
        //   simple:  Suggestions ON (no story concept)
        let allow = isOn;
        let titleText = '';
        if (isEndless) {
            allow = true;  // never gate in endless — see comment above
            titleText = 'Add another story-beat row using the default prompt';
        } else if (simpleNeedsAdd) {
            // Simple mode pre-first-batch — + bootstraps the initial
            // suggestion stack. Badge swaps to ↻ refresh once the
            // first batch lands.
            titleText = isOn
                ? 'Click to fetch your first batch of suggestions'
                : 'Suggestions are off — click to enable + fetch a batch';
        }
        addBtn.disabled = !allow;
        addBtn.classList.toggle('opacity-40', !allow);
        addBtn.classList.toggle('cursor-not-allowed', !allow);
        addBtn.title = titleText;
    }
    if (lbl) {
        const p = _getPromptById(_getDefaultPromptId());
        // Cache may not be loaded yet on first paint — show the canonical
        // built-in default (matches DEFAULT_SUGGEST_PROMPTS[0] in config.py)
        // rather than a placeholder, so the badge always reads as a real
        // option even before /settings has resolved.
        lbl.textContent = p ? p.title : 'Yes, and…';
    }
}
window._refreshSuggestBadge = _refreshSuggestBadge;

// Popover picker — lists every active prompt; selecting one updates either
// the global default OR the specific endless row (when invoked from there).
// `targetRowIdx` is null for the unified badge / non-endless modes.
let _pickerActiveTargetRowIdx = null;
function _openSuggestPromptPicker(targetRowIdx) {
    _pickerActiveTargetRowIdx = (typeof targetRowIdx === 'number') ? targetRowIdx : null;
    const popover = document.getElementById('subjects-suggest-prompt-popover');
    if (!popover) return;
    const anchor = (targetRowIdx != null)
        ? document.querySelector(`[data-endless-row-prompt-btn="${targetRowIdx}"]`)
        : document.getElementById('subjects-suggest-prompt-name');
    const active = _getActivePrompts();
    const currentId = (targetRowIdx != null)
        ? (_getEndlessRowPrompts()[targetRowIdx] || _getDefaultPromptId())
        : _getDefaultPromptId();
    popover.innerHTML = active.map(p => {
        const sel = p.id === currentId ? 'bg-base-300/60' : '';
        return `<button type="button" class="w-full text-left px-2 py-1 rounded ${sel} hover:bg-base-300/40"
            onclick="_pickPromptForActiveTarget('${p.id}')">
            <span class="font-semibold">${_htmlEscape(p.title)}</span>
        </button>`;
    }).join('') + `
        <div class="border-t border-base-300/60 my-1"></div>
        <button type="button" class="w-full text-left px-2 py-1 rounded hover:bg-base-300/40 text-base-content/60"
            onclick="document.getElementById('subjects-suggest-prompt-popover').classList.add('hidden'); openSettings(); setTimeout(() => { const el = document.querySelector('#settings-modal input[type=radio][aria-label=&quot;LLM&quot;]'); if (el) { el.checked = true; el.dispatchEvent(new Event('change', {bubbles:true})); } }, 200);">
            ✎ Manage prompts in Settings
        </button>`;
    if (anchor) {
        const r = anchor.getBoundingClientRect();
        popover.style.position = 'fixed';
        // Position the popover so it does NOT overlap the chip stack
        // below the badge — measure available space and flip up when
        // there isn't enough room below the anchor.
        popover.style.visibility = 'hidden';
        popover.classList.remove('hidden');
        const popH = popover.offsetHeight || 200;
        popover.classList.add('hidden');
        popover.style.visibility = '';
        const spaceBelow = window.innerHeight - r.bottom;
        if (spaceBelow < popH + 16) {
            // Not enough room below — open ABOVE the anchor (right-aligned).
            popover.style.left = r.left + 'px';
            popover.style.top = Math.max(8, r.top - popH - 4) + 'px';
        } else {
            popover.style.left = r.left + 'px';
            popover.style.top = (r.bottom + 4) + 'px';
        }
    }
    popover.classList.remove('hidden');
    // Dismiss on outside click.
    setTimeout(() => {
        const onDocClick = (e) => {
            if (!popover.contains(e.target) && e.target !== anchor) {
                popover.classList.add('hidden');
                document.removeEventListener('click', onDocClick);
            }
        };
        document.addEventListener('click', onDocClick);
    }, 0);
}
window._openSuggestPromptPicker = _openSuggestPromptPicker;

function _pickPromptForActiveTarget(id) {
    const popover = document.getElementById('subjects-suggest-prompt-popover');
    if (popover) popover.classList.add('hidden');
    if (_pickerActiveTargetRowIdx != null) {
        const arr = _getEndlessRowPrompts();
        while (arr.length <= _pickerActiveTargetRowIdx) arr.push(_getDefaultPromptId());
        arr[_pickerActiveTargetRowIdx] = id;
        _setEndlessRowPrompts(arr);
        if (typeof _regenEndlessRow === 'function') _regenEndlessRow(_pickerActiveTargetRowIdx);
    } else {
        _setDefaultPromptId(id);
    }
    _pickerActiveTargetRowIdx = null;
}
window._pickPromptForActiveTarget = _pickPromptForActiveTarget;

document.addEventListener('DOMContentLoaded', async () => {
    await _loadSuggestPrompts();
    _refreshSuggestBadge();
});

// LLM-availability gate for Subjects modes. Endless / Simple / Chat all
// need a reachable LLM; Raw is the only mode that works without one
// (per-stage prompts pre-filled OR queued verbatim with no rewrite).
// Pings /llm/health on load and every 60s — quick probe, low overhead.
let _llmAvailableCached = null;
async function _updateLlmAvailability() {
    // /llm/health is a passive indicator only. The pipeline auto-suspends
    // the LLM (SIGSTOP) during GPU-heavy stages, so 'unreachable' is the
    // NORMAL state mid-iter — disabling LLM-using modes during suspend
    // would lock users out for 90 % of the runtime. We just track it on
    // body.llm-down so CSS / future indicators can react if wanted; the
    // mode buttons stay enabled regardless.
    let ok = true;
    try {
        const r = await fetch('/llm/health');
        if (r.status === 404) {
            ok = true;
        } else {
            const d = await r.json();
            ok = !!(d && d.ok);
        }
    } catch (_) { ok = true; }
    _llmAvailableCached = ok;
    document.body.classList.toggle('llm-down', !ok);
    // Make sure no leftover disable from earlier builds is sticky.
    ['endless', 'simple', 'chat'].forEach(mode => {
        const btn = document.querySelector(`.subjects-mode-pill button[data-subj-mode="${mode}"]`);
        if (!btn) return;
        btn.disabled = false;
        btn.classList.remove('opacity-50', 'cursor-not-allowed');
        const orig = btn.getAttribute('data-orig-title');
        if (orig) {
            btn.title = orig;
            btn.removeAttribute('data-orig-title');
        }
    });
    if (typeof _updateEndlessEnabled === 'function') _updateEndlessEnabled();
}
window._updateLlmAvailability = _updateLlmAvailability;
document.addEventListener('DOMContentLoaded', () => {
    _updateLlmAvailability();
    setInterval(_updateLlmAvailability, 60_000);
});

// Queue pause/resume — soft-stops new iter starts via pause.flag.
// Three-state visual:
//   running  → button label "Pause", pill hidden (isRendering hides it)
//   pausing  → button label "Pausing…" (disabled), pill "⏸ Pausing…"
//   paused   → button label "Resume",  pill "⏸ Paused"
// _pausePending is set on click and cleared by the first WS tick that
// confirms scheduler.paused === true, or by the 5s poll.
let _pausePending = false;

function _applyPauseButtonState(paused) {
    const btn = document.getElementById('btn-queue-pause');
    if (!btn) return;
    const lbl = document.getElementById('btn-queue-pause-label');
    const icon = document.getElementById('btn-queue-pause-icon');
    if (_pausePending && !paused) {
        // Intermediate: sent pause request, waiting for backend confirmation.
        if (lbl) lbl.textContent = 'Pausing…';
        btn.disabled = true;
        btn.classList.add('text-warning');
        btn.title = 'Waiting for active stage to finish before pausing…';
        if (icon) icon.innerHTML = '<circle cx="12" cy="12" r="9" stroke-dasharray="3 3"/>';
        return;
    }
    // Clear pending once paused is confirmed.
    if (paused) _pausePending = false;
    btn.disabled = false;
    if (lbl) lbl.textContent = paused ? 'Resume' : 'Pause';
    btn.classList.toggle('text-warning', paused);
    btn.title = paused
        ? 'Resume — clear pause.flag so new iters can start'
        : 'Pause new iter starts (active iter finishes)';
    if (icon) {
        icon.innerHTML = paused
            ? '<polygon points="5 3 19 12 5 21 5 3"/>'
            : '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
    }
}

async function _refreshPauseButton() {
    try {
        const r = await fetch('/queue/pause-state');
        if (r.status === 404) return;
        const d = await r.json();
        _applyPauseButtonState(!!(d && d.paused));
    } catch (_) { /* fail silently */ }
}

async function _toggleQueuePause() {
    try {
        const cur = await fetch('/queue/pause-state').then(r => r.json()).catch(() => ({ paused: false }));
        const goingPause = !cur.paused;
        if (goingPause) {
            // Optimistic UI: show pausing state immediately.
            _pausePending = true;
            _applyPauseButtonState(false);
        }
        const next = goingPause ? 'pause' : 'resume';
        await fetch('/queue/' + next, { method: 'POST' });
        if (!goingPause) {
            // Resuming — clear pending and refresh immediately.
            _pausePending = false;
            await _refreshPauseButton();
        }
        // On pause: _refreshPauseButton / WS tick will resolve pending state.
    } catch (e) {
        _pausePending = false;
        if (typeof _seedToast === 'function') _seedToast(`Pause toggle failed: ${e.message}`, 'error');
        await _refreshPauseButton();
    }
}
window._refreshPauseButton = _refreshPauseButton;
window._toggleQueuePause = _toggleQueuePause;
document.addEventListener('DOMContentLoaded', () => {
    _refreshPauseButton();
    setInterval(_refreshPauseButton, 5_000);
});

// Disk-guard UI — polls /disk/guard, toggles the warning banner beneath
// the Queue button + disables the Queue button when blocked. Click on the
// warning banner opens Settings → General → Disk guard so the user can
// raise the threshold if the reading is wrong. Fail-open on 404 (the
// endpoint is new; older servers shouldn't lock users out).
async function _refreshDiskGuardUI() {
    const warn = document.getElementById('subjects-disk-warn');
    const detail = document.getElementById('subjects-disk-warn-detail');
    const queueBtn = document.getElementById('btn-start-stop-inline');
    if (!warn) return;
    let ok = true;
    let info = null;
    try {
        const r = await fetch('/disk/guard');
        if (r.status === 404) { ok = true; }
        else {
            info = await r.json();
            ok = !!(info && info.ok);
        }
    } catch (_) { ok = true; /* fail open */ }
    if (ok) {
        warn.classList.add('hidden');
        if (queueBtn) {
            queueBtn.disabled = false;
            queueBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
        return;
    }
    warn.classList.remove('hidden');
    if (detail && info) {
        detail.textContent = ` ${info.free_gb} GB / ${info.free_pct}% free (threshold: ${info.threshold_gb} GB or ${info.threshold_pct}%).`;
    }
    if (queueBtn) {
        queueBtn.disabled = true;
        queueBtn.classList.add('opacity-50', 'cursor-not-allowed');
        queueBtn.title = 'Disk-low — adjust threshold in Settings → General → Disk guard';
    }
}
window._refreshDiskGuardUI = _refreshDiskGuardUI;
document.addEventListener('DOMContentLoaded', () => {
    _refreshDiskGuardUI();
    setInterval(_refreshDiskGuardUI, 30_000);
});

// PWA: register service worker (scoped to /) for installable desktop-icon experience.
// Also detect when a new SW takes control (cache version bumped) and surface
// a toast inviting the user to reload — they don't have to, but the UI may be
// stale until they do.
let _newVersionShown = false;
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker
            .register('/sw.js', { scope: '/' })
            .then((reg) => {
                // Fires when an updated SW finishes installing AND is activating
                // (sw.js calls skipWaiting() so it claims control immediately).
                navigator.serviceWorker.addEventListener('controllerchange', () => {
                    if (_newVersionShown) return;
                    _newVersionShown = true;
                    _showNewVersionToast();
                });
            })
            .catch((err) => console.warn('SW registration failed:', err));
    });
}

function _showNewVersionToast() {
    if (document.getElementById('new-version-toast')) return;
    // (Note: the alert class for THIS toast is set inline below — keep it
    // aligned with the themed-primary used by _seedToast's default kind.)
    const t = document.createElement('div');
    t.id = 'new-version-toast';
    t.className = 'toast toast-end z-50';
    t.innerHTML = `
        <div class="alert toast-themed-primary shadow-lg flex items-center gap-3 max-w-sm">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 flex-none" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            <div class="flex-1">
                <div class="font-semibold text-sm">New version available</div>
                <div class="text-[11px] opacity-80">Refresh to load the latest dashboard.</div>
            </div>
            <button type="button" class="btn btn-sm btn-primary flex-none" onclick="location.reload()">Refresh</button>
            <button type="button" class="btn btn-sm btn-ghost btn-circle flex-none" aria-label="Dismiss" onclick="this.closest('#new-version-toast').remove()">✕</button>
        </div>
    `;
    document.body.appendChild(t);
}

let ws;
let gH = Array(15).fill(0);
let vH = Array(15).fill(0);
let dH = [];
let lH = Array(15).fill(0);
let _diskTickCounter = 0;
let _stageStartTs = null;
let _lastStage = null;
let _jobStartTs = null;
let _lastJobIndex = null;
let _wsConnected = false;
// Per-job per-stage actuals — populated as we observe step transitions.
// Format: { <video_index>: { <stage>: { duration_s: 12.4, eta_s: 24 } } }
// Reset when video_index changes (new job).
let _jobActuals = {};

// Tracks which (v_idx, stage) completion badges have already been shown so
// renderPipelineStrip only runs the cross-fade animation once per stage flip
// (subsequent re-renders skip the .stage-just-completed class). Cleared per
// fresh page load — staleness across reloads is fine, the badge stays put.
const _displayedDoneStages = new Set();
// Track which done-queue item is expanded so the WS state tick (which
// re-renders the queue list wholesale via innerHTML) doesn't clobber the
// user's open <details>. Single string = single-open enforcement: opening
// item B implicitly collapses item A.
let _openDoneItem = null;

// Rolling history of recent stage durations (per stage name) persisted in
// localStorage so ETAs survive page reloads. Format:
//   { "Concept": [12.4, 9.8, 11.2], "Base Image": [...], ... }
// v2 bumps the key so cold defaults (below) take effect for everyone — v1
// had stale ~5s samples from failed iters that polluted the rolling avg
// and made image-gen ETAs read 44s for a 4-minute job.
const _STAGE_HISTORY_KEY = 'slopfinity_stage_durations_v2';
const _STAGE_HISTORY_KEEP = 5;
function _loadStageHistory() {
    try { return JSON.parse(localStorage.getItem(_STAGE_HISTORY_KEY) || '{}') || {}; }
    catch { return {}; }
}
function _saveStageDuration(stage, seconds) {
    if (!stage || !isFinite(seconds) || seconds <= 0) return;
    const hist = _loadStageHistory();
    const arr = Array.isArray(hist[stage]) ? hist[stage] : [];
    arr.push(seconds);
    while (arr.length > _STAGE_HISTORY_KEEP) arr.shift();
    hist[stage] = arr;
    try { localStorage.setItem(_STAGE_HISTORY_KEY, JSON.stringify(hist)); } catch { }
}
// Conservative defaults — tripled from the original observed-warm-run
// numbers because real cold starts on Strix Halo (model load + 8 denoise
// steps + VAE) routinely hit the upper end. ETAs should over-estimate
// until we accumulate enough samples to trust them.
const _STAGE_DEFAULT_SECONDS = {
    'Concept': 24,
    'Base Image': 540,
    'Video Chains': 1800,
    'Audio': 180,
    'TTS': 60,
    'Post Process': 180,
    'Final Merge': 60,
};
// Until the rolling history has at least this many samples, lean on the
// default — small samples shouldn't yank the displayed ETA around. Once
// we have ≥3 real measurements the rolling average takes over.
const _STAGE_TRUST_AT = 3;
function _stageAvgSeconds(stage) {
    const arr = _loadStageHistory()[stage];
    const dflt = _STAGE_DEFAULT_SECONDS[stage] ?? null;
    if (!arr || arr.length === 0) return dflt;
    if (arr.length < _STAGE_TRUST_AT && dflt != null) {
        // Blend: weight the default by (TRUST_AT - n) and the samples by n.
        const sampleAvg = arr.reduce((a, b) => a + b, 0) / arr.length;
        return ((dflt * (_STAGE_TRUST_AT - arr.length)) + (sampleAvg * arr.length)) / _STAGE_TRUST_AT;
    }
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}
// Update the small ≈Ns hint next to each pipeline step + the total ETA badge.
function _renderStageEtas() {
    document.querySelectorAll('#stage-steps li[data-stage]').forEach(li => {
        const avg = _stageAvgSeconds(li.dataset.stage);
        let hint = li.querySelector('.stage-eta');
        if (avg == null) { if (hint) hint.remove(); return; }
        if (!hint) {
            hint = document.createElement('span');
            hint.className = 'stage-eta ml-1 opacity-50 text-[9px] font-mono';
            li.appendChild(hint);
        }
        hint.innerHTML = '≈' + _fmtElapsedHtml(avg * 1000);
    });
    // Total ETA badge near the timers — sum of all stages' rolling averages.
    const total = ['Concept', 'Base Image', 'Video Chains', 'Audio', 'TTS', 'Post Process', 'Final Merge']
        .map(_stageAvgSeconds)
        .filter(x => x != null)
        .reduce((a, b) => a + b, 0);
    const eta = document.getElementById('h-c-eta');
    if (eta) eta.innerHTML = total > 0 ? 'ETA ' + _fmtElapsedHtml(total * 1000) : '';
}

function _fmtElapsed(ms) {
    const elapsed = Math.max(0, Math.floor(ms / 1000));
    if (elapsed >= 3600) {
        const h = Math.floor(elapsed / 3600);
        const m = Math.floor((elapsed % 3600) / 60);
        return `${h}h${String(m).padStart(2, '0')}m`;
    }
    const m = Math.floor(elapsed / 60);
    const s = elapsed % 60;
    return m ? `${m}m${String(s).padStart(2, '0')}s` : `${s}s`;
}

// Same shape as _fmtElapsed but returns HTML with letter-units wrapped in
// <span class="time-unit"> so CSS can dim them. Safe to insert via innerHTML
// (no user input — output is fully derived from a numeric ms argument).
function _fmtElapsedHtml(ms) {
    return _fmtElapsed(ms).replace(/([hms]+)/g, '<span class="time-unit">$1</span>');
}

// Compact 1-or-2-unit duration label. Used in the per-stage output
// reveal so each row's right-edge time chip stays narrow but exact.
//   < 60s            → "Xs"          (45s)
//   60s – 59m59s     → "XmYs" or "Xm" when seconds are 0
//   ≥ 1h             → "XhYm" or "Xh" when minutes are 0
// Total seconds is ceil()'d so the label never under-reports duration —
// matches the prior "rounded up" intent of the chip while letting two
// units show through when the leftover is non-zero.
function _fmtRoundUp(ms) {
    if (ms == null) return '';
    if (ms < 0) ms = 0;
    const totalSec = Math.ceil(ms / 1000);
    if (totalSec < 60) return `${totalSec}s`;
    if (totalSec < 3600) {
        const m = Math.floor(totalSec / 60);
        const s = totalSec - m * 60;
        return s > 0 ? `${m}m${s}s` : `${m}m`;
    }
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec - h * 3600) / 60);
    return m > 0 ? `${h}h${m}m` : `${h}h`;
}

// ----- Done queue items: thumbnail / mini-player / asset link expanded card.
// The summary row stays compact (badge · prompt · duration · asset count).
// Expanding the <details> reveals a join-vertical list of asset previews:
//   image  → 96x54 <img>
//   video  → <video preload=metadata> with poster auto-grabbed by browser
//   audio  → <audio controls>
//   other  → just the link
// Each asset row also gets [Open] / [Download] links and a truncated filename.
function _truncMiddle(s, max) {
    if (!s || s.length <= max) return s || '';
    const head = Math.ceil((max - 1) / 2);
    const tail = Math.floor((max - 1) / 2);
    return s.slice(0, head) + '…' + s.slice(-tail);
}

function _doneAssetKind(name) {
    const n = (name || '').toLowerCase();
    if (n.endsWith('.mp4') || n.endsWith('.webm') || n.endsWith('.mov')) return 'video';
    if (n.endsWith('.wav') || n.endsWith('.mp3') || n.endsWith('.ogg') || n.endsWith('.flac')) return 'audio';
    if (n.endsWith('.png') || n.endsWith('.jpg') || n.endsWith('.jpeg') || n.endsWith('.webp') || n.endsWith('.gif')) return 'image';
    return 'other';
}

function _renderDoneAssetRow(filename) {
    const url = `/files/${encodeURIComponent(filename)}`;
    const kind = _doneAssetKind(filename);
    const safe = _htmlEscape(filename);
    const trunc = _htmlEscape(_truncMiddle(filename, 40));
    let preview = '';
    if (kind === 'image') {
        preview = `<a href="${url}" target="_blank" rel="noopener" class="block flex-none"><img src="${url}" alt="" class="rounded bg-black object-cover" style="width:96px;height:54px;" loading="lazy"></a>`;
    } else if (kind === 'video') {
        // preload=metadata coaxes the browser to grab the first frame as a poster.
        preview = `<a href="${url}" target="_blank" rel="noopener" class="block flex-none relative" style="width:96px;height:54px;"><video src="${url}" preload="metadata" muted playsinline class="rounded bg-black w-full h-full object-cover"></video><span class="absolute inset-0 flex items-center justify-center pointer-events-none text-white text-lg drop-shadow">▶</span></a>`;
    } else if (kind === 'audio') {
        preview = `<div class="flex-none" style="width:96px;"><audio controls preload="none" class="w-full h-8"><source src="${url}"></audio></div>`;
    } else {
        preview = `<div class="flex-none flex items-center justify-center bg-base-300 rounded font-mono text-[10px] text-base-content/60" style="width:96px;height:54px;">file</div>`;
    }
    const onClickInfo = `onclick='openAssetInfo(${JSON.stringify(filename)})'`;
    return `<div class="join-item card card-compact bg-base-200 p-2">
        <div class="flex items-start gap-2">
            ${preview}
            <div class="flex flex-col gap-1 min-w-0 flex-1">
                <span class="font-mono text-[10px] truncate cursor-pointer" title="${safe}" ${onClickInfo}>${trunc}</span>
                <div class="flex items-center gap-1 flex-wrap">
                    <a href="${url}" target="_blank" rel="noopener" class="btn btn-ghost btn-xs px-1 h-5 min-h-0">Open</a>
                    <a href="${url}" download class="btn btn-ghost btn-xs px-1 h-5 min-h-0">Download</a>
                </div>
            </div>
        </div>
    </div>`;
}

function _renderDoneItem(q) {
    const dur = q.duration_s ? _fmtElapsedHtml(q.duration_s * 1000) : '';
    const failed = q.succeeded === false;
    // Failed verdict uses badge-error (DaisyUI --er token) — same colour
    // contract as the ticker's maxed-out columns (.ticker-col.bg-error) and
    // the active progress bar's failed state (.slop-progress-failed,
    // hsl(var(--er))). One token = one "this is bad" signal across the UI.
    const cls = failed ? 'badge-error' : 'badge-success';
    const sym = failed ? '✗' : '✓';
    const verdict = failed ? 'failed' : 'done';
    const promptEsc = _htmlEscape(q.prompt || '');
    // Backwards-compat: pre-asset-tracking done records only have v_idx /
    // image_only. Synthesize a best-guess single-asset list from that so old
    // history items still render a thumbnail.
    let assets = Array.isArray(q.assets) ? q.assets.filter(Boolean) : [];
    if (!assets.length) {
        const v = q.v_idx || 0;
        if (v) {
            // Prefer real on-disk filenames from the resolver cache; fall
            // back to the legacy synthesized names so the row still renders
            // a thumbnail even when the cache is cold (e.g. immediately
            // after page load before the WS / SSR ingestion populates).
            const cached = _assetsByVidx.get(v) || {};
            if (q.image_only) {
                assets = [cached.base || `slop_${v}_base.png`];
            } else {
                assets = [
                    cached.final || `FINAL_${v}.mp4`,
                    cached.base || `slop_${v}_base.png`,
                ];
            }
            // Trigger an async resolve so the next WS re-render (~1Hz)
            // picks up the real names if either slot was a synthesised
            // fallback.
            if (!cached.base || (!q.image_only && !cached.final)) {
                _resolveVidxAssets(v);
            }
        }
    }
    const tsHuman = q.completed_ts
        ? new Date(q.completed_ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        : '';
    const headFile = assets[0] ? _htmlEscape(_truncMiddle(assets[0], 40)) : '';
    const assetCountBadge = assets.length > 1
        ? `<span class="badge badge-xs badge-ghost aspect-square min-w-5 h-5 px-0 rounded-full font-mono" title="${assets.length} assets">${assets.length}</span>`
        : '';
    const metaParts = [
        tsHuman ? `<span class="text-[9px] font-mono text-base-content/60">${tsHuman}</span>` : '',
        dur ? `<span class="text-[9px] font-mono text-base-content/60">${dur}</span>` : '',
        headFile ? `<span class="text-[9px] font-mono opacity-70 hidden sm:inline truncate max-w-[12rem]">${headFile}</span>` : '',
    ].filter(Boolean).join('<span class="text-base-content/30">·</span>');
    const metaGroup = metaParts
        ? `<span class="inline-flex items-center gap-2 px-2 py-0.5 rounded-md bg-base-300/40 border border-base-300/60">${metaParts}</span>`
        : '';
    const previewList = assets.length
        ? `<div class="join join-vertical w-full mt-2">${assets.map(_renderDoneAssetRow).join('')}</div>`
        : `<div class="text-[10px] text-base-content/50 italic mt-2">no asset recorded</div>`;
    // Persist open state across WS-driven re-renders. The list innerHTML is
    // replaced ~1Hz, which would otherwise wipe the user's expansion. We
    // re-emit `open` based on the module-level _openDoneItem on every render
    // (must NOT rely on the previous DOM since it's already gone).
    const qid = String(q.ts || q.completed_ts || 0);
    const openAttr = (qid !== '0' && qid === _openDoneItem) ? ' open' : '';
    return `<li class="bg-base-200/40 rounded-md opacity-80 hover:opacity-100" data-q-status="done">
        <details data-q-id="${qid}"${openAttr}>
            <summary class="cursor-pointer p-2 flex items-center gap-2 text-xs flex-wrap">
                <span class="badge badge-xs ${cls}">${sym} ${verdict}</span>
                <span class="font-semibold truncate flex-1" title="${promptEsc}">${promptEsc}</span>
                ${assetCountBadge}
                ${metaGroup}
            </summary>
            <div class="px-2 pb-2 pt-0 border-t border-base-300/50">
                ${previewList}
            </div>
        </details>
    </li>`;
}

// Stage order for "is this stage already done?" lookups in the pipeline strip.
// Stage execution order: Audio + TTS now run BEFORE Base Image + Video.
// The audio-driven-chains feature uses the music/voice durations to size
// the video chain count so the final video matches the audio length —
// that requires audio to be measured before the video chain loop starts.
// Music/voice don't depend on the image; only the prompt (Concept).
const _STAGE_ORDER = ['Concept', 'Audio', 'TTS', 'Base Image', 'Video Chains', 'Post Process', 'Final Merge'];
// [canonicalStage, shortAcronym, displayLabel, activeVerb, tone].
// Module-scope so both renderPipelineStrip (per-item completed history) and
// _buildActiveJobProgressBar (top-of-card active bar) read the same table.
const _STAGES_META = [
    ['Concept', 'T', 'Text', 'Texting', 'accent'],
    ['Audio', 'M', 'Music', 'Composing', 'secondary'],
    ['TTS', 'S', 'Voice', 'Voicing', 'warning'],
    ['Base Image', 'I', 'Image', 'Imaging', 'info'],
    ['Video Chains', 'V', 'Video', 'Rendering parts', 'success'],
    ['Post Process', 'U', 'Upscale', 'Upscaling', 'warning'],
    ['Final Merge', 'F', 'Merge', 'Merging', 'accent'],
];
function _stageDoneBefore(curStage, candidate) {
    const ci = _STAGE_ORDER.indexOf(curStage);
    const xi = _STAGE_ORDER.indexOf(candidate);
    return ci > -1 && xi > -1 && xi < ci;
}

// Map a canonical stage name (e.g. "Base Image", "Video Chains") to its
// user-friendly short label ("Image", "Video"). Unknown stages fall
// through unchanged. Mirrors _STAGES_META[i][2] without forcing every
// caller to dig through the table by hand. Use this everywhere a stage
// name is rendered to the user — labels, tooltips, badges, status
// strings — so nothing leaks the internal "Base Image"/"Video Chains"
// terminology.
function _stageDisplayName(canonical) {
    const row = _STAGES_META.find(r => r[0] === canonical);
    return (row && row[2]) || canonical || '';
}
window._stageDisplayName = _stageDisplayName;

// Build the top-of-card segmented progress bar markup. Pulls all timing data
// from the same _lastTick / _stageStartTs / _jobActuals globals as the
// per-item renderPipelineStrip did before — but emits ONE bar at the
// queue-card level, not one per active item. Header above the bar shows just
// the activity spinner + verb. Each segment carries its own inline
// elapsed/ETA timing under the label so users read "actual / planned" per
// stage at a glance. Total elapsed / ETA sits in the footer row below.
// Route a progress-bar segment-label click to the most relevant
// settings surface for that stage. Concept → Settings → LLM tab; TTS →
// Settings → Speech tab; everything else → the Pipeline popup (which
// hosts every model select + per-stage tuning).
function _openSegSettings(stage) {
    if (stage === 'Concept') {
        if (typeof openSettings === 'function') {
            openSettings();
            // Settings is async; flip the LLM radio after the modal opens.
            setTimeout(() => {
                const r = document.querySelector('input[name="settings_tabs"][aria-label="LLM"]');
                if (r) r.checked = true;
            }, 60);
        }
        return;
    }
    if (stage === 'TTS') {
        if (typeof openSettings === 'function') {
            openSettings();
            setTimeout(() => {
                const r = document.querySelector('input[name="settings_tabs"][aria-label="Speech"]');
                if (r) r.checked = true;
            }, 60);
        }
        return;
    }
    if (typeof openPipeline === 'function') openPipeline();
}
window._openSegSettings = _openSegSettings;

function _buildActiveJobProgressBar(d) {
    const state = (d && d.state) || {};
    const curStep = state.step || '';
    if (!curStep) return '';
    const v = state.video_index || 1;
    const curIdx = _STAGE_ORDER.indexOf(curStep);
    const stageDurations = _STAGE_ORDER.map(s => _stageAvgSeconds(s) || 30);
    const totalSec = stageDurations.reduce((a, b) => a + b, 0);
    const segWidths = stageDurations.map(dur => (dur / totalSec) * 100);
    const stageElapsedSec = _stageStartTs ? (Date.now() - _stageStartTs) / 1000 : 0;
    const curStageAvg = stageDurations[curIdx] || 30;
    const stageProgressFraction = curIdx < 0 || !_stageStartTs
        ? 0
        : Math.min(1, stageElapsedSec / curStageAvg);
    const isOverrun = curIdx >= 0 && _stageStartTs && stageElapsedSec > curStageAvg;
    // Activity text is no longer derived here — the queue-header label is
    // backend-driven via the `render_heartbeat` WS event with a TTL.
    const jobElapsedMs = _jobStartTs ? (Date.now() - _jobStartTs) : 0;
    const totalElapsedHTML = _fmtElapsedHtml(jobElapsedMs);
    const totalEtaHTML = _fmtElapsedHtml(totalSec * 1000);
    const actuals = (_jobActuals[v] || {});
    const segments = _STAGE_ORDER.map((s, i) => {
        const isPast = curIdx >= 0 && i < curIdx;
        const isCurrent = i === curIdx;
        const cls = isPast ? 'pipeline-seg-past'
            : isCurrent ? 'pipeline-seg-current'
                : 'pipeline-seg-future';
        const overrunCls = (isCurrent && isOverrun) ? ' pipeline-seg-overrun' : '';
        const localFill = isPast ? 100 : (isCurrent ? stageProgressFraction * 100 : 0);
        const meta = _STAGES_META.find(x => x[0] === s) || [, , s, , 'primary'];
        const tone = meta[4];
        const shortLabel = meta[2];
        // Inline per-segment timing under the label. Past stages render their
        // recorded actual; current stage renders live elapsed (refreshed by
        // the 1 Hz ticker against [data-seg-elapsed]); future stages render
        // an em-dash. Each is paired with the model's ETA so users read
        // "actual / planned" for every step at a glance.
        let elapsedHtml;
        let etaSec;
        if (isPast) {
            const a = actuals[s];
            elapsedHtml = a ? _fmtElapsedHtml(a.duration_s * 1000) : '—';
            etaSec = (a && a.eta_s) || stageDurations[i];
        } else if (isCurrent) {
            elapsedHtml = _fmtElapsedHtml(stageElapsedSec * 1000);
            etaSec = stageDurations[i];
        } else {
            elapsedHtml = '—';
            etaSec = stageDurations[i];
        }
        const etaHtml = etaSec ? _fmtElapsedHtml(etaSec * 1000) : '—';
        return `<div class="pipeline-seg ${cls}${overrunCls}" style="flex: 1 1 0;" data-stage="${s}" data-tone="${tone}">
            <div class="pipeline-seg-top">
                <button type="button" class="pipeline-seg-label cursor-pointer" onclick="event.stopPropagation(); _openSegSettings(${JSON.stringify(s)})" title="Open settings for ${_htmlEscape(s)}">${shortLabel}</button>
            </div>
            <div class="pipeline-seg-bottom">
                <div class="pipeline-seg-fill bg-${tone}" style="width: ${localFill}%"></div>
                <span class="pipeline-seg-timing"><span class="pipeline-seg-time-chip" data-seg-elapsed>${elapsedHtml}</span><span class="opacity-60"> / ETA <span class="pipeline-seg-time-chip">${etaHtml}</span></span></span>
            </div>
        </div>`;
    }).join('');
    // Push initial Total elapsed/ETA into the external footer row
    // (#queue-progress-footer in template) — see the 1Hz tick handler
    // which keeps these in sync as the bar progresses.
    try {
        const tot = document.getElementById('queue-progress-total');
        const elEl = document.getElementById('queue-total-elapsed');
        const etEl = document.getElementById('queue-total-eta');
        if (tot) tot.style.display = '';
        if (elEl) elEl.innerHTML = totalElapsedHTML;
        if (etEl) etEl.innerHTML = totalEtaHTML;
    } catch (_) { }
    return `
        <div class="pipeline-bar-wrap"
             data-pipeline-bar
             data-cur-stage="${curStep || ''}">
            <div class="pipeline-bar relative overflow-hidden rounded-md bg-base-200 border border-base-300">
                <div class="flex" data-pipeline-segments>${segments}</div>
            </div>
        </div>
    `;
}

async function editItem(ts, currentPrompt) {
    if (!ts) return;
    const next = window.prompt('Edit prompt for this queue item:', currentPrompt || '');
    if (next == null) return;          // cancelled
    if (!next.trim()) return;          // refuse empty
    if (next.trim() === (currentPrompt || '').trim()) return; // no change
    try {
        await fetch('/queue/edit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ts, prompt: next.trim() }),
        });
    } catch (e) { console.warn('edit failed', e); }
}

async function toggleItemInfinity(ts) {
    if (!ts) return;
    try {
        await fetch('/queue/toggle-infinity', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ts }),
        });
    } catch (e) { console.warn('toggle-infinity failed', e); }
}

async function toggleItemPolymorphic(ts) {
    if (!ts) return;
    try {
        await fetch('/queue/toggle-polymorphic', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ts }),
        });
    } catch (e) { console.warn('toggle-polymorphic failed', e); }
}
window.toggleItemPolymorphic = toggleItemPolymorphic;

// Endless Story auto-cycle: when the navbar toggle is on, fetch a fresh
// suggestion batch every 12 s and APPEND a new row (don't replace existing).
// Cap at _SUGGEST_MAX_ROWS (=50); FIFO eviction kicks in beyond that.
let _endlessStoryTimer = null;
function _endlessCycleMs() {
    try {
        const v = parseInt(localStorage.getItem('slopfinity-endless-cycle-s') || '12', 10);
        if (isFinite(v) && v >= 4 && v <= 60) return v * 1000;
    } catch (_) { }
    return 12000;
}
// Allow the Slop Config modal's interval slider to restart the timer
// in place — without this, a new interval value wouldn't take effect
// until the next stop/start cycle.
function _restartEndlessCycle() {
    const t = document.getElementById('endless-story-toggle');
    if (!t || !t.checked) return;
    if (_endlessStoryTimer) {
        clearInterval(_endlessStoryTimer);
        _endlessStoryTimer = null;
    }
    t.dispatchEvent(new Event('change', { bubbles: true }));
}
window._restartEndlessCycle = _restartEndlessCycle;
function _wireEndlessStoryCycle() {
    const t = document.getElementById('endless-story-toggle');
    if (!t) return;
    // The 12 s auto-append cycle has been REMOVED — it spammed new rows
    // every tick after Start Story, surprising users who expected the
    // Prompt card to stay calm until they manually pressed +. The +
    // button is now the canonical way to add a row (with prefetch making
    // it instant), and per-row 🔄 refresh handles the "give me different
    // chips for THIS row" case. Wiring is kept as a no-op so existing
    // localStorage state + UI toggle stay valid; the change-listener
    // below now only manages the toggle's persisted state.
    const start = () => { /* intentional no-op — see comment above */ };
    if (false) { // dead code, kept inside an unreachable block so the
        // pre-existing closure capture (recentChips / promptId / etc.)
        // remains for any future callers that want to revive it.
        _endlessStoryTimer = setInterval(async () => {
            if (typeof _isSuggestionsHidden === 'function' && _isSuggestionsHidden()) return;
            if (typeof _autoSuggestDisabled === 'function' && _autoSuggestDisabled()) return;
            if (!_endlessRunning) return;
            if (typeof _getSubjectsMode === 'function' && _getSubjectsMode() !== 'endless') return;
            // Direct fetch + APPEND (vs regenSuggestions which clears the stack)
            // so existing rows stay visible while new ones queue below them.
            try {
                // Endless Story mode = continuation of the existing chip
                // history. Grab the most-recent visible chip texts and
                // pass them as `subjects=` so the server can prompt the
                // LLM to extend the story rather than generate isolated
                // fresh subjects. `endless=1` flips the server's
                // user-message wording.
                const { stack } = _getSuggestStack();
                const recentChips = Array.from(
                    stack ? stack.querySelectorAll('button[data-suggest]') : []
                ).map(b => b.dataset.suggest).filter(Boolean);
                // Dedupe (chips are duplicated in each row for the
                // marquee wraparound) and keep the last 6 unique.
                const seen = new Set(); const uniq = [];
                for (const t of recentChips) {
                    if (!seen.has(t)) { seen.add(t); uniq.push(t); }
                    if (uniq.length >= 6) break;
                }
                const qs = new URLSearchParams({ n: '6', endless: '1' });
                if (uniq.length) qs.set('subjects', uniq.join('\n'));
                const r = await fetch('/subjects/suggest?' + qs.toString());
                const d = await r.json();
                const arr = (d && d.suggestions) || [];
                if (arr.length && typeof _appendSuggestBatchRow === 'function') {
                    // Pass promptId + rowIdx so the cycle's auto-appended
                    // rows render with the same subject/refresh/minus
                    // lead cluster as the rows added by the user. Without
                    // these opts the row would render as a bare marquee
                    // (no dropdown), which read as "the simple-mode
                    // suggestions are leaking into endless".
                    const existingRows = stack ? stack.querySelectorAll('.suggest-marquee-row').length : 0;
                    const promptId = (typeof _getDefaultPromptId === 'function')
                        ? _getDefaultPromptId() : 'yes-and';
                    // Persist the new row's prompt so refresh / minus on
                    // it stays in sync with _ENDLESS_ROW_PROMPTS_KEY.
                    if (typeof _getEndlessRowPrompts === 'function'
                        && typeof _setEndlessRowPrompts === 'function') {
                        const arrIds = _getEndlessRowPrompts();
                        arrIds.push(promptId);
                        _setEndlessRowPrompts(arrIds);
                    }
                    _appendSuggestBatchRow(arr, { promptId, rowIdx: existingRows });
                }
            } catch (_) { }
        }, _endlessCycleMs());
    };
    const stop = () => {
        if (_endlessStoryTimer) { clearInterval(_endlessStoryTimer); _endlessStoryTimer = null; }
    };
    t.addEventListener('change', () => { t.checked ? start() : stop(); });
    // Hydrate from localStorage on first load — the toggle is now its
    // own piece of state (decoupled from Infinity / queue settings).
    try {
        const persisted = localStorage.getItem('slopfinity-endless-story') === '1';
        if (persisted !== t.checked) {
            t.checked = persisted;
            const host = document.getElementById('subjects-endless-story');
            if (host) host.setAttribute('aria-pressed', persisted);
        }
    } catch (_) { }
    if (t.checked) start();
}
document.addEventListener('DOMContentLoaded', _wireEndlessStoryCycle);

// ---------------------------------------------------------------------------
// Subjects card mode pill: 'raw' (queue verbatim) vs 'endless' (LLM
// auto-cycles story continuations from a seed). State persists in
// localStorage. The pill swaps the queue button label between
// {Queue Slop, Start Story, I'm Feeling Lucky} based on mode + seed
// presence + running state.
// ---------------------------------------------------------------------------
const _SUBJ_MODE_KEY = 'slopfinity-subjects-mode';
let _endlessRunning = false;

function _getSubjectsMode() {
    try {
        const v = localStorage.getItem(_SUBJ_MODE_KEY);
        if (v === 'endless') return 'endless';
        if (v === 'chat') return 'chat';
        if (v === 'raw') return 'raw';
        return 'simple';  // default: single textarea, LLM rewrites at queue time
    } catch (_) { return 'simple'; }
}

function _getSuggestStack() {
    const mode = _getSubjectsMode();
    const isEndless = mode === 'endless';
    return {
        stack: document.getElementById(isEndless ? 'subject-chips-stack-endless' : 'subject-chips-stack-simple'),
        placeholder: document.getElementById(isEndless ? 'subject-chips-empty-endless' : 'subject-chips-empty-simple')
    };
}

function _setSubjectsMode(mode) {
    if (mode !== 'simple' && mode !== 'raw' && mode !== 'endless' && mode !== 'chat') mode = 'simple';
    try { localStorage.setItem(_SUBJ_MODE_KEY, mode); } catch (_) { }
    document.querySelectorAll('.subjects-mode-pill button[data-subj-mode]').forEach(b => {
        const active = b.getAttribute('data-subj-mode') === mode;
        b.classList.toggle('subj-mode-active', active);
        b.classList.toggle('btn-outline', !active);
        b.setAttribute('aria-pressed', String(active));
    });
    // Switching out of endless ends the running story cleanly.
    if (mode !== 'endless' && _endlessRunning) _endEndlessStory();
    // Switching INTO endless: there is no longer a separate Start Story
    // state — entering endless mode IS starting the story. The big
    // action button stays as the regular Queue Slop button (see
    // _updateSubjectsActionLabel + _subjectsAction); chip clicks /
    // manual typing append per-beat rows; the story pane stays visible
    // for the entire duration of endless mode.
    if (mode === 'endless' && !_endlessRunning) {
        _endlessRunning = true;
        document.body.classList.add('endless-running');
        if (!window._endlessStoryStartTs) window._endlessStoryStartTs = Date.now();
        // Make sure suggestion-row prompts are seeded so _renderEndlessRows
        // produces at least one chip row. Idempotent — _setEndlessRowPrompts
        // is just a localStorage write.
        try {
            if (!_getEndlessRowPrompts().length) {
                const defaultId = (typeof _getDefaultPromptId === 'function')
                    ? _getDefaultPromptId() : 'yes-and';
                _setEndlessRowPrompts([defaultId]);
            }
        } catch (_) { }
        // Seed an empty first beat if the log is empty — gives the user
        // a row to type into / a target for the very first chip click.
        try {
            const existing = _loadEndlessLogLines();
            if (!existing.length) {
                _persistEndlessLogLines(['']);
                _setEndlessActiveBeatIdx(0);
            }
        } catch (_) { }
    }
    _updateSubjectsActionLabel();
    // Story pane is always visible in endless mode now (no separate
    // Start state — the pane IS the endless surface).
    const pane = document.getElementById('subjects-story-pane');
    if (pane) pane.classList.toggle('hidden', mode !== 'endless');
    // Chat pane is the inverse — visible only in chat mode.
    const chatPane = document.getElementById('subjects-chat-pane');
    if (chatPane) chatPane.classList.toggle('hidden', mode !== 'chat');
    // The big Queue button is irrelevant in chat mode (the chat input is
    // the only action). Hide it; restore on mode switch.
    const bigBtn = document.getElementById('btn-start-stop-inline');
    if (bigBtn) bigBtn.classList.toggle('hidden', mode === 'chat');
    // Per-mode textarea sizing — raw needs space (multi-line subjects),
    // endless and chat compact down (the seed/prompt is short, the
    // story-log or chat-bubble pane below it does the heavy lifting).
    // Body class drives the actual CSS rules in app.css (.subj-mode-*).
    const body = document.body;
    body.classList.remove('subj-mode-simple', 'subj-mode-raw', 'subj-mode-endless', 'subj-mode-chat');
    body.classList.add('subj-mode-' + mode);
    const ta = document.getElementById('p-core');
    if (ta) {
        ta.rows = (mode === 'simple' || mode === 'raw') ? 5 : 2;
        // Per-mode placeholder is the canonical mode-hint surface —
        // shows when the textarea is empty and gets out of the way
        // the moment the user types. NO separate heading-above-the-
        // input; the placeholder IS the heading. Keep each one
        // self-contained: tells the user (a) what this field is
        // for in this mode and (b) what action to take next.
        const placeholders = {
            simple: 'Your idea — type it here.\n\nThe LLM rewrites your idea into per-stage prompts when you click Queue Slop. Pick from the suggestions below if you want a starter.',
            // Raw hides #p-core entirely; placeholder unused but kept
            // so a mode-switch sequence doesn't leave stale copy.
            raw: '(raw mode uses the per-stage prompts above directly — no subject seed)',
            endless: 'Optional opening seed.\n\nType beats directly into the story rows below — Enter adds the next row. Click any suggestion chip to append it as a new beat. Press Queue Slop when you\'re ready to render.',
            chat: 'Ask the assistant — e.g. "queue 3 short clips of dragons"',
        };
        ta.placeholder = placeholders[mode] || placeholders.raw;
    }
    // Chat mode owns its own input (#subjects-chat-input) — sync its
    // placeholder too so the same descriptive-copy convention applies.
    const chatInput = document.getElementById('subjects-chat-input');
    if (chatInput) {
        chatInput.placeholder = 'Ask the assistant — e.g. "queue 3 short clips of dragons", "what\'s running?", "cancel job 4"';
    }
    // Render any persisted history on first switch into chat. Reply
    // suggestions ONLY auto-fire when the assistant has already
    // responded in the conversation — pre-first-turn we wait for the
    // user to send something, so we don't burn LLM cycles on
    // "starter chips" the user didn't ask for. Manual regen via the
    // refresh button still works (regenSuggestions → _renderChatReplies).
    if (mode === 'chat') {
        _renderChatLog();
        const hist = (typeof _getChatHistory === 'function') ? _getChatHistory() : [];
        const hasAsst = hist.some(m => m && m.role === 'assistant' && (m.content || '').trim());
        if (hasAsst && typeof _renderChatReplies === 'function') _renderChatReplies();
    }
    // Switching INTO endless: rehydrate the persisted story log if we
    // have one. Survives layout switches (mobile-nav prev/next), full
    // reloads, and accidental tab-aways. The pane visibility is owned
    // by `_endlessRunning`; the LOG content is owned by localStorage.
    // We always paint the saved text into the pre — even when
    // _endlessRunning is currently false (e.g. user came back after
    // closing the browser), so the user sees their progress even
    // before they re-Start.
    if (mode === 'endless') {
        const log = document.getElementById('subjects-story-log');
        if (log) {
            const savedLines = _loadEndlessLogLines();
            // Only paint if the host is empty AND there is saved content —
            // avoids clobbering an in-progress edit on a quick tab flip.
            if (savedLines.length && !log.children.length) {
                _renderEndlessLog();
                // Show the pane if there's saved content even if not
                // currently running — gives the user a "your story is
                // still here, press Start to continue" affordance.
                const pane = document.getElementById('subjects-story-pane');
                if (pane) pane.classList.remove('hidden');
            }
        }
    }
    // Switching INTO endless while no story is running: wipe whatever
    // chips the previous mode left behind and show the "press Start
    // Story" hint. Without this the simple/chat marquee rows remain
    // visible after a mode swap, which contradicts the rule that
    // endless suggestions only exist while a story is in flight.
    // Switching OUT of endless: re-render the cached chips so simple
    // mode isn't stuck with an empty stack until the user clicks ↻.
    const { stack: stackBox } = _getSuggestStack();
    if (mode === 'endless' && !_endlessRunning) {
        if (stackBox) stackBox.innerHTML =
            "";
    } else if (mode === 'simple') {
        if (stackBox && stackBox.querySelector('.suggest-marquee-row') === null
            && typeof _renderCachedSuggestions === 'function') {
            _renderCachedSuggestions();
        }
    }
    // The Suggestions toggle stays freely user-controlled in every mode.
    // Endless's Suggestions-required constraint is enforced at the
    // Start-Story button (disabled until Suggestions is on) — not by
    // forcing the toggle. _updateSubjectsActionLabel handles that.
    const sugLabel = document.getElementById('subjects-suggestions-toggle');
    if (sugLabel) {
        sugLabel.classList.remove('pointer-events-none', 'opacity-70');
        sugLabel.title = 'Suggestions — show/hide auto-suggestion controls. Required for Endless story mode.';
    }
    // Repaint the unified Suggestions badge — its right-edge action button
    // swaps between "↻ refresh-all" (simple/chat) and "+ add row" (endless),
    // so a mode change has to refire the icon swap immediately.
    if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
}
window._setSubjectsMode = _setSubjectsMode;

function _updateSubjectsActionLabel() {
    const btn = document.getElementById('btn-start-stop-inline');
    if (!btn) return;
    const mode = _getSubjectsMode();
    const ta = document.getElementById('p-core');
    const seed = (ta && ta.value || '').trim();
    if (mode === 'endless') {
        // Endless mode no longer has a separate Start Story state —
        // the big button is the regular Queue Slop button and behaves
        // identically to simple/raw. Defer to the legacy queue-label
        // builder so labels like "Queue Slop / Queue Infinite Slop /
        // Generate ASAP …" stay consistent across modes.
        btn.disabled = false;
        btn.classList.remove('opacity-70', 'opacity-50', 'cursor-not-allowed');
        btn.title = 'Queue clips for the current story beats';
        const txtNode = document.getElementById('btn-start-stop-inline-text') || btn;
        txtNode.textContent = (window.__SLOPFINITY_DEFAULT_QUEUE_LABEL__ || 'Queue Slop');
        if (typeof _updateStartBtn === 'function') _updateStartBtn();
    } else if (mode === 'chat') {
        // Big button is hidden in chat mode; nothing to update.
    } else {
        // Simple / Raw — both defer to the legacy label-builder which
        // handles "Queue Slop / Queue Infinite Slop / Generate ASAP …".
        // Reset any sticky endless-mode state (Start Story / Story Running)
        // on the way back so the button label tracks the new mode.
        btn.disabled = false;
        btn.classList.remove('opacity-70', 'opacity-50', 'cursor-not-allowed');
        btn.title = 'Start/queue generation';
        // Set a sensible default text BEFORE _updateStartBtn runs so even
        // if that helper is missing or no-ops, the label isn't a leftover.
        const txtNode = document.getElementById('btn-start-stop-inline-text') || btn;
        txtNode.textContent = (window.__SLOPFINITY_DEFAULT_QUEUE_LABEL__ || 'Queue Slop');
        if (typeof _updateStartBtn === 'function') _updateStartBtn();
    }
}
window._updateSubjectsActionLabel = _updateSubjectsActionLabel;

// Queue-status chip under the Queue Slop button. Pulls depth + activity
// from the WS tick payload (passed in as `d`). Best-effort; missing
// fields fall back to '—' / 'idle'. Updates two spans:
//   #btn-queue-info-depth   — pending count or 'queue empty'
//   #btn-queue-info-status  — current step OR mode OR 'idle' / 'paused'
function _updateQueueStatusChip(d) {
    // Two copies of the chip: the shared one under #subjects-input-row
    // (visible in simple/endless) and the raw-mode in-pane copy.
    // Update both — whichever is in the active pane will be visible;
    // the other stays hidden but synchronised in case the user mode-
    // swaps before the next tick.
    const depthEls = [
        document.getElementById('btn-queue-info-depth'),
        document.getElementById('btn-queue-info-depth-raw'),
    ].filter(Boolean);
    const statusEls = [
        document.getElementById('btn-queue-info-status'),
        document.getElementById('btn-queue-info-status-raw'),
    ].filter(Boolean);
    if (!depthEls.length || !statusEls.length) return;
    const depthEl = depthEls[0];   // for class toggle below; both same content
    const statusEl = statusEls[0];
    const queue = (d && d.queue) || [];
    const pending = queue.filter(x => x && (x.status === 'pending' || x.status == null)).length;
    const working = queue.filter(x => x && x.status === 'working').length;
    // Infinity-mode tally — count how many of the visible queue items
    // are flagged infinity (will auto-re-loop after each completion).
    // Surfaces as a small ∞ marker after the count so the user knows
    // "of the N queued, K of them won't stop on their own". Empty
    // string when none are infinity-flagged, so the chip stays clean
    // for the common case.
    const infinityCount = queue.filter(x => x && x.infinity
        && (x.status === 'pending' || x.status === 'working' || x.status == null)).length;
    const infinityMark = infinityCount > 0
        ? ` <span class="text-primary" title="${infinityCount} infinity-mode item${infinityCount === 1 ? '' : 's'} (auto-re-loop)">∞${infinityCount}</span>`
        : '';
    // Drop the leading "0+" noise when nothing is in-flight — the user
    // pointed out that "0+49 queued" reads as weird math. Keep the split
    // ONLY when both counts are non-zero so the chip still distinguishes
    // working-vs-pending at a glance during active runs.
    let depthText;
    if (pending === 0 && working === 0) depthText = 'queue empty';
    else if (working === 0) depthText = `${pending} queued${infinityMark}`;
    else if (pending === 0) depthText = `${working} running${infinityMark}`;
    else depthText = `${working} running · ${pending} queued${infinityMark}`;
    // Detect a count INCREASE (item just queued) so we can pulse the chip
    // — pure-grow visual feedback that the click did something. We pulse
    // the depth element specifically (it's the count, not the status),
    // and only on increase (decrements happen as items finish, which the
    // user doesn't need a celebration for).
    const total = pending + working;
    const prevTotal = (typeof _updateQueueStatusChip._prevTotal === 'number')
        ? _updateQueueStatusChip._prevTotal : total;
    const grew = total > prevTotal;
    _updateQueueStatusChip._prevTotal = total;
    depthEls.forEach(el => {
        // innerHTML — `depthText` may include the ∞-marker span when
        // any queued item is infinity-flagged. Other contributors are
        // plain numbers (no XSS risk; queue is server-controlled state,
        // not user-typed text inserted as raw HTML).
        el.innerHTML = depthText;
        el.classList.toggle('text-primary', working > 0);
        if (grew) {
            // Restart the animation by removing + forcing reflow + re-adding.
            // CSS keyframe runs ~1.6 s and removes itself via animationend.
            el.classList.remove('queue-count-pulse');
            void el.offsetWidth;
            el.classList.add('queue-count-pulse');
        }
    });
    // Simplified status: just Idle / Generating / Paused. Earlier the
    // chip leaked the internal stage name ('base image' / 'video chains')
    // which read like jargon. Three-state is enough for "is the box
    // doing work" at a glance — Queue card has the per-stage detail.
    const state = (d && d.state) || {};
    let status = 'Idle';
    if (state.step) {
        status = 'Generating';
    } else if (state.mode && state.mode !== 'Idle') {
        status = 'Generating';
    }
    if (d && d.paused) status = 'Paused';
    statusEls.forEach(el => {
        el.textContent = status;
        // Theme-coloured status: Generating reads as the active app accent
        // (text-primary) rather than text-success — the rest of the dashboard
        // already uses primary as the "this is the live signal" colour
        // (ticker bars, mode-pill active fill, etc.) so success-green stood
        // out as a one-off semantic. Paused stays text-warning because that
        // genuinely IS a caution state, not a theme accent.
        el.classList.toggle('text-warning', status === 'Paused');
        el.classList.toggle('text-primary', status === 'Generating');
    });
    // Focus-mode FAB badges: paint the queue tally on whichever flanking
    // FAB points TOWARD queue from the user's current focused layout.
    //   - On Prompt   → next is Queue → badge on focus-fab-next
    //   - On Slop     → prev is Queue → badge on focus-fab-prev (NEW —
    //                   the user wanted a queue tally visible from the
    //                   Slop layout's nav cluster)
    //   - On Queue    → user is already there, neither badge shows
    const totalText = total > 99 ? '99+' : String(total);
    const curMode = (typeof _mobileNavCurrentIdx === 'function')
        ? _mobileNavCurrentIdx() : -1;
    const nextStep = (typeof _MOBILE_NAV_ORDER !== 'undefined' && curMode >= 0)
        ? _MOBILE_NAV_ORDER[curMode + 1] : null;
    const prevStep = (typeof _MOBILE_NAV_ORDER !== 'undefined' && curMode >= 0)
        ? _MOBILE_NAV_ORDER[curMode - 1] : null;
    const fabNext = document.getElementById('focus-fab-next-badge');
    if (fabNext) {
        const show = total > 0 && nextStep && nextStep.layout === 'queue';
        fabNext.classList.toggle('hidden', !show);
        if (show) fabNext.textContent = totalText;
    }
    const fabPrev = document.getElementById('focus-fab-prev-badge');
    if (fabPrev) {
        const show = total > 0 && prevStep && prevStep.layout === 'queue';
        fabPrev.classList.toggle('hidden', !show);
        if (show) fabPrev.textContent = totalText;
    }
    // Mobile bottom-nav: append the count to whichever directional label
    // (prev or next) names "Queue", so the user sees the tally on the
    // arrow that would take them there. _mobileNavRefresh sets the base
    // label; we append "(N)" here on every WS tick.
    const mNavPrevLbl = document.getElementById('mobile-nav-prev-label');
    const mNavNextLbl = document.getElementById('mobile-nav-next-label');
    if (mNavPrevLbl && prevStep) {
        const base = prevStep.label || '';
        mNavPrevLbl.textContent = (prevStep.layout === 'queue' && total > 0)
            ? `${base} (${totalText})` : base;
    }
    if (mNavNextLbl && nextStep) {
        const base = nextStep.label || '';
        mNavNextLbl.textContent = (nextStep.layout === 'queue' && total > 0)
            ? `${base} (${totalText})` : base;
    }
    // Stitch hooks — every tick, refresh the stitch button label so its
    // "(waiting for N)" count tracks the live queue depth, then check
    // whether the queue has drained enough to fire a deferred stitch.
    if (typeof _refreshStitchButton === 'function') _refreshStitchButton();
    if (typeof _maybeFirePendingStitch === 'function') _maybeFirePendingStitch();
}
window._updateQueueStatusChip = _updateQueueStatusChip;

// ---------------------------------------------------------------------------
// Chat mode — tool-using assistant. The LLM has tools to queue clips,
// inspect status, list recent outputs, etc. Client owns conversation
// history (localStorage), sends it on each turn, renders the returned
// message list including tool-call chips.
// ---------------------------------------------------------------------------
const _CHAT_HISTORY_KEY = 'slopfinity-chat-history-v1';
const _CHAT_HISTORY_MAX = 50;

function _getChatHistory() {
    try {
        const raw = localStorage.getItem(_CHAT_HISTORY_KEY);
        const arr = raw ? JSON.parse(raw) : [];
        return Array.isArray(arr) ? arr : [];
    } catch (_) { return []; }
}
function _setChatHistory(arr) {
    const trimmed = (Array.isArray(arr) ? arr : []).slice(-_CHAT_HISTORY_MAX);
    try { localStorage.setItem(_CHAT_HISTORY_KEY, JSON.stringify(trimmed)); } catch (_) { }
    // Mirror to the tree as the active chain — see _chatTree* below.
    _chatSyncTreeFromHistory(trimmed);
}

// ── Chat message tree (forking + branch nav) ───────────────────────────────
// Each message is a node with a `parent` pointer. Editing a user message
// creates a NEW sibling node sharing the same parent — the original chain
// still exists; the active chain just switches to the new sibling. Branch
// nav under each forked message lets you flip between siblings.
//
// Storage:
//   { nodes: { [id]: { id, role, content, tool_calls, name, parent, ts } },
//     active: <leaf id>, nextId: int }
// Active chain = walk from `active` back to root via parent pointers,
// reverse → array of message objects (matches the legacy flat-array shape
// `_getChatHistory` returns, so render + send paths Just Work).
const _CHAT_TREE_KEY = 'slopfinity-chat-tree-v1';

function _chatGetTree() {
    try {
        const raw = localStorage.getItem(_CHAT_TREE_KEY);
        if (raw) {
            const t = JSON.parse(raw);
            if (t && t.nodes && typeof t.nextId === 'number') return t;
        }
    } catch (_) { }
    return { nodes: {}, active: null, nextId: 1 };
}
function _chatSetTree(tree) {
    try { localStorage.setItem(_CHAT_TREE_KEY, JSON.stringify(tree)); } catch (_) { }
}
// Active chain: walk parent pointers from active leaf back to root,
// then reverse so chronological order matches the legacy history array.
function _chatActiveChain() {
    const tree = _chatGetTree();
    if (!tree.active || !tree.nodes[tree.active]) return [];
    const chain = [];
    let cur = tree.active;
    const seen = new Set();
    while (cur && tree.nodes[cur] && !seen.has(cur)) {
        seen.add(cur);
        chain.push(tree.nodes[cur]);
        cur = tree.nodes[cur].parent;
    }
    return chain.reverse();
}
// Sync the tree from a flat history array (called by _setChatHistory).
// Strategy: if the LEGACY array's tail extends the current active chain,
// just append new nodes. Otherwise rebuild from scratch — preserves the
// fork tree across normal sends, lets explicit forks (via _chatForkAt)
// override.
function _chatSyncTreeFromHistory(hist) {
    const tree = _chatGetTree();
    const chain = _chatActiveChain();
    // If hist exactly matches the current chain, no-op.
    const matches = hist.length === chain.length && hist.every((m, i) =>
        chain[i] && chain[i].role === m.role && chain[i].content === m.content
        && JSON.stringify(chain[i].tool_calls || null) === JSON.stringify(m.tool_calls || null)
    );
    if (matches) return;
    // If hist is a strict EXTENSION of the chain, append the new nodes
    // to the active leaf (preserves any sibling branches elsewhere).
    if (hist.length > chain.length && chain.every((c, i) => hist[i]
        && hist[i].role === c.role && hist[i].content === c.content)) {
        let parent = tree.active;
        for (let i = chain.length; i < hist.length; i++) {
            const id = String(tree.nextId++);
            tree.nodes[id] = { id, ...hist[i], parent, ts: Date.now() };
            parent = id;
        }
        tree.active = parent;
        _chatSetTree(tree);
        return;
    }
    // Divergence (e.g. truncation from refresh / external mutation):
    // rebuild a flat chain. Sibling branches are dropped — safe because
    // truncate-and-refresh is the user's explicit "undo this branch" path.
    const fresh = { nodes: {}, active: null, nextId: 1 };
    let parent = null;
    hist.forEach(m => {
        const id = String(fresh.nextId++);
        fresh.nodes[id] = { id, ...m, parent, ts: Date.now() };
        parent = id;
    });
    fresh.active = parent;
    _chatSetTree(fresh);
}
// Fork at the user message with id `nodeId`: create a NEW sibling node
// (same parent) carrying `newContent`, set it as the active leaf, save
// the resulting linear chain via _setChatHistory so the legacy send
// path picks it up. Returns the new node's id.
function _chatForkAt(nodeId, newContent) {
    const tree = _chatGetTree();
    const orig = tree.nodes[nodeId];
    if (!orig) return null;
    const id = String(tree.nextId++);
    tree.nodes[id] = {
        id,
        role: orig.role,           // preserved (only edit user roles in UI)
        content: newContent,
        parent: orig.parent,        // shared parent → siblings
        ts: Date.now(),
    };
    tree.active = id;
    _chatSetTree(tree);
    // Mirror to the legacy flat history so _sendChatMessage's history
    // loop sees the truncated + edited chain.
    const chain = _chatActiveChain();
    try { localStorage.setItem(_CHAT_HISTORY_KEY, JSON.stringify(chain)); } catch (_) { }
    return id;
}
// Switch the active leaf to whichever leaf descends from `targetNodeId`.
// If the target node is itself a leaf (no children), use it directly.
// Otherwise walk down its descendant chain to a leaf — preserves the
// "you were viewing branch B; now switching to branch A's most-recent
// continuation" UX.
function _chatSwitchActiveTo(targetNodeId) {
    const tree = _chatGetTree();
    if (!tree.nodes[targetNodeId]) return;
    // Find a leaf descendant. BFS through the children of targetNodeId.
    const childrenOf = (id) => Object.values(tree.nodes).filter(n => n.parent === id).map(n => n.id);
    let cur = targetNodeId;
    let kids = childrenOf(cur);
    const seen = new Set([cur]);
    while (kids.length && !seen.has(kids[0])) {
        // Pick the most-recently-created child (highest ts) so the
        // user lands on their LATEST continuation of that branch.
        const next = kids.sort((a, b) => (tree.nodes[b].ts || 0) - (tree.nodes[a].ts || 0))[0];
        seen.add(next);
        cur = next;
        kids = childrenOf(cur);
    }
    tree.active = cur;
    _chatSetTree(tree);
    // Mirror to legacy history so _renderChatLog (which reads from
    // _getChatHistory) picks up the new chain.
    const chain = _chatActiveChain();
    try { localStorage.setItem(_CHAT_HISTORY_KEY, JSON.stringify(chain)); } catch (_) { }
}
// Sibling lookup for the branch nav. Returns array of sibling node ids
// (the message at this position across forks) sorted by creation time.
function _chatSiblingsOf(nodeId) {
    const tree = _chatGetTree();
    const node = tree.nodes[nodeId];
    if (!node) return [];
    return Object.values(tree.nodes)
        .filter(n => n.parent === node.parent && n.role === node.role)
        .sort((a, b) => (a.ts || 0) - (b.ts || 0))
        .map(n => n.id);
}
window._chatForkAt = _chatForkAt;
window._chatSwitchActiveTo = _chatSwitchActiveTo;
window._chatSiblingsOf = _chatSiblingsOf;
window._chatActiveChain = _chatActiveChain;

// Switch to a sibling branch via the chat-branch-pill, re-render, then
// scroll the FORK-FROM user message into view. The fork-from message is
// the one carrying the branch-nav (i.e. the user message at the same
// chain position as the target node), and it shares its `nodeId` value
// with the target's siblings — so we can grab `targetNodeId` directly
// and scrollIntoView the bubble that has `data-node-id="<targetNodeId>"`.
// Without this, the chat log auto-scrolls to the end (set by
// _renderChatLog), which can hide the fork point if the new branch is
// long. See user's request: "when i toggle between forked threads, can
// we jump to the message that the fork was from?".
window._chatToggleFork = function (targetNodeId) {
    _chatSwitchActiveTo(targetNodeId);
    _renderChatLog();
    // scrollIntoView after the next paint so layout is settled. The
    // user-bubble carrying this id is the fork-from message itself.
    requestAnimationFrame(() => {
        const el = document.querySelector(`[data-node-id="${targetNodeId}"]`);
        if (el && typeof el.scrollIntoView === 'function') {
            try { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
            catch (_) { el.scrollIntoView(); }
        }
    });
};

function _resetChat() {
    if (!confirm('Clear chat history? This wipes the current conversation.')) return;
    _setChatHistory([]);
    _renderChatLog();
    // History wiped → don't auto-fetch starter chips. Reply suggestions
    // are reserved for "after the assistant responds" (per the chat
    // suggestion-cache scoping rule). User can manually regen via the
    // refresh button if they want chips before sending.
    const host = document.getElementById('subjects-chat-replies');
    if (host) host.innerHTML = '';
}
window._resetChat = _resetChat;

function _renderChatLog() {
    const log = document.getElementById('subjects-chat-log');
    if (!log) return;
    const history = _getChatHistory();
    if (!history.length) {
        log.innerHTML = '<div class="text-[10px] opacity-50 italic">no messages yet — try "queue 3 short clips of dragons" or "what\'s running?"</div>';
        return;
    }
    // DaisyUI chat / chat-bubble for proper bubble styling (avatar slot
    // omitted; chat-start/end handles left/right layout). Tool calls
    // render as compact mono chips inside the assistant bubble.
    // Helper: pretty-print a string the LLM/server returned. If it parses
    // as JSON, indent it; otherwise return as-is. Used inside <details>
    // expanders for assistant tool_calls + tool result messages.
    const _prettyJson = (s) => {
        if (typeof s !== 'string' || !s.trim()) return '';
        try {
            const obj = JSON.parse(s);
            return JSON.stringify(obj, null, 2);
        } catch (_) { return s; }
    };
    // Bubble action icons — appear dim on hover, light up on direct hover.
    // Each bubble gets a copy icon (copies its text content); assistant
    // bubbles ALSO get a refresh icon (re-asks the LLM with the same
    // history minus the last assistant turn). Wrapped in
    // `.chat-bubble-actions` so CSS handles opacity transitions.
    const _bubbleActions = (text, withRefresh, msgIdx) => {
        const escText = (text || '').replace(/"/g, '&quot;').replace(/\n/g, '&#10;');
        return `<div class="chat-bubble-actions">
            <button type="button" class="chat-bubble-action" title="Copy"
                onclick='_copyBubbleText(this, "${escText}")'>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round" class="w-3 h-3">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
            </button>
            ${withRefresh ? `<button type="button" class="chat-bubble-action" title="Re-ask"
                onclick='_refreshAssistantTurn(${msgIdx})'>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round" class="w-3 h-3">
                    <path d="M21 12a9 9 0 0 0-15-6.7L3 8"/>
                    <path d="M3 3v5h5"/>
                    <path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"/>
                    <path d="M21 21v-5h-5"/>
                </svg>
            </button>` : ''}
        </div>`;
    };
    // Also pull the active-chain nodes so each rendered message can
    // be paired with its tree id (drives the Edit + branch-nav UI).
    const _chain = (typeof _chatActiveChain === 'function') ? _chatActiveChain() : [];
    // Format a single tool_call into a clean human-friendly one-liner:
    //   queue_status()
    //   cancel_job(id=42)
    //   add_to_queue(prompt="dragons in fog…", count=3)
    // - String args truncate to 30 chars + ellipsis, kept in quotes.
    // - Numbers/bools render as-is via JSON.stringify.
    // - At most 3 args; remaining keys collapse to a trailing `…`.
    // Returns plain text — caller is responsible for HTML-escaping.
    const _toolCallSummary = (c) => {
        const fn = (c && c.function) || {};
        const name = fn.name || 'unknown';
        let parsed = {};
        try {
            parsed = typeof fn.arguments === 'string' ? JSON.parse(fn.arguments) : (fn.arguments || {});
        } catch (_) {
            return `${name}(…)`;
        }
        const entries = Object.entries(parsed || {});
        if (!entries.length) return `${name}()`;
        const head = entries.slice(0, 3).map(([k, v]) => {
            if (typeof v === 'string') {
                const trimmed = v.length > 30 ? v.slice(0, 30) + '…' : v;
                return `${k}="${trimmed}"`;
            }
            // Numbers, bools, null, arrays, objects — JSON.stringify is fine.
            // For nested objects we still want a compact repr.
            return `${k}=${JSON.stringify(v)}`;
        });
        const argShort = head.join(', ') + (entries.length > 3 ? ', …' : '');
        return `${name}(${argShort})`;
    };
    // Render-helper: build the inner blocks for ONE thinking item (either
    // an assistant tool_calls turn or a tool-result turn). Used inside the
    // fused thinking-run wrapper. Now renders as PLAIN <div>s (no nested
    // <details>) because the whole run is wrapped in ONE master <details>
    // expander — see the runItems map below. The inline label still shows
    // the tool name + a one-line preview for context once expanded.
    const _renderThinkingItemInner = (m) => {
        const role = m.role || '';
        if (role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length) {
            return m.tool_calls.map(c => {
                const fn = (c.function || {});
                let prettyHtml = '';
                const summary = _toolCallSummary(c);
                try {
                    const parsed = typeof fn.arguments === 'string' ? JSON.parse(fn.arguments) : (fn.arguments || {});
                    prettyHtml = _renderKvPretty(parsed, 0);
                } catch (_) {
                    prettyHtml = `<pre class="kv-raw whitespace-pre-wrap break-all text-[10px]">${_htmlEscape(String(fn.arguments || ''))}</pre>`;
                }
                // Plain div (not <details>) — the master <details> wrapping
                // the whole run is the SOLE collapse surface; nested per-item
                // collapses inside it would be annoying noise.
                return `<div class="chat-thought-item text-[10px] mt-1">
                    <div class="font-mono opacity-80">→ ${_htmlEscape(summary)}</div>
                    <div class="mt-1 p-2 bg-base-300/40 rounded">${prettyHtml}</div>
                </div>`;
            }).join('');
        }
        if (role === 'tool') {
            const raw = m.content || '';
            const oneLine = (raw || '').split('\n')[0] || '';
            const preview = oneLine.length > 80 ? oneLine.slice(0, 80) + '…' : oneLine;
            const name = m.name || 'tool';
            return `<div class="chat-thought-item text-[10px] mt-1">
                <div class="font-mono opacity-80">↳ ${_htmlEscape(name)}: ${_htmlEscape(preview)}</div>
                <div class="mt-1 p-2 bg-base-300/40 rounded">${_renderKvPrettySafe(raw)}</div>
            </div>`;
        }
        return '';
    };
    // Headline summary (collapsed state): the MOST RECENT tool_call's
    // clean one-liner, so a long run reads as "currently doing X" not
    // "3 tool calls". The full chronological history lives inside the
    // master <details> body, revealed on click.
    const _runHeadlineSummary = (runItems) => {
        for (let j = runItems.length - 1; j >= 0; j--) {
            const m = runItems[j];
            if (m && m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length) {
                const last = m.tool_calls[m.tool_calls.length - 1];
                return _toolCallSummary(last);
            }
        }
        const t = runItems.find(m => m && m.role === 'tool');
        return t ? `${t.name || 'tool'}(…)` : 'thinking…';
    };
    // Animated cogs (active state). The 'done' state shares the same
    // markup but CSS gates the @keyframes animation to .chat-thought-active
    // — see app.css. So in-flight runs spin and resolved runs sit static.
    const _cogsHTML = `<span class="chat-cogs" aria-hidden="true" title="thinking…">
        <svg class="chat-cog" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zm9.4 3.5c0-.5 0-1-.1-1.5l2-1.5-2-3.4-2.3.8c-.8-.7-1.7-1.2-2.7-1.5L15.8 2h-3.6l-.5 2.4c-1 .3-1.9.8-2.7 1.5l-2.3-.8-2 3.4 2 1.5c-.1.5-.1 1-.1 1.5s0 1 .1 1.5l-2 1.5 2 3.4 2.3-.8c.8.7 1.7 1.2 2.7 1.5l.5 2.4h3.6l.5-2.4c1-.3 1.9-.8 2.7-1.5l2.3.8 2-3.4-2-1.5c.1-.5.1-1 .1-1.5z"/>
        </svg>
        <svg class="chat-cog-rev" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zm9.4 3.5c0-.5 0-1-.1-1.5l2-1.5-2-3.4-2.3.8c-.8-.7-1.7-1.2-2.7-1.5L15.8 2h-3.6l-.5 2.4c-1 .3-1.9.8-2.7 1.5l-2.3-.8-2 3.4 2 1.5c-.1.5-.1 1-.1 1.5s0 1 .1 1.5l-2 1.5 2 3.4 2.3-.8c.8.7 1.7 1.2 2.7 1.5l.5 2.4h3.6l.5-2.4c1-.3 1.9-.8 2.7-1.5l2.3.8 2-3.4-2-1.5c.1-.5.1-1 .1-1.5z"/>
        </svg>
    </span>`;
    const isThinkingMsg = (m) => {
        if (!m) return false;
        const r = m.role || '';
        return r === 'tool' || (r === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length > 0);
    };
    // Two-pass render: walk history fusing consecutive thinking-run
    // messages (assistant w/ tool_calls + tool results) into ONE bubble,
    // emit standalone bubbles for user + assistant-text turns. The fused
    // bubble's `chat-thought-active` class flips to `chat-thought-done`
    // when the run is followed by an assistant-text turn — gating cog
    // animation. Old behavior (one bubble per message + cogs spinning
    // forever) made replayed conversations look like every step was
    // still computing.
    const out = [];
    let i = 0;
    while (i < history.length) {
        const m = history[i];
        const role = m.role || '';
        const idx = i;
        const node = _chain[idx] || null;
        const nodeId = node && node.id;
        if (isThinkingMsg(m)) {
            // Fuse the run.
            const runItems = [];
            const runStart = i;
            while (i < history.length && isThinkingMsg(history[i])) {
                runItems.push(history[i]);
                i++;
            }
            // In-flight if NOTHING follows the run yet. Once any non-
            // thinking message lands behind it (assistant text, user reply,
            // etc.), the run is done — cogs go static.
            const inflight = i >= history.length;
            const stateCls = inflight ? ' chat-thought-active' : ' chat-thought-done';
            // Chronological inner: every tool_call (one per call entry) +
            // every tool result, in the order they appeared. The master
            // <details> hides this list by default. The headline (visible
            // collapsed) is the most-recent tool_call's clean one-liner —
            // so a long run reads as "currently doing X" not "3 tool calls".
            const inner = runItems.map(_renderThinkingItemInner).join('');
            // Master expander: cogs + headline visible always; full
            // chronological history (calls + results) lives inside the
            // <details> body and is hidden by default.
            const headline = _runHeadlineSummary(runItems);
            out.push(`<div class="chat chat-start" data-msg-idx="${runStart}"><div class="chat-thought text-xs${stateCls}">${_cogsHTML}<details class="chat-thought-detail-wrap"><summary class="chat-thought-summary font-mono cursor-pointer">${_htmlEscape(headline)}</summary><div class="chat-thought-history">${inner}</div></details></div></div>`);
            continue;
        }
        if (role === 'user') {
            const sibs = (nodeId && typeof _chatSiblingsOf === 'function')
                ? _chatSiblingsOf(nodeId) : [];
            const isForked = sibs.length > 1;
            // Branch-switcher pills only appear when there are already multiple
            // forks — these are navigation, not an action, so they stay outside
            // the bubble as before.
            const navHTML = isForked ? `<div class="chat-branch-nav" data-pos-idx="${idx}">
                ${sibs.map((sid, j) => {
                const isActive = sid === nodeId;
                return `<button type="button" class="chat-branch-pill${isActive ? ' chat-branch-active' : ''}"
                        title="Switch to branch ${j + 1} of ${sibs.length}"
                        onclick="_chatToggleFork('${sid}')">${j + 1}</button>`;
            }).join('')}
            </div>` : '';
            // Edit button (pencil) — triggers inline edit + fork.
            const editBtn = nodeId ? `<button type="button" class="chat-bubble-action chat-bubble-edit"
                title="Edit + fork conversation"
                onclick='_chatBeginEdit("${nodeId}", this)'>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round" class="w-3 h-3">
                    <path d="M12 20h9"/>
                    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
                </svg>
            </button>` : '';
            // Fork button — now lives inside the bubble alongside copy + edit.
            // Identical action to edit (both open the inline edit form which
            // creates a new branch); the fork icon makes the branching intent
            // explicit for users who haven't discovered the edit pencil yet.
            const forkBtn = nodeId ? `<button type="button" class="chat-bubble-action"
                title="Fork conversation from this message"
                onclick='_chatBeginEdit("${nodeId}", this)'>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"
                     stroke-linecap="round" stroke-linejoin="round" class="w-3 h-3">
                    <circle cx="6" cy="3" r="2"/><circle cx="6" cy="21" r="2"/><circle cx="18" cy="9" r="2"/>
                    <path d="M6 5v6a4 4 0 0 0 4 4h4"/><path d="M6 13v6"/>
                </svg>
            </button>` : '';
            const baseActions = _bubbleActions(m.content || '', false, idx);
            const actions = baseActions.replace('</div>', editBtn + forkBtn + '</div>');
            out.push(`<div class="chat chat-end" data-msg-idx="${idx}" data-node-id="${nodeId || ''}">
                <div class="chat-bubble chat-bubble-primary text-xs whitespace-pre-wrap relative chat-bubble-host">${_htmlEscape(m.content || '')}${actions}</div>
                ${navHTML}
            </div>`);
            i++;
            continue;
        }

        if (role === 'assistant') {
            // tool_calls case is handled by the thinking-run walker above —
            // here we only handle the speak-only assistant turn.
            const content = (m.content || '').trim();
            const body = content ? `<div class="whitespace-pre-wrap">${_htmlEscape(content)}</div>` : '';
            out.push(`<div class="chat chat-start"><div class="chat-bubble text-xs relative chat-bubble-host">${body}${_bubbleActions(content, true, idx)}</div></div>`);
            i++;
            continue;
        }
        // Unknown role — skip silently.
        i++;
    }
    log.innerHTML = out.join('');
    log.scrollTop = log.scrollHeight;
}
window._renderChatLog = _renderChatLog;

// Begin inline edit of a user message identified by tree node id.
// Replaces the bubble's content with a textarea + Send/Cancel buttons.
// On Send: forks at this node with the new content + truncates the
// active chain + re-fires _sendChatMessage so the assistant replies
// to the edited turn. On Cancel: re-renders to restore.
window._chatBeginEdit = function (nodeId, btn) {
    const tree = _chatGetTree();
    const node = tree.nodes[nodeId];
    if (!node || node.role !== 'user') return;
    const bubble = btn.closest('.chat-bubble');
    if (!bubble) return;
    const original = node.content || '';
    bubble.innerHTML = `<div class="chat-bubble-edit-form">
        <textarea class="chat-bubble-edit-input textarea textarea-bordered text-xs w-full"
            rows="3">${_htmlEscape(original)}</textarea>
        <div class="flex gap-1 mt-1 justify-end">
            <button type="button" class="btn btn-xs btn-ghost"
                onclick='_renderChatLog()'>Cancel</button>
            <button type="button" class="btn btn-xs btn-primary"
                onclick='_chatCommitEdit("${nodeId}", this)'>Send fork →</button>
        </div>
    </div>`;
    const ta = bubble.querySelector('textarea');
    if (ta) {
        ta.focus();
        // Move cursor to end.
        ta.selectionStart = ta.selectionEnd = ta.value.length;
    }
};

window._chatCommitEdit = async function (nodeId, btn) {
    const ta = btn.closest('.chat-bubble-edit-form').querySelector('textarea');
    const newText = (ta && ta.value || '').trim();
    if (!newText) return;
    // Fork: creates a sibling user node + sets it as active leaf +
    // mirrors to legacy history (truncated to before the parent +
    // ending at the new node).
    const newId = _chatForkAt(nodeId, newText);
    if (!newId) return;
    // Re-render to show the new branch immediately.
    _renderChatLog();
    // Trigger the assistant's reply for the new branch. _sendChatMessage
    // reads the input field; we stuff our edited text in there + invoke
    // it. Behavior matches "send" except history is already correct.
    const input = document.getElementById('subjects-chat-input');
    if (input) {
        // The forked chain ALREADY ends with the user's edited message,
        // so don't re-append by submitting through the input. Instead
        // call /chat directly with the active history.
        const sendBtn = document.getElementById('subjects-chat-send');
        if (sendBtn) {
            sendBtn.disabled = true;
            let stxt = document.getElementById('subjects-chat-send-text') || sendBtn;
            stxt.textContent = '…';
        }
        const chain = _chatActiveChain();
        try {
            const d = await _fetchChatWithRetry(chain);
            if (d && d.ok && Array.isArray(d.messages)) {
                _setChatHistory(d.messages);
                _renderChatLog();
            } else {
                const after = _getChatHistory();
                after.push({ role: 'assistant', content: `Error: ${(d && d.error) || 'chat request failed'}` });
                _setChatHistory(after);
                _renderChatLog();
            }
        } catch (e) {
            const after = _getChatHistory();
            after.push({ role: 'assistant', content: `Network error: ${e.message}` });
            _setChatHistory(after);
            _renderChatLog();
        } finally {
            if (sendBtn) {
                sendBtn.disabled = false;
                let stxt = document.getElementById('subjects-chat-send-text') || sendBtn;
                stxt.textContent = 'Send';
            }
        }
    }
};

// Render a JSON value as a styled key-value tree instead of raw text.
// Each scalar gets a type-aware color:
//   string  → quoted, theme primary
//   number  → mono, theme secondary
//   bool    → italic, theme accent
//   null    → muted "—"
//   object  → recursive nested .kv-pretty block
//   array   → recursive numbered nested block
// Used by the chat-log tool-call args + tool-result body so the
// `<pre>${JSON.stringify}</pre>` blob becomes a scannable list.
function _renderKvPretty(value, depth) {
    depth = depth || 0;
    if (value === null || value === undefined) {
        return '<span class="kv-null">—</span>';
    }
    const t = typeof value;
    if (t === 'string') {
        return `<span class="kv-string">${_htmlEscape(value)}</span>`;
    }
    if (t === 'number') {
        return `<span class="kv-number">${value}</span>`;
    }
    if (t === 'boolean') {
        return `<span class="kv-bool">${value ? 'true' : 'false'}</span>`;
    }
    if (Array.isArray(value)) {
        if (!value.length) return '<span class="kv-empty">[]</span>';
        return `<div class="kv-pretty kv-array">${value.map((v, i) =>
            `<div class="kv-row"><span class="kv-key">${i}</span>${_renderKvPretty(v, depth + 1)}</div>`
        ).join('')}</div>`;
    }
    if (t === 'object') {
        const entries = Object.entries(value);
        if (!entries.length) return '<span class="kv-empty">{}</span>';
        return `<div class="kv-pretty kv-object">${entries.map(([k, v]) =>
            `<div class="kv-row"><span class="kv-key">${_htmlEscape(k)}</span>${_renderKvPretty(v, depth + 1)}</div>`
        ).join('')}</div>`;
    }
    return _htmlEscape(String(value));
}
window._renderKvPretty = _renderKvPretty;
// Best-effort JSON-or-string → pretty render. Falls back to a `<pre>`
// blob if the input doesn't parse as JSON. Used for tool-result content
// where we may receive either a JSON string or already-parsed object.
function _renderKvPrettySafe(raw) {
    if (raw == null) return '<span class="kv-null">—</span>';
    let parsed = raw;
    if (typeof raw === 'string') {
        try { parsed = JSON.parse(raw); }
        catch (_) {
            // Not JSON — render as a plain pre block, theme-aligned.
            return `<pre class="kv-raw whitespace-pre-wrap break-all text-[10px]">${_htmlEscape(raw)}</pre>`;
        }
    }
    return _renderKvPretty(parsed, 0);
}

// Copy a bubble's text to clipboard. Brief tick-flash via ::after on
// success so the user sees the action landed without a disruptive toast.
window._copyBubbleText = function (btn, text) {
    if (!text || !navigator.clipboard) return;
    navigator.clipboard.writeText(text.replace(/&#10;/g, '\n').replace(/&quot;/g, '"'))
        .then(() => {
            if (btn) {
                btn.classList.add('copied');
                setTimeout(() => btn.classList.remove('copied'), 900);
            }
        })
        .catch(() => { });
};

// Re-ask the LLM for a fresh assistant turn at index `msgIdx`. Truncates
// the chat history to the user message that PRECEDED this assistant turn,
// then re-fires _sendChatMessage equivalent. Easier than polluting
// /chat with a "regenerate" parameter — we just rewind history and
// re-submit the last user message verbatim.
window._refreshAssistantTurn = function (msgIdx) {
    const history = (typeof _getChatHistory === 'function') ? _getChatHistory() : [];
    if (!history.length || msgIdx < 0 || msgIdx >= history.length) return;
    // Walk backward from msgIdx to find the preceding user message.
    let userIdx = -1;
    for (let i = Math.min(msgIdx - 1, history.length - 1); i >= 0; i--) {
        if (history[i] && history[i].role === 'user') { userIdx = i; break; }
    }
    if (userIdx < 0) return;
    const userText = history[userIdx].content || '';
    if (!userText.trim()) return;
    // Truncate history to BEFORE the user turn — _sendChatMessage will
    // re-append it from the input field.
    const trimmed = history.slice(0, userIdx);
    if (typeof _setChatHistory === 'function') _setChatHistory(trimmed);
    _renderChatLog();
    const input = document.getElementById('subjects-chat-input');
    if (input) {
        input.value = userText;
        if (typeof _sendChatMessage === 'function') _sendChatMessage();
    }
};

// ---------------------------------------------------------------------------
// Shared chat fetch with exponential-backoff retry.
// Only retries on pure network failures (TypeError / Failed to fetch).
// Application-level errors (4xx, 5xx JSON) are returned immediately.
//
// Between attempts we push a temporary assistant message showing the
// retry countdown so the user knows the client is still trying. That
// placeholder is replaced in-place once either a real response lands
// or all retries are exhausted.
//
// Backoff schedule (maxRetries = 5): 2 s, 4 s, 8 s, 16 s, 32 s.
// ---------------------------------------------------------------------------
async function _fetchChatWithRetry(messages, maxRetries) {
    if (maxRetries === undefined) maxRetries = 5;
    const BASE_DELAY_MS = 2000;

    // We track whether we already inserted a placeholder bubble so we
    // can overwrite it on the next attempt instead of appending a new one.
    let placeholderPushed = false;

    const _setPlaceholder = (text) => {
        const hist = _getChatHistory();
        if (placeholderPushed) {
            // Overwrite the last entry (the placeholder).
            hist[hist.length - 1] = { role: 'assistant', content: text, _retry_placeholder: true };
        } else {
            hist.push({ role: 'assistant', content: text, _retry_placeholder: true });
            placeholderPushed = true;
        }
        _setChatHistory(hist);
        _renderChatLog();
    };

    const _clearPlaceholder = () => {
        if (!placeholderPushed) return;
        const hist = _getChatHistory();
        if (hist.length && hist[hist.length - 1]._retry_placeholder) {
            hist.pop();
            _setChatHistory(hist);
        }
        placeholderPushed = false;
    };

    let lastErr;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const r = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages }),
            });
            // HTTP-level success — clear any retry placeholder and return
            // the parsed body to the caller for normal ok/error handling.
            _clearPlaceholder();
            return await r.json();
        } catch (e) {
            lastErr = e;
            const retriesLeft = maxRetries - attempt;
            if (retriesLeft <= 0) break;

            const delaySec = (BASE_DELAY_MS / 1000) * Math.pow(2, attempt);
            _setPlaceholder(`⚠ Network error: ${e.message} — retrying in ${delaySec}s… (${retriesLeft} attempt${retriesLeft === 1 ? '' : 's'} left)`);

            // Countdown: update the bubble every second so the user sees
            // the timer tick down rather than a static message.
            for (let remaining = delaySec; remaining > 0; remaining--) {
                await new Promise(res => setTimeout(res, 1000));
                _setPlaceholder(`⚠ Network error: ${lastErr.message} — retrying in ${remaining - 1 > 0 ? remaining - 1 + 's' : 'now'}… (${retriesLeft} attempt${retriesLeft === 1 ? '' : 's'} left)`);
            }
        }
    }

    // All retries exhausted — clear placeholder, let caller render final error.
    _clearPlaceholder();
    throw lastErr;
}
window._fetchChatWithRetry = _fetchChatWithRetry;

// Chat send + pending-message queue
//
// When the LLM is in-flight (`_chatInflight === true`), pressing Send
// stashes the new text into `_chatPendingMessage` instead of dropping it.
// On completion, _sendChatMessage drains the pending slot by recursing
// once with the queued text. Single-slot, replace-with-newest semantics:
// double-types preserve the most recent intent. Cancel via the × button
// on the pending bubble before the in-flight call returns.
// ---------------------------------------------------------------------------
let _chatInflight = false;
let _chatPendingMessage = null;

function _renderPendingChat() {
    const el = document.getElementById('subjects-chat-pending');
    const txt = document.getElementById('subjects-chat-pending-text');
    if (!el || !txt) return;
    if (_chatPendingMessage) {
        txt.textContent = _chatPendingMessage;
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
        txt.textContent = '';
    }
}

function _cancelPendingChat() {
    _chatPendingMessage = null;
    _renderPendingChat();
}
window._cancelPendingChat = _cancelPendingChat;

async function _sendChatMessage() {
    const input = document.getElementById('subjects-chat-input');
    const sendBtn = document.getElementById('subjects-chat-send');
    if (!input) return;
    const text = (input.value || '').trim();
    if (!text) return;

    // If the LLM is already thinking, queue this message instead of
    // dropping it. Single-slot: replace-with-newest preserves the
    // latest intent (rapid retypes win). Clear input so the user can
    // see their message was accepted; show the pending bubble so they
    // know it'll fire after the current reply lands.
    if (_chatInflight) {
        _chatPendingMessage = text;
        input.value = '';
        _renderPendingChat();
        return;
    }

    input.value = '';
    _chatInflight = true;
    if (sendBtn) {
        sendBtn.disabled = true;
        let stxt = document.getElementById('subjects-chat-send-text') || sendBtn;
        stxt.textContent = '…';
    }
    const history = _getChatHistory();
    history.push({ role: 'user', content: text });
    _setChatHistory(history);
    _renderChatLog();
    try {
        const d = await _fetchChatWithRetry(history);
        if (d && d.ok && Array.isArray(d.messages)) {
            _setChatHistory(d.messages);
            _renderChatLog();
        } else {
            const fallback = (d && d.error) || 'chat request failed';
            const after = _getChatHistory();
            after.push({ role: 'assistant', content: `Error: ${fallback}` });
            _setChatHistory(after);
            _renderChatLog();
        }
    } catch (e) {
        const after = _getChatHistory();
        after.push({ role: 'assistant', content: `Network error: ${e.message}` });
        _setChatHistory(after);
        _renderChatLog();
    }
    _chatInflight = false;
    if (sendBtn) {
        sendBtn.disabled = false;
        let stxt = document.getElementById('subjects-chat-send-text') || sendBtn;
        stxt.textContent = 'Send';
    }
    if (input) input.focus();
    // Refresh reply suggestions against the new latest assistant turn so
    // the user has 4 contextual continuation chips ready.
    if (typeof _renderChatReplies === 'function') _renderChatReplies();

    // Drain pending: if the user queued another message during this
    // turn, fire it now. Stash + clear the slot before recursing so a
    // network error in the queued send doesn't loop on stale state.
    if (_chatPendingMessage) {
        const queued = _chatPendingMessage;
        _chatPendingMessage = null;
        _renderPendingChat();
        if (input) input.value = queued;
        await _sendChatMessage();
    }
}
window._sendChatMessage = _sendChatMessage;

// Single click handler on the big queue button — forks on mode.
async function _subjectsAction() {
    const mode = _getSubjectsMode();
    if (mode === 'chat') return; // chat input is the only action; big button is hidden
    // STORY MODE — sync the persisted beat lines into p-core before
    // falling through to toggleInfinity(). Agent G's overhaul made the
    // story pane always-visible, chip clicks append new beats, manual
    // typing creates new rows on Enter — but toggleInfinity() reads
    // p-core for its subjects, so we still need to materialise the
    // beat list there at queue time. Without this sync the user would
    // queue an empty subject list (the bug the user just reported as
    // "Queue reverted to older UI and didn't submit").
    if (mode === 'endless') {
        const lines = (typeof _loadEndlessLogLines === 'function') ? _loadEndlessLogLines() : [];
        const cleaned = (lines || []).map(s => (s || '').trim()).filter(Boolean);
        if (cleaned.length) {
            const ta = document.getElementById('p-core');
            if (ta) {
                ta.value = cleaned.join('\n');
                ta.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    }
    // Simple + Raw + Story all hit the legacy queue path. Raw exposes
    // the per-stage prompt panel for power users to pre-fill via AI
    // Magic; Simple keeps the textarea as the only input; Story turns
    // the per-beat list into the multi-line subject list (above).
    if (typeof toggleInfinity === 'function') return toggleInfinity();
}
window._subjectsAction = _subjectsAction;

async function _startEndlessStory() {
    if (_endlessRunning) return;
    const ta = document.getElementById('p-core');
    if (!ta) return;
    let seed = (ta.value || '').trim();
    if (!seed) {
        // I'm Feeling Lucky — fetch a single story-opener from the LLM.
        try {
            const r = await fetch('/subjects/suggest?n=1&opener=1');
            const d = await r.json();
            const arr = (d && d.suggestions) || [];
            seed = (arr[0] || '').trim();
        } catch (_) { }
        if (!seed) {
            console.warn('I\'m Feeling Lucky: no opener returned');
            return;
        }
        ta.value = seed;
        ta.dispatchEvent(new Event('input', { bubbles: true }));
    }
    _endlessRunning = true;
    // Snapshot start-time (ms epoch) so _stitchEndlessStory can later
    // ask /assets for FINAL_*.mp4 clips that completed AFTER this
    // moment — i.e. the clips this story produced. Survives Submit
    // (we want to be able to stitch after the story ends) and is
    // cleared on Reset.
    window._endlessStoryStartTs = Date.now();
    document.body.classList.add('endless-running');
    ta.readOnly = true;
    ta.classList.add('opacity-70');
    // Reset the story log to just the seed AND persist it so layout
    // switches / reloads can rehydrate (see _hydrateEndlessLog below).
    // Storage is a list of rows; the seed is the first row.
    _persistEndlessLogLines(seed ? [seed] : []);
    // New story → active beat is the seed (idx 0). Resetting here keeps a
    // stale active-idx from a prior story from pointing past the new end.
    if (typeof _setEndlessActiveBeatIdx === 'function') _setEndlessActiveBeatIdx(0);
    if (typeof _renderEndlessLog === 'function') _renderEndlessLog();
    const pane = document.getElementById('subjects-story-pane');
    if (pane) pane.classList.remove('hidden');
    // Start with EXACTLY ONE row using the currently-selected default
    // suggestion prompt — equivalent to pressing the + button once.
    // The user adds more rows manually via + (each new + uses whatever
    // their dropdown is set to AT THAT MOMENT). Previous behavior of
    // pre-loading the saved _ENDLESS_ROW_PROMPTS_KEY (often 4+ rows)
    // surprised users who'd just pressed Start Story expecting one
    // fresh start.
    try {
        const defaultId = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : 'yes-and';
        _setEndlessRowPrompts([defaultId]);
    } catch (_) { }
    // Kick the existing endless cycle on (it reads the hidden checkbox).
    const t = document.getElementById('endless-story-toggle');
    if (t && !t.checked) {
        t.checked = true;
        t.dispatchEvent(new Event('change', { bubbles: true }));
    }
    _updateSubjectsActionLabel();
    // + button gates on _endlessRunning — repaint so it un-disables now
    // that the story is live.
    if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
    // Fire one fresh batch immediately so the user sees movement.
    if (typeof regenSuggestions === 'function') regenSuggestions().catch(() => { });
}

// Stop Endless mode and tear down its UI state. `clearLog`:
//   true  → reset variant: also wipes the story log (destructive,
//           confirmed by caller)
//   false → submit variant: leaves the log so the user can copy / inspect
//
// The Story-mode redesign removed the "Start Story" gate, so the pane
// itself stays visible whenever subj-mode is endless. This function
// only tears down ephemeral state (the running flag, the log, the
// auto-cycle toggle); the mode pill is the boundary.
function _endEndlessStory(clearLog) {
    if (clearLog === undefined) clearLog = true; // legacy callers
    _endlessRunning = false;
    document.body.classList.remove('endless-running');
    const ta = document.getElementById('p-core');
    if (ta) {
        ta.readOnly = false;
        ta.classList.remove('opacity-70');
    }
    const pane = document.getElementById('subjects-story-pane');
    // Hide the pane only when leaving endless mode entirely (caller
    // _setSubjectsMode handles that path). Pure Submit/Reset stays in
    // endless and keeps the pane visible — the log just gets cleared.
    const stillEndless = (typeof _getSubjectsMode === 'function')
        && _getSubjectsMode() === 'endless';
    if (pane && clearLog && !stillEndless) pane.classList.add('hidden');
    if (clearLog) {
        const log = document.getElementById('subjects-story-log');
        if (log) log.innerHTML = '';
        _clearEndlessLog();
        // Clear the active-beat pointer too — otherwise the next Start
        // Story rehydrates with whatever idx the previous story left
        // behind, which can outlive the seed-only single-beat reset.
        try { localStorage.removeItem(_ENDLESS_ACTIVE_BEAT_KEY); } catch (_) { }
    }
    // Stop the cycle.
    const t = document.getElementById('endless-story-toggle');
    if (t && t.checked) {
        t.checked = false;
        t.dispatchEvent(new Event('change', { bubbles: true }));
    }
    // If we're still in endless mode (e.g. user pressed Submit/Reset to
    // start a fresh story), re-arm running + reseed the empty first
    // beat so the per-beat input UI keeps the same shape after the
    // wipe. This avoids a "blank pane" state where the user has to
    // toggle modes to get a typeable row back.
    if (stillEndless) {
        _endlessRunning = true;
        document.body.classList.add('endless-running');
        try {
            const existing = _loadEndlessLogLines();
            if (!existing.length) {
                _persistEndlessLogLines(['']);
                _setEndlessActiveBeatIdx(0);
                _renderEndlessLog();
            }
        } catch (_) { }
    }
    _updateSubjectsActionLabel();
    // + button gates on _endlessRunning — repaint so it re-disables now
    // that the story has ended.
    if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
}
window._endEndlessStory = _endEndlessStory;

// Submit — stop the auto-cycle and tear down the story UI. Same destructive
// teardown as Reset (story log + pane go away), just without the confirm
// prompt: pressing Submit is itself the user's "I'm done" intent. Already-
// queued clips keep running on the worker side; this only clears the
// composer surface so the user can move on to the next thing.
function _submitEndlessStory() {
    _endEndlessStory(true);
    // Also empty the seed textarea so the next Start Story doesn't inherit
    // the just-submitted seed — mirrors what the user expects after a
    // successful submit ("the story stuff disappears").
    const ta = document.getElementById('p-core');
    if (ta) {
        ta.value = '';
        ta.dispatchEvent(new Event('input', { bubbles: true }));
    }
}
window._submitEndlessStory = _submitEndlessStory;

// Reset — destructive: clears the log and hides the pane. Confirms first.
// Also clears the stitch-cutoff timestamp so a subsequent Stitch click
// (after starting a NEW story) doesn't pull in clips from the old one.
function _resetEndlessStory() {
    if (!confirm('Reset story? This clears the story log. Already-queued clips keep running.')) return;
    _endEndlessStory(true);
    window._endlessStoryStartTs = 0;
}
window._resetEndlessStory = _resetEndlessStory;

function _copyEndlessStory() {
    // Read from persisted list (the editable rows would otherwise leak
    // the per-row "×" button glyphs into log.textContent).
    const lines = _loadEndlessLogLines();
    const text = lines.join('\n').trim();
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('subjects-story-copy');
        if (btn) {
            btn.classList.add('text-success');
            setTimeout(() => btn.classList.remove('text-success'), 900);
        }
    }).catch(err => console.warn('copy story failed', err));
}
window._copyEndlessStory = _copyEndlessStory;

// Stitch every FINAL_*.mp4 produced since the current (or just-submitted)
// story started into one combined STORY_<ts>.mp4 via /story/stitch.
// Strategy: capture the start timestamp when _startEndlessStory fires
// (window._endlessStoryStartTs), then on click pull /assets, filter
// to FINAL_*.mp4 with mtime >= start_ts, sort by mtime ascending
// (chronological), confirm count + total duration with the user,
// and POST to /story/stitch. Server runs ffmpeg concat (stream copy
// — fast + lossless when source clips share codec params, which they
// do since the same model + settings produced them).
// Click-to-queue flow:
//   1. If queue is empty AND there are already FINAL_*.mp4 clips for
//      this story → fire the stitch right now (legacy fast path).
//   2. If queue has pending/working items → schedule a "pending stitch"
//      and let _updateQueueStatusChip's tick handler fire it the moment
//      total = 0. Button label changes to "Stitch queued (waiting for N)"
//      and a click while queued cancels.
//   3. If neither → toast "no clips yet, queue something first".
//
// The pending state is held on `window._pendingStitch = { startTs, queuedAt }`
// so the WS tick handler can spot it without poking module-private state.
async function _stitchEndlessStory() {
    const startTs = window._endlessStoryStartTs || 0;
    if (!startTs) {
        if (typeof _toast === 'function') _toast('No story start time recorded — start a story first', 'warning');
        return;
    }
    // Toggle: if a pending stitch is already scheduled, this click cancels it.
    if (window._pendingStitch) {
        window._pendingStitch = null;
        _refreshStitchButton();
        if (typeof _toast === 'function') _toast('Stitch cancelled', 'warning');
        return;
    }
    // Snapshot current queue depth via the chip helper's last-seen value.
    const total = (typeof _updateQueueStatusChip === 'function'
        && typeof _updateQueueStatusChip._prevTotal === 'number')
        ? _updateQueueStatusChip._prevTotal : 0;
    if (total > 0) {
        // Schedule — don't fire yet. The WS tick handler will trigger
        // _runStitchNow() the moment total drops to 0.
        window._pendingStitch = { startTs, queuedAt: Date.now() };
        _refreshStitchButton();
        if (typeof _toast === 'function') {
            _toast(`Stitch queued — will run when ${total} clip${total === 1 ? '' : 's'} finish${total === 1 ? 'es' : ''}.`, 'success');
        }
        return;
    }
    // Queue already empty — fire immediately.
    return _runStitchNow(startTs, { confirmFirst: true });
}
window._stitchEndlessStory = _stitchEndlessStory;

// Update the stitch button's label/state based on the current pending
// flag and last-seen queue depth. Called from the click handler and
// from every WS tick (via _updateQueueStatusChip) so the user sees the
// "(waiting for N)" count tick down as clips finish.
function _refreshStitchButton() {
    const btn = document.getElementById('subjects-story-stitch');
    if (!btn) return;
    const lbl = btn.querySelector('span:last-of-type') || btn;
    const pending = window._pendingStitch;
    if (!pending) {
        lbl.textContent = 'Stitch';
        btn.classList.remove('btn-warning');
        btn.title = 'Concatenate this story\'s FINAL_*.mp4 clips into one mp4. Click while queued items pending → schedules stitch to run when queue drains.';
        return;
    }
    const total = (typeof _updateQueueStatusChip._prevTotal === 'number')
        ? _updateQueueStatusChip._prevTotal : 0;
    lbl.textContent = total > 0
        ? `Stitch queued (waiting for ${total})`
        : 'Stitch queued (running…)';
    btn.classList.add('btn-warning');
    btn.title = 'Stitch is scheduled to fire when the queue drains. Click again to cancel.';
}
window._refreshStitchButton = _refreshStitchButton;

// Tick-driven trigger — _updateQueueStatusChip calls this on every WS
// state event. When a pending stitch is set AND queue total == 0, fire
// the actual ffmpeg call (no confirm — the user already confirmed by
// clicking Stitch earlier; queueing-vs-immediate is just timing).
function _maybeFirePendingStitch() {
    const pending = window._pendingStitch;
    if (!pending) return;
    const total = (typeof _updateQueueStatusChip._prevTotal === 'number')
        ? _updateQueueStatusChip._prevTotal : -1;
    if (total !== 0) return;
    // Small debounce — wait at least 2 s after queue empties so any
    // straggler "Final Merge" stage that's about to write FINAL_*.mp4
    // has a moment to finish writing its file before we glob /assets.
    if (!pending._emptyAt) {
        pending._emptyAt = Date.now();
        return;
    }
    if (Date.now() - pending._emptyAt < 2000) return;
    // Clear the flag BEFORE firing so subsequent ticks don't re-trigger
    // while ffmpeg is still running.
    const startTs = pending.startTs;
    window._pendingStitch = null;
    _refreshStitchButton();
    _runStitchNow(startTs, { confirmFirst: false });
}
window._maybeFirePendingStitch = _maybeFirePendingStitch;

// The actual ffmpeg-concat path — split out so both the immediate-fire
// case and the deferred-after-queue-drain case go through the same
// fetch + toast handling.
async function _runStitchNow(startTs, opts) {
    opts = opts || {};
    let assets = [];
    try {
        const r = await fetch('/assets?limit=200');
        const d = await r.json();
        assets = (d && d.assets) || [];
    } catch (e) {
        if (typeof _toast === 'function') _toast(`Stitch: /assets fetch failed (${e.message})`, 'error');
        return;
    }
    const startSec = Math.floor(startTs / 1000);
    const candidates = assets
        .filter(a => a && a.file && a.file.startsWith('FINAL_') && a.file.endsWith('.mp4'))
        .filter(a => (a.mtime || 0) >= startSec - 5)
        .sort((a, b) => (a.mtime || 0) - (b.mtime || 0));
    if (!candidates.length) {
        if (typeof _toast === 'function') _toast('Stitch: no FINAL_*.mp4 clips have completed for this story yet', 'warning');
        return;
    }
    const filenames = candidates.map(a => a.file);
    if (opts.confirmFirst) {
        const ok = confirm(
            `Stitch ${filenames.length} clip${filenames.length === 1 ? '' : 's'} into one mp4?\n\n`
            + filenames.map((f, i) => `${i + 1}. ${f}`).join('\n')
            + '\n\n(Stream copy — fast and lossless. Output: STORY_<timestamp>.mp4)'
        );
        if (!ok) return;
    }
    const btn = document.getElementById('subjects-story-stitch');
    if (btn) { btn.disabled = true; btn.classList.add('opacity-50'); }
    try {
        const res = await fetch('/story/stitch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filenames }),
        });
        const data = await res.json();
        if (data && data.ok) {
            if (typeof _toast === 'function') {
                _toast(`Stitched ${data.n_inputs} clips → ${data.output}`, 'success');
            }
        } else {
            if (typeof _toast === 'function') {
                _toast(`Stitch failed: ${(data && data.error) || res.statusText}`, 'error');
            }
        }
    } catch (e) {
        if (typeof _toast === 'function') _toast(`Stitch network error: ${e.message}`, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.classList.remove('opacity-50'); }
    }
}

// Story-log persistence — survives layout switches (mobile-nav prev/next,
// View dropdown), full reloads, etc. Stored under a single key alongside
// the existing _endlessStoryStartTs so Reset/Submit can wipe both atomically.
//
// Storage shape: JSON-encoded array of strings (one per row). The loader
// also accepts the legacy `\n`-joined plain-string shape and migrates it
// on first read so existing user state isn't lost when the editable-row
// UI lands.
const _ENDLESS_STORY_LOG_KEY = 'slopfinity-endless-story-log';

function _serializeEndlessLog(lines) {
    try { return JSON.stringify(Array.isArray(lines) ? lines : []); }
    catch (_) { return '[]'; }
}

// Read the saved log as an array of lines. Handles both the new
// JSON-array shape and the legacy `\n`-joined string shape — splitting
// the legacy form on newlines preserves the user's content even when
// they upgrade across this change.
function _loadEndlessLogLines() {
    let raw;
    try { raw = localStorage.getItem(_ENDLESS_STORY_LOG_KEY); }
    catch (_) { return []; }
    if (!raw) return [];
    // New shape: JSON array
    if (raw.length && raw.charCodeAt(0) === 91 /* '[' */) {
        try {
            const arr = JSON.parse(raw);
            if (Array.isArray(arr)) return arr.map(s => String(s || ''));
        } catch (_) { /* fall through to legacy */ }
    }
    // Legacy shape: plain `\n`-joined string. Split + drop empty trailing
    // lines so re-rendering doesn't show ghost empty rows.
    const lines = String(raw).split('\n');
    while (lines.length && !lines[lines.length - 1]) lines.pop();
    return lines;
}

function _persistEndlessLogLines(lines) {
    try { localStorage.setItem(_ENDLESS_STORY_LOG_KEY, _serializeEndlessLog(lines)); }
    catch (_) { }
}

// Back-compat shim — older callers passed a single string. Now they
// pass either a string (treated as the seed = single row) or an array.
function _persistEndlessLog(textOrLines) {
    if (Array.isArray(textOrLines)) {
        _persistEndlessLogLines(textOrLines);
        return;
    }
    const s = String(textOrLines || '');
    _persistEndlessLogLines(s ? [s] : []);
}
function _loadEndlessLog() {
    // Returns the legacy `\n`-joined string view for callers that still
    // want plaintext (e.g. _copyEndlessStory). Internal renderers use
    // _loadEndlessLogLines directly.
    return _loadEndlessLogLines().join('\n');
}
function _clearEndlessLog() {
    try { localStorage.removeItem(_ENDLESS_STORY_LOG_KEY); } catch (_) { }
}

// === Active-beat tracking =============================================
// Each beat is one editable input. The "active beat" is the one that
// currently has focus (or, if nothing's focused, the last one focused).
// Suggestion chips write into the active beat instead of appending to a
// global log. Persisted across reloads under its own key so the user's
// position survives layout swaps.
const _ENDLESS_ACTIVE_BEAT_KEY = 'slopfinity-endless-active-beat-idx';

function _getEndlessActiveBeatIdx() {
    try {
        const v = parseInt(localStorage.getItem(_ENDLESS_ACTIVE_BEAT_KEY) || '0', 10);
        return Number.isInteger(v) && v >= 0 ? v : 0;
    } catch (_) { return 0; }
}
function _setEndlessActiveBeatIdx(idx) {
    if (!Number.isInteger(idx) || idx < 0) return;
    try { localStorage.setItem(_ENDLESS_ACTIVE_BEAT_KEY, String(idx)); }
    catch (_) { }
    // Repaint just the active highlight without a full re-render (avoids
    // blowing away focus on the input the user just clicked into).
    try {
        const log = document.getElementById('subjects-story-log');
        if (!log) return;
        log.querySelectorAll('.endless-row').forEach(row => {
            const rIdx = parseInt(row.getAttribute('data-row-idx'), 10);
            row.classList.toggle('beat-active', rIdx === idx);
        });
    } catch (_) { }
}
window._getEndlessActiveBeatIdx = _getEndlessActiveBeatIdx;
window._setEndlessActiveBeatIdx = _setEndlessActiveBeatIdx;

// Render the editable beats from the persisted line list into
// #subjects-story-log. Each row is: drag-handle (≡) + single-line
// <input type="text"> + per-row + (add new beat below) + × (delete).
// The active beat gets `beat-active` so CSS can outline it. Edits commit
// on `input` so every keystroke immediately persists; deletes drop the
// row and re-render. Add-next inserts a fresh empty beat below and makes
// it active. Pressing Enter in any input also adds a fresh empty beat
// below + focuses it.
function _renderEndlessLog() {
    const log = document.getElementById('subjects-story-log');
    if (!log) return;
    const lines = _loadEndlessLogLines();
    if (!lines.length) {
        log.innerHTML = `<div class="endless-empty-state flex flex-col items-center justify-center gap-1 py-6 text-center text-xs text-base-content/50 italic select-none">
            <span class="text-2xl opacity-60" aria-hidden="true">📜</span>
            <span>No beats yet — type your first one above, or click a suggestion chip to seed it.</span>
        </div>`;
        return;
    }
    // Clamp the persisted active idx so it can't point past the end after
    // a delete-from-the-bottom or a fresh story load.
    let activeIdx = _getEndlessActiveBeatIdx();
    if (activeIdx >= lines.length) {
        activeIdx = lines.length - 1;
        _setEndlessActiveBeatIdx(activeIdx);
    }
    const rows = lines.map((line, idx) => {
        // <input value> uses the attribute escape, but the HTML-string
        // assemble here means we still want HTML-safe escaping for "&,
        // <, >, ', ". _htmlEscape handles all five.
        const safe = _htmlEscape(line || '');
        const activeCls = idx === activeIdx ? ' beat-active' : '';
        return `<div class="endless-row beat-row group flex items-center gap-1 px-1 py-0.5 rounded hover:bg-base-300/40${activeCls}"
                     data-row-idx="${idx}" draggable="true">
            <span class="beat-drag-handle flex-none self-stretch flex items-center text-base-content/60 hover:text-base-content/90 select-none cursor-grab"
                  title="Drag to reorder this beat" aria-label="Drag handle">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-3 h-3">
                    <line x1="4" y1="8" x2="20" y2="8" />
                    <line x1="4" y1="12" x2="20" y2="12" />
                    <line x1="4" y1="16" x2="20" y2="16" />
                </svg>
            </span>
            <input type="text" spellcheck="false"
                  class="endless-row-text beat-input flex-1 min-w-0 bg-transparent outline-none focus:bg-base-100/60 rounded px-1 text-sm"
                  data-row-idx="${idx}"
                  placeholder="Type a beat — Enter for next row"
                  value="${safe}" />
            <button type="button" class="endless-row-add btn btn-ghost btn-xs btn-square text-primary/70 hover:text-primary opacity-75 group-hover:opacity-100 flex-none"
                    data-row-idx="${idx}" title="Add a new beat below this one" aria-label="Add beat below">+</button>
            <button type="button" class="endless-row-del btn btn-ghost btn-xs btn-square text-error/70 hover:text-error opacity-75 group-hover:opacity-100 flex-none"
                    data-row-idx="${idx}" title="Delete this beat" aria-label="Delete beat">×</button>
        </div>`;
    });
    log.innerHTML = rows.join('');
    log.scrollTop = log.scrollHeight;
}
window._renderEndlessLog = _renderEndlessLog;

// Suggestion-chip routing: chip click APPENDS a NEW beat row prefilled
// with the chip text, marks it active, and focuses its input. The
// previous behaviour wrote into the currently-active beat (replacing if
// empty, space-appending otherwise) — that surprised users who clicked
// a chip expecting "add this as a new beat" and instead got their
// in-progress beat clobbered. Now every chip click is a new row;
// manual typing in any input still edits that beat in place.
function _storyAcceptIntoActive(text) {
    const t = String(text || '').trim();
    if (!t) return;
    const lines = _loadEndlessLogLines();
    // If the active beat is empty, fill it instead of appending — this
    // keeps the very first chip-click after entering endless mode from
    // leaving an orphan empty row at the top.
    let idx = _getEndlessActiveBeatIdx();
    if (lines.length && idx < lines.length && !String(lines[idx] || '').trim()) {
        lines[idx] = t;
        _persistEndlessLogLines(lines);
        _renderEndlessLog();
        try {
            const log = document.getElementById('subjects-story-log');
            const inp = log && log.querySelector(`.beat-input[data-row-idx="${idx}"]`);
            if (inp) inp.focus();
        } catch (_) { }
        return;
    }
    // Otherwise append a NEW beat at the end + make it active + focus it.
    lines.push(t);
    _persistEndlessLogLines(lines);
    const newIdx = lines.length - 1;
    _setEndlessActiveBeatIdx(newIdx);
    _renderEndlessLog();
    try {
        const log = document.getElementById('subjects-story-log');
        const inp = log && log.querySelector(`.beat-input[data-row-idx="${newIdx}"]`);
        if (inp) inp.focus();
    } catch (_) { }
}
window._storyAcceptIntoActive = _storyAcceptIntoActive;

// Insert a fresh empty beat at `afterIdx + 1`, persist, mark it active,
// re-render, then focus the new input so the user can immediately type
// or click a suggestion chip into it.
function _storyAddBeatAfter(afterIdx) {
    const lines = _loadEndlessLogLines();
    const at = Math.max(0, Math.min(lines.length, afterIdx + 1));
    lines.splice(at, 0, '');
    _persistEndlessLogLines(lines);
    _setEndlessActiveBeatIdx(at);
    _renderEndlessLog();
    // Focus the freshly-rendered input so the next keystroke / chip click
    // lands in the new beat without an extra click.
    try {
        const log = document.getElementById('subjects-story-log');
        const newSpan = log && log.querySelector(`.beat-input[data-row-idx="${at}"]`);
        if (newSpan) newSpan.focus();
    } catch (_) { }
}
window._storyAddBeatAfter = _storyAddBeatAfter;

// Move the beat at `from` to `to` (insertion-style — `to` is where the
// row should END UP after the splice). Persists + re-renders. Used by
// the HTML5 drag-and-drop handlers below.
function _storyReorderBeat(from, to) {
    if (from === to) return;
    const lines = _loadEndlessLogLines();
    if (!Number.isInteger(from) || from < 0 || from >= lines.length) return;
    if (!Number.isInteger(to) || to < 0 || to > lines.length) return;
    const [moved] = lines.splice(from, 1);
    const insertAt = to > from ? to - 1 : to;
    lines.splice(insertAt, 0, moved);
    _persistEndlessLogLines(lines);
    // Active beat follows the dragged row so its highlight stays with the
    // user's intent ("the beat I just moved is the one I'm working on").
    const oldActive = _getEndlessActiveBeatIdx();
    let newActive = oldActive;
    if (oldActive === from) newActive = insertAt;
    else if (from < oldActive && to > oldActive) newActive = oldActive - 1;
    else if (from > oldActive && to <= oldActive) newActive = oldActive + 1;
    _setEndlessActiveBeatIdx(newActive);
    _renderEndlessLog();
}
window._storyReorderBeat = _storyReorderBeat;

// Delegated handlers for the editable beats. One listener per concern
// (input + click + focus + drag-events) on the document — beats wiring
// per-row on every render.
(function _wireEndlessLogEditors() {
    document.addEventListener('input', (e) => {
        const inp = e.target.closest && e.target.closest('#subjects-story-log .endless-row-text');
        if (!inp) return;
        const row = inp.closest('.endless-row');
        const idx = row ? parseInt(row.getAttribute('data-row-idx'), 10) : -1;
        if (!Number.isInteger(idx) || idx < 0) return;
        const lines = _loadEndlessLogLines();
        if (idx >= lines.length) return;
        // <input>.value (single-line) — preserves text exactly without
        // HTML-paste leaks that contenteditable used to allow.
        lines[idx] = (inp.value !== undefined ? inp.value : (inp.textContent || ''));
        _persistEndlessLogLines(lines);
    });
    // Enter in an input → append a NEW empty beat below + focus it.
    // Mirrors the per-row + button so the keyboard-only flow is fast:
    // user types beat → Enter → keeps typing the next beat. Shift+Enter
    // is reserved (input can't break to newline anyway, but we still
    // skip the new-row append on shift so users with muscle-memory
    // multi-line don't surprise themselves).
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' || e.shiftKey) return;
        const inp = e.target.closest && e.target.closest('#subjects-story-log .endless-row-text');
        if (!inp) return;
        const row = inp.closest('.endless-row');
        const idx = row ? parseInt(row.getAttribute('data-row-idx'), 10) : -1;
        if (!Number.isInteger(idx) || idx < 0) return;
        e.preventDefault();
        if (typeof _storyAddBeatAfter === 'function') _storyAddBeatAfter(idx);
    });
    // Focus → mark this beat active. focusin bubbles (focus does not).
    document.addEventListener('focusin', (e) => {
        const inp = e.target.closest && e.target.closest('#subjects-story-log .endless-row-text');
        if (!inp) return;
        const row = inp.closest('.endless-row');
        const idx = row ? parseInt(row.getAttribute('data-row-idx'), 10) : -1;
        if (!Number.isInteger(idx) || idx < 0) return;
        _setEndlessActiveBeatIdx(idx);
    });
    document.addEventListener('click', (e) => {
        const addBtn = e.target.closest && e.target.closest('#subjects-story-log .endless-row-add');
        if (addBtn) {
            e.preventDefault();
            e.stopPropagation();
            const idx = parseInt(addBtn.getAttribute('data-row-idx'), 10);
            if (Number.isInteger(idx) && idx >= 0) _storyAddBeatAfter(idx);
            return;
        }
        const delBtn = e.target.closest && e.target.closest('#subjects-story-log .endless-row-del');
        if (delBtn) {
            e.preventDefault();
            e.stopPropagation();
            const idx = parseInt(delBtn.getAttribute('data-row-idx'), 10);
            if (!Number.isInteger(idx) || idx < 0) return;
            const lines = _loadEndlessLogLines();
            if (idx >= lines.length) return;
            lines.splice(idx, 1);
            _persistEndlessLogLines(lines);
            // Keep active idx in range; if we deleted the active one,
            // step back to the previous beat.
            let active = _getEndlessActiveBeatIdx();
            if (idx === active) active = Math.max(0, active - 1);
            else if (idx < active) active -= 1;
            _setEndlessActiveBeatIdx(Math.max(0, active));
            _renderEndlessLog();
            return;
        }
        // Click anywhere else inside a row → treat the row as the active
        // beat (click a chip inside the drag-handle area still selects).
        const row = e.target.closest && e.target.closest('#subjects-story-log .endless-row');
        if (row) {
            const idx = parseInt(row.getAttribute('data-row-idx'), 10);
            if (Number.isInteger(idx) && idx >= 0) _setEndlessActiveBeatIdx(idx);
        }
    });

    // === HTML5 drag-and-drop reordering =============================
    // We track the source row idx on dragstart, paint a `.drag-over`
    // outline on whichever row the cursor is over, then on drop call
    // _storyReorderBeat(from, to). The drop target is calculated from
    // the row the pointer is over PLUS the cursor's vertical position
    // within that row's bounding box (top half = insert before; bottom
    // half = insert after) — gives the user fine-grained control on
    // adjacent beats without needing a separate gap drop-zone.
    let _dragFromIdx = -1;
    document.addEventListener('dragstart', (e) => {
        const row = e.target.closest && e.target.closest('#subjects-story-log .endless-row');
        if (!row) return;
        _dragFromIdx = parseInt(row.getAttribute('data-row-idx'), 10);
        if (!Number.isInteger(_dragFromIdx)) { _dragFromIdx = -1; return; }
        row.classList.add('dragging');
        try {
            // dataTransfer must be set or some browsers cancel the drag.
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', String(_dragFromIdx));
        } catch (_) { }
    });
    document.addEventListener('dragend', (e) => {
        const log = document.getElementById('subjects-story-log');
        if (log) log.querySelectorAll('.endless-row.dragging, .endless-row.drag-over')
            .forEach(r => r.classList.remove('dragging', 'drag-over'));
        _dragFromIdx = -1;
    });
    document.addEventListener('dragover', (e) => {
        const row = e.target.closest && e.target.closest('#subjects-story-log .endless-row');
        if (!row) return;
        if (_dragFromIdx < 0) return;
        e.preventDefault(); // required to allow drop
        try { e.dataTransfer.dropEffect = 'move'; } catch (_) { }
        const log = document.getElementById('subjects-story-log');
        if (log) log.querySelectorAll('.endless-row.drag-over').forEach(r => {
            if (r !== row) r.classList.remove('drag-over');
        });
        row.classList.add('drag-over');
    });
    document.addEventListener('dragleave', (e) => {
        const row = e.target.closest && e.target.closest('#subjects-story-log .endless-row');
        if (!row) return;
        // Only clear when the pointer truly leaves the row (not just
        // crosses into a child element). relatedTarget === null means
        // the drag left the window.
        if (!e.relatedTarget || !row.contains(e.relatedTarget)) {
            row.classList.remove('drag-over');
        }
    });
    document.addEventListener('drop', (e) => {
        const row = e.target.closest && e.target.closest('#subjects-story-log .endless-row');
        if (!row) return;
        if (_dragFromIdx < 0) return;
        e.preventDefault();
        const toIdx = parseInt(row.getAttribute('data-row-idx'), 10);
        if (!Number.isInteger(toIdx)) return;
        // Top half = drop before this row; bottom half = drop after.
        const rect = row.getBoundingClientRect();
        const after = (e.clientY - rect.top) > rect.height / 2;
        const insertionTo = after ? toIdx + 1 : toIdx;
        const from = _dragFromIdx;
        _dragFromIdx = -1;
        _storyReorderBeat(from, insertionTo);
    });
})();

// Suggestion-chip → story routing.
// Pre-redesign this APPENDED a fresh row per chip; the per-beat redesign
// now writes the chip into the ACTIVE beat input via
// _storyAcceptIntoActive (replacing if empty, space-appending if not).
// Function name kept for back-compat with the call site in
// _buildSuggestChip.
function _appendToEndlessLog(text) {
    if (!_endlessRunning || _getSubjectsMode() !== 'endless') return;
    if (typeof _storyAcceptIntoActive === 'function') {
        _storyAcceptIntoActive(text);
        return;
    }
    // Fallback (shouldn't normally trigger): append as a new beat.
    const lines = _loadEndlessLogLines();
    lines.push(String(text || ''));
    _persistEndlessLogLines(lines);
    _renderEndlessLog();
}
window._appendToEndlessLog = _appendToEndlessLog;

document.addEventListener('DOMContentLoaded', () => {
    _setSubjectsMode(_getSubjectsMode());
    _updateSubjectsActionLabel();
    // Paint any saved story rows on first load so a refresh shows the
    // user's in-progress story (the pane stays hidden until they're in
    // endless mode + have content — see the _setSubjectsMode hydration
    // above for the visibility rule).
    try {
        if (typeof _renderEndlessLog === 'function') _renderEndlessLog();
    } catch (_) { /* ignore */ }
});

// Quick read-only popup for the LLM-rewritten prompt of the active job.
// Lighter than openAssetInfo (which is for files); this is just text.
function showPromptPeek(text, stage) {
    const existing = document.getElementById('prompt-peek-modal');
    if (existing) existing.remove();
    const dlg = document.createElement('dialog');
    dlg.id = 'prompt-peek-modal';
    dlg.className = 'modal';
    // Header labels which stage this prompt feeds. Maps the canonical
    // stage names to the user-facing role ("Image" / "Video" / etc) so
    // the modal title reads "Video prompt" rather than "Video Chains
    // prompt". Falls back to a neutral label when stage is unknown.
    const _stageRoleLabel = {
        'Concept': 'Concept',
        'Base Image': 'Image',
        'Video Chains': 'Video',
        'Audio': 'Music',
        'TTS': 'Voice',
        'Post Process': 'Upscale',
        'Final Merge': 'Final',
    };
    const _label = stage ? (_stageRoleLabel[stage] || stage) : 'LLM-rewritten';
    const _subtitle = stage
        ? `Sent to the ${_label.toLowerCase()} model for this iteration. Editable in <button type="button" class="link link-primary" onclick="document.getElementById('prompt-peek-modal').close(); openPromptsEdit('${_htmlEscape(stage)}')">Pipeline → ${_htmlEscape(_label)}</button>.`
        : 'The LLM-rewritten prompt the runner used.';
    dlg.innerHTML = `<div class="modal-box bg-base-200 border border-base-100 max-w-2xl">
        <h3 class="font-bold text-sm text-accent uppercase tracking-widest mb-1">${_htmlEscape(_label)} prompt</h3>
        <p class="text-[10px] text-base-content/60 italic mb-3">${_subtitle}</p>
        <div class="text-xs whitespace-pre-wrap font-mono bg-base-300/50 p-3 rounded">${_htmlEscape(text || '(empty)')}</div>
        <div class="modal-action">
            <button id="prompt-peek-copy" class="btn btn-sm btn-secondary btn-outline">Copy</button>
            <form method="dialog"><button class="btn btn-sm btn-primary">Close</button></form>
        </div>
    </div>
    <form method="dialog" class="modal-backdrop"><button>close</button></form>`;
    document.body.appendChild(dlg);
    const copyBtn = dlg.querySelector('#prompt-peek-copy');
    if (copyBtn) {
        copyBtn._prompt = text || '';
        copyBtn.addEventListener('click', (e) => {
            navigator.clipboard.writeText(copyBtn._prompt || '').catch(() => { });
            copyBtn.textContent = '✓ Copied';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
        });
    }
    dlg.showModal();
}

// Per-stage prompts quick-edit modal — shortcut to image_prompt /
// video_prompt / music_prompt / tts_prompt. Live-pickup is a property
// of the runner: it re-reads config.json on entry to each stage, so a
// save before the stage starts will be honoured automatically. Stages
// that already ran (or are running right now) render disabled with a
// "locked" badge — editing them after the fact is a no-op.
//
// _STAGE_ORDER drives status: any stage with index < curIdx is done,
// the index == curIdx stage is running, > curIdx is pending. Concept
// is intentionally absent from the editable map — the LLM has already
// emitted the rewritten prompt by the time any image/video/etc stage
// starts, so editing "Concept" wouldn't change downstream behaviour.
const _PROMPTS_STAGE_MAP = [
    ['Base Image', 'image', 'image_prompt'],
    ['Video Chains', 'video', 'video_prompt'],
    ['Audio', 'audio', 'music_prompt'],
    ['TTS', 'tts', 'tts_prompt'],
];
const _PROMPTS_ROLE_TO_KEY = {
    image: 'image_prompt',
    video: 'video_prompt',
    audio: 'music_prompt',
    tts: 'tts_prompt',
};

function _buildPromptRow(stage, role, value, { locked, status, currentPrompt }) {
    const lockTitle = status === 'done'
        ? 'Stage complete — edits will not apply'
        : 'Stage in progress — edits will not apply';
    const lockNote = locked
        ? `<span class="badge badge-xs badge-neutral ml-2" title="${lockTitle}">locked: ${status}</span>`
        : `<span class="badge badge-xs badge-success ml-2" title="Stage has not run yet — edits will apply">live</span>`;
    // Empty values render as a placeholder hint that surfaces what the
    // runner will fall back to (the active job's `current_prompt`, if
    // any). Non-empty values render as the textarea's actual `value` —
    // the placeholder is only visible when the field is empty.
    const cur = currentPrompt || '';
    const placeholder = cur
        ? `Default: ${cur.slice(0, 80)}${cur.length > 80 ? '…' : ''} — type here to override`
        : `(use built-in default for ${role})`;
    const pullBtn = (!locked && cur)
        ? `<button type="button" class="btn btn-ghost btn-xs ml-2" onclick="_pullFromConcept(this)" title="Copy the active Concept into this textarea so you can edit it">↧ Pull from Concept</button>`
        : '';
    return `
        <div class="form-control" data-prompt-stage="${stage}" data-prompt-role="${role}">
            <label class="label py-0.5">
                <span class="label-text text-[10px] uppercase tracking-widest opacity-70">
                    ${_stageDisplayName(stage)} (${role})
                </span>
                <span class="flex items-center">
                    ${lockNote}
                    ${pullBtn}
                </span>
            </label>
            <textarea class="textarea textarea-bordered textarea-sm font-mono text-[11px]"
                      ${locked ? 'disabled' : ''}
                      rows="2"
                      placeholder="${_htmlEscape(placeholder)}">${_htmlEscape(value || '')}</textarea>
        </div>
    `;
}

// Wired up via inline onclick from `_buildPromptRow`. Reads the active
// Concept from the modal's `data-current-prompt` attribute (set in
// `openPromptsEdit`) so it stays in sync with the snapshot the modal
// was opened from, even if a later WS tick mutates `_lastTick`.
function _pullFromConcept(btn) {
    const modal = document.getElementById('prompts-edit-modal');
    if (!modal) return;
    const cur = modal.dataset.currentPrompt || '';
    if (!cur) return;
    const row = btn.closest('[data-prompt-stage]');
    if (!row) return;
    const ta = row.querySelector('textarea');
    if (!ta || ta.disabled) return;
    ta.value = cur;
    ta.focus();
}

function openPromptsEdit(focusStage = null) {
    const modal = document.getElementById('prompts-edit-modal');
    const rowsEl = document.getElementById('prompts-edit-rows');
    if (!modal || !rowsEl) return;
    const config = (_lastTick && _lastTick.config) || {};
    const state = (_lastTick && _lastTick.state) || {};
    const curStage = state.step || '';
    const curIdx = _STAGE_ORDER.indexOf(curStage);
    // The "Concept" prompt is the LLM's per-video rewrite — the runner
    // sets it on `state.current_prompt` at the top of each iteration.
    // Snapshot it onto the modal element so `_pullFromConcept` and the
    // placeholder text both refer to the same value even if a later WS
    // tick changes `_lastTick.state.current_prompt` while the modal is
    // open.
    const currentPrompt = (state && state.current_prompt) || '';
    modal.dataset.currentPrompt = currentPrompt;
    const rows = [];
    if (currentPrompt) {
        rows.push(`
            <div class="form-control" data-prompt-stage="Concept" data-prompt-role="concept">
                <label class="label py-0.5">
                    <span class="label-text text-[10px] uppercase tracking-widest opacity-70">
                        Concept (LLM-generated, this video)
                    </span>
                    <span class="badge badge-xs badge-info ml-2">read-only</span>
                </label>
                <textarea class="textarea textarea-bordered textarea-sm font-mono text-[11px] opacity-80"
                          disabled rows="2">${_htmlEscape(currentPrompt)}</textarea>
            </div>
        `);
    }
    for (const [stage, role, cfgKey] of _PROMPTS_STAGE_MAP) {
        const i = _STAGE_ORDER.indexOf(stage);
        // When the fleet is idle (curIdx === -1) treat every stage as
        // pending — the next started job will pick up whatever the user
        // saves here.
        let status;
        if (curIdx < 0) status = 'pending';
        else if (i < curIdx) status = 'done';
        else if (i === curIdx) status = 'running';
        else status = 'pending';
        const locked = status !== 'pending';
        rows.push(_buildPromptRow(stage, role, config[cfgKey] || '', { locked, status, currentPrompt }));
    }
    rowsEl.innerHTML = rows.join('');
    if (focusStage) {
        const ta = rowsEl.querySelector(`[data-prompt-stage="${focusStage}"] textarea`);
        if (ta && !ta.disabled) setTimeout(() => ta.focus(), 50);
    }
    modal.showModal();
}

async function savePromptsEdit() {
    const rowsEl = document.getElementById('prompts-edit-rows');
    if (!rowsEl) return;
    const payload = {};
    rowsEl.querySelectorAll('[data-prompt-stage]').forEach(row => {
        const role = row.dataset.promptRole;
        const ta = row.querySelector('textarea');
        if (!ta || ta.disabled) return;
        const cfgKey = _PROMPTS_ROLE_TO_KEY[role];
        if (cfgKey) payload[cfgKey] = ta.value;
    });
    const modal = document.getElementById('prompts-edit-modal');
    if (Object.keys(payload).length === 0) {
        if (modal) modal.close();
        return;
    }
    try {
        await fetch('/config', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(payload),
        });
        // The fleet runner re-reads config.json on each stage entry, so
        // saving here means any not-yet-started stage will pick up the
        // new prompt without further coordination.
    } catch (e) {
        console.warn('save prompts failed', e);
    }
    if (modal) modal.close();
}

// Hard-terminate the run_fleet.py orchestrator process on the host.
// Asks confirm() first because this isn't recoverable mid-iter — any
// in-flight stage gets killed; cancel-flag-based /cancel-all is the
// gentler option for normal cancellation.
async function terminateRunner() {
    if (!confirm("SIGTERM the run_fleet.py orchestrator?\n\nUse this when the runner is stuck (e.g. hung LLM call). The current iteration's in-flight stages will be killed. You'll need to relaunch the runner manually afterwards.")) return;
    try {
        const r = await fetch('/runner/terminate', { method: 'POST' });
        const data = await r.json();
        if (data.ok) {
            const pids = (data.killed || []).join(', ');
            alert(pids ? `Sent SIGTERM to pid ${pids}.` : (data.note || 'No runner process found.'));
        } else {
            alert(`Terminate failed: ${data.error || 'unknown error'}`);
        }
    } catch (e) {
        alert(`Terminate failed: ${e}`);
    }
}
window.terminateRunner = terminateRunner;

async function cancelItem(ts) {
    // ts=0 = the synthetic "running" row that doesn't have a queue entry of
    // its own (the fleet popped it before processing). For those, /cancel-all
    // is the right hammer — it writes cancel.flag so the running iter aborts.
    const url = ts ? '/queue/cancel' : '/cancel-all';
    const body = ts ? JSON.stringify({ ts }) : undefined;
    try {
        const opts = { method: 'POST' };
        if (ts) {
            opts.headers = { 'Content-Type': 'application/json' };
            opts.body = body;
        }
        await fetch(url, opts);
    } catch (e) { console.warn('cancel failed', e); }
}

// Sync the navbar "Idle / Processing / Connection Lost" pill. Called on each
// WS state tick (live state) and on ws.onclose / onerror (lost connection).
function _updateConnPill(isRendering, modeStr, step) {
    // Tri-state header pill (Queue card):
    //   - WS dead       → ⚠ Connection Lost (error)
    //   - all paused    → ⏸ Paused (warning)        ← scheduler.paused = true
    //   - actively working → hidden               (per-item active node speaks)
    //   - nothing to do → ⚠ Idle (ghost)
    const pill = document.getElementById('conn-pill');
    const text = document.getElementById('conn-pill-text');
    if (!pill || !text) return;
    const setPill = (cls, label, animStyle = null) => {
        pill.className = cls;
        if (!animStyle) {
            text.textContent = label;
            pill.removeAttribute('data-anim-style');
        } else {
            pill.dataset.animStyle = animStyle;
            let html = '';
            for (let i = 0; i < label.length; i++) {
                const ch = label[i];
                const isSpace = /\\s/.test(ch);
                const pos = (i / label.length).toFixed(4);
                html += `<span class="render-anim-char"${isSpace ? ' data-space="1"' : ''} `
                    + `style="--char-pos:${pos};--char-i:${i};">`
                    + (isSpace ? '&nbsp;' : _htmlEscape(ch))
                    + '</span>';
            }
            text.innerHTML = html;
        }
        pill.style.display = '';
    };
    if (!_wsConnected) {
        return setPill(
            'badge badge-sm badge-error rounded-full gap-1 normal-case font-mono mx-auto',
            '⚠ Connection Lost'
        );
    }
    const paused = !!(_lastTick && _lastTick.scheduler && _lastTick.scheduler.paused);
    // Clear _pausePending when WS confirms paused — no poll lag needed.
    if (paused && _pausePending) {
        _pausePending = false;
        _applyPauseButtonState(true);
    }
    if (_pausePending) {
        const animStyle = window._pausingAnimStyle || 'pulse';
        return setPill(
            `badge badge-sm badge-warning rounded-full gap-1 normal-case font-mono mx-auto render-anim`,
            '⏸ Pausing…',
            animStyle
        );
    }
    if (paused) {
        return setPill(
            'badge badge-sm badge-warning rounded-full gap-1 normal-case font-mono mx-auto',
            '⏸ Paused'
        );
    }
    if (isRendering) {
        pill.style.display = 'none';
        return;
    }
    setPill(
        'badge badge-sm badge-ghost rounded-full gap-1 normal-case font-mono mx-auto',
        '⚠ Idle'
    );
}

async function requeueItem(ts) {
    if (!ts) return;
    try {
        const r = await fetch('/queue/requeue', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ts }),
        });
        // Read the body so a 404 / "not requeueable" surfaces in the
        // console with a real reason instead of a generic "requeue failed".
        const j = await r.json().catch(() => null);
        if (!r.ok || !(j && j.ok)) {
            console.warn('requeue not applied:', j || `HTTP ${r.status}`);
        }
    } catch (e) { console.warn('requeue fetch failed', e); }
}

// Tick stage + total elapsed once a second so they don't jump only on WS ticks.
setInterval(() => {
    if (!_isRendering) return;
    // Update the SINGLE top-of-card segmented pipeline bar in place. We avoid
    // re-templating the bar every second — instead we mutate the current
    // .pipeline-seg-fill width, the per-segment inline elapsed timer
    // ([data-seg-elapsed] inside .pipeline-seg-current), the activity text,
    // and the total elapsed/ETA footer. The bar's CSS width transition stays
    // smooth because we never thrash innerHTML on the segments.
    const stageDurations = _STAGE_ORDER.map(s => _stageAvgSeconds(s) || 30);
    const totalSec = stageDurations.reduce((a, b) => a + b, 0);
    const host = document.getElementById('active-job-progress-bar');
    if (!host) return;
    const bar = host.querySelector('[data-pipeline-bar]');
    if (!bar) return;
    const curStage = bar.dataset.curStage || '';
    const curIdx = _STAGE_ORDER.indexOf(curStage);
    if (curIdx < 0 || !_stageStartTs) return;
    const stageElapsed = (Date.now() - _stageStartTs) / 1000;
    const stageAvg = stageDurations[curIdx] || 30;
    const fraction = Math.min(1, stageElapsed / stageAvg);
    const isOverrun = stageElapsed > stageAvg;
    // Update current segment's fill + inline elapsed timer.
    const curSeg = bar.querySelector('.pipeline-seg-current');
    if (curSeg) {
        const fill = curSeg.querySelector('.pipeline-seg-fill');
        if (fill) fill.style.width = (fraction * 100) + '%';
        curSeg.classList.toggle('pipeline-seg-overrun', isOverrun);
        const segElapsedEl = curSeg.querySelector('[data-seg-elapsed]');
        if (segElapsedEl) segElapsedEl.innerHTML = _fmtElapsedHtml(stageElapsed * 1000);
    }
    // Activity text is now backend-driven via the `render_heartbeat` WS
    // event — _applyRenderHeartbeat() owns the queue-header-activity DOM,
    // gated by the heartbeat's TTL so a stalled scheduler can't leave the
    // spinner stuck on. The 1Hz expiry ticker (setInterval below) keeps
    // the label hiding itself if heartbeats stop arriving.
    if (typeof _applyRenderHeartbeat === 'function') _applyRenderHeartbeat();
    // Total elapsed / ETA — now lives in the external #queue-progress-footer
    // row (template), so they sit on the same line as the bulk-action buttons.
    const totEl = document.getElementById('queue-total-elapsed');
    const totEt = document.getElementById('queue-total-eta');
    if (totEl && _jobStartTs) totEl.innerHTML = _fmtElapsedHtml(Date.now() - _jobStartTs);
    if (totEt && totalSec > 0) totEt.innerHTML = _fmtElapsedHtml(totalSec * 1000);
}, 1000);

const $ = (id) => document.getElementById(id);

function statusClass(st) {
    if (st === 'danger') return 'badge-error';
    if (st === 'warn') return 'badge-warning';
    return 'badge-success';
}

function progressClass(st) {
    if (st === 'danger') return 'progress-error';
    if (st === 'warn') return 'progress-warning';
    return 'progress-success';
}

function _statusRank(st) {
    if (st === 'danger') return 2;
    if (st === 'warn') return 1;
    return 0;
}

function _statusEmoji(st) {
    if (st === 'danger') return '🔴';
    if (st === 'warn') return '🟡';
    return '🟢';
}

function updateOutputsDisk(d) {
    if (!d) return;
    const el = document.getElementById('d-v');
    if (!el) return;
    el.textContent = `${d.pct}%`;
    // Tone matches the disk ticker columns (bg-primary baseline + bg-error
    // when the disk slice in the ticker reads pressure). Aligns with the
    // RAM/GPU/Load convention where the percentage label colour tracks the
    // ticker's most-recent column tint.
    el.className = 'font-mono font-black ' + (
        d.status === 'danger' ? 'text-error' :
            d.status === 'warn' ? 'text-warning' : 'text-primary'
    );
    // 100 % celebration — uses the shared _pickCelebrateClass helper
    // so this honours the user's Settings → Display animation choice
    // (or rotates randomly when 'random' is selected).
    if (d.pct >= 100) {
        const has = Array.from(el.classList).some(c => c.startsWith('celebrate-'));
        if (!has) {
            el.classList.add(_pickCelebrateClass());
        }
    } else {
        CELEBRATE_STYLES.forEach(s => el.classList.remove(s));
    }
    const freeGb = (d.free_gb !== undefined) ? d.free_gb : (d.total_gb - d.used_gb);
    // Update the used/free label beneath the percentage (mirrors RAM r-v line).
    const drEl = document.getElementById('d-r');
    if (drEl) drEl.textContent = `${d.used_gb} used / ${Math.round(freeGb * 10) / 10} GB free`;
    // Wrapper carries the tooltip with both numbers spelled out.
    const wrap = el.closest('span[title]');
    if (wrap) wrap.title = `${d.used_gb} GB used / ${Math.round(freeGb * 10) / 10} GB free · ticker spans ~1 hour`;
}

function updateStorage(storage) {
    if (!storage || !storage.length) return;
    const pill = $('st-pill');
    // Find worst (highest pct) mount and whether any non-ok exist.
    let worst = storage[0];
    let anyAlert = false;
    storage.forEach(s => {
        if (s.pct > worst.pct) worst = s;
        if (_statusRank(s.status) > 0) anyAlert = true;
    });

    if (pill) {
        pill.className = 'badge badge-lg cursor-pointer tooltip tooltip-left ' + statusClass(worst.status);
        pill.innerText = `${_statusEmoji(worst.status)} ${worst.pct}%`;
        pill.setAttribute(
            'data-tip',
            storage.map(s => `${s.mount}: ${s.used_gb}/${s.total_gb} GB (${s.pct}%)`).join(' · ')
        );
        pill.style.display = anyAlert ? '' : 'none';
    }

    const list = $('storage-modal-list');
    if (list) {
        list.innerHTML = storage.map(s => `
            <div class="flex items-center justify-between gap-3 bg-base-100 p-3 rounded-lg border border-base-300">
                <div class="flex-1">
                    <div class="text-[10px] uppercase tracking-widest text-base-content/50">${_htmlEscape(s.mount)}</div>
                    <div class="text-sm font-mono">${s.used_gb} / ${s.total_gb} GB</div>
                </div>
                <span class="badge badge-lg ${statusClass(s.status)}">${s.pct}%</span>
            </div>`).join('');
    }
}

function switchOutputTab(which) {
    document.querySelectorAll('[data-output-tab]').forEach(t => {
        t.classList.toggle('tab-active', t.getAttribute('data-output-tab') === which);
    });
    document.querySelectorAll('[data-output-panel]').forEach(p => {
        p.style.display = p.getAttribute('data-output-panel') === which ? 'block' : 'none';
    });
}

function openStorageModal() {
    const m = $('storage-modal');
    if (m && typeof m.showModal === 'function') m.showModal();
}

function updateRam(ram) {
    if (!ram) return;
    const el = $('ram-est');
    if (!el) return;
    const txt = el.querySelector('.ram-txt');
    const bar = el.querySelector('.ram-bar');
    const bd = el.querySelector('.ram-breakdown');
    if (txt) txt.innerText = `${ram.estimated_gb} / ${ram.budget_gb || 128} GB unified`;
    if (bar) {
        bar.className = 'ram-bar progress w-full mt-1 ' + progressClass(ram.status);
        bar.value = Math.min(100, (ram.estimated_gb / (ram.budget_gb || 128)) * 100);
    }
    // Per-model WILL-USE rows. We hide entries with gb===0 unless they're
    // overhead, so the table doesn't get polluted with `Voice  —  0.0 GB`
    // lines when the user has TTS off. The server emits role-keyed entries
    // in pipeline order; we render them as a fixed-width table.
    if (bd && Array.isArray(ram.breakdown)) {
        const rows = [];
        for (const e of ram.breakdown) {
            if (e.role !== 'overhead' && (!e.gb || e.gb <= 0)) continue;
            const stage = (e.stage || '').toString();
            const label = (e.label || e.model || '').toString();
            const labelTrim = label.length > 22 ? label.slice(0, 21) + '…' : label;
            const gb = (Math.round((e.gb || 0) * 10) / 10).toFixed(1);
            rows.push(
                `<div class="flex items-baseline gap-2">` +
                `<span class="opacity-70 w-14 flex-none">${_htmlEscape(stage)}</span>` +
                `<span class="flex-1 truncate">${_htmlEscape(labelTrim)}</span>` +
                `<span class="font-mono text-right flex-none">${gb} GB</span>` +
                `</div>`
            );
        }
        rows.push('<div class="opacity-50 my-1">────────────────────────────</div>');
        const total = (Math.round((ram.estimated_gb || 0) * 10) / 10).toFixed(1);
        const budget = ram.budget_gb || 128;
        rows.push(
            `<div class="flex items-baseline justify-between gap-2">` +
            `<b>Total</b>` +
            `<span><b class="font-mono">${total} GB</b> <span class="opacity-60">/ ${budget} GB unified</span></span>` +
            `</div>`
        );
        bd.innerHTML = rows.join('');
    }
    el.className = 'alert p-2 ' + (ram.status === 'danger' ? 'alert-error' : ram.status === 'warn' ? 'alert-warning' : 'alert-success');
    el.id = 'ram-est';
}

// Render the Belady-MIN load plan inside the Pipeline popup. Lazy: only
// fetched when the user expands <details id="load-plan-details">. The plan
// is advisory — surfaced so the user can see which model boundaries the
// scheduler *could* skip cold-loads at if Phase 2 wires it in. See
// docs/memory-stage-planner-design.md.
async function loadPlanRender() {
    const body = $('load-plan-body');
    if (!body) return;
    body.innerHTML = '<span class="opacity-50">loading…</span>';
    let plan;
    try {
        const r = await fetch('/pipeline/plan');
        plan = await r.json();
    } catch (e) {
        body.innerHTML = '<span class="text-error">failed to fetch /pipeline/plan</span>';
        return;
    }
    if (!plan || !Array.isArray(plan.decisions)) {
        body.innerHTML = '<span class="text-error">empty plan</span>';
        return;
    }
    const rows = [];
    // Header: budget + projected savings.
    const sv = plan.savings || {};
    const savedSec = sv.est_saved_seconds || 0;
    const savedMin = Math.floor(savedSec / 60);
    const savedTail = savedSec % 60;
    rows.push(
        `<div class="opacity-70">budget ${plan.budget_gb} GB · ` +
        `${plan.queued_jobs_planned} queued + active · ` +
        `<b>${sv.naive_loads || 0} → ${sv.planned_loads || 0} loads</b> ` +
        `(saved ${sv.saved_loads || 0} ≈ ${savedMin}m${savedTail}s)</div>`
    );
    rows.push('<div class="opacity-50">────────────────────────────</div>');
    // Per-step decision rows. The plan flattens active+queued jobs into
    // ONE sequence of stage steps — without job boundaries, repeated
    // model loads across jobs read as if the planner is duplicating
    // work. Detect job boundaries via stage-cycle wrap (each job
    // proceeds image → video → audio → tts → upscale; when the next
    // stage's index is <= current's, we've started a new job) and
    // insert a divider so the user can read 3 jobs as 3 jobs.
    const _STAGE_CYCLE = ['image', 'video', 'audio', 'tts', 'upscale'];
    const _stageIdx = (s) => {
        const i = _STAGE_CYCLE.indexOf(s);
        return i < 0 ? 99 : i;
    };
    let jobNum = 1;
    let prevStageIdx = -1;
    rows.push(`<div class="text-primary font-bold mt-1">— Job ${jobNum} (active) —</div>`);
    for (let i = 0; i < plan.decisions.length; i++) {
        const d = plan.decisions[i];
        const curIdx = _stageIdx(d.step.stage);
        if (i > 0 && curIdx <= prevStageIdx) {
            jobNum += 1;
            rows.push(`<div class="text-primary font-bold mt-2">— Job ${jobNum} (queued) —</div>`);
        }
        prevStageIdx = curIdx;
        const tag = d.load && d.load.length ? 'LOAD ' : 'HIT  ';
        const cls = d.load && d.load.length ? '' : 'text-success';
        // Stage labels: convert internal slugs to user-facing 'X Model'
        // strings. The internal slug ('image', 'video', 'audio', etc.)
        // is the role-key the planner uses; the display label reads
        // as a noun phrase aligned with how the user thinks of each
        // model role.
        const _stageLabel = ({
            concept: 'Text Model ',
            image: 'Image Model',
            video: 'Video Model',
            audio: 'Music Model',
            tts: 'Speech Model',
            upscale: 'Upscale Model',
            merge: 'Final Merge',
        })[d.step.stage] || (d.step.stage || '');
        const stage = _stageLabel.padEnd(13);
        const model = (d.step.model || '').padEnd(12);
        const gb = (Math.round((d.step.gb || 0) * 10) / 10).toFixed(1);
        rows.push(
            `<div class="${cls}">` +
            `<span class="opacity-60">${(i + 1).toString().padStart(2, ' ')}.</span> ` +
            `${_htmlEscape(tag)} ${_htmlEscape(stage)} ${_htmlEscape(model)} ` +
            `<span class="opacity-50">${gb} GB</span>` +
            `</div>`
        );
        if (d.evict && d.evict.length) {
            rows.push(
                `<div class="text-warning pl-6">` +
                `↳ evict ${_htmlEscape(d.evict.join(', '))}` +
                `</div>`
            );
        }
    }
    rows.push('<div class="opacity-50">────────────────────────────</div>');
    rows.push(
        `<div class="opacity-60 italic text-[10px] mt-1">` +
        `Each job needs its own image / video / audio model loaded. ` +
        `Repeated model names across jobs are correct — Belady's MIN ` +
        `keeps a model resident across jobs only when the unified ` +
        `memory budget allows.</div>`
    );
    rows.push(
        `<div class="opacity-60 italic">` +
        `Advisory only — scheduler still cold-loads on every stage. ` +
        `Phase 2: wire into acquire_gpu.</div>`
    );
    body.innerHTML = rows.join('');
}

// Show/hide the slopped sub-select for a given role and populate it on demand.
// Wired via inline onchange handlers on cfg-base / cfg-audio / cfg-tts.
//   selId    - 'base' | 'audio' | 'tts'  (matches cfg-<id>)
//   roleName - 'image' | 'audio' | 'tts' (server /pipeline/slopped role param)
async function _onPseudoChanged(selId, roleName) {
    const top = $('cfg-' + selId);
    if (!top) return;
    // Image role uses a thumbnail grid (with a hidden input for compat); audio
    // and tts still use the original plain <select> sub-selector.
    if (roleName === 'image') {
        const wrap = $('cfg-' + selId + '-slopped-wrap');
        const grid = $('cfg-' + selId + '-slopped-grid');
        const hidden = $('cfg-' + selId + '-slopped');
        if (!wrap || !grid || !hidden) return;
        if (top.value !== '__slopped__') {
            wrap.classList.add('hidden');
            return;
        }
        wrap.classList.remove('hidden');
        if (grid.dataset.loaded === '1') {
            // Already populated; just re-show the highlight for any saved value.
            _slopppedHighlight(grid, hidden.value);
            return;
        }
        grid.innerHTML = '<div class="col-span-full text-[11px] italic opacity-60 p-2">loading…</div>';
        try {
            const r = await fetch('/pipeline/slopped?role=' + encodeURIComponent(roleName));
            const j = await r.json();
            const files = ((j && j.files) || []).slice(0, 60);
            if (!files.length) {
                grid.innerHTML = '<div class="col-span-full text-[11px] italic opacity-60 p-2">No PNG files yet — generate something first.</div>';
                grid.dataset.loaded = '1';
                return;
            }
            grid.innerHTML = files.map(name => {
                const safe = _htmlEscape(name);
                const trunc = _htmlEscape(_truncMiddle(name, 18));
                const url = '/files/' + encodeURIComponent(name);
                return `<button type="button" class="slopped-cell relative aspect-square bg-black rounded overflow-hidden border-2 border-transparent hover:border-primary focus:border-primary focus:outline-none" data-fname="${safe}" title="${safe}">`
                    + `<img src="${url}" loading="lazy" class="w-full h-full object-cover" alt="">`
                    + `<span class="absolute bottom-0 inset-x-0 px-1 py-0.5 text-[8px] font-mono bg-black/60 text-white truncate">${trunc}</span>`
                    + `</button>`;
            }).join('');
            grid.dataset.loaded = '1';
            if (!grid.dataset.bound) {
                grid.addEventListener('click', (e) => {
                    const cell = e.target.closest('.slopped-cell');
                    if (!cell) return;
                    e.stopPropagation();
                    e.preventDefault();
                    _slopppedHighlight(grid, 'slopped:' + cell.dataset.fname);
                    hidden.value = 'slopped:' + cell.dataset.fname;
                    hidden.dispatchEvent(new Event('change', { bubbles: true }));
                });
                grid.dataset.bound = '1';
            }
            _slopppedHighlight(grid, hidden.value);
        } catch (e) {
            grid.innerHTML = '<div class="col-span-full text-[11px] italic opacity-60 p-2">(error loading files)</div>';
        }
        return;
    }
    // Non-image roles: keep legacy <select> behavior.
    const sub = $('cfg-' + selId + '-slopped');
    if (!sub) return;
    if (top.value !== '__slopped__') {
        sub.classList.add('hidden');
        sub.innerHTML = '';
        return;
    }
    sub.classList.remove('hidden');
    if (sub.options.length === 0) {
        sub.innerHTML = '<option value="">loading…</option>';
        try {
            const r = await fetch('/pipeline/slopped?role=' + encodeURIComponent(roleName));
            const j = await r.json();
            const files = (j && j.files) || [];
            if (!files.length) {
                sub.innerHTML = '<option value="">(no existing files)</option>';
                return;
            }
            sub.innerHTML = '<option value="">— pick a file —</option>'
                + files.map(f => `<option value="slopped:${_htmlEscape(f)}">${_htmlEscape(f)}</option>`).join('');
        } catch (e) {
            sub.innerHTML = '<option value="">(error)</option>';
        }
    }
}

// Highlight the cell whose data-fname matches the (possibly `slopped:`-prefixed) value.
function _slopppedHighlight(grid, val) {
    const target = (val || '').startsWith('slopped:') ? val.slice('slopped:'.length) : '';
    grid.querySelectorAll('.slopped-cell').forEach(c => {
        if (target && c.dataset.fname === target) c.classList.add('border-primary');
        else c.classList.remove('border-primary');
    });
}

function schedBadgeClass(type) {
    if (type === 'stage_start') return 'badge-info';
    if (type === 'stage_end') return 'badge-success';
    if (type === 'budget_block') return 'badge-warning';
    if (type === 'oom_retry') return 'badge-error';
    if (type === 'emergency_free') return 'badge-error';
    return 'badge-ghost';
}

// Track the asset currently shown in the info modal so the delete button knows what to remove.
let _currentAssetFilename = null;

async function deleteCurrentAsset() {
    const filename = _currentAssetFilename;
    if (!filename) return;
    if (!confirm(`Delete this asset?\n\n${filename}\n\nThis cannot be undone.`)) return;
    try {
        const r = await fetch('/asset/' + encodeURIComponent(filename), { method: 'DELETE' });
        const j = await r.json();
        if (!j.ok) { alert('Delete failed: ' + (j.error || 'unknown')); return; }
        document.querySelectorAll('#preview-grid > [data-slop-kind]').forEach(card => {
            const t = card.querySelector('[title]');
            if (t && t.getAttribute('title') === filename) card.remove();
        });
        const d = document.getElementById('asset-info-modal');
        if (d && d.close) d.close();
        _currentAssetFilename = null;
    } catch (e) {
        alert('Delete error: ' + String(e));
    }
}

// Asset card click → metadata popover
async function openAssetInfo(filename) {
    _currentAssetFilename = filename;
    const d = document.getElementById('asset-info-modal');
    if (!d) return;
    const body = document.getElementById('asset-info-body');
    const media = document.getElementById('asset-info-media');
    // daisyUI skeleton placeholders read better than a single loader for
    // a structured panel — show a media-shaped block + a column of
    // text-line skeletons while the /asset/<file> fetch is in flight.
    if (media) media.innerHTML = '<div class="skeleton w-full aspect-video"></div>';
    if (body) body.innerHTML = `
        <div class="grid grid-cols-[min-content_1fr] gap-x-3 gap-y-2 text-xs">
            <div class="skeleton h-3 w-12"></div><div class="skeleton h-3 w-3/4"></div>
            <div class="skeleton h-3 w-12"></div><div class="skeleton h-3 w-1/3"></div>
            <div class="skeleton h-3 w-12"></div><div class="skeleton h-3 w-2/3"></div>
            <div class="skeleton h-3 w-12"></div><div class="skeleton h-3 w-1/4"></div>
            <div class="skeleton h-3 w-12"></div><div class="skeleton h-3 w-2/3"></div>
            <div class="skeleton h-3 w-12"></div><div class="skeleton h-3 w-full"></div>
        </div>
    `;
    if (d.showModal) d.showModal();
    try {
        const r = await fetch('/asset/' + encodeURIComponent(filename));
        const m = await r.json();
        if (!m.ok) {
            // Surface the filename so the user can confirm what was
            // requested. "not found" with no filename is undebuggable.
            console.warn('[asset-info] failed:', filename, m);
            body.innerHTML = `<div class="alert alert-error text-xs"><div>${m.error || 'error'}</div><div class="font-mono text-[10px] mt-1 opacity-70">${_htmlEscape(filename)}</div></div>`;
            return;
        }
        const isV = filename.endsWith('.mp4');
        const isA = filename.endsWith('.wav');
        const mediaHtml = isV
            ? `<video controls autoplay loop class="w-full aspect-video rounded bg-black"><source src="${m.url}"></video>`
            : isA
                ? `<audio controls class="w-full"><source src="${m.url}"></audio>`
                : `<img src="${m.url}" class="w-full rounded bg-black" />`;
        media.innerHTML = mediaHtml;
        const badgeColor = ({ final: 'badge-accent', chain: 'badge-primary', image: 'badge-secondary', audio: 'badge-warning' })[m.kind] || 'badge-ghost';
        // Prompt-notes link to the Markdown sidecar (`<file>.md`). The
        // sidecar is only written for FINAL/iter outputs by run_fleet.py's
        // `_write_md_sidecar`, so this row is conditional on a successful
        // fetch below. We render the row first with a placeholder and
        // hydrate it after the MD fetch resolves.
        const mdHref = '/files/' + encodeURIComponent(filename) + '.md';
        body.innerHTML = `
            <div id="asset-info-md-row" class="hidden mb-2">
                <a href="${mdHref}" target="_blank" rel="noopener" class="link link-primary text-xs uppercase tracking-widest">Prompt notes</a>
                <pre id="asset-info-md-preview" class="text-xs whitespace-pre-wrap leading-snug bg-base-200/40 rounded p-2 max-h-48 overflow-y-auto mt-1"></pre>
            </div>
            <div class="grid grid-cols-[min-content_1fr] gap-x-3 gap-y-1 text-xs font-mono">
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">File</div><div class="truncate">${m.filename}</div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Kind</div><div><span class="badge badge-xs ${badgeColor}">${m.kind}</span></div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Model</div><div>${m.model || '—'}</div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Size</div><div>${m.size_human}</div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Created</div><div>${m.mtime_human} <span class="text-base-content/50">(${m.age_seconds}s ago)</span></div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Prompt</div><div class="whitespace-pre-wrap italic ${m.prompt ? '' : 'text-base-content/40'}">${m.prompt || '(no sidecar captured yet — fleet writes prompts to state.json only while active)'}</div>
            </div>
        `;
        // For FINAL_*.mp4, append a Components section listing the source
        // chain mp4s + base.png + any music/tts wavs that were merged into
        // this final asset. Best-effort; failures are non-fatal.
        if (filename.startsWith('FINAL_') && filename.endsWith('.mp4')) {
            renderAssetComponents(filename, body).catch(() => undefined);
        }
        // Best-effort MD-sidecar fetch (only FINAL/iter outputs have one).
        // 404 silently leaves the row hidden. textContent is enough for v1
        // — the raw Markdown reads fine in a monospace <pre>.
        try {
            const mdHref = `/files/${encodeURIComponent(filename)}.md`;
            const mdRes = await fetch(mdHref);
            if (mdRes.ok) {
                const txt = await mdRes.text();
                const row = document.getElementById('asset-info-md-row');
                const pre = document.getElementById('asset-info-md-preview');
                if (row && pre) {
                    pre.textContent = txt;
                    row.classList.remove('hidden');
                }
            }
        } catch (_e) { /* non-fatal — asset still renders */ }
    } catch (e) {
        body.innerHTML = `<div class="alert alert-error text-xs">${String(e)}</div>`;
    }
}

// Fetch and render the "Components" section for a FINAL_*.mp4 asset.
// Each row links to /files/<name> and shows a small thumbnail when the
// component is a png/jpg/mp4 (audio rows skip the thumbnail).
async function renderAssetComponents(filename, body) {
    if (!body) return;
    const section = document.createElement('div');
    section.className = 'mt-3 pt-3 border-t border-base-content/10';
    section.innerHTML = `
        <div class="text-[10px] uppercase tracking-widest text-base-content/50 mb-2">Components</div>
        <div class="space-y-1 text-xs">
            <div class="skeleton h-6 w-full"></div>
            <div class="skeleton h-6 w-full"></div>
        </div>
    `;
    body.appendChild(section);
    let data;
    try {
        const r = await fetch('/asset/components/' + encodeURIComponent(filename));
        data = await r.json();
    } catch (e) {
        section.querySelector('.space-y-1').innerHTML = `<div class="text-base-content/40">components unavailable: ${String(e)}</div>`;
        return;
    }
    if (!data || !data.ok) {
        section.querySelector('.space-y-1').innerHTML = `<div class="text-base-content/40">components unavailable</div>`;
        return;
    }
    const list = data.components || [];
    if (list.length === 0) {
        section.querySelector('.space-y-1').innerHTML = `<div class="text-base-content/40">(no source components found)</div>`;
        return;
    }
    const kindBadge = (k) => {
        const map = { final: 'badge-accent', chain: 'badge-primary', video: 'badge-primary', image: 'badge-secondary', audio: 'badge-warning' };
        return map[k] || 'badge-ghost';
    };
    const rowHtml = (c) => {
        const isVid = c.file.endsWith('.mp4');
        const isImg = c.file.endsWith('.png') || c.file.endsWith('.jpg');
        const thumb = isVid
            ? `<video class="w-12 h-8 rounded bg-black object-cover" preload="metadata" muted playsinline><source src="${c.url}#t=0.1"></video>`
            : isImg
                ? `<img src="${c.url}" class="w-12 h-8 rounded bg-black object-cover" loading="lazy" />`
                : `<div class="w-12 h-8 rounded bg-base-300 flex items-center justify-center text-[9px] text-base-content/50">wav</div>`;
        const partTxt = (c.part != null && c.of != null) ? `part ${c.part} of ${c.of}` : (c.part != null ? `part ${c.part}` : '');
        const modelTxt = c.model ? `<span class="badge badge-xs badge-ghost">${c.model}</span>` : '';
        const kindTxt = `<span class="badge badge-xs ${kindBadge(c.kind)}">${c.kind || '—'}</span>`;
        return `
            <div class="flex items-center gap-2 py-1">
                <button type="button" class="flex-none" onclick='event.stopPropagation(); openAssetInfo(${JSON.stringify(c.file)})' title="Open ${c.file}">
                    ${thumb}
                </button>
                <div class="flex-1 min-w-0">
                    <a href="${c.url}" target="_blank" rel="noopener" class="link link-hover truncate block font-mono text-[11px]" title="${c.file}">${c.file}</a>
                    <div class="flex flex-wrap items-center gap-1 mt-0.5">
                        ${kindTxt}
                        ${modelTxt}
                        ${partTxt ? `<span class="text-[10px] text-base-content/60">${partTxt}</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    };
    section.querySelector('.space-y-1').innerHTML = list.map(rowHtml).join('');
}

// Click-to-info wiring for any asset card in the Slop feed
// Delegated click for the per-stage prompt-peek badges. Stashing the prompt
// text in data-prompt-text avoids attribute-quoting bugs when the prompt
// contains apostrophes or quotes (which used to silently break onclick).
document.addEventListener('click', (e) => {
    const peek = e.target.closest('.slop-prompt-peek');
    if (peek) {
        e.stopPropagation();
        const txt = peek.getAttribute('data-prompt-text') || '';
        const stage = peek.getAttribute('data-prompt-stage') || '';
        if (typeof showPromptPeek === 'function') showPromptPeek(txt, stage);
        return;
    }
});

document.addEventListener('click', (e) => {
    const card = e.target.closest('#preview-grid > [data-slop-kind]');
    if (!card) return;
    // Avoid opening info when clicking the native media controls
    if (e.target.closest('video, audio')) return;
    // Prefer the explicit data-slop-file dataset (set by both SSR cards
    // at index.html:1685 and JS-built cards in _buildSlopCard). Fall back
    // to the [title] attribute scan for any legacy markup. Reading
    // dataset.slopFile first dodges the case where querySelector('[title]')
    // picks up a nested control's title (e.g. native media-control
    // tooltips or future hover-FAB tooltips) instead of the filename span.
    const filename = card.dataset.slopFile
        || (card.querySelector('[title]') && card.querySelector('[title]').getAttribute('title'));
    if (filename) openAssetInfo(filename);
});

// Done-queue <details> open-state tracker. The queue list is wholesale
// re-rendered on every WS state tick, so we cannot rely on the DOM to hold
// the user's expansion. We listen for `toggle` in the capture phase
// (it doesn't bubble), update the module-level _openDoneItem, and enforce
// single-open by collapsing any other open done-item.
document.addEventListener('toggle', (e) => {
    const det = e.target;
    if (!(det instanceof HTMLElement)) return;
    if (det.tagName !== 'DETAILS') return;
    const qid = det.getAttribute('data-q-id');
    if (!qid) return;
    if (det.open) {
        _openDoneItem = qid;
        // Single-open enforcement: collapse any other done item currently open.
        document.querySelectorAll('details[data-q-id][open]').forEach(other => {
            if (other !== det) other.removeAttribute('open');
        });
    } else if (_openDoneItem === qid) {
        _openDoneItem = null;
    }
}, true);

// Pending/active queue-item open tracker. Same problem as done-items: the
// queue list re-renders ~1 Hz off WS ticks, which would drop any <details>
// the user manually expanded. We persist user-toggled open-ness across
// re-renders by stashing the queue item's `data-q-ts` in a module-level
// Set; the renderItem template reads from this set when emitting `open`.
const _openPendingItems = new Set();
// Mirror set for the per-iter Video Chains <details> inside the output
// reveal — keyed by data-video-chain (the v_idx) so each iter's expand
// state survives WS re-renders.
const _openVideoChains = new Set();

// Queue-render signature cache. The WS tick handler ticks ~1 Hz and used to
// rebuild #q-list.innerHTML on every tick — even when the underlying queue
// data was identical to the previous tick. That detach/reattach was the
// root cause of dropdown glyph flicker, transient focus loss on the ⋯
// menus, and image/video thumbnail poster reloads. We now compute a
// stable signature over the fields the render path actually consumes and
// skip the rebuild when it matches the previous tick. _openPendingItems
// preserves user-expanded <details> across whatever rebuilds DO happen,
// so this is purely an "elide redundant work" optimisation.
let _lastQueueSig = null;
// Tracks the empty<->has-items transition for the gated
// _refreshCardVisibility() call inside the rebuild block. Initialised to
// null so the first tick after page load forces a refresh.
let _lastQueueHadItems = null;

// Animated thumbnail cycle. Any <video data-anim-thumb> element gets
// its currentTime stepped through {0, 50%, 95%} of its duration on a
// shared 800ms timer, yielding a 3-frame GIF-like preview without
// re-encoding anything. Cheap because we only seek + repaint, never
// play. Skips elements that haven't loaded metadata yet (duration is
// still NaN) and elements that are off-screen via getBoundingClientRect.
setInterval(() => {
    if (typeof _isUiToggleOn === 'function' && !_isUiToggleOn('outputThumbs')) return;
    const vs = document.querySelectorAll('video[data-anim-thumb]');
    if (!vs.length) return;
    const vh = window.innerHeight || 0;
    vs.forEach(v => {
        if (!isFinite(v.duration) || v.duration <= 0) return;
        const r = v.getBoundingClientRect();
        if (r.bottom < 0 || r.top > vh) return;  // off-screen
        const frame = (parseInt(v.dataset.animFrame || '0', 10) + 1) % 3;
        v.dataset.animFrame = String(frame);
        const t = frame === 0 ? 0 : (frame === 1 ? v.duration * 0.5 : v.duration * 0.95);
        try { v.currentTime = t; } catch (_) { }
    });
}, 800);
document.addEventListener('toggle', (e) => {
    const det = e.target;
    if (!(det instanceof HTMLElement)) return;
    if (det.tagName !== 'DETAILS') return;
    // Queue item reveal: persist by ts.
    const li = det.closest('li[data-q-ts]');
    if (li) {
        const ts = parseInt(li.getAttribute('data-q-ts') || '0', 10);
        if (ts) {
            if (det.open) _openPendingItems.add(ts);
            else _openPendingItems.delete(ts);
        }
        // When the user expands a queue item that's near the bottom of
        // the viewport, the freshly-revealed body extends below the
        // fold and there's nothing to draw the eye to it. After the
        // browser finishes painting the new content, scroll the item
        // into view so the entire reveal is visible — using
        // block:'nearest' so we only scroll if needed (no jump when
        // the item is already fully on-screen).
        if (det.open) {
            requestAnimationFrame(() => requestAnimationFrame(() => {
                try {
                    const r = li.getBoundingClientRect();
                    const vh = window.innerHeight || document.documentElement.clientHeight;
                    if (r.bottom > vh - 8) {
                        li.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                } catch (_) { }
            }));
        }
    }
    // Video Chains reveal: persist by v_idx.
    if (det.classList.contains('video-chain-details')) {
        const v = det.getAttribute('data-video-chain') || '';
        if (v) {
            if (det.open) _openVideoChains.add(v);
            else _openVideoChains.delete(v);
        }
    }
}, true);

// Map a filename → display badge for the model + role that produced it.
// Returns { label, color, border, part?, kind } where kind is the filter bucket.
function _slopBadgeMeta(file) {
    const isMp4 = file.endsWith('.mp4');
    const isWav = file.endsWith('.wav');
    const isFinal = isMp4 && /^FINAL_/i.test(file);
    // Chain-stage PNGs (v<N>_base.png, v<N>_f<M>.png) belong to the video
    // pipeline even though they're images, so the video filter chip should
    // surface them alongside chains + FINAL mp4s.
    const isChainPng = /^v\d+_(base|f\d+)\.png$/i.test(file);
    const kind = isMp4 ? 'video' : isWav ? 'audio' : (isChainPng ? 'video' : 'image');
    if (isFinal) {
        const m = file.match(/^FINAL_([^.]+)\.mp4$/i);
        return { label: 'FINAL · V' + (m ? m[1] : '?'), color: 'badge-accent', border: 'border-accent', kind };
    }
    let model = '';
    let part = '';
    const knownModels = new Set(['qwen', 'ernie', 'ltx-2.3', 'ltx-bridge', 'wan2.2', 'wan2.5', 'heartmula', 'qwen-tts', 'kokoro']);
    const testMatch = file.match(/^test_([a-z0-9.-]+)_/i);
    if (testMatch && knownModels.has(testMatch[1].toLowerCase())) {
        model = testMatch[1].toLowerCase();
    } else if (/^ltx_base_/i.test(file)) {
        model = 'ltx-2.3';
    } else if (/^v\d+_base\.png$/i.test(file)) {
        // Base image of a video pipeline iter — produced by whatever
        // base_model is configured (qwen by default). We can't tell the
        // exact model from the filename alone; default to qwen which is
        // overwhelmingly the common case. Sidecar JSON (now written by
        // the fleet) will let us be authoritative in a follow-up.
        model = 'qwen';
    } else if (/^v\d+_c\d+\.mp4$/i.test(file)) {
        model = 'ltx-2.3';
        const pm = file.match(/_c(\d+)\.mp4$/i);
        if (pm) part = pm[1];
    } else if (/^v\d+_f\d+\.png$/i.test(file)) {
        model = 'ltx-bridge';
    } else if (isMp4) {
        // Generic mp4 fallback — assume LTX-2.3 since that's the only video model wired in.
        model = 'ltx-2.3';
    }
    const map = {
        'qwen': { label: 'Qwen Image', color: 'badge-info' },
        'ernie': { label: 'Ernie Image', color: 'badge-error' },
        'ltx-2.3': { label: isMp4 ? 'LTX-2.3 Video' : 'LTX-2.3 Image', color: 'badge-success' },
        'ltx-bridge': { label: 'LTX Bridge', color: 'badge-success' },
        'wan2.2': { label: 'Wan 2.2 Video', color: 'badge-info' },
        'wan2.5': { label: 'Wan 2.5 Video', color: 'badge-info' },
        'heartmula': { label: 'Heartmula Music', color: 'badge-secondary' },
        'qwen-tts': { label: 'Qwen TTS', color: 'badge-warning' },
        'kokoro': { label: 'Kokoro TTS', color: 'badge-warning' },
    };
    const borderByKind = { 'video': 'border-primary', 'image': 'border-secondary', 'audio': 'border-warning' };
    if (model && map[model]) {
        return { label: map[model].label, color: map[model].color, border: borderByKind[kind], part, kind };
    }
    const fallback = {
        'video': { label: '🎬 video', color: 'badge-primary' },
        'image': { label: '🖼 image', color: 'badge-secondary' },
        'audio': { label: '🔊 audio', color: 'badge-warning' },
    }[kind];
    return { label: fallback.label, color: fallback.color, border: borderByKind[kind], part, kind };
}

// Map fleet stage → which model role's badge should pulse + which asset
// filename was produced for that stage. Used by the queue-item reveal so
// the active badge spins, and completed stages append a clickable asset link.
const _STAGE_ROLE = {
    'Concept': 'llm',
    'Base Image': 'base',
    'Video Chains': 'video',
    'Audio': 'audio',
    'TTS': 'tts',
    'Post Process': 'upscale',
    'Final Merge': 'video',
};
// ---- Asset filename resolution --------------------------------------------
// The fleet runner now writes slug-based filenames (e.g.
//   v1_sterile_chrome_corridors_algorithms_shep_base.png
// ) rather than the legacy `v{N}_base.png` shape this client used to
// synthesize. We maintain a module-level v_idx -> {base, video, final, ...}
// map populated from three sources:
//   1. WS `new_file` events (push the real filename as it lands)
//   2. SSR initial DOM (parse filenames out of the seeded preview-grid cards)
//   3. /assets infinite-scroll pages (parse each item.file as it streams in)
// When a synthesis-time lookup misses, `_resolveVidxAssets` fires a one-shot
// fetch to /assets/by-vidx as a fallback so subsequent re-renders pick up
// the real name. We always fall back to the synthesized name on a true miss
// so the link/thumbnail never breaks outright — at worst it 404s the same
// way the pre-fix code did.
const _assetsByVidx = new Map();   // v_idx -> {base?, video?, final?, audio?, tts?}
const _vidxResolveInflight = new Set();  // dedupe concurrent /assets/by-vidx calls

function _ingestAssetFilename(filename) {
    if (!filename || typeof filename !== 'string') return;
    // Base image: v{N}_<anything>_base.png (also matches the legacy
    // v{N}_base.png since "_base.png" is a valid match for the `(.+)_base.png`
    // tail when <anything> is empty — see the explicit base regex below).
    let m = filename.match(/^v(\d+)(?:_.+)?_base\.png$/);
    if (m) {
        const v = parseInt(m[1], 10);
        const rec = _assetsByVidx.get(v) || {};
        rec.base = filename;
        _assetsByVidx.set(v, rec);
        return;
    }
    // Final merge: FINAL_{N}.mp4 (with optional suffix like FINAL_{N}_x.mp4).
    m = filename.match(/^FINAL_(\d+)(?:[._].*)?\.mp4$/);
    if (m) {
        const v = parseInt(m[1], 10);
        const rec = _assetsByVidx.get(v) || {};
        rec.final = filename;
        _assetsByVidx.set(v, rec);
        return;
    }
    // Video chain segment: v{N}_c{M}.mp4 or v{N}_<slug>_c{M}.mp4.
    m = filename.match(/^v(\d+)(?:_.+)?_c(\d+)\.mp4$/);
    if (m) {
        const v = parseInt(m[1], 10);
        const rec = _assetsByVidx.get(v) || {};
        // Keep the highest c_idx as the representative video (that's the
        // latest chain segment for this v_idx).
        const cIdx = parseInt(m[2], 10);
        const prevC = rec._video_c_idx || 0;
        if (cIdx >= prevC) {
            rec.video = filename;
            rec._video_c_idx = cIdx;
            _assetsByVidx.set(v, rec);
        }
        return;
    }
    // Bare v{N}.mp4 or v{N}.wav — generic per-vidx artefact.
    m = filename.match(/^v(\d+)\.(mp4|wav)$/);
    if (m) {
        const v = parseInt(m[1], 10);
        const rec = _assetsByVidx.get(v) || {};
        if (m[2] === 'mp4') rec.video = rec.video || filename;
        else rec.audio = rec.audio || filename;
        _assetsByVidx.set(v, rec);
        return;
    }
    // ffmpeg last-frame bridge between video chains: v{N}_f{M}.png
    // (and v{N}_<slug>_f{M}.png if a slug variant ever lands).
    m = filename.match(/^v(\d+)(?:_.+)?_f(\d+)\.png$/);
    if (m) {
        const v = parseInt(m[1], 10);
        const i = parseInt(m[2], 10);
        const rec = _assetsByVidx.get(v) || {};
        rec.bridges = rec.bridges || {};
        rec.bridges[i] = filename;
        _assetsByVidx.set(v, rec);
        return;
    }
}

async function _resolveVidxAssets(v_idx) {
    if (!v_idx) return;
    if (_vidxResolveInflight.has(v_idx)) return;
    const cached = _assetsByVidx.get(v_idx);
    // If we already have base + final, no need to refetch — those are the
    // two slots the dashboard actually links to today.
    if (cached && cached.base && cached.final) return;
    _vidxResolveInflight.add(v_idx);
    try {
        const r = await fetch(`/assets/by-vidx?v_idx=${v_idx}`);
        if (!r.ok) return;
        const j = await r.json();
        if (j && j.assets && typeof j.assets === 'object') {
            const rec = _assetsByVidx.get(v_idx) || {};
            Object.assign(rec, j.assets);
            _assetsByVidx.set(v_idx, rec);
        }
    } catch (_) {
        // Best-effort; the synthesized fallback still renders.
    } finally {
        _vidxResolveInflight.delete(v_idx);
    }
}

const _STAGE_ASSET = (stage, v_idx, c_idx) => {
    if (!v_idx) return null;
    const cached = _assetsByVidx.get(v_idx);
    if (stage === 'Base Image') {
        if (cached && cached.base) return cached.base;
        // Fire-and-forget resolve so the next re-render picks up the real
        // name. Returning the synthesized fallback uses the current
        // "slop_<idx>_" prefix; the resolver tries both that and the
        // legacy "v<idx>_" form before settling on a real on-disk match.
        _resolveVidxAssets(v_idx);
        return `slop_${v_idx}_base.png`;
    }
    if (stage === 'Video Chains' && c_idx > 0) {
        // Per-chain assets are numerically suffixed (_c{M}.mp4). Use the
        // current prefix; resolver picks the real on-disk filename.
        return `slop_${v_idx}_c${c_idx}.mp4`;
    }
    if (stage === 'Final Merge') {
        if (cached && cached.final) return cached.final;
        _resolveVidxAssets(v_idx);
        return `FINAL_${v_idx}.mp4`;
    }
    return null;
};
const _STAGE_TEXT = {
    'Concept': 'generating prompts',
    'Base Image': 'rendering image',
    'Video Chains': 'rendering video part',
    'Audio': 'composing music',
    'TTS': 'recording voiceover',
    'Post Process': 'upscaling',
    'Final Merge': 'merging final',
};

// Backend-driven activity heartbeat. The server emits a `render_heartbeat`
// WS event every ~2 s while a stage is running, carrying an `expires_ts`
// (~15 s in the future). The queue-header-activity label only shows while
// `Date.now() < _renderHeartbeat.expiresAt` — so if the backend stalls,
// the next heartbeat never arrives and the 1 Hz expiry ticker hides the
// spinner cleanly. Replaces the old client-side derivation off state.step.
let _renderHeartbeat = null;
// Track whether the heartbeat was live on the previous tick so we can detect
// the idle→live transition and fire the ignite burst exactly once per render.
let _renderWasLive = false;
// Timer handle for the ignite-class cleanup fallback.
let _renderIgniteTimer = null;
// Render the heartbeat text as a row of per-character spans so the assembly-
// line CSS keyframes (bounce/border-pulse) can target each glyph individually.
// Each char carries its position fraction `--char-pos` (0..1) so the 1 Hz
// fill pass can compare it to the host's `--progress` value and flip a
// `.filled` class for inverted text on the "completed" leading characters.
// Spaces are tagged with data-space="1" so the fill pass skips them.
function _paintRenderText(host, text) {
    if (!host) return;
    const safe = (text == null ? '' : String(text));
    if (host.dataset.renderedText === safe) return;
    host.dataset.renderedText = safe;
    host.classList.add('render-anim');
    const total = Math.max(1, safe.length);
    let html = '';
    for (let i = 0; i < safe.length; i++) {
        const ch = safe[i];
        const isSpace = ch === ' ';
        const pos = (i / total).toFixed(4);
        // index drives the per-char animation-delay so the assembly-line
        // bounce ripples left-to-right rather than firing in unison.
        html += `<span class="render-anim-char"${isSpace ? ' data-space="1"' : ''} `
            + `style="--char-pos:${pos};--char-i:${i};">`
            + (isSpace ? '&nbsp;' : _htmlEscape(ch))
            + '</span>';
    }
    host.innerHTML = html;
}

// Drive the per-char block-fill from the current stage's ETA progress.
// Mirrors _buildActiveJobProgressBar's stageProgressFraction so the queue
// header animation and the progress bar agree on "how far through this
// stage we are". Defaults to 30s when the stage has no historical avg yet
// (first run) and 0 fraction when there's no live stage timestamp.
function _updateRenderTextProgress(host) {
    if (!host) return;
    const curStep = (_lastTick && _lastTick.state && _lastTick.state.step) || '';
    let frac = 0;
    if (curStep && _stageStartTs) {
        const avg = _stageAvgSeconds(curStep) || 30;
        frac = Math.min(1, ((Date.now() - _stageStartTs) / 1000) / avg);
        if (!isFinite(frac) || frac < 0) frac = 0;
    }
    host.style.setProperty('--progress', frac.toFixed(4));
    // Once we settle past the assembly-line bounce window, mark the host
    // so the keyframe animation is paused via CSS — the fill takes over.
    host.dataset.settled = frac >= 0.2 ? '1' : '0';
    const chars = host.children;
    for (let i = 0; i < chars.length; i++) {
        const c = chars[i];
        if (c.dataset.space === '1') {
            // Spaces never fill — they remain blank dividers.
            if (c.classList.contains('filled')) c.classList.remove('filled');
            continue;
        }
        const pos = parseFloat(c.style.getPropertyValue('--char-pos')) || 0;
        const shouldFill = pos <= frac && frac > 0;
        if (shouldFill !== c.classList.contains('filled')) {
            c.classList.toggle('filled', shouldFill);
        }
    }
}

// Animation styles cycled through every ~5 s on the render-anim host
// so the heartbeat reads as varied motion (bounce → wobble → pulse →
// jump → wave → swap → repeat). Order is shuffled once on first paint
// so the same stage doesn't always start with the same style.
// Rendering animation style roster — motion-first; colour-only styles
// (pulse, ping, ripple, morph) were moved to the paused palette.
// Default rendering style is 'bounce' (index 0); user can pick any
// from Settings → Appearance. New styles: card, bucket, spiral, mobius,
// breathe, sway, fountain.
const _RENDER_ANIM_STYLES = [
    'bounce', 'wobble', 'jump', 'wave', 'swap',
    'spin', 'flip', 'glitch', 'shake', 'orbit', 'typewriter',
    'nod', 'stretch', 'skew', 'drop', 'slide', 'pacman', 'cube',
    'card', 'bucket', 'spiral', 'mobius', 'breathe', 'sway', 'fountain',
];
let _renderAnimStyleIdx = Math.floor(Math.random() * _RENDER_ANIM_STYLES.length);
function _rotateRenderAnimStyle(host) {
    if (!host) return;
    _renderAnimStyleIdx = (_renderAnimStyleIdx + 1) % _RENDER_ANIM_STYLES.length;
    host.dataset.animStyle = _RENDER_ANIM_STYLES[_renderAnimStyleIdx];
}

// Fire the gradient-explosion ignite burst on the render-anim host.
// Adds .render-ignite which triggers the CSS keyframes, then removes it
// once the animation ends (animationend) or after 3.1 s max (fallback).
function _fireRenderIgnite(host) {
    if (!host) return;
    // Clear any in-flight cleanup.
    if (_renderIgniteTimer) { clearTimeout(_renderIgniteTimer); _renderIgniteTimer = null; }
    host.classList.remove('render-ignite');
    // Force reflow so re-adding the class restarts the animation.
    void host.offsetWidth;
    host.classList.add('render-ignite');
    const cleanup = () => {
        host.classList.remove('render-ignite');
        host.removeEventListener('animationend', onEnd);
        if (_renderIgniteTimer) { clearTimeout(_renderIgniteTimer); _renderIgniteTimer = null; }
    };
    const onEnd = (e) => { if (e.target === host) cleanup(); };
    host.addEventListener('animationend', onEnd, { once: true });
    // Safety fallback in case animationend never fires (e.g. display:none).
    _renderIgniteTimer = setTimeout(cleanup, 3200);
}

function _applyRenderHeartbeat() {
    const headerAct = document.getElementById('queue-header-activity');
    if (!headerAct) return;
    const txtEl = headerAct.querySelector('[data-queue-header-activity-text]');
    const live = !!(_renderHeartbeat && Date.now() < _renderHeartbeat.expiresAt);
    headerAct.style.display = live ? 'inline-flex' : 'none';
    if (live && txtEl && _renderHeartbeat.text) {
        // Repaint per-char spans only when the underlying text actually
        // changes (cheap idempotent guard inside _paintRenderText), then
        // run the fill pass every tick so the bar advances live.
        _paintRenderText(txtEl, _renderHeartbeat.text);
        _updateRenderTextProgress(txtEl);
        // Initial style assignment — only on first paint.
        if (!txtEl.dataset.animStyle) {
            txtEl.dataset.animStyle = _RENDER_ANIM_STYLES[_renderAnimStyleIdx];
        }
        // Fire the ignite burst on the idle→live transition.
        if (!_renderWasLive) {
            _fireRenderIgnite(txtEl);
        }
    }
    if (!live && _renderHeartbeat) _renderHeartbeat = null;
    _renderWasLive = live;
}
// Cycle animation styles every 5 s while a heartbeat is live.
setInterval(() => {
    if (!_renderHeartbeat || Date.now() > _renderHeartbeat.expiresAt) return;
    const headerAct = document.getElementById('queue-header-activity');
    if (!headerAct) return;
    const txtEl = headerAct.querySelector('[data-queue-header-activity-text]');
    _rotateRenderAnimStyle(txtEl);
}, 5000);
// 1 Hz expiry tick — fires regardless of WS traffic so a stalled backend
// hides the label within ~1 s of the TTL elapsing.
setInterval(_applyRenderHeartbeat, 1000);

// =====================================================================
// Chat-thinking ephemeral bubble — driven by `chat_thinking` WS events.
// The /chat endpoint emits {phase:'received'|'calling'|'done'} signals
// while it processes a user message. Client renders an in-flight thought
// bubble (NOT in _chatGetTree — purely ephemeral) at the bottom of the
// chat log while phase is 'received' or 'calling'. Each signal pushes
// an 8 s dead-man timeout into the future; if heartbeats stop arriving,
// a 1 Hz expiry tick auto-hides the bubble. A 'done' signal hides
// immediately. The bubble carries `.chat-thought-active` so the cogs
// CSS animation runs.
// =====================================================================
let _chatThinkingExpiresAt = 0;
const _CHAT_THINKING_EPHEMERAL_ID = 'chat-thinking-ephemeral';

function _chatThinkingEphemeralHTML() {
    // Re-use the same cogs SVG markup that `_renderChatLog` emits so the
    // visuals match exactly. Inline copy to avoid coupling to render-helper
    // closures. Active class drives @keyframes via app.css.
    return `<div class="chat chat-start" id="${_CHAT_THINKING_EPHEMERAL_ID}">
        <div class="chat-thought text-xs chat-thought-active">
            <span class="chat-cogs" aria-hidden="true" title="thinking…">
                <svg class="chat-cog" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zm9.4 3.5c0-.5 0-1-.1-1.5l2-1.5-2-3.4-2.3.8c-.8-.7-1.7-1.2-2.7-1.5L15.8 2h-3.6l-.5 2.4c-1 .3-1.9.8-2.7 1.5l-2.3-.8-2 3.4 2 1.5c-.1.5-.1 1-.1 1.5s0 1 .1 1.5l-2 1.5 2 3.4 2.3-.8c.8.7 1.7 1.2 2.7 1.5l.5 2.4h3.6l.5-2.4c1-.3 1.9-.8 2.7-1.5l2.3.8 2-3.4-2-1.5c.1-.5.1-1 .1-1.5z"/>
                </svg>
                <svg class="chat-cog-rev" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zm9.4 3.5c0-.5 0-1-.1-1.5l2-1.5-2-3.4-2.3.8c-.8-.7-1.7-1.2-2.7-1.5L15.8 2h-3.6l-.5 2.4c-1 .3-1.9.8-2.7 1.5l-2.3-.8-2 3.4 2 1.5c-.1.5-.1 1-.1 1.5s0 1 .1 1.5l-2 1.5 2 3.4 2.3-.8c.8.7 1.7 1.2 2.7 1.5l.5 2.4h3.6l.5-2.4c1-.3 1.9-.8 2.7-1.5l2.3.8 2-3.4-2-1.5c.1-.5.1-1 .1-1.5z"/>
                </svg>
            </span>
            <span class="opacity-70 text-[10px]">thinking…</span>
        </div>
    </div>`;
}

function _showChatThinkingBubble() {
    const log = document.getElementById('subjects-chat-log');
    if (!log) return;
    let el = document.getElementById(_CHAT_THINKING_EPHEMERAL_ID);
    if (!el) {
        // Append to the end of the log; the placeholder text in an empty
        // log gets pushed up, which is fine — the cogs read as "thinking".
        log.insertAdjacentHTML('beforeend', _chatThinkingEphemeralHTML());
        el = document.getElementById(_CHAT_THINKING_EPHEMERAL_ID);
    }
    if (el) {
        log.scrollTop = log.scrollHeight;
    }
}

function _hideChatThinkingBubble() {
    const el = document.getElementById(_CHAT_THINKING_EPHEMERAL_ID);
    if (el && el.parentNode) el.parentNode.removeChild(el);
    _chatThinkingExpiresAt = 0;
}

function _applyChatThinkingExpiry() {
    if (_chatThinkingExpiresAt && Date.now() > _chatThinkingExpiresAt) {
        _hideChatThinkingBubble();
    }
}

function _onChatThinkingSignal(d) {
    const phase = d && d.phase;
    if (phase === 'received' || phase === 'calling') {
        // Push the dead-man timeout 8 s into the future. As long as
        // heartbeats arrive at <8 s intervals, the bubble keeps spinning.
        _chatThinkingExpiresAt = Date.now() + 8000;
        _showChatThinkingBubble();
    } else if (phase === 'done') {
        _hideChatThinkingBubble();
    }
}

// 1 Hz dead-man check — fires regardless of WS traffic so a stalled
// backend (no `done`, no heartbeats) auto-hides the bubble within ~1 s
// of the 8 s window elapsing.
setInterval(_applyChatThinkingExpiry, 1000);

// Pretty display label for a configured model id (from config snapshot).
function _modelDisplayName(id, role) {
    if (!id || id === 'none') return '';
    const map = {
        'qwen': 'Qwen Image',
        'qwen-image': 'Qwen Image',
        'ernie': 'Ernie Image',
        'ltx-2.3': role === 'image' ? 'LTX-2.3 Image' : 'LTX-2.3 Video',
        'wan2.2': 'Wan 2.2 Video',
        'wan2.5': 'Wan 2.5 Video',
        'heartmula': 'Heartmula Music',
        'qwen-tts': 'Qwen TTS',
        'kokoro': 'Kokoro TTS',
    };
    return map[id] || id;
}

// Build badge HTML strings from a config snapshot, including the LLM that wrote the prompt.
// Badge and progress-bar colour theming.
// badge_theme: "themed" (default) | "custom"
// badge_custom_color: any CSS colour string (only used when theme=custom)
// Hydrated from config on load; updated live when user changes settings.
let _badgeTheme = 'themed';
let _badgeCustomColor = '#7c3aed';

// Returns the class string OR inline style needed for a badge element.
// When themed: returns 'badge-<tone>' (DaisyUI token, follows active theme).
// When custom: returns '' for class + the inline style is applied by callers
// using _badgeStyle().
function _badgeToneCls(tone) {
    return _badgeTheme === 'themed' ? `badge-${tone}` : '';
}
function _badgeStyleAttr() {
    return _badgeTheme === 'custom'
        ? ` style="background:${_badgeCustomColor};color:#fff;border-color:${_badgeCustomColor};"`
        : '';
}

window.loadBadgeThemeSetting = function(cfg) {
    _badgeTheme = (cfg && cfg.badge_theme) || 'themed';
    _badgeCustomColor = (cfg && cfg.badge_custom_color) || '#7c3aed';
    window._pausingAnimStyle = (cfg && cfg.pausing_anim_style) || 'pulse';

    // Sync the settings UI controls if they're already in the DOM.
    const modeThemed  = document.getElementById('badge-theme-themed');
    const modeCustom  = document.getElementById('badge-theme-custom');
    const colorPicker = document.getElementById('badge-custom-color');
    const pickerRow   = document.getElementById('badge-color-picker-row');
    const animSelect  = document.getElementById('pausing-anim-select');

    if (modeThemed)  modeThemed.checked  = (_badgeTheme === 'themed');
    if (modeCustom)  modeCustom.checked  = (_badgeTheme === 'custom');
    if (colorPicker) colorPicker.value   = _badgeCustomColor;
    if (pickerRow)   pickerRow.style.display = (_badgeTheme === 'custom') ? '' : 'none';
    if (animSelect)  animSelect.value    = window._pausingAnimStyle;
};

window.saveBadgeTheme = async function(theme) {
    _badgeTheme = theme;
    const pickerRow = document.getElementById('badge-color-picker-row');
    if (pickerRow) pickerRow.style.display = (theme === 'custom') ? '' : 'none';
    await _saveConfig({ badge_theme: theme });
};

window.saveBadgeCustomColor = async function(color) {
    _badgeCustomColor = color;
    await _saveConfig({ badge_custom_color: color });
};

window.savePausingAnimStyle = async function(style) {
    window._pausingAnimStyle = style;
    await _saveConfig({ pausing_anim_style: style });
};

// When `qTs` is provided (the queue item's timestamp / id), each badge becomes a
// clickable button that opens the per-stage settings popup. The badges remain
// rendered as <span>s in contexts where there is no item to associate with
// (e.g. the live "Will use" preview).
function _configModelBadges(snap, llmModelId, activeRole, qTs) {
    // Order matches the pipeline strip: LLM → Image → Video → Music → Voice → Post.
    // activeRole (optional) — string like 'llm'/'base'/'video'/'audio'/'tts'/
    // 'upscale' — that badge gets an inline spinner so the user can see at
    // a glance which model is currently working.
    const spin = role => activeRole === role
        ? '<span class="loading loading-spinner loading-xs mr-1"></span>'
        : '';
    // Render either a clickable <button> (when we know which queue item this
    // strip belongs to) or a passive <span>. The button stops propagation so
    // it doesn't toggle the parent <details> reveal (same gotcha as PR #44).
    const mk = (role, tone, title, label) => {
        const cls = _badgeToneCls(tone);
        const sty = _badgeStyleAttr();
        if (qTs == null) {
            return `<span class="badge badge-xs ${cls} gap-1" title="${_htmlEscape(title)}"${sty}>${spin(role)}${label}</span>`;
        }
        return `<button type="button" class="badge badge-xs ${cls} gap-1 cursor-pointer" title="${_htmlEscape(title)} — click for settings" onclick='event.stopPropagation(); openModelSettingsPopup(${JSON.stringify(role)}, ${qTs})'${sty}>${spin(role)}${label}</button>`;
    };
    const out = [];
    if (llmModelId) {
        const short = llmModelId.replace(/^\//, '').replace(/\.gguf$/i, '');
        out.push(mk('llm', 'accent', `prompt LLM: ${llmModelId}`, _htmlEscape(short)));
    }
    if (snap.base_model) out.push(mk('base', 'info', 'image model', _htmlEscape(_modelDisplayName(snap.base_model, 'image'))));
    if (snap.video_model) out.push(mk('video', 'success', 'video model', _htmlEscape(_modelDisplayName(snap.video_model, 'video'))));
    if (snap.audio_model && snap.audio_model !== 'none') out.push(mk('audio', 'secondary', 'music model', _htmlEscape(_modelDisplayName(snap.audio_model, 'audio'))));
    if (snap.tts_model && snap.tts_model !== 'none') out.push(mk('tts', 'warning', 'voice model', _htmlEscape(_modelDisplayName(snap.tts_model, 'audio'))));
    if (snap.upscale_model && snap.upscale_model !== 'none') out.push(mk('upscale', 'warning', 'upscaler', _htmlEscape(snap.upscale_model)));
    return out;
}


// Open a small read-only popup describing the settings for one stage of one
// queue item. Pulled fresh from `_lastTick` so values reflect the snapshot
// captured when the item was queued (with global config as a fallback for
// fields that aren't snapshotted, e.g. LLM provider).
function openModelSettingsPopup(role, qTs) {
    const tick = _lastTick || {};
    const cfg = tick.config || {};
    const queue = tick.queue || [];
    // qTs === 0 is the synthetic "currently running" placeholder used by
    // renderItem; treat it as "use the live config snapshot".
    let q = null;
    if (qTs) q = queue.find(x => (x.ts || 0) === qTs) || null;
    const snap = (q && q.config_snapshot) || cfg;
    const llm = (cfg.llm || {});
    const promptText = (q && q.prompt) || (tick.state && tick.state.current_prompt) || '';

    const titleMap = {
        llm: 'LLM — prompt rewriter',
        base: 'Image stage',
        video: 'Video stage',
        audio: 'Music stage',
        tts: 'Voice stage',
        upscale: 'Post-process stage',
    };
    const title = titleMap[role] || 'Stage settings';

    // Each row is a [label, value] pair; missing values render as em-dash.
    const rows = [];
    const push = (k, v) => rows.push([k, (v == null || v === '') ? '—' : v]);

    if (role === 'llm') {
        push('Provider', llm.provider || 'auto');
        push('Model', llm.model_id || 'auto-pick');
        push('Base URL', llm.base_url);
        push('Rewrite mode', cfg.enhancer_prompt ? 'cinematic-director (custom)' : 'default');
        push('Subject', promptText);
    } else if (role === 'base') {
        push('Model', _modelDisplayName(snap.base_model, 'image'));
        push('Size', snap.size);
        push('Quality ramp', snap.quality_ramp ? 'on' : 'off');
        push('Prompt override', snap.image_prompt_override);
    } else if (role === 'video') {
        push('Model', _modelDisplayName(snap.video_model, 'video'));
        push('Frames', snap.frames);
        push('Parts', snap.chains);
        push('Quality ramp', snap.video_quality_ramp ? 'on' : 'off');
    } else if (role === 'audio') {
        push('Model', _modelDisplayName(snap.audio_model, 'audio'));
        push('Music gain', (snap.music_gain_db != null) ? `${snap.music_gain_db} dB` : null);
        push('Fade', (snap.fade_s != null) ? `${snap.fade_s}s` : null);
    } else if (role === 'tts') {
        push('Model', _modelDisplayName(snap.tts_model, 'audio'));
        push('Voice preset', snap.tts_voice || snap.voice_preset);
        push('Voice gain', (snap.voice_gain_db != null) ? `${snap.voice_gain_db} dB` : null);
    } else if (role === 'upscale') {
        push('Upscaler', snap.upscale_model);
        push('Upscale on', snap.upscale ? 'yes' : 'no');
        push('Consolidation', snap.consolidation);
    }

    const body = rows.map(([k, v]) =>
        `<div class="text-base-content/50 uppercase tracking-widest text-[10px]">${_htmlEscape(k)}</div>` +
        `<div class="font-mono text-xs whitespace-pre-wrap break-words">${_htmlEscape(String(v))}</div>`
    ).join('');

    const titleEl = document.getElementById('model-settings-title');
    const bodyEl = document.getElementById('model-settings-body');
    if (titleEl) titleEl.textContent = title;
    if (bodyEl) {
        // Footer copy now contains an inline "Pipeline Settings" link
        // instead of a duplicate Open-Pipeline button in the modal-action
        // row. One affordance, less visual clutter — the link routes
        // through the same openPipeline() handler the button used.
        bodyEl.innerHTML = `<div class="grid grid-cols-[min-content_1fr] gap-x-3 gap-y-1">${body}</div>` +
            `<div class="text-[10px] text-base-content/40 italic mt-3">Read-only snapshot from when this item was queued. Use <a href="#" class="link link-primary" onclick="event.preventDefault();document.getElementById('model-settings-modal').close();openPipeline();">Pipeline Settings</a> to edit defaults for future items.</div>`;
    }
    const d = document.getElementById('model-settings-modal');
    if (d && d.showModal) d.showModal();
}

// Slop filter chips — toggle visibility of cards by data-slop-kind.
function _applySlopFilters() {
    const enabled = {};
    document.querySelectorAll('[data-slop-filter]').forEach(cb => {
        enabled[cb.dataset.slopFilter] = cb.checked;
    });
    // The 'assets' chip is a meta-filter: when OFF (default), the gallery
    // shows only final products (FINAL_*.mp4 — `data-slop-final="1"` on the
    // card). When ON, intermediate assets (chain mp4s, base pngs, bridge
    // frames) become visible too. The kind chips (video/image/audio) still
    // gate by media type independently.
    const showAssets = enabled.assets === true; // default false = only finals
    // Secondary 'frames' chip — only meaningful WHEN assets is ON. Frames
    // are bridge PNGs ffmpeg extracts as the last frame of each video
    // chain (handed to the next chain as its first frame). They're
    // technically intermediate assets so they qualify under 'assets',
    // but visually they're stills FROM videos already in the grid →
    // noisy by default. When assets is OFF, the frame state is moot
    // (frames are excluded by the assets filter regardless). When
    // assets is ON + frames is OFF, hide just the frame cards.
    const showFrames = enabled.frames === true;
    // Update the 'frames' chip's enabled/disabled visual + interaction
    // state to match the assets master chip — disabled when assets is
    // off, since toggling it has no effect there.
    const framesLabel = document.querySelector('.slop-filter-frames-label');
    const framesInput = document.querySelector('[data-slop-filter="frames"]');
    if (framesInput) framesInput.disabled = !showAssets;
    let vCount = 0;
    let iCount = 0;
    document.querySelectorAll('#preview-grid > [data-slop-kind]').forEach(card => {
        const kindOk = enabled[card.dataset.slopKind] !== false;
        const isFinal = card.dataset.slopFinal === '1';
        const isFrame = card.dataset.slopFrame === '1';
        const isSeed = card.dataset.slopSeed === '1';
        // Seeds (user-uploaded source images) are exempt from the
        // assets gate — without this, an upload would silently land
        // in the grid but stay hidden until the user toggled assets
        // on, which produced the "I uploaded but nothing happened"
        // confusion. Seeds also bypass the frames gate (they aren't
        // ffmpeg-extracted bridges).
        const passesAssets = isSeed || showAssets || isFinal;
        // Frames are a sub-class of assets — only show them when BOTH
        // assets and frames are on. (Finals are not frames so they're
        // unaffected.) Non-frame cards bypass this gate entirely.
        const passesFrames = !isFrame || (showAssets && showFrames);
        const visible = kindOk && passesAssets && passesFrames;
        card.style.display = visible ? '' : 'none';

        if (visible) {
            if (card.dataset.slopKind === 'video') vCount++;
            else if (card.dataset.slopKind === 'image') iCount++;
        }
    });

    // Update Slop Bar stats
    const vidSpan = document.getElementById('slop-bar-count-video');
    const imgSpan = document.getElementById('slop-bar-count-image');
    if (vidSpan) vidSpan.textContent = `${vCount} vids`;
    if (imgSpan) imgSpan.textContent = `${iCount} imgs`;
    // Empty-state class — set when no card is currently visible, so the
    // CSS `::before` empty-state hint surfaces. Cleared otherwise.
    const grid = document.getElementById('preview-grid');
    if (grid) {
        const anyVisible = vCount + iCount > 0;
        grid.classList.toggle('slop-empty-state', !anyVisible);
    }
}
document.addEventListener('change', e => {
    if (e.target.matches('[data-slop-filter]')) _applySlopFilters();
});
// Resizable split between Subjects (left) and Queue (right). The flex-basis
// of each side is persisted in localStorage so the user's preferred ratio
// survives reloads.
function _initSplitDivider() {
    const row = document.getElementById('split-row');
    const left = document.getElementById('split-left');
    const right = document.getElementById('split-right');
    const handle = document.getElementById('split-divider');
    if (!row || !left || !right || !handle) return;
    const KEY = 'slopfinity_split_ratio_v1';
    const applyRatio = (ratio) => {
        const r = Math.max(0.2, Math.min(0.8, ratio));
        left.style.flex = `${r} 1 0`;
        right.style.flex = `${1 - r} 1 0`;
    };
    const stored = parseFloat(localStorage.getItem(KEY));
    if (isFinite(stored)) applyRatio(stored);
    let dragging = false;
    handle.addEventListener('mousedown', (e) => { dragging = true; e.preventDefault(); document.body.style.userSelect = 'none'; });
    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const rect = row.getBoundingClientRect();
        const ratio = (e.clientX - rect.left) / rect.width;
        applyRatio(ratio);
        try { localStorage.setItem(KEY, String(Math.max(0.2, Math.min(0.8, ratio)))); } catch { }
    });
    document.addEventListener('mouseup', () => { dragging = false; document.body.style.userSelect = ''; });
}

document.addEventListener('DOMContentLoaded', () => {
    _applySlopFilters();
    _initSplitDivider();

    // data-slop-expanded was the toggle state for the now-removed
    // <details> wrapper around slop. With slop always inline, the
    // body flag is meaningless — and the position:fixed CSS that
    // keyed off it caused slop to overlay the upper pane. Both the
    // attribute and the CSS rules are gone.
    if (typeof _updateSingleLabels === 'function') _updateSingleLabels();
    if (typeof _updateChaosEnabled === 'function') _updateChaosEnabled();
    if (typeof _updateTerminateEnabled === 'function') _updateTerminateEnabled();
    if (typeof _updateGenModePill === 'function') _updateGenModePill();
    if (typeof _renderStageEtas === 'function') _renderStageEtas();
    // Suggestions on page load — passive only:
    //   1. Render cached chips for SIMPLE mode if a cache exists (no LLM
    //      call, just localStorage hydration so the user sees the last
    //      session's last batch).
    //   2. Endless idle shows the "press Start Story" hint.
    //   3. NO auto-fire on page load (old design — would burn LLM cycles
    //      every time the dashboard opened, even when the user hadn't
    //      asked for suggestions). The user clicks ↻ Refresh on the
    //      Suggestions badge when they actually want a fresh batch.
    const _curMode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
    const _endlessIdle = (_curMode === 'endless' && !_endlessRunning);
    if (_endlessIdle) {
        const { stack: box } = _getSuggestStack();
        if (box) box.innerHTML = "";
    } else {
        _renderCachedSuggestions(); // simple-mode-only; no-op for raw/chat
    }
    // Repaint the badge so the + ↔ ↻ swap reflects whether rows
    // ended up in the stack (cache hit) or not (empty → + bootstraps).
    if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
});

function updateOutputs(o) {
    if (!o) return;
    const f = document.getElementById('out-finals');
    const c = document.getElementById('out-chains');
    const b = document.getElementById('out-base');
    const l = document.getElementById('out-latest'); // legacy, removed from layout
    const pill = document.getElementById('out-latest-pill');
    const link = document.getElementById('out-latest-link');
    const thumb = document.getElementById('out-latest-thumb');
    if (pill && link) {
        if (o.latest_final) {
            link.textContent = o.latest_final;
            link.title = o.latest_final;
            link.href = '/files/' + encodeURIComponent(o.latest_final);
            link.onclick = (e) => { e.preventDefault(); openAssetInfo(o.latest_final); };
            // Tiny thumbnail — for mp4 use the <video> first-frame poster, for
            // png/jpg just an <img>. Click also opens the asset-info modal.
            if (thumb) {
                const url = '/files/' + encodeURIComponent(o.latest_final);
                const isMp4 = /\.mp4$/i.test(o.latest_final);
                thumb.innerHTML = isMp4
                    ? `<video src="${url}" muted preload="metadata" class="w-full h-full object-cover" title="${_htmlEscape(o.latest_final)}"></video>`
                    : `<img src="${url}" class="w-full h-full object-cover" title="${_htmlEscape(o.latest_final)}">`;
                thumb.onclick = () => openAssetInfo(o.latest_final);
            }
            pill.style.display = '';
        } else {
            pill.style.display = 'none';
        }
    }
    if (f) f.textContent = o.finals ?? 0;
    if (c) c.textContent = o.chains ?? 0;
    if (b) b.textContent = o.base_images ?? 0;
    if (l) {
        l.textContent = o.latest_final ? `latest: ${o.latest_final}` : '';
        l.style.display = o.latest_final ? 'block' : 'none';
    }
    // Chip counts come from the actual rendered cards — server-side counters
    // (`finals`/`chains`/`base_images`) under-count now that chain-stage PNGs
    // (v<N>_base, v<N>_f<M>) are bucketed under the `video` filter kind.
    const chipV = document.querySelector('[data-chip-count="video"]');
    const chipI = document.querySelector('[data-chip-count="image"]');
    const chipA = document.querySelector('[data-chip-count="audio"]');
    if (chipV) chipV.textContent = document.querySelectorAll('#preview-grid > [data-slop-kind="video"]').length;
    if (chipI) chipI.textContent = document.querySelectorAll('#preview-grid > [data-slop-kind="image"]').length;
    // 'audio' is the legacy chip — count both music + speech under it for
    // back-compat in case the old chip is still in the DOM during transition.
    if (chipA) chipA.textContent = document.querySelectorAll('#preview-grid > [data-slop-kind="music"], #preview-grid > [data-slop-kind="speech"]').length;
    const chipM = document.querySelector('[data-chip-count="music"]');
    const chipS = document.querySelector('[data-chip-count="speech"]');
    if (chipM) chipM.textContent = document.querySelectorAll('#preview-grid > [data-slop-kind="music"]').length;
    if (chipS) chipS.textContent = document.querySelectorAll('#preview-grid > [data-slop-kind="speech"]').length;
    // Parts chip — count of intermediate/partial assets (chain mp4s, base
    // pngs, bridge frames) currently in the gallery. Anything NOT marked
    // data-slop-final="1" is a part. This fixes the chip showing "0" when
    // partials are clearly visible — the count was never wired.
    const chipP = document.querySelector('[data-chip-count="parts"]')
        || document.querySelector('[data-chip-count="assets"]');
    if (chipP) chipP.textContent = document.querySelectorAll('#preview-grid > [data-slop-kind][data-slop-final="0"]').length;
    // 'frames' chip count — bridge PNGs only. Counted independently of
    // visibility so the badge shows the TOTAL pool the chip would
    // toggle, not the currently-visible subset (matches every other
    // chip-count's semantics).
    const chipF = document.querySelector('[data-chip-count="frames"]');
    if (chipF) chipF.textContent = document.querySelectorAll('#preview-grid > [data-slop-frame="1"]').length;
}

function updateScheduler(sc) {
    if (!sc) return;
    const events = (sc.events || []).slice(-5);
    // Settings-modal Scheduler tab badge — the only place scheduler state
    // surfaces in the UI now (the main-page strip was removed; the Settings
    // → Scheduler tab is the proper home for this content).
    const statusBadge = $('sched-status-badge');
    if (statusBadge) {
        statusBadge.innerText = sc.paused ? '⏸ Paused' : '▶ Running';
        statusBadge.className = 'badge font-mono ' + (sc.paused ? 'badge-warning' : 'badge-success');
    }
    const tipFor = e => JSON.stringify(e).replace(/"/g, '&quot;');
    const labelFor = e => (e.stage && e.model) ? `${e.stage}/${e.model}` : (e.type || 'event');
    const renderEvents = (container, idPrefix) => {
        if (!container) return;
        if (!events.length) {
            container.innerHTML = '<span class="text-[11px] text-base-content/40 italic">no events yet</span>';
            return;
        }
        container.innerHTML = events.map((e, i) =>
            `<span id="${idPrefix}-${i}" class="badge badge-sm ${schedBadgeClass(e.type)} tooltip tooltip-bottom" data-tip="${tipFor(e)}">${e.type}: ${labelFor(e)}</span>`
        ).join('');
    };
    renderEvents($('sched-recent'), 'sched-recent-event');
}

// Cache last WS tick for diagnostics-copy / manual refresh.
let _lastTick = null;

// Rolling GPU utilization history. Used to gate automatic LLM suggestion
// fetches on a sustained-idle GPU, instead of the older queue/fleet-mode
// heuristic which missed ad-hoc GPU work (manual ComfyUI runs, etc.) and
// fired spuriously during fleet-stage transitions.
const _GPU_IDLE_THRESHOLD_PCT = 5;
const _GPU_IDLE_REQUIRED_SECONDS = 3;
const _GPU_HISTORY_MAX = 30; // keep ~30 s of samples (WS ticks ~1 Hz)
const _gpuPctHistory = []; // each entry: { ts: ms, pct: 0..100 }

// ---------------------------------------------------------------------------
// Auto-fetch gating audit — every auto-fetch surface that talks to the LLM
// (or anything else that competes with running pipelines) MUST consult
// BOTH gates before firing:
//
//   1. _autoSuggestDisabled() — Settings → LLM → Generation toggle. When
//      ON, every auto path bails immediately. The 🎲 button is NOT gated.
//   2. _isGpuIdleEnough() — GPU has been at <=5% for >=3 consecutive
//      seconds. Catches ad-hoc GPU work the older queue/fleet check
//      missed. The 🎲 button is also NOT gated by this.
//
// Gated callers (auto-fetch surfaces) AFTER the marquee rewrite:
//   - tryAutoSuggest (page-load auto-suggest):
//       gated by _autoSuggestDisabled(); then by _isGpuIdleEnough()
//       (retries every 1 s while GPU busy).
//   - _maybePrefetch (pointerenter / idle triggers):
//       gated by _autoSuggestDisabled() then _isGpuIdleEnough();
//       silent no-op when either fails.
//   - _resetPrefetchIdleTimer's drain → _appendSuggestBatchRow: pulls
//       a previously-buffered batch into a new marquee row. No fetch
//       happens, so this path is unaffected by both gates BUT the
//       top-up _maybePrefetch() it then triggers IS gated.
//
// NOT gated (explicit user intent):
//   - regenSuggestions() — the 🎲 Suggest button. Always fetches fresh.
// ---------------------------------------------------------------------------

// Returns true when the user has flipped the Settings → LLM → Generation
// "Disable automatic suggestion fetches" toggle ON. Falls back to false
// (auto-suggest enabled) until the first WS tick lands.
// Aliased so older call sites that say _autoSuggestDisabled() keep
// working — the previous agent renamed the canonical helper without
// updating every caller, which threw "is not defined" at runtime and
// silently broke later JS execution (subjects card stopped rendering).
window._autoSuggestDisabled = function () { return _isAutoSuggestDisabled(); };
function _isAutoSuggestDisabled() {
    if (!_lastTick || !_lastTick.config) return false;
    const c = _lastTick.config;
    if (c.auto_suggest_enabled !== undefined) return !c.auto_suggest_enabled;
    return !!c.suggest_auto_disabled;
}

// Spiffy mode: per-row prompt-name pill cluster in simple mode. The
// `endless` mode renders this cluster unconditionally; simple mode opts
// in via the Settings → Generation toggle. Default off. The override
// captures the optimistic state when the user toggles the checkbox so
// effects are immediate (no Save → tick round-trip needed for feedback).
let _perRowPromptsOverride = null;
function _isPerRowPromptsEnabled() {
    if (_perRowPromptsOverride !== null) return _perRowPromptsOverride;
    if (!_lastTick || !_lastTick.config) return false;
    return !!_lastTick.config.suggest_per_row_prompts;
}
window._isPerRowPromptsEnabled = _isPerRowPromptsEnabled;

// Inline handler invoked from the Settings → Generation checkbox.
// Captures the optimistic override and, when flipping OFF→ON in simple
// mode with no rows yet, auto-seeds an empty row using the currently
// selected prompt so the user immediately sees the cluster's effect.
function _onPerRowPromptsToggle(checked) {
    _perRowPromptsOverride = !!checked;
    if (!checked) return;
    const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : '';
    if (mode !== 'simple') return;

    const { stack } = _getSuggestStack();
    if (!stack) return;
    if (stack.querySelector('.suggest-marquee-row')) return;
    const promptId = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : null;
    if (!promptId) return;
    if (typeof _appendSuggestBatchRow === 'function') {
        _appendSuggestBatchRow([], { promptId, rowIdx: 0 });
    }
}
window._onPerRowPromptsToggle = _onPerRowPromptsToggle;

function _isGpuIdleEnough() {
    const now = Date.now();
    const windowMs = _GPU_IDLE_REQUIRED_SECONDS * 1000;
    // Keep samples covering the window plus the most recent one preceding
    // it (so a 2 s tick rate still produces evidence spanning >=3 s).
    const recent = _gpuPctHistory.filter(e => now - e.ts <= windowMs + 2500);
    // Need >=2 samples AND the oldest one must be at least REQUIRED_SECONDS
    // old, otherwise we don't yet have evidence covering the full window.
    if (recent.length < 2) return false;
    const oldest = recent[0];
    if (now - oldest.ts < windowMs) return false;
    return recent.every(e => e.pct <= _GPU_IDLE_THRESHOLD_PCT);
}

async function schedPost(path) {
    try {
        const r = await fetch(path, { method: 'POST' });
        return r.json();
    } catch (e) {
        console.warn('schedPost failed', path, e);
        return { ok: false, error: String(e) };
    }
}

async function saveSchedSafety(v) {
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scheduler: { memory_safety_gb: parseInt(v, 10) } }),
        });
    } catch (e) {
        console.warn('saveSchedSafety failed', e);
    }
}

async function saveSchedUsePlanner(checked) {
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scheduler: { use_planner: !!checked } }),
        });
    } catch (e) {
        console.warn('saveSchedUsePlanner failed', e);
    }
}

// CPU offload toggles for LLM + TTS. Persisted under
// scheduler.{llm_cpu_only, tts_cpu_only}; the orchestrator + memory
// planner read these to skip the GPU-idle wait for the relevant stage
// and exclude the model from GPU budget accounting.
// VAE-grid detection prefs. Post the partial scheduler payload so the
// orchestrator + post-pass agree on whether to write sidecars.
async function saveVaeGridDetect(checked) {
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scheduler: { vae_grid_detect: !!checked } }),
        });
    } catch (e) {
        console.warn('saveVaeGridDetect failed', e);
    }
}
window.saveVaeGridDetect = saveVaeGridDetect;

async function saveVaeGridMethod(value) {
    if (value !== 'post-pass' && value !== 'comfyui') return;
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scheduler: { vae_grid_method: value } }),
        });
    } catch (e) {
        console.warn('saveVaeGridMethod failed', e);
    }
}
window.saveVaeGridMethod = saveVaeGridMethod;

// Lazy VAE-grid badge tagger. Walks the slop preview-grid for cards
// whose data-slop-file looks decoded (mp4 / png) and asks the server
// for the (cached) result. Cards with has_grid=true get a ⚠ chip
// appended to their card-body. Skipped when detect is off in config
// or when a card is already tagged. Throttled to one in-flight fetch
// at a time so a hundred cards don't fan out into a hundred GETs.
let _vaeGridFetchActive = false;
async function _scanSlopForGridArtefacts() {
    if (_vaeGridFetchActive) return;
    const cfg = (_lastTick && _lastTick.config && _lastTick.config.scheduler) || {};
    if (cfg.vae_grid_detect === false) return;
    const cards = document.querySelectorAll('#preview-grid > [data-slop-kind]:not([data-vae-grid-checked])');
    if (!cards.length) return;
    _vaeGridFetchActive = true;
    try {
        for (const card of cards) {
            const file = card.getAttribute('data-slop-file');
            const kind = card.getAttribute('data-slop-kind');
            // Only image + video assets have a meaningful VAE grid; audio
            // skips. Final mp4s pass — the server samples a frame.
            if (kind !== 'image' && kind !== 'video') {
                card.setAttribute('data-vae-grid-checked', '1');
                continue;
            }
            if (!file) {
                card.setAttribute('data-vae-grid-checked', '1');
                continue;
            }
            try {
                const r = await fetch('/vae_grid?file=' + encodeURIComponent(file));
                const d = await r.json();
                card.setAttribute('data-vae-grid-checked', '1');
                if (d && d.has_grid) {
                    const body = card.querySelector('.card-body');
                    if (body && !body.querySelector('.vae-grid-chip')) {
                        const chip = document.createElement('span');
                        chip.className = 'vae-grid-chip badge badge-xs badge-warning';
                        chip.title = `VAE grid artefact detected (period ${d.peak_freq}px, score ${d.score})`;
                        chip.textContent = '⚠ vae grid';
                        body.querySelector('.flex.flex-wrap')?.prepend(chip);
                    }
                }
            } catch (_) {
                card.setAttribute('data-vae-grid-checked', '1');
            }
        }
    } finally {
        _vaeGridFetchActive = false;
    }
}
window._scanSlopForGridArtefacts = _scanSlopForGridArtefacts;
// Trigger scan after each WS tick — cheap because already-checked
// cards are skipped via the data-attribute guard.
setInterval(_scanSlopForGridArtefacts, 4000);

// Three-way CPU mode selector for LLM + TTS.
// Persisted under scheduler.{llm_cpu_mode, tts_cpu_mode}.
// mode: "gpu" | "smart" | "cpu"
async function saveSchedCpuMode(role, mode) {
    if (role !== 'llm' && role !== 'tts') return;
    if (!['gpu', 'smart', 'cpu'].includes(mode)) return;
    const field = role === 'llm' ? 'llm_cpu_mode' : 'tts_cpu_mode';
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scheduler: { [field]: mode } }),
        });
    } catch (e) {
        console.warn('saveSchedCpuMode failed', e);
    }
}
window.saveSchedCpuMode = saveSchedCpuMode;
// Legacy alias kept for any Jinja templates that still call the old name
// with a boolean — converts True→cpu, False→gpu, else smart.
async function saveSchedCpuOffload(role, checked) {
    return saveSchedCpuMode(role, checked ? 'cpu' : 'gpu');
}
window.saveSchedCpuOffload = saveSchedCpuOffload;

// Per-model loading preferences. Reads every checkbox in the
// "Per-model loading preferences" grid and POSTs the union as
// model_loading.{sticky,eager_unload}: [model_ids].
async function saveModelLoadingPrefs() {
    const sticky = Array.from(document.querySelectorAll('[data-loadpref-sticky]'))
        .filter(el => el.checked)
        .map(el => el.getAttribute('data-loadpref-sticky'));
    const eager = Array.from(document.querySelectorAll('[data-loadpref-eager]'))
        .filter(el => el.checked)
        .map(el => el.getAttribute('data-loadpref-eager'));
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_loading: { sticky, eager_unload: eager } }),
        });
    } catch (e) {
        console.warn('saveModelLoadingPrefs failed', e);
    }
}

// Pre-tick the per-model loading checkboxes from the persisted config.
// Called from openSettings() so the grid reflects the current state
// every time the modal opens.
function _hydrateModelLoadingPrefs(cfg) {
    const ml = (cfg && cfg.model_loading) || {};
    const sticky = new Set(ml.sticky || []);
    const eager = new Set(ml.eager_unload || []);
    document.querySelectorAll('[data-loadpref-sticky]').forEach(el => {
        el.checked = sticky.has(el.getAttribute('data-loadpref-sticky'));
    });
    document.querySelectorAll('[data-loadpref-eager]').forEach(el => {
        el.checked = eager.has(el.getAttribute('data-loadpref-eager'));
    });
}

function updateDiagnostics(d) {
    const g = $('diag-gpu'); if (g && d.stats) g.innerText = d.stats.gpu + '%';
    const v = $('diag-vram'); if (v && d.stats) v.innerText = d.stats.vram + '%';
    const r = $('diag-ram'); if (r && d.stats) r.innerText = `${d.stats.ram_u} / ${Math.round(d.stats.ram_t)} GB`;
}

async function refreshDiagStatus() {
    const pre = $('diag-status');
    if (!pre) return;
    try {
        const r = await fetch('/scheduler/status');
        const d = await r.json();
        pre.innerText = JSON.stringify(d, null, 2);
    } catch (e) {
        pre.innerText = 'error: ' + String(e);
    }
}

async function copyDiagnostics() {
    const badge = $('diag-copy-badge');
    const blob = {
        stats: _lastTick && _lastTick.stats,
        storage: _lastTick && _lastTick.storage,
        ram: _lastTick && _lastTick.ram,
        scheduler: _lastTick && _lastTick.scheduler,
    };
    try {
        await navigator.clipboard.writeText(JSON.stringify(blob, null, 2));
        if (badge) { badge.className = 'badge badge-success badge-sm font-mono'; badge.innerText = 'copied'; }
    } catch (e) {
        if (badge) { badge.className = 'badge badge-error badge-sm font-mono'; badge.innerText = 'fail'; }
    }
    setTimeout(() => {
        if (badge) { badge.className = 'badge badge-ghost badge-sm font-mono'; badge.innerText = 'idle'; }
    }, 1500);
}

function updateRefresh() {
    const el = $('refresh-interval');
    if (!el) return;
    // Value is read on demand; WS is authoritative when connected.
    return parseInt(el.value || '5000', 10);
}

function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => {
        _wsConnected = true;
        const w = $('refresh-wrapper');
        if (w) w.style.display = 'none';
        const dot = $('live-dot');
        if (dot) {
            const inner = dot.querySelector('.rounded-full:last-child');
            const ping = dot.querySelector('.animate-ping');
            if (inner) inner.classList.replace('bg-error', 'bg-success');
            if (ping) ping.classList.replace('bg-error', 'bg-success');
        }
        _updateConnPill(_isRendering, _lastTick && _lastTick.state && _lastTick.state.mode);
    };
    ws.onmessage = e => {
        const d = JSON.parse(e.data);
        if (d.type === 'render_heartbeat') {
            // Backend-driven activity label with TTL — see _applyRenderHeartbeat.
            _renderHeartbeat = {
                text: d.text || 'working…',
                expiresAt: (typeof d.expires_ts === 'number' ? d.expires_ts * 1000 : Date.now() + 15000),
            };
            _applyRenderHeartbeat();
            return;
        }
        if (d.type === 'chat_thinking') {
            // Backend chat-endpoint lifecycle: received / calling / done.
            // 8 s dead-man timeout in case heartbeats stop arriving — see
            // _onChatThinkingSignal + _applyChatThinkingExpiry.
            _onChatThinkingSignal(d);
            return;
        }
        if (d.type === 'state') {
            // First successful WS state tick = the dashboard has live
            // data. Drop the splash now even if the 2.5s timeout
            // hasn't fired yet — feels snappy on a warm cache.
            if (typeof _hideSplash === 'function') _hideSplash();
            // Tone the percentage colour with the latest ticker column —
            // text-error in the ALARM band, otherwise the per-pill tone
            // class. Keeps the number visually in sync with the bar colour.
            //
            // ALL tickers in this dashboard are INVERTED (low = bad). For an
            // AI workload the desired state is "the box is being used" —
            // high GPU, high RAM (model resident), high CPU load (workers
            // running), high disk usage (results being saved). A near-zero
            // reading means the fleet has stalled or starved, which is
            // exactly the alarming case. So every ticker passes
            // {invert:true} and the alarm threshold is `pct < 20`. (See
            // _tickerHTML below for the matching ticker-column logic.)
            const _toneClass = (pct, baseTone, opts) => {
                const invert = !!(opts && opts.invert);
                const isAlarming = invert ? (pct < 20) : (pct > 80);
                return 'font-mono font-black ' + (isAlarming ? 'text-error' : 'text-' + baseTone);
            };
            // When a ticker hits 100% (full primary), apply a randomized
            // celebration animation to the percentage label. The animation
            // is picked ONCE on transition into 100% and persists until the
            // value falls back below — re-randomized on each new arrival
            // at 100% so users don't see the same wiggle every time.
            const _toneCelebrate = (el, pct) => {
                if (!el) return;
                if (pct >= 100) {
                    // Apply via _pickCelebrateClass so the user's Settings →
                    // Display choice is honoured (or random rotation if
                    // 'random'). Sticky once applied — only re-pick on the
                    // transition into 100 % (no celebrate-* class yet).
                    const has = Array.from(el.classList).some(c => c.startsWith('celebrate-'));
                    if (!has) {
                        el.classList.add(_pickCelebrateClass());
                    }
                } else {
                    CELEBRATE_STYLES.forEach(s => el.classList.remove(s));
                }
            };
            const gpuPct = d.stats.gpu;
            const gpuEl = $('g-v');
            if (gpuEl) { gpuEl.innerText = gpuPct + '%'; gpuEl.className = _toneClass(gpuPct, 'primary', { invert: true }); _toneCelebrate(gpuEl, gpuPct); }
            // Subtitle rows: GPU marketing name under GPU; CPU marketing name
            // under Load. stats.gpu_name / stats.cpu_name are detected once
            // at module import in stats.py and don't change. Only set when
            // non-empty so the "—" placeholder remains on detection failure.
            const _gName = $('g-name');
            if (_gName && d.stats.gpu_name && _gName.textContent !== d.stats.gpu_name) {
                _gName.textContent = d.stats.gpu_name;
                _gName.title = d.stats.gpu_name;
            }
            const _lName = $('l-name');
            if (_lName && d.stats.cpu_name && _lName.textContent !== d.stats.cpu_name) {
                _lName.textContent = d.stats.cpu_name;
                _lName.title = d.stats.cpu_name;
            }
            // Strix Halo has unified memory — rocm-smi's VRAM% always reads 0,
            // so derive RAM% from the host meminfo numbers (ram_u / ram_t).
            const ramPct = d.stats.ram_t > 0 ? Math.round((d.stats.ram_u / d.stats.ram_t) * 100) : 0;
            const ramEl = $('v-v');
            if (ramEl) { ramEl.innerText = ramPct + '%'; ramEl.className = _toneClass(ramPct, 'primary', { invert: true }); _toneCelebrate(ramEl, ramPct); }
            $('r-v').innerText = d.stats.ram_u + ' / ' + Math.round(d.stats.ram_t) + ' GB';

            gH.push(d.stats.gpu); vH.push(ramPct);
            if (gH.length > 15) gH.shift();
            if (vH.length > 15) vH.shift();
            // Use DaisyUI bg-* utility classes so the ticker tints match the
            // active theme (and switch when the user changes themes). For
            // PRESSURE metrics (RAM, Disk, Load) high% = bad → >80% flips to
            // bg-error to flag pressure regardless of base tone. GPU is
            // INVERTED — high GPU% means "the fleet is doing useful work",
            // which is the desired state for this app — so a *low* GPU% is
            // the alarming case (idle / starved). Pass `invert: true` to flip
            // the threshold so 0–20% renders bg-error instead of 80–100%.
            // (Disk / RAM / Load keep the default "high = bad" semantics
            // because they genuinely indicate resource pressure.)
            // Each column's TINT scales with its value: full primary (or
            // tone) at 100 %, fading toward 30 % opacity at 0 %. Above the
            // alarm threshold (>80 % normal, <20 % when inverted) we flip
            // to bg-error at full opacity so pressure POPS regardless of
            // the surrounding gradient. Net: a row of low-value columns
            // looks subdued, a row at 100 % looks fully saturated, and an
            // alarm column stands out as a distinct red against either.
            const _tickerHTML = (vals, tone, opts) => {
                const invert = !!(opts && opts.invert);
                return vals.map(v => {
                    const isAlarming = invert ? (v < 20) : (v > 80);
                    const cls = isAlarming ? 'bg-error' : ('bg-' + tone);
                    // Map 0..100 → 0.30..1.0 opacity for the non-alarm tone.
                    // Alarm columns stay at full opacity (clarity > consistency).
                    const opacity = isAlarming ? 1 : Math.max(0.3, Math.min(1, 0.3 + (v / 100) * 0.7));
                    return `<div class="ticker-col ${cls}" style="height:${Math.max(5, (v / 100) * 30)}px;opacity:${opacity.toFixed(2)}"></div>`;
                }).join('');
            };
            $('g-t').innerHTML = _tickerHTML(gH, 'primary', { invert: true });
            $('v-t').innerHTML = _tickerHTML(vH, 'primary', { invert: true });

            // Load average (1m) — same INVERTED tone pattern as GPU/RAM:
            // primary normally, text-error when low (< 20 %) since a
            // sleeping CPU during an AI session usually means the workers
            // have stalled rather than "great, plenty of headroom".
            const loadPct = (typeof d.stats.load_pct === 'number') ? d.stats.load_pct : 0;
            const loadEl = $('l-v');
            if (loadEl) { loadEl.innerText = loadPct + '%'; loadEl.className = _toneClass(loadPct, 'primary', { invert: true }); _toneCelebrate(loadEl, loadPct); }
            const loadParent = loadEl && loadEl.parentElement;
            if (loadParent && d.stats.load_1m != null) {
                loadParent.title = `1m: ${d.stats.load_1m.toFixed(2)} · 5m: ${d.stats.load_5m.toFixed(2)} · 15m: ${d.stats.load_15m.toFixed(2)} (load average / cpu count)`;
            }
            lH.push(loadPct);
            if (lH.length > 15) lH.shift();
            const lt = $('l-t');
            if (lt) lt.innerHTML = _tickerHTML(lH, 'primary', { invert: true });

            // Disk ticker: usage barely moves, so sample at 1/120 the rate of
            // GPU/RAM. With WS broadcasting every 2 s that's one new column
            // every ~4 minutes; 15 columns ≈ 1 hour of history (matches the
            // "Disk (1hr)" label).
            _diskTickCounter = (_diskTickCounter || 0) + 1;
            if (d.outputs_disk && (_diskTickCounter % 120 === 1 || dH.length === 0)) {
                // Prefill the rolling buffer with the current value the first
                // time we see one, so the ticker shows a flat line at the
                // current usage instead of a single tiny column slowly growing.
                if (dH.length === 0) dH = Array(15).fill(d.outputs_disk.pct);
                else dH.push(d.outputs_disk.pct);
                if (dH.length > 15) dH.shift();
                const dt = $('d-t');
                // Disk is the ONE ticker that keeps non-inverted (pressure)
                // semantics — unlike GPU/RAM/Load, a full disk is a real
                // problem (no headroom for outputs, /disk/guard kicks in
                // and blocks the queue). High% = bad here.
                if (dt) dt.innerHTML = _tickerHTML(dH, 'primary');
            }

            $('h-m').innerText = d.state.mode;
            $('h-pr').innerText = '"' + d.state.current_prompt + '"';
            // Stage / job start timestamps come from the BACKEND now —
            // server-side broadcast loop tracks step + video_index transitions
            // and stamps `state.stage_started_ts` / `state.job_started_ts`.
            // We also accumulate per-job per-stage actuals so completed
            // stages can stick around as a "Stage · 12s (ETA 24s)" history
            // line in the queue panel.
            if (d.state.step !== _lastStage) {
                if (_lastStage && _stageStartTs) {
                    const dur = (Date.now() - _stageStartTs) / 1000;
                    _saveStageDuration(_lastStage, dur);
                    _renderStageEtas();
                }
                _lastStage = d.state.step;
            }
            if (d.state.video_index !== _lastJobIndex) {
                _lastJobIndex = d.state.video_index;
            }
            // Hydrate _jobActuals from backend so refresh doesn't lose the
            // Text/Image/Video timing rows. Backend tracks stage transitions
            // in state.stage_actuals = { stage_name: { duration_s, ended_ts } }
            // for the current job; we re-shape it into our lookup format.
            const v = d.state.video_index;
            if (v && d.state.stage_actuals) {
                _jobActuals[v] = _jobActuals[v] || {};
                for (const [stage, info] of Object.entries(d.state.stage_actuals)) {
                    _jobActuals[v][stage] = {
                        duration_s: info.duration_s,
                        eta_s: _stageAvgSeconds(stage),
                    };
                }
            }
            // Convert backend ts (seconds since epoch) to JS Date.now() base.
            const stageTs = d.state.stage_started_ts;
            const jobTs = d.state.job_started_ts;
            _stageStartTs = stageTs ? stageTs * 1000 : Date.now();
            _jobStartTs = jobTs ? jobTs * 1000 : Date.now();
            const stageEl = $('h-c');
            if (stageEl && _stageStartTs) stageEl.innerHTML = '⏱ ' + _fmtElapsedHtml(Date.now() - _stageStartTs);
            const totalEl = $('h-c-total');
            if (totalEl && _jobStartTs) totalEl.innerHTML = 'Σ ' + _fmtElapsedHtml(Date.now() - _jobStartTs);
            // Progress bar tracks subject-through-list as a rough lifetime indicator.
            $('h-p').value = d.state.total_videos
                ? (d.state.video_index / Math.max(1, d.state.total_videos)) * 100
                : 0;
            updateStageSteps(d.state);

            document.querySelectorAll('.wf-step').forEach(s => s.classList.remove('active', 'done'));
            const steps = ['Concept', 'Base-Image', 'Video-Chains', 'Audio-Music', 'Post-Process', 'Final-Merge'];
            let hit = false;
            const cur = d.state.step ? d.state.step.replace(/ /g, '-') : '';
            steps.forEach(s => {
                const el = $('s-' + s);
                if (el) {
                    if (s === cur) { el.classList.add('active'); hit = true; }
                    else if (!hit) el.classList.add('done');
                }
            });

            const isRunning = d.state && d.state.mode && d.state.mode !== 'Idle';
            _isRendering = isRunning;
            _updateStartBtn();
            // Pass mode AND step so the action verb mapping ("Imaging" / "Videoing" /
            // …) doesn't depend on _lastTick which is set later in this handler.
            _updateConnPill(isRunning, d.state && d.state.mode, d.state && d.state.step);
            const qLen = d.queue.length + (isRunning ? 1 : 0);
            $('q-count').innerText = qLen;
            _refreshFailedActionsVisibility(d.queue);
            const qList = $('q-list');
            const cfg = d.config || {};
            const llmModelId = (cfg.llm && cfg.llm.model_id) || '';
            // Tone+label rows live at module scope as _STAGES_META so the
            // top-of-card bar (_buildActiveJobProgressBar) can read the same
            // table. See the const definition near _STAGE_ORDER.
            const STAGES = _STAGES_META;
            // Build the unified Video Chains collapsible. Summary line
            // displays "video parts · <count> · <slug>" + a video preview of
            // chain 1; expanded body interleaves chain mp4 rows and bridge
            // png rows for parts 1..c, all using the same 48×27 thumbnail
            // style (video uses preload="metadata" so the browser shows the
            // first frame as a poster). Text is whitespace-nowrap so
            // narrow viewports don't break "extract last frame N →" across
            // two lines.
            const _buildVideoChainCollapsible = (v, c, q, modelLabel, timingHtml, isStageActive, roundedMs) => {
                const cached = _assetsByVidx.get(v) || {};
                const bridges = cached.bridges || {};
                const cfgSnap = (q && q.config_snapshot) || (_lastTick && _lastTick.config) || {};
                // Best-effort slug recovery from the running prompt, so the
                // summary reads "v3_<slug> · 4 parts" instead of just "v3".
                const _slug = cached.slug || '';
                const stem = _slug ? `slop_${v}_${_slug}` : `slop_${v}`;
                const c1Name = `${stem}_c1.mp4`;
                const c1Href = `/files/${encodeURIComponent(c1Name)}`;
                const thumbCls = "rounded bg-black object-cover flex-none";
                const thumbStyle = "width:48px;height:27px;";
                // Per-part rows: each chain mp4 is rendered as
                //   "<video model> · part i"  + thumbnail.
                // The currently-being-rendered part (only when stage is the
                // active one) gets a loading spinner inside the chip — same
                // visual contract as the active model badge in the summary.
                // c here = chain_index from state (count of FINISHED chains
                // for the active stage; total chains for completed stages).
                // For active: the IN-FLIGHT part is c+1 (the one being
                // rendered right now, frames landing on disk); we cap at
                // total_chains so we don't run past the end.
                const partRows = [];
                const _activeChainIdx = isStageActive
                    ? ((_lastTick && _lastTick.state && _lastTick.state.chain_index) || 0) + 1
                    : 0;
                const _totalChains = isStageActive
                    ? ((_lastTick && _lastTick.state && _lastTick.state.total_chains) || c)
                    : c;
                const _maxPart = Math.max(c, _activeChainIdx);
                const _partLabel = modelLabel || 'Video';
                for (let i = 1; i <= _maxPart; i++) {
                    const chainName = `${stem}_c${i}.mp4`;
                    const chainHref = `/files/${encodeURIComponent(chainName)}`;
                    const isActivePart = isStageActive && i === _activeChainIdx;
                    const _spinner = isActivePart
                        ? '<span class="loading loading-spinner loading-xs mr-1"></span>'
                        : '';
                    const _badgeTone = isActivePart ? 'badge-warning' : 'badge-success';
                    // Poster picks the on-disk PNG that's visually closest to
                    // chain i's first frame: bridge i-1 (the extracted last
                    // frame of the previous chain, which became the input
                    // image for this one) for i>1, or the base image for i=1.
                    // This guarantees a thumbnail even when the browser
                    // wouldn't auto-render a poster from preload="metadata".
                    const _posterName = i === 1
                        ? `${stem}_base.png`
                        : (bridges[i - 1] || `${stem}_f${i - 1}.png`);
                    const _posterAttr = `poster="/files/${encodeURIComponent(_posterName)}"`;
                    // Per-part time chip — we don't track each chain's
                    // duration separately on the server side, so we
                    // approximate by splitting the total Video Chains
                    // duration evenly across N parts. Marked with a tilde
                    // ("~3m") so the user knows it's an estimate, not
                    // measured. Bridge rows get an em-dash since ffmpeg
                    // extract is sub-second.
                    const _perPartMs = (roundedMs && _totalChains > 0) ? (roundedMs / _totalChains) : null;
                    const _perPartChip = _perPartMs
                        ? `<span class="flex-none w-12 text-right font-mono text-[9px] text-base-content/60" title="estimated (total chain time ÷ part count)">~${_fmtRoundUp(_perPartMs)}</span>`
                        : `<span class="flex-none w-12" aria-hidden="true"></span>`;
                    partRows.push(`<div class="flex items-center gap-2 mt-1 text-[9px] font-mono ${isActivePart ? '' : 'opacity-80'} pl-4 border-l border-base-300/50 ml-1" data-chain-row="${v}:${i}">
                        <span class="min-w-0 flex items-center gap-2 overflow-hidden fade-edges-r [&>a]:truncate [&>a]:min-w-0">
                            <a href="${chainHref}" target="_blank" rel="noopener" class="inline-flex items-center gap-1 min-w-0">
                                <video src="${chainHref}" ${_posterAttr} class="${thumbCls}" style="${thumbStyle}" preload="metadata" muted playsinline data-anim-thumb onerror="this.style.display='none'"></video>
                                <span class="truncate">${_htmlEscape(chainName)}</span>
                            </a>
                        </span>
                        <span class="flex-none text-right ml-auto">
                            <span class="badge badge-xs ${_badgeTone}">${_spinner}${_htmlEscape(_partLabel)} · part ${i}</span>
                        </span>
                        ${_perPartChip}
                    </div>`);
                    if (i < _maxPart) {
                        // Bridge i = extracted last frame of chain i, fed as
                        // input to chain i+1. Render it whenever there's a
                        // chain i+1 in this view (so completed chains always
                        // show their bridge, including the one feeding the
                        // currently-active chain). Previously gated on i<c
                        // which dropped the bridge between the last finished
                        // chain and the in-flight one.
                        const bridgeName = bridges[i] || `${stem}_f${i}.png`;
                        const bridgeHref = `/files/${encodeURIComponent(bridgeName)}`;
                        partRows.push(`<div class="flex items-center gap-2 mt-1 text-[9px] font-mono opacity-60 pl-4 border-l border-base-300/50 ml-1" data-ffmpeg-bridge="${v}:${i}">
                            <span class="min-w-0 flex items-center gap-2 overflow-hidden fade-edges-r [&>a]:truncate [&>a]:min-w-0">
                                <a href="${bridgeHref}" target="_blank" rel="noopener" class="inline-flex items-center gap-1 min-w-0">
                                    <img src="${bridgeHref}" class="${thumbCls}" style="${thumbStyle}" loading="lazy" onerror="this.style.display='none'">
                                    <span class="truncate">${_htmlEscape(bridgeName)}</span>
                                </a>
                            </span>
                            <span class="flex-none text-right ml-auto">
                                <span class="badge badge-xs badge-warning opacity-80" title="ffmpeg extracts the last frame of chain ${i} as the input image for chain ${i + 1}">✓ ffmpeg · bridge ${i}</span>
                            </span>
                            <span class="flex-none w-12 text-right font-mono text-[9px] text-base-content/60" title="ffmpeg frame extract is sub-second">&lt;1s</span>
                        </div>`);
                    }
                }
                // Summary uses the runner's video model id (e.g. "LTX-2.3 Video")
                // rather than the generic word "chain", because the user already
                // knows it's a chain — the model identity is what's actually
                // useful at a glance. Timing/ETA renders on the right so the
                // chain step shows progress inline even when it's the active
                // (in-flight) stage and there's no completed-row counterpart.
                const _label = modelLabel || 'Video';
                // Layout matches the standard per-stage output row:
                //   [disclosure + thumb + filename (flex-1)] [model badge (right)] [time chip]
                // The disclosure arrow sits inside the left cluster so the
                // affordance reads as "click row to expand", same idiom
                // other rows use for the copy button glyph slot.
                const _roundedTime = (roundedMs != null) ? _fmtRoundUp(roundedMs) : '';
                const _timeTitle = (timingHtml || '').replace(/<[^>]+>/g, '');
                const _vcTimeChip = _roundedTime
                    ? `<span class="flex-none w-12 text-right font-mono text-[9px] text-base-content/60" title="${_htmlEscape(_timeTitle)}">${_roundedTime}</span>`
                    : `<span class="flex-none w-12" aria-hidden="true"></span>`;
                const _vcOpen = _openVideoChains.has(String(v)) ? ' open' : '';
                // No single playable mp4 exists for the chain stage —
                // only after Final Merge concatenates the parts. Until
                // then the summary's left cluster reads "expand" /
                // "collapse" (an affordance label, no filename). Once
                // FINAL_<v>.mp4 is on disk, surface its filename + thumb.
                const _final = cached.final || '';
                let _vcLeftCluster;
                if (_final) {
                    const _finalHref = `/files/${encodeURIComponent(_final)}`;
                    _vcLeftCluster = `<span class="video-chain-arrow inline-block transition-transform flex-none">▸</span>
                            <a href="${_finalHref}" target="_blank" rel="noopener" class="inline-flex items-center gap-1 min-w-0 truncate" onclick="event.stopPropagation()">
                                <video src="${_finalHref}" class="${thumbCls}" style="${thumbStyle}" preload="metadata" muted data-anim-thumb onerror="this.style.display='none'"></video>
                                <span class="truncate">${_htmlEscape(_final)}</span>
                            </a>`;
                } else {
                    _vcLeftCluster = `<span class="video-chain-arrow inline-block transition-transform flex-none">▸</span>
                            <span class="opacity-70 italic vc-toggle-label" aria-hidden="true"></span>`;
                }
                return `<details class="mt-1 video-chain-details" data-video-chain="${v}"${_vcOpen}>
                    <summary class="cursor-pointer list-none flex items-center gap-2 text-[9px] font-mono">
                        <span class="min-w-0 flex items-center gap-2 overflow-hidden fade-edges-r">
                            ${_vcLeftCluster}
                        </span>
                        <span class="flex-none text-right ml-auto">
                            <span class="badge badge-xs badge-success opacity-70">✓ ${_htmlEscape(_label)} · ${_totalChains}p</span>
                        </span>
                        ${_vcTimeChip}
                    </summary>
                    ${partRows.join('')}
                </details>`;
            };

            const renderPipelineStrip = (q, opts) => {
                const isActive = !!(opts && opts.running);
                const curStep = isActive ? (opts.step || '') : null;
                const v = isActive ? ((_lastTick && _lastTick.state && _lastTick.state.video_index) || 1) : 0;
                const c = isActive ? ((_lastTick && _lastTick.state && _lastTick.state.chain_index) || 0) : 0;
                if (!isActive) return '';
                // Per-item strip now ONLY emits the completed-stages history
                // ("Output" block) + active-stage bridge thumbnails. The
                // segmented progress bar was hoisted to the top of the queue
                // card — see _buildActiveJobProgressBar() — so the user reads
                // ONE bar regardless of which queue row is expanded.
                // Each completed stage of THIS job becomes a single line:
                //   [asset-link badge]  ⏱ actual / ETA was-eta
                // Stage's clickable badge replaces the spinner+text it had
                // while running. Active stage gets a fresh in-progress line
                // below it. The user can read top-to-bottom = full history.
                const actuals = (_jobActuals[v] || {});
                // Track stages that are *first* appearing as completed in
                // this render — they get .stage-just-completed for the
                // cross-fade animation. Pre-existing completions render plain
                // so we don't restart the animation on every WS tick.
                const justCompleted = new Set();
                const completedLines = STAGES
                    .filter(([s]) => _stageDoneBefore(curStep, s))
                    .map(([s, , label, , tone]) => {
                        const key = `${v}:${s}`;
                        const isFresh = !_displayedDoneStages.has(key);
                        if (isFresh) {
                            justCompleted.add(key);
                            _displayedDoneStages.add(key);
                        }
                        let assetBadge;
                        // Per-stage prompt badges — one per stage that has a
                        // generated prompt, styled the same as file-asset badges.
                        // Click → showPromptPeek modal with the full text.
                        const _PROMPT_FIELD_BY_STAGE = {
                            'Concept': null, // uses state.current_prompt below
                            'Base Image': 'image_prompt',
                            'Video Chains': 'video_prompt',
                            'Audio': 'music_prompt',
                            'TTS': 'tts_prompt',
                        };
                        const cfgSnap = (q && q.config_snapshot) || (_lastTick && _lastTick.config) || {};
                        let promptForStage = '';
                        if (s === 'Concept') {
                            promptForStage = (_lastTick && _lastTick.state && _lastTick.state.current_prompt) || '';
                        } else if (_PROMPT_FIELD_BY_STAGE[s]) {
                            promptForStage = cfgSnap[_PROMPT_FIELD_BY_STAGE[s]] || '';
                        }
                        // Stash the prompt text in a data attribute to avoid
                        // attribute-quoting nightmares when the prompt has '
                        // or " in it (which would break onclick parsing).
                        // A delegated click handler (wired once at startup)
                        // reads the data-prompt-text on click.
                        // Prompt button now opens the per-stage prompts
                        // editor (used to open a read-only peek). The model
                        // badge on the right takes over the "show what was
                        // sent" role via openModelSettingsPopup, so the two
                        // buttons effectively swap responsibilities — click
                        // 📝 to edit prompts, click the model to see the
                        // settings/snapshot for that stage.
                        const _promptEditTarget = (s === 'Concept') ? 'Base Image' : s;
                        const promptBadge = promptForStage
                            ? `<button type="button" class="badge badge-xs badge-primary cursor-pointer font-mono text-[9px]" title="${_stageDisplayName(s)} prompts — click to view + edit" onclick='event.stopPropagation(); openPromptsEdit(${JSON.stringify(_promptEditTarget)})'>📝 prompts →</button>`
                            : '';
                        if (s === 'Concept') {
                            assetBadge = promptBadge;
                        } else {
                            const asset = _STAGE_ASSET(s, v, c);
                            // Filename = plain text link, NOT a badge. The
                            // badge formatting was making the asset row read
                            // as another model chip — but the filename is
                            // just an output reference. Click opens the file
                            // in a new tab; click the thumbnail (below) for
                            // the asset-info modal.
                            const _isVid = asset && /\.mp4$/i.test(asset);
                            const _isImg = asset && /\.(png|jpe?g|webp)$/i.test(asset);
                            const _href = asset ? `/files/${encodeURIComponent(asset)}` : '';
                            const _thumbCls = "rounded bg-black object-cover flex-none";
                            const _thumbStyle = "width:32px;height:18px;";
                            const _thumb = asset && _isVid
                                ? `<button type="button" class="flex-none" onclick='event.stopPropagation(); openAssetInfo(${JSON.stringify(asset)})' title="Open ${asset} in info modal"><video class="${_thumbCls}" style="${_thumbStyle}" preload="metadata" muted playsinline data-anim-thumb onerror="this.style.display='none'"><source data-src="${_href}"></video></button>`
                                : asset && _isImg
                                    ? `<button type="button" class="flex-none" onclick='event.stopPropagation(); openAssetInfo(${JSON.stringify(asset)})' title="Open ${asset} in info modal"><img data-src="${_href}" class="${_thumbCls}" style="${_thumbStyle}" loading="lazy" onerror="this.style.display='none'"></button>`
                                    : '';
                            // Tiny copy-button glyph that sits next to the
                            // filename. Reads its target from data-copy-text
                            // so quotes inside the path don't break the
                            // attribute. Briefly tints success on copy.
                            const _copyBtn = asset
                                ? `<button type="button" class="flex-none w-4 h-4 inline-flex items-center justify-center opacity-50 hover:opacity-100 cursor-pointer" title="Copy ${_htmlEscape(asset)}" aria-label="Copy filename" data-copy-text="${_htmlEscape(asset)}" onclick="event.stopPropagation(); navigator.clipboard.writeText(this.dataset.copyText || '').then(()=>{this.classList.add('text-success');setTimeout(()=>this.classList.remove('text-success'),900);}).catch(()=>{});"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-3 h-3"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>`
                                : '';
                            const fileBadge = asset
                                ? `${_thumb}<a href="${_href}" target="_blank" rel="noopener" class="link link-hover font-mono text-[9px] truncate min-w-0" title="Open ${asset} in a new tab" onclick="event.stopPropagation()">${asset}</a>${_copyBtn}`
                                : '';
                            // Stage shows BOTH the prompt badge (if a per-stage
                            // prompt exists from /enhance/distribute) AND the
                            // file asset link + thumbnail.
                            assetBadge = [promptBadge, fileBadge].filter(Boolean).join(' ');
                        }
                        const a = actuals[s];
                        // Timing chip is wrapped in a flex-none, fixed-width
                        // right-aligned container so the elapsed/ETA columns
                        // line up vertically across every per-stage row in the
                        // expanded queue item — each row reads as
                        //   "✓ Image    v3_base.png    [   3m22s / ETA 9m  ]"
                        // with the bracketed part forming a clean column.
                        // 2× overrun = bold + error tint on the elapsed time
                        // so the user spots stages that went WAY past their
                        // rolling-average ETA. Threshold is a multiplier on
                        // eta_s; only kicks in when eta_s is known and
                        // duration_s > 2 × eta_s.
                        const _overrun2x = a && a.eta_s && a.duration_s > 2 * a.eta_s;
                        const _elapsedCls = _overrun2x
                            ? "font-mono text-[9px] font-bold text-error"
                            : "font-mono text-[9px]";
                        const timing = a
                            ? `<span class="${_elapsedCls}">${_fmtElapsedHtml(a.duration_s * 1000)}</span><span class="opacity-50 text-[9px]">${a.eta_s ? ' / ETA ' + _fmtElapsedHtml(a.eta_s * 1000) : ''}</span>`
                            : '';
                        // Per-stage meta string — what the user wants to see in
                        // the right-edge cluster alongside the model badge.
                        // Image stages → resolution; Video parts → duration in
                        // seconds (frames/24); Audio + TTS → configured length;
                        // Final Merge → total length (chains × frames / 24).
                        const _aspectToRes = { '1:1': '1024×1024', '4:3': '1152×864', '3:4': '864×1152', '16:9': '1280×720', '9:16': '720×1280' };
                        const _frames = Number(cfgSnap.frames) || 0;
                        const _chains = Number(cfgSnap.chains) || 0;
                        let _meta = '';
                        if (s === 'Base Image') _meta = _aspectToRes[cfgSnap.size] || cfgSnap.size || '';
                        else if (s === 'Video Chains' && _frames) _meta = `${(_frames / 24).toFixed(1)}s × ${_chains}`;
                        else if ((s === 'Audio' || s === 'TTS') && cfgSnap.audio_duration_s) _meta = `${Math.round(cfgSnap.audio_duration_s)}s`;
                        else if (s === 'Final Merge' && _frames && _chains) _meta = `${((_frames * _chains) / 24).toFixed(1)}s`;
                        const _metaHtml = _meta
                            ? `<span class="text-base-content/60 font-mono text-[9px] flex-none">${_htmlEscape(_meta)}</span>`
                            : '';
                        const timingCol = `<span class="flex-none w-32 text-right font-mono">${timing}</span>`;
                        // Stage label on the LEFT, asset link + duration on
                        // the RIGHT (push with ml-auto). Reads as a list:
                        // "✓ Image                        v3_base.png  3m22s / ETA 9m"
                        const animCls = isFresh ? ' stage-just-completed' : '';
                        // The stage label itself doubles as a shortcut to the
                        // prompts editor focused on that stage. For Concept,
                        // it opens the editor at Base Image (Concept itself
                        // isn't editable but Text is the natural entry point
                        // — replaces the standalone `prompts →` button).
                        // Resolve which model the runner used for THIS stage from
                        // the snapshot, so the badge shows e.g. "✓ Qwen Image"
                        // or "✓ LTX-2.3" rather than the generic stage verb.
                        // Falls back to the stage label if no model is recorded.
                        const _STAGE_MODEL_FIELD = {
                            'Concept': null, // uses snap.llm.model_id below
                            'Base Image': 'base_model',
                            'Video Chains': 'video_model',
                            'Audio': 'audio_model',
                            'TTS': 'tts_model',
                            'Post Process': 'upscale_model',
                        };
                        let modelLabel = label;
                        if (s === 'Concept') {
                            const llmId = (cfgSnap.llm && cfgSnap.llm.model_id) || llmModelId || '';
                            if (llmId) modelLabel = llmId.replace(/^.*\//, '').replace(/\.gguf$/i, '');
                        } else {
                            const modelId = cfgSnap[_STAGE_MODEL_FIELD[s] || ''];
                            if (modelId && modelId !== 'none') {
                                const role = (s === 'Audio') ? 'audio' : (s === 'Video Chains' ? 'video' : (s === 'Base Image' ? 'image' : s.toLowerCase()));
                                modelLabel = (typeof _modelDisplayName === 'function')
                                    ? (_modelDisplayName(modelId, role) || modelId)
                                    : modelId;
                            }
                        }
                        // Stage → role for openModelSettingsPopup. Final
                        // Merge has no model settings (it's an ffmpeg
                        // consolidation), so it stays a passive span.
                        const _STAGE_SETTINGS_ROLE = {
                            'Concept': 'llm',
                            'Base Image': 'base',
                            'Video Chains': 'video',
                            'Audio': 'audio',
                            'TTS': 'tts',
                            'Post Process': 'upscale',
                        };
                        // ✓ if this stage actually produced output, ✗ if it
                        // was skipped (model = none, or no asset on disk for
                        // the asset-producing stages). Concept is always ✓
                        // because the prompt itself is the artifact.
                        let _stageHasOutput;
                        if (s === 'Concept') {
                            _stageHasOutput = true;
                        } else if (s === 'Final Merge') {
                            _stageHasOutput = !!_STAGE_ASSET(s, v, c);
                        } else {
                            const _fld = _STAGE_MODEL_FIELD[s];
                            const _mid = _fld ? cfgSnap[_fld] : null;
                            _stageHasOutput = !!(_mid && _mid !== 'none');
                        }
                        const _stageGlyph = _stageHasOutput ? '✓' : '✗';
                        const _stageBadgeCls = _stageHasOutput
                            ? `badge badge-xs badge-${tone} opacity-70 cursor-pointer`
                            : 'badge badge-xs badge-ghost opacity-50 cursor-pointer';
                        const _settingsRole = _STAGE_SETTINGS_ROLE[s];
                        // CPU badge — shows next to the model glyph when
                        // the active scheduler config has CPU offload on
                        // for this stage. _lastTick.config.scheduler is
                        // the live source of truth (defaults to ON when
                        // unset, matching the Settings UI promise).
                        let _cpuBadge = '';
                        const _sched = (_lastTick && _lastTick.config && _lastTick.config.scheduler) || {};
                        // Resolve cpu_mode: "cpu" always shows badge, "smart" shows "⚡"
                        // badge (resolved dynamically), "gpu" shows nothing.
                        const _cpuMode = (s === 'Concept')
                            ? (_sched.llm_cpu_mode || (_sched.llm_cpu_only === false ? 'gpu' : _sched.llm_cpu_only === true ? 'cpu' : 'smart'))
                            : (s === 'TTS')
                                ? (_sched.tts_cpu_mode || (_sched.tts_cpu_only === false ? 'gpu' : _sched.tts_cpu_only === true ? 'cpu' : 'smart'))
                                : null;
                        if (_cpuMode === 'cpu') {
                            _cpuBadge = `<span class="inline-flex items-center align-middle mr-0.5 opacity-70" title="Running on CPU (forced)"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-3 h-3"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="2" x2="9" y2="4"/><line x1="15" y1="2" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="22"/><line x1="15" y1="20" x2="15" y2="22"/><line x1="20" y1="9" x2="22" y2="9"/><line x1="20" y1="15" x2="22" y2="15"/><line x1="2" y1="9" x2="4" y2="9"/><line x1="2" y1="15" x2="4" y2="15"/></svg></span>`;
                        } else if (_cpuMode === 'smart') {
                            _cpuBadge = `<span class="inline-flex items-center align-middle mr-0.5 opacity-70" title="Smart mode: GPU if idle, else CPU">⚡</span>`;
                        }
                        const stageLabelHtml = _settingsRole
                            ? `<button type="button" class="${_stageBadgeCls}" title="${_stageDisplayName(s)} — ${modelLabel}. Click for settings" onclick='event.stopPropagation(); openModelSettingsPopup(${JSON.stringify(_settingsRole)}, ${q.ts || 0})'>${_stageGlyph} ${_cpuBadge}${_htmlEscape(modelLabel)}</button>`
                            : `<span class="badge badge-xs badge-${tone} opacity-70" title="${_stageDisplayName(s)} — ${modelLabel}">${_stageGlyph} ${_cpuBadge}${_htmlEscape(modelLabel)}</span>`;
                        // Right-edge cluster mirrors the summary row's
                        //   [model-badge] [menu]
                        // by sitting the model in the same min-w-[7rem]
                        // column and a coarse rounded-up duration chip in
                        // the menu's slot (w-7). 2× overrun keeps its bold
                        // error tint so wildly-overshooting stages still
                        // jump out at a glance.
                        // Every completed-stage row renders SOMETHING in the
                        // time column — actual duration when available, '—'
                        // otherwise. The placeholder makes LLM/TTS rows
                        // visually consistent with Image/Video rows even
                        // when the orchestrator's stage_actuals didn't
                        // capture a transition (rare; happens when a stage
                        // completes faster than the 1 Hz heartbeat).
                        const _roundedTime = a ? _fmtRoundUp(a.duration_s * 1000) : '';
                        const _timeCls = _overrun2x
                            ? "flex-none w-12 text-right font-mono text-[9px] font-bold text-error"
                            : "flex-none w-12 text-right font-mono text-[9px] text-base-content/60";
                        const timeChip = _roundedTime
                            ? `<span class="${_timeCls}" title="${_htmlEscape(_fmtElapsed(a.duration_s * 1000))}${a.eta_s ? ' / ETA ' + _fmtElapsed(a.eta_s * 1000) : ''}">${_roundedTime}</span>`
                            : `<span class="${_timeCls} opacity-40" title="duration not recorded">—</span>`;
                        // min-w-0 on the LEFT cluster + truncate on the
                        // filename link clamps the asset column to whatever
                        // horizontal space remains after model + time claim
                        // their flex-none widths. Without this, a long
                        // filename would push the right-side columns
                        // off-screen rather than truncating itself.
                        // Layout: asset content sits at the left of the row;
                        // model badge + time chip pinned to the right via
                        // ml-auto. Asset uses min-w-0 + truncate so long
                        // filenames clip rather than push the right
                        // cluster off-screen. Earlier `flex-1` on the
                        // asset column + min-w-[7rem] on the badge was
                        // padding ~120px of empty whitespace between
                        // filename and badge — now they sit adjacent.
                        let row = `<div class="flex items-center gap-2 mt-1${animCls}" data-stage-row="${key}">
                            <span class="min-w-0 flex items-center gap-2 overflow-hidden fade-edges-r [&>a]:truncate [&>a]:min-w-0">${assetBadge}</span>
                            <span class="flex-none text-right ml-auto">${stageLabelHtml}</span>
                            ${timeChip}
                        </div>`;
                        // Video Chains stage emits a single collapsible whose
                        // SUMMARY shows "<N> parts · <slug>" with the first
                        // chain's preview, and whose body interleaves chain
                        // mp4 rows + bridge png rows for parts 1..c. The
                        // overarching collapsible title represents the entire
                        // video-generation chain for this iter.
                        if (s === 'Video Chains' && v && c > 0) {
                            row += _buildVideoChainCollapsible(v, c, q, modelLabel, timing, /* isStageActive */ false, a ? a.duration_s * 1000 : null);
                        }
                        return row;
                    }).join('');
                // Active Video Chains: same collapsible component, mid-flight.
                // Resolve modelLabel + a live timing chip so the user can see
                // chain progress without an expanded reveal.
                let activeBridgesHtml = '';
                if (curStep === 'Video Chains' && v && c > 0) {
                    const cfgSnap = (q && q.config_snapshot) || (_lastTick && _lastTick.config) || {};
                    const _vid = cfgSnap.video_model;
                    let _activeLabel = (_vid && _vid !== 'none' && typeof _modelDisplayName === 'function')
                        ? (_modelDisplayName(_vid, 'video') || _vid)
                        : 'Video';
                    // Annotate with the active chain-handoff strategy so the user
                    // can see whether multi-frame keyframing is in play. FLF2V
                    // mode (per-chain seeds) wins over plain handoff K.
                    if (q && q.seeds_mode === 'per-chain' && Array.isArray(q.seed_images) && q.seed_images.length >= 2) {
                        _activeLabel += ` · FLF2V (${q.seed_images.length} kf)`;
                    } else if (c > 1) {
                        const _k = Math.max(1, Math.min(8, parseInt(cfgSnap.chain_handoff_keyframes ?? 4, 10) || 4));
                        if (_k > 1) _activeLabel += ` · K=${_k}`;
                    }
                    const _stageElapsed = _stageStartTs ? (Date.now() - _stageStartTs) : 0;
                    const _stageEta = _stageAvgSeconds('Video Chains') || 0;
                    // Mid-flight 2× overrun → bold + error so the user spots
                    // a chain stage that's pushed way past its rolling avg.
                    const _liveOverrun2x = _stageEta && _stageElapsed > 2 * _stageEta * 1000;
                    const _liveElapsedCls = _liveOverrun2x
                        ? "font-mono text-[9px] font-bold text-error"
                        : "font-mono text-[9px]";
                    const _activeTiming = `<span class="${_liveElapsedCls}">${_fmtElapsedHtml(_stageElapsed)}</span>${_stageEta ? `<span class="opacity-50 text-[9px]"> / ETA ${_fmtElapsedHtml(_stageEta * 1000)}</span>` : ''}`;
                    activeBridgesHtml = _buildVideoChainCollapsible(v, c, q, _activeLabel, _activeTiming, /* isStageActive */ true, _stageElapsed);
                }
                // Strip the .stage-just-completed class after the animation
                // finishes (600 ms total budget) so the next WS re-render
                // doesn't accidentally restart it via DOM diffing quirks.
                if (justCompleted.size) {
                    setTimeout(() => {
                        justCompleted.forEach(key => {
                            document.querySelectorAll(`[data-stage-row="${key}"]`).forEach(el => {
                                el.classList.remove('stage-just-completed');
                            });
                        });
                    }, 600);
                }
                // No more per-item segmented bar — see _buildActiveJobProgressBar()
                // which renders ONE bar at the top of the queue card. This
                // function now returns the completed-stages history block only.
                const hasOutput = !!(completedLines || activeBridgesHtml);
                return hasOutput
                    ? `${completedLines}${activeBridgesHtml}`
                    : '';
            };
            const renderItem = (q, opts) => {
                const snap = (q && q.config_snapshot) || cfg;
                const activeRole = (opts && opts.running && opts.step) ? _STAGE_ROLE[opts.step] : null;
                const badges = _configModelBadges(snap, llmModelId, activeRole, q.ts || 0);
                const meta = `<span class="opacity-50">size</span> ${_htmlEscape(snap.size || '1:1')} <span class="opacity-30">·</span> ${snap.frames || 17} <span class="opacity-50">frames</span>`;
                const promptEsc = _htmlEscape(q.prompt || '');
                const isActive = !!(opts && opts.running);
                const isCancelled = q.status === 'cancelled';
                const showDateTime = (cfg && cfg.show_date_time) || (snap && snap.show_date_time);
                const tsHtml = (showDateTime && q.ts)
                    ? `<span class="opacity-40 font-mono text-[9px] mr-1">${new Date(q.ts * 1000).toLocaleString(undefined, { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>`
                    : '';
                // ♾ / polymorphic: transparent (icon-only) — they're decorative
                // status hints, not call-to-action chips.
                const infBadge = q.infinity
                    ? `<span class="text-base font-bold leading-none flex-none" title="Infinity — re-queues itself after every completion">♾</span>`
                    : '';
                const polyBadge = q.chaos
                    ? `<span class="text-secondary flex-none" title="Polymorphic — randomized model selections per cycle"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="w-3 h-3"><path d="M16 3h5v5"/><path d="M4 20l16-16"/><path d="M21 16v5h-5"/><path d="M15 15l6 6"/><path d="M4 4l5 5"/></svg></span>`
                    : '';
                const fastBadge = q.fast_track
                    ? `<span class="text-warning flex-none" title="Fast Track — chains=2 frames=17 tier=low, audio/tts/upscale skipped (~3 min/clip)">🏃</span>`
                    : '';
                // Surface "random" / "slopped" model selections as their own
                // status icons in the summary, alongside infinity/polymorphic.
                // Single icon per kind regardless of how many roles use it —
                // the tooltip lists which roles. Reads as a quiet hint that
                // this iteration's choices weren't fully deterministic.
                const _modelFields = ['base_model', 'video_model', 'audio_model', 'tts_model', 'upscale_model'];
                const _randomRoles = _modelFields.filter(f => snap[f] === '__random__');
                const _sloppedRoles = _modelFields.filter(f => {
                    const v = snap[f];
                    return v === '__slopped__' || (typeof v === 'string' && v.startsWith('slopped:'));
                });
                const _roleLabel = { base_model: 'image', video_model: 'video', audio_model: 'music', tts_model: 'voice', upscale_model: 'upscale' };
                const randomBadge = _randomRoles.length
                    ? `<span class="text-warning flex-none" title="Random model selection per cycle for: ${_randomRoles.map(r => _roleLabel[r]).join(', ')}">🎲</span>`
                    : '';
                const sloppedBadge = _sloppedRoles.length
                    ? `<span class="text-info flex-none" title="Slopped (reusing existing asset) for: ${_sloppedRoles.map(r => _roleLabel[r]).join(', ')}">♻︎</span>`
                    : '';
                // Hoist the *active* model badge (the one with the loading
                // spinner) into the summary so collapsing the <details> still
                // reveals what's currently working. Completed stages show
                // their model in the Output block, so the full reveal row of
                // badges is now redundant and dropped below.
                const activeBadge = isActive
                    ? (badges.find(b => b.includes('loading-spinner')) || '')
                    : '';
                // Cancelled items keep their badge (so the strikethrough has a
                // label). Active gets nothing (ring+timers signal it). Pending
                // items also drop the chip — being in the queue says it all.
                const statusChip = isCancelled
                    ? `<span class="badge badge-xs badge-ghost text-[9px]">cancelled</span>`
                    : '';
                // Hamburger dropdown — Cancel today, room to grow tomorrow.
                const infToggleLabel = q.infinity ? '▶ Make Single' : '♾ Make Infinite';
                const promptForJs = JSON.stringify(q.prompt || '');
                // The dropdown lives inside <summary>, so any click inside it
                // would bubble up and toggle the <details> — and the focus
                // shift would close the dropdown before the <a> click fired,
                // making selections appear to "pass through". Stop propagation
                // on the wrapper (and the action <a>s) so clicks land on the
                // intended action instead of the summary toggle.
                const stop = `event.stopPropagation()`;
                // Action labels: word LEFT, icon RIGHT (justify-between).
                // The label is the meaning the user scans for; the icon is
                // the affordance. Right-aligned icons line up vertically
                // across the menu, making it easier to scan symbols at a
                // glance ("there's the cancel ✕"). Was leading-glyph-then-
                // text which wasn't aligning the icons across rows.
                // _ml renders one row consistently — split label/icon via
                // a small helper rather than templating each <li> twice.
                const _mlRow = (label, icon, handler, extraCls) =>
                    `<li><a class="flex items-center justify-between gap-3${extraCls ? ' ' + extraCls : ''}" onclick="${handler}"><span>${label}</span><span class="opacity-70 font-mono text-base">${icon}</span></a></li>`;
                // toggleItemInfinity already builds its own `infToggleLabel` string
                // ("♾ Infinity" / "✖ Disable Infinity") with a leading icon — split it
                // back into icon + label so the same right-align rule applies.
                const _splitInfLbl = infToggleLabel.split(/\s+/, 2);
                const infIcon = _splitInfLbl[0] || '♾';
                const infLabel = _splitInfLbl[1] || 'Infinity';
                const menuHTML = isCancelled
                    ? `<div class="dropdown dropdown-end" onclick="${stop}">
                        <label tabindex="0" class="btn btn-ghost btn-xs btn-square" title="Actions" onclick="${stop}">⋯</label>
                        <ul tabindex="0" class="dropdown-content menu menu-xs p-1 shadow bg-base-300 rounded-box z-10 w-44">
                            ${_mlRow('Re-queue', '↻', `event.stopPropagation();requeueItem(${q.ts || 0})`)}
                        </ul>
                       </div>`
                    : `<div class="dropdown dropdown-end" onclick="${stop}">
                        <label tabindex="0" class="btn btn-ghost btn-xs btn-square" title="Actions" onclick="${stop}">⋯</label>
                        <ul tabindex="0" class="dropdown-content menu menu-xs p-1 shadow bg-base-300 rounded-box z-10 w-44">
                            ${_mlRow('Edit prompt', '✎', `event.stopPropagation();editItem(${q.ts || 0}, ${promptForJs})`)}
                            ${_mlRow(infLabel, infIcon, `event.stopPropagation();toggleItemInfinity(${q.ts || 0})`)}
                            ${_mlRow(q.chaos ? 'Disable Polymorphic' : 'Enable Polymorphic', q.chaos ? '✖' : '⤳', `event.stopPropagation();toggleItemPolymorphic(${q.ts || 0})`)}
                            ${_mlRow('Cancel', '✕', `event.stopPropagation();cancelItem(${q.ts || 0})`, 'text-error')}
                        </ul>
                       </div>`;
                const cls = `bg-base-200 rounded-md${isCancelled ? ' opacity-50 slop-cancelled-fade' : ''}${isActive ? ' ring-2 ring-primary' : ''}`;
                // Strip shows on EVERY item — pending items get all-dimmed
                // dots, active gets the verb on the current node.
                const stripHTML = isCancelled ? '' : renderPipelineStrip(q, opts);
                // Always-visible row + collapsible reveal. Active items get the
                // reveal pre-opened so the user sees stage progress without an
                // extra click. The reveal hosts model badges, size·frames meta,
                // and the live pipeline strip + timers.
                // Explicit chevron — clicking the <summary> toggles the
                // <details> for free, so we just need a visual affordance
                // here. The chevron rotates 90° when [open] via the
                // `details[open] > summary .q-row-chevron` rule in app.css.
                // Uses pointer-events:none on the SVG so the click lands on
                // <summary> (not the chevron) — no extra handler, no risk
                // of double-toggle.
                const chevronHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="q-row-chevron w-3 h-3 flex-none text-base-content/60" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>`;
                return `<li class="${cls}" data-q-ts="${q.ts || 0}" data-q-status="${isCancelled ? 'cancelled' : (isActive ? 'active' : 'pending')}">
                    <details ${_openPendingItems.has(q.ts || 0) ? 'open' : ''}>
                        <summary class="cursor-pointer p-2 flex items-center gap-2 text-xs flex-wrap">
                            ${chevronHTML}
                            <span class="flex items-center gap-1 flex-none">
                                ${tsHtml}${statusChip}${infBadge}${polyBadge}${fastBadge}${randomBadge}${sloppedBadge}
                            </span>
                            <span class="flex-1 min-w-0">
                                <span class="font-semibold truncate${isCancelled ? ' line-through' : ''}" title="${promptEsc}">${promptEsc}</span>
                            </span>
                            <span class="flex items-center gap-2 flex-none ml-auto">
                                ${(() => {
                        const _fpp = snap.frames || 17;
                        const _ch = parseInt(snap.chains, 10) || 1;
                        const _tot = _fpp * _ch;
                        // Dropped the trailing "(N parts)" suffix — every
                        // queued item already has a chain badge in its
                        // expanded reveal, so the count was duplicated noise
                        // in the always-visible meta row. The number we
                        // surface here is the TOTAL frame count (chains
                        // multiplied through) since that's what determines
                        // the clip's final length.
                        const _frames = _ch > 1 ? _tot : _fpp;
                        // "frames" word collapses to bare "f" at compact
                        // widths via the .frames-long / .frames-short pair
                        // — same trick as the stats-navbar tight-mode icons.
                        return `<span class="text-base-content/60 font-mono text-[10px] flex-none" title="aspect ratio · total frames"><span>aspect ${_htmlEscape(snap.size || '1:1')} · ${_frames}</span><span class="frames-long"> frames</span><span class="frames-short">f</span></span>`;
                    })()}
                                <span class="flex-none min-w-[7rem] text-right" style="margin-right:1.5rem">${activeBadge}</span>
                                ${menuHTML}
                            </span>
                        </summary>
                        <div class="px-2 pb-2 pt-0 flex flex-col gap-1 border-t border-base-300/50">
                            ${stripHTML}
                        </div>
                    </details>
                </li>`;
            };
            // Signature-gate the queue rebuild. Every WS tick (~1 Hz) used to
            // call qList.innerHTML = … unconditionally, which detached and
            // re-attached the dropdown ⋯ menus, badges, and thumbnails — the
            // visible "flicker / random highlight" the user reported. We now
            // hash the data the render path consumes and skip the rebuild
            // when nothing relevant has changed. The hash covers per-item
            // fields renderItem reads + the active-stage info that drives
            // the synthesized "running" row, plus done-item completion ts.
            const _nowSecForSig = Date.now() / 1000;
            const _qSigPayload = [
                isRunning,
                d.state && d.state.mode,
                d.state && d.state.step,
                d.state && d.state.video_index,
                d.state && d.state.chain_index,
                d.state && d.state.current_prompt,
                qLen,
                (d.queue || []).map(q => [
                    q.ts || 0,
                    q.status || '',
                    q.prompt || '',
                    q.infinity ? 1 : 0,
                    q.chaos ? 1 : 0,
                    q.fast_track ? 1 : 0,
                    q.completed_ts || 0,
                    q.cancelled_ts || 0,
                    q.v_idx || 0,
                    // Cancelled items fade out after 5s; flip a bit when the
                    // window expires so the gated rebuild fires once and
                    // drops the row from the inline strip.
                    (q.status === 'cancelled' && (_nowSecForSig - (q.cancelled_ts || 0) < 5)) ? 1 : 0,
                ]),
            ];
            const _qSig = JSON.stringify(_qSigPayload);
            const _queueChanged = _qSig !== _lastQueueSig;
            if (_queueChanged) _lastQueueSig = _qSig;
            if (qList && _queueChanged) {
                if (!qLen) {
                    qList.innerHTML = '<li class="text-[10px] text-base-content/40 italic p-2">queue empty — click Generate to add</li>';
                } else {
                    const items = [];
                    if (isRunning) {
                        // While the LLM is still rewriting the concept (state.current_prompt
                        // empty), surface the source subject the runner picked instead
                        // of an opaque "(running)". For infinity mode that's the indexed
                        // theme; otherwise the first /n line of #p-core.
                        let displayPrompt = d.state.current_prompt;
                        if (!displayPrompt) {
                            const themes = (cfg && cfg.infinity_themes) || [];
                            const idx = (cfg && cfg.infinity_index) || 0;
                            if (themes.length) {
                                displayPrompt = themes[idx % themes.length];
                            } else {
                                const ta = document.getElementById('p-core');
                                const first = (ta && ta.value || '').split(/\r?\n/).find(l => l.trim());
                                displayPrompt = first || '(starting…)';
                            }
                        }
                        const runItem = {
                            prompt: displayPrompt,
                            config_snapshot: cfg,
                            ts: 0,
                        };
                        items.push(renderItem(runItem, { running: true, step: d.state.step }));
                    }
                    // Cancelled items fade out for ~5 s, then disappear from view.
                    // The data still persists server-side until the 48 h prune sweep.
                    const nowSec = Date.now() / 1000;
                    const visibleQueue = d.queue.filter(q => {
                        if (q.status !== 'cancelled') return true;
                        const age = nowSec - (q.cancelled_ts || 0);
                        return age < 5;
                    });
                    // Inline queue cap is dynamic by layout:
                    //   default / multi-pane → 6 (glanceable, drawer for rest)
                    //   queue-focused        → up to ~30 to fill the card's
                    //                          full vertical real estate the
                    //                          user gave it. Math: viewport
                    //                          minus header/footer reserve
                    //                          divided by ~70 px row height.
                    //                          Floor at 6, ceiling at 30 so
                    //                          we don't render thousands on
                    //                          a giant 4k display either.
                    // Done items get the SAME cap so they fill the bottom of
                    // the card too rather than always trailing 6 deep.
                    const isQueueFocused = document.body.dataset.layout === 'queue'
                        || document.body.dataset.layout === 'subj-queue'
                        || document.body.dataset.layout === 'queue-slop';
                    let inlineCap = 6;
                    if (isQueueFocused) {
                        const avail = Math.max(0, window.innerHeight - 220);
                        inlineCap = Math.max(6, Math.min(30, Math.floor(avail / 70)));
                    }
                    const pendingOnly = visibleQueue.filter(q => q.status !== 'done');
                    items.push(...pendingOnly.slice(0, inlineCap).map(q => renderItem(q, {})));
                    // Done items (newest first) — full audit log of completed
                    // iters. Same dynamic cap so done rows can also fill
                    // available height in the focused layout.
                    const doneOnly = visibleQueue.filter(q => q.status === 'done')
                        .slice().sort((a, b) => (b.completed_ts || 0) - (a.completed_ts || 0));
                    doneOnly.slice(0, inlineCap).forEach(q => {
                        items.push(_renderDoneItem(q));
                    });
                    qList.innerHTML = items.join('');
                    qList.querySelectorAll('li[data-q-ts]').forEach(li => PriorityLoader.register(li));
                }
                // Default-layout queue card visibility hinges on whether
                // q-list has any <li data-q-ts>. Only refresh on the
                // empty<->has-items transition — the rest of the time
                // _refreshCardVisibility would walk localStorage and
                // toggle classes for nothing, which was contributing to
                // the per-tick churn we just gated above. Other callers
                // (layout-mode change, card show/hide, DOMContentLoaded,
                // slop drawer toggle) still trigger it where it matters.
                const _hadItems = !!_lastQueueHadItems;
                const _hasItemsNow = qLen > 0;
                if (_hadItems !== _hasItemsNow) {
                    _lastQueueHadItems = _hasItemsNow;
                    _refreshCardVisibility();
                }
            }
            // Top-of-card segmented progress bar — single instance, hosted at
            // #active-job-progress-bar. Hidden when nothing's running so the
            // card collapses cleanly.
            const barHost = document.getElementById('active-job-progress-bar');
            if (barHost) {
                if (isRunning) {
                    barHost.innerHTML = _buildActiveJobProgressBar(d);
                    barHost.style.display = '';
                } else {
                    barHost.innerHTML = '';
                    barHost.style.display = 'none';
                    // Hide the Total elapsed / ETA cluster on the footer row
                    // when nothing's running. Bulk-action buttons still show
                    // (their visibility is independent — driven by queue contents).
                    const tot = document.getElementById('queue-progress-total');
                    if (tot) tot.style.display = 'none';
                }
            }
            // Drawer is now self-managed via /queue/paginated. If it's open
            // AND showing page 1, refresh that page so new completions show
            // up live; otherwise leave it alone — user navigates pages with
            // the prev/next/filter controls.
            const qDrawerToggle = $('queue-drawer-toggle');
            if (qDrawerToggle && qDrawerToggle.checked && _queueDrawerOffset === 0) {
                _loadQueueDrawerPage();
            }

            // Empty-state hint: only when idle AND queue empty.
            const hint = $('empty-state-hint');
            if (hint) {
                hint.style.display = (d.state.mode === 'Idle' && d.queue.length === 0) ? 'flex' : 'none';
            }

            // Hero collapse (compact line when Idle; full card otherwise).
            const hero = $('hero-card');
            const heroIdle = $('hero-idle');
            const statsNav = $('stats-navbar');
            const gpuIdle = (d.stats.gpu || 0) < 2 && (d.stats.vram || 0) < 2;
            const isIdle = d.state.mode === 'Idle';
            if (hero && heroIdle) {
                if (isIdle) {
                    hero.style.display = 'none';
                    heroIdle.style.display = 'flex';
                } else {
                    hero.style.display = 'block';
                    heroIdle.style.display = 'none';
                }
            }
            // Stats navbar: dim stat values when both GPU+VRAM are near zero
            // (no active inference). Keeps the navbar present for tickers but
            // de-emphasises meaningless "0%" numbers.
            if (statsNav) statsNav.classList.toggle('opacity-60', isIdle && gpuIdle);

            // Output section: hide when all three grids are empty; show ONE empty card instead.
            const hasAny = document.querySelector('#preview-grid > *') ||
                document.querySelector('#v-grid > *') ||
                document.querySelector('#i-grid > *');
            const outSec = $('output-section');
            const outEmpty = $('output-empty');
            if (outSec) outSec.style.display = hasAny ? 'block' : 'none';
            if (outEmpty) outEmpty.style.display = hasAny ? 'none' : 'flex';

            updateStorage(d.storage);
            updateOutputsDisk(d.outputs_disk);
            updateRam(d.ram);
            updateScheduler(d.scheduler);
            updateOutputs(d.outputs);
            // Populate the collapsible top-section summary line — visible
            // only when the user has collapsed the top pane. Kept in sync
            // each WS tick so re-expanding/re-collapsing reflects current
            // state.
            try {
                const sumEl = document.getElementById('top-collapsible-summary');
                if (sumEl) {
                    const mode = (d.state && d.state.mode) || 'Idle';
                    const v = d.state && d.state.video_index;
                    const tot = d.state && d.state.total_videos;
                    const pending = (d.queue || []).filter(q => q.status == null || q.status === 'pending').length;
                    const done = (d.queue || []).filter(q => q.status === 'done' && q.succeeded !== false).length;
                    const failed = (d.queue || []).filter(q => q.status === 'done' && q.succeeded === false).length;
                    const parts = [];
                    parts.push(mode === 'Idle' ? '<span class="opacity-60">Idle</span>' : `<b>${mode}</b>`);
                    if (v && tot) parts.push(`video ${v}/${tot}`);
                    if (pending) parts.push(`queue ${pending} pending`);
                    if (done) parts.push(`<span class="text-success">${done} done</span>`);
                    if (failed) parts.push(`<span class="text-error">${failed} failed</span>`);
                    sumEl.innerHTML = parts.join(' · ');
                }
            } catch (_) { /* summary is best-effort */ }
            _lastTick = d;
            // Queue-status chip under the Queue Slop button. Surfaces
            // depth + activity so the user knows things are progressing
            // even when the Queue card isn't on screen (focused layouts,
            // mobile). Best-effort — failures don't break the WS tick.
            try {
                if (typeof _updateQueueStatusChip === 'function') _updateQueueStatusChip(d);
            } catch (_) { /* chip is purely informational */ }
            // Push GPU% sample for the auto-suggest idle gate.
            try {
                const gpuPct = Number((d.stats && (d.stats.gpu ?? d.stats.gpu_pct ?? d.stats.gpu_util)) || 0);
                _gpuPctHistory.push({ ts: Date.now(), pct: gpuPct });
                if (_gpuPctHistory.length > _GPU_HISTORY_MAX) _gpuPctHistory.shift();
            } catch (_) { /* ignore */ }
            _renderSubjectsModels();
            updateDiagnostics(d);
        }
        if (d.type === 'new_file') {
            const file = d.file;
            // Seed the v_idx -> filename cache before any rendering uses
            // it; downstream pipeline-strip / done-list re-renders will
            // then resolve real names instead of synthesized ones.
            _ingestAssetFilename(file);
            const g = $('preview-grid');
            if (!g) return;
            const outSec = $('output-section');
            const outEmpty = $('output-empty');
            if (outSec) outSec.style.display = 'block';
            if (outEmpty) outEmpty.style.display = 'none';
            // Build via the shared card factory so SSR / WS-push / infinite
            // scroll all produce identical markup. `pulse=true` flashes the
            // newcomer; autoplay=true so a fresh video starts playing.
            const c = _buildSlopCard(file, { pulse: true, autoplay: true });
            if (!c) return;
            g.prepend(c);
            const ring = $('live-ring');
            if (ring) ring.style.display = 'inline-block';
            setTimeout(() => c.classList.remove('animate-pulse'), 3000);
            // Apply current filter state to the new card
            _applySlopFilters();
            // Note: no longer capping to 64 cards — infinite scroll loads
            // older content into the same grid, so a hard cap would defeat
            // it. WS-pushed files prepend; the bottom of the grid grows as
            // the user scrolls.
        }
    };
    ws.onclose = () => {
        _wsConnected = false;
        _updateConnPill(false);
        const w = $('refresh-wrapper');
        if (w) w.style.display = '';
        const dot = $('live-dot');
        if (dot) {
            const inner = dot.querySelector('.rounded-full:last-child');
            const ping = dot.querySelector('.animate-ping');
            if (inner) inner.classList.replace('bg-success', 'bg-error');
            if (ping) ping.classList.replace('bg-success', 'bg-error');
        }
        setTimeout(connect, 2000);
    };
}

function _concatStagePrompts() {
    const parts = [
        $('p-image') && $('p-image').value,
        $('p-video') && $('p-video').value,
        $('p-music') && $('p-music').value,
        $('p-tts') && $('p-tts').value,
    ].filter(Boolean);
    const concat = parts.join('\n\n');
    if ($('p-in')) $('p-in').value = concat;
    return concat;
}

// ---- Fan-out (P2) ----------------------------------------------------------
const STAGE_NAMES = ['image', 'video', 'music', 'tts'];
let _fanoutPending = null; // { stages: {...}, preserved_ok, preserved_dropped }

function _stageVal(name) {
    const el = $('p-' + name);
    return el ? el.value : '';
}

function _setStageVal(name, v) {
    const el = $('p-' + name);
    if (el) el.value = v;
}

function _lockedList() {
    return STAGE_NAMES.filter(n => {
        const b = $('lock-' + n);
        return b && b.dataset.locked === '1';
    });
}

function _setLock(name, locked) {
    const b = $('lock-' + name);
    if (!b) return;
    if (locked) {
        b.dataset.locked = '1';
        b.innerText = '🔒';
        b.title = 'Locked — your text will be preserved';
        b.classList.remove('badge-ghost');
        b.classList.add('badge-warning');
    } else {
        b.dataset.locked = '0';
        b.innerText = '🔓';
        b.title = 'Unlocked — will be overwritten';
        b.classList.remove('badge-warning');
        b.classList.add('badge-ghost');
    }
}

function _wireLockListeners() {
    STAGE_NAMES.forEach(n => {
        const ta = $('p-' + n);
        if (!ta) return;
        const sync = () => _setLock(n, (ta.value || '').length >= 1);
        ta.addEventListener('input', sync);
        sync();
    });
}

function _htmlEscape(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

// Render the fan-out as 4 stacked / 2x2 boxes — every stage prompt visible
// at once. Replaces the older tab-switcher UI; the user wanted to see all
// suggestions side-by-side rather than clicking through tabs.
function _renderFanoutPreview(stages) {
    STAGE_NAMES.forEach(n => {
        const box = $('fanout-' + n);
        if (box) box.textContent = (stages && stages[n]) || '';
        // Mirror the lock state from the corresponding p-<stage> textarea so
        // the user can see at a glance which boxes will be skipped on Accept.
        const lockBadge = $('fanout-' + n + '-lock');
        const srcLock = $('lock-' + n);
        if (lockBadge && srcLock) {
            const locked = srcLock.dataset.locked === '1';
            lockBadge.textContent = locked ? '🔒' : '🔓';
            lockBadge.title = locked
                ? 'Locked — your existing text will be preserved'
                : 'Unlocked — will be overwritten on Accept';
            lockBadge.classList.toggle('badge-warning', locked);
            lockBadge.classList.toggle('badge-ghost', !locked);
        }
    });
}

// Single-tab enhancer — rewrite ONE stage's prompt, leaving the others alone.
async function enhanceStage(stage) {
    const fieldMap = { image: 'p-image', video: 'p-video', music: 'p-music', tts: 'p-tts' };
    const fieldId = fieldMap[stage];
    if (!fieldId) return;
    const ta = document.getElementById(fieldId);
    if (!ta) return;
    const core = ($('p-core') && $('p-core').value) || '';
    const seed = (ta.value || '').trim() || core.trim();
    if (!seed) return;
    if (!(await _ramGuardCheck())) return;
    const sys = {
        image: 'Rewrite the prompt as a detailed visual still-frame description for an AI image generator. Lighting, texture, mood. Under 60 words. Output ONLY the rewritten prompt.',
        video: 'Rewrite the prompt as a motion/camera description for an AI video generator. Camera movement, pacing, transitions. Under 60 words. Output ONLY the rewritten prompt.',
        music: 'Rewrite the prompt as a short mood/genre description suitable for a music generator (instruments, tempo, vibe). Under 30 words. Output ONLY the description.',
        tts: 'Rewrite the prompt as a one or two sentence voiceover line spoken in first or third person. Output ONLY the line.',
    }[stage] || 'Rewrite the prompt for ' + stage;
    const orig = ta.value;
    ta.value = '✨ rewriting...';
    ta.disabled = true;
    // 180-s ceiling — reasoning models (qwen3.5-claude-distill etc.) can
    // burn 60-120 s on a single rewrite. AbortController cuts the fetch
    // cleanly so the textarea unfreezes; the catch surfaces the timeout.
    const controller = new AbortController();
    const t0 = Date.now();
    const timer = setTimeout(() => controller.abort(), 180_000);
    try {
        const res = await fetch('/enhance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: `${sys}\n\nSubject: ${seed}` }),
            signal: controller.signal,
        });
        const r = await res.json();
        ta.value = (r.suggestion || orig).trim();
    } catch (e) {
        const elapsed = Math.round((Date.now() - t0) / 1000);
        if (e && e.name === 'AbortError') {
            ta.value = orig;
            if (typeof _seedToast === 'function') {
                _seedToast(`Rewrite timed out after ${elapsed}s — LLM is slow`, 'warning');
            }
        } else {
            ta.value = orig;
            if (typeof _seedToast === 'function') {
                _seedToast(`Rewrite failed: ${e && e.message || e}`, 'error');
            }
        }
    } finally {
        clearTimeout(timer);
        ta.disabled = false;
    }
}

async function enhance() {
    const core = ($('p-core') && $('p-core').value) || '';
    const stages = {
        image: _stageVal('image'),
        video: _stageVal('video'),
        music: _stageVal('music'),
        tts: _stageVal('tts'),
    };
    if (!core.trim() && !Object.values(stages).some(v => v && v.trim())) return;
    if (!(await _ramGuardCheck())) return;
    const locked = _lockedList();
    const preview = $('fanout-preview');
    if (preview) preview.classList.remove('hidden');
    const warn = $('fanout-warn');
    if (warn) { warn.classList.add('hidden'); warn.innerText = ''; }
    // Loading placeholder — drop a "thinking" hint into each of the 4 boxes
    // so the preview reflects active work even before the LLM responds.
    STAGE_NAMES.forEach(n => {
        const box = $('fanout-' + n);
        if (box) box.textContent = '✨ rewriting...';
    });
    try {
        const res = await fetch('/enhance/distribute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ core, stages, locked, preserve_tokens: [] }),
        });
        const r = await res.json();
        if (r && r.stages) {
            _fanoutPending = r;
            _renderFanoutPreview(r.stages);
            if (!r.preserved_ok && warn) {
                warn.classList.remove('hidden');
                warn.innerText = 'Restored dropped tokens: ' + (r.preserved_dropped || []).join(', ');
            }
        } else {
            STAGE_NAMES.forEach(n => {
                const box = $('fanout-' + n);
                if (box) box.textContent = '(no response)';
            });
        }
    } catch (e) {
        STAGE_NAMES.forEach(n => {
            const box = $('fanout-' + n);
            if (box) box.textContent = 'Error: ' + e;
        });
    }
}

function acceptFanout() {
    if (!_fanoutPending || !_fanoutPending.stages) return;
    STAGE_NAMES.forEach(n => {
        const v = _fanoutPending.stages[n];
        if (typeof v === 'string') _setStageVal(n, v);
        _setLock(n, (_stageVal(n) || '').length >= 1);
    });
    _concatStagePrompts();
    const preview = $('fanout-preview');
    if (preview) preview.classList.add('hidden');
    _fanoutPending = null;
}

function editFanout() {
    if (!_fanoutPending || !_fanoutPending.stages) return;
    STAGE_NAMES.forEach(n => {
        const el = $('fe-' + n);
        if (el) el.value = _fanoutPending.stages[n] || '';
    });
    const m = $('fanout-edit-modal');
    if (m && typeof m.showModal === 'function') m.showModal();
}

function commitFanoutEdit() {
    if (!_fanoutPending) _fanoutPending = { stages: {} };
    STAGE_NAMES.forEach(n => {
        const el = $('fe-' + n);
        if (el) _fanoutPending.stages[n] = el.value;
    });
    const m = $('fanout-edit-modal');
    if (m && typeof m.close === 'function') m.close();
    _renderFanoutPreview(_fanoutPending.stages);
}

// Legacy — kept so any stray caller doesn't explode.
function applyAi() {
    const s = $('ai-s') && $('ai-s').innerText;
    if ($('p-in') && s) $('p-in').value = s;
    const box = $('ai-box');
    if (box) box.classList.add('hidden');
}

async function inject(prio, terminate, concurrent, opts) {
    // Per-stage override fields (set in Pipeline modal). If they're all empty
    // the prompt comes from the Subjects textarea instead — newline = one job.
    const stageConcat = _concatStagePrompts();
    const subjectsRaw = ($('p-core') && $('p-core').value || '').trim();
    const subjects = subjectsRaw
        ? subjectsRaw.split(/\r?\n/).map(s => s.trim()).filter(Boolean)
        : [];
    const stages = {
        image: $('p-image') ? $('p-image').value : '',
        video: $('p-video') ? $('p-video').value : '',
        music: $('p-music') ? $('p-music').value : '',
        tts: $('p-tts') ? $('p-tts').value : '',
    };
    // Decide what to enqueue: stage-override concat (one job), or one job per
    // subject line if the user is driving from the Subjects textarea.
    const prompts = stageConcat ? [stageConcat] : (subjects.length ? subjects : []);
    if (!prompts.length) {
        console.warn('inject: no prompt available — Prompt textarea empty and no stage overrides set');
        return;
    }
    // Send one /inject per prompt. Multiple subjects (newline-separated)
    // become multiple queue items in order.
    for (const promptText of prompts) {
        const f = new FormData();
        f.append('prompt', promptText);
        f.append('priority', prio);
        if (terminate) f.append('terminate', '1');
        if (concurrent) f.append('concurrent', '1');
        if (opts && opts.infinity) f.append('infinity', '1');
        if (opts && opts.whenIdle) f.append('when_idle', '1');
        if (opts && opts.chaos) f.append('chaos', '1');
        if (opts && opts.fastTrack) f.append('fast_track', '1');
        // Seeds — staged via the Subjects-card picker. Forward the list +
        // mode so the server can fan out (per-task) or carry through to the
        // chain loop (per-chain, FLF2V). Empty list = no-op on backend.
        const _stagedSeeds = (typeof _getStagedSeeds === 'function') ? _getStagedSeeds() : [];
        if (_stagedSeeds.length) {
            f.append('seed_images', JSON.stringify(_stagedSeeds));
            f.append('seeds_mode', (typeof _getSeedsMode === 'function') ? _getSeedsMode() : 'per-task');
        }
        if (stageConcat) f.append('stage_prompts', JSON.stringify(stages));
        await fetch('/inject', { method: 'POST', body: f });
    }
    // Only blank the per-stage overrides — leave the Subjects textarea alone
    // so the user can re-queue the same set quickly if they want to.
    ['p-image', 'p-video', 'p-music', 'p-tts', 'p-in'].forEach(id => { if ($(id)) $(id).value = ''; });
    // Toast confirmation so the user knows the submit landed even when
    // the Queue card isn't on screen (focused layouts, mobile, etc).
    if (typeof _toast === 'function') {
        const n = prompts.length;
        const first = (prompts[0] || '').slice(0, 60).replace(/\s+/g, ' ').trim();
        const msg = n === 1
            ? `Queued: ${first}${(prompts[0] || '').length > 60 ? '…' : ''}`
            : `Queued ${n} jobs · first: ${first}${(prompts[0] || '').length > 60 ? '…' : ''}`;
        _toast(msg, 'success');
    }
}

// Deprecated: the standalone "Will use" badge row was removed; the RAM
// estimator breakdown now renders role + model + GB together. Keep this
// stub so existing callers don't error.
function _renderSubjectsModels() { }

async function updatePipeline() {
    // Resolve a model select's value, swapping in the slopped sub-select
    // value when the user has picked __slopped__ AND chosen a concrete file.
    // If they haven't picked yet, leave `__slopped__` so the server's random
    // fallback kicks in (rather than persisting the sentinel).
    const _resolve = (topId, subId) => {
        const top = $(topId) ? $(topId).value : '';
        if (top !== '__slopped__') return top;
        const sub = $(subId);
        const subVal = sub ? sub.value : '';
        return subVal && subVal.startsWith('slopped:') ? subVal : '__slopped__';
    };
    const baseVal = _resolve('cfg-base', 'cfg-base-slopped');
    const audioVal = _resolve('cfg-audio', 'cfg-audio-slopped');
    const ttsVal = _resolve('cfg-tts', 'cfg-tts-slopped');
    const body = {
        infinity_mode: $('inf-on') ? $('inf-on').checked : false,
        // Prefer newline-based subjects from #p-core textarea; fall back to
        // legacy comma-separated hidden #inf-themes shim for compat.
        infinity_themes: _subjectsFromTextarea().length
            ? _subjectsFromTextarea()
            : ($('inf-themes') ? $('inf-themes').value.split(',').map(s => s.trim()).filter(Boolean) : []),
        base_model: baseVal,
        video_model: $('cfg-video') ? $('cfg-video').value : '',
        audio_model: audioVal,
        tts_model: ttsVal,
        upscale_model: $('cfg-upscale') ? $('cfg-upscale').value : '',
        frames: $('cfg-frames') ? parseInt($('cfg-frames').value, 10) :
            ($('cfg-video') && $('cfg-video').value.includes('wan') ? 81 : 49),
    };
    if ($('cfg-chains')) body.chains = parseInt($('cfg-chains').value, 10);
    if ($('cfg-size')) body.size = $('cfg-size').value;
    if ($('cfg-tier')) body.tier = $('cfg-tier').value;
    if ($('cfg-consolidation')) body.consolidation = $('cfg-consolidation').value;
    if ($('cfg-music-gain-db')) body.music_gain_db = parseInt($('cfg-music-gain-db').value, 10);
    if ($('cfg-fade-s')) body.fade_s = parseFloat($('cfg-fade-s').value);
    if ($('chaos-on')) body.chaos_mode = $('chaos-on').checked;
    if ($('when-idle-on')) body.when_idle = $('when-idle-on').checked;
    if ($('date-time-on')) body.show_date_time = $('date-time-on').checked;
    await fetch('/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    // Snappy feedback: recompute RAM estimate immediately without waiting for WS tick.
    try {
        const qs = new URLSearchParams({
            base: body.base_model,
            video: body.video_model,
            audio: body.audio_model,
            upscale: body.upscale_model,
            tts: body.tts_model,
        });
        const res = await fetch('/ram_estimate?' + qs.toString());
        if (res.ok) updateRam(await res.json());
    } catch (e) { /* WS tick will catch up */ }
}

// ── Queue drawer pagination ────────────────────────────────────────────────
// State for the "View all" drawer. Server-paginated via /queue/paginated;
// the drawer renders one page at a time so 1000+ done items stay snappy.
let _queueDrawerOffset = 0;
let _queueDrawerLimit = 25;
let _queueDrawerFilter = 'all';
let _queueDrawerTotal = 0;

function _renderQueueDrawerItem(q) {
    if (q && q.status === 'done') {
        // Reuse the canonical done-item renderer so the drawer matches
        // the inline strip's markup (thumbnails, asset rows, etc.).
        return `<ul class="m-0 p-0 list-none">${_renderDoneItem(q)}</ul>`;
    }
    const cls = q && q.status === 'cancelled' ? 'border-warning/40' :
        q && q.status === 'pending' ? 'border-base-200' : 'border-base-200';
    const badge = q && q.status === 'cancelled' ? '<span class="badge badge-xs badge-warning mr-1">cancelled</span>' :
        q && q.status === 'pending' ? '<span class="badge badge-xs badge-info mr-1">pending</span>' : '';
    const prompt = _htmlEscape(((q && q.prompt) || '').substring(0, 200));
    return `<div class="bg-base-300 p-3 rounded text-xs border ${cls}">${badge}${prompt}</div>`;
}

async function _loadQueueDrawerPage() {
    const body = $('queue-drawer-body');
    const info = $('queue-drawer-page-info');
    const totalEl = $('queue-drawer-total');
    const prev = $('queue-drawer-prev');
    const next = $('queue-drawer-next');
    if (!body) return;
    try {
        const qs = new URLSearchParams({
            offset: String(_queueDrawerOffset),
            limit: String(_queueDrawerLimit),
            filter: _queueDrawerFilter,
        });
        const res = await fetch('/queue/paginated?' + qs.toString());
        const r = await res.json();
        const items = Array.isArray(r.items) ? r.items : [];
        _queueDrawerTotal = r.total || 0;
        if (_queueDrawerTotal === 0) {
            body.innerHTML = '<div class="text-xs text-base-content/50 italic text-center p-4">No queued items yet.</div>';
        } else {
            body.innerHTML = items.map(_renderQueueDrawerItem).join('');
        }
        const totalPages = Math.max(1, Math.ceil(_queueDrawerTotal / _queueDrawerLimit));
        const currentPage = Math.floor(_queueDrawerOffset / _queueDrawerLimit) + 1;
        if (info) info.textContent = `Page ${currentPage} / ${totalPages}`;
        if (totalEl) totalEl.textContent = `${_queueDrawerTotal} item${_queueDrawerTotal === 1 ? '' : 's'}`;
        if (prev) prev.classList.toggle('btn-disabled', _queueDrawerOffset <= 0);
        if (next) next.classList.toggle('btn-disabled', !r.has_more);
    } catch (e) {
        body.innerHTML = '<div class="text-xs text-error italic text-center p-4">Failed to load queue page.</div>';
    }
}

function _wireQueueDrawerControls() {
    const toggle = $('queue-drawer-toggle');
    if (toggle && !toggle._paginatedWired) {
        toggle.addEventListener('change', () => {
            if (toggle.checked) {
                _queueDrawerOffset = 0;
                _loadQueueDrawerPage();
            }
        });
        toggle._paginatedWired = true;
    }
    const prev = $('queue-drawer-prev');
    if (prev && !prev._wired) {
        prev.addEventListener('click', () => {
            if (_queueDrawerOffset <= 0) return;
            _queueDrawerOffset = Math.max(0, _queueDrawerOffset - _queueDrawerLimit);
            _loadQueueDrawerPage();
        });
        prev._wired = true;
    }
    const next = $('queue-drawer-next');
    if (next && !next._wired) {
        next.addEventListener('click', () => {
            if (_queueDrawerOffset + _queueDrawerLimit >= _queueDrawerTotal) return;
            _queueDrawerOffset += _queueDrawerLimit;
            _loadQueueDrawerPage();
        });
        next._wired = true;
    }
    const filt = $('queue-drawer-filter');
    if (filt && !filt._wired) {
        filt.addEventListener('change', () => {
            _queueDrawerFilter = filt.value || 'all';
            _queueDrawerOffset = 0;
            _loadQueueDrawerPage();
        });
        filt._wired = true;
    }
    const psz = $('queue-drawer-pagesize');
    if (psz && !psz._wired) {
        psz.addEventListener('change', () => {
            const v = parseInt(psz.value, 10);
            if (Number.isFinite(v) && v > 0) {
                _queueDrawerLimit = v;
                _queueDrawerOffset = 0;
                _loadQueueDrawerPage();
            }
        });
        psz._wired = true;
    }
}

if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _wireQueueDrawerControls);
    } else {
        _wireQueueDrawerControls();
    }
}

function openQueueDrawer() {
    const t = $('queue-drawer-toggle');
    if (t) {
        t.checked = true;
        _queueDrawerOffset = 0;
        _loadQueueDrawerPage();
    }
}

async function clearFailedQueue() {
    const btn = document.getElementById('btn-clear-failed');
    if (btn) btn.disabled = true;
    try {
        const r = await fetch('/queue/clear-failed', { method: 'POST' });
        const j = await r.json();
        if (!j.ok) console.warn('clear-failed:', j);
    } catch (e) {
        console.warn('clear-failed fetch:', e);
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function requeueFailedQueue() {
    const btn = document.getElementById('btn-requeue-failed');
    if (btn) btn.disabled = true;
    try {
        const r = await fetch('/queue/requeue-failed', { method: 'POST' });
        const j = await r.json();
        if (!j.ok) console.warn('requeue-failed:', j);
    } catch (e) {
        console.warn('requeue-failed fetch:', e);
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function clearCompletedQueue() {
    const btn = document.getElementById('btn-clear-completed');
    if (btn) btn.disabled = true;
    try {
        const r = await fetch('/queue/clear-completed', { method: 'POST' });
        const j = await r.json();
        if (!j.ok) console.warn('clear-completed:', j);
    } catch (e) {
        console.warn('clear-completed fetch:', e);
    } finally {
        if (btn) btn.disabled = false;
    }
}

// Toggles the queue-header actions based on what kinds of items the
// queue contains. Failed items expose Requeue + Clear-Failed; completed
// items expose Clear-Completed. Hidden by default so the header stays
// clean when there's nothing to act on.
function _refreshFailedActionsVisibility(queue) {
    const items = queue || [];
    const anyFailed = items.some(q => q.status === 'done' && q.succeeded === false);
    const anyCompleted = items.some(q => q.status === 'done' && q.succeeded !== false);
    for (const id of ['btn-requeue-failed', 'btn-clear-failed']) {
        const btn = document.getElementById(id);
        if (btn) btn.style.display = anyFailed ? '' : 'none';
    }
    const mobileRequeue = document.getElementById('mobile-nav-requeue-failed');
    if (mobileRequeue) mobileRequeue.classList.toggle('hidden', !anyFailed);
    const btnDone = document.getElementById('btn-clear-completed');
    if (btnDone) btnDone.style.display = anyCompleted ? '' : 'none';
}

async function generateTts() {
    const text = $('tts-in') ? $('tts-in').value.trim() : '';
    const voice = $('tts-voice') ? $('tts-voice').value : 'ryan';
    const statusEl = $('tts-status');
    const previewEl = $('tts-preview');
    if (!text) { if (statusEl) statusEl.innerText = 'empty'; return; }
    if (!(await _ramGuardCheck())) {
        if (statusEl) { statusEl.innerText = 'cancelled'; statusEl.className = 'badge badge-xs badge-ghost'; }
        return;
    }
    if (statusEl) { statusEl.innerText = 'synth...'; statusEl.className = 'badge badge-xs badge-warning'; }
    try {
        const res = await fetch('/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice }),
        });
        const r = await res.json();
        const url = r.url || r.audio_path;
        if ((r.ok || r.status === 'ok') && url) {
            if (previewEl) { previewEl.src = url + '?t=' + Date.now(); previewEl.play().catch(() => { }); }
            if (statusEl) { statusEl.innerText = 'ok'; statusEl.className = 'badge badge-xs badge-success'; }
        } else if (statusEl) {
            statusEl.innerText = 'err';
            statusEl.className = 'badge badge-xs badge-error';
        }
    } catch (e) {
        if (statusEl) { statusEl.innerText = 'err'; statusEl.className = 'badge badge-xs badge-error'; }
    }
}

// -------------------- Settings modal (local-only LLM) --------------------

const PROVIDER_DEFAULTS = {
    lmstudio: { port: 1234, path: '/v1' },
    ollama: { port: 11434, path: '' },
    vllm: { port: 8000, path: '/v1' },
    llamacpp: { port: 8080, path: '/v1' },
    custom: { port: 8080, path: '/v1' },
};

function computeBaseUrl() {
    const scheme = $('set-scheme').value;
    const host = $('set-host').value.trim() || 'localhost';
    const port = $('set-port').value;
    const path = $('set-path').value || '';
    const p = path && !path.startsWith('/') ? '/' + path : path;
    const url = `${scheme}://${host}${port ? ':' + port : ''}${p}`;
    $('set-base').value = url;
    return url;
}

function onSettingsUrlChanged() { computeBaseUrl(); }

function applyProviderDefaults() {
    const prov = $('set-provider').value;
    const d = PROVIDER_DEFAULTS[prov] || PROVIDER_DEFAULTS.custom;
    $('set-port').value = d.port;
    $('set-path').value = d.path;
    computeBaseUrl();
}

function parseBaseUrl(url) {
    try {
        const u = new URL(url);
        return {
            scheme: u.protocol.replace(':', ''),
            host: u.hostname,
            port: u.port || (u.protocol === 'https:' ? '443' : '80'),
            path: u.pathname && u.pathname !== '/' ? u.pathname.replace(/\/$/, '') : '',
        };
    } catch (e) {
        return null;
    }
}

function toggleApiKeyReveal() {
    const el = $('set-api-key');
    const btn = $('set-api-reveal');
    if (!el || !btn) return;
    if (el.type === 'password') { el.type = 'text'; btn.innerText = 'hide'; }
    else { el.type = 'password'; btn.innerText = 'show'; }
}

// LLM model selector now lives in the Pipeline popup (cfg-llm) so all
// per-iteration model choices are co-located. Connection details
// (provider/host/port/key) stay in Settings → LLM. The handler below
// shows/hides the free-text custom-id input and persists the choice
// via partial /settings POST so the orchestrator picks up the new
// model_id on its next iteration.
function onCfgLlmChanged() {
    const sel = $('cfg-llm');
    const custom = $('cfg-llm-custom');
    if (!sel) return;
    let modelId = sel.value;
    if (modelId === '__custom__') {
        if (custom) { custom.classList.remove('hidden'); custom.focus(); }
        modelId = (custom && custom.value || '').trim();
        if (!modelId) return; // wait for user to type
    } else if (custom) {
        custom.classList.add('hidden');
    }
    fetch('/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ llm: { model_id: modelId } }),
    }).catch(err => console.warn('persist llm model_id failed', err));
}

async function reloadModels() {
    const sel = $('cfg-llm');
    if (!sel) return;
    // Connection details (provider, host, port, key) live in Settings →
    // LLM and are read from the inputs here. If the Settings modal hasn't
    // been opened yet this session those inputs may not be populated —
    // fall back to the values baked into the config snapshot.
    const base = (typeof computeBaseUrl === 'function') ? computeBaseUrl() : '';
    const provider = $('set-provider') ? $('set-provider').value : '';
    const apiKey = $('set-api-key') ? $('set-api-key').value : '';
    const prevSelected = sel.dataset.selected || sel.value || '';
    sel.innerHTML = '<option value="">(loading…)</option>';
    try {
        const qs = new URLSearchParams({ base_url: base, provider, api_key: apiKey || '' });
        const r = await fetch('/settings/models?' + qs.toString());
        const d = await r.json();
        const models = (d && d.models) || [];
        sel.innerHTML = '';
        if (!models.length) {
            const opt = document.createElement('option');
            opt.value = ''; opt.innerText = '(none found)';
            sel.appendChild(opt);
        }
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m; opt.innerText = m;
            if (m === prevSelected) opt.selected = true;
            sel.appendChild(opt);
        });
        const customOpt = document.createElement('option');
        customOpt.value = '__custom__'; customOpt.innerText = '⌨ Custom…';
        sel.appendChild(customOpt);
    } catch (e) {
        sel.innerHTML = '<option value="">(error)</option>';
    }
}

async function probeLan() {
    const box = $('set-probe-chips');
    if (!box) return;
    box.innerHTML = '<span class="badge badge-ghost badge-sm">scanning…</span>';
    try {
        const r = await fetch('/settings/probe');
        const d = await r.json();
        const eps = (d && d.endpoints) || [];
        box.innerHTML = '';
        if (!eps.length) { box.innerHTML = '<span class="badge badge-warning badge-sm">no local endpoints found</span>'; return; }
        eps.forEach(ep => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'badge badge-primary badge-outline cursor-pointer';
            chip.innerText = `${ep.provider} :${ep.port} (${ep.model_count})`;
            chip.onclick = () => {
                $('set-provider').value = ep.provider;
                const parsed = parseBaseUrl(ep.base_url);
                if (parsed) {
                    $('set-scheme').value = parsed.scheme;
                    $('set-host').value = parsed.host;
                    $('set-port').value = parsed.port;
                    $('set-path').value = parsed.path;
                }
                computeBaseUrl();
                reloadModels();
            };
            box.appendChild(chip);
        });
    } catch (e) {
        box.innerHTML = '<span class="badge badge-error badge-sm">probe failed</span>';
    }
}

async function testSettings() {
    const badge = $('set-test-badge');
    if (!badge) return;
    badge.className = 'badge badge-warning font-mono text-xs';
    badge.innerText = 'testing…';
    // Model selector is in Pipeline popup now (cfg-llm). Fall back to
    // the live config snapshot when the Pipeline popup hasn't been opened
    // this session — connection-test should still work without forcing
    // the user to open Pipeline first.
    const sel = $('cfg-llm');
    let model_id = sel ? sel.value : '';
    if (model_id === '__custom__') {
        const c = $('cfg-llm-custom');
        model_id = c ? c.value.trim() : '';
    }
    if (!model_id) {
        model_id = (_lastTick && _lastTick.config && _lastTick.config.llm && _lastTick.config.llm.model_id) || '';
    }
    const payload = {
        base_url: computeBaseUrl(),
        provider: $('set-provider').value,
        model_id,
        api_key: $('set-api-key').value || '',
    };
    try {
        const r = await fetch('/settings/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const d = await r.json();
        if (d.ok) {
            const mc = d.model_count != null ? `${d.model_count} models · ` : '';
            badge.className = 'badge badge-success font-mono text-xs';
            badge.innerText = `✓ ${mc}${d.latency_ms} ms`;
        } else {
            badge.className = 'badge badge-error font-mono text-xs';
            badge.innerText = `✗ ${(d.error || 'error').slice(0, 60)}`;
        }
    } catch (e) {
        badge.className = 'badge badge-error font-mono text-xs';
        badge.innerText = '✗ network';
    }
}

// Shortcut from "Suggestion Prompt?" link → open Settings, switch to LLM tab,
// scroll the custom-suggestion-prompt textarea into view, focus it.
async function openSettingsToSuggestionPrompt() {
    await openSettings();
    // The Settings modal uses a tabs-radio pattern; flip the LLM tab on.
    const llmTab = document.querySelector('input[name="settings_tabs"][aria-label="LLM"]');
    if (llmTab) llmTab.checked = true;
    // Wait a frame for the tab content to render, then focus the textarea.
    requestAnimationFrame(() => {
        const ta = document.getElementById('set-suggest-custom-prompt');
        if (ta) {
            ta.scrollIntoView({ behavior: 'smooth', block: 'center' });
            setTimeout(() => ta.focus(), 200);
        }
    });
}

// Drawer-style settings open/close. The legacy `<dialog>` was unworkable
// at narrow viewports — the modal-box capped at max-w-4xl and the inner
// 9-tab strip overflowed in ways that broke touch scrolling. The drawer
// pattern (matching the queue drawer) gives the user a full-height side
// panel they can swipe / tap-overlay to dismiss. The legacy
// `settings-modal.close()` callsites still work via the openSettings()
// shim — it sets the drawer-toggle checkbox to false.
function _setSettingsOpen(open) {
    const tog = document.getElementById('settings-drawer-toggle');
    if (tog) {
        tog.checked = !!open;
        tog.dispatchEvent(new Event('change', { bubbles: true }));
    }
}
window._setSettingsOpen = _setSettingsOpen;

// Settings tab strip — page-scroll the overflow strip ±240 px when the
// user clicks the chevron overlays. Also wires up an `at-start`/`at-end`
// class toggle so the matching arrow fades out at the edges (CSS owns
// the visual state). 240 px ≈ 2-3 tab widths per click.
function _settingsTabsScroll(direction) {
    const strip = document.getElementById('settings-tab-strip');
    if (!strip) return;
    strip.scrollBy({ left: direction * 240, behavior: 'smooth' });
}
window._settingsTabsScroll = _settingsTabsScroll;
function _refreshSettingsTabsArrows() {
    const strip = document.getElementById('settings-tab-strip');
    const wrap = strip && strip.parentElement;
    if (!strip || !wrap) return;
    const atStart = strip.scrollLeft <= 4;
    const atEnd = strip.scrollLeft + strip.clientWidth >= strip.scrollWidth - 4;
    wrap.classList.toggle('at-start', atStart);
    wrap.classList.toggle('at-end', atEnd);
}
document.addEventListener('DOMContentLoaded', () => {
    const strip = document.getElementById('settings-tab-strip');
    if (!strip) return;
    strip.addEventListener('scroll', _refreshSettingsTabsArrows, { passive: true });
    // Also recompute when the drawer opens (strip dims may have been 0
    // before the drawer-content hydrated).
    const tog = document.getElementById('settings-drawer-toggle');
    if (tog) tog.addEventListener('change', () => setTimeout(_refreshSettingsTabsArrows, 60));
    _refreshSettingsTabsArrows();
});

async function openSettings() {
    const modal = $('settings-modal');
    if (!modal) return;
    // Open the drawer FIRST so any subsequent scrollIntoView / focus work
    // happens against a visible panel.
    _setSettingsOpen(true);
    // Pre-fill the Subjects-mode tunables (Endless cycle interval + Fresh
    // marquee toggle) — they live in localStorage, not in the server
    // settings payload, so hydrate them directly before the modal opens.
    try {
        const interval = parseInt(localStorage.getItem('slopfinity-endless-cycle-s') || '12', 10);
        const slider = document.getElementById('slop-endless-interval');
        const lbl = document.getElementById('slop-endless-interval-val');
        if (slider) slider.value = String(interval);
        if (lbl) lbl.innerText = String(interval);
        const fresh = localStorage.getItem('slopfinity-fresh') === '1';
        const fresEl = document.getElementById('slop-fresh-toggle-modal');
        if (fresEl) fresEl.checked = fresh;
        // Simple-mode initial-rows hydrate (default 3).
        const rows = parseInt(localStorage.getItem('slopfinity-simple-initial-rows') || '3', 10);
        const rowsClamped = Math.max(1, Math.min(10, isFinite(rows) ? rows : 3));
        const rowsSlider = document.getElementById('slop-simple-rows');
        const rowsLabel = document.getElementById('slop-simple-rows-val');
        if (rowsSlider) rowsSlider.value = String(rowsClamped);
        if (rowsLabel) rowsLabel.innerText = String(rowsClamped);
        // Visuals density hydrate (default 1.0; range 0.5-1.5).
        const dens = parseFloat(localStorage.getItem('slopfinity-visuals-density') || '1');
        const densClamped = Math.max(0.5, Math.min(1.5, isFinite(dens) ? dens : 1));
        const densSlider = document.getElementById('set-visuals-density');
        const densLabel = document.getElementById('set-visuals-density-val');
        if (densSlider) densSlider.value = String(densClamped);
        if (densLabel) densLabel.innerText = String(densClamped);
    } catch (_) { }
    try {
        const [sr, br] = await Promise.all([
            fetch('/settings').then(r => r.json()),
            fetch('/branding').then(r => r.json()),
        ]);
        const llm = sr.llm || {};
        $('set-provider').value = llm.provider || 'lmstudio';
        const parsed = parseBaseUrl(llm.base_url || 'http://localhost:1234/v1');
        if (parsed) {
            $('set-scheme').value = parsed.scheme;
            $('set-host').value = parsed.host;
            $('set-port').value = parsed.port;
            $('set-path').value = parsed.path;
        }
        computeBaseUrl();
        $('set-api-key').value = sr.llm_has_api_key ? '***' : '';
        $('set-api-key').type = 'password';
        $('set-api-reveal').innerText = 'show';
        $('set-temp').value = llm.temperature ?? 0.7;
        $('set-temp-val').innerText = $('set-temp').value;
        $('set-retries').value = llm.max_retries ?? 2;
        $('set-timeout').value = llm.timeout_s ?? 60;
        // Disk guard thresholds (Settings → General → Disk guard).
        const dpct = $('set-disk-min-pct');
        const dgb = $('set-disk-min-gb');
        if (dpct) dpct.value = String(sr.disk_min_pct ?? 1);
        if (dgb) dgb.value = String(sr.disk_min_gb ?? 5);
        renderAutoSuspendList(sr.auto_suspend);
        // Cloud-endpoints toggle (Settings → LLM). Server-side default
        // is OFF — show the local-only provider list. When ON, future
        // cloud entries in the registry would surface here too.
        const allowCloud = $('set-allow-cloud-endpoints');
        if (allowCloud) allowCloud.checked = !!sr.allow_cloud_endpoints;
        filterProviderDropdown(!!sr.allow_cloud_endpoints);
        const fleetPrompt = $('set-fleet-prompt');
        if (fleetPrompt) fleetPrompt.value = sr.philosophical_prompt || '';
        const sugUseSub = $('set-suggest-use-subjects');
        if (sugUseSub) {
            // Default ON when the server hasn't returned the key yet.
            sugUseSub.checked = (sr.suggest_use_subjects === undefined)
                ? true
                : !!sr.suggest_use_subjects;
        }
        const sugCustom = $('set-suggest-custom-prompt');
        if (sugCustom) sugCustom.value = sr.suggest_custom_prompt || '';
        const sugPerRow = $('set-suggest-per-row-prompts');
        if (sugPerRow) sugPerRow.checked = !!sr.suggest_per_row_prompts;

        // Part 3 — Positive/Scored config hydration
        const sugAutoEn = $('set-suggest-auto-enabled');
        if (sugAutoEn) sugAutoEn.checked = sr.auto_suggest_enabled ?? !sr.suggest_auto_disabled;

        const idleThr = $('set-idle-throttle');
        if (idleThr) {
            idleThr.value = sr.idle_throttle_pct ?? (sr.when_idle ? 10 : 0);
            const lbl = $('set-idle-throttle-val'); if (lbl) lbl.innerText = idleThr.value;
        }

        const creatScore = $('set-creativity-score');
        if (creatScore) {
            creatScore.value = sr.creativity_score ?? (sr.chaos_mode ? 8 : 5);
            const lbl = $('set-creativity-score-val'); if (lbl) lbl.innerText = creatScore.value;
        }

        const qualScore = $('cfg-quality-score');
        if (qualScore) {
            let def = 5;
            if (sr.tier === 'low') def = 2;
            else if (sr.tier === 'med') def = 5;
            else if (sr.tier === 'high') def = 9;
            qualScore.value = sr.quality_score ?? def;
            const lbl = $('cfg-quality-score-val'); if (lbl) lbl.innerText = qualScore.value;
        }

        const concBud = $('set-concurrent-budget');
        if (concBud) {
            concBud.value = sr.concurrency_budget_gb ?? (sr.concurrent ? 8 : 0);
            const lbl = $('set-concurrent-budget-val'); if (lbl) lbl.innerText = concBud.value;
        }
        // ...
        // Prompts tab — pre-fill each textarea with the stored override
        // (empty when the user hasn't customised it). Placeholder shows the
        // built-in default so users know what they'd be overriding.
        _hydratePromptField('set-prompts-enhancer',
            sr.enhancer_prompt, sr.enhancer_prompt_default);
        _hydratePromptField('set-prompts-fanout',
            sr.fanout_system_prompt, sr.fanout_system_prompt_default);
        _hydratePromptField('set-prompts-fleet-user',
            sr.fleet_user_prompt_template, sr.fleet_user_prompt_template_default);
        _hydratePromptField('set-prompts-infinity',
            sr.infinity_user_prompt_template, sr.infinity_user_prompt_template_default);
        _hydratePromptField('set-prompts-chaos',
            sr.chaos_suggest_system_prompt, sr.chaos_suggest_system_prompt_default);
        _hydratePromptField('set-prompts-void',
            sr.void_fallback_template, sr.void_fallback_template_default);
        // Pre-stage the model_id on cfg-llm (Pipeline popup) so the
        // selector reflects the current choice when the popup opens
        // before reloadModels finishes the /settings/models fetch.
        const modelSel = $('cfg-llm');
        if (modelSel) {
            modelSel.dataset.selected = llm.model_id || '';
            if (llm.model_id) {
                modelSel.innerHTML = '';
                const o = document.createElement('option');
                o.value = llm.model_id; o.innerText = llm.model_id; o.selected = true;
                modelSel.appendChild(o);
            }
        }
        const bsel = $('set-branding');
        bsel.innerHTML = '';
        const profiles = (br && br.profiles) || [];
        profiles.forEach(name => {
            const o = document.createElement('option');
            o.value = name; o.innerText = name;
            if (name === (br && br.active)) o.selected = true;
            bsel.appendChild(o);
        });
        $('set-test-badge').className = 'badge badge-ghost font-mono text-xs';
        $('set-test-badge').innerText = 'idle';
        // Hydrate scheduler safety-margin slider from full settings payload.
        const safety = (sr.scheduler && sr.scheduler.memory_safety_gb) ?? 10;
        const safetyEl = $('sched-safety');
        if (safetyEl) {
            safetyEl.value = safety;
            const lbl = $('sched-safety-val'); if (lbl) lbl.innerText = safety;
        }
        const usePlanner = !!(sr.scheduler && sr.scheduler.use_planner);
        const planEl = $('sched-use-planner');
        if (planEl) planEl.checked = usePlanner;
        // CPU mode: 3-way "gpu" | "smart" | "cpu". Fall back to deriving
        // from the old llm_cpu_only boolean for config files that haven't
        // been re-saved since the migration.
        const sched = sr.scheduler || {};
        const llmMode = sched.llm_cpu_mode ||
            (sched.llm_cpu_only === false ? 'gpu' : sched.llm_cpu_only === true ? 'cpu' : 'smart');
        const ttsMode = sched.tts_cpu_mode ||
            (sched.tts_cpu_only === false ? 'gpu' : sched.tts_cpu_only === true ? 'cpu' : 'smart');
        // Hydrate a <select id="sched-llm-cpu-mode"> / <select id="sched-tts-cpu-mode">
        // if present, otherwise fall back to the legacy checkbox.
        const llmModeEl = $('sched-llm-cpu-mode');
        const ttsModeEl = $('sched-tts-cpu-mode');
        if (llmModeEl) llmModeEl.value = llmMode;
        if (ttsModeEl) ttsModeEl.value = ttsMode;
        // Legacy checkbox compat — keep it in sync with the mode.
        const llmEl = $('sched-llm-cpu-only');
        const ttsEl = $('sched-tts-cpu-only');
        if (llmEl) llmEl.checked = llmMode === 'cpu';
        if (ttsEl) ttsEl.checked = ttsMode === 'cpu';

        // Per-model loading-prefs grid is in the same Scheduler tab.
        if (typeof _hydrateModelLoadingPrefs === 'function') _hydrateModelLoadingPrefs(sr);
        // (Endless + Fresh prefs migrated to the Subjects-card Slop
        // Config modal — see openSlopConfig.)
        // Hydrate theme selector from localStorage (falling back to branding default).
        const themeSel = $('theme-select');
        if (themeSel) {
            const brandingDefault = (br && br.theme && br.theme.default) || themeSel.value;
            const current = localStorage.getItem('slopfinity-theme') || brandingDefault;
            if (current) {
                const opt = Array.from(themeSel.options).find(o => o.value === current);
                if (opt) themeSel.value = current;
            }
        }
        // Drawer is already open from the early _setSettingsOpen(true)
        // call above. Keep `.showModal()` as a defensive fallback so a
        // future re-introduction of the dialog form still works.
        if (typeof modal.showModal === 'function') {
            try { modal.showModal(); } catch (_) { /* not a dialog → drawer mode */ }
        }
        reloadModels();
    } catch (e) {
        console.error('openSettings failed', e);
    }
}

// ---------------------------------------------------------------------------
// Auto-suspend list UI — replaces PR #40's single LLM toggle. Each row is
// an entry from config.auto_suspend; users can enable/disable, change the
// suspension method, edit method-specific config, and add/remove entries.
// ---------------------------------------------------------------------------

const AUTO_SUSPEND_METHODS = [
    { value: 'sigstop', label: 'pause via SIGSTOP', fields: ['process_name'] },
    { value: 'rest_unload', label: 'unload via REST', fields: ['endpoint'] },
    { value: 'docker_stop', label: 'docker stop', fields: ['container'] },
    { value: 'sigterm', label: 'SIGTERM (one-shot)', fields: ['process_name'] },
    // The script method runs an arbitrary shell command on suspend.
    // The `command` text input is rendered by the renderAutoSuspendRow
    // logic below — empty string falls back to a hardcoded default.
    { value: 'script', label: 'shell script (custom)', fields: ['command'] },
];

// Cloud-endpoints filter for the LLM provider <select>. Toggled by the
// "Allow cloud LLM endpoints" checkbox (Settings → LLM). Today every
// option in the dropdown is a local provider, so this is a no-op gate;
// the wiring is in place so a future cloud entry tagged with
// data-provider-tier="cloud" gets hidden when the toggle is OFF.
// TODO(cloud-endpoints): once cloud providers land in slopfinity/llm/
// providers.py, add data-provider-tier="cloud" to the matching <option>
// elements in templates/index.html — this gate already enforces them.
function filterProviderDropdown(allow) {
    const sel = document.getElementById('set-provider');
    if (!sel) return;
    Array.from(sel.options).forEach(o => {
        const tier = o.getAttribute('data-provider-tier') || 'local';
        const hide = tier === 'cloud' && !allow;
        o.hidden = hide;
        o.disabled = hide;
    });
    // If the currently selected option got hidden, fall back to the first
    // visible (local) option so the form never submits a hidden cloud value.
    if (sel.options[sel.selectedIndex] && sel.options[sel.selectedIndex].hidden) {
        const firstVis = Array.from(sel.options).find(o => !o.hidden);
        if (firstVis) sel.value = firstVis.value;
    }
}

function autoSuspendMethodMeta(method) {
    return AUTO_SUSPEND_METHODS.find(m => m.value === method) || AUTO_SUSPEND_METHODS[0];
}

function renderAutoSuspendList(entries) {
    const host = $('auto-suspend-list');
    if (!host) return;
    host.innerHTML = '';
    const list = Array.isArray(entries) ? entries : [];
    list.forEach((e, i) => host.appendChild(renderAutoSuspendRow(e, i)));
}

function renderAutoSuspendRow(entry, idx) {
    const row = document.createElement('div');
    row.className = 'flex flex-wrap items-center gap-2 p-1 rounded border border-base-300/40';
    row.dataset.asIdx = String(idx);
    row.dataset.asId = entry.id || `entry-${idx}`;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'checkbox checkbox-xs checkbox-primary';
    cb.checked = !!entry.enabled;
    cb.dataset.asField = 'enabled';
    row.appendChild(cb);

    const label = document.createElement('span');
    label.className = 'text-xs flex-1 min-w-[10rem]';
    label.innerText = entry.label || entry.id || `Service ${idx + 1}`;
    row.appendChild(label);

    const sel = document.createElement('select');
    sel.className = 'select select-bordered select-xs';
    sel.dataset.asField = 'method';
    AUTO_SUSPEND_METHODS.forEach(m => {
        const o = document.createElement('option');
        o.value = m.value; o.innerText = m.label;
        if (m.value === entry.method) o.selected = true;
        sel.appendChild(o);
    });
    sel.onchange = () => {
        // Re-render this row so method-specific fields match the new method.
        const list = readAutoSuspendList();
        list[idx] = { ...list[idx], method: sel.value };
        renderAutoSuspendList(list);
    };
    row.appendChild(sel);

    // Method-specific config field(s).
    const meta = autoSuspendMethodMeta(entry.method);
    meta.fields.forEach(f => {
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'input input-bordered input-xs flex-1 min-w-[8rem] font-mono text-[11px]';
        inp.placeholder = f;
        inp.value = entry[f] || '';
        inp.dataset.asField = f;
        row.appendChild(inp);
    });

    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'btn btn-ghost btn-xs';
    del.innerText = '×';
    del.title = 'Remove';
    del.onclick = () => {
        const list = readAutoSuspendList();
        list.splice(idx, 1);
        renderAutoSuspendList(list);
    };
    row.appendChild(del);

    return row;
}

function readAutoSuspendList() {
    const host = $('auto-suspend-list');
    if (!host) return [];
    const out = [];
    host.querySelectorAll('[data-as-idx]').forEach(row => {
        const e = { id: row.dataset.asId };
        row.querySelectorAll('[data-as-field]').forEach(el => {
            const f = el.dataset.asField;
            if (el.type === 'checkbox') e[f] = !!el.checked;
            else e[f] = el.value;
        });
        // Preserve label by id where we can recognize it.
        const known = (window.__autoSuspendLabels || {})[e.id];
        if (known) e.label = known;
        out.push(e);
    });
    return out;
}

function addAutoSuspendEntry() {
    const list = readAutoSuspendList();
    const id = `service-${Date.now().toString(36)}`;
    list.push({
        id, label: 'New service', enabled: false,
        method: 'sigstop', process_name: '',
    });
    renderAutoSuspendList(list);
}

async function saveSettings() {
    // Model selector is in Pipeline popup now (cfg-llm). Fall back to
    // the live config snapshot when the Pipeline popup hasn't been opened
    // this session — connection-test should still work without forcing
    // the user to open Pipeline first.
    const sel = $('cfg-llm');
    let model_id = sel ? sel.value : '';
    if (model_id === '__custom__') {
        const c = $('cfg-llm-custom');
        model_id = c ? c.value.trim() : '';
    }
    if (!model_id) {
        model_id = (_lastTick && _lastTick.config && _lastTick.config.llm && _lastTick.config.llm.model_id) || '';
    }
    const body = {
        llm: {
            provider: $('set-provider').value,
            base_url: computeBaseUrl(),
            model_id,
            api_key: $('set-api-key').value,
            temperature: parseFloat($('set-temp').value),
            max_retries: parseInt($('set-retries').value, 10),
            timeout_s: parseInt($('set-timeout').value, 10),
        },
        auto_suspend: readAutoSuspendList(),
        allow_cloud_endpoints: $('set-allow-cloud-endpoints')
            ? !!$('set-allow-cloud-endpoints').checked
            : false,
        philosophical_prompt: $('set-fleet-prompt') ? $('set-fleet-prompt').value.trim() : null,
        suggest_use_subjects: $('set-suggest-use-subjects') ? $('set-suggest-use-subjects').checked : true,
        suggest_custom_prompt: $('set-suggest-custom-prompt') ? $('set-suggest-custom-prompt').value.trim() : '',
        suggest_per_row_prompts: $('set-suggest-per-row-prompts') ? $('set-suggest-per-row-prompts').checked : false,
        suggest_auto_disabled: !$('set-suggest-auto-enabled').checked,
        auto_suggest_enabled: $('set-suggest-auto-enabled').checked,
        idle_throttle_pct: parseInt($('set-idle-throttle').value, 10),
        creativity_score: parseInt($('set-creativity-score').value, 10),
        quality_score: parseInt($('cfg-quality-score').value, 10),
        concurrency_budget_gb: parseFloat($('set-concurrent-budget').value),
        when_idle: parseInt($('set-idle-throttle').value, 10) > 0,
        chaos_mode: parseInt($('set-creativity-score').value, 10) > 5,
        concurrent: parseFloat($('set-concurrent-budget').value) > 0,
        disk_min_pct: $('set-disk-min-pct') ? parseFloat($('set-disk-min-pct').value || '0') : 1,
        disk_min_gb: $('set-disk-min-gb') ? parseFloat($('set-disk-min-gb').value || '0') : 5,
    };
    // Prompts tab — collect every textarea verbatim. Empty string is
    // meaningful (server interprets it as "use built-in default").
    _collectPromptField(body, 'set-prompts-enhancer', 'enhancer_prompt');
    _collectPromptField(body, 'set-prompts-fanout', 'fanout_system_prompt');
    _collectPromptField(body, 'set-prompts-fleet-user', 'fleet_user_prompt_template');
    _collectPromptField(body, 'set-prompts-infinity', 'infinity_user_prompt_template');
    _collectPromptField(body, 'set-prompts-chaos', 'chaos_suggest_system_prompt');
    _collectPromptField(body, 'set-prompts-void', 'void_fallback_template');
    // Optimistically close the drawer FIRST so Save feels instant.
    // The POSTs run in background; if /settings or /branding fails, the
    // server-side state is unchanged but the user already sees the UI
    // collapse — far better UX than waiting on the round-trip with the
    // drawer frozen open. Was: close fired AFTER both awaits (~300-800ms
    // delay on a slow LLM-host loop), reading as "Save did nothing".
    const modal = $('settings-modal');
    if (modal && typeof modal.close === 'function') {
        try { modal.close(); } catch (_) { /* drawer mode → close via toggle */ }
    }
    if (typeof window._setSettingsOpen === 'function') window._setSettingsOpen(false);
    await fetch('/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    const bsel = $('set-branding');
    if (bsel && bsel.value) {
        await fetch('/branding', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ active: bsel.value }),
        });
    }
}

async function resetFleetPrompt() {
    // Clearing the textarea is sufficient — server treats empty string as
    // "use built-in default" and the next openSettings hydrate will leave
    // it blank too. Persist immediately so the user doesn't need to also
    // hit Save to commit the reset.
    const el = $('set-fleet-prompt');
    if (el) el.value = '';
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ philosophical_prompt: '' }),
        });
    } catch (e) {
        console.error('resetFleetPrompt failed', e);
    }
}

// ---------------------------------------------------------------------------
// Prompts tab helpers — hydrate / collect / reset for the textareas added in
// the Settings → Prompts panel. The server treats empty string as "use the
// built-in default" so reset is just `el.value = ''` + a POST.
// ---------------------------------------------------------------------------

function _hydratePromptField(elId, override, dflt) {
    const el = $(elId);
    if (!el) return;
    el.value = override || '';
    // Show the built-in default as a placeholder so the user can see what
    // they'd be replacing without first having to clear their override.
    if (typeof dflt === 'string' && dflt) {
        el.placeholder = dflt;
    }
}

function _collectPromptField(body, elId, key) {
    const el = $(elId);
    if (el) body[key] = el.value;
}

async function resetPromptField(elId, key) {
    const el = $(elId);
    if (el) el.value = '';
    try {
        const payload = {};
        payload[key] = '';
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
    } catch (e) {
        console.error('resetPromptField failed', e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const p = $('set-provider');
    if (p) p.addEventListener('change', applyProviderDefaults);
});

// -------------------- Single-page layout helpers --------------------

function openPipeline() {
    const d = document.getElementById('pipeline-modal');
    if (d && d.showModal) d.showModal();
    // Pre-select the active model_id so the dropdown reflects current state
    // before the /settings/models fetch completes.
    const sel = document.getElementById('cfg-llm');
    if (sel) {
        const cur = (_lastTick && _lastTick.config && _lastTick.config.llm && _lastTick.config.llm.model_id) || '';
        if (cur) sel.dataset.selected = cur;
    }
    if (typeof reloadModels === 'function') reloadModels();
}

function _subjectsFromTextarea() {
    const v = (document.getElementById('p-core') || {}).value || '';
    return v.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
}

// Generate-button behaviour:
// Each click ALWAYS queues a new job — there is no longer a global "Stop
// Infinity" mode. If the Infinity toggle is on, the queued item carries an
// `infinity:true` flag so the fleet re-appends it after every completion;
// multiple infinity items round-robin against each other. To stop one, click
// the ✕ on its queue row.
async function toggleInfinity() {
    const inf = document.getElementById('inf-on');
    const nowToggle = document.getElementById('now-on');
    const termToggle = document.getElementById('term-on');
    const concToggle = document.getElementById('concurrent-on');
    const idleToggle = document.getElementById('when-idle-on');
    const chaosToggle = document.getElementById('chaos-on');
    const prio = nowToggle && nowToggle.checked ? 'now' : 'next';
    const terminate = !!(termToggle && termToggle.checked);
    const concurrent = !!(concToggle && concToggle.checked);
    const infinity = !!(inf && inf.checked);
    const whenIdle = !!(idleToggle && idleToggle.checked);
    const chaos = !!(chaosToggle && chaosToggle.checked);
    await inject(prio, terminate, concurrent, { infinity, whenIdle, chaos });
    _updateStartBtn();
}

// Tracks whether the fleet is actually rendering right now (driven by the WS
// state broadcast). Toggling the Infinity checkbox alone doesn't mean we've
// started — the button label must reflect real state, not just the toggle.
let _isRendering = false;

// Compose the start-button label from every Generation toggle that affects
// the queued action. Pattern:
//   "[Terminate and ]Queue [Infinite ][Polymorphic ]Slop[ (now[, when idle])]"
// so the user can see — without opening the modal — exactly what the next
// click will do. Each toggle's onchange must call _updateStartBtn() so the
// label stays in sync.
function _composeStartBtnLabel() {
    const $ = (id) => document.getElementById(id);
    const term = !!($('term-on') && $('term-on').checked);
    const inf = !!($('inf-on') && $('inf-on').checked);
    const idle = !!($('when-idle-on') && $('when-idle-on').checked);
    const chaos = !!($('chaos-on') && $('chaos-on').checked);
    const now = !!($('now-on') && $('now-on').checked);

    let label = '';
    if (term) label += 'Terminate and ';
    label += 'Queue ';
    if (inf) label += 'Infinite ';
    if (chaos) label += 'Polymorphic ';
    label += 'Slop';

    const modifiers = [];
    if (now) modifiers.push('asap');
    if (idle) modifiers.push('when idle');
    if (modifiers.length) label += ' (' + modifiers.join(', ') + ')';
    return label;
}

function _updateStartBtn() {
    const b = document.getElementById('btn-start-stop');
    if (!b) return;
    // Each click queues a new item. The Infinity toggle in the Generation tab
    // makes the queued item re-loop after each completion (cancel via the ✕ on
    // its queue row). Never use this button to stop running jobs.
    b.textContent = _composeStartBtnLabel();
}

// Polymorphic + When Idle only make sense when the fleet is looping —
// gray them out otherwise.
// Terminate Existing only makes sense as part of a Generate-ASAP
// insert (it cancels the running job to make room for the new one).
// When ASAP is off, disable the Terminate input + grey its row, and
// force-uncheck it so a stale "true" can't bleed into the next click.
function _updateTermEnabled() {
    const now = document.getElementById('now-on');
    const term = document.getElementById('term-on');
    const row = document.getElementById('term-row');
    const enabled = !!(now && now.checked);
    if (term) {
        term.disabled = !enabled;
        if (!enabled && term.checked) {
            term.checked = false;
            term.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
    if (row) row.classList.toggle('opacity-60', !enabled);
}
window._updateTermEnabled = _updateTermEnabled;
document.addEventListener('DOMContentLoaded', () => {
    if (typeof _updateTermEnabled === 'function') _updateTermEnabled();
});

function _updateChaosEnabled() {
    // Polymorphic + When-Idle used to be force-disabled whenever Infinity
    // Mode was off, on the rationale that they "only make sense in a loop."
    // That tied Endless Story (a Subjects-card surface for the same flag)
    // visually to those toggles in a way users found confusing — flipping
    // Endless Story off would suddenly grey out Polymorphic / When-Idle.
    // Leaving them editable independently: they sit dormant when Infinity
    // is off, and take effect the moment a looping run starts. No more
    // surprise greying-out. Function kept as a stub so existing call sites
    // stay valid.
    const inf = document.getElementById('inf-on');
    const enabled = !!(inf && inf.checked);
    // Light visual hint (50% opacity on the ROW only, not disabled) so the
    // user still sees that those modifiers are inert without infinity, but
    // can click them in advance.
    ['chaos-row', 'when-idle-row'].forEach(rowId => {
        const r = document.getElementById(rowId);
        if (r) r.classList.toggle('opacity-60', !enabled);
    });
    // Make sure the toggles themselves are NOT disabled (clear any stale
    // state from before this fix landed).
    ['chaos-on', 'when-idle-on'].forEach(id => {
        const t = document.getElementById(id);
        if (t) t.disabled = false;
    });
}

// Back-compat shim — Terminate is now a flat flag; older callers may still
// invoke this. No-op preserves the call site.
function _updateTerminateEnabled() { }

// Build a one-line summary of the Generation tab toggles for the header pill,
// so the user can glance at it without opening the modal.
function _updateGenModePill() {
    const pill = document.getElementById('gen-mode-pill');
    if (!pill) return;
    const inf = document.getElementById('inf-on');
    const now = document.getElementById('now-on');
    const term = document.getElementById('term-on');
    const parts = [];
    parts.push(inf && inf.checked ? '♾ Infinity' : '▶ Single');
    if (term && term.checked) parts.push('🛑 terminate');
    else if (now && now.checked) parts.push('⏯ asap');
    else parts.push('queue');
    // +idle / +poly / +concurrent dropped — those are global settings now
    // (Diagnostics + Triggers tabs), not per-iteration knobs, so they
    // don't belong alongside the queue-mode chips.
    pill.textContent = parts.join(' · ');
}

// Single-label toggle pattern: each label sits to the right of the toggle and
// swaps text + emphasis based on the toggle state. data-on-label / data-off-label
// hold the words; bold + full-opacity when on, dim when off.
function _updateSingleLabels() {
    document.querySelectorAll('.single-label').forEach(el => {
        const t = document.getElementById(el.dataset.toggle);
        if (!t) return;
        const isOn = t.checked;
        const onText = el.dataset.onLabel || '';
        const offText = el.dataset.offLabel || onText;
        el.textContent = isOn ? onText : offText;
        el.classList.toggle('font-bold', isOn);
        el.classList.toggle('opacity-50', !isOn);
    });
}
// Back-compat shims — older onchange handlers may still reference these.
function _updateSideLabels() { _updateSingleLabels(); }
function _updatePrioLabel() { _updateSingleLabels(); _updateStartBtn(); }

// Toggle highlight on subject chips that are currently present in the
// Subjects textarea — re-clicking a present chip is a no-op (use Shift+click
// to replace). Called on textarea input + after each chip click.
function _refreshChipHighlights() {
    const ta = document.getElementById('p-core');
    if (!ta) return;
    const lines = new Set(ta.value.split(/\r?\n/).map(s => s.trim().toLowerCase()).filter(Boolean));
    // Walks every marquee row in the stack — chips appear duplicated in
    // each row's track for the seamless loop, so the same data-suggest may
    // map to multiple buttons; toggling them all keeps the visual in sync.
    const { stack } = _getSuggestStack();
    if (!stack) return;
    stack.querySelectorAll('button[data-suggest]').forEach(btn => {
        const here = lines.has((btn.dataset.suggest || '').toLowerCase());
        btn.classList.toggle('btn-primary', !here);
        btn.classList.toggle('btn-outline', here);
        btn.classList.toggle('btn-success', here);
        btn.classList.remove('opacity-60');
    });
}

// Cache key for the most-recent suggestion batch. Used to avoid hitting
// the LLM on every page load — auto-suggest renders from cache; the
// 🎲 Suggest button still fetches fresh and overwrites the cache.
const _SUGGEST_CACHE_KEY = 'slopfinity_suggestions_v1';

// Build a single suggestion chip <button>. Same click semantics as before
// the marquee rewrite: bare click appends, Shift+click replaces, present
// chips are no-ops. data-suggest carries the raw string so the highlight
// pass can match against the textarea content.
function _buildSuggestChip(s) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn btn-outline btn-primary btn-xs normal-case';
    b.textContent = s;
    b.title = 'Click: queue this prompt as a one-shot · Shift+click: append to the Prompt textarea';
    b.dataset.suggest = s;
    b.addEventListener('click', async (e) => {
        // Default click: inject the chip text as a single queue item — does
        // NOT touch the Subjects textarea. Shift+click: legacy behavior of
        // appending into Subjects (for the user who actually wants to build
        // an infinity-themes list manually).
        if (e.shiftKey) {
            const ta = document.getElementById('p-core');
            if (!ta) return;
            const present = ta.value.split(/\r?\n/).map(x => x.trim().toLowerCase()).includes(s.toLowerCase());
            if (present) return;
            ta.value = (ta.value.trim() ? ta.value.trimEnd() + '\n' : '') + s;
            ta.dispatchEvent(new Event('input', { bubbles: true }));
            updatePipeline();
            _refreshChipHighlights();
            return;
        }
        // One-shot inject: send this single prompt straight to the queue.
        try {
            const f = new FormData();
            f.append('prompt', s);
            f.append('priority', 'next');
            await fetch('/inject', { method: 'POST', body: f });
            // Endless mode: also append the accepted chip text to the
            // story-log pane so the user accumulates a high-level
            // story they can copy out before image/video rewrites.
            if (typeof _appendToEndlessLog === 'function') _appendToEndlessLog(s);
            // Kick a fresh suggestion fetch ASAP. Rationale: the LLM just
            // finished serving the batch this chip came from, so compute
            // is probably free this instant — start generating the next
            // batch in the background so it's ready before the user is
            // ready to pick again. Both branches respect the GPU-idle +
            // auto-suggest gates (silent no-op when compute is busy).
            try {
                const autoOff = (typeof window._autoSuggestDisabled === 'function')
                    && window._autoSuggestDisabled();
                const gpuOk = (typeof _isGpuIdleEnough !== 'function')
                    || _gpuPctHistory.length === 0
                    || _isGpuIdleEnough();
                if (!autoOff && gpuOk) {
                    const inEndless = (typeof _getSubjectsMode === 'function')
                        && _getSubjectsMode() === 'endless'
                        && (typeof _endlessRunning !== 'undefined' && _endlessRunning);
                    if (inEndless) {
                        // Regen THIS row with the new story tail so the next
                        // beat's continuations reflect the just-picked chip.
                        // Wait for the chip-disappear animation (~2.2 s) so
                        // the user sees pulse → collapse → fresh row arrive.
                        const rowEl = e.target.closest('.suggest-marquee-row');
                        const { stack } = _getSuggestStack();
                        const allRows = stack ? Array.from(stack.querySelectorAll('.suggest-marquee-row')) : [];
                        const rowIdx = rowEl ? allRows.indexOf(rowEl) : -1;
                        if (rowIdx >= 0 && typeof _regenEndlessRow === 'function') {
                            setTimeout(() => _regenEndlessRow(rowIdx), 2300);
                        }
                    } else if (typeof window._maybePrefetch === 'function') {
                        // Simple mode: top up the prefetch buffer. The
                        // existing idle-drain timer will swap a new row in
                        // when the user pauses — non-jarring.
                        window._maybePrefetch();
                    }
                }
            } catch (_) { /* kick-on-pick is best-effort, never block inject */ }
            // Two-phase exit: pulse-with-success-tint for ~1.4 s so the
            // user gets visual confirmation that this prompt is now in
            // the queue, then collapse over ~0.7 s. Both phases are
            // driven by CSS keyframes on the .chip-disappear class.
            // Trailing chips re-flow leftward filling the gap; the
            // marquee animation continues uninterrupted because the
            // track is still flex-row and translateX(-50%) still loops
            // the remaining content.
            const { stack: matchStack } = _getSuggestStack();
            const matches = matchStack ? matchStack.querySelectorAll(`button[data-suggest="${CSS.escape(s)}"]`) : [];
            matches.forEach(el => el.classList.add('chip-disappear'));
            // Snapshot which tracks need re-measuring AFTER removal — the
            // marquee duration is set as a CSS var based on track width, so
            // a narrower track (post-removal) translates fewer pixels per
            // second over the SAME duration → looks slower. Recompute.
            const tracksToReMeasure = new Set();
            matches.forEach(el => {
                const tr = el.closest('.suggest-marquee-track');
                if (tr) tracksToReMeasure.add(tr);
            });
            // Match the CSS animation total (1.4 s pulse + 0.7 s collapse
            // = 2.1 s; +100 ms slack so the collapse completes visually
            // before we yank the node).
            setTimeout(() => {
                matches.forEach(el => el.remove());
                tracksToReMeasure.forEach(track => {
                    const row = track.parentElement;
                    if (!row) return;
                    const tw = track.scrollWidth;
                    const rw = row.clientWidth;
                    if (tw / 2 <= rw) {
                        row.classList.add('no-overflow');
                        track.style.removeProperty('--marquee-duration');
                    } else {
                        row.classList.remove('no-overflow');
                        const px = tw / 2;
                        const seconds = Math.min(180, Math.max(40, px / 60));
                        track.style.setProperty('--marquee-duration', seconds + 's');
                    }
                });
            }, 2200);
        } catch (err) {
            console.warn('chip inject failed', err);
        }
    });
    return b;
}

// Hard cap on rows in the marquee stack — oldest row evicted (FIFO)
// when the Fresh toggle is on; new batches are dropped once full when
// Fresh is off so the user can pin a curated set on screen.
// Hard cap on rows in the marquee stack. Bumped from 5 → 50 — the user
// wants room to keep many rows pinned while curating (Fresh OFF mode).
// Fresh ON still FIFO-evicts oldest at the cap; the higher number just
// means evictions only happen at extreme depth.
const _SUGGEST_MAX_ROWS = 50;
function _isFreshMode() {
    try { return localStorage.getItem('slopfinity-fresh') === '1'; }
    catch (_) { return false; }
}

// Append a fresh marquee row to #subject-chips-stack. Items are duplicated
// (items + items) so the keyframes' translateX(-50%) yields a seamless
// loop. After insertion, measure the duplicated track width: if it fits
// inside the row, mark `.no-overflow` and skip the animation; otherwise
// scale the duration so a single chip travels at ~60 px/s, clamped to
// 40..180 s for the full track. Hover/focus-within pauses via CSS.
function _appendSuggestBatchRow(items, opts) {
    // Optional diagnostic for the "+ adds two rows" class of bugs. Enable by
    // setting localStorage['slopfinity-debug-rows']='1' in DevTools — every
    // entry logs a timestamped warn with opts + items.length + a stack so
    // you can see WHO double-fired. Off by default so production stays
    // quiet.
    try {
        if (typeof localStorage !== 'undefined'
            && localStorage.getItem('slopfinity-debug-rows') === '1') {
            console.warn('[debug-rows] _appendSuggestBatchRow', {
                t: Date.now(),
                items: (items && items.length) || 0,
                opts: opts || null,
            }, new Error('stack').stack);
        }
    } catch (_) { /* never block render on diagnostic */ }
    const { stack, placeholder } = _getSuggestStack();
    if (!stack || !items) return;
    // Empty items is allowed when opts has a promptId/rowIdx (endless
    // mode placeholder row — render the lead cluster + empty mask so
    // the row scaffold exists while the fetch is in flight; chips
    // arrive later via _addEndlessRow's swap). Without this branch we
    // had to pass [' '] to dodge the guard, which the marquee
    // duplicator turned into [' ', ' '] = two visible empty chips.
    const hasLeadOpts = opts && typeof opts.rowIdx === 'number' && opts.promptId;
    if (!items.length && !hasLeadOpts) return;
    // First batch — drop the placeholder span if present.
    if (placeholder) placeholder.remove();

    // Fresh OFF + at cap → drop the new batch instead of evicting old
    // ones. This lets the user pin a curated set on screen. Fresh ON
    // (default once user opts in) keeps the FIFO eviction below.
    if (!_isFreshMode() && stack.children.length >= _SUGGEST_MAX_ROWS) {
        return;
    }

    const row = document.createElement('div');
    const isEndless = (typeof _getSubjectsMode === 'function') && _getSubjectsMode() === 'endless';
    const isSimple = (typeof _getSubjectsMode === 'function') && _getSubjectsMode() === 'simple';
    const spiffySimple = isSimple && (typeof _isPerRowPromptsEnabled === 'function') && _isPerRowPromptsEnabled();
    const hasLead = (isEndless || spiffySimple) && opts && typeof opts.rowIdx === 'number' && opts.promptId;
    row.className = 'suggest-marquee-row entering' + (hasLead ? ' with-lead' : '');
    // Endless mode rows get a leading prompt-name chip + per-row refresh
    // icon. Click chip → popover picker scoped to THIS row's prompt; click
    // refresh → re-fetch this row only via _regenEndlessRow(rowIdx).
    if (hasLead) {
        const promptObj = (typeof _getPromptById === 'function') ? _getPromptById(opts.promptId) : null;
        const ttl = promptObj ? promptObj.title : opts.promptId;
        const lead = document.createElement('div');
        // Right-aligned cluster (CSS: order:2 to flip past the mask) with
        // a fold-toggle button visible only at compact viewports. The
        // three controls (prompt-name / refresh / minus) collapse on
        // compact + expand inline when the fold button is clicked.
        lead.className = 'endless-row-lead flex items-center gap-1 flex-none';
        lead.setAttribute('data-endless-row-lead', String(opts.rowIdx));
        lead.innerHTML = `
            <button type="button" class="endless-row-fold btn btn-ghost btn-xs btn-square"
                data-row-fold
                title="Show row controls"
                onclick="this.parentElement.classList.toggle('lead-expanded')">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round"
                    stroke-linejoin="round" class="w-3 h-3">
                    <circle cx="12" cy="5" r="1.5"/>
                    <circle cx="12" cy="12" r="1.5"/>
                    <circle cx="12" cy="19" r="1.5"/>
                </svg>
            </button>
            <div class="endless-row-controls flex items-center gap-1">
                <button type="button" class="btn btn-ghost btn-xs gap-1 px-1.5"
                    data-endless-row-prompt-btn="${opts.rowIdx}"
                    data-row-prompt-btn
                    title="Click to swap this row's prompt"
                    onclick="_openSuggestPromptPicker(${opts.rowIdx})">
                    <span class="font-semibold text-[10px]" data-row-prompt-label>${_htmlEscape(ttl)}</span>
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round"
                        stroke-linejoin="round" class="w-2.5 h-2.5">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                </button>
                <button type="button" class="btn btn-ghost btn-xs btn-square"
                    data-row-refresh
                    title="Refresh this row" onclick="_regenEndlessRow(${opts.rowIdx})">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round"
                        stroke-linejoin="round" class="w-3 h-3">
                        <path d="M21 12a9 9 0 0 0-15-6.7L3 8"/>
                        <path d="M3 3v5h5"/>
                        <path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"/>
                        <path d="M21 21v-5h-5"/>
                    </svg>
                </button>
                <button type="button" class="btn btn-ghost btn-xs btn-square opacity-60 hover:opacity-100 hover:text-error"
                    data-row-remove
                    title="Remove this row" onclick="_removeEndlessRow(${opts.rowIdx})">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2.4" stroke-linecap="round"
                        stroke-linejoin="round" class="w-3 h-3">
                        <line x1="5" y1="12" x2="19" y2="12"/>
                    </svg>
                </button>
            </div>`;
        row.appendChild(lead);
    }
    const track = document.createElement('div');
    track.className = 'suggest-marquee-track';
    const all = [...items, ...items]; // duplicate for seamless wraparound
    all.forEach(s => track.appendChild(_buildSuggestChip(s)));
    if (hasLead) {
        // Wrap the track in a masked viewport so the row's lead cluster
        // stays fully opaque + interactive while the marquee fades at
        // its own edges within its own clipping context.
        const mask = document.createElement('div');
        mask.className = 'suggest-marquee-mask';
        mask.appendChild(track);
        row.appendChild(mask);
    } else {
        row.appendChild(track);
    }
    // ── FLIP (First-Last-Invert-Play) — Option C of the marquee bump fix.
    // Snapshot every existing row's bounding rect BEFORE the insertion,
    // then perform the insertion, then snapshot AFTER. For any row whose
    // top moved, apply an inverse translateY so it visually starts at its
    // OLD position, then animate to translateY(0). The browser still
    // performed the layout shift, but the user only sees a smooth
    // animation — no abrupt bump above the new row. Safe-guarded against
    // browsers without getBoundingClientRect (impossible in practice but
    // catch keeps us defensive). */
    const _flipBefore = new Map();
    try {
        stack.querySelectorAll('.suggest-marquee-row').forEach((r) => {
            _flipBefore.set(r, r.getBoundingClientRect().top);
        });
    } catch (_) { /* FLIP best-effort */ }

    // Honor opts.insertAtIdx — used by _regenEndlessRow to put the
    // refreshed row back in its ORIGINAL position rather than appending
    // it at the end of the stack. Without this, swapping a row's
    // suggestion-prompt via the picker silently moved that row to the
    // bottom, which read as "the dropdown change didn't take effect".
    if (opts && typeof opts.insertAtIdx === 'number') {
        const existing = stack.querySelectorAll('.suggest-marquee-row');
        const before = existing[opts.insertAtIdx] || null;
        stack.insertBefore(row, before);
    } else {
        stack.appendChild(row);
    }

    // FLIP — apply inverse transforms to existing rows that moved.
    try {
        _flipBefore.forEach((oldTop, r) => {
            if (!r.isConnected) return;            // row was removed
            const newTop = r.getBoundingClientRect().top;
            const dy = oldTop - newTop;
            if (Math.abs(dy) < 0.5) return;        // didn't move — skip
            // Step 1: snap to the old position with no transition.
            r.style.transition = 'none';
            r.style.transform = `translateY(${dy}px)`;
            // Step 2: next frame — clear transform with a transition so
            // the row glides to its new position. Using two rAFs to make
            // sure the no-transition snap commits before the easing kicks.
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    r.style.transition = 'transform 380ms cubic-bezier(0.22, 0.61, 0.36, 1)';
                    r.style.transform = '';
                });
            });
            // Cleanup: drop the inline transition once the slide finishes
            // so it doesn't interfere with future hover/marquee animations.
            const cleanup = (e) => {
                if (e.propertyName !== 'transform') return;
                r.style.transition = '';
                r.removeEventListener('transitionend', cleanup);
            };
            r.addEventListener('transitionend', cleanup);
        });
    } catch (_) { /* FLIP best-effort */ }
    // Drop the .entering class once the slide-in completes so future
    // hover/focus rules apply normally and the transform doesn't linger
    // as inline animation state.
    row.addEventListener('animationend', e => {
        if (e.animationName === 'suggest-row-enter') row.classList.remove('entering');
    }, { once: true });

    requestAnimationFrame(() => {
        const tw = track.scrollWidth;
        const rw = row.clientWidth;
        if (tw / 2 <= rw) {
            row.classList.add('no-overflow');
        } else {
            const px = tw / 2; // single-copy width
            // Speed math: aim for ~60 px/s on average. Clamp to 40..180 s
            // so very short overflows don't sprint and very long ones
            // don't crawl. seconds = clamp(px / 60, 40, 180).
            const seconds = Math.min(180, Math.max(40, px / 60));
            track.style.setProperty('--marquee-duration', seconds + 's');
        }
        _refreshChipHighlights();
    });

    // FIFO eviction — drop oldest rows past the cap. Only runs when
    // Fresh is on (when off, the cap-check above already short-circuits
    // before we reach here, so children.length stays ≤ cap).
    if (_isFreshMode()) {
        while (stack.children.length > _SUGGEST_MAX_ROWS) {
            stack.removeChild(stack.firstElementChild);
        }
    }
}

// Wholesale replace the stack with a single fresh row (cache hydrate,
// 🎲 click). Additive use cases (auto-suggest top-up, prefetch consume)
// call _appendSuggestBatchRow directly.
function _renderSuggestChips(arr) {
    const { stack, placeholder } = _getSuggestStack();
    if (!stack) return;
    stack.innerHTML = '';
    if (!arr.length) {
        const span = document.createElement('span');
        span.className = 'text-[10px] italic text-warning';
        span.textContent = 'no suggestions';
        stack.appendChild(span);
        return;
    }
    _appendSuggestBatchRow(arr);
}

// Render cached suggestions from localStorage if any exist. Returns true
// if it rendered, false if cache was empty.
//
// Caching is SIMPLE-MODE ONLY. Endless beats are story-state-dependent
// (the seed and prior beats matter); chat replies are conversation-
// state-dependent (the assistant's last turn matters); raw doesn't use
// suggestions at all. Serving a stale cached batch in any of those
// modes would render confusingly out-of-context chips. Only simple
// mode — where suggestions are evergreen "ideas matching your
// subject text" — benefits from cross-session persistence.
function _renderCachedSuggestions() {
    const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
    if (mode !== 'simple') return false;
    try {
        const raw = localStorage.getItem(_SUGGEST_CACHE_KEY);
        if (!raw) return false;
        const arr = JSON.parse(raw);
        if (!Array.isArray(arr) || !arr.length) return false;
        _renderSuggestChips(arr);
        return true;
    } catch { return false; }
}

// 🎲 Suggest button entry point. INTENTIONALLY NOT gated by
// _isGpuIdleEnough() — manual user click always wins. See the audit
// block above _isGpuIdleEnough for the full inventory.
// Single fetch helper used by every mode-specific branch. fresh=1 tells
// the server to bypass cache + nudge the LLM with a salt so successive
// calls actually differ. promptId picks ONE named prompt's system text.
async function _fetchSuggestBatch(opts) {
    const o = opts || {};
    const subjects = (($('p-core') && $('p-core').value) || '').trim();
    const qs = '?n=' + (o.n || 6)
        + (subjects ? '&subjects=' + encodeURIComponent(subjects) : '')
        + (o.promptId ? '&prompt_id=' + encodeURIComponent(o.promptId) : '')
        + (o.fresh ? '&fresh=1&_t=' + Date.now() : '');
    try {
        const r = await fetch('/subjects/suggest' + qs);
        const data = await r.json();
        const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
        const dict = (data && data.suggestions) || {};
        const arr = mode === 'endless' ? (dict.story || []) : (dict.simple || []);
        // Filter out anything that looks like an error string from the
        // LLM / HTTP layer leaking through as a "suggestion". Errors
        // are NOT suggestions — they shouldn't render as clickable
        // chips. Pattern: leading "Error", "HTTP <code>", "timeout",
        // "timed out", "internal server error", or strings that are
        // wholly enclosed in <error>…</error>-like markers.
        // Suggestion chips should be SHORT, evergreen prompts the user
        // can click to queue. Anything that looks like an error response,
        // an LLM scaffolding leak (markdown headers, system-prompt
        // restatements, numbered list intros), or a generic instruction
        // gets filtered. Errors are not suggestions.
        const looksLikeJunk = (s) => {
            if (typeof s !== 'string') return true;
            const t = s.trim();
            if (!t) return true;
            // Length sanity — real suggestions are ≤ 120 chars.
            if (t.length > 200) return true;
            // Obvious error markers.
            if (/^error[:\s]/i.test(t)) return true;
            if (/^http\s*[345][0-9]{2}/i.test(t)) return true;  // 3xx/4xx/5xx
            if (/\btimed?\s*out\b/i.test(t)) return true;
            if (/^internal\s*server\s*error/i.test(t)) return true;
            if (/^<error/i.test(t)) return true;
            if (/^\(empty\)$/i.test(t)) return true;
            // Markdown / formatting leaks — LLM is repeating its own
            // scaffolding instead of giving us a chip.
            if (/^\*\*/.test(t)) return true;            // **bold header**
            if (/^#{1,6}\s/.test(t)) return true;         // # markdown heading
            if (/^[-*]\s+\*\*/.test(t)) return true;      // - **bold list item
            if (/^\d+[.)]\s+\*\*/.test(t)) return true;   // 1. **bold numbered**
            // System-prompt restatements — the LLM is echoing the
            // instructions back. These are giveaways.
            if (/\bconstraint\s*check/i.test(t)) return true;
            if (/^generated\s+concepts/i.test(t)) return true;
            if (/^need\s+\d+\s+(short|visual|prompt)/i.test(t)) return true;
            if (/^plain\s+text\s+only/i.test(t)) return true;
            if (/\bsystem\s*prompt\b/i.test(t)) return true;
            if (/\beach\s+idea\s+must\b/i.test(t)) return true;
            if (/\bideas?\s+for\s+(ai|llm)\b/i.test(t)) return true;
            if (/\bfleet\s*concepts?\b/i.test(t)) return true;
            if (/^output:?\s*$/i.test(t)) return true;
            if (/^(here\s+(are|is)|i'?ll|i\s+will|let\s+me)\s+/i.test(t)) return true;
            return false;
        };
        const unique = [];
        const seen = new Set();
        arr.forEach(s => {
            if (!s || typeof s !== 'string') return;
            const t = s.trim();
            const low = t.toLowerCase();
            if (seen.has(low)) return;
            if (looksLikeJunk(t)) return;
            seen.add(low);
            unique.push(t);
        });
        return unique;
    } catch (_) { return []; }
}
window._fetchSuggestBatch = _fetchSuggestBatch;

// Per-mode regen entry. Branches on subjects mode.
//   raw     — bail (suggestions skipped entirely)
//   simple  — N rows (default 3) all using the user's default prompt_id
//   endless — N rows, EACH using its own prompt_id from _getEndlessRowPrompts
//             (per-row label clickable to swap → row regenerates)
//   chat    — 4 reply chips, no marquee, derived from chat history
async function regenSuggestions(n = 6) {
    if (!(await _ramGuardCheck())) return;
    // Clicking refresh while the Suggestions toggle is OFF auto-enables
    // it — the user clearly wants suggestions, even if the toggle was
    // turned off earlier. Saves them an extra click on the badge.
    const sugInput = document.getElementById('subjects-suggestions-toggle-input');
    if (sugInput && !sugInput.checked && typeof toggleSuggestionsHidden === 'function') {
        toggleSuggestionsHidden(false);
    }
    const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
    const stackArea = document.getElementById('subjects-suggestions-area');
    if (mode === 'raw') {
        // Hide the suggestions area entirely; raw mode is LLM-free.
        if (stackArea) stackArea.classList.add('hidden');
        return;
    }
    if (stackArea) stackArea.classList.remove('hidden');
    // Spin the unified-badge refresh icon CONTINUOUSLY while the fetch
    // is in flight — matches the per-row refresh treatment in endless.
    // The 700 ms tap-spin from the click delegate is just the click
    // ack; this is the actual loading affordance for simple/chat modes.
    const refreshSvg = document.querySelector('#subjects-suggest-btn svg');
    if (refreshSvg) refreshSvg.classList.add('refresh-spinning');
    try {
        if (mode === 'chat') return await _renderChatReplies();
        const { stack: box } = _getSuggestStack();
        if (!box) return;
        if (mode === 'endless') return await _renderEndlessRows(n);
        return await _renderSimpleRows(n);
    } finally {
        if (refreshSvg) refreshSvg.classList.remove('refresh-spinning');
    }
}
window.regenSuggestions = regenSuggestions;

// SIMPLE mode — 3 rows, all of the user's default prompt. Server-cache
// hits row 1 → row 2/3 use fresh=1 to get genuinely different batches.
//
// Mode-check at every await boundary: if the user swaps modes mid-fetch,
// the in-flight result must not paint into the new mode's stack (would
// leak as 'naked' simple-mode rows in endless / chat). We capture the
// mode at start and bail if it changes.
async function _renderSimpleRows(n) {
    const startMode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
    const stillSimple = () => (typeof _getSubjectsMode === 'function')
        ? _getSubjectsMode() === startMode : true;
    const { stack: box } = _getSuggestStack();
    if (!box) return;
    const promptId = _getDefaultPromptId();
    // SWAP-ON-SUCCESS: only clear the existing chips AFTER the new
    // batch lands. Previously we dropped a spinner into the box first,
    // so a slow refresh erased the user's current chips for several
    // seconds. Spinner now lives only when the box is empty (first
    // load); existing chips persist + get a dim-while-loading class
    // so the user knows they're stale.
    // Snapshot the existing row count BEFORE we touch the box. If the
    // user has manually added rows via the +/− pill (so the current row
    // count > the localStorage initial-rows default), we want regen to
    // PRESERVE that count rather than collapsing back to the default.
    // Specifically: switching the suggestion prompt fires regenSuggestions
    // which lands here — without this, going from 7 rows back to 3 every
    // prompt swap is a UX trap the user reported.
    const _existingRowCount = box.querySelectorAll('.suggest-marquee-row').length;
    const hasExistingChips = _existingRowCount > 0;
    if (!hasExistingChips) {
        box.innerHTML = '<span class="loading loading-dots loading-xs"></span>';
    } else {
        box.classList.add('row-loading');
    }
    // fresh:true on the FIRST batch too — clicking ↻ refresh should
    // get a genuinely new batch from the LLM, not a server-cache hit
    // from minutes ago. Server cache fired between consecutive ↻
    // clicks because we passed fresh:false here, so the user saw
    // identical chips no matter how many times they refreshed.
    const arr = await _fetchSuggestBatch({ n, promptId, fresh: true });
    if (!stillSimple()) { box.classList.remove('row-loading'); return; }
    box.classList.remove('row-loading');
    if (!arr.length) {
        if (!hasExistingChips) {
            box.innerHTML = '<span class="text-[10px] italic text-warning">LLM unreachable</span>';
        }
        // If we DID have chips before, keep them — empty fetch shouldn't
        // wipe the user's current view.
        return;
    }
    try { localStorage.setItem(_SUGGEST_CACHE_KEY, JSON.stringify(arr)); } catch { }
    _renderSuggestChips(arr);
    // Initial-rows count is configurable in Settings (default 3, clamped
    // 1..10). Persisted in localStorage as 'slopfinity-simple-initial-rows'.
    // The +/- pill still adds/removes rows manually after the initial
    // render — this just sets the starting depth.
    let TARGET_ROWS = 3;
    try {
        const stored = parseInt(localStorage.getItem('slopfinity-simple-initial-rows') || '3', 10);
        if (Number.isFinite(stored)) TARGET_ROWS = Math.max(1, Math.min(10, stored));
    } catch (_) { }
    for (let i = 1; i < TARGET_ROWS; i++) {
        await new Promise(r => setTimeout(r, 800));
        if (!stillSimple()) return;
        const t = _lastTick && _lastTick.stats ? _lastTick.stats : {};
        const ramT = t.ram_t || 0;
        const ramPct = ramT > 0 ? Math.round((t.ram_u / ramT) * 100) : 0;
        if ((t.gpu || 0) >= 80 || ramPct >= 80 || (t.load_pct || 0) >= 80) break;
        const more = await _fetchSuggestBatch({ n, promptId, fresh: true });
        if (!stillSimple()) return;
        if (more && more.length) _appendSuggestBatchRow(more);
    }
}

// ENDLESS mode — one row per entry in _getEndlessRowPrompts; each row
// has a clickable prompt-name chip at its start (label inside
// _appendSuggestBatchRow) so the user can swap that row's prompt and
// trigger a re-fetch.
//
// Endless suggestions ONLY run while a story is active. Pre-start the
// chip-stack just shows a hint so we don't burn LLM cycles on a seed
// that hasn't been chosen yet.
async function _renderEndlessRows(n) {
    // Mode-check guard: if user swaps to non-endless mid-fetch, bail
    // before painting (otherwise endless rows leak into simple/chat).
    const stillEndless = () => (typeof _getSubjectsMode === 'function')
        && _getSubjectsMode() === 'endless' && _endlessRunning;
    const { stack: box } = _getSuggestStack();
    if (!box) return;
    if (!_endlessRunning) {
        box.innerHTML = "";
        return;
    }
    box.innerHTML = '<div class="flex items-center gap-2 text-[11px] text-base-content/70"><span class="loading loading-spinner loading-sm text-primary"></span><span>Generating story beats…</span></div>';
    const rowPrompts = _getEndlessRowPrompts();
    if (!rowPrompts.length) {
        box.innerHTML = '<span class="text-[10px] italic text-warning">No active suggestion prompts. Add one in Settings → LLM.</span>';
        return;
    }
    box.innerHTML = ''; // wipe spinner; rows append below
    for (let i = 0; i < rowPrompts.length; i++) {
        const t = _lastTick && _lastTick.stats ? _lastTick.stats : {};
        const ramT = t.ram_t || 0;
        const ramPct = ramT > 0 ? Math.round((t.ram_u / ramT) * 100) : 0;
        if (i > 0 && ((t.gpu || 0) >= 80 || ramPct >= 80 || (t.load_pct || 0) >= 80)) break;
        const promptId = rowPrompts[i];
        // ALL endless rows pass fresh:true. Story beats depend on the
        // current seed + prior beats, so a server-cache hit from an
        // earlier session would be off-context. Cache is reserved for
        // simple mode (see _renderCachedSuggestions doc).
        const arr = await _fetchSuggestBatch({ n, promptId, fresh: true });
        if (!stillEndless()) return;  // mode swap during fetch — abandon
        if (arr && arr.length) _appendSuggestBatchRow(arr, { promptId, rowIdx: i });
        if (i < rowPrompts.length - 1) {
            await new Promise(r => setTimeout(r, 700));
            if (!stillEndless()) return;
        }
    }
}

// Re-fetch a SINGLE endless row in place — invoked by the picker after the
// user swaps that row's prompt. Replaces the row's chip content; rest of
// the marquee stack is untouched.
//
// CRITICAL: per-row inflight guard. Without this, rapid chip-clicks on the
// same row schedule multiple `setTimeout(() => _regenEndlessRow(N), 2300)`
// calls (chip click handler in _buildSuggestChip). Each runs independently:
// both look up `rows[N]` BEFORE either has finished fetching, both call
// `row.remove()`, both then `_appendSuggestBatchRow(..., insertAtIdx: N)`.
// The second one's `existing[N]` lookup catches the first one's just-
// inserted row and `insertBefore`s a SECOND new row in front of it ⇒ ghost
// rows accumulate per click. A small per-index Set blocks the race.
const _endlessRegenInflight = new Set();
async function _regenEndlessRow(rowIdx) {
    if (_endlessRegenInflight.has(rowIdx)) return;
    _endlessRegenInflight.add(rowIdx);
    try {
        return await _regenEndlessRowImpl(rowIdx);
    } finally {
        _endlessRegenInflight.delete(rowIdx);
    }
}
async function _regenEndlessRowImpl(rowIdx) {
    const { stack, placeholder } = _getSuggestStack();
    if (!stack) return;
    const rowPrompts = _getEndlessRowPrompts();
    const promptId = rowPrompts[rowIdx];
    if (!promptId) return;
    const rows = stack.querySelectorAll('.suggest-marquee-row');
    const row = rows[rowIdx];
    // Spin the row's refresh icon + swap chips for a loading shimmer
    // INSIDE the marquee mask so the lead cluster (subject/refresh/minus)
    // stays interactive and visible while the new batch loads.
    const refreshBtn = row ? row.querySelector('[data-row-refresh]') : null;
    if (refreshBtn) refreshBtn.classList.add('row-refresh-spinning');
    const mask = row ? row.querySelector('.suggest-marquee-mask') : null;
    // The spinning refresh button is the loading affordance — don't double
    // up with a "regenerating…" label inside the mask. Just dim the
    // existing chips so they read as "stale" without yanking the row's
    // visual structure away from the user. If there are no existing
    // chips (shouldn't happen on regen but be safe), the .row-loading
    // class lets CSS hold a min-height.
    if (mask) {
        mask.classList.add('row-loading');
    } else if (row) {
        row.classList.add('row-loading');
    }
    let arr = [];
    try {
        arr = await _fetchSuggestBatch({ n: 6, promptId, fresh: true });
    } finally {
        if (refreshBtn) refreshBtn.classList.remove('row-refresh-spinning');
    }
    if (mask) mask.classList.remove('row-loading');
    if (row) row.classList.remove('row-loading');
    if (!arr.length) {
        if (mask) mask.innerHTML = '<span class="text-[10px] italic text-warning px-2">empty batch</span>';
        else if (row) row.innerHTML = '<span class="text-[10px] italic text-warning px-2">empty batch</span>';
        return;
    }
    // Re-render the row by removing it + appending fresh; preserves order
    // by re-walking from rowIdx and replacing only that one.
    if (row) row.remove();
    _appendSuggestBatchRow(arr, { promptId, rowIdx, insertAtIdx: rowIdx });
}
window._regenEndlessRow = _regenEndlessRow;

// Add a new endless row using the currently-selected default prompt id.
// Wired from the unified badge's "+" button when in endless mode.
//
// UX: render the row scaffold (subject chip + refresh + minus) IMMEDIATELY
// with a spinning refresh button so the user gets instant feedback that
// the click landed. Then await the batch and swap the chips in. Without
// this the badge appears to do nothing for 2-10 s while the LLM thinks.
let _addEndlessRowInflight = false;
async function _addEndlessRow() {
    if (_addEndlessRowInflight) return;
    _addEndlessRowInflight = true;
    try {
        const arr = _getEndlessRowPrompts();
        const newId = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : 'yes-and';
        arr.push(newId);
        _setEndlessRowPrompts(arr);
        const idx = arr.length - 1;
        // Append a placeholder row with the lead cluster + an EMPTY mask
        // (no chips — _appendSuggestBatchRow allows items=[] when opts
        // has promptId/rowIdx). The row scaffold exists in the DOM before
        // the fetch resolves; .row-loading on the mask preserves height
        // and dims the (empty) area; the spinning refresh icon is the
        // actual loading affordance. Previously passed [' '] which the
        // marquee duplicator turned into [' ', ' '] = two visible empty
        // chips next to the lead cluster.
        _appendSuggestBatchRow([], { promptId: newId, rowIdx: idx });
        const { stack, placeholder } = _getSuggestStack();
        // BUG FIX: previously looked up the placeholder by `[idx]` where
        // idx came from the prompt-array length. If the prompt array
        // and DOM row count had drifted (e.g. a dead-code endless cycle
        // had pushed a prompt without rendering, or a prior render
        // failed mid-flight), idx pointed at the WRONG row. mask/oldTrack
        // came back null, the fallback at line 9430 ran, and a SECOND
        // _appendSuggestBatchRow call appended the real batch — leaving
        // the empty placeholder visible AND adding a fresh row beneath:
        // "when I add a row it adds two, but one is empty". The
        // placeholder is always appended LAST (no insertAtIdx), so look
        // it up by last-child instead of by index. Stays correct even
        // when the prompt array and DOM disagree.
        const allRows = stack ? stack.querySelectorAll('.suggest-marquee-row') : [];
        const row = allRows.length ? allRows[allRows.length - 1] : null;
        const refreshBtn = row ? row.querySelector('[data-row-refresh]') : null;
        const mask = row ? row.querySelector('.suggest-marquee-mask') : null;
        if (refreshBtn) refreshBtn.classList.add('row-refresh-spinning');
        if (mask) mask.classList.add('row-loading');
        try {
            const batch = await _fetchSuggestBatch({ n: 6, promptId: newId, fresh: true });
            if (batch && batch.length) {
                // Swap the placeholder content in-place: rebuild the marquee
                // track with the real chips, leaving the lead cluster intact.
                const oldTrack = mask ? mask.querySelector('.suggest-marquee-track') : null;
                if (mask && oldTrack) {
                    const newTrack = document.createElement('div');
                    newTrack.className = 'suggest-marquee-track';
                    [...batch, ...batch].forEach(s => newTrack.appendChild(_buildSuggestChip(s)));
                    mask.replaceChild(newTrack, oldTrack);
                } else {
                    // Fallback: row scaffold missing — drop & re-append fresh.
                    // Guarded so we don't double-append when the placeholder
                    // exists but only the inner mask/track went missing for
                    // some other reason.
                    if (row) row.remove();
                    _appendSuggestBatchRow(batch, { promptId: newId, rowIdx: idx });
                }
            }
        } finally {
            if (refreshBtn) refreshBtn.classList.remove('row-refresh-spinning');
            if (mask) mask.classList.remove('row-loading');
        }
    } finally {
        _addEndlessRowInflight = false;
    }
}
window._addEndlessRow = _addEndlessRow;

// Router for the "+" badge button. Dispatches by current mode:
//   endless → _addEndlessRow (append a new story-beat row)
//   simple  → first-time bootstrap: regenSuggestions to render the
//             initial chip stack. After that the badge swaps to ↻
//             refresh (see _refreshSuggestBadge), so this path only
//             fires on the very first click in a fresh session.
//   other   → fall back to regenSuggestions (chat replies).
function _addOrFirstSuggestion() {
    const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
    if (mode === 'endless') return _addEndlessRow();
    if (typeof regenSuggestions === 'function') {
        return regenSuggestions().then(() => {
            // Repaint the badge so the + → ↻ swap happens immediately
            // once the first batch lands. Without this the user has to
            // change modes (or trigger another _refreshSuggestBadge) to
            // see the affordance flip.
            if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
        });
    }
}
window._addOrFirstSuggestion = _addOrFirstSuggestion;

// Simple mode row-stack controls (separate from the Suggestions ↻
// refresh button). + adds one fresh row to the BOTTOM of the stack;
// − removes the bottom row. Both honor the current default prompt id.
// _refreshSuggestBadge toggles the cluster's visibility based on mode.
async function _addSimpleRow() {
    const promptId = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : 'yes-and';
    // FAST PATH: if the prefetch buffer has a batch tagged for THIS
    // promptId, render it instantly — the LLM round-trip already
    // happened in the background while the user was reading the prior
    // row. Then kick another prefetch so the NEXT + click is also
    // instant. Stale batches (different promptId) are dropped inside
    // _consumePrefetchedBatch, not handed to us.
    if (typeof window._consumePrefetchedBatch === 'function') {
        const pre = window._consumePrefetchedBatch(promptId);
        if (pre && pre.length && typeof _appendSuggestBatchRow === 'function') {
            _appendSuggestBatchRow(pre);
            if (typeof window._maybePrefetch === 'function') window._maybePrefetch();
            if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
            return;
        }
    }
    // SLOW PATH: nothing buffered (cold start, GPU was busy, prompt
    // just swapped, etc.) — fetch fresh and also kick prefetch so the
    // next click can hit the fast path.
    const arr = await _fetchSuggestBatch({ n: 6, promptId, fresh: true });
    if (arr && arr.length && typeof _appendSuggestBatchRow === 'function') {
        _appendSuggestBatchRow(arr);
    }
    if (typeof window._maybePrefetch === 'function') window._maybePrefetch();
    if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
}
function _removeSimpleRow() {
    const { stack, placeholder } = _getSuggestStack();
    if (!stack) return;
    const rows = stack.querySelectorAll('.suggest-marquee-row');
    if (rows.length) rows[rows.length - 1].remove();
    if (typeof _refreshSuggestBadge === 'function') _refreshSuggestBadge();
}
window._addSimpleRow = _addSimpleRow;
window._removeSimpleRow = _removeSimpleRow;

// Remove the row at rowIdx — drops its prompt from the persisted array
// and removes its DOM node. Subsequent rows shift indices; we re-render
// every row's leading chip so the minus/picker handlers stay synced.
function _removeEndlessRow(rowIdx) {
    const arr = _getEndlessRowPrompts();
    if (rowIdx < 0 || rowIdx >= arr.length) return;
    arr.splice(rowIdx, 1);
    _setEndlessRowPrompts(arr);
    const { stack, placeholder } = _getSuggestStack();
    if (!stack) return;
    const rows = stack.querySelectorAll('.suggest-marquee-row');
    if (rows[rowIdx]) rows[rowIdx].remove();
    // Re-walk surviving rows + re-attach lead clusters with updated indices
    // so each row's minus/picker call references its NEW position.
    const survivingRows = stack.querySelectorAll('.suggest-marquee-row');
    survivingRows.forEach((row, i) => {
        const lead = row.querySelector('[data-endless-row-lead]');
        if (lead) {
            const promptId = arr[i];
            const promptObj = (typeof _getPromptById === 'function') ? _getPromptById(promptId) : null;
            const ttl = promptObj ? promptObj.title : promptId;
            lead.setAttribute('data-endless-row-lead', String(i));
            lead.querySelector('[data-row-prompt-btn]')?.setAttribute('onclick', `_openSuggestPromptPicker(${i})`);
            lead.querySelector('[data-row-prompt-label]') && (lead.querySelector('[data-row-prompt-label]').textContent = ttl);
            lead.querySelector('[data-row-refresh]')?.setAttribute('onclick', `_regenEndlessRow(${i})`);
            lead.querySelector('[data-row-remove]')?.setAttribute('onclick', `_removeEndlessRow(${i})`);
        }
    });
}
window._removeEndlessRow = _removeEndlessRow;

// CHAT mode — 4 reply chips beneath the input, no marquee. Each chip is
// a one-click "send this as your next message" shortcut.
//   With history:  4 contextual continuations of the conversation
//   Empty history: 4 conversation-starter prompts the user can pick to
//                  open a session (e.g. "queue 3 dragons", "what's running?")
// ---------------------------------------------------------------------------
// Chat suggestion-cluster controls — radio mode (helpful/playful/terse) +
// numeric stepper for chip count. Both persisted in localStorage so the
// user's pref survives reload.
//
// NOTE: the MODE pill is currently UI-only — wiring the value through to the
// /subjects/suggest fetch (so e.g. "playful" actually changes prompt tone)
// is a follow-up. The current chat suggestions already honor the user's
// _getDefaultPromptId() registry pick, and overlapping the chat pane with
// a SECOND, narrower mode dimension would conflict with that. Surface for
// now, plumb later. See TODO inside _chatSuggestRefreshCluster.
//
// The COUNT stepper IS wired: _renderChatReplies reads
// `slopfinity-chat-suggest-count` and adjusts how many chips it displays
// (server still returns up to n=6 buffer either way).
// ---------------------------------------------------------------------------
const _CHAT_SUGGEST_COUNT_KEY = 'slopfinity-chat-suggest-count';
const _CHAT_SUGGEST_MODE_KEY = 'slopfinity-chat-suggest-mode';
const _CHAT_SUGGEST_COUNT_MIN = 1;
const _CHAT_SUGGEST_COUNT_MAX = 6;

function _getChatSuggestCount() {
    try {
        const raw = parseInt(localStorage.getItem(_CHAT_SUGGEST_COUNT_KEY) || '', 10);
        if (Number.isFinite(raw) && raw >= _CHAT_SUGGEST_COUNT_MIN && raw <= _CHAT_SUGGEST_COUNT_MAX) return raw;
    } catch (_) { }
    return 3;
}
function _setChatSuggestCount(n) {
    const clamped = Math.max(_CHAT_SUGGEST_COUNT_MIN, Math.min(_CHAT_SUGGEST_COUNT_MAX, n | 0));
    try { localStorage.setItem(_CHAT_SUGGEST_COUNT_KEY, String(clamped)); } catch (_) { }
    const lbl = document.getElementById('chat-suggest-count');
    if (lbl) {
        lbl.textContent = String(clamped);
        lbl.setAttribute('data-count', String(clamped));
    }
    // Re-slice synchronously from the cached batch first so the rows
    // update IMMEDIATELY in lockstep with the tally. Only re-fetch when
    // we don't have enough cached suggestions to satisfy the new count.
    const synced = (typeof _resliceChatReplies === 'function') ? _resliceChatReplies(clamped) : false;
    if (!synced && typeof _renderChatReplies === 'function') _renderChatReplies();
    return clamped;
}
function _chatSuggestCountStep(delta) {
    return _setChatSuggestCount(_getChatSuggestCount() + (delta | 0));
}
window._getChatSuggestCount = _getChatSuggestCount;
window._setChatSuggestCount = _setChatSuggestCount;
window._chatSuggestCountStep = _chatSuggestCountStep;

// Chat-mode suggestion mode is now driven by the SHARED suggest_prompts
// system (the same Yes-and / Plot Twist / Concrete Detail / … entries
// simple mode uses). Pills are rendered into #chat-suggest-prompt-pills
// from _getActivePrompts(), clicking a pill calls _setDefaultPromptId so
// chat + simple share one selection. The legacy helpful/playful/terse
// scaffold (and slopfinity-chat-suggest-mode localStorage key) is gone.
function _renderChatSuggestPromptPills() {
    const host = document.getElementById('chat-suggest-prompt-pills');
    if (!host) return;
    const active = (typeof _getActivePrompts === 'function') ? _getActivePrompts() : [];
    if (!active.length) {
        host.innerHTML = '';
        return;
    }
    const cur = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : '';
    host.innerHTML = active.map(p => {
        const isActive = p.id === cur;
        return `<button type="button"
            class="btn btn-xs rounded-full btn-primary ${isActive ? '' : 'btn-outline'} normal-case chat-suggest-prompt-pill"
            data-chat-prompt-id="${_htmlEscape(p.id)}"
            title="${_htmlEscape(p.title)}"
            onclick="_onChatSuggestPromptClick('${_htmlEscape(p.id)}')">${_htmlEscape(p.title)}</button>`;
    }).join('');
}
window._renderChatSuggestPromptPills = _renderChatSuggestPromptPills;

function _onChatSuggestPromptClick(id) {
    if (typeof _setDefaultPromptId === 'function') _setDefaultPromptId(id);
    _renderChatSuggestPromptPills();
    if (typeof _renderChatReplies === 'function') _renderChatReplies();
}
window._onChatSuggestPromptClick = _onChatSuggestPromptClick;

// Hydrate the cluster's controls from localStorage and wire change handlers.
// Idempotent — safe to call multiple times (re-binds are cheap and the
// element set is small).
function _initChatSuggestCluster() {
    // Count badge.
    const lbl = document.getElementById('chat-suggest-count');
    if (lbl) {
        const n = _getChatSuggestCount();
        lbl.textContent = String(n);
        lbl.setAttribute('data-count', String(n));
    }
    // Prompt-pill cluster (driven by suggest_prompts; shared with
    // simple-mode dropdown via _setDefaultPromptId). _loadSuggestPrompts
    // hydrates the cache from /settings — fire-and-forget, then re-render
    // when ready. The first paint uses _SUGGEST_PROMPTS_FALLBACK so the
    // cluster never reads as empty during initial load.
    if (typeof _loadSuggestPrompts === 'function') {
        _loadSuggestPrompts().then(() => _renderChatSuggestPromptPills()).catch(() => { });
    }
    _renderChatSuggestPromptPills();
}
window._initChatSuggestCluster = _initChatSuggestCluster;
document.addEventListener('DOMContentLoaded', _initChatSuggestCluster);

async function _renderChatReplies() {
    const host = document.getElementById('subjects-chat-replies');
    if (!host) return;
    const history = (typeof _getChatHistory === 'function') ? _getChatHistory() : [];
    // No bouncing dots while we wait — the unified badge's refresh button
    // gets a spinning icon instead (see _spinRefreshBriefly). The reply
    // strip stays empty rather than flashing a placeholder that immediately
    // gets replaced.
    let subjects = '';
    if (history.length) {
        const lastAsst = [...history].reverse().find(m => m.role === 'assistant' && (m.content || '').trim());
        const lastUser = [...history].reverse().find(m => m.role === 'user' && (m.content || '').trim());
        subjects = `Last assistant: ${(lastAsst && lastAsst.content) || '(none)'}\nLast user: ${(lastUser && lastUser.content) || '(none)'}`;
    }
    // Honor the user's selected prompt_id so swapping the dropdown actually
    // changes the reply style (Yes-and / Plot Twist / Concrete Detail / …).
    // Empty subjects → server's default "give me N ideas" prompt picks up
    // the named prompt's system text via prompt_id.
    const promptId = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : '';
    let arr = [];
    try {
        // Fetch n=6 (2 extra) so we have spares ready when the user
        // clicks a chip → that chip animates out → we slide-in the
        // next from the buffer. Mirrors the endless-row consume-and-
        // refill UX. Server returns up to 6; we display 3.
        const qs = '?n=6&fresh=1&chat=1&_t=' + Date.now()
            + (subjects ? '&subjects=' + encodeURIComponent(subjects) : '')
            + (promptId ? '&prompt_id=' + encodeURIComponent(promptId) : '');
        const r = await fetch('/subjects/suggest' + qs);
        const d = await r.json();
        const dict = (d && d.suggestions) || {};
        arr = dict.chat || [];
    } catch (_) { }
    if (!arr.length) {
        host.innerHTML = '<span class="text-[10px] italic opacity-60">no replies (LLM unreachable)</span>';
        return;
    }
    // Stash the spare buffer on the host element so the click handler
    // can pull from it without re-fetching. Survives across renders
    // until the next regen wipes innerHTML. Display count is user-
    // configurable via the +/- stepper in the suggestions header
    // (localStorage `slopfinity-chat-suggest-count`, default 3, clamped
    // to [1, 6]). Whatever's left over after the displayed slice is the
    // prefetch buffer for consume-and-refill.
    const displayCount = (typeof _getChatSuggestCount === 'function') ? _getChatSuggestCount() : 3;
    // Stash the full fetched batch so the +/- stepper can re-slice
    // synchronously without re-fetching when the user changes count.
    host._chatReplyArr = arr;
    host._chatReplyBuffer = arr.slice(displayCount);
    // Reply chips share the same primary-outline aesthetic as every
    // OTHER suggestion chip in the dashboard (simple-mode marquee
    // chips via _buildSuggestChip, endless-row chips). Was btn-ghost
    // (no border, faded text) which read as "different cluster" and
    // visually drifted from the rest of the suggestion language.
    // Layout overrides (full-width left-justified text, normal-case
    // for sentence-fragment readability) layered on top of the
    // shared `btn btn-outline btn-primary btn-xs` base.
    // 3 chips displayed (was 4 — user pref). Each chip wires through
    // _consumeChatReply on click instead of the bare-IIFE inline
    // handler so we can ALSO fade the chip out + slide a spare in
    // from the buffer (mirrors endless-row consume UX). Single-quote
    // outer attribute is critical — JSON.stringify produces a
    // "-wrapped string and would close a "-wrapped onclick at the
    // first inner ". (See chat-suggestion-send.spec.js.)
    host.innerHTML = arr.slice(0, displayCount).map(s => {
        // The onclick attribute uses single-quote outer wrapping because
        // JSON.stringify(s) is "-wrapped (would close a "-wrapped attr).
        // BUT if `s` contains a literal apostrophe (e.g. "what's running")
        // the JSON output is `"what's running"` — the inner `'` closes
        // our outer single-quote attribute and the rest becomes garbage
        // ("Invalid or unexpected token"). Solution: HTML-escape the
        // JSON output for the attribute context — `&#39;` survives
        // attribute parsing AND the browser un-escapes it before JS
        // evaluation. Same fix applies to `&` (must precede the others).
        const payload = JSON.stringify(s)
            .replace(/&/g, '&amp;')
            .replace(/'/g, '&#39;');
        return `<button type="button" class="chat-reply-chip btn btn-outline btn-primary btn-xs normal-case w-full justify-start text-xs whitespace-normal text-left h-auto py-1.5"
            onclick='_consumeChatReply(this, ${payload})'>
            ${_htmlEscape(s)}
        </button>`;
    }).join('');
}
window._renderChatReplies = _renderChatReplies;

// Re-render the chat replies row from the cached batch (no network).
// Returns true when it was able to render the requested count from cache,
// false when we don't have enough cached items and a fresh fetch is needed.
// Drives the synchronous half of the +/- stepper UX so tally and chips
// update in lockstep without waiting on /subjects/suggest.
function _resliceChatReplies(displayCount) {
    const host = document.getElementById('subjects-chat-replies');
    if (!host) return false;
    const arr = host._chatReplyArr;
    if (!Array.isArray(arr) || arr.length < displayCount) return false;
    host._chatReplyBuffer = arr.slice(displayCount);
    host.innerHTML = arr.slice(0, displayCount).map(s => {
        // See _renderChatReplies for the apostrophe-escape rationale.
        const payload = JSON.stringify(s)
            .replace(/&/g, '&amp;')
            .replace(/'/g, '&#39;');
        return `<button type="button" class="chat-reply-chip btn btn-outline btn-primary btn-xs normal-case w-full justify-start text-xs whitespace-normal text-left h-auto py-1.5"
            onclick='_consumeChatReply(this, ${payload})'>
            ${_htmlEscape(s)}
        </button>`;
    }).join('');
    return true;
}
window._resliceChatReplies = _resliceChatReplies;

// Consume one chat reply chip:
//   1. fade the clicked chip out (CSS class .chip-disappear, ~700 ms)
//   2. submit the text via the standard chat input → _sendChatMessage
//   3. if the host has a spare from the prefetch buffer, slide it
//      into a NEW chip in the just-vacated slot
//   4. if the buffer is empty, just leave the row at 2 chips until
//      the next assistant turn re-renders (which fetches fresh n=6)
window._consumeChatReply = function (btn, text) {
    // Fire the send first so there's no perceived lag — the chat
    // input + _sendChatMessage path is the canonical send.
    const input = document.getElementById('subjects-chat-input');
    if (input) {
        input.value = text;
        input.focus();
        if (typeof _sendChatMessage === 'function') _sendChatMessage();
    }
    // Animate the chip out of the row.
    if (btn) {
        btn.classList.add('chip-disappear');
        const host = btn.parentElement;
        const spare = host && host._chatReplyBuffer && host._chatReplyBuffer.shift();
        // After the chip-disappear animation completes, either swap
        // in a fresh chip from the buffer or just remove the slot.
        setTimeout(() => {
            if (!btn.isConnected) return;
            if (spare) {
                const fresh = document.createElement('button');
                fresh.type = 'button';
                fresh.className = 'chat-reply-chip btn btn-outline btn-primary btn-xs normal-case w-full justify-start text-xs whitespace-normal text-left h-auto py-1.5 chip-arriving';
                fresh.textContent = spare;
                fresh.onclick = () => window._consumeChatReply(fresh, spare);
                btn.replaceWith(fresh);
                // drop the entry class on next frame so the slide-in plays
                requestAnimationFrame(() => requestAnimationFrame(() => fresh.classList.remove('chip-arriving')));
            } else {
                btn.remove();
            }
        }, 700);
    }
};

// Visual feedback on the unified-badge refresh icon: when an action is
// pending (dropdown swap → regen, manual refresh click), spin the icon
// for ~1.5 s OR until the next render hooks in.
function _spinRefreshBriefly(ms) {
    const btn = document.getElementById('subjects-suggest-btn');
    if (!btn) return;
    const svg = btn.querySelector('svg');
    if (!svg) return;
    svg.classList.add('animate-spin');
    setTimeout(() => svg.classList.remove('animate-spin'), ms || 1500);
}
window._spinRefreshBriefly = _spinRefreshBriefly;

// Universal refresh-tap acknowledgement. Any clickable element that LOOKS
// like a refresh affordance (data-row-refresh, #subjects-suggest-btn,
// #btn-suggest, .btn-refresh, plus the per-row refresh in suggest rows)
// gets a one-shot 600 ms rotation on its inner SVG. Independent of the
// underlying action's completion — purely a "yes I heard the click".
document.addEventListener('click', (e) => {
    const target = e.target.closest(
        '[data-row-refresh],' +
        '#subjects-suggest-btn,' +
        '#subjects-suggest-prompt-name + #subjects-suggest-btn,' +
        '.btn-refresh,' +
        '[data-refresh-tap]'
    );
    if (!target) return;
    const svg = target.querySelector('svg');
    if (!svg) return;
    svg.classList.remove('refresh-tapping'); // restart even if mid-spin
    void svg.offsetWidth;                    // force reflow so the keyframe re-fires
    svg.classList.add('refresh-tapping');
    setTimeout(() => svg.classList.remove('refresh-tapping'), 700);
}, { capture: true });

// ---------------------------------------------------------------------------
// Multi-row marquee chip area — replaces the single-row carousel from #76.
//
// Each batch arrives as a new <div class="suggest-marquee-row"> appended
// to #subject-chips-stack. Inside the row, a <div class="suggest-marquee-track">
// holds the chips duplicated (items + items) so the CSS @keyframes
// translateX(-50%) loop is visually seamless. Speed scales with track
// length (60 px/s target, clamped 40..180 s) and pauses on hover/focus.
// FIFO cap of _SUGGEST_MAX_ROWS = 4 evicts oldest rows.
//
// Composition with #76's prefetch state machine: _maybePrefetch still
// fills _prefetchedBatches; the consumer is now an idle-time top-up that
// appends a new row when buffered batches are present. Manual 🎲 still
// renders fresh and bypasses the prefetch buffer.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Subjects suggestion prefetch — preemptive cache from #76, repurposed
// to feed the multi-row marquee. Driven by hover/idle signals on the
// Subjects card; consumed by an idle-time top-up that calls
// _appendSuggestBatchRow when a buffered batch is ready. Every fetch path
// here is gated by _isGpuIdleEnough() AND by _autoSuggestDisabled() (the
// new Settings toggle). Manual 🎲 stays exempt from both.
//
// Invariants:
// - At most one inflight request (_prefetchInflight bool).
// - At most _PREFETCH_CAP buffered batches (FIFO).
// - 30 s backoff after empty/errored response.
// - Cancel idle timer when the page is hidden.
// ---------------------------------------------------------------------------

const _PREFETCH_CAP = 3;
const _PREFETCH_BACKOFF_MS = 30_000;
const _PREFETCH_IDLE_TRIGGER_MS = 8_000;
const _prefetchedBatches = []; // FIFO of string[]
let _prefetchInflight = false;
let _prefetchBackoffUntil = 0;
let _prefetchIdleTimer = null;
let _prefetchStats = { triggered: 0, fetched: 0, consumed: 0, skippedGpu: 0, skippedBackoff: 0, skippedFull: 0 };

// Consume the oldest buffered batch. If `forPromptId` is provided,
// skip + drop any leading entries that were generated under a
// DIFFERENT prompt (they're stale relative to the user's current
// dropdown choice and would surprise them if rendered now).
function _consumePrefetchedBatch(forPromptId) {
    while (_prefetchedBatches.length) {
        const head = _prefetchedBatches[0];
        if (forPromptId && head.promptId && head.promptId !== forPromptId) {
            _prefetchedBatches.shift(); // drop stale
            continue;
        }
        _prefetchedBatches.shift();
        _prefetchStats.consumed += 1;
        return head.suggestions;
    }
    return null;
}
// Drop every buffered batch (called when the user swaps prompts —
// batches were generated for the OLD prompt and will read as wrong-
// topic suggestions if the user later presses + with the new prompt
// active). Caller pairs this with a _maybePrefetch() to start
// refilling under the new prompt. `discarded` stat tracks dumps
// for telemetry / dev console.
function _dropPrefetchedBatches() {
    const n = _prefetchedBatches.length;
    _prefetchedBatches.length = 0;
    _prefetchStats.discarded = (_prefetchStats.discarded || 0) + n;
    return n;
}
window._dropPrefetchedBatches = _dropPrefetchedBatches;

function _maybePrefetch() {
    if (_isSuggestionsHidden()) return;
    // SIMPLE MODE ONLY. Endless story-beats need their own per-row
    // prompt context (lead cluster); chat replies render into a
    // different DOM target; raw doesn't use suggestions at all.
    // Speculative prefetch into a generic batch buffer doesn't
    // make sense for any of those.
    const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
    if (mode !== 'simple') return;
    _prefetchStats.triggered += 1;
    // Settings toggle — no auto-fetches when disabled. Manual 🎲 unaffected.
    if (_autoSuggestDisabled()) return;
    if (typeof _isGpuIdleEnough === 'function' && _gpuPctHistory.length > 0 && !_isGpuIdleEnough()) {
        _prefetchStats.skippedGpu += 1;
        return;
    }
    if (_prefetchInflight) return;
    if (_prefetchedBatches.length >= _PREFETCH_CAP) {
        _prefetchStats.skippedFull += 1;
        return;
    }
    if (Date.now() < _prefetchBackoffUntil) {
        _prefetchStats.skippedBackoff += 1;
        return;
    }
    _prefetchInflight = true;
    const subjects = (($('p-core') && $('p-core').value) || '').trim();
    // Capture the promptId AT FETCH TIME and ship it on the wire so
    // the server's own promptId is respected, AND tag the cached
    // batch with it so a later prompt-swap can reject stale entries
    // in _consumePrefetchedBatch / drop them via _dropPrefetchedBatches.
    const promptId = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : '';
    const qs = '?n=6'
        + (subjects ? '&subjects=' + encodeURIComponent(subjects) : '')
        + (promptId ? '&prompt_id=' + encodeURIComponent(promptId) : '');
    fetch('/subjects/suggest' + qs)
        .then(r => r.json())
        .then(d => {
            const dict = (d && d.suggestions) || {};
            const arr = dict.simple || [];
            if (arr.length) {
                _prefetchedBatches.push({ promptId, suggestions: arr });
                _prefetchStats.fetched += 1;
            } else {
                _prefetchBackoffUntil = Date.now() + _PREFETCH_BACKOFF_MS;
            }
        })
        .catch(() => { _prefetchBackoffUntil = Date.now() + _PREFETCH_BACKOFF_MS; })
        .finally(() => { _prefetchInflight = false; });
}
// Exposed so the chip-click handler in _buildSuggestChip can kick a
// fresh prefetch the moment the user picks a suggestion — "compute is
// probably free RIGHT NOW (the LLM just finished the last batch), so
// start the next one ASAP so it lands in the buffer before the user
// is ready to pick again." All gates inside _maybePrefetch (mode,
// GPU-idle, inflight, cap, backoff) still apply.
window._maybePrefetch = _maybePrefetch;
// Exposed for the simple-mode + button (`_addSimpleRow`) so it can
// INSTANT-render a prefetched batch instead of awaiting a fresh
// round-trip. Pass the current promptId so stale (post-swap)
// batches are skipped.
window._consumePrefetchedBatch = _consumePrefetchedBatch;

function _resetPrefetchIdleTimer() {
    if (_prefetchIdleTimer) clearTimeout(_prefetchIdleTimer);
    if (document.hidden) return;
    _prefetchIdleTimer = setTimeout(() => {
        // SIMPLE MODE ONLY. The drain previously fired regardless of
        // mode and called _appendSuggestBatchRow(pre) with no opts —
        // which in endless mode appended naked rows (no subject /
        // refresh / minus lead cluster), reading as 'simple-mode
        // rows leaking into endless'. The user explicitly asked
        // for the old auto-load behaviour to go: "we got rid of
        // the initial default load suggestions row right? that was
        // old design". So endless / chat / raw skip both drain
        // AND the speculative prefetch.
        const mode = (typeof _getSubjectsMode === 'function') ? _getSubjectsMode() : 'simple';
        if (mode !== 'simple') return;
        const curPid = (typeof _getDefaultPromptId === 'function') ? _getDefaultPromptId() : '';
        const pre = _consumePrefetchedBatch(curPid);
        if (pre) {
            _appendSuggestBatchRow(pre);
            _maybePrefetch();
        } else {
            _maybePrefetch();
        }
    }, _PREFETCH_IDLE_TRIGGER_MS);
}

let _suggestPrefetchWired = false;
function _wireSuggestPrefetch() {
    if (_suggestPrefetchWired) return;
    const { stack, placeholder } = _getSuggestStack();
    const ta = document.getElementById('p-core');
    if (!stack) return;
    _suggestPrefetchWired = true;

    // Hover the stack → top up the buffer (gated). Drain happens via the
    // idle timer below so the marquee doesn't grow rows under the cursor.
    stack.addEventListener('pointerenter', () => _maybePrefetch());

    if (ta) {
        ta.addEventListener('input', _resetPrefetchIdleTimer);
        ta.addEventListener('focus', _resetPrefetchIdleTimer);
    }
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && _prefetchIdleTimer) {
            clearTimeout(_prefetchIdleTimer);
            _prefetchIdleTimer = null;
        } else if (!document.hidden) {
            _resetPrefetchIdleTimer();
        }
    });
    _resetPrefetchIdleTimer();
}
document.addEventListener('DOMContentLoaded', _wireSuggestPrefetch);

// Devtools helper — `_dumpSuggestPrefetchStats()` from the console.
window._dumpSuggestPrefetchStats = function () {
    return {
        cap: _PREFETCH_CAP,
        buffered: _prefetchedBatches.length,
        inflight: _prefetchInflight,
        backoffMsRemaining: Math.max(0, _prefetchBackoffUntil - Date.now()),
        stats: { ..._prefetchStats },
    };
};

function updateStageSteps(state) {
    if (!state) return;
    const steps = document.querySelectorAll('#stage-steps li[data-stage]');
    if (!steps.length) return;
    const currentStage = state.step || '';
    const order = ['Concept', 'Base Image', 'Video Chains', 'Audio', 'TTS', 'Post Process', 'Final Merge'];
    const idx = order.indexOf(currentStage);
    steps.forEach((el, i) => {
        el.classList.remove('step-primary', 'step-accent');
        if (idx < 0) return;
        if (i < idx) el.classList.add('step-primary');
        else if (i === idx) el.classList.add('step-accent');
    });
    const chainCounter = document.getElementById('chain-counter');
    if (chainCounter) {
        // Element id is legacy ("chain-counter"); display text now reads
        // "Part X of Y" to match the user-facing rename. The state keys
        // (`chain_index`, `total_chains`) come straight from the runner and
        // stay unchanged.
        chainCounter.textContent = (state.step === 'Video Chains' && state.chain_index)
            ? `Part ${state.chain_index} of ${state.total_chains}` : '';
    }
}

// No-op stubs for reference-modal inline handlers that don't yet exist client-side.
if (typeof window._onAudioChanged !== 'function') {
    window._onAudioChanged = function () { /* reserved for future audio-dependent UI */ };
}

connect();
_wireLockListeners();

// ---------------------------------------------------------------------------
// Top-section collapsible — persist open/closed in localStorage so the user's
// preference survives reloads. The <details> element starts `open` in the
// template so first-load matches existing behaviour; if the user has previously
// collapsed it, we strip the attribute on init.
// ---------------------------------------------------------------------------
(function wireTopCollapsible() {
    const init = () => {
        const COLLAPSE_KEY = 'slopfinity_top_collapsed';
        const top = document.getElementById('top-collapsible');
        if (!top) return;
        try {
            if (localStorage.getItem(COLLAPSE_KEY) === '1') top.removeAttribute('open');
        } catch (_) { }
        top.addEventListener('toggle', () => {
            try { localStorage.setItem(COLLAPSE_KEY, top.open ? '0' : '1'); } catch (_) { }
        });
    };
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

// ===========================================================================
// Draggable horizontal splitter between the Subjects/Queue (upper) and the
// Slop output (lower). Persists upper-section height as a 0..1 fraction of
// the splitter container in localStorage. Double-click resets to 50/50
// (cleared key → both panes use natural flex). Touch / pen / mouse all work
// via Pointer Events; Pointer Capture keeps drag tracking even when the
// cursor leaves the handle.
// ===========================================================================
(function wireUiSplit() {
    const init = () => {
        const handle = document.getElementById('ui-split-handle');
        const upper = document.getElementById('ui-split-upper');
        const lower = document.getElementById('ui-split-lower');
        if (!handle || !upper || !lower) return;
        // Storage key changed from `_pct` (legacy %, container-relative) to
        // `_px` (absolute pixel height of the upper pane). The container no
        // longer has a fixed height, so percent math has nothing to multiply
        // against. Pixel storage is bounded by [200 px, 80 vh] on read.
        const KEY = 'slopfinity_ui_split_upper_px';
        // Helper — broadcast that the panes resized so internal autogrow /
        // re-layout code (textarea autogrow, suggestion-chip filler, etc.)
        // can recompute against the new available height.
        const _emitSplitResize = () => {
            try { window.dispatchEvent(new Event('resize')); } catch (_) { }
            if (typeof window._autogrowSubjects === 'function') {
                try { window._autogrowSubjects(); } catch (_) { }
            }
        };
        const _bounds = () => ({
            min: 200,
            max: Math.max(220, Math.floor(window.innerHeight * 0.8)),
        });
        const stored = parseFloat(localStorage.getItem(KEY));
        if (!Number.isNaN(stored) && stored >= 100) {
            const { min, max } = _bounds();
            upper.style.height = Math.max(min, Math.min(max, stored)) + 'px';
        }
        let dragging = false;
        let startY = 0;
        let startUpperPx = 0;
        handle.addEventListener('pointerdown', (e) => {
            // Skip on viewports where the media query made the handle inert.
            if (window.matchMedia('(max-width: 768px)').matches) return;
            dragging = true;
            startY = e.clientY;
            startUpperPx = upper.getBoundingClientRect().height;
            try { handle.setPointerCapture(e.pointerId); } catch (_) { }
            handle.classList.add('dragging');
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
        handle.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            const { min, max } = _bounds();
            const newUpper = Math.max(min, Math.min(max, startUpperPx + (e.clientY - startY)));
            upper.style.height = newUpper + 'px';
            // Lower pane is naturally below in the flex column and content-
            // sized; no math needed for it.
            _emitSplitResize();
        });
        const stop = (e) => {
            if (!dragging) return;
            dragging = false;
            try { handle.releasePointerCapture(e.pointerId); } catch (_) { }
            handle.classList.remove('dragging');
            document.body.style.userSelect = '';
            const px = upper.getBoundingClientRect().height;
            if (px >= 100) localStorage.setItem(KEY, String(Math.round(px)));
        };
        handle.addEventListener('pointerup', stop);
        handle.addEventListener('pointercancel', stop);
        // Double-click resets to default (clear the override + remove storage).
        handle.addEventListener('dblclick', () => {
            upper.style.height = '';
            localStorage.removeItem(KEY);
            _emitSplitResize();
        });
    };
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

// ===========================================================================
// Slop card factory + infinite scroll (PR: feat/infinite-scroll-slop-view)
// ---------------------------------------------------------------------------
// `_buildSlopCard` is the single source of truth for how a slop-feed card
// renders. Originally inlined in the WS `new_file` handler; now also called
// by the IntersectionObserver-driven infinite-scroll loader so that initial
// SSR cards, live WS pushes, and lazily-fetched older cards all produce
// identical DOM (filter chips, delete handlers, count chips all key off
// `data-slop-kind` and the .card.card-compact selector).
//
// Pagination shape (server: GET /assets):
//   { items: [{file, mtime, kind}], offset, limit, total, has_more }
//
// SSR renders the first 64 cards (see index.html `all_assets[:64]`); the
// observer takes over from offset=64 on. WS new_file events prepend new
// cards (they don't shift the offset — the offset tracks how far down
// historical mtime we've fetched, not the number of children in the grid).
// ===========================================================================
function _buildSlopCard(file, opts = {}) {
    if (!file) return null;
    const isV = file.endsWith('.mp4');
    const isWav = file.endsWith('.wav');
    const kind = isV ? 'video' : isWav ? 'audio' : 'image';
    const meta = _slopBadgeMeta(file);
    const c = document.createElement('div');
    const pulseClass = opts.pulse ? ' animate-pulse' : '';
    c.className = `card card-compact bg-base-100 shadow-lg border ${meta.border} card-hover overflow-hidden${pulseClass}`;
    c.dataset.slopKind = meta.kind || kind;
    // Match the SSR card contract — _applySlopFilters reads both flags.
    // isFinal: FINAL_*.mp4 (the merged video the pipeline builds last).
    // isFrame: ffmpeg-extracted bridge PNG between video chains —
    //   slop_<N>_<slug>_f<M>.png OR legacy v<N>_f<M>.png.
    c.dataset.slopFinal = (isV && /^FINAL_/i.test(file)) ? '1' : '0';
    c.dataset.slopFrame = /^(?:v\d+|slop_\d+)(?:_.+)?_f\d+\.png$/i.test(file) ? '1' : '0';
    c.dataset.slopFile = file;
    const partBadge = meta.part
        ? `<span class="badge badge-xs badge-ghost">part ${meta.part}</span>`
        : '';
    const autoplayAttr = opts.autoplay ? 'autoplay' : '';
    let media;
    if (isV) {
        media = `<figure class="bg-black aspect-video flex items-center justify-center overflow-hidden"><video controls ${autoplayAttr} muted loop preload="metadata" class="w-full h-full object-contain"><source data-src="/files/${file}"></video></figure>`;
    } else if (isWav) {
        media = `<figure class="bg-black aspect-video flex items-center justify-center overflow-hidden"><audio controls class="w-full mx-2"><source data-src="/files/${file}"></audio></figure>`;
    } else {
        media = `<figure class="bg-black aspect-video flex items-center justify-center overflow-hidden"><img data-src="/files/${file}" class="w-full h-full object-contain" loading="lazy"></figure>`;
    }
    c.innerHTML = `${media}
        <div class="card-body !p-2 bg-base-200/60 gap-1">
            <div class="flex flex-wrap items-center gap-1">
                <span class="badge badge-xs ${meta.color}">${meta.label}</span>
                ${partBadge}
            </div>
            <span class="text-[10px] font-mono text-base-content/60 truncate" title="${file}">${file}</span>
        </div>`;

    // Throttle loading
    PriorityLoader.register(c);

    return c;
}

(function wireInfiniteScroll() {
    const init = () => {
        const sentinel = document.getElementById('preview-grid-sentinel');
        const grid = document.getElementById('preview-grid');
        if (!sentinel || !grid) return;
        // Seed the v_idx resolver cache from the SSR-rendered cards on
        // the page. Each card has its filename in the trailing
        // `<span title="..."` attribute (see _buildSlopCard above) — we
        // walk those once at init time so the very first pipeline-strip
        // / done-list paint can use real names instead of synthesizing
        // v{N}_base.png.
        try {
            grid.querySelectorAll('[data-slop-kind] span[title]').forEach(span => {
                const f = span.getAttribute('title');
                if (f) _ingestAssetFilename(f);
            });
        } catch (_) { /* ignore */ }

        let loading = false;
        let exhausted = false;
        // SSR seeds the first 64 cards; older content starts at offset=64.
        // WS-pushed files prepend without bumping offset (offset tracks
        // historical mtime depth, not DOM child count).
        let nextOffset = 64;
        const PAGE = 48;

        async function loadMore() {
            if (loading || exhausted) return;
            loading = true;
            try {
                const r = await fetch(`/assets?offset=${nextOffset}&limit=${PAGE}`);
                if (!r.ok) { exhausted = true; return; }
                const j = await r.json();
                const items = (j && j.items) || [];
                if (!items.length) { exhausted = true; return; }
                // Append before the sentinel so it stays at the bottom of
                // the grid container.
                items.forEach(item => {
                    // Seed the resolver cache so older paginated content
                    // also feeds v_idx -> real-filename lookups.
                    _ingestAssetFilename(item.file);
                    const card = _buildSlopCard(item.file);
                    if (card) grid.appendChild(card);
                });
                nextOffset += items.length;
                if (!j.has_more) exhausted = true;
                // Keep current filter chip state applied to freshly inserted cards.
                try { _applySlopFilters(); } catch (_) { /* ignore */ }
            } catch (e) {
                console.warn('infinite-scroll fetch failed:', e);
            } finally {
                loading = false;
            }
        }

        // The page body is now the scroll container — the lower pane no
        // longer has its own overflow cap. `root: null` ties the observer
        // to the viewport so the sentinel fires as the user scrolls the
        // page, not an inner pane.
        const observer = new IntersectionObserver((entries) => {
            if (entries.some(e => e.isIntersecting)) loadMore();
        }, {
            root: null,
            rootMargin: '300px',
            threshold: 0,
        });
        observer.observe(sentinel);
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

async function refreshLlmPool() {
    const list = document.getElementById('llm-pool-list');
    if (!list) return;
    try {
        const r = await fetch('/llm/pool');
        if (!r.ok) throw new Error('failed');
        const data = await r.json();
        
        let html = '';
        const addNode = (role, node) => {
            if (!node || !node.url) return;
            const statusColor = node.ok ? 'text-success' : 'text-error';
            const statusIcon = node.ok ? '✓' : '⚠';
            html += `
                <div class="bg-base-200 p-2 rounded flex justify-between items-center text-xs">
                    <div class="flex items-center gap-2">
                        <span class="badge badge-outline badge-xs uppercase opacity-70">${role}</span>
                        <span class="font-mono opacity-80">${node.url}</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="font-mono text-[10px] opacity-60">${node.selected_model || 'no model'}</span>
                        <span class="${statusColor} font-bold" title="${node.error || 'OK'}">${statusIcon}</span>
                    </div>
                </div>
            `;
        };
        addNode('Primary', data.primary);
        addNode('CPU', data.cpu);
        if (data.failovers) {
            data.failovers.forEach((f, i) => addNode(`Failover ${i+1}`, f));
        }
        list.innerHTML = html || '<div class="text-xs opacity-50 italic">No pool endpoints configured in .env</div>';
    } catch (e) {
        list.innerHTML = '<div class="text-xs text-error">Failed to load pool status</div>';
    }
}
window.refreshLlmPool = refreshLlmPool;

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(refreshLlmPool, 1000);
});
