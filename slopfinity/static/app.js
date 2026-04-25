// Slopfinity dashboard client.

// Theme persistence — apply any previously-chosen theme as early as possible
// to avoid a flash of unstyled/default theme on load.
(function () {
  const saved = localStorage.getItem('slopfinity-theme');
  if (saved) document.documentElement.dataset.theme = saved;
})();

function applyTheme(name) {
  if (!name) return;
  document.documentElement.dataset.theme = name;
  localStorage.setItem('slopfinity-theme', name);
}
window.applyTheme = applyTheme;

// PWA: register service worker (scoped to /) for installable desktop-icon experience.
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker
            .register('/sw.js', { scope: '/' })
            .catch((err) => console.warn('SW registration failed:', err));
    });
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
    try { localStorage.setItem(_STAGE_HISTORY_KEY, JSON.stringify(hist)); } catch {}
}
// Conservative defaults — tripled from the original observed-warm-run
// numbers because real cold starts on Strix Halo (model load + 8 denoise
// steps + VAE) routinely hit the upper end. ETAs should over-estimate
// until we accumulate enough samples to trust them.
const _STAGE_DEFAULT_SECONDS = {
    'Concept':        24,
    'Base Image':    540,
    'Video Chains': 1800,
    'Audio':         180,
    'TTS':            60,
    'Post Process':  180,
    'Final Merge':    60,
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
    const cls = q.succeeded === false ? 'badge-error' : 'badge-success';
    const sym = q.succeeded === false ? '✗' : '✓';
    const promptEsc = _htmlEscape(q.prompt || '');
    // Backwards-compat: pre-asset-tracking done records only have v_idx /
    // image_only. Synthesize a best-guess single-asset list from that so old
    // history items still render a thumbnail.
    let assets = Array.isArray(q.assets) ? q.assets.filter(Boolean) : [];
    if (!assets.length) {
        const v = q.v_idx || 0;
        if (v) {
            if (q.image_only) assets = [`v${v}_base.png`];
            else assets = [`FINAL_${v}.mp4`, `v${v}_base.png`];
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
                <span class="badge badge-xs ${cls}">${sym} done</span>
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
const _STAGE_ORDER = ['Concept', 'Base Image', 'Video Chains', 'Audio', 'TTS', 'Post Process', 'Final Merge'];
function _stageDoneBefore(curStage, candidate) {
    const ci = _STAGE_ORDER.indexOf(curStage);
    const xi = _STAGE_ORDER.indexOf(candidate);
    return ci > -1 && xi > -1 && xi < ci;
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

// Quick read-only popup for the LLM-rewritten prompt of the active job.
// Lighter than openAssetInfo (which is for files); this is just text.
function showPromptPeek(text) {
    const existing = document.getElementById('prompt-peek-modal');
    if (existing) existing.remove();
    const dlg = document.createElement('dialog');
    dlg.id = 'prompt-peek-modal';
    dlg.className = 'modal';
    // Store the raw prompt on the button; let the click handler read it back.
    // Avoids HTML-attribute-quoting hell (JSON.stringify of a prompt with
    // any double-quote or backtick will corrupt the inline onclick).
    dlg.innerHTML = `<div class="modal-box bg-base-200 border border-base-100 max-w-2xl">
        <h3 class="font-bold text-sm text-accent uppercase tracking-widest mb-2">LLM-rewritten prompt</h3>
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
            navigator.clipboard.writeText(copyBtn._prompt || '').catch(() => {});
            copyBtn.textContent = '✓ Copied';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
        });
    }
    dlg.showModal();
}

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
    const setPill = (cls, label) => {
        pill.className = cls;
        text.textContent = label;
        pill.style.display = '';
    };
    if (!_wsConnected) {
        return setPill(
            'badge badge-sm badge-error rounded-full gap-1 normal-case font-mono mx-auto',
            '⚠ Connection Lost'
        );
    }
    const paused = !!(_lastTick && _lastTick.scheduler && _lastTick.scheduler.paused);
    if (paused) {
        return setPill(
            'badge badge-sm badge-warning rounded-full gap-1 normal-case font-mono mx-auto',
            '⏸ Paused'
        );
    }
    if (isRendering) {
        // Active node in a queue item already shows the verb — hide the pill.
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
        await fetch('/queue/requeue', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ts }),
        });
    } catch (e) { console.warn('requeue failed', e); }
}

// Tick stage + total elapsed once a second so they don't jump only on WS ticks.
setInterval(() => {
    if (!_isRendering) return;
    // Live update the active queue item's badges. Format must match the
    // renderItem template exactly or the badges flicker between the two.
    document.querySelectorAll('[data-q-status="active"] [data-q-stage-elapsed]').forEach(el => {
        if (_stageStartTs) el.innerHTML = _fmtElapsedHtml(Date.now() - _stageStartTs);
    });
    document.querySelectorAll('[data-q-status="active"] [data-q-job-elapsed]').forEach(el => {
        if (_jobStartTs) el.innerHTML = _fmtElapsedHtml(Date.now() - _jobStartTs);
    });
    // Total ETA from rolling stage averages.
    const totalEta = _STAGE_ORDER
        .map(_stageAvgSeconds)
        .filter(x => x != null)
        .reduce((a, b) => a + b, 0);
    if (totalEta > 0) {
        document.querySelectorAll('[data-q-status="active"] [data-q-job-eta]').forEach(el => {
            el.innerHTML = 'ETA ' + _fmtElapsedHtml(totalEta * 1000);
        });
    }
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
    el.className = 'font-mono font-black ' + (
        d.status === 'danger' ? 'text-error' :
        d.status === 'warn' ? 'text-warning' : 'text-accent'
    );
    if (el.parentElement) el.parentElement.title = `${d.used_gb} / ${d.total_gb} GB`;
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
    const bd  = el.querySelector('.ram-breakdown');
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
    // Per-step decision rows.
    for (let i = 0; i < plan.decisions.length; i++) {
        const d = plan.decisions[i];
        const tag = d.load && d.load.length ? 'LOAD ' : 'HIT  ';
        const cls = d.load && d.load.length ? '' : 'text-success';
        const stage = (d.step.stage || '').padEnd(11);
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
    if (body) body.innerHTML = '<span class="loading loading-dots loading-sm"></span>';
    if (media) media.innerHTML = '';
    if (d.showModal) d.showModal();
    try {
        const r = await fetch('/asset/' + encodeURIComponent(filename));
        const m = await r.json();
        if (!m.ok) {
            body.innerHTML = `<div class="alert alert-error text-xs">${m.error || 'error'}</div>`;
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
        const badgeColor = ({final:'badge-accent',chain:'badge-primary',image:'badge-secondary',audio:'badge-warning'})[m.kind] || 'badge-ghost';
        body.innerHTML = `
            <div class="grid grid-cols-[min-content_1fr] gap-x-3 gap-y-1 text-xs font-mono">
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">File</div><div class="truncate">${m.filename}</div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Kind</div><div><span class="badge badge-xs ${badgeColor}">${m.kind}</span></div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Model</div><div>${m.model || '—'}</div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Size</div><div>${m.size_human}</div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Created</div><div>${m.mtime_human} <span class="text-base-content/50">(${m.age_seconds}s ago)</span></div>
                <div class="text-base-content/50 uppercase tracking-widest text-[10px]">Prompt</div><div class="whitespace-pre-wrap italic ${m.prompt ? '' : 'text-base-content/40'}">${m.prompt || '(no sidecar captured yet — fleet writes prompts to state.json only while active)'}</div>
            </div>
        `;
    } catch (e) {
        body.innerHTML = `<div class="alert alert-error text-xs">${String(e)}</div>`;
    }
}

// Click-to-info wiring for any asset card in the Slop feed
document.addEventListener('click', (e) => {
    const card = e.target.closest('#preview-grid > [data-slop-kind]');
    if (!card) return;
    // Avoid opening info when clicking the native media controls
    if (e.target.closest('video, audio')) return;
    const nameSpan = card.querySelector('[title]');
    const filename = nameSpan ? nameSpan.getAttribute('title') : null;
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
    'Concept':       'llm',
    'Base Image':    'base',
    'Video Chains':  'video',
    'Audio':         'audio',
    'TTS':           'tts',
    'Post Process':  'upscale',
    'Final Merge':   'video',
};
const _STAGE_ASSET = (stage, v_idx, c_idx) => {
    if (!v_idx) return null;
    if (stage === 'Base Image') return `v${v_idx}_base.png`;
    if (stage === 'Video Chains' && c_idx > 0) return `v${v_idx}_c${c_idx}.mp4`;
    if (stage === 'Final Merge') return `FINAL_${v_idx}.mp4`;
    return null;
};
const _STAGE_TEXT = {
    'Concept':       'generating prompts',
    'Base Image':    'rendering image',
    'Video Chains':  'rendering video chain',
    'Audio':         'composing music',
    'TTS':           'recording voiceover',
    'Post Process':  'polishing',
    'Final Merge':   'merging final',
};

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
    const mk = (role, klass, title, label) => {
        if (qTs == null) {
            return `<span class="badge badge-xs ${klass} gap-1" title="${_htmlEscape(title)}">${spin(role)}${label}</span>`;
        }
        return `<button type="button" class="badge badge-xs ${klass} gap-1 cursor-pointer" title="${_htmlEscape(title)} — click for settings" onclick='event.stopPropagation(); openModelSettingsPopup(${JSON.stringify(role)}, ${qTs})'>${spin(role)}${label}</button>`;
    };
    const out = [];
    if (llmModelId) {
        // Strip leading path (huggingface-style "owner/repo/...") but KEEP
        // the colon-separated name:tag — `qwen3:4b` should display as
        // `qwen3:4b`, not just `4b`.
        const short = llmModelId.replace(/^.*\//, '').replace(/\.gguf$/i, '');
        out.push(mk('llm', 'badge-accent', `prompt LLM: ${llmModelId}`, _htmlEscape(short)));
    }
    if (snap.base_model) out.push(mk('base', 'badge-info', 'image model', _htmlEscape(_modelDisplayName(snap.base_model, 'image'))));
    if (snap.video_model) out.push(mk('video', 'badge-success', 'video model', _htmlEscape(_modelDisplayName(snap.video_model, 'video'))));
    if (snap.audio_model && snap.audio_model !== 'none') out.push(mk('audio', 'badge-secondary', 'music model', _htmlEscape(_modelDisplayName(snap.audio_model, 'audio'))));
    if (snap.tts_model && snap.tts_model !== 'none') out.push(mk('tts', 'badge-warning', 'voice model', _htmlEscape(_modelDisplayName(snap.tts_model, 'audio'))));
    if (snap.upscale_model && snap.upscale_model !== 'none') out.push(mk('upscale', 'badge-warning', 'upscaler', _htmlEscape(snap.upscale_model)));
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
        llm:     'LLM — prompt rewriter',
        base:    'Image stage',
        video:   'Video stage',
        audio:   'Music stage',
        tts:     'Voice stage',
        upscale: 'Post-process stage',
    };
    const title = titleMap[role] || 'Stage settings';

    // Each row is a [label, value] pair; missing values render as em-dash.
    const rows = [];
    const push = (k, v) => rows.push([k, (v == null || v === '') ? '—' : v]);

    if (role === 'llm') {
        push('Provider',     llm.provider || 'auto');
        push('Model',        llm.model_id || 'auto-pick');
        push('Base URL',     llm.base_url);
        push('Rewrite mode', cfg.enhancer_prompt ? 'cinematic-director (custom)' : 'default');
        push('Subject',      promptText);
    } else if (role === 'base') {
        push('Model',        _modelDisplayName(snap.base_model, 'image'));
        push('Size',         snap.size);
        push('Quality ramp', snap.quality_ramp ? 'on' : 'off');
        push('Prompt override', snap.image_prompt_override);
    } else if (role === 'video') {
        push('Model',        _modelDisplayName(snap.video_model, 'video'));
        push('Frames',       snap.frames);
        push('Chains',       snap.chains);
        push('Quality ramp', snap.video_quality_ramp ? 'on' : 'off');
    } else if (role === 'audio') {
        push('Model',        _modelDisplayName(snap.audio_model, 'audio'));
        push('Music gain',   (snap.music_gain_db != null) ? `${snap.music_gain_db} dB` : null);
        push('Fade',         (snap.fade_s != null) ? `${snap.fade_s}s` : null);
    } else if (role === 'tts') {
        push('Model',        _modelDisplayName(snap.tts_model, 'audio'));
        push('Voice preset', snap.tts_voice || snap.voice_preset);
        push('Voice gain',   (snap.voice_gain_db != null) ? `${snap.voice_gain_db} dB` : null);
    } else if (role === 'upscale') {
        push('Upscaler',     snap.upscale_model);
        push('Upscale on',   snap.upscale ? 'yes' : 'no');
        push('Consolidation', snap.consolidation);
    }

    const body = rows.map(([k, v]) =>
        `<div class="text-base-content/50 uppercase tracking-widest text-[10px]">${_htmlEscape(k)}</div>` +
        `<div class="font-mono text-xs whitespace-pre-wrap break-words">${_htmlEscape(String(v))}</div>`
    ).join('');

    const titleEl = document.getElementById('model-settings-title');
    const bodyEl  = document.getElementById('model-settings-body');
    if (titleEl) titleEl.textContent = title;
    if (bodyEl) {
        bodyEl.innerHTML = `<div class="grid grid-cols-[min-content_1fr] gap-x-3 gap-y-1">${body}</div>` +
            `<div class="text-[10px] text-base-content/40 italic mt-3">Read-only snapshot from when this item was queued. Use “Open Pipeline Advanced” to edit defaults for future items.</div>`;
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
    document.querySelectorAll('#preview-grid > [data-slop-kind]').forEach(card => {
        card.style.display = enabled[card.dataset.slopKind] === false ? 'none' : '';
    });
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
        try { localStorage.setItem(KEY, String(Math.max(0.2, Math.min(0.8, ratio)))); } catch {}
    });
    document.addEventListener('mouseup', () => { dragging = false; document.body.style.userSelect = ''; });
}

document.addEventListener('DOMContentLoaded', () => {
    _applySlopFilters();
    _initSplitDivider();
    if (typeof _updateSingleLabels === 'function') _updateSingleLabels();
    if (typeof _updateChaosEnabled === 'function') _updateChaosEnabled();
    if (typeof _updateTerminateEnabled === 'function') _updateTerminateEnabled();
    if (typeof _updateGenModePill === 'function') _updateGenModePill();
    if (typeof _renderStageEtas === 'function') _renderStageEtas();
    // Suggestions on page load:
    //   1. Render cached chips immediately (no LLM call) so the row isn't
    //      empty during the first second.
    //   2. If no cache AND queue is idle, fire one fetch to populate cache.
    //   3. The 🎲 Suggest button always fetches fresh + overwrites cache.
    const hadCache = _renderCachedSuggestions();
    if (!hadCache && typeof regenSuggestions === 'function') {
        const tryAutoSuggest = () => {
            const t = _lastTick;
            if (!t) return setTimeout(tryAutoSuggest, 250);
            // Preferred gate: GPU has been at <=5% for >=3 consecutive
            // seconds. This catches ad-hoc GPU users (manual ComfyUI
            // runs, transient spikes) that the old queue/fleet check
            // missed, and avoids firing during brief fleet-stage gaps.
            if (_gpuPctHistory.length > 0) {
                if (!_isGpuIdleEnough()) {
                    console.info('skipping auto-suggest: GPU busy');
                    return setTimeout(tryAutoSuggest, 1000);
                }
            } else {
                // Fallback for the very first ticks before any GPU
                // history accumulates — keep the old heuristic so we
                // don't block forever if WS stats are unavailable.
                const queueBusy = (t.queue || []).some(x => x.status == null || x.status === 'pending');
                const fleetBusy = t.state && t.state.mode && t.state.mode !== 'Idle';
                if (queueBusy || fleetBusy) {
                    console.info('skipping auto-suggest: queue active');
                    return;
                }
            }
            regenSuggestions().catch(() => {});
        };
        setTimeout(tryAutoSuggest, 500);
    }
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
    if (chipA) chipA.textContent = document.querySelectorAll('#preview-grid > [data-slop-kind="audio"]').length;
}

function updateScheduler(sc) {
    if (!sc) return;
    const events = (sc.events || []).slice(-5);
    // Hide the whole strip when we have nothing to show (reduces visual noise when idle).
    const strip = $('sched-strip');
    if (strip) strip.style.display = (events.length > 0 || sc.paused) ? 'flex' : 'none';
    const pauseBadge = $('sched-pause-badge');
    if (pauseBadge) {
        pauseBadge.innerText = sc.paused ? 'paused' : 'live';
        pauseBadge.className = 'badge badge-sm font-mono ' + (sc.paused ? 'badge-warning' : 'badge-ghost');
    }
    // Settings-modal Scheduler tab badge
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
    renderEvents($('sched-timeline'), 'sched-event');
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
// GPU-idle gate audit — every auto-fetch surface that talks to the LLM
// (or anything else that competes with running pipelines) MUST consult
// _isGpuIdleEnough() first. Manual user-driven actions (button clicks,
// keyboard shortcuts) are intentionally NOT gated — explicit intent wins.
//
// Gated callers (auto-fetch surfaces):
//   - tryAutoSuggest (page-load auto-suggest, ~line 1092):
//       gated; retries every 1 s while GPU busy.
//   - _wireSuggestCarousel right-overlay click → fresh-fetch fallback
//       (when prefetch FIFO is empty): gated; shows brief "is-loading"
//       hint and bails out.
//   - _maybePrefetch (pointerenter / scroll / idle triggers):
//       gated; silent no-op when busy.
//
// NOT gated (explicit user intent):
//   - regenSuggestions() — the 🎲 Suggest button. Manual click always wins.
//   - _wireSuggestCarousel right-overlay → _consumePrefetchedBatch path:
//       no fetch, just dequeues a previously-prefetched batch. The
//       opportunistic top-up _maybePrefetch() it kicks IS gated.
//
// Definition: GPU has been at <=5% for >=3 consecutive seconds. Catches
// ad-hoc GPU work the older queue/fleet check missed.
// ---------------------------------------------------------------------------
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
        if (d.type === 'state') {
            // Tone the percentage colour with the latest ticker column —
            // text-error above 80 %, otherwise the per-pill tone class. Keeps
            // the number visually in sync with the bar colour.
            const _toneClass = (pct, baseTone) => 'font-mono font-black ' + (pct > 80 ? 'text-error' : 'text-' + baseTone);
            const gpuPct = d.stats.gpu;
            const gpuEl = $('g-v');
            if (gpuEl) { gpuEl.innerText = gpuPct + '%'; gpuEl.className = _toneClass(gpuPct, 'primary'); }
            // Strix Halo has unified memory — rocm-smi's VRAM% always reads 0,
            // so derive RAM% from the host meminfo numbers (ram_u / ram_t).
            const ramPct = d.stats.ram_t > 0 ? Math.round((d.stats.ram_u / d.stats.ram_t) * 100) : 0;
            const ramEl = $('v-v');
            if (ramEl) { ramEl.innerText = ramPct + '%'; ramEl.className = _toneClass(ramPct, 'secondary'); }
            $('r-v').innerText = d.stats.ram_u + ' / ' + Math.round(d.stats.ram_t) + ' GB';

            gH.push(d.stats.gpu); vH.push(ramPct);
            if (gH.length > 15) gH.shift();
            if (vH.length > 15) vH.shift();
            // Use DaisyUI bg-* utility classes so the ticker tints match the
            // active theme (and switch when the user changes themes). >80% gets
            // bg-error to flag pressure regardless of base tone.
            const _tickerHTML = (vals, tone) => vals.map(v => {
                const cls = v > 80 ? 'bg-error' : ('bg-' + tone);
                return `<div class="ticker-col ${cls}" style="height:${Math.max(5, (v / 100) * 30)}px"></div>`;
            }).join('');
            $('g-t').innerHTML = _tickerHTML(gH, 'primary');
            $('v-t').innerHTML = _tickerHTML(vH, 'secondary');

            // Load average (1m) — same tone-flip pattern as GPU/RAM: text-info
            // normally, text-error above 80 % so the colour tracks the ticker.
            const loadPct = (typeof d.stats.load_pct === 'number') ? d.stats.load_pct : 0;
            const loadEl = $('l-v');
            if (loadEl) { loadEl.innerText = loadPct + '%'; loadEl.className = _toneClass(loadPct, 'info'); }
            const loadParent = loadEl && loadEl.parentElement;
            if (loadParent && d.stats.load_1m != null) {
                loadParent.title = `1m: ${d.stats.load_1m.toFixed(2)} · 5m: ${d.stats.load_5m.toFixed(2)} · 15m: ${d.stats.load_15m.toFixed(2)} (load average / cpu count)`;
            }
            lH.push(loadPct);
            if (lH.length > 15) lH.shift();
            const lt = $('l-t');
            if (lt) lt.innerHTML = _tickerHTML(lH, 'info');

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
                if (dt) dt.innerHTML = _tickerHTML(dH, 'accent');
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
            const qList = $('q-list');
            const cfg = d.config || {};
            const llmModelId = (cfg.llm && cfg.llm.model_id) || '';
            // [canonicalStage, shortAcronym, displayLabel, activeVerb, tone].
            // Tone matches the model badge color for the worker that runs
            // the stage, so the pipeline strip and the Selected-Models row
            // share a visual language: Image=info(qwen), Video=success(ltx),
            // Music=secondary(heartmula), Voice=warning(qwen-tts), Concept=
            // accent(LLM), Final=accent(FINAL accent).
            const STAGES = [
                ['Concept',      'T', 'Text',  'Texting',    'accent'],
                ['Base Image',   'I', 'Image', 'Imaging',    'info'],
                ['Video Chains', 'V', 'Video', 'Videoing',   'success'],
                ['Audio',        'M', 'Music', 'Composing',  'secondary'],
                ['TTS',          'S', 'Voice', 'Voicing',    'warning'],
                ['Post Process', 'X', 'Post',  'Polishing',  'warning'],
                ['Final Merge',  'F', 'Final', 'Merging',    'accent'],
            ];
            const renderPipelineStrip = (q, opts) => {
                const isActive = !!(opts && opts.running);
                const curStep = isActive ? (opts.step || '') : null;
                const v = isActive ? ((_lastTick && _lastTick.state && _lastTick.state.video_index) || 1) : 0;
                const c = isActive ? ((_lastTick && _lastTick.state && _lastTick.state.chain_index) || 0) : 0;
                if (!isActive) return '';
                // Activity line: what the fleet is doing right now in plain
                // English.
                const activityText = curStep && _STAGE_TEXT[curStep] ? `${_STAGE_TEXT[curStep]}…` : 'working…';
                // Match the 1Hz interval handler exactly (no '⏱ '/'Σ ' prefix
                // — the labels next to the badges already convey what they
                // mean) so live updates don't flicker.
                const stageNow = _stageStartTs ? _fmtElapsedHtml(Date.now() - _stageStartTs) : _fmtElapsedHtml(0);
                const stageAvg = _stageAvgSeconds(curStep);
                const stageEtaTxt = stageAvg != null ? 'ETA ' + _fmtElapsedHtml(stageAvg * 1000) : '';
                const jobNow2 = _jobStartTs ? _fmtElapsedHtml(Date.now() - _jobStartTs) : _fmtElapsedHtml(0);
                const totalEtaSec2 = _STAGE_ORDER.map(_stageAvgSeconds).filter(x => x != null).reduce((a, b) => a + b, 0);
                const totalEtaTxt2 = totalEtaSec2 > 0 ? 'ETA ' + _fmtElapsedHtml(totalEtaSec2 * 1000) : '';
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
                    .map(([s,,label,,tone]) => {
                        const key = `${v}:${s}`;
                        const isFresh = !_displayedDoneStages.has(key);
                        if (isFresh) {
                            justCompleted.add(key);
                            _displayedDoneStages.add(key);
                        }
                        let assetBadge;
                        if (s === 'Concept') {
                            const promptText = (_lastTick && _lastTick.state && _lastTick.state.current_prompt) || '(no prompt captured)';
                            assetBadge = `<button type="button" class="badge badge-xs badge-outline cursor-pointer" title="LLM-rewritten prompt" onclick='showPromptPeek(${JSON.stringify(promptText)})'>prompt →</button>`;
                        } else {
                            const asset = _STAGE_ASSET(s, v, c);
                            assetBadge = asset
                                ? `<button type="button" class="badge badge-xs badge-outline cursor-pointer font-mono text-[9px]" title="${s} → ${asset}" onclick='openAssetInfo(${JSON.stringify(asset)})'>${asset} →</button>`
                                : '';
                        }
                        const a = actuals[s];
                        const timing = a
                            ? `<span class="font-mono text-[9px]">${_fmtElapsedHtml(a.duration_s * 1000)}</span><span class="opacity-50 text-[9px]">${a.eta_s ? ' / ETA ' + _fmtElapsedHtml(a.eta_s * 1000) : ''}</span>`
                            : '';
                        // Stage label on the LEFT, asset link + duration on
                        // the RIGHT (push with ml-auto). Reads as a list:
                        // "✓ Image                        v3_base.png  3m22s / ETA 9m"
                        const animCls = isFresh ? ' stage-just-completed' : '';
                        return `<div class="flex items-center gap-2 mt-1${animCls}" data-stage-row="${key}">
                            <span class="badge badge-xs badge-${tone} opacity-70">✓ ${label}</span>
                            <span class="ml-auto flex items-center gap-2">${assetBadge}${timing}</span>
                        </div>`;
                    }).join('');
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
                return `
                    ${completedLines ? `<div class="text-[9px] uppercase tracking-widest text-base-content/50 mt-2">Output</div>${completedLines}` : ''}
                    <div class="flex items-center gap-2 text-[10px] mt-1">
                        <span class="loading loading-spinner loading-xs text-primary"></span>
                        <span class="italic text-base-content/70 flex-1 truncate">${activityText}</span>
                        <span class="badge badge-xs badge-primary font-mono text-[9px]" data-q-stage-elapsed>${stageNow}</span>
                        ${stageEtaTxt ? `<span class="badge badge-xs badge-outline font-mono text-[9px] opacity-70" data-q-stage-eta>${stageEtaTxt}</span>` : ''}
                    </div>
                    <div class="flex items-center justify-end gap-1 mt-1 text-[9px]">
                        <span class="text-base-content/50">Total</span>
                        <span class="badge badge-xs badge-ghost font-mono" data-q-job-elapsed>${jobNow2}</span>
                        ${totalEtaTxt2 ? `<span class="badge badge-xs badge-outline font-mono opacity-70" data-q-job-eta>${totalEtaTxt2}</span>` : ''}
                    </div>
                `;
            };
            const renderItem = (q, opts) => {
                const snap = (q && q.config_snapshot) || cfg;
                const activeRole = (opts && opts.running && opts.step) ? _STAGE_ROLE[opts.step] : null;
                const badges = _configModelBadges(snap, llmModelId, activeRole, q.ts || 0);
                const meta = `${_htmlEscape(snap.size || '1:1')}·${snap.frames || 17}f`;
                const promptEsc = _htmlEscape(q.prompt || '');
                const isActive = !!(opts && opts.running);
                const isCancelled = q.status === 'cancelled';
                const infBadge = q.infinity
                    ? `<span class="badge badge-xs badge-primary text-base font-bold" title="Infinity — re-queues itself after every completion">♾</span>`
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
                const menuHTML = isCancelled
                    ? `<div class="dropdown dropdown-end" onclick="${stop}">
                        <label tabindex="0" class="btn btn-ghost btn-xs btn-square" title="Actions" onclick="${stop}">⋯</label>
                        <ul tabindex="0" class="dropdown-content menu menu-xs p-1 shadow bg-base-300 rounded-box z-10 w-40">
                            <li><a onclick="event.stopPropagation();requeueItem(${q.ts || 0})">↻ Re-queue</a></li>
                        </ul>
                       </div>`
                    : `<div class="dropdown dropdown-end" onclick="${stop}">
                        <label tabindex="0" class="btn btn-ghost btn-xs btn-square" title="Actions" onclick="${stop}">⋯</label>
                        <ul tabindex="0" class="dropdown-content menu menu-xs p-1 shadow bg-base-300 rounded-box z-10 w-40">
                            <li><a onclick='event.stopPropagation();editItem(${q.ts || 0}, ${promptForJs})'>✎ Edit prompt</a></li>
                            <li><a onclick="event.stopPropagation();toggleItemInfinity(${q.ts || 0})">${infToggleLabel}</a></li>
                            <li><a onclick="event.stopPropagation();cancelItem(${q.ts || 0})" class="text-error">✕ Cancel</a></li>
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
                return `<li class="${cls}" data-q-ts="${q.ts || 0}" data-q-status="${isCancelled ? 'cancelled' : (isActive ? 'active' : 'pending')}">
                    <details ${isActive ? 'open' : ''}>
                        <summary class="cursor-pointer p-2 flex items-center gap-2 text-xs">
                            ${statusChip}${infBadge}
                            <span class="font-semibold truncate flex-1${isCancelled ? ' line-through' : ''}" title="${promptEsc}">${promptEsc}</span>
                            ${menuHTML}
                        </summary>
                        <div class="px-2 pb-2 pt-0 flex flex-col gap-1 border-t border-base-300/50">
                            <div class="flex items-center gap-1 flex-wrap text-[10px] mt-1">
                                ${badges.join('')}
                                <span class="text-base-content/50 font-mono ml-auto">${meta}</span>
                            </div>
                            ${stripHTML}
                        </div>
                    </details>
                </li>`;
            };
            if (qList) {
                if (!qLen) {
                    qList.innerHTML = '<li class="text-[10px] text-base-content/40 italic p-2">queue empty — click Generate to add</li>';
                } else {
                    const items = [];
                    if (isRunning) {
                        const runItem = {
                            prompt: d.state.current_prompt || '(running)',
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
                    // Cap the inline queue at 6 items so the card stays
                    // glanceable; the "View all →" button opens the drawer
                    // which shows the entire queue.
                    // Pending items first (max 6 visible inline).
                    const pendingOnly = visibleQueue.filter(q => q.status !== 'done');
                    items.push(...pendingOnly.slice(0, 6).map(q => renderItem(q, {})));
                    // Done items (newest first) — full audit log of completed
                    // iters, each with its actual duration and an asset link.
                    // Limit to most-recent 6 inline; the drawer shows them all.
                    const doneOnly = visibleQueue.filter(q => q.status === 'done')
                        .slice().sort((a, b) => (b.completed_ts || 0) - (a.completed_ts || 0));
                    doneOnly.slice(0, 6).forEach(q => {
                        items.push(_renderDoneItem(q));
                    });
                    qList.innerHTML = items.join('');
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
            _lastTick = d;
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
    const sys = {
        image: 'Rewrite the prompt as a detailed visual still-frame description for an AI image generator. Lighting, texture, mood. Under 60 words. Output ONLY the rewritten prompt.',
        video: 'Rewrite the prompt as a motion/camera description for an AI video generator. Camera movement, pacing, transitions. Under 60 words. Output ONLY the rewritten prompt.',
        music: 'Rewrite the prompt as a short mood/genre description suitable for a music generator (instruments, tempo, vibe). Under 30 words. Output ONLY the description.',
        tts: 'Rewrite the prompt as a one or two sentence voiceover line spoken in first or third person. Output ONLY the line.',
    }[stage] || 'Rewrite the prompt for ' + stage;
    const orig = ta.value;
    ta.value = '✨ thinking...';
    ta.disabled = true;
    try {
        const res = await fetch('/enhance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: `${sys}\n\nSubject: ${seed}` }),
        });
        const r = await res.json();
        ta.value = (r.suggestion || orig).trim();
    } catch (e) {
        ta.value = orig;
    } finally {
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
    const locked = _lockedList();
    const preview = $('fanout-preview');
    if (preview) preview.classList.remove('hidden');
    const warn = $('fanout-warn');
    if (warn) { warn.classList.add('hidden'); warn.innerText = ''; }
    // Loading placeholder — drop a "thinking" hint into each of the 4 boxes
    // so the preview reflects active work even before the LLM responds.
    STAGE_NAMES.forEach(n => {
        const box = $('fanout-' + n);
        if (box) box.textContent = '✨ thinking...';
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
        console.warn('inject: no prompt available — Subjects textarea empty and no stage overrides set');
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
        if (stageConcat) f.append('stage_prompts', JSON.stringify(stages));
        await fetch('/inject', { method: 'POST', body: f });
    }
    // Only blank the per-stage overrides — leave the Subjects textarea alone
    // so the user can re-queue the same set quickly if they want to.
    ['p-image', 'p-video', 'p-music', 'p-tts', 'p-in'].forEach(id => { if ($(id)) $(id).value = ''; });
}

// Deprecated: the standalone "Will use" badge row was removed; the RAM
// estimator breakdown now renders role + model + GB together. Keep this
// stub so existing callers don't error.
function _renderSubjectsModels() {}

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
    const baseVal  = _resolve('cfg-base',  'cfg-base-slopped');
    const audioVal = _resolve('cfg-audio', 'cfg-audio-slopped');
    const ttsVal   = _resolve('cfg-tts',   'cfg-tts-slopped');
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
    if ($('cfg-tier')) body.tier = $('cfg-tier').value;
    if ($('cfg-consolidation')) body.consolidation = $('cfg-consolidation').value;
    if ($('cfg-music-gain-db')) body.music_gain_db = parseInt($('cfg-music-gain-db').value, 10);
    if ($('cfg-fade-s')) body.fade_s = parseFloat($('cfg-fade-s').value);
    if ($('chaos-on')) body.chaos_mode = $('chaos-on').checked;
    if ($('when-idle-on')) body.when_idle = $('when-idle-on').checked;
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

async function generateTts() {
    const text = $('tts-in') ? $('tts-in').value.trim() : '';
    const voice = $('tts-voice') ? $('tts-voice').value : 'ryan';
    const statusEl = $('tts-status');
    const previewEl = $('tts-preview');
    if (!text) { if (statusEl) statusEl.innerText = 'empty'; return; }
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

function onModelSelectChanged() {
    const sel = $('set-model');
    const custom = $('set-model-custom');
    if (!sel || !custom) return;
    if (sel.value === '__custom__') {
        custom.classList.remove('hidden');
        custom.focus();
    } else {
        custom.classList.add('hidden');
    }
}

async function reloadModels() {
    const sel = $('set-model');
    if (!sel) return;
    const base = computeBaseUrl();
    const provider = $('set-provider').value;
    const apiKey = $('set-api-key').value;
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
    const sel = $('set-model');
    let model_id = sel.value;
    if (model_id === '__custom__') model_id = $('set-model-custom').value.trim();
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

async function openSettings() {
    const modal = $('settings-modal');
    if (!modal) return;
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
        renderAutoSuspendList(sr.auto_suspend);
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
        const modelSel = $('set-model');
        modelSel.dataset.selected = llm.model_id || '';
        modelSel.innerHTML = '';
        if (llm.model_id) {
            const o = document.createElement('option');
            o.value = llm.model_id; o.innerText = llm.model_id; o.selected = true;
            modelSel.appendChild(o);
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
        modal.showModal();
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
];

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
    const sel = $('set-model');
    let model_id = sel.value;
    if (model_id === '__custom__') model_id = $('set-model-custom').value.trim();
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
    };
    const fleetPrompt = $('set-fleet-prompt');
    if (fleetPrompt) {
        // Empty string is meaningful here (server interprets it as "use built-in default").
        body.philosophical_prompt = fleetPrompt.value;
    }
    const sugUseSub = $('set-suggest-use-subjects');
    if (sugUseSub) body.suggest_use_subjects = !!sugUseSub.checked;
    const sugCustom = $('set-suggest-custom-prompt');
    if (sugCustom) body.suggest_custom_prompt = sugCustom.value;
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
    const modal = $('settings-modal');
    if (modal && modal.close) modal.close();
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

document.addEventListener('DOMContentLoaded', () => {
    const p = $('set-provider');
    if (p) p.addEventListener('change', applyProviderDefaults);
});

// -------------------- Single-page layout helpers --------------------

function openPipeline() {
  const d = document.getElementById('pipeline-modal');
  if (d && d.showModal) d.showModal();
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

function _updateStartBtn() {
  const b = document.getElementById('btn-start-stop');
  if (!b) return;
  // Each click queues a new item. The Infinity toggle in the Generation tab
  // makes the queued item re-loop after each completion (cancel via the ✕ on
  // its queue row). Never use this button to stop running jobs.
  const inf = document.getElementById('inf-on');
  const now = document.getElementById('now-on');
  const term = document.getElementById('term-on');
  const termOn = !!(term && term.checked);
  const infOn = !!(inf && inf.checked);
  if (termOn && infOn) { b.textContent = 'Terminate and Queue Infinite Slop'; return; }
  if (termOn) { b.textContent = 'Terminate & Queue'; return; }
  if (infOn) {
    b.textContent = (now && now.checked) ? 'Queue Infinite Slop (now)' : 'Queue Infinite Slop';
    return;
  }
  b.textContent = (now && now.checked) ? 'Queue Now' : 'Queue Slop';
}

// Polymorphic + When Idle only make sense when the fleet is looping —
// gray them out otherwise.
function _updateChaosEnabled() {
  const inf = document.getElementById('inf-on');
  const enabled = !!(inf && inf.checked);
  [
    ['chaos-on', 'chaos-row'],
    ['when-idle-on', 'when-idle-row'],
  ].forEach(([toggleId, rowId]) => {
    const t = document.getElementById(toggleId);
    const r = document.getElementById(rowId);
    if (t) t.disabled = !enabled;
    if (r) r.classList.toggle('opacity-40', !enabled);
  });
}

// Back-compat shim — Terminate is now a flat flag; older callers may still
// invoke this. No-op preserves the call site.
function _updateTerminateEnabled() {}

// Build a one-line summary of the Generation tab toggles for the header pill,
// so the user can glance at it without opening the modal.
function _updateGenModePill() {
  const pill = document.getElementById('gen-mode-pill');
  if (!pill) return;
  const inf = document.getElementById('inf-on');
  const idle = document.getElementById('when-idle-on');
  const poly = document.getElementById('chaos-on');
  const now = document.getElementById('now-on');
  const term = document.getElementById('term-on');
  const conc = document.getElementById('concurrent-on');
  const parts = [];
  parts.push(inf && inf.checked ? '♾ Infinity' : '▶ Single');
  if (inf && inf.checked && idle && idle.checked) parts.push('+idle');
  if (inf && inf.checked && poly && poly.checked) parts.push('+poly');
  if (term && term.checked) parts.push('🛑 terminate');
  else if (now && now.checked) parts.push('⏯ now');
  else parts.push('queue');
  if (conc && conc.checked) parts.push('+concurrent');
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
    document.querySelectorAll('#subject-chips button[data-suggest]').forEach(btn => {
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

// Build a single suggestion chip <button>. Pass repeat=true to mark the
// chip as filler from a wrap-around batch — the click handler still works
// identically; the data-attribute lets _refillSuggestChips drop fillers
// before re-measuring on resize.
function _buildSuggestChip(s, repeat) {
    const b = document.createElement('button');
    b.className = 'btn btn-outline btn-primary btn-xs normal-case';
    b.textContent = s;
    b.title = 'Click: append · Shift+click: replace · ✓ = already in your subjects';
    b.dataset.suggest = s;
    if (repeat) b.dataset.suggestionRepeat = 'true';
    b.addEventListener('click', (e) => {
        const ta = document.getElementById('p-core');
        if (!ta) return;
        const present = ta.value.split(/\r?\n/).map(x => x.trim().toLowerCase()).includes(s.toLowerCase());
        if (e.shiftKey) ta.value = s;
        else if (present) return;
        else ta.value = (ta.value.trim() ? ta.value.trimEnd() + '\n' : '') + s;
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        updatePipeline();
        _refreshChipHighlights();
    });
    return b;
}

// Most recently rendered base batch — used by _refillSuggestChips on
// resize so we can re-fill from the same source list without re-fetching.
let _lastSuggestBatch = [];

// Pixel threshold under which we consider the gap "filled" — also the
// minimum gap we'll leave at the bottom so chips don't touch the next
// element.
const _SUGGEST_FILL_THRESHOLD = 32;
const _SUGGEST_FILL_MAX_REPEATS = 4;

// Measure the vertical gap between the bottom of the chips container and
// the top of the next sibling (the Start button row). The user wants the
// chip area to visually fill this whitespace so there's no awkward gap
// between the last chip row and the bottom-anchored Start button.
function _suggestGapPx(box) {
    const anchor = box && box.nextElementSibling;
    if (!anchor) return 0;
    const boxRect = box.getBoundingClientRect();
    const anchorRect = anchor.getBoundingClientRect();
    return Math.max(0, anchorRect.top - boxRect.bottom);
}

// Append repeats of the base batch until the gap below the chips is
// smaller than the threshold or we hit the hard repeat cap. Filler chips
// are tagged with data-suggestion-repeat="true" so we can strip them on
// resize before re-measuring.
function _refillSuggestChips() {
    const box = document.getElementById('subject-chips');
    if (!box || !_lastSuggestBatch.length) return;
    // Strip previous fillers — keep only the original batch.
    box.querySelectorAll('button[data-suggestion-repeat="true"]').forEach(el => el.remove());
    let repeats = 0;
    while (repeats < _SUGGEST_FILL_MAX_REPEATS) {
        const gap = _suggestGapPx(box);
        if (gap < _SUGGEST_FILL_THRESHOLD) break;
        _lastSuggestBatch.forEach(s => box.appendChild(_buildSuggestChip(s, true)));
        repeats += 1;
    }
    _refreshChipHighlights();
}

function _renderSuggestChips(arr) {
    const box = document.getElementById('subject-chips');
    if (!box) return;
    if (!arr.length) {
        box.innerHTML = '<span class="text-[10px] italic text-warning">no suggestions</span>';
        _lastSuggestBatch = [];
        return;
    }
    box.innerHTML = '';
    _lastSuggestBatch = arr.slice();
    arr.forEach(s => box.appendChild(_buildSuggestChip(s, false)));
    // Defer the fill measurement until layout settles — the chips have
    // just been inserted and getBoundingClientRect on a freshly attached
    // node is fine, but a rAF gives the browser a clean frame.
    requestAnimationFrame(() => _refillSuggestChips());
    _refreshChipHighlights();
}

// Watch for viewport / card-size changes so the fill recalculates when
// the user drags the split-row divider or resizes the window. Set up
// once — multiple calls are no-ops.
let _suggestResizeWired = false;
function _wireSuggestResize() {
    if (_suggestResizeWired) return;
    const box = document.getElementById('subject-chips');
    if (!box) return;
    _suggestResizeWired = true;
    let pending = false;
    const schedule = () => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => { pending = false; _refillSuggestChips(); });
    };
    window.addEventListener('resize', schedule);
    if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(schedule);
        // Observe the card-body so divider drags / textarea growth retrigger.
        const card = box.closest('.card-body') || box.parentElement;
        if (card) ro.observe(card);
    }
}
document.addEventListener('DOMContentLoaded', _wireSuggestResize);

// Render cached suggestions from localStorage if any exist. Returns true
// if it rendered, false if cache was empty.
function _renderCachedSuggestions() {
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
async function regenSuggestions(n = 6) {
  const box = document.getElementById('subject-chips');
  if (!box) return;
  box.innerHTML = '<span class="loading loading-dots loading-xs"></span>';
  try {
    // Forward the current Subjects textarea so the server can match
    // style/theme when `suggest_use_subjects` is enabled. The server ignores
    // it when the toggle is off, so it's safe to always send.
    const subjects = (($('p-core') && $('p-core').value) || '').trim();
    const qs = '?n=' + n + (subjects ? '&subjects=' + encodeURIComponent(subjects) : '');
    const r = await fetch('/subjects/suggest' + qs);
    const data = await r.json();
    const arr = data.suggestions || [];
    if (!arr.length) { box.innerHTML = '<span class="text-[10px] italic text-warning">LLM unreachable</span>'; return; }
    try { localStorage.setItem(_SUGGEST_CACHE_KEY, JSON.stringify(arr)); } catch {}
    // Manual 🎲 click: render fresh AND seed the carousel ring so paging
    // back through the batch history works. _renderSuggestChips clears the
    // strip first; the seed call below records the same batch as a fresh
    // ring entry so the carousel doesn't think the strip is empty.
    _renderSuggestChips(arr);
    if (typeof _seedSuggestBatchFromRender === 'function') {
        _seedSuggestBatchFromRender(arr);
    }
  } catch (e) {
    box.innerHTML = '<span class="text-[10px] italic text-error">error</span>';
  }
}

// ---------------------------------------------------------------------------
// Subjects suggestion carousel — hover-overlay paging (restoring PR #56).
//
// The strip (#subject-chips) is wrapped in a frame (#subject-chips-frame)
// with two 32 px gradient overlays that fade in on parent-hover. Click the
// right overlay to page right one client-width; if already at the right
// edge, fetch a new batch and append it (with a divider). Click the left
// overlay to page left.
//
// Composition with the surviving fill scaffold (_refillSuggestChips):
// _renderSuggestChips replaces the strip wholesale on cache-load and on
// 🎲 fetch, then runs the fill repeats. The carousel ring buffer
// (_suggestBatches) records each appended batch separately so paging back
// through earlier batches restores them; the cap of 10 prevents unbounded
// DOM growth. Filler "repeat" chips are not stored in the ring — they are
// regenerated by _refillSuggestChips on resize.
//
// Gating: the right-overlay fetch path consults _isGpuIdleEnough() before
// hitting /subjects/suggest. The 🎲 button path is intentionally NOT gated
// (manual user intent always wins).
// ---------------------------------------------------------------------------

const _SUGGEST_BATCH_CAP = 10;
// Each entry: { id: number, items: string[], domStart: number }
// where domStart is the index of the first chip of this batch in the strip
// (used to slice the strip down when the cap is exceeded).
let _suggestBatches = [];

// Called by _renderSuggestChips' callers (cache hydrate, 🎲 fetch) to
// reset the ring after a wholesale strip rebuild — the strip now contains
// exactly one batch.
function _seedSuggestBatchFromRender(items) {
    _suggestBatches = [{ id: Date.now(), items: items.slice(), domStart: 0 }];
}

function _suggestFrame() { return document.getElementById('subject-chips-frame'); }
function _suggestStrip() { return document.getElementById('subject-chips'); }

// Append a new batch to the strip — inserts a divider between previous
// chips and new ones, scrolls the first new chip into view, and prunes
// oldest batches when the ring exceeds _SUGGEST_BATCH_CAP.
function _appendSuggestBatch(batch) {
    const strip = _suggestStrip();
    if (!strip || !batch || !Array.isArray(batch.items) || !batch.items.length) return;
    // Strip "click 🎲" placeholder if present.
    const placeholder = strip.querySelector('span.italic');
    if (placeholder && !strip.querySelector('button[data-suggest]')) {
        strip.innerHTML = '';
        _suggestBatches = [];
    }
    // Drop any filler-repeat chips before appending — _refillSuggestChips
    // will regenerate them after the new batch is in.
    strip.querySelectorAll('button[data-suggestion-repeat="true"]').forEach(el => el.remove());

    const domStart = strip.children.length;
    if (_suggestBatches.length > 0) {
        const div = document.createElement('span');
        div.className = 'suggest-batch-divider';
        div.setAttribute('aria-hidden', 'true');
        strip.appendChild(div);
    }
    const firstChip = _buildSuggestChip(batch.items[0], false);
    strip.appendChild(firstChip);
    for (let i = 1; i < batch.items.length; i++) {
        strip.appendChild(_buildSuggestChip(batch.items[i], false));
    }
    _suggestBatches.push({ id: batch.id || Date.now(), items: batch.items.slice(), domStart });

    // Prune oldest batches past the cap by removing their DOM nodes.
    while (_suggestBatches.length > _SUGGEST_BATCH_CAP) {
        const dropped = _suggestBatches.shift();
        const next = _suggestBatches[0];
        const cutEnd = next ? next.domStart : strip.children.length;
        // Remove children [0, cutEnd) — but cutEnd indices are pre-prune;
        // since the dropped batch is the first, indices 0..(cutEnd-1) are
        // exactly its chips + the trailing divider.
        for (let i = 0; i < cutEnd && strip.firstElementChild; i++) {
            strip.removeChild(strip.firstElementChild);
        }
        // Re-base remaining batches' domStart.
        _suggestBatches.forEach(b => { b.domStart = Math.max(0, b.domStart - cutEnd); });
    }

    // _lastSuggestBatch drives _refillSuggestChips' repeat fill — keep it
    // pointed at the most recently appended batch so resizes still fill.
    _lastSuggestBatch = batch.items.slice();

    // Scroll the new batch's first chip into view (right edge).
    requestAnimationFrame(() => {
        try { firstChip.scrollIntoView({ behavior: 'smooth', inline: 'start', block: 'nearest' }); } catch {}
        _refreshChipHighlights();
    });
    // Newly-appended chips can extend the scrollWidth past clientWidth
    // (revealing the right arrow) or, after pruning, contract it (hiding
    // both arrows). Refresh the .can-scroll classes now and again on the
    // next frame, after the smooth-scroll above has updated scrollLeft.
    _updateOverlayVisibility();
    requestAnimationFrame(_updateOverlayVisibility);
}

// Toggle the .can-scroll class on each overlay based on whether the strip
// actually has content to scroll to in that direction. CSS gates the
// hover-reveal on .can-scroll, so an overlay without it stays invisible
// even on hover. The 4-pixel slack on each edge avoids flicker at the
// exact start/end of the scroll range (sub-pixel rounding, smooth-scroll
// landings, and the scroll-snap padding can leave scrollLeft a hair off
// of zero or scrollWidth-clientWidth even at "true" edge).
function _updateOverlayVisibility() {
    const frame = document.getElementById('subject-chips-frame');
    const left = document.getElementById('subject-chips-overlay-left');
    const right = document.getElementById('subject-chips-overlay-right');
    const strip = document.getElementById('subject-chips');
    if (!frame || !left || !right || !strip) return;
    const canLeft = strip.scrollLeft > 4;
    const canRight = strip.scrollLeft + strip.clientWidth < strip.scrollWidth - 4;
    left.classList.toggle('can-scroll', canLeft);
    right.classList.toggle('can-scroll', canRight);
}

// Are we within ~8 px of the right edge of the strip's scroll range?
function _atRightEdge() {
    const strip = _suggestStrip();
    if (!strip) return false;
    return strip.scrollLeft + strip.clientWidth >= strip.scrollWidth - 8;
}

// Wire the overlay click / keyboard handlers. Idempotent — safe to call
// multiple times.
let _suggestCarouselWired = false;
function _wireSuggestCarousel() {
    if (_suggestCarouselWired) return;
    const left = document.getElementById('subject-chips-overlay-left');
    const right = document.getElementById('subject-chips-overlay-right');
    const strip = _suggestStrip();
    if (!left || !right || !strip) return;
    _suggestCarouselWired = true;

    const pageLeft = () => {
        strip.scrollBy({ left: -strip.clientWidth, behavior: 'smooth' });
    };
    const pageRight = () => {
        if (!_atRightEdge()) {
            strip.scrollBy({ left: strip.clientWidth, behavior: 'smooth' });
            return;
        }
        // Right-edge: prefer prefetched batch, fall back to fresh fetch.
        if (typeof _consumePrefetchedBatch === 'function') {
            const pre = _consumePrefetchedBatch();
            if (pre) {
                _appendSuggestBatch({ id: Date.now(), items: pre });
                if (typeof _maybePrefetch === 'function') _maybePrefetch();
                return;
            }
        }
        // No prefetch ready — gate fresh fetch on GPU idle. Manual 🎲 stays ungated.
        if (typeof _isGpuIdleEnough === 'function' && _gpuPctHistory.length > 0 && !_isGpuIdleEnough()) {
            console.info('skipping right-overlay fetch: GPU busy');
            right.classList.add('is-loading');
            setTimeout(() => right.classList.remove('is-loading'), 600);
            return;
        }
        right.classList.add('is-loading');
        const subjects = (($('p-core') && $('p-core').value) || '').trim();
        const qs = '?n=6' + (subjects ? '&subjects=' + encodeURIComponent(subjects) : '');
        fetch('/subjects/suggest' + qs)
            .then(r => r.json())
            .then(d => {
                if (d && Array.isArray(d.suggestions) && d.suggestions.length) {
                    _appendSuggestBatch({ id: Date.now(), items: d.suggestions });
                }
            })
            .catch(() => {})
            .finally(() => { right.classList.remove('is-loading'); });
    };

    const keyHandler = (fn) => (e) => {
        if (e.type === 'click' || e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fn();
        }
    };
    left.addEventListener('click', keyHandler(pageLeft));
    left.addEventListener('keydown', keyHandler(pageLeft));
    right.addEventListener('click', keyHandler(pageRight));
    right.addEventListener('keydown', keyHandler(pageRight));

    // Drive overlay visibility from actual scroll state — the CSS only
    // reveals overlays that have .can-scroll, so without these listeners
    // the arrows would never appear. rAF-coalesce the scroll listener so
    // we don't thrash classList during smooth-scroll.
    let _ovRaf = 0;
    strip.addEventListener('scroll', () => {
        if (_ovRaf) return;
        _ovRaf = requestAnimationFrame(() => {
            _ovRaf = 0;
            _updateOverlayVisibility();
        });
    }, { passive: true });
    window.addEventListener('resize', _updateOverlayVisibility);
    _updateOverlayVisibility();
}
document.addEventListener('DOMContentLoaded', _wireSuggestCarousel);

// ---------------------------------------------------------------------------
// Subjects suggestion prefetch — preemptive cache for carousel paging
// (restoring PR #57). Driven entirely by hover/scroll/idle signals on the
// Subjects card; consumed by the carousel right-overlay click handler above
// (_consumePrefetchedBatch). Every fetch path here is gated by
// _isGpuIdleEnough() so we don't compete with running pipelines / ad-hoc
// GPU users (manual ComfyUI runs, etc.).
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

function _consumePrefetchedBatch() {
    const next = _prefetchedBatches.shift();
    if (next) _prefetchStats.consumed += 1;
    return next || null;
}

function _maybePrefetch() {
    _prefetchStats.triggered += 1;
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
    const qs = '?n=6' + (subjects ? '&subjects=' + encodeURIComponent(subjects) : '');
    fetch('/subjects/suggest' + qs)
        .then(r => r.json())
        .then(d => {
            if (d && Array.isArray(d.suggestions) && d.suggestions.length) {
                _prefetchedBatches.push(d.suggestions);
                _prefetchStats.fetched += 1;
            } else {
                _prefetchBackoffUntil = Date.now() + _PREFETCH_BACKOFF_MS;
            }
        })
        .catch(() => { _prefetchBackoffUntil = Date.now() + _PREFETCH_BACKOFF_MS; })
        .finally(() => { _prefetchInflight = false; });
}

function _resetPrefetchIdleTimer() {
    if (_prefetchIdleTimer) clearTimeout(_prefetchIdleTimer);
    if (document.hidden) return;
    _prefetchIdleTimer = setTimeout(() => { _maybePrefetch(); }, _PREFETCH_IDLE_TRIGGER_MS);
}

let _suggestPrefetchWired = false;
function _wireSuggestPrefetch() {
    if (_suggestPrefetchWired) return;
    const frame = document.getElementById('subject-chips-frame');
    const strip = document.getElementById('subject-chips');
    const right = document.getElementById('subject-chips-overlay-right');
    const ta = document.getElementById('p-core');
    if (!frame || !strip) return;
    _suggestPrefetchWired = true;

    frame.addEventListener('pointerenter', () => _maybePrefetch());
    if (right) right.addEventListener('pointerenter', () => _maybePrefetch());

    // Scroll-near-right-edge trigger (within 1.5 viewports of the right).
    strip.addEventListener('scroll', () => {
        const remaining = strip.scrollWidth - (strip.scrollLeft + strip.clientWidth);
        if (remaining < strip.clientWidth * 1.5) _maybePrefetch();
    });

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

// Hide the entire carousel frame (not just the strip) when the surviving
// fill scaffold says there's no vertical room. We piggyback on the
// existing _refillSuggestChips path: when called, sync frame visibility
// to the strip's computed display.
const _origRefillSuggestChips = typeof _refillSuggestChips === 'function' ? _refillSuggestChips : null;
if (_origRefillSuggestChips) {
    _refillSuggestChips = function () {
        _origRefillSuggestChips();
        const frame = _suggestFrame();
        const strip = _suggestStrip();
        if (frame && strip) {
            // If the strip is hidden by other layout logic, hide the frame too.
            const hidden = strip.style.display === 'none';
            frame.style.display = hidden ? 'none' : '';
        }
    };
}

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
    chainCounter.textContent = (state.step === 'Video Chains' && state.chain_index)
      ? `${state.chain_index}/${state.total_chains}` : '';
  }
}

// No-op stubs for reference-modal inline handlers that don't yet exist client-side.
if (typeof window._onAudioChanged !== 'function') {
  window._onAudioChanged = function () { /* reserved for future audio-dependent UI */ };
}

connect();
_wireLockListeners();

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
        const upper  = document.getElementById('ui-split-upper');
        const lower  = document.getElementById('ui-split-lower');
        if (!handle || !upper || !lower) return;
        const KEY = 'slopfinity_ui_split_upper_pct';
        // Restore stored fraction (sanity-bounded so a corrupt value can't
        // collapse a pane to invisibility).
        // Helper — broadcast that the panes resized so internal autogrow /
        // re-layout code (textarea autogrow, suggestion-chip filler, etc.)
        // can recompute against the new available height.
        const _emitSplitResize = () => {
            try { window.dispatchEvent(new Event('resize')); } catch (_) {}
            if (typeof window._autogrowSubjects === 'function') {
                try { window._autogrowSubjects(); } catch (_) {}
            }
        };
        const stored = parseFloat(localStorage.getItem(KEY));
        if (!Number.isNaN(stored) && stored > 0.05 && stored < 0.95) {
            upper.style.flex = `0 0 ${stored * 100}%`;
            lower.style.flex = '1 1 0';
        }
        let dragging = false;
        let startY = 0;
        let startUpperPx = 0;
        let containerPx = 0;
        handle.addEventListener('pointerdown', (e) => {
            // Skip on viewports where the media query made the handle inert.
            if (window.matchMedia('(max-width: 768px)').matches) return;
            dragging = true;
            startY = e.clientY;
            startUpperPx = upper.getBoundingClientRect().height;
            containerPx = handle.parentElement.getBoundingClientRect().height;
            try { handle.setPointerCapture(e.pointerId); } catch (_) {}
            handle.classList.add('dragging');
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });
        handle.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            // Reserve 120 px min for each pane + 8 px for the handle itself.
            const newUpper = Math.max(120, Math.min(containerPx - 120 - 8, startUpperPx + (e.clientY - startY)));
            const pct = newUpper / containerPx;
            upper.style.flex = `0 0 ${pct * 100}%`;
            lower.style.flex = '1 1 0';
            _emitSplitResize();
        });
        const stop = (e) => {
            if (!dragging) return;
            dragging = false;
            try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
            handle.classList.remove('dragging');
            document.body.style.userSelect = '';
            const pct = upper.getBoundingClientRect().height / handle.parentElement.getBoundingClientRect().height;
            if (pct > 0.05 && pct < 0.95) localStorage.setItem(KEY, String(pct));
        };
        handle.addEventListener('pointerup', stop);
        handle.addEventListener('pointercancel', stop);
        // Double-click resets to 50/50 (= clear the override + remove storage).
        handle.addEventListener('dblclick', () => {
            upper.style.flex = '1 1 0';
            lower.style.flex = '1 1 0';
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
    const partBadge = meta.part
        ? `<span class="badge badge-xs badge-ghost">part ${meta.part}</span>`
        : '';
    const autoplayAttr = opts.autoplay ? 'autoplay' : '';
    let media;
    if (isV) {
        media = `<figure class="bg-black aspect-video flex items-center justify-center overflow-hidden"><video controls ${autoplayAttr} muted loop preload="metadata" class="w-full h-full object-contain"><source src="/files/${file}"></video></figure>`;
    } else if (isWav) {
        media = `<figure class="bg-black aspect-video flex items-center justify-center overflow-hidden"><audio controls class="w-full mx-2"><source src="/files/${file}"></audio></figure>`;
    } else {
        media = `<figure class="bg-black aspect-video flex items-center justify-center overflow-hidden"><img src="/files/${file}" class="w-full h-full object-contain" loading="lazy"></figure>`;
    }
    c.innerHTML = `${media}
        <div class="card-body !p-2 bg-base-200/60 gap-1">
            <div class="flex flex-wrap items-center gap-1">
                <span class="badge badge-xs ${meta.color}">${meta.label}</span>
                ${partBadge}
            </div>
            <span class="text-[10px] font-mono text-base-content/60 truncate" title="${file}">${file}</span>
        </div>`;
    return c;
}

(function wireInfiniteScroll() {
    const init = () => {
        const sentinel = document.getElementById('preview-grid-sentinel');
        const grid = document.getElementById('preview-grid');
        const lower = document.getElementById('ui-split-lower');
        if (!sentinel || !grid) return;

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

        // The lower pane is the actual scroll container (PR #74's splitter
        // gives it `overflow-y: auto` via .ui-split-pane). Falling back to
        // the viewport (`root: null`) keeps the observer sensible if the
        // splitter element isn't present.
        const observer = new IntersectionObserver((entries) => {
            if (entries.some(e => e.isIntersecting)) loadMore();
        }, {
            root: lower || null,
            rootMargin: '200px',
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
