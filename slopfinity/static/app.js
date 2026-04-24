// Slopfinity dashboard client.

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

function updateStorage(storage) {
    if (!storage) return;
    const ids = { '/': 'st-root', '/mnt/data': 'st-data', '/mnt/downloads': 'st-downloads' };
    storage.forEach(s => {
        const el = $(ids[s.mount]);
        if (!el) return;
        const badge = el.querySelector('.st-badge');
        const val = el.querySelector('.st-val');
        if (val) val.innerText = `${s.used_gb} / ${s.total_gb} GB`;
        if (badge) {
            badge.className = 'st-badge badge badge-sm ' + statusClass(s.status);
            badge.innerText = s.pct + '%';
        }
    });
}

function updateRam(ram) {
    if (!ram) return;
    const el = $('ram-est');
    if (!el) return;
    const txt = el.querySelector('.ram-txt');
    const bar = el.querySelector('.ram-bar');
    if (txt) txt.innerText = `${ram.estimated_gb} / ${ram.budget_gb || 128} GB unified`;
    if (bar) {
        bar.className = 'ram-bar progress w-full ' + progressClass(ram.status);
        bar.value = Math.min(100, (ram.estimated_gb / (ram.budget_gb || 128)) * 100);
    }
    el.className = 'alert p-2 ' + (ram.status === 'danger' ? 'alert-error' : ram.status === 'warn' ? 'alert-warning' : 'alert-success');
    el.id = 'ram-est';
}

function schedBadgeClass(type) {
    if (type === 'stage_start') return 'badge-info';
    if (type === 'stage_end') return 'badge-success';
    if (type === 'budget_block') return 'badge-warning';
    if (type === 'oom_retry') return 'badge-error';
    if (type === 'emergency_free') return 'badge-error';
    return 'badge-ghost';
}

function updateScheduler(sc) {
    if (!sc) return;
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
    const events = (sc.events || []).slice(-5);
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
        const w = $('refresh-wrapper');
        if (w) w.style.display = 'none';
    };
    ws.onmessage = e => {
        const d = JSON.parse(e.data);
        if (d.type === 'state') {
            $('g-v').innerText = d.stats.gpu + '%';
            $('v-v').innerText = d.stats.vram + '%';
            $('r-v').innerText = d.stats.ram_u + ' / ' + Math.round(d.stats.ram_t) + ' GB';

            gH.push(d.stats.gpu); vH.push(d.stats.vram);
            if (gH.length > 15) gH.shift();
            if (vH.length > 15) vH.shift();
            $('g-t').innerHTML = gH.map(v => `<div class="ticker-col" style="height:${Math.max(5, (v / 100) * 30)}px; background:${v > 80 ? '#ff5555' : '#ff79c6'}"></div>`).join('');
            $('v-t').innerHTML = vH.map(v => `<div class="ticker-col" style="height:${Math.max(5, (v / 100) * 30)}px; background:#bd93f9"></div>`).join('');

            $('h-m').innerText = d.state.mode;
            $('h-pr').innerText = '"' + d.state.current_prompt + '"';
            $('h-c').innerText = `V ${d.state.video_index}/${d.state.total_videos} | C ${d.state.chain_index}/${d.state.total_chains}`;
            $('h-p').value = (d.state.video_index / Math.max(1, d.state.total_videos)) * 100;

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

            $('q-count').innerText = d.queue.length;
            $('q-list').innerHTML = d.queue.length
                ? d.queue.map(q => `<div class="bg-base-300 p-2 rounded mb-2 text-xs border border-base-200">${(q.prompt || '').substring(0, 60)}...</div>`).join('')
                : '<div class="text-xs text-base-content/50 italic text-center p-4">Queue empty</div>';

            updateStorage(d.storage);
            updateRam(d.ram);
            updateScheduler(d.scheduler);
            _lastTick = d;
            updateDiagnostics(d);
        }
        if (d.type === 'new_file') {
            const isV = d.file.endsWith('.mp4');
            const g = $(isV ? 'preview-grid' : 'i-grid');
            if (!g) return;
            const c = document.createElement('div');
            c.className = 'card bg-base-100 shadow-2xl border-2 border-primary card-hover animate-pulse';
            c.innerHTML = isV
                ? `<figure><video controls autoplay loop class="w-full aspect-video object-cover"><source src="/files/${d.file}"></video></figure><div class="p-3 text-xs font-mono bg-base-200 text-primary truncate">${d.file}</div>`
                : `<figure><img src="/files/${d.file}" class="w-full aspect-video object-cover"></figure><div class="p-2 text-[10px] font-mono bg-base-200 truncate">${d.file}</div>`;
            g.prepend(c);
            const ring = $('live-ring');
            if (isV && ring) ring.style.display = 'inline-block';
            setTimeout(() => c.classList.remove('animate-pulse', 'border-2', 'border-primary'), 3000);
            if (isV && g.children.length > 4) g.removeChild(g.lastChild);
        }
    };
    ws.onclose = () => {
        const w = $('refresh-wrapper');
        if (w) w.style.display = '';
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

function _renderFanoutPreview(stages) {
    const tabs = $('fanout-tabs');
    const body = $('fanout-body');
    if (!tabs || !body) return;
    tabs.innerHTML = STAGE_NAMES.map((n, i) =>
        `<a role="tab" class="tab ${i === 0 ? 'tab-active' : ''}" data-stage="${n}" onclick="_switchFanoutTab('${n}')">${n}</a>`
    ).join('');
    body.dataset.stages = JSON.stringify(stages);
    _switchFanoutTab('image');
}

function _switchFanoutTab(stage) {
    const tabs = $('fanout-tabs');
    const body = $('fanout-body');
    if (!tabs || !body) return;
    tabs.querySelectorAll('.tab').forEach(el => {
        el.classList.toggle('tab-active', el.dataset.stage === stage);
    });
    const stages = JSON.parse(body.dataset.stages || '{}');
    const before = _stageVal(stage);
    const after = stages[stage] || '';
    body.innerHTML = `
        <div class="grid grid-cols-1 gap-2">
            <div>
                <div class="text-[10px] uppercase opacity-60">Before</div>
                <div class="bg-base-100 p-2 rounded text-xs whitespace-pre-wrap">${_htmlEscape(before) || '<em class="opacity-40">(empty)</em>'}</div>
            </div>
            <div>
                <div class="text-[10px] uppercase opacity-60">After</div>
                <div class="bg-base-100 p-2 rounded text-xs whitespace-pre-wrap border-l-2 border-primary">${_htmlEscape(after) || '<em class="opacity-40">(empty)</em>'}</div>
            </div>
        </div>`;
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
    const body = $('fanout-body');
    if (body) body.innerHTML = '<em class="opacity-60">Conjuring brilliance...</em>';
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
        } else if (body) {
            body.innerText = '(no response)';
        }
    } catch (e) {
        if (body) body.innerText = 'Error: ' + e;
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

async function inject(prio) {
    const concat = _concatStagePrompts();
    const stages = {
        image: $('p-image') ? $('p-image').value : '',
        video: $('p-video') ? $('p-video').value : '',
        music: $('p-music') ? $('p-music').value : '',
        tts: $('p-tts') ? $('p-tts').value : '',
    };
    const f = new FormData();
    f.append('prompt', concat);
    f.append('priority', prio);
    f.append('stage_prompts', JSON.stringify(stages));
    await fetch('/inject', { method: 'POST', body: f });
    ['p-image', 'p-video', 'p-music', 'p-tts', 'p-in'].forEach(id => { if ($(id)) $(id).value = ''; });
}

async function updatePipeline() {
    const body = {
        infinity_mode: $('inf-on') ? $('inf-on').checked : false,
        infinity_themes: $('inf-themes') ? $('inf-themes').value.split(',').map(s => s.trim()) : [],
        base_model: $('cfg-base') ? $('cfg-base').value : '',
        video_model: $('cfg-video') ? $('cfg-video').value : '',
        audio_model: $('cfg-audio') ? $('cfg-audio').value : '',
        upscale_model: $('cfg-upscale') ? $('cfg-upscale').value : '',
        frames: $('cfg-video') && $('cfg-video').value.includes('wan') ? 81 : 49,
    };
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
        });
        const res = await fetch('/ram_estimate?' + qs.toString());
        if (res.ok) updateRam(await res.json());
    } catch (e) { /* WS tick will catch up */ }
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
        modal.showModal();
        reloadModels();
    } catch (e) {
        console.error('openSettings failed', e);
    }
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
    };
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

document.addEventListener('DOMContentLoaded', () => {
    const p = $('set-provider');
    if (p) p.addEventListener('change', applyProviderDefaults);
});

connect();
_wireLockListeners();
