// Slopfinity dashboard client.
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
    const tl = $('sched-timeline');
    if (!tl) return;
    const events = (sc.events || []).slice(-5);
    if (!events.length) {
        tl.innerHTML = '<span class="text-[11px] text-neutral-content/40 italic">no events yet</span>';
        return;
    }
    tl.innerHTML = events.map((e, i) => {
        const label = (e.stage && e.model) ? `${e.stage}/${e.model}` : (e.type || 'event');
        const tip = JSON.stringify(e).replace(/"/g, '&quot;');
        return `<span id="sched-event-${i}" class="badge badge-sm ${schedBadgeClass(e.type)} tooltip tooltip-bottom" data-tip="${tip}">${e.type}: ${label}</span>`;
    }).join('');
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

async function enhance() {
    _concatStagePrompts();
    const p = ($('p-in') && $('p-in').value) || '';
    if (!p) return;
    const box = $('ai-box');
    if (box) box.classList.remove('hidden');
    if ($('ai-s')) $('ai-s').innerText = 'Conjuring brilliance...';
    const res = await fetch('/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: p, distribute: true }),
    });
    const r = await res.json();
    if (r && r.stages) {
        if ($('p-image')) $('p-image').value = r.stages.image || '';
        if ($('p-video')) $('p-video').value = r.stages.video || '';
        if ($('p-music')) $('p-music').value = r.stages.music || '';
        if ($('p-tts')) $('p-tts').value = r.stages.tts || '';
        _concatStagePrompts();
        if ($('ai-s')) $('ai-s').innerText = 'Distributed across stages. Edit or Accept.';
    } else if ($('ai-s')) {
        $('ai-s').innerText = r.suggestion || '(no response)';
    }
}

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

connect();
