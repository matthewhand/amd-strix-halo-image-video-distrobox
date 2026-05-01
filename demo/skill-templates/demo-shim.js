// =====================================================================
// demo-shim.js — generic static-demo network hijack.
// Loaded BEFORE the app's main JS. Replaces fetch + WebSocket +
// EventSource with in-memory mocks driven by ./fixtures/*.json.
// App code is unchanged.
//
// CUSTOMIZE PER PROJECT: only the fixture JSON files. Don't fork this.
// =====================================================================
(function () {
  'use strict';

  if (window.__DEMO_SHIM_LOADED__) return;
  window.__DEMO_SHIM_LOADED__ = true;
  window.__IS_DEMO__ = true;

  const _origFetch = window.fetch.bind(window);
  const _OrigWS = window.WebSocket;

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const jitter = (base, spread) => Math.max(20, base + Math.random() * spread);

  // ---- Fixture loader ------------------------------------------------
  const FIXTURE_BASE = './fixtures/';
  const fixtures = { network: { static: {}, decks: {} }, ticker: null };
  let fixturesReady = null;

  async function loadFixtures() {
    if (fixturesReady) return fixturesReady;
    fixturesReady = (async () => {
      try {
        const [n, t] = await Promise.all([
          _origFetch(FIXTURE_BASE + 'network.json').then(r => r.json()),
          _origFetch(FIXTURE_BASE + 'ticker.json').then(r => r.json()),
        ]);
        fixtures.network = n || { static: {}, decks: {} };
        fixtures.ticker = t || { stage_timeline: [], completed_pool: [], fake_stats: {} };
      } catch (e) {
        console.error('[demo] fixture load failed', e);
      }
    })();
    return fixturesReady;
  }

  // ---- Round-robin deck cursors -------------------------------------
  const _deckCursors = {};
  function nextFromDeck(key, deck) {
    const idx = (_deckCursors[key] = (_deckCursors[key] ?? -1) + 1) % deck.length;
    return deck[idx];
  }

  // ---- Ticker (scripted progression) --------------------------------
  class DemoTicker {
    constructor() {
      this.queue = [];
      this.assets = [];
      this.subscribers = new Set();
      this._loop();
    }
    enqueue(prompt) {
      const ts = Math.floor(Date.now() / 1000) + Math.random();
      const item = { ts, prompt: prompt || '(demo prompt)', status: 'pending', stage: 'queued', progress: 0 };
      this.queue.unshift(item);
      this._broadcast({ type: 'queue', items: this.queue });
      this._scheduleProgression(item);
    }
    _scheduleProgression(item) {
      const stages = (fixtures.ticker?.stage_timeline) || [{ stage: 'render', dur_ticks: 3 }];
      let acc = 0;
      stages.forEach(s => {
        acc += s.dur_ticks;
        setTimeout(() => {
          if (item.status === 'cancelled') return;
          item.stage = s.stage;
          item.status = 'running';
          item.progress = 0;
          this._broadcast({ type: 'queue', items: this.queue });
        }, acc * 1500);
      });
      setTimeout(() => {
        if (item.status === 'cancelled') return;
        item.status = 'done';
        item.progress = 100;
        const pool = fixtures.ticker?.completed_pool || [];
        if (pool.length) {
          const filename = nextFromDeck('completed', pool);
          this.assets.unshift({ file: filename, ts: item.ts, mtime: Date.now() / 1000 });
          this._broadcast({ type: 'new_file', file: filename });
        }
        this._broadcast({ type: 'queue', items: this.queue });
      }, (acc + 1) * 1500);
    }
    cancel(ts) {
      const it = this.queue.find(q => q.ts === ts);
      if (it) it.status = 'cancelled';
      this._broadcast({ type: 'queue', items: this.queue });
    }
    snapshot() { return { items: this.queue, total: this.queue.length, ok: true }; }
    _broadcast(msg) { this.subscribers.forEach(fn => { try { fn(msg); } catch (e) {} }); }
    subscribe(fn) { this.subscribers.add(fn); return () => this.subscribers.delete(fn); }
    _loop() {
      setInterval(() => {
        this._broadcast({
          type: 'state',
          ts: Date.now(),
          queue: this.queue,
          assets: this.assets.slice(0, 30),
          stats: fixtures.ticker?.fake_stats || {},
        });
      }, 2000);
    }
  }

  const ticker = new DemoTicker();
  window.__demoTicker = ticker;

  // ---- fetch hijack -------------------------------------------------
  window.fetch = async function (input, init) {
    await loadFixtures();

    // Resolve URL + method (the app may pass Request objects or strings)
    let url, method;
    if (typeof input === 'string') {
      url = input;
      method = (init?.method || 'GET').toUpperCase();
    } else if (input instanceof Request) {
      url = input.url;
      method = (input.method || 'GET').toUpperCase();
    } else {
      url = String(input);
      method = (init?.method || 'GET').toUpperCase();
    }

    // Don't hijack absolute external URLs (CDN scripts, etc.) or fixture loads
    let u;
    try { u = new URL(url, location.origin); } catch (e) { return _origFetch(input, init); }
    if (u.origin !== location.origin) return _origFetch(input, init);
    if (u.pathname.startsWith('/fixtures/') || u.pathname.startsWith('/samples/') || u.pathname.startsWith('/static/')) {
      return _origFetch(input, init);
    }

    const path = u.pathname;

    // 1. Static fixtures
    if (fixtures.network.static && fixtures.network.static[path]) {
      const fx = fixtures.network.static[path];
      await sleep(jitter(fx.delay ?? 80, 120));
      return _jsonResponse(fx.body, fx.status ?? 200);
    }

    // 2. Deck endpoints (path-prefix match, ignore query string)
    for (const key of Object.keys(fixtures.network.decks || {})) {
      if (path === key) {
        const deck = fixtures.network.decks[key];
        const body = nextFromDeck(key, deck.responses);
        await sleep(jitter(deck.delay ?? 600, 400));
        return _jsonResponse(body);
      }
    }

    // 3. Scripted progression (project-customizable; default Slopfinity-style)
    if (path === '/inject' && method === 'POST') {
      const body = await _parseBody(input, init);
      ticker.enqueue(body?.prompt);
      await sleep(180);
      return _jsonResponse({ ok: true, demo: true, ts: Date.now() / 1000 });
    }
    if (path === '/queue/paginated' || path === '/queue') {
      await sleep(50);
      return _jsonResponse(ticker.snapshot());
    }
    if (path === '/assets' || path === '/outputs') {
      await sleep(60);
      return _jsonResponse({ files: ticker.assets, ok: true });
    }

    // 4. Sample binary (asset routes redirect to bundled samples)
    if (path.startsWith('/asset/') || path.startsWith('/files/')) {
      const fname = path.split('/').filter(Boolean).pop();
      if (!fname || fname === 'asset' || fname === 'files') {
        // Listing endpoint, not a binary fetch — return the assets list
        await sleep(40);
        return _jsonResponse({ files: ticker.assets, ok: true });
      }
      return _origFetch('./samples/' + fname).catch(() =>
        _jsonResponse({ ok: false, error: 'sample missing' }, 404)
      );
    }

    // 5. Echo (mutating endpoints with no real effect — toast + ack)
    if (method !== 'GET') {
      await sleep(jitter(120, 80));
      _showEchoToast(method + ' ' + path);
      return _jsonResponse({ ok: true, demo: true });
    }

    // 6. Fallback: unknown GET — log and return generic ok
    console.warn('[demo] unmocked', method, path);
    return _jsonResponse({ ok: true, demo: true, _unmocked: path });
  };

  function _jsonResponse(body, status = 200) {
    return new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    });
  }

  async function _parseBody(input, init) {
    try {
      if (input instanceof Request) return await input.clone().json();
      const raw = init?.body;
      if (typeof raw === 'string') return JSON.parse(raw);
      if (raw instanceof FormData) {
        const obj = {};
        raw.forEach((v, k) => obj[k] = v);
        return obj;
      }
      return raw || {};
    } catch (e) {
      return {};
    }
  }

  // ---- WebSocket shim -----------------------------------------------
  window.WebSocket = class DemoWS {
    constructor(url) {
      this.url = String(url);
      this.readyState = 0;
      this.binaryType = 'blob';
      // Listener registries (so addEventListener works too)
      this._listeners = { open: [], message: [], close: [], error: [] };
      setTimeout(() => {
        this.readyState = 1;
        this._fire('open', { type: 'open' });
        this._unsub = ticker.subscribe(msg => {
          this._fire('message', { data: JSON.stringify(msg) });
        });
      }, 80);
    }
    addEventListener(ev, fn) { this._listeners[ev]?.push(fn); }
    removeEventListener(ev, fn) {
      const a = this._listeners[ev]; if (!a) return;
      const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1);
    }
    send() { /* no-op in demo */ }
    close() {
      if (this.readyState === 3) return;
      this.readyState = 3;
      this._unsub?.();
      this._fire('close', { type: 'close', code: 1000 });
    }
    _fire(ev, data) {
      const handler = this['on' + ev];
      try { handler && handler.call(this, data); } catch (e) {}
      (this._listeners[ev] || []).forEach(fn => { try { fn.call(this, data); } catch (e) {} });
    }
  };
  Object.defineProperty(window.WebSocket, 'CONNECTING', { value: 0 });
  Object.defineProperty(window.WebSocket, 'OPEN', { value: 1 });
  Object.defineProperty(window.WebSocket, 'CLOSING', { value: 2 });
  Object.defineProperty(window.WebSocket, 'CLOSED', { value: 3 });

  // ---- Banner + reset + AI-limited badges ---------------------------
  async function _injectBanner() {
    try {
      const html = await _origFetch('./static/demo-banner.html').then(r => r.text());
      const wrap = document.createElement('div');
      wrap.innerHTML = html;
      while (wrap.firstChild) document.body.prepend(wrap.firstChild);
      document.getElementById('demo-reset')?.addEventListener('click', () => {
        try { localStorage.clear(); sessionStorage.clear(); } catch (e) {}
        location.reload();
      });
    } catch (e) {
      console.warn('[demo] banner load failed', e);
    }
  }

  function _showEchoToast(label) {
    const t = document.createElement('div');
    t.className = 'demo-toast';
    t.textContent = `Demo: change won't persist (${label})`;
    document.body.appendChild(t);
    setTimeout(() => t.classList.add('demo-toast-fade'), 2000);
    setTimeout(() => t.remove(), 2800);
  }

  // ---- Boot ---------------------------------------------------------
  loadFixtures().then(() => {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', _injectBanner);
    } else {
      _injectBanner();
    }
  });
})();
