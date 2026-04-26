// Slopfinity minimal service worker — cache-first app shell.
// Network falls through for API, WS, files, branding, config, tts, etc.
// Bump whenever shell assets (app.js, app.css, templates/index.html) change
// in a way that must invalidate users' caches. Browsers delete any cache
// whose name differs on next activate.
const CACHE = 'slopfinity-shell-v190';
const SHELL = [
  '/',
  '/static/app.css',
  '/static/app.js',
  '/static/manifest.webmanifest',
  '/static/icons/icon-180.png',
  '/static/icons/icon-192.png',
  '/static/icons/icon-256.png',
  '/static/icons/icon-384.png',
  '/static/icons/icon-512.png',
];

// Paths we must NEVER cache — always go to network.
const NETWORK_ONLY_PREFIXES = [
  '/ws',
  '/files/',
  '/api',
  '/branding',
  '/enhance',
  '/inject',
  '/config',
  '/tts',
  '/llm/',
  '/ram_estimate',
  '/storage',
  '/vae_grid',
  '/upload',
  '/seeds/',
  '/manifest.webmanifest', // prefer live/dynamic branded manifest
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).catch(() => undefined)
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
     .then(() => self.clients.matchAll({ type: 'window' }))
     .then((clients) => {
        // After the new SW activates, every controlled tab is still painting
        // the OLD shell because the OLD SW served the current page load.
        // Force a single reload of each tab so the new cache (already
        // pre-populated by the install handler's addAll(SHELL)) is what
        // paints next. This converts the typical "two-reload" SW upgrade
        // dance into a single reload from the user's perspective.
        clients.forEach((c) => {
          try { c.navigate(c.url); } catch (_) {}
        });
     })
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Skip network-only paths entirely.
  if (NETWORK_ONLY_PREFIXES.some((p) => url.pathname === p || url.pathname.startsWith(p))) {
    return; // default network fetch
  }

  // Cache-first for the shell.
  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req)
        .then((resp) => {
          if (resp && resp.ok && (url.pathname === '/' || url.pathname.startsWith('/static/'))) {
            const copy = resp.clone();
            caches.open(CACHE).then((cache) => cache.put(req, copy)).catch(() => undefined);
          }
          return resp;
        })
        .catch(() => cached);
    })
  );
});
