/**
 * Browser API stubs for Vitest (Node environment).
 *
 * app.js uses localStorage, window globals, and document — none of which
 * exist in Node. This file sets up minimal stubs so pure functions can
 * be extracted and tested without a browser.
 *
 * Import this file at the top of each JS test file.
 */

// ----- localStorage stub -----
const _store = {};
global.localStorage = {
    getItem: (k) => (k in _store ? _store[k] : null),
    setItem: (k, v) => { _store[k] = String(v); },
    removeItem: (k) => { delete _store[k]; },
    clear: () => { Object.keys(_store).forEach(k => delete _store[k]); },
};

// ----- window stub -----
global.window = global;

// ----- fetch stub (overridable per test) -----
global.fetch = async () => ({
    ok: true,
    json: async () => ({ ok: true, messages: [] }),
});

// ----- minimal document stub -----
global.document = {
    getElementById: () => null,
    querySelector: () => null,
};

export function clearStorage() {
    Object.keys(_store).forEach(k => delete _store[k]);
}
