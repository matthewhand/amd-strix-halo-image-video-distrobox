// Extended Playwright test that auto-attaches three error listeners to
// every page and fails the test if any of them collect anything by the
// time the test ends:
//
//   • page.on('pageerror')  — uncaught JS exceptions in the browser
//   • page.on('console')    — console.error(...) calls (warn ignored)
//   • page.on('response')   — 5xx server-side failures
//                             (4xx is validation — left alone; specs
//                             that probe 4xx paths don't need to opt out)
//
// Drop-in replacement for `@playwright/test`:
//
//   const { test, expect } = require('./_fixtures');
//
// Per-test/describe opt-out for tests that intentionally trigger errors:
//
//   test.use({ ignoreErrors: { console: [/known noise/i], status5xx: true } });
//
// Filename starts with `_` so testMatch doesn't pick it up.

const base = require('@playwright/test');

const matches = (text, patterns) =>
  (patterns || []).some(p => p instanceof RegExp ? p.test(text) : String(text).includes(p));

exports.test = base.test.extend({
  // Option-style fixture: tests/describes can override via test.use().
  ignoreErrors: [
    { pageerror: [], console: [], status5xx: false },
    { option: true },
  ],

  // Auto-fixture: runs for every test, no need to declare it in the
  // test signature. Attaches listeners on setup, asserts clean on teardown.
  _errorSentry: [
    async ({ page, ignoreErrors }, use) => {
      const pageErrors = [];
      const consoleErrors = [];
      const serverErrors = [];

      page.on('pageerror', (e) => {
        const msg = String(e?.stack || e?.message || e);
        if (!matches(msg, ignoreErrors.pageerror)) pageErrors.push(msg);
      });
      page.on('console', (m) => {
        if (m.type() !== 'error') return;
        const txt = m.text();
        if (!matches(txt, ignoreErrors.console)) consoleErrors.push(txt);
      });
      page.on('response', (r) => {
        if (ignoreErrors.status5xx) return;
        if (r.status() >= 500) {
          serverErrors.push(`${r.status()} ${r.request().method()} ${r.url()}`);
        }
      });

      await use(undefined);

      // Hard assertions — matches the contract _sweep-smoke.spec.js
      // established. A regression here turns the test red, which is
      // the entire point of the global fixture.
      base.expect(pageErrors,    'uncaught browser pageerror').toEqual([]);
      base.expect(consoleErrors, 'browser console.error calls').toEqual([]);
      base.expect(serverErrors,  'server 5xx responses').toEqual([]);
    },
    { auto: true },
  ],
});

exports.expect = base.expect;
// Pass-through re-exports for things specs sometimes import alongside test/expect.
exports.request = base.request;
exports.devices = base.devices;

