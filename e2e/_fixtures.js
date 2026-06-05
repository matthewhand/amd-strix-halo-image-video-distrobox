// Shared Playwright fixtures for the Slopfinity e2e suite.
//
// This module layers TWO auto-fixtures onto base @playwright/test:
//
//   (1) BACKEND GATING (_backendGuard)
//       A chunk of the suite drives the *interactive* suggestion / story
//       machinery — endless story rows, chat replies, live suggestion
//       batches, the story log. Those flows need a real, reachable LLM
//       (and in some cases the ComfyUI fleet) behind the dashboard. In CI
//       there is no backend, so they fail with 30s timeouts and pollute
//       the signal.
//
//       Specs that need that backend import { test, expect } from this
//       module instead of '@playwright/test'. The auto-fixture below skips
//       every test in such a file UNLESS SLOPFINITY_HAS_LLM is set.
//
//         * CI (no backend):   SLOPFINITY_HAS_LLM unset → these specs SKIP
//                              (reported as skipped, not failed)
//         * Local / staging:   SLOPFINITY_HAS_LLM=1 npx playwright test → run
//
//       Specs that only assert static layout / chrome (smoke, layouts, …)
//       keep importing '@playwright/test' directly and always run.
//
//   (2) ERROR SENTRY (_errorSentry)
//       Auto-attaches three error listeners to every page and fails the
//       test if any of them collect anything by the time the test ends:
//
//         • page.on('pageerror')  — uncaught JS exceptions in the browser
//         • page.on('console')    — console.error(...) calls (warn ignored)
//         • page.on('response')   — 5xx server-side failures
//                                   (4xx is validation — left alone; specs
//                                   that probe 4xx paths don't need to opt out)
//
//       Drop-in replacement for `@playwright/test`:
//
//         const { test, expect } = require('./_fixtures');
//
//       Per-test/describe opt-out for tests that intentionally trigger errors:
//
//         test.use({ ignoreErrors: { console: [/known noise/i], status5xx: true } });
//
// Filename starts with `_` so testMatch doesn't pick it up.

const base = require('@playwright/test');

const HAS_LLM = !!process.env.SLOPFINITY_HAS_LLM;

const matches = (text, patterns) =>
  (patterns || []).some(p => p instanceof RegExp ? p.test(text) : String(text).includes(p));

const test = base.test.extend({
  // `auto` fixtures run for every test that uses this `test` object,
  // before the test body. Calling test.skip(condition, reason) here
  // skips the test cleanly when no backend is configured.
  _backendGuard: [
    async ({}, use) => {
      base.test.skip(
        !HAS_LLM,
        'Requires a live LLM/ComfyUI backend — set SLOPFINITY_HAS_LLM=1 to run locally.'
      );
      await use(undefined);
    },
    { auto: true },
  ],

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

module.exports = { test, expect: base.expect, HAS_LLM };
// Pass-through re-exports for things specs sometimes import alongside test/expect.
module.exports.request = base.request;
module.exports.devices = base.devices;
