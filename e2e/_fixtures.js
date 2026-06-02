// Shared Playwright fixtures for the Slopfinity e2e suite.
//
// BACKEND GATING
// --------------
// A chunk of the suite drives the *interactive* suggestion / story
// machinery — endless story rows, chat replies, live suggestion
// batches, the story log. Those flows need a real, reachable LLM (and
// in some cases the ComfyUI fleet) behind the dashboard. In CI there is
// no backend, so they fail with 30s timeouts and pollute the signal.
//
// Specs that need that backend import { test, expect } from this module
// instead of '@playwright/test'. The auto-fixture below skips every test
// in such a file UNLESS SLOPFINITY_HAS_LLM is set in the environment.
//
//   * CI (no backend):   SLOPFINITY_HAS_LLM unset → these specs SKIP
//                        (reported as skipped, not failed → clean signal)
//   * Local / staging:   SLOPFINITY_HAS_LLM=1 npx playwright test → they run
//
// Specs that only assert static layout / chrome (smoke, layouts, …)
// keep importing '@playwright/test' directly and always run.
const base = require('@playwright/test');

const HAS_LLM = !!process.env.SLOPFINITY_HAS_LLM;

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
});

module.exports = { test, expect: base.expect, HAS_LLM };
