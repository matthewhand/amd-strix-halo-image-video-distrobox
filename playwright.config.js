// Slopfinity Playwright config — chromium against the live dashboard
// on :9099. Override the URL with SLOPFINITY_URL=http://other:9099
// npx playwright test.
//
// REPORTING:
//   - `list`  — CLI text (each test pass/fail inline)
//   - `html`  — browseable report at playwright-report/index.html with
//                EVERY screenshot, video, trace, and step inline.
//                Open with `npx playwright show-report` (auto-opens a
//                local server on a free port) OR open the index.html
//                directly. Each test card expands to show its
//                screenshot timeline + console + network log.
//
// OUTPUT:
//   - `outputDir` — per-test artifacts (screenshots, videos, traces)
//                land under `test-results/`. The HTML report indexes
//                everything from there. Both dirs are git-ignored.
//
// SCREENSHOTS:
//   - `screenshot: 'on'` captures a PNG at the END of every test
//     (and on failure). Was 'only-on-failure' which meant the HTML
//     report had nothing visual for passing tests. Flipping to 'on'
//     makes the report a full visual catalogue of the dashboard
//     across every layout / mode / viewport combination.
//   - Specs that already do explicit `page.screenshot({ path: ... })`
//     keep writing to those custom paths AND get the auto-capture.
//
// VIDEOS:
//   - 'retain-on-failure' — only kept when a test fails (videos are
//     ~MB each so we don't keep them for green runs).
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
    testDir: './e2e',
    timeout: 30_000,
    expect: { timeout: 5_000 },
    fullyParallel: false, // dashboard is shared state
    // CI lane vs local lane:
    //   * Local: 'list' reporter (per-test inline pass/fail) + html for review
    //   * CI:    'dot' (concise) + json/junit (machine-readable) + html
    //            (uploaded as artifact only on failure). Reading the JSON
    //            programmatically lets the failure-extractor build a
    //            compact `failure-packet.md` instead of model-flooding the
    //            raw 600-line CI log.
    reporter: process.env.CI
        ? [
            ['dot'],
            ['json', { outputFile: 'artifacts/results.json' }],
            ['junit', { outputFile: 'artifacts/results.xml' }],
            ['html', { outputFolder: 'playwright-report', open: 'never' }],
        ]
        : [
            ['list'],
            ['html', { outputFolder: 'playwright-report', open: 'never' }],
        ],
    outputDir: 'test-results',
    use: {
        baseURL: process.env.SLOPFINITY_URL || 'http://localhost:9099',
        // CI keeps screenshots/videos only on failure to keep artifacts
        // small. Local keeps the full visual catalogue ('on') so the
        // HTML report is browsable for green runs too.
        trace: 'retain-on-failure',
        screenshot: process.env.CI ? 'only-on-failure' : 'on',
        video: 'retain-on-failure',
        viewport: { width: 1440, height: 900 },
        // Block the Slopfinity service worker — its activate handler
        // calls clients.navigate(c.url) to force a one-shot reload on
        // upgrade, which destroys the test's page context mid-evaluate
        // and produces flaky "Execution context was destroyed" errors.
        // Tests assert layout / visibility — none of them care about SW
        // cache behaviour.
        serviceWorkers: 'block',
    },
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
    ],
});
