// Minimal Playwright config — no fancy projects, just chromium against
// the live Slopfinity dashboard on :9099. Override the URL with
// SLOPFINITY_URL=http://other:9099 npx playwright test.
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
    testDir: './e2e',
    timeout: 30_000,
    expect: { timeout: 5_000 },
    fullyParallel: false, // dashboard is shared state
    reporter: [['list']],
    use: {
        baseURL: process.env.SLOPFINITY_URL || 'http://localhost:9099',
        trace: 'retain-on-failure',
        screenshot: 'only-on-failure',
        viewport: { width: 1440, height: 900 },
    },
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
    ],
});
