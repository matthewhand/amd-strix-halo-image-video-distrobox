// Disposable sweep spec — captures console errors, page errors, and failed
// network responses on a cold load of http://localhost:9099. NOT part of the
// regular suite; run explicitly. Safe to delete after the AFK report ships.
const { test, expect } = require('./_fixtures');

test('cold-load console + network smoke', async ({ page }) => {
  const consoleErrors = [];
  const pageErrors = [];
  const failedResponses = [];

  page.on('console', m => {
    if (m.type() === 'error') consoleErrors.push(m.text());
  });
  page.on('pageerror', e => pageErrors.push(String(e)));
  page.on('response', r => {
    if (r.status() >= 400) failedResponses.push(`${r.status()} ${r.request().method()} ${r.url()}`);
  });

  await page.goto('http://localhost:9099/', { waitUntil: 'load' });
  // Splash overlay hides at ~3.1s; WS hydration on top. 12s per handoff doc.
  await page.waitForTimeout(12000);

  console.log('=== CONSOLE ERRORS ===');
  for (const e of consoleErrors) console.log('  •', e);
  console.log('=== PAGE ERRORS ===');
  for (const e of pageErrors) console.log('  •', e);
  console.log('=== FAILED RESPONSES ===');
  for (const e of failedResponses) console.log('  •', e);

  // Surface to test report — soft assertions so we get full picture
  expect.soft(failedResponses, 'failed network responses').toEqual([]);
  expect.soft(pageErrors, 'page errors').toEqual([]);
  expect.soft(consoleErrors, 'console errors').toEqual([]);
});
