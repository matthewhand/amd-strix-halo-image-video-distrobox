/**
 * Catalog walker: one deterministic capture per inventory ID.
 *
 * Usage:
 *   SLOP_URL=http://127.0.0.1:9099 \
 *   CATALOG_OUT=e2e/catalog-out \
 *   npx playwright test e2e/feature-settings-catalog.spec.js
 *
 * Writes:
 *   {CATALOG_OUT}/captures/<id>.png
 *   {CATALOG_OUT}/meta/<id>.json
 *   {CATALOG_OUT}/coverage.json
 */
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const BASE = process.env.SLOP_URL || 'http://127.0.0.1:9099';
const ROOT = path.resolve(__dirname, '..');
const INV_PATH = process.env.INVENTORY_JSON
  || path.join(ROOT, 'slopfinity', 'ui_inventory.json');
const OUT = process.env.CATALOG_OUT
  || path.join(ROOT, 'e2e', 'catalog-out');

function loadInventory() {
  const raw = JSON.parse(fs.readFileSync(INV_PATH, 'utf8'));
  return raw.items || [];
}

async function openSettings(page) {
  // Prefer drawer toggle
  const gear = page.locator('button[aria-label="Settings"]').first();
  if (await gear.count()) {
    await gear.click({ timeout: 5000 }).catch(() => {});
  } else {
    await page.evaluate(() => {
      if (typeof openSettings === 'function') openSettings();
      const t = document.getElementById('settings-drawer-toggle');
      if (t) t.checked = true;
    });
  }
  await page.waitForTimeout(200);
}

async function selectSettingsTab(page, tab) {
  if (!tab) return;
  const radio = page.locator(`input[name="settings_tabs"][aria-label="${tab}"]`);
  if (await radio.count()) {
    await radio.check({ force: true }).catch(async () => {
      await radio.click({ force: true });
    });
    await page.waitForTimeout(120);
  }
}

async function prepareSurface(page, item) {
  if (item.surface === 'settings' || (item.id || '').startsWith('feature.settings')) {
    await openSettings(page);
    if (item.tab) await selectSettingsTab(page, item.tab);
  }
  if ((item.id || '').startsWith('feature.mode.')) {
    const mode = item.id.split('.').pop();
    await page.evaluate((m) => {
      if (typeof _setSubjectsMode === 'function') _setSubjectsMode(m);
    }, mode);
    await page.waitForTimeout(100);
  }
  if ((item.id || '').startsWith('feature.layout.')) {
    const layout = item.id.replace('feature.layout.', '').replace(/_/g, '-');
    await page.evaluate((v) => {
      if (typeof selectLayoutView === 'function') selectLayoutView(v);
      else {
        const r = document.querySelector(`input[name="layout-view"][value="${v}"]`);
        if (r) { r.checked = true; r.dispatchEvent(new Event('change', { bubbles: true })); }
      }
    }, layout);
    await page.waitForTimeout(100);
  }
  if ((item.id || '').startsWith('feature.chat') || item.kind === 'chat') {
    await page.evaluate(() => {
      if (typeof _setSubjectsMode === 'function') _setSubjectsMode('chat');
    });
    await page.waitForTimeout(80);
  }
}

function safeName(id) {
  return String(id).replace(/[^\w.\-]+/g, '_');
}

test.describe.configure({ mode: 'serial' });

test('feature/settings catalog coverage', async ({ page }) => {
  test.setTimeout(600_000);
  const items = loadInventory();
  expect(items.length).toBeGreaterThan(40);

  fs.mkdirSync(path.join(OUT, 'captures'), { recursive: true });
  fs.mkdirSync(path.join(OUT, 'meta'), { recursive: true });

  await page.goto(BASE + '/?catalog=1&t=' + Date.now(), { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.waitForTimeout(800);
  // cache-bust static if needed
  await page.reload({ waitUntil: 'networkidle' }).catch(() => {});
  await page.waitForTimeout(400);

  const results = [];
  let captured = 0;
  let failed = 0;

  for (const item of items) {
    const id = item.id;
    const name = safeName(id);
    const shotPath = path.join(OUT, 'captures', `${name}.png`);
    const metaPath = path.join(OUT, 'meta', `${name}.json`);
    const row = {
      id,
      ok: false,
      selector: item.selector,
      surface: item.surface,
      kind: item.kind,
      tab: item.tab || null,
      error: null,
      shot: shotPath,
      visible: null,
      box: null,
    };
    try {
      await prepareSurface(page, item);
      const loc = page.locator(item.selector).first();
      const count = await loc.count();
      if (count === 0) {
        // fallback: full viewport shot still records the surface state
        await page.screenshot({ path: shotPath, fullPage: false });
        row.error = 'selector_not_found';
        row.ok = false;
        failed += 1;
      } else {
        // ensure in view
        await loc.scrollIntoViewIfNeeded().catch(() => {});
        const box = await loc.boundingBox().catch(() => null);
        row.box = box;
        row.visible = await loc.isVisible().catch(() => false);
        // Prefer element screenshot; fall back to page clip / full
        try {
          if (box && box.width > 2 && box.height > 2) {
            await loc.screenshot({ path: shotPath, timeout: 10_000 });
          } else {
            await page.screenshot({ path: shotPath, fullPage: false });
          }
        } catch (e) {
          await page.screenshot({ path: shotPath, fullPage: false });
          row.error = String(e.message || e).slice(0, 200);
        }
        // Structural OK if element exists (hidden radios still count as coverage)
        row.ok = count > 0 && fs.existsSync(shotPath) && fs.statSync(shotPath).size > 50;
        if (row.ok) captured += 1;
        else failed += 1;
      }
    } catch (e) {
      row.error = String(e.message || e).slice(0, 300);
      failed += 1;
      try {
        await page.screenshot({ path: shotPath, fullPage: false });
      } catch (_) { /* ignore */ }
    }
    fs.writeFileSync(metaPath, JSON.stringify({ item, result: row }, null, 2));
    results.push(row);
  }

  const coverage = {
    generated_at: new Date().toISOString(),
    base: BASE,
    inventory: INV_PATH,
    out: OUT,
    total: items.length,
    captured: results.filter((r) => r.ok).length,
    failed: results.filter((r) => !r.ok).length,
    missing_ids: results.filter((r) => !r.ok).map((r) => r.id),
    results,
  };
  fs.writeFileSync(path.join(OUT, 'coverage.json'), JSON.stringify(coverage, null, 2));
  // Human summary
  const summary = [
    `# Catalog coverage`,
    ``,
    `- total: ${coverage.total}`,
    `- captured_ok: ${coverage.captured}`,
    `- failed: ${coverage.failed}`,
    coverage.failed ? `- missing: ${coverage.missing_ids.join(', ')}` : `- missing: (none)`,
    ``,
  ].join('\n');
  fs.writeFileSync(path.join(OUT, 'coverage.md'), summary);

  expect(coverage.captured, `missing captures: ${coverage.missing_ids.slice(0, 20).join(', ')}`).toBe(coverage.total);
});
