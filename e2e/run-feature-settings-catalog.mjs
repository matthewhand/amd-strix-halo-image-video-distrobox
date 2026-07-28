/**
 * Fast standalone catalog walker (no Playwright test runner timeout).
 *
 *   node e2e/run-feature-settings-catalog.mjs
 *
 * Env:
 *   SLOP_URL (default http://127.0.0.1:9099)
 *   INVENTORY_JSON
 *   CATALOG_OUT
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { chromium } from 'playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const BASE = process.env.SLOP_URL || 'http://127.0.0.1:9099';
const INV_PATH = process.env.INVENTORY_JSON || path.join(ROOT, 'slopfinity', 'ui_inventory.json');
const OUT = process.env.CATALOG_OUT || path.join(ROOT, 'e2e', 'catalog-out');

function safeName(id) {
  return String(id).replace(/[^\w.\-]+/g, '_');
}

async function ensureCatalogHideStyle(page) {
  await page.evaluate(() => {
    if (document.getElementById('catalog-hide-settings-style')) return;
    const s = document.createElement('style');
    s.id = 'catalog-hide-settings-style';
    s.textContent = `
      body.catalog-settings-closed #settings-drawer .drawer-side {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
        transform: translateX(100%) !important;
      }
      body.catalog-settings-closed #settings-drawer .drawer-overlay {
        display: none !important;
      }
      body:not(.catalog-settings-closed) #settings-drawer .drawer-side {
        /* restore when open */
      }
    `;
    document.head.appendChild(s);
  });
}

async function closeSettings(page) {
  await ensureCatalogHideStyle(page);
  try {
    const tog = page.locator('#settings-drawer-toggle');
    if (await tog.count()) {
      await tog.setChecked(false, { force: true }).catch(() => {});
    }
  } catch (_) {}
  await page.evaluate(() => {
    document.body.classList.add('catalog-settings-closed');
    try {
      if (typeof _setSettingsOpen === 'function') _setSettingsOpen(false);
    } catch (_) {}
    const t = document.getElementById('settings-drawer-toggle');
    if (t) {
      t.checked = false;
      t.dispatchEvent(new Event('change', { bubbles: true }));
    }
    const q = document.getElementById('queue-drawer-toggle');
    if (q) q.checked = false;
    document.querySelectorAll('dialog[open]').forEach((d) => {
      try { d.close(); } catch (_) {}
    });
    // Ensure main is visible (splash may have left opacity 0)
    const main = document.querySelector('main');
    if (main) {
      main.style.opacity = '1';
      main.style.visibility = 'visible';
    }
    const splash = document.getElementById('splash-overlay');
    if (splash) splash.remove();
  });
  await page.waitForTimeout(60);
}

async function openSettings(page) {
  await ensureCatalogHideStyle(page);
  await page.evaluate(() => {
    document.body.classList.remove('catalog-settings-closed');
    try {
      if (typeof openSettings === 'function') openSettings();
    } catch (_) {}
    if (typeof _setSettingsOpen === 'function') _setSettingsOpen(true);
    const t = document.getElementById('settings-drawer-toggle');
    if (t) {
      t.checked = true;
      t.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
  try {
    await page.locator('#settings-drawer-toggle').setChecked(true, { force: true });
  } catch (_) {}
  await page.waitForTimeout(120);
}

async function openPipelineModal(page) {
  await closeSettings(page);
  await page.evaluate(() => {
    if (typeof openPipeline === 'function') openPipeline();
    else {
      const d = document.getElementById('pipeline-modal');
      if (d && d.showModal) d.showModal();
    }
  });
  await page.waitForTimeout(100);
}

async function prepare(page, item) {
  const id = item.id || '';
  const isPipelineModal =
    id.startsWith('settings.PipelineModal') ||
    item.tab === 'PipelineModal' ||
    id === 'feature.pipeline.modal' ||
    id.includes('pipeline.open');
  const needsSettingsDrawer =
    !isPipelineModal &&
    (id.startsWith('feature.settings') ||
      (id.startsWith('settings.') && item.tab && item.tab !== 'PipelineModal') ||
      (item.surface === 'settings' && item.tab && item.tab !== 'PipelineModal'));

  // Feature / pipeline: ensure settings drawer closed first
  if (!needsSettingsDrawer) {
    await closeSettings(page);
  }

  if (isPipelineModal) {
    await openPipelineModal(page);
  } else if (needsSettingsDrawer) {
    await openSettings(page);
    if (item.tab) {
      await page.evaluate((tab) => {
        const r = document.querySelector(`input[name="settings_tabs"][aria-label="${tab}"]`);
        if (r) {
          r.checked = true;
          r.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }, item.tab);
      await page.waitForTimeout(60);
    }
  }

  if (id.startsWith('feature.mode.')) {
    const mode = id.split('.').pop();
    await page.evaluate((m) => {
      if (typeof selectLayoutView === 'function') selectLayoutView('default');
      if (typeof _setSubjectsMode === 'function') _setSubjectsMode(m);
    }, mode);
    await page.waitForTimeout(80);
  }
  if (id.startsWith('feature.layout.')) {
    const layout = id.replace('feature.layout.', '').replace(/_/g, '-');
    await page.evaluate((v) => {
      if (typeof selectLayoutView === 'function') selectLayoutView(v);
      else {
        const r = document.querySelector(`input[name="layout-view"][value="${v}"]`);
        if (r) {
          r.checked = true;
          r.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    }, layout);
    await page.waitForTimeout(120);
  }
  if (id.startsWith('feature.chat') || item.kind === 'chat') {
    await page.evaluate(() => {
      if (typeof selectLayoutView === 'function') selectLayoutView('subjects');
      if (typeof _setSubjectsMode === 'function') _setSubjectsMode('chat');
    });
    await page.waitForTimeout(80);
  }
  if (id.startsWith('feature.queue')) {
    await page.evaluate(() => {
      if (typeof selectLayoutView === 'function') selectLayoutView('queue');
    });
    await page.waitForTimeout(80);
  }
  if (id.startsWith('feature.gallery')) {
    await page.evaluate(() => {
      if (typeof selectLayoutView === 'function') selectLayoutView('gallery');
    });
    // Activate the specific filter chip so each filter capture is distinct.
    const m = id.match(/feature\.gallery\.filter\.(\w+)/);
    if (m) {
      const kind = m[1];
      await page.evaluate((k) => {
        // Turn all filters off, then enable only the target kind so the
        // viewport differs per inventory ID (MD5 must not be identical).
        document.querySelectorAll('[data-slop-filter]').forEach((cb) => {
          cb.checked = cb.getAttribute('data-slop-filter') === k;
        });
        if (typeof _applySlopFilters === 'function') _applySlopFilters();
        // Banner text in meta via data attribute for debugging
        document.documentElement.setAttribute('data-catalog-filter', k);
      }, kind);
      await page.waitForTimeout(80);
    }
    // empty CTA: clear grid so the CTA is actually shown
    if (id.includes('empty_cta')) {
      await page.evaluate(() => {
        const g = document.getElementById('preview-grid');
        if (g) {
          g.innerHTML = '';
          if (typeof _applySlopFilters === 'function') _applySlopFilters();
        }
      });
    }
    await page.waitForTimeout(60);
  }
  if (id.includes('prompt.core') || id.includes('suggest.')) {
    await page.evaluate(() => {
      if (typeof selectLayoutView === 'function') selectLayoutView('subjects');
      if (typeof _setSubjectsMode === 'function') _setSubjectsMode('simple');
    });
    await page.waitForTimeout(60);
  }
}

async function captureOne(page, item) {
  const name = safeName(item.id);
  const shotPath = path.join(OUT, 'captures', `${name}.png`);
  const metaPath = path.join(OUT, 'meta', `${name}.json`);
  const row = {
    id: item.id,
    ok: false,
    selector: item.selector,
    surface: item.surface,
    kind: item.kind,
    tab: item.tab || null,
    error: null,
    shot: shotPath,
    count: 0,
  };
  try {
    await prepare(page, item);
    // Record whether settings drawer leaked open on a feature shot
    row.settings_drawer_open = await page.evaluate(() => {
      const t = document.getElementById('settings-drawer-toggle');
      return !!(t && t.checked);
    });
    if (row.settings_drawer_open && !(item.id || '').startsWith('settings.') && !(item.id || '').startsWith('feature.settings')) {
      await closeSettings(page);
      row.settings_drawer_open = await page.evaluate(() => {
        const t = document.getElementById('settings-drawer-toggle');
        return !!(t && t.checked);
      });
    }
    const handle = await page.$(item.selector);
    row.count = handle ? 1 : 0;
    if (handle) {
      await handle.evaluate((el) => {
        try {
          el.scrollIntoView({ block: 'center', inline: 'nearest' });
        } catch (_) {}
      }).catch(() => {});
      const box = await handle.boundingBox().catch(() => null);
      row.box = box;
      // Prefer a meaningful context shot: small controls get parent card /
      // settings modal / viewport so human review is not a blank 100-byte chip.
      const area = box ? box.width * box.height : 0;
      const forceViewport =
        (item.id || '').startsWith('feature.mode.') ||
        (item.id || '').startsWith('feature.layout.') ||
        (item.id || '').startsWith('feature.queue') ||
        (item.id || '').startsWith('feature.gallery') ||
        (item.id || '').startsWith('feature.chat') ||
        (item.id || '').includes('pipeline.open') ||
        (item.id || '').includes('settings.open') ||
        (item.id || '').includes('prompt.core') ||
        (item.id || '') === 'feature.main';
      try {
        if (forceViewport || area < 2000) {
          await page.screenshot({ path: shotPath, fullPage: false, timeout: 4000 });
        } else if (area >= 8000) {
          await handle.screenshot({ path: shotPath, timeout: 2500 });
        } else {
          const ctx = await handle.evaluateHandle((el) => {
            return (
              el.closest('#settings-modal, .tab-content:not([style*="display: none"]), .modal-box, .card, #subjects-chat-pane, #preview-grid, #q-list, main') ||
              el.parentElement ||
              el
            );
          });
          const ctxEl = ctx.asElement();
          if (ctxEl) {
            try {
              await ctxEl.screenshot({ path: shotPath, timeout: 2500 });
            } catch (_) {
              await page.screenshot({ path: shotPath, fullPage: false, timeout: 4000 });
            }
          } else {
            await page.screenshot({ path: shotPath, fullPage: false, timeout: 4000 });
          }
        }
      } catch (e) {
        await page.screenshot({ path: shotPath, fullPage: false, timeout: 4000 });
        row.error = `shot_fallback: ${String(e.message || e).slice(0, 120)}`;
      }
    } else {
      await page.screenshot({ path: shotPath, fullPage: false, timeout: 4000 });
      row.error = 'selector_not_found';
    }
    row.ok = fs.existsSync(shotPath) && fs.statSync(shotPath).size > 80;
  } catch (e) {
    row.error = String(e.message || e).slice(0, 300);
    try {
      await page.screenshot({ path: shotPath, fullPage: false, timeout: 3000 });
    } catch (_) {}
  }
  fs.writeFileSync(metaPath, JSON.stringify({ item, result: row }, null, 2));
  return row;
}

async function main() {
  const items = JSON.parse(fs.readFileSync(INV_PATH, 'utf8')).items || [];
  if (!items.length) {
    console.error('empty inventory', INV_PATH);
    process.exit(2);
  }
  fs.mkdirSync(path.join(OUT, 'captures'), { recursive: true });
  fs.mkdirSync(path.join(OUT, 'meta'), { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(8000);
  await page.goto(`${BASE}/?catalog=1&t=${Date.now()}`, { waitUntil: 'domcontentloaded', timeout: 30000 });
  // Splash overlays the whole UI for ~2.5s (or until first WS tick). Catalog
  // must force-hide it or every shot is the splash logo + open drawer scrap.
  await page.evaluate(() => {
    try {
      if (typeof _hideSplash === 'function') _hideSplash();
    } catch (_) {}
    const el = document.getElementById('splash-overlay');
    if (el) el.remove();
    const main = document.querySelector('main');
    if (main) {
      main.style.opacity = '1';
      main.style.visibility = 'visible';
    }
  });
  await page.waitForTimeout(200);

  const results = [];
  let i = 0;
  for (const item of items) {
    i += 1;
    let row = await captureOne(page, item);
    if (!row.ok) {
      // One retry after soft reload (handles transient navigation / context loss)
      try {
        await page.goto(`${BASE}/?catalog=retry&t=${Date.now()}`, {
          waitUntil: 'domcontentloaded',
          timeout: 20000,
        });
        await page.evaluate(() => {
          try { if (typeof _hideSplash === 'function') _hideSplash(); } catch (_) {}
          const el = document.getElementById('splash-overlay');
          if (el) el.remove();
          const main = document.querySelector('main');
          if (main) main.style.opacity = '1';
        });
        await page.waitForTimeout(150);
        row = await captureOne(page, item);
        if (row.ok) row.error = (row.error ? row.error + '; ' : '') + 'recovered_after_retry';
      } catch (e) {
        row.error = String(e.message || e).slice(0, 200);
      }
    }
    results.push(row);
    if (i % 20 === 0 || !row.ok) {
      console.log(`[${i}/${items.length}] ${row.ok ? 'OK' : 'FAIL'} ${item.id}${row.error ? ' — ' + row.error : ''}`);
    }
  }
  await browser.close();

  const coverage = {
    generated_at: new Date().toISOString(),
    base: BASE,
    inventory: INV_PATH,
    out: OUT,
    total: items.length,
    captured: results.filter((r) => r.ok).length,
    failed: results.filter((r) => !r.ok).length,
    missing_ids: results.filter((r) => !r.ok).map((r) => r.id),
    selector_not_found: results.filter((r) => r.error === 'selector_not_found').map((r) => r.id),
    results,
  };
  fs.writeFileSync(path.join(OUT, 'coverage.json'), JSON.stringify(coverage, null, 2));
  fs.writeFileSync(
    path.join(OUT, 'coverage.md'),
    [
      '# Catalog coverage',
      '',
      `- total: ${coverage.total}`,
      `- captured_ok: ${coverage.captured}`,
      `- failed: ${coverage.failed}`,
      coverage.failed
        ? `- missing: ${coverage.missing_ids.join(', ')}`
        : '- missing: (none)',
      '',
    ].join('\n'),
  );
  console.log(JSON.stringify({ total: coverage.total, captured: coverage.captured, failed: coverage.failed }, null, 2));
  process.exit(coverage.failed === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
