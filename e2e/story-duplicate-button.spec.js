const { test, expect } = require('@playwright/test');

test('Story mode should only have one Queue Slop button visible', async ({ page }) => {
    await page.goto('/');
    
    // Switch to endless mode
    await page.click('button[data-subj-mode="endless"]');
    
    // Wait for layout switch
    await page.waitForTimeout(500);
    
    // Check buttons that say "Queue Slop"
    const visibleQueueButtons = await page.$$eval('button', buttons => 
        buttons.filter(b => b.textContent.includes('Queue Slop') && window.getComputedStyle(b).display !== 'none' && b.getBoundingClientRect().width > 0).length
    );
    
    expect(visibleQueueButtons).toBe(1);

    // Check story-specific action buttons (should only be 1 each)
    for (const label of ['Submit', 'Reset', 'Auto-Stitch', 'Copy story']) {
        const count = await page.$$eval('button, label', (els, lab) => 
            els.filter(e => e.textContent.includes(lab) && window.getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().width > 0).length
        , label);
        expect(count, `Expected exactly 1 visible "${label}" button/label`).toBe(1);
    }
});
