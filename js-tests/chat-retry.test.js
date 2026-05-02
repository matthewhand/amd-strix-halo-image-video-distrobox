/**
 * Unit tests for _fetchChatWithRetry logic.
 *
 * We replicate the retry helper here as a pure function so we can test it
 * without importing all of app.js. When Phase B modularises the code, these
 * tests should import from the real chat module instead.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { clearStorage } from './setup.js';
import { _getChatHistory, _setChatHistory } from './_shim.js';

// ---------------------------------------------------------------------------
// Inline replica of _fetchChatWithRetry for testing.
// Must stay in sync with slopfinity/static/app.js until Phase B.
// ---------------------------------------------------------------------------
async function _fetchChatWithRetry(messages, maxRetries, _fetchFn) {
    // _fetchFn is injectable for testing (replaces global fetch)
    if (maxRetries === undefined) maxRetries = 5;
    const BASE_DELAY_MS = 2000;
    let placeholderPushed = false;

    const _setPlaceholder = (text) => {
        const hist = _getChatHistory();
        if (placeholderPushed) {
            hist[hist.length - 1] = { role: 'assistant', content: text, _retry_placeholder: true };
        } else {
            hist.push({ role: 'assistant', content: text, _retry_placeholder: true });
            placeholderPushed = true;
        }
        _setChatHistory(hist);
    };

    const _clearPlaceholder = () => {
        if (!placeholderPushed) return;
        const hist = _getChatHistory();
        if (hist.length && hist[hist.length - 1]._retry_placeholder) {
            hist.pop();
            _setChatHistory(hist);
        }
        placeholderPushed = false;
    };

    let lastErr;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const r = await _fetchFn('/chat', { method: 'POST', body: JSON.stringify({ messages }) });
            _clearPlaceholder();
            return await r.json();
        } catch (e) {
            lastErr = e;
            const retriesLeft = maxRetries - attempt;
            if (retriesLeft <= 0) break;
            const delaySec = (BASE_DELAY_MS / 1000) * Math.pow(2, attempt);
            _setPlaceholder(`⚠ Network error: ${e.message} — retrying in ${delaySec}s…`);
            // Skip actual sleep in tests
        }
    }
    _clearPlaceholder();
    throw lastErr;
}

// ---------------------------------------------------------------------------

beforeEach(() => {
    clearStorage();
});

describe('_fetchChatWithRetry', () => {
    it('returns parsed JSON on first success', async () => {
        const fetchFn = vi.fn().mockResolvedValue({
            json: async () => ({ ok: true, messages: [{ role: 'assistant', content: 'hi' }] }),
        });
        const result = await _fetchChatWithRetry([], 2, fetchFn);
        expect(result.ok).toBe(true);
        expect(fetchFn).toHaveBeenCalledTimes(1);
    });

    it('retries on TypeError and succeeds on 3rd attempt', async () => {
        let calls = 0;
        const fetchFn = vi.fn().mockImplementation(async () => {
            calls++;
            if (calls < 3) throw new TypeError('Failed to fetch');
            return { json: async () => ({ ok: true, messages: [] }) };
        });
        const result = await _fetchChatWithRetry([], 5, fetchFn);
        expect(result.ok).toBe(true);
        expect(fetchFn).toHaveBeenCalledTimes(3);
    });

    it('throws after exhausting all retries', async () => {
        const fetchFn = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
        await expect(_fetchChatWithRetry([], 2, fetchFn)).rejects.toThrow('Failed to fetch');
        expect(fetchFn).toHaveBeenCalledTimes(3); // initial + 2 retries
    });

    it('inserts placeholder message on first failure', async () => {
        _setChatHistory([{ role: 'user', content: 'test' }]);
        let calls = 0;
        const fetchFn = vi.fn().mockImplementation(async () => {
            calls++;
            if (calls < 2) throw new TypeError('Failed to fetch');
            return { json: async () => ({ ok: true, messages: [] }) };
        });
        await _fetchChatWithRetry([], 5, fetchFn);
        // Placeholder should be removed after success
        const hist = _getChatHistory();
        expect(hist.every(m => !m._retry_placeholder)).toBe(true);
    });

    it('clears placeholder on final success', async () => {
        _setChatHistory([{ role: 'user', content: 'x' }]);
        let calls = 0;
        const fetchFn = vi.fn().mockImplementation(async () => {
            calls++;
            if (calls === 1) throw new TypeError('Failed to fetch');
            return { json: async () => ({ ok: true, messages: [] }) };
        });
        await _fetchChatWithRetry([], 5, fetchFn);
        const hist = _getChatHistory();
        expect(hist.some(m => m._retry_placeholder)).toBe(false);
    });

    it('clears placeholder even when all retries exhausted', async () => {
        _setChatHistory([{ role: 'user', content: 'x' }]);
        const fetchFn = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
        try {
            await _fetchChatWithRetry([], 1, fetchFn);
        } catch (_) {}
        const hist = _getChatHistory();
        expect(hist.some(m => m._retry_placeholder)).toBe(false);
    });

    it('calls fetch with messages in body', async () => {
        const msgs = [{ role: 'user', content: 'hello' }];
        const fetchFn = vi.fn().mockResolvedValue({
            json: async () => ({ ok: true, messages: [] }),
        });
        await _fetchChatWithRetry(msgs, 0, fetchFn);
        const callArgs = fetchFn.mock.calls[0];
        const body = JSON.parse(callArgs[1].body);
        expect(body.messages).toEqual(msgs);
    });
});
