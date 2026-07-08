// LIVE: prove the chat loop discovers tools via mcp-openapi-proxy and
// the LLM actually picks one to answer a real question.
//
// What this proves (against the real :9099 backend + a real LLM):
//   1. /chat returns tool_audit containing at least one entry whose
//      name starts with `api_` (the TOOL_NAME_PREFIX we configured on
//      mcp-openapi-proxy).
//   2. The MCP tool call returned ok:true and surfaced real data the
//      LLM then summarised in its final assistant bubble.
//
// Prerequisites for this spec to NOT skip:
//   * lmstudio (or compatible) is running with a tool-capable model
//     (`gpt-oss-20b` is known-good).
//   * `uvx` is on PATH so slopfinity can spawn `mcp-openapi-proxy`.
//     The first chat request will be SLOW (~5-15s) while uvx hydrates
//     its package cache; subsequent requests reuse the singleton.
//   * The `mcp` python SDK is installed in slopfinity's runtime
//     environment (`pip install -r requirements-slopfinity.txt`).
//
// If `uvx` is missing OR the proxy fails to spawn, the chat handler
// degrades gracefully (still serves the 13 hardcoded tools) — in that
// case this spec FAILS, which is what we want: a regression that
// silently disabled MCP would otherwise go unnoticed.

const { test, expect } = require('../_fixtures');

const BASE = process.env.SLOPFINITY_URL || 'http://localhost:9099';

test.use({ ignoreErrors: { console: [/Failed to load resource.*404/i] } });

test.describe('chat-mcp-tool (live :9099 + real LLM + mcp-openapi-proxy)', () => {
    // First /chat call spawns uvx subprocess — generous timeout.
    test.setTimeout(120_000);

    test('LLM picks an api_* tool when asked for slopfinity state', async ({ page, request }) => {
        // Skip cleanly if no LLM provider configured.
        const health = await request.get(`${BASE}/llm/health`)
            .then(r => r.ok() ? r.json() : null)
            .catch(() => null);
        test.skip(!health || health.ok === false, `No LLM available — /llm/health=${JSON.stringify(health)}`);

        // Bypass the chat UI and hit /chat directly so we can inspect
        // tool_audit. (The UI path is covered by chat-compose-real.)
        // The prompt is engineered to invite a state-query tool call —
        // "how many assets" maps cleanly onto api_get_assets which
        // returns {total: N, items: [...]}.
        const res = await request.post(`${BASE}/chat`, {
            data: {
                messages: [{
                    role: 'user',
                    content: 'Use a tool to tell me exactly how many assets are in the slopfinity output directory right now.',
                }],
            },
            timeout: 110_000,
        });
        expect(res.ok()).toBeTruthy();

        const body = await res.json();
        expect(body.ok).toBe(true);
        expect(Array.isArray(body.tool_audit)).toBe(true);

        // At least one MCP-proxied tool call. The exact tool name
        // depends on how mcp-openapi-proxy slugifies /assets — could
        // be api_get_assets, api_assets_get, etc. We accept any
        // `api_*` whose result was ok.
        const apiCalls = body.tool_audit.filter(t => (t.name || '').startsWith('api_'));
        expect(apiCalls.length, `expected ≥1 api_* tool call, got: ${JSON.stringify(body.tool_audit.map(t => t.name))}`).toBeGreaterThan(0);
        const okApi = apiCalls.find(t => t.result && t.result.ok !== false);
        expect(okApi, `expected at least one api_* call with ok:true result`).toBeTruthy();

        // The final assistant bubble references the data the tool
        // returned. Don't pin to a specific number (the asset count
        // changes); just assert the bubble has substantive content.
        const lastAssistant = [...body.messages].reverse()
            .find(m => m.role === 'assistant' && (m.content || '').trim());
        expect(lastAssistant, 'expected an assistant reply').toBeTruthy();
        expect(lastAssistant.content.length).toBeGreaterThan(10);
    });
});
