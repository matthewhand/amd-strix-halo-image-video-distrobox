/**
 * Thin extract of pure utility functions from app.js for unit testing.
 * This file is NOT served to the browser — it's a test-only module.
 * When Phase B modularises app.js, these should point at the real module.
 */

export function _htmlEscape(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

export function _renderKvPretty(value, depth) {
    depth = depth || 0;
    if (value === null || value === undefined) {
        return '<span class="kv-null">—</span>';
    }
    const t = typeof value;
    if (t === 'string') {
        return `<span class="kv-string">${_htmlEscape(value)}</span>`;
    }
    if (t === 'number') {
        return `<span class="kv-number">${value}</span>`;
    }
    if (t === 'boolean') {
        return `<span class="kv-bool">${value ? 'true' : 'false'}</span>`;
    }
    if (Array.isArray(value)) {
        if (!value.length) return '<span class="kv-empty">[]</span>';
        return `<div class="kv-pretty kv-array">${value.map((v, i) =>
            `<div class="kv-row"><span class="kv-key">${i}</span>${_renderKvPretty(v, depth + 1)}</div>`
        ).join('')}</div>`;
    }
    if (t === 'object') {
        const entries = Object.entries(value);
        if (!entries.length) return '<span class="kv-empty">{}</span>';
        return `<div class="kv-pretty kv-object">${entries.map(([k, v]) =>
            `<div class="kv-row"><span class="kv-key">${_htmlEscape(k)}</span>${_renderKvPretty(v, depth + 1)}</div>`
        ).join('')}</div>`;
    }
    return _htmlEscape(String(value));
}

export function _renderKvPrettySafe(raw) {
    if (raw == null) return '<span class="kv-null">—</span>';
    let parsed = raw;
    if (typeof raw === 'string') {
        try { parsed = JSON.parse(raw); }
        catch (_) {
            return `<pre class="kv-raw whitespace-pre-wrap break-all text-[10px]">${_htmlEscape(raw)}</pre>`;
        }
    }
    return _renderKvPretty(parsed, 0);
}

// ---------------------------------------------------------------------------
// Chat history helpers (extracted from app.js chat section)
// ---------------------------------------------------------------------------

const _CHAT_HISTORY_KEY = 'slopfinity-chat-history-v1';
const _CHAT_HISTORY_MAX = 50;
const _CHAT_TREE_KEY = 'slopfinity-chat-tree-v1';

export function _getChatHistory() {
    try {
        const raw = localStorage.getItem(_CHAT_HISTORY_KEY);
        const arr = raw ? JSON.parse(raw) : [];
        return Array.isArray(arr) ? arr : [];
    } catch (_) { return []; }
}

export function _setChatHistory(arr) {
    const trimmed = (Array.isArray(arr) ? arr : []).slice(-_CHAT_HISTORY_MAX);
    try { localStorage.setItem(_CHAT_HISTORY_KEY, JSON.stringify(trimmed)); } catch (_) { }
}

export function _chatGetTree() {
    try {
        const raw = localStorage.getItem(_CHAT_TREE_KEY);
        if (raw) {
            const t = JSON.parse(raw);
            if (t && t.nodes && typeof t.nextId === 'number') return t;
        }
    } catch (_) { }
    return { nodes: {}, active: null, nextId: 1 };
}

export function _chatSetTree(tree) {
    try { localStorage.setItem(_CHAT_TREE_KEY, JSON.stringify(tree)); } catch (_) { }
}

export function _chatActiveChain() {
    const tree = _chatGetTree();
    if (!tree.active || !tree.nodes[tree.active]) return [];
    const chain = [];
    let cur = tree.active;
    const seen = new Set();
    while (cur && tree.nodes[cur] && !seen.has(cur)) {
        seen.add(cur);
        chain.push(tree.nodes[cur]);
        cur = tree.nodes[cur].parent;
    }
    return chain.reverse();
}

export function _chatForkAt(nodeId, newText) {
    const tree = _chatGetTree();
    const node = tree.nodes[nodeId];
    if (!node) return null;
    const newId = String(tree.nextId++);
    tree.nodes[newId] = {
        id: newId,
        role: 'user',
        content: newText,
        parent: node.parent,
        ts: Date.now(),
    };
    tree.active = newId;
    _chatSetTree(tree);
    return newId;
}

export function _chatSiblingsOf(nodeId) {
    const tree = _chatGetTree();
    const node = tree.nodes[nodeId];
    if (!node) return [];
    const parentId = node.parent;
    return Object.values(tree.nodes)
        .filter(n => n.parent === parentId && n.role === 'user')
        .map(n => n.id);
}
