import { describe, it, expect, beforeEach } from 'vitest';
import { clearStorage } from './setup.js';
import {
    _getChatHistory,
    _setChatHistory,
    _chatActiveChain,
    _chatGetTree,
    _chatSetTree,
    _chatForkAt,
    _chatSiblingsOf,
} from './_shim.js';

beforeEach(() => {
    clearStorage();
});

describe('_getChatHistory / _setChatHistory', () => {
    it('returns empty array when nothing stored', () => {
        expect(_getChatHistory()).toEqual([]);
    });

    it('round-trips messages correctly', () => {
        const msgs = [{ role: 'user', content: 'hello' }];
        _setChatHistory(msgs);
        expect(_getChatHistory()).toEqual(msgs);
    });

    it('trims history to 50 items', () => {
        const msgs = Array.from({ length: 60 }, (_, i) => ({ role: 'user', content: `m${i}` }));
        _setChatHistory(msgs);
        const stored = _getChatHistory();
        expect(stored.length).toBe(50);
        // Most recent 50 preserved
        expect(stored[0].content).toBe('m10');
    });

    it('handles corrupt localStorage gracefully', () => {
        localStorage.setItem('slopfinity-chat-history-v1', '{not json}');
        expect(_getChatHistory()).toEqual([]);
    });

    it('handles non-array stored value gracefully', () => {
        localStorage.setItem('slopfinity-chat-history-v1', '"a string"');
        expect(_getChatHistory()).toEqual([]);
    });
});

describe('chat tree', () => {
    it('returns empty tree when nothing stored', () => {
        const tree = _chatGetTree();
        expect(tree.nodes).toEqual({});
        expect(tree.active).toBeNull();
        expect(tree.nextId).toBe(1);
    });

    it('round-trips tree correctly', () => {
        const tree = { nodes: { '1': { id: '1', role: 'user', content: 'hi', parent: null, ts: 1 } }, active: '1', nextId: 2 };
        _chatSetTree(tree);
        expect(_chatGetTree()).toEqual(tree);
    });
});

describe('_chatActiveChain', () => {
    it('returns empty array when no active node', () => {
        expect(_chatActiveChain()).toEqual([]);
    });

    it('returns single node chain', () => {
        _chatSetTree({
            nodes: { '1': { id: '1', role: 'user', content: 'hi', parent: null } },
            active: '1',
            nextId: 2,
        });
        const chain = _chatActiveChain();
        expect(chain.length).toBe(1);
        expect(chain[0].content).toBe('hi');
    });

    it('returns chain in chronological order (root first)', () => {
        _chatSetTree({
            nodes: {
                '1': { id: '1', role: 'user', content: 'first', parent: null },
                '2': { id: '2', role: 'assistant', content: 'reply', parent: '1' },
                '3': { id: '3', role: 'user', content: 'second', parent: '2' },
            },
            active: '3',
            nextId: 4,
        });
        const chain = _chatActiveChain();
        expect(chain.map(m => m.content)).toEqual(['first', 'reply', 'second']);
    });

    it('handles cycle protection (broken tree)', () => {
        // Node points to itself as parent — should not infinite loop
        _chatSetTree({
            nodes: { '1': { id: '1', role: 'user', content: 'bad', parent: '1' } },
            active: '1',
            nextId: 2,
        });
        const chain = _chatActiveChain();
        expect(chain.length).toBe(1);
    });
});

describe('_chatForkAt', () => {
    it('creates a sibling node', () => {
        _chatSetTree({
            nodes: {
                '1': { id: '1', role: 'user', content: 'original', parent: null },
            },
            active: '1',
            nextId: 2,
        });
        const newId = _chatForkAt('1', 'edited');
        expect(newId).toBe('2');
        const tree = _chatGetTree();
        expect(tree.nodes['2'].content).toBe('edited');
        expect(tree.nodes['2'].parent).toBeNull(); // same parent as '1'
        expect(tree.active).toBe('2');
    });

    it('returns null for unknown nodeId', () => {
        _chatSetTree({ nodes: {}, active: null, nextId: 1 });
        expect(_chatForkAt('nonexistent', 'x')).toBeNull();
    });

    it('increments nextId', () => {
        _chatSetTree({
            nodes: { '5': { id: '5', role: 'user', content: 'x', parent: null } },
            active: '5',
            nextId: 6,
        });
        _chatForkAt('5', 'y');
        expect(_chatGetTree().nextId).toBe(7);
    });
});

describe('_chatSiblingsOf', () => {
    it('returns only self when no forks', () => {
        _chatSetTree({
            nodes: { '1': { id: '1', role: 'user', content: 'a', parent: null } },
            active: '1',
            nextId: 2,
        });
        const sibs = _chatSiblingsOf('1');
        expect(sibs).toContain('1');
        expect(sibs.length).toBe(1);
    });

    it('returns both siblings after a fork', () => {
        _chatSetTree({
            nodes: {
                '1': { id: '1', role: 'user', content: 'a', parent: null },
                '2': { id: '2', role: 'user', content: 'b', parent: null },
            },
            active: '2',
            nextId: 3,
        });
        const sibs = _chatSiblingsOf('1');
        expect(sibs).toContain('1');
        expect(sibs).toContain('2');
        expect(sibs.length).toBe(2);
    });

    it('returns empty array for unknown node', () => {
        _chatSetTree({ nodes: {}, active: null, nextId: 1 });
        expect(_chatSiblingsOf('999')).toEqual([]);
    });
});
