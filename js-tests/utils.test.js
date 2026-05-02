import { describe, it, expect } from 'vitest';
import { _htmlEscape, _renderKvPretty, _renderKvPrettySafe } from './_shim.js';

describe('_htmlEscape', () => {
    it('escapes ampersand', () => {
        expect(_htmlEscape('a&b')).toBe('a&amp;b');
    });
    it('escapes less-than', () => {
        expect(_htmlEscape('<script>')).toBe('&lt;script&gt;');
    });
    it('escapes double quote', () => {
        expect(_htmlEscape('"hello"')).toBe('&quot;hello&quot;');
    });
    it('escapes single quote', () => {
        expect(_htmlEscape("it's")).toBe('it&#39;s');
    });
    it('leaves safe strings unchanged', () => {
        expect(_htmlEscape('hello world')).toBe('hello world');
    });
    it('handles null/undefined gracefully', () => {
        expect(_htmlEscape(null)).toBe('');
        expect(_htmlEscape(undefined)).toBe('');
    });
    it('converts non-string input to string', () => {
        expect(_htmlEscape(42)).toBe('42');
    });
});

describe('_renderKvPretty', () => {
    it('renders null as em-dash span', () => {
        expect(_renderKvPretty(null)).toContain('kv-null');
    });
    it('renders string with kv-string class', () => {
        const out = _renderKvPretty('hello');
        expect(out).toContain('kv-string');
        expect(out).toContain('hello');
    });
    it('escapes HTML in string values', () => {
        const out = _renderKvPretty('<b>bold</b>');
        expect(out).toContain('&lt;b&gt;');
    });
    it('renders number with kv-number class', () => {
        const out = _renderKvPretty(42);
        expect(out).toContain('kv-number');
        expect(out).toContain('42');
    });
    it('renders boolean true', () => {
        expect(_renderKvPretty(true)).toContain('true');
        expect(_renderKvPretty(false)).toContain('false');
    });
    it('renders empty array', () => {
        expect(_renderKvPretty([])).toContain('kv-empty');
    });
    it('renders array with indexed kv-row entries', () => {
        const out = _renderKvPretty(['a', 'b']);
        expect(out).toContain('kv-array');
        expect(out).toContain('kv-row');
    });
    it('renders empty object', () => {
        expect(_renderKvPretty({})).toContain('kv-empty');
    });
    it('renders object with kv-key entries', () => {
        const out = _renderKvPretty({ foo: 'bar' });
        expect(out).toContain('kv-object');
        expect(out).toContain('foo');
        expect(out).toContain('bar');
    });
    it('escapes object keys', () => {
        const out = _renderKvPretty({ '<evil>': 'x' });
        expect(out).toContain('&lt;evil&gt;');
    });
    it('renders nested objects recursively', () => {
        const out = _renderKvPretty({ a: { b: 1 } });
        expect(out).toContain('kv-number');
    });
});

describe('_renderKvPrettySafe', () => {
    it('returns null span for null input', () => {
        expect(_renderKvPrettySafe(null)).toContain('kv-null');
    });
    it('parses valid JSON string', () => {
        const out = _renderKvPrettySafe('{"x":1}');
        expect(out).toContain('kv-number');
    });
    it('returns pre block for non-JSON string', () => {
        const out = _renderKvPrettySafe('not json at all');
        expect(out).toContain('<pre');
        expect(out).toContain('kv-raw');
    });
    it('escapes HTML in non-JSON string', () => {
        const out = _renderKvPrettySafe('<b>bad</b>');
        expect(out).toContain('&lt;b&gt;');
    });
    it('passes already-parsed object through', () => {
        const out = _renderKvPrettySafe({ key: 'val' });
        expect(out).toContain('key');
    });
});
