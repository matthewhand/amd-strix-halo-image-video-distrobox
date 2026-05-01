#!/usr/bin/env python3
"""
build_demo.py — generic helper for the static-demo-builder skill.

Reads a project's production index.html (or template) and writes a
demo-mode entrypoint into the dist/demo/ bundle:

  * Render any Jinja `{{ ... }}` interpolations from a JSON context
    (so the demo HTML is fully baked, not template source).
  * Strip any leftover Jinja `{% ... %}` and `{# ... #}` blocks.
  * Inject the demo sentinel and the demo-shim.js <script>.
  * Add `<meta name="robots" content="noindex,nofollow">` and a
    canonical link if a real production URL is provided.
  * Strip any third-party analytics / error-reporter <script> tags
    whose src matches a configurable host allowlist.

Usage:
  python3 build_demo.py <out_dir> \\
      --src=slopfinity/templates/index.html \\
      --canonical=https://github.com/you/your-app \\
      --jinja-context=demo/fixtures/jinja-context.json \\
      --strip-jinja \\
      [--strip-script-host=plausible.io,sentry.io]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote as _urlencode

try:
    import jinja2
except ImportError:  # pragma: no cover
    jinja2 = None


SENTINEL = (
    '<script>window.__IS_DEMO__=true;</script>\n'
    '<link rel="stylesheet" href="./static/demo-banner.css">\n'
)
SHIM_TAG = '<script src="./static/demo-shim.js"></script>\n'


def inject_meta(html: str, canonical: str | None) -> str:
    head_match = re.search(r"<head[^>]*>", html, re.IGNORECASE)
    if not head_match:
        return html
    inject = '\n<meta name="robots" content="noindex,nofollow">\n'
    if canonical:
        inject += f'<link rel="canonical" href="{canonical}">\n'
    end = head_match.end()
    return html[:end] + inject + html[end:]


def inject_sentinel(html: str) -> str:
    head_match = re.search(r"<head[^>]*>", html, re.IGNORECASE)
    if not head_match:
        return SENTINEL + html
    end = head_match.end()
    return html[:end] + "\n" + SENTINEL + html[end:]


def inject_shim_before_first_app_script(html: str) -> str:
    pat = re.compile(
        r'<script\s+[^>]*src=["\'](?:[^"\']*)?app\.js[^"\']*["\'][^>]*></script>',
        re.IGNORECASE,
    )
    m = pat.search(html)
    if not m:
        # Fallback: inject right before </body>
        return html.replace("</body>", SHIM_TAG + "</body>", 1)
    return html[: m.start()] + SHIM_TAG + html[m.start():]


def strip_third_party_scripts(html: str, hosts: list[str]) -> str:
    if not hosts:
        return html
    pat = re.compile(
        r'<script\s+[^>]*src=["\']([^"\']+)["\'][^>]*></script>',
        re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        src = m.group(1)
        if any(h in src for h in hosts):
            return f'<!-- demo: stripped {src} -->'
        return m.group(0)

    return pat.sub(repl, html)


def strip_jinja_blocks(html: str) -> str:
    """Remove Jinja `{% ... %}` and `{# ... #}` tags so demo HTML is static.
    Also strip any leftover `{{ ... }}` interpolations — anything that
    survived a `jinja_render` pass is loop-local debris from a `{% for %}`
    block whose iterator was stripped, and rendering it as text would
    leak template syntax into the demo HTML.
    Final pass: tags whose `src` or `href` ends up empty / orphan-prefix
    (e.g. `src="/files/"` from a stripped `{{ f }}`) become broken
    requests in the browser. Strip those tags entirely so the demo
    doesn't 404 on dead loop-body markup."""
    html = re.sub(r"{#.*?#}", "", html, flags=re.DOTALL)
    html = re.sub(r"{%.*?%}", "", html, flags=re.DOTALL)
    html = re.sub(r"\{\{[^}]*\}\}", "", html)
    # Remove tags with orphan src/href ending in '/' (e.g. /files/, /asset/)
    html = re.sub(
        r'<(source|img|script|video|audio|link|iframe)\b[^>]*\b(src|href)=["\'][^"\']*/["\'][^>]*/?>',
        "",
        html,
        flags=re.IGNORECASE,
    )
    return html


def jinja_render(html: str, ctx_path: Path) -> str:
    """Render the template against a JSON context. Tolerates missing keys
    (chainable Undefined yields '' for unknown attrs) so demo doesn't crash
    on every detail.

    Note: we render the template first, then `--strip-jinja` cleans up
    the {% %} blocks (they remained as text after Jinja saw them as
    statements during render but couldn't execute them safely without
    full app context).
    """
    if not ctx_path:
        return html
    if jinja2 is None:
        print("[build_demo] WARNING: jinja2 not installed — skipping render", file=sys.stderr)
        return html
    if not ctx_path.exists():
        print(f"[build_demo] jinja context missing: {ctx_path}", file=sys.stderr)
        return html

    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))

    env = jinja2.Environment(
        undefined=jinja2.ChainableUndefined,
        autoescape=False,
    )
    env.filters["urlencode"] = lambda s: _urlencode(s or "")

    # Recursively wrap dicts in a class supporting dot-access for Jinja
    class _DotDict(dict):
        def __getattr__(self, name):
            v = self.get(name)
            if v is None:
                return jinja2.ChainableUndefined(name=name)
            if isinstance(v, dict):
                return _DotDict(v)
            if isinstance(v, list):
                return [_DotDict(x) if isinstance(x, dict) else x for x in v]
            return v

    rendered_ctx = {k: _DotDict(v) if isinstance(v, dict) else v for k, v in ctx.items()}
    try:
        rendered = env.from_string(html).render(**rendered_ctx)
    except jinja2.exceptions.TemplateError as e:
        print(f"[build_demo] jinja render warning: {e}", file=sys.stderr)
        rendered = html
    return rendered


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path, help="dist/demo target directory")
    p.add_argument("--src", required=True, type=Path, help="source index.html (template ok)")
    p.add_argument("--canonical", default=None, help="canonical URL of the real app")
    p.add_argument("--strip-script-host", default="", help="comma-separated list of script src hosts to strip")
    p.add_argument("--strip-jinja", action="store_true", help="strip raw Jinja {% %} and {# #} tags")
    p.add_argument("--jinja-context", default=None, type=Path, help="JSON file with Jinja context for {{ ... }}")
    args = p.parse_args()

    if not args.src.exists():
        print(f"[build_demo] source not found: {args.src}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    html = args.src.read_text(encoding="utf-8")

    # Render Jinja FIRST so {{ ... }} are baked in.
    if args.jinja_context:
        html = jinja_render(html, args.jinja_context)
    if args.strip_jinja:
        html = strip_jinja_blocks(html)

    hosts = [h.strip() for h in args.strip_script_host.split(",") if h.strip()]
    html = strip_third_party_scripts(html, hosts)
    html = inject_meta(html, args.canonical)
    html = inject_sentinel(html)
    html = inject_shim_before_first_app_script(html)

    out = args.out_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"[build_demo] wrote {out} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
