# Slopfinity demo bundle

This directory holds the project-specific inputs to the
[`static-demo-builder`](https://github.com/anthropics/claude-code) skill.

## Layout

```
demo/
├── fixtures/             # Network + ticker mock data (JSON)
├── samples/              # Pre-rendered slop assets (PNG / MP4)
├── skill-templates/      # Copy of the skill's app-agnostic templates
└── README.md             # This file
```

## Build the demo

```bash
make demo           # produces dist/demo/
make demo-serve     # builds + serves at http://localhost:8765/
```

## Embed in your marketing site

```html
<iframe src="https://matthewhand.github.io/amd-strix-halo-image-video-distrobox/"
        width="100%" height="900" loading="lazy"
        sandbox="allow-scripts allow-same-origin allow-forms"></iframe>
```

## Refresh fixtures after a UI change

When new endpoints land in the real app, re-run the inventory:

```bash
grep -nE "fetch\(['\"\`]" slopfinity/static/app.js | sort -u
```

Add any new endpoints to `fixtures/network.json` (static, deck, or echo —
see `skill-templates/SKILL.md` for guidance).

## Refresh samples

Replace `demo/samples/*.png` with newer pre-rendered outputs. Keep the
filenames listed in `demo/fixtures/ticker.json#completed_pool` in sync.

## Limitations

The demo is a **canned, single-user, non-AI experience**. The banner and
inline ⓘ badges call this out on every AI-limited surface. If a visitor
hits an unmocked endpoint, the shim returns `{ok: true, _unmocked: …}`
and logs to the console — adjust `fixtures/network.json` to cover it.
