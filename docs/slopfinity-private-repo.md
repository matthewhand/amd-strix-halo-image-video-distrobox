# Slopfinity private repo status

**Private home:** https://github.com/matthewhand/slopfinity (private)

**Public toolbox:** https://github.com/matthewhand/amd-strix-halo-image-video-distrobox

## Current layout (2026-07-13)

| Location | State |
|----------|--------|
| Parent `slopfinity/` | Still an **in-tree** package (not a git submodule). |
| Private `main` | Full monorepo-shaped dump (Dockerfile, compose, nested `slopfinity/…`). Last push ~2026-06-13 — stale; the canonical code lives in this toolbox repo's `slopfinity/` in-tree package. |
| Private `wip/from-toolbox-20260713` | Snapshot of current toolbox WIP (fleet / stage-gate / LTX / WAN / service-registry). **Do not force-push over `main` without review.** |

## Why not submodule yet

`git submodule add … slopfinity` against private `main` would nest as `slopfinity/slopfinity/server.py` and break `bin/slopfinity`, compose mounts, and imports.

**Before converting the parent path to a submodule:**

1. Reshape private repo so dashboard code lives at package-friendly paths (or pin a branch that already does).
2. Keep private `main` history/backup; land extract on a branch first.
3. Then replace parent in-tree `slopfinity/` with a submodule pin + `git submodule update --init`.

## Contract (stable either way)

- Toolbox owns backends (ComfyUI / ROCm image / TTS / music launchers).
- Slopfinity owns the dashboard and talks over **URLs + env** (`SLOPFINITY_*`, `TTS_WORKER_URL`, Settings → Endpoints).
- See `README.slopfinity.md` for bind/security/env details.

## Remotes in this checkout

```text
origin     → matthewhand/amd-strix-halo-image-video-distrobox
slopfinity → matthewhand/slopfinity
```

Push WIP mirror only:

```bash
git push -u slopfinity HEAD:wip/from-toolbox-YYYYMMDD
```

## See also

- Runtime / ownership boundary (toolbox vs dashboard): [`slopfinity-toolbox-boundary.md`](slopfinity-toolbox-boundary.md)
- Operator docs: [`../README.slopfinity.md`](../README.slopfinity.md)

## Public vs private surface

| Surface | Repo / path | Audience |
|---------|-------------|----------|
| **Public toolbox** | `matthewhand/amd-strix-halo-image-video-distrobox` — includes in-tree `slopfinity/` | Anyone cloning the toolbox |
| **Private mirror** | `matthewhand/slopfinity` — extract / history WIP | Maintainers only |
| **Not public** | Model caches, `dist/demo/`, runtime `config.json`, private tokens | Never publish |

Canonical **code** for the dashboard today: **toolbox in-tree `slopfinity/`**, not private `main`.

## See also (index)

- Full docs map: [`slopfinity-docs-index.md`](slopfinity-docs-index.md)

