# Slopfinity documentation index

**Last verified:** 2026-07-13  
**Verification method:** `git status` / `git remote -v`; `test -f` on entrypoints; `wc -l` on `slopfinity/` hot paths; `ls` `demo/`, `Makefile` demo targets, `.github/workflows/`; `rg` for overclaims (`gh-pages`, `race-free`, `anthropics/claude-code`, LICENSE); adversarial report `/tmp/goal-achievement-skeptic-report.md`.

Prefer **accuracy over marketing**. If a claim is not true in this tree, fix the doc in the same change set as the code.

## Canonical surfaces (start here)

| Priority | Path | Audience | What it answers |
|----------|------|----------|-----------------|
| 1 | [`../README.slopfinity.md`](../README.slopfinity.md) | Operators | How to run the dashboard, env, security, demo honesty, architecture LOC |
| 2 | [`../README.md` §14](../README.md#14-slopfinity-dashboard-ui) | Toolbox users | UI layout mermaid (current), link-out to product docs |
| 3 | [`slopfinity-toolbox-boundary.md`](slopfinity-toolbox-boundary.md) | Contributors | Ownership, runtime contract, NOT shipped, private sync rules |
| 4 | [`slopfinity-private-repo.md`](slopfinity-private-repo.md) | Maintainers | Private mirror vs in-tree package; submodule hazards |

## Supporting / design (may lag code)

| Path | Notes |
|------|-------|
| [`USER_GUIDE.md`](USER_GUIDE.md) | Operator aid; screenshot-era UI may lag `templates/` + `static/app.js` |
| [`USER_GUIDE-network-services.md`](USER_GUIDE-network-services.md) | Network worker lifecycle notes |
| `docs/*-design.md` | Historical / planned designs — not source of truth for current behavior |
| [`../demo/README.md`](../demo/README.md) | Static demo source notes |
| [`../slopfinity/PWA.md`](../slopfinity/PWA.md) | PWA / service worker details |

## Cross-link mesh (must stay consistent)

```text
README.md §14  ──►  README.slopfinity.md
       │                    │
       ├────────────────────┼──►  docs/slopfinity-toolbox-boundary.md
       │                    │              │
       └────────────────────┴──►  docs/slopfinity-private-repo.md
                                          │
                    docs/slopfinity-docs-index.md (this file) ◄── all of the above should link back when edited
```

## Honesty checklist (docs PRs)

- [ ] No claim that gh-pages / public demo is live unless `.github/workflows` deploys it and `dist/demo` is produced in CI.
- [ ] `demo/skill-templates/` is a **local** static-demo builder — do not attribute to third-party skill registries unless true.
- [ ] Queue is **not** race-free; residual multi-writer races are documented in `slopfinity/config.py`.
- [ ] Do not claim a root `LICENSE` / Apache-2.0 until `LICENSE` exists at repo root.
- [ ] Coordinator is **opt-in**; `run_fleet.py` remains primary runner until code + docs flip together.
- [ ] LOC tables: re-run `wc -l` on listed paths when editing architecture sections.
- [ ] Private `main` is **stale packaging**; canonical dashboard = in-tree `slopfinity/`.

## Commands that exist (preview / demo)

```bash
# Dashboard (standalone loopback)<0x0Apip install -r requirements-slopfinity.txt<0x0ASLOPFINITY_BIND_HOST=127.0.0.1 SLOPFINITY_EXP_DIR=$PWD/comfy-outputs/experiments python3 dark_server.py

# Bundled stack (toolbox compose profile)<0x0Adocker compose --profile slop up -d   # after override yaml as in README.slopfinity.md

# Static demo source → local dist (not deployed)<0x0Amake demo          # → dist/demo/<0x0Amake demo-serve    # http.server :8765 on dist/demo<0x0Amake demo-clean
```

`make demo` requires `demo/skill-templates/build_demo.py` and copies `slopfinity/static/{app.js,app.css}`. There is **no** active `gh-pages` workflow under `.github/workflows/` (only a template under `demo/skill-templates/gh-pages-deploy.yml`).

## Audit artifacts (host `/tmp`, not committed)

| Path | Content |
|------|---------|
| `/tmp/slopfinity-claude-docs-audit.log` | Prior automated docs-audit narrative |
| `/tmp/goal-achievement-skeptic-report.md` | Adversarial G1–G7 goal completion report |
| `/tmp/docs-followup-improvements.md` | This session's ≥3 follow-up fixes list |
