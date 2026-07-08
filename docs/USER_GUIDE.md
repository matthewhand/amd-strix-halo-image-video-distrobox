# Slopfinity user guide

Slopfinity is the dashboard that drives the AMD Strix Halo image / video / audio toolbox. This guide walks every visible surface of the app through the screenshots captured by the Playwright `*-shots.spec.js` suites under `e2e/`.

Every screenshot in `e2e/artifacts/` is referenced below. If you add a new artifact and it isn't linked here, the docs-drift gate (see [Documentation maintenance](#documentation-maintenance)) will fail.

## Table of contents

1. [Quick start](#1-quick-start)
2. [The Pipeline (image / video / audio / voice / upscale)](#2-the-pipeline)
3. [Prompt card modes](#3-prompt-card-modes)
4. [Chat compose & suggestions](#4-chat-compose--suggestions)
5. [Queue submission](#5-queue-submission)
6. [Slop gallery & previews](#6-slop-gallery--previews)
7. [Endless / Infinity mode](#7-endless--infinity-mode)
8. [Story grouping](#8-story-grouping)
9. [Layouts](#9-layouts)
10. [Responsive viewports](#10-responsive-viewports)
11. [Mobile](#11-mobile)
12. [Settings](#12-settings)
13. [Documentation maintenance](#13-documentation-maintenance)

---

## 1. Quick start

Slopfinity boots into the default split-pane layout. The smoke spec verifies the page renders cleanly with no console errors.

![Smoke test](../e2e/artifacts/smoke.png)

The default layout shows the Prompt card on the left, the Queue card on the right, and the Slop gallery below. A floating nav at the bottom of the viewport jumps between focus modes.

![Default layout](../e2e/artifacts/layout-default.png)

Visit `http://localhost:9099/` (or whatever `SLOPFINITY_BIND_PORT` your `bin/slopfinity up` set). Open the dashboard from any device on the LAN — bind host defaults to `127.0.0.1`, override with `SLOPFINITY_BIND_HOST=0.0.0.0 bin/slopfinity up` to expose on the network.

---

## 2. The Pipeline

The Pipeline modal is where you pick which models to invoke at each stage: **base image → video → music → voice → upscale**. Open it via the ⚙ Pipeline button in the navbar.

### Voice model: DramaBox

DramaBox (Resemble AI's LTX-2.3 expressive TTS) is one of three engines in the Voice model dropdown alongside Kokoro and Qwen-TTS.

![DramaBox dropdown option](../e2e/artifacts/dramabox-dropdown-selected.png)

Selecting **DramaBox (expressive)** reveals an extras panel with a voice picker (bundled reference WAVs) and an "expressive speech" checkbox that asks the LLM enhancer to shape the output into DramaBox's quote-and-stage-direction format.

![DramaBox extras panel](../e2e/artifacts/dramabox-extras-panel.png)

Pick **Custom WAV…** in the voice picker to upload your own reference recording (≥10 s of clean speech). The upload is hashed by sha256, stored under `<EXP_DIR>/dramabox_voices/`, and added to the picker as a stable option.

![Custom WAV uploaded](../e2e/artifacts/dramabox-after-upload.png)

When the **Shape LLM prompts for expressive speech** checkbox is on, the concept-stage enhancer appends a one-paragraph addendum that teaches the LLM to emit DramaBox's prompt grammar (`"quoted dialogue" stage-direction`). It's a no-op for kokoro / qwen.

![Expressive checkbox on](../e2e/artifacts/dramabox-expressive-on.png)

### Other model dropdowns

Each model dropdown — base image, video, music, voice, upscale — has a 🎲 **Random** option. When picked, `/config` resolves it to a uniformly-chosen real model from that role's pool (defined in `slopfinity/routers/config.py::_RANDOM_CANDIDATES`). Use it with Infinity mode to rotate through every model over a run.

---

## 3. Prompt card modes

The Prompt card switches between four modes via the pill row at its top.

| Mode | What it does |
| --- | --- |
| Simple | one prompt, one image |
| Raw | exactly-as-typed (no LLM rewrite) |
| Endless | seed → auto-cycle story continuations |
| Chat | LLM conversation that hands off prompts to the queue |

### Simple

![Simple — full viewport](../e2e/artifacts/pane-simple-full.png)

The cropped card view shows just the Prompt card body:

![Simple — card](../e2e/artifacts/pane-simple-card.png)

Focused (single-card) layout via `?layout=subjects`:

![Simple — focused](../e2e/artifacts/pane-simple-focused.png)

### Raw

![Raw — full viewport](../e2e/artifacts/pane-raw-full.png)

![Raw — card](../e2e/artifacts/pane-raw-card.png)

![Raw — focused](../e2e/artifacts/pane-raw-focused.png)

### Endless

![Endless — full viewport](../e2e/artifacts/pane-endless-full.png)

![Endless — card](../e2e/artifacts/pane-endless-card.png)

![Endless — focused](../e2e/artifacts/pane-endless-focused.png)

### Chat

![Chat — full viewport](../e2e/artifacts/pane-chat-full.png)

![Chat — card](../e2e/artifacts/pane-chat-card.png)

![Chat — focused](../e2e/artifacts/pane-chat-focused.png)

---

## 4. Chat compose & suggestions

The chat pane lets you converse with the configured LLM provider. Each assistant reply can be sent to the queue with one click.

### Compose lifecycle

Happy path: a message is submitted, the assistant replies, the reply renders inline.

![Compose happy](../e2e/artifacts/chat-compose-happy.png)

Error path: the network call fails and the UI surfaces it without losing the typed prompt.

![Compose error](../e2e/artifacts/chat-compose-error.png)

Live mode (against the real server) confirms the same roundtrip works end-to-end.

![Compose live](../e2e/artifacts/chat-compose-live.png)

### Mocked compose across layouts

The same chat surface in the default layout, with mocked backend:

![Chat mocked — default](../e2e/artifacts/pane-chat-mocked-default.png)

…and the long-conversation variant where many bubbles scroll:

![Chat mocked — long, default](../e2e/artifacts/pane-chat-mocked-long-default.png)

…the focused single-pane variant:

![Chat mocked — long, focused](../e2e/artifacts/pane-chat-mocked-long-focused.png)

…the mobile viewport:

![Chat mocked — mobile](../e2e/artifacts/pane-chat-mocked-mobile.png)

…and the Subjects-card focus mode:

![Chat mocked — subjects](../e2e/artifacts/pane-chat-mocked-subjects.png)

### Suggestion chips & thought bubbles

The chat pane suggests follow-up prompts as chips beneath each assistant turn — clicking one fills the compose box.

![Suggestion chips](../e2e/artifacts/pane-chat-suggestion-chips.png)

When the LLM is mid-stream you'll see a "thought bubble" placeholder rather than empty space.

![Thought bubbles](../e2e/artifacts/pane-chat-thought-bubbles.png)

---

## 5. Queue submission

The Queue card shows pending, working, and done items. Submitting a prompt from any mode lands here.

### Happy path

![Queue submit — happy](../e2e/artifacts/queue-submit-happy.png)

### Empty submission no-op

Clicking submit with no text is silently ignored — no spurious queue entries.

![Queue submit — empty no-op](../e2e/artifacts/queue-submit-empty-noop.png)

### Live-server drawer

Against a real server, the queue drawer animates open with the freshly queued item visible.

![Queue submit — live drawer](../e2e/artifacts/queue-submit-live-drawer.png)

---

## 6. Slop gallery & previews

The Slop gallery aggregates generated assets (images, video chains, music, speech) from `<EXP_DIR>/`. As of the DramaBox integration it also walks `<EXP_DIR>/tts/` so TTS WAVs appear alongside everything else.

### Clicking an image preview

Happy path: clicking a tile opens the asset-info modal with the file inline.

![Slop image click — happy](../e2e/artifacts/slop-image-click-happy.png)

Same flow against a live server, confirming network roundtrip:

![Slop image click — live](../e2e/artifacts/slop-image-click-live.png)

When an asset is referenced but the file is missing, the modal surfaces a clear error with the filename instead of a generic "not found".

![Slop image click — missing](../e2e/artifacts/slop-image-click-missing.png)

The asset-info modal carries **← Prev / Next →** buttons that walk the visible preview grid (greyed out at the ends). Arrow keys do the same while the modal is open.

The gallery-only layout shows just the preview grid:

![Layout — gallery](../e2e/artifacts/layout-gallery.png)

---

## 7. Endless / Infinity mode

Endless mode auto-cycles continuations from a seed prompt. You'll see the Prompt card change copy and a "running" badge appear on the Queue card.

Before starting — pre-start state:

![Endless pre-start](../e2e/artifacts/endless-prestart.png)

Mid-run, no rows yet drawn:

![Endless running](../e2e/artifacts/endless-running.png)

With rows scrolling in:

![Endless running with rows](../e2e/artifacts/endless-running-with-rows.png)

Multiple rows after a few cycles:

![Endless rows × 3](../e2e/artifacts/endless-rows-3.png)

---

## 8. Story grouping

When a single chat turn generates multiple "story beats", they share a `story_id` and collapse under a group header in the queue with a "view all" button.

![Story — view all grouped](../e2e/artifacts/story-01-view-all-grouped.png)

---

## 9. Layouts

The view dropdown in the navbar (or `?layout=<name>` query) flips between focus configurations.

| Layout | Screenshot |
| --- | --- |
| Default — Prompt + Queue + Slop | `layout-default.png` (see [§1](#1-quick-start)) |
| Subjects — Prompt only | shown below |
| Queue — Queue only | shown below |
| Gallery — Slop only | shown below |
| Subjects + Queue | shown below |
| Subjects + Slop | shown below |
| Queue + Slop | shown below |

![Layout — subjects](../e2e/artifacts/layout-subjects.png)

![Layout — queue](../e2e/artifacts/layout-queue.png)

![Layout — gallery](../e2e/artifacts/layout-gallery.png) (also shown in §6)

![Layout — subjects + queue](../e2e/artifacts/layout-subj-queue.png)

![Layout — subjects + slop](../e2e/artifacts/layout-subj-slop.png)

![Layout — queue + slop](../e2e/artifacts/layout-queue-slop.png)

---

## 10. Responsive viewports

The dashboard adapts to three viewport sizes (compact / medium / large). Captured at 1920 px, 1280 px, and 768 px.

### Large viewport (≥1280 px)

| Section | Screenshot |
| --- | --- |
| Default | `deepui-large-default.png` |
| Gallery | `deepui-large-gallery.png` |
| Queue   | `deepui-large-queue.png` |
| Chrome (navbar / chips) | `deep-ui-large-chrome.png` |
| Settings drawer | `deep-ui-large-settings.png` |

![Large default](../e2e/artifacts/deepui-large-default.png)
![Large gallery](../e2e/artifacts/deepui-large-gallery.png)
![Large queue](../e2e/artifacts/deepui-large-queue.png)
![Large chrome](../e2e/artifacts/deep-ui-large-chrome.png)
![Large settings](../e2e/artifacts/deep-ui-large-settings.png)

### Medium viewport (~1280 px)

![Medium default](../e2e/artifacts/deepui-medium-default.png)
![Medium gallery](../e2e/artifacts/deepui-medium-gallery.png)
![Medium queue](../e2e/artifacts/deepui-medium-queue.png)
![Medium chrome](../e2e/artifacts/deep-ui-medium-chrome.png)
![Medium settings](../e2e/artifacts/deep-ui-medium-settings.png)

### Compact viewport (~768 px)

![Compact default](../e2e/artifacts/deepui-compact-default.png)
![Compact gallery](../e2e/artifacts/deepui-compact-gallery.png)
![Compact queue](../e2e/artifacts/deepui-compact-queue.png)
![Compact chrome](../e2e/artifacts/deep-ui-compact-chrome.png)
![Compact settings](../e2e/artifacts/deep-ui-compact-settings.png)

---

## 11. Mobile

Below 1024 px the dashboard redirects to a single-card layout with a bottom nav.

When you land on a multi-pane layout on a narrow screen, the app redirects:

![Mobile — default redirect](../e2e/artifacts/mobile-default-redirect.png)

Single-card Prompt view:

![Mobile — prompt](../e2e/artifacts/mobile-prompt.png)

Single-card Queue view:

![Mobile — queue](../e2e/artifacts/mobile-queue.png)

Single-card Slop view:

![Mobile — slop](../e2e/artifacts/mobile-slop.png)

The bottom nav-bar's ends are now persistent **📖 Guide** and **⭐ GitHub** anchors (replacing the disappearing prev/next arrows at the cycle ends).

---

## 12. Settings

The Settings drawer (⚙ in the navbar) holds the rest of the configuration: LLM provider details, disk-guard thresholds, branding, model preferences, etc.

![Settings drawer open](../e2e/artifacts/settings-open.png)

Per-viewport-size copies of the settings drawer are captured under [§10 Responsive viewports](#10-responsive-viewports).

---

## 13. Documentation maintenance

### Regenerating screenshots

Bring the dashboard up and run the shot specs:

```bash
bin/slopfinity up                                # uvicorn at :9099
SLOPFINITY_URL=http://localhost:9099 \
  npx playwright test e2e/*-shots.spec.js        # writes e2e/artifacts/*.png
```

Individual specs:

| Spec | What it produces |
| --- | --- |
| `e2e/smoke.spec.js` | `smoke.png` |
| `e2e/layouts.spec.js` | `layout-*.png` |
| `e2e/prompt-pane-shots.spec.js` | `pane-*.png` |
| `e2e/dramabox-shots.spec.js` | `dramabox-*.png` |
| `e2e/chat-mocked.spec.js` | `chat-compose-*.png`, `pane-chat-mocked-*.png` |
| `e2e/chat-suggestion-send.spec.js` | `pane-chat-suggestion-chips.png` |
| `e2e/queue-submit-roundtrip.spec.js` | `queue-submit-*.png` |
| `e2e/slop-image-click.spec.js` | `slop-image-click-*.png` |
| `e2e/endless-running.spec.js`, `e2e/endless-rows.spec.js` | `endless-*.png` |
| `e2e/mobile-nav-shots.spec.js` | `mobile-*.png` |
| `e2e/deep-ui-*.spec.js` | `deepui-*.png`, `deep-ui-*.png` |
| `e2e/settings-drawer.spec.js` | `settings-open.png` |
| `e2e/story-grouping.spec.js` | `story-*.png` |

### Adding a new screenshot

1. Add the `page.screenshot({ path: 'e2e/artifacts/<name>.png' })` call to a new or existing `*-shots.spec.js`.
2. Run the spec — confirm the PNG lands in `e2e/artifacts/`.
3. Reference the file in this guide (`![alt](../e2e/artifacts/<name>.png)`) in the most relevant section.
4. The drift gate (TODO: `scripts/check-userguide-drift.js`) will block CI if an artifact is unreferenced.

### Two-suite split

| Dir | Backend | Use |
| --- | --- | --- |
| `e2e/` | Mocked via `page.route` | CI-safe. `make e2e`. |
| `e2e/live/` | Real slopfinity server on :9099 | Local-only. `make e2e-live` or `E2E_INCLUDE_LIVE=1 npx playwright test`. |
