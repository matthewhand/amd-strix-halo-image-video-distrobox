# Slopfinity build helpers.
#
# Today this is mostly for the Tailwind bundle — replacing the play-CDN
# script tag in templates/index.html with a pre-compiled, minified CSS
# we ship from /static/tailwind.css. The standalone Tailwind binary
# (single file, no Node) is downloaded into bin/ on first use.
#
#   make tailwind        — one-shot build of slopfinity/static/tailwind.css
#   make tailwind-watch  — rebuild on file change (good while iterating)
#   make tailwind-clean  — drop the binary + the generated CSS
#
# License + offline deployment:
#   The Tailwind CLI (and the CSS it emits) is MIT-licensed. You can
#   redistribute the generated tailwind.css inside an offline package
#   (AppImage / tarball / rootfs) freely; ship the upstream LICENSE
#   from https://github.com/tailwindlabs/tailwindcss alongside it to
#   stay compliant. The CLI BINARY itself isn't needed at runtime —
#   only at build time. So an offline deployment looks like:
#     1. `make tailwind` on a machine with internet access.
#     2. Bundle slopfinity/static/tailwind.css with the rest of the
#        static assets (no need to ship bin/tailwindcss).
#     3. Drop the play-CDN <script src="https://cdn.tailwindcss.com">
#        tag in templates/index.html for a local <link> to the
#        bundled file.

TAILWIND_VERSION := latest
TAILWIND_BIN     := bin/tailwindcss
TAILWIND_INPUT   := src/tailwind.css
TAILWIND_OUTPUT  := slopfinity/static/tailwind.css
TAILWIND_CONTENT := slopfinity/templates/*.html,slopfinity/static/app.js

# Pick the right release asset for the host architecture.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Linux)
  ifeq ($(UNAME_M),x86_64)
    TAILWIND_ASSET := tailwindcss-linux-x64
  else ifeq ($(UNAME_M),aarch64)
    TAILWIND_ASSET := tailwindcss-linux-arm64
  else
    $(error Unsupported Linux arch: $(UNAME_M))
  endif
else ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    TAILWIND_ASSET := tailwindcss-macos-arm64
  else
    TAILWIND_ASSET := tailwindcss-macos-x64
  endif
else
  $(error Unsupported OS: $(UNAME_S))
endif

TAILWIND_URL := https://github.com/tailwindlabs/tailwindcss/releases/$(TAILWIND_VERSION)/download/$(TAILWIND_ASSET)

.PHONY: tailwind tailwind-watch tailwind-clean lint e2e e2e-live e2e-all help up dev down logs status

help:
	@echo "Targets:"
	@echo "  up              Start the dashboard (uvicorn only)"
	@echo "  dev             Start uvicorn + tailwind --watch via overmind"
	@echo "  down            Stop whatever's running on the dashboard port"
	@echo "  logs            Tail the server log"
	@echo "  status          Show what's bound on the dashboard port"
	@echo "  tailwind        Build $(TAILWIND_OUTPUT) (downloads CLI on first run)"
	@echo "  tailwind-watch  Rebuild on change (Ctrl-C to stop)"
	@echo "  tailwind-clean  Remove the CLI binary + generated CSS"
	@echo "  lint            stylelint app.css + node --check app.js/sw.js + ast.parse python"
	@echo "  e2e             Playwright CI suite (e2e/, mocked, no AI required)"
	@echo "  e2e-live        Playwright local-only suite (e2e/live/, real :9099 server)"
	@echo "  e2e-all         Both suites in one run"

# Launcher delegates. Real logic lives in bin/slopfinity so the shell
# script is also usable on systems without make.
up:     ; @bin/slopfinity up
dev:    ; @bin/slopfinity dev
down:   ; @bin/slopfinity down
logs:   ; @bin/slopfinity logs
status: ; @bin/slopfinity status

# Download the standalone Tailwind binary on first use, cache in bin/.
$(TAILWIND_BIN):
	@mkdir -p $(dir $@)
	@echo "→ downloading Tailwind CLI ($(TAILWIND_ASSET)) …"
	@curl -sSL --fail -o $@ "$(TAILWIND_URL)"
	@chmod +x $@
	@echo "→ $@ ready"

# Seed input file if it doesn't exist. Bare-bones — you can extend with
# @apply rules + custom layers later.
$(TAILWIND_INPUT):
	@mkdir -p $(dir $@)
	@printf '@tailwind base;\n@tailwind components;\n@tailwind utilities;\n' > $@
	@echo "→ created $@ stub"

tailwind: $(TAILWIND_BIN) $(TAILWIND_INPUT)
	@echo "→ building $(TAILWIND_OUTPUT) …"
	@$(TAILWIND_BIN) -i $(TAILWIND_INPUT) -o $(TAILWIND_OUTPUT) \
		--content "$(TAILWIND_CONTENT)" --minify
	@echo "→ done. Now swap the CDN <script> in templates/index.html for"
	@echo "  <link rel=\"stylesheet\" href=\"/static/tailwind.css\">"

tailwind-watch: $(TAILWIND_BIN) $(TAILWIND_INPUT)
	@$(TAILWIND_BIN) -i $(TAILWIND_INPUT) -o $(TAILWIND_OUTPUT) \
		--content "$(TAILWIND_CONTENT)" --watch

tailwind-clean:
	@rm -f $(TAILWIND_BIN) $(TAILWIND_OUTPUT)
	@echo "→ removed CLI + generated CSS"

# Lint the static assets. stylelint catches structural CSS bugs (the
# kind that would silently invalidate every rule below an unmatched
# brace, e.g.); node --check parses the JS without executing; py_compile
# does the same for the Python entry points. Run before pushing UI/JS
# changes — also wired into .github/workflows/lint.yml on every PR.
lint:
	@command -v npm >/dev/null || { echo "npm required for stylelint"; exit 1; }
	@test -d node_modules || npm ci --silent || npm install --silent
	npx stylelint slopfinity/static/app.css
	node --check slopfinity/static/app.js
	node --check slopfinity/static/sw.js
	python3 -m py_compile slopfinity/server.py run_fleet.py
	@echo "→ lint OK"

# End-to-end smoke test against the live Slopfinity dashboard. Loads
# the page in chromium, asserts no JS pageerror fired, and checks that
# the three primary cards (Subjects / Queue / Slop) all render with
# non-zero size. Catches the silent "JS dies, page renders blank" class
# of regressions that pure stylelint + node --check miss.
#
#   make e2e            — run against http://localhost:9099 by default
#   SLOPFINITY_URL=...  — override target
e2e:
	@command -v npm >/dev/null || { echo "npm required for Playwright"; exit 1; }
	@test -d node_modules/@playwright/test || npm install --silent --save-dev @playwright/test@1.59.1
	@test -d node_modules/playwright/.local-browsers 2>/dev/null || npx playwright install chromium
	npx playwright test --reporter=list
	@echo "→ e2e OK (CI suite — mocked)"

# Local-only — hits the real slopfinity server on :9099. AI workers may or
# may not be running; these specs assert the wire contract end-to-end, not
# generated output. Each test cleans up after itself via POST /queue/cancel.
e2e-live:
	@command -v npm >/dev/null || { echo "npm required for Playwright"; exit 1; }
	@test -d node_modules/@playwright/test || npm install --silent --save-dev @playwright/test@1.59.1
	@test -d node_modules/playwright/.local-browsers 2>/dev/null || npx playwright install chromium
	E2E_INCLUDE_LIVE=1 npx playwright test e2e/live --reporter=list
	@echo "→ e2e-live OK (real :9099 server)"

# Run both suites in a single invocation. CI runs `make e2e`; local QA runs
# `make e2e-all` to cover the wire contract + the mocked frontend behavior.
e2e-all:
	@command -v npm >/dev/null || { echo "npm required for Playwright"; exit 1; }
	@test -d node_modules/@playwright/test || npm install --silent --save-dev @playwright/test@1.59.1
	@test -d node_modules/playwright/.local-browsers 2>/dev/null || npx playwright install chromium
	E2E_INCLUDE_LIVE=1 npx playwright test --reporter=list
	@echo "→ e2e-all OK (CI + live)"

# ─────────────────────────────────────────────────────────────────────
# Static demo bundle (static-demo-builder skill)
# Produces dist/demo/ — a self-hosted interactive demo of the dashboard
# that runs entirely from canned fixtures with no backend.
# ─────────────────────────────────────────────────────────────────────

SKILL_TEMPLATES := demo/skill-templates

.PHONY: demo demo-serve demo-clean demo-smoke

demo:
	@rm -rf dist/demo
	@mkdir -p dist/demo/static dist/demo/samples dist/demo/fixtures
	@python3 $(SKILL_TEMPLATES)/build_demo.py dist/demo \
		--src=slopfinity/templates/index.html \
		--canonical=https://github.com/matthewhand/amd-strix-halo-image-video-distrobox \
		--strip-jinja
	@cp slopfinity/static/app.js  dist/demo/static/
	@cp slopfinity/static/app.css dist/demo/static/
	@cp -r slopfinity/static/icons dist/demo/static/icons 2>/dev/null || true
	@cp slopfinity/static/manifest.webmanifest dist/demo/static/ 2>/dev/null || true
	@echo "// no-op SW for demo bundle" > dist/demo/static/sw.js
	@cp $(SKILL_TEMPLATES)/demo-shim.js     dist/demo/static/
	@cp $(SKILL_TEMPLATES)/demo-banner.html dist/demo/static/
	@cp $(SKILL_TEMPLATES)/demo-banner.css  dist/demo/static/
	@cp demo/fixtures/*.json dist/demo/fixtures/
	@cp demo/samples/* dist/demo/samples/ 2>/dev/null || true
	@cp demo/README.md dist/demo/
	@echo "Demo bundle ready: $$(du -sh dist/demo | cut -f1) at dist/demo/"
	@echo "  → make demo-serve   (or)   python3 -m http.server -d dist/demo 8765"

demo-serve: demo
	@cd dist/demo && python3 -m http.server 8765

demo-smoke: demo
	@(cd dist/demo && python3 -m http.server 8766 >/dev/null 2>&1 &) && sleep 1
	@DEMO_URL=http://localhost:8766/ npx playwright test e2e/demo-smoke.spec.js --reporter=list || true
	@pkill -f "http.server 8766" 2>/dev/null || true

demo-clean:
	@rm -rf dist/demo
