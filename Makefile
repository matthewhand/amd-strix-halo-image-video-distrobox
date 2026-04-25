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

.PHONY: tailwind tailwind-watch tailwind-clean help

help:
	@echo "Targets:"
	@echo "  tailwind        Build $(TAILWIND_OUTPUT) (downloads CLI on first run)"
	@echo "  tailwind-watch  Rebuild on change (Ctrl-C to stop)"
	@echo "  tailwind-clean  Remove the CLI binary + generated CSS"

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
