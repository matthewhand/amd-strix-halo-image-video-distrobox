# Agent playbook — debugging a Playwright CI failure

When the **Playwright Tests** workflow goes red and an agent (or human)
needs to figure out why, **read in this order** to keep the agent's
context window unflooded:

## 0. Know the CI shape first

The `.github/workflows/playwright.yml` job is now **mock-backed**, which
changes what a red run means. Before triaging, internalise this:

- **A mock LLM backend is started.** The job runs
  `tests/mock_llm_server.py` on `127.0.0.1:11500` (port 11500 avoids the
  Ollama default 11434) and points the dashboard's LLM pool at it via
  `SLOPFINITY_LLM_PRIMARY_URL` / `SLOPFINITY_LLM_CPU_URL`. The
  CPU/failover endpoints are blanked so the pool never probes a real
  Ollama / LAN box. So `/llm/health` is green and the suggestion cluster
  renders **enabled**.
- **The backend-gated specs now RUN.** The test step sets
  `SLOPFINITY_HAS_LLM=1`, which flips the gate in `e2e/_fixtures.js` so
  the previously-skipped chat / suggestion / endless / story specs
  execute against the mock instead of self-skipping. (See the gate
  section below.) The job is still `continue-on-error: true`, so a red
  run does **not** block merge — but the failure packet is the signal.
- **A placeholder output is seeded** into `SLOPFINITY_EXP_DIR` (a 1x1
  PNG + a stub `FINAL_*.mp4`) so the Slop output card renders its
  populated layout rather than the empty state.

When a backend-gated spec fails, suspect (in order): the mock's canned
response shape drifting from what the spec asserts
(`artifacts/mock-llm.log`), the dashboard not pointed at the mock, or a
genuine UI regression. When a static / chrome spec fails, the mock is
irrelevant — it's a layout / contract regression.

## 1. `failure-packet.md` (always present)

Compact summary, ≤20 entries. Generated from `results.json` by
`scripts/extract-failures.mjs`. One section per failed spec:

```
## <spec title>
- Project: chromium
- Status: failed
- File: e2e/foo.spec.js
- Error: <first line of the assertion error / timeout reason>
- Attachments: trace.zip, screenshot.png
```

This is the entire surface 80% of failures need. If the error first-line
identifies the cause (selector not found / timeout / 500 from server),
you're done — go fix the code.

## 2. `failed-tests.json` (always present)

Structured per-test data. Same content as the packet but full error
messages (not just first lines), full stack traces, attachment paths,
project metadata. Read this only if the markdown packet's first-line
truncation hid context you need.

## 3. `playwright-report/` (only present when CI is RED)

The full daisyUI'd HTML report with screenshots, traces, network logs,
console logs. Heavy — only download this artifact when the structured
data isn't enough. `npx playwright show-trace <trace.zip>` opens an
interactive debugger.

`test-results/` carries the raw per-test artifacts (PNGs, MP4s, traces)
that the HTML report indexes.

## 4. `artifacts/slopfinity.log` + `artifacts/mock-llm.log` (only on red runs)

`slopfinity.log` is the dashboard's stdout/stderr during the test run.
Useful when the failure is "the dashboard 500'd" rather than "the spec
is wrong" — a 500 in slopfinity.log means the bug is in
`slopfinity/server.py`, not the test.

`mock-llm.log` is the stdout/stderr of `tests/mock_llm_server.py`. Read
it when a backend-gated spec (chat / suggestion / endless / story) fails
in a way that smells like bad data from the LLM — e.g. the suggestion
cluster renders but with the wrong shape, or `/subjects/suggest` returns
something the spec doesn't expect. Both logs are bundled in the
`playwright-failure-artifacts` upload (failure-only).

---

## The `SLOPFINITY_HAS_LLM` gate

Specs that drive the *interactive* LLM machinery (suggestion batches,
endless story rows, chat replies, the story log) import
`{ test, expect }` from `e2e/_fixtures.js` instead of
`@playwright/test`. That module installs an `auto` fixture that calls
`test.skip(!process.env.SLOPFINITY_HAS_LLM, ...)` — so:

- **`SLOPFINITY_HAS_LLM` unset** → those specs report as **skipped**
  (clean signal, not a red).
- **`SLOPFINITY_HAS_LLM=1`** → those specs **run**.

CI sets `SLOPFINITY_HAS_LLM=1` in the test step (the mock backend is up),
so the gated specs run there. The specs currently importing the fixture
(`grep -l _fixtures e2e/*.spec.js`):

```
e2e/chat-suggest-header.spec.js
e2e/chat-suggestion-send.spec.js
e2e/endless-rows.spec.js
e2e/endless-running.spec.js
e2e/story-log-editable.spec.js
e2e/spiffy-toggle.spec.js
e2e/suggestion-contracts.spec.js
```

Static layout / chrome specs (smoke, layouts, settings, etc.) import
`@playwright/test` directly and always run.

> Note: the comment block at the top of `playwright.yml` still describes
> the *old* behaviour where `SLOPFINITY_HAS_LLM` was left unset and the
> gated specs skipped. The **test step actually sets it to `1`**
> (`Run Playwright tests` step) — trust the step env, not the stale
> header comment.

## Running the e2e suite locally against a mock

Reproduce the CI lane without a real backend:

```bash
# 1. Start the mock LLM on 11500 (avoid a host Ollama on 11434).
LLM_MOCK_HOST=127.0.0.1 LLM_MOCK_PORT=11500 \
  python3 tests/mock_llm_server.py &

# 2. Boot the dashboard pointed at the mock, with a writable exp dir.
export SLOPFINITY_EXP_DIR="$(mktemp -d)"
export SLOPFINITY_LLM_PRIMARY_URL=http://127.0.0.1:11500/v1
export SLOPFINITY_LLM_CPU_URL=http://127.0.0.1:11500/v1
python3 dark_server.py &

# 3. Run the suite with the gate on so the backend specs execute.
SLOPFINITY_HAS_LLM=1 npx playwright test
```

Drop `SLOPFINITY_HAS_LLM` to run only the always-on static/chrome specs.
To run a single gated spec: `SLOPFINITY_HAS_LLM=1 npx playwright test
e2e/suggestion-contracts.spec.js`.

## Why this order

The CI artifacts pyramid is **deliberately compact at the top**. A 200-
failure regression has a 200-line packet. A 1-failure regression has a
20-line packet. Either way you read the same bytes per spec.

The HTML report and traces are 50-200 MB each — flooding them through
an LLM's 200k context window is wasteful when the first-line error
identifies 80% of failures already. Escalate to traces only when the
packet says e.g. "Timeout waiting for selector" and you genuinely need
to see what the page looked like at the timeout moment.

## How to read the artifacts in a CI re-run

```bash
gh run download <run-id> --name failure-packet -D /tmp/packet
cat /tmp/packet/failure-packet.md       # 30 seconds, done in 90% of cases

gh run download <run-id> --name test-structured-results -D /tmp/results
jq '.[] | {title, error}' /tmp/results/failed-tests.json | head -50

# Only when the above isn't enough:
gh run download <run-id> --name playwright-failure-artifacts -D /tmp/heavy
npx playwright show-report /tmp/heavy/playwright-report
npx playwright show-trace /tmp/heavy/test-results/.../trace.zip
```

## Job summary on the GitHub UI

The workflow appends the test summary + failure packet to
`$GITHUB_STEP_SUMMARY` so they render directly in the GitHub Actions
run page. **Look there first** before downloading artifacts — it's the
same content, no download needed.

## Adding new specs

When you write a new e2e spec, the failure-packet system needs no
changes. The Playwright JSON reporter captures every spec
automatically; the extractor walks the JSON tree and pulls failures
regardless of project / suite / nesting.

If a spec genuinely needs a richer failure context than first-line of
error, add a `test.info().attach('debug-state', { body: JSON.stringify(state), contentType: 'application/json' })` call in the spec's failing path — Playwright bundles attachments into the trace, and they show up in the packet's "Attachments:" line.

## Local-vs-CI behavior of the Playwright config

| | Local (no `CI` env) | CI (`CI=1`) |
|---|---|---|
| Reporters | list, html | dot, json, junit, html |
| Screenshots | always on | only on failure |
| Videos | retain on failure | retain on failure |
| HTML report | open: never | open: never |
| trace | retain on failure | retain on failure |

The local lane keeps screenshots-on-every-test so the HTML report is a
full visual catalogue when you're debugging interactively. The CI lane
keeps artifacts small so the upload is fast and focused.
