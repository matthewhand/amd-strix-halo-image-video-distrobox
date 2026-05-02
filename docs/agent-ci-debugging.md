# Agent playbook — debugging a Playwright CI failure

When the **Playwright Tests** workflow goes red and an agent (or human)
needs to figure out why, **read in this order** to keep the agent's
context window unflooded:

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

## 4. `artifacts/slopfinity.log` (only on red runs)

The dashboard's stdout/stderr during the test run. Useful when the
failure is "the dashboard 500'd" rather than "the spec is wrong" — a
500 in slopfinity.log means the bug is in `slopfinity/server.py`, not
the test.

---

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
