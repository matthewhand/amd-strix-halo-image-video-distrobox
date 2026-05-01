#!/usr/bin/env node
// Read Playwright's `artifacts/results.json` (json reporter output) and
// emit two compact files:
//   * artifacts/failed-tests.json — structured array, one entry per failure
//   * artifacts/failure-packet.md  — human-readable summary, capped at 20
//
// Used by .github/workflows/playwright.yml to build a "failure packet"
// the LLM can read first instead of the raw CI firehose. The packet is
// uploaded as an artifact on every run; downstream consumers (a
// "diagnose-failure" agent, a Slack notifier, etc.) can pull just this
// without scrolling through 600 lines of node output.

import fs from 'node:fs';
import path from 'node:path';

const RESULTS = process.argv[2] || 'artifacts/results.json';
const OUT_DIR = path.dirname(RESULTS);
const OUT_FAILED = path.join(OUT_DIR, 'failed-tests.json');
const OUT_PACKET = path.join(OUT_DIR, 'failure-packet.md');

if (!fs.existsSync(RESULTS)) {
  console.error(`[extract-failures] no results file at ${RESULTS}`);
  fs.writeFileSync(OUT_FAILED, '[]');
  fs.writeFileSync(OUT_PACKET, '# Failure packet\n\nNo results.json (test run did not complete?)\n');
  process.exit(0);
}

const report = JSON.parse(fs.readFileSync(RESULTS, 'utf8'));
const failures = [];

function walkSuite(suite, parents = []) {
  const here = [...parents, suite.title].filter(Boolean);
  for (const child of suite.suites ?? []) walkSuite(child, here);
  for (const spec of suite.specs ?? []) {
    for (const test of spec.tests ?? []) {
      const bad = (test.results ?? []).find(r => ['failed', 'timedOut'].includes(r.status));
      if (!bad) continue;
      failures.push({
        suite: parents.filter(Boolean),
        title: spec.title,
        project: test.projectName,
        status: bad.status,
        error: bad.error?.message ?? '',
        location: spec.file,
        // Optional: attachments (trace/screenshot paths) — useful when
        // the agent wants to escalate beyond the packet.
        attachments: (bad.attachments ?? []).map(a => ({ name: a.name, path: a.path })),
      });
    }
  }
}

for (const suite of report.suites ?? []) walkSuite(suite);

fs.writeFileSync(OUT_FAILED, JSON.stringify(failures, null, 2));

const md = [
  '# Failure packet',
  '',
  `Generated: ${new Date().toISOString()}`,
  `Failed tests: ${failures.length}`,
  '',
  ...(failures.length === 0
    ? ['_No failures._']
    : failures.slice(0, 20).flatMap(f => [
        `## ${f.title}`,
        `- Project: ${f.project ?? 'unknown'}`,
        `- Status: ${f.status}`,
        `- File: ${f.location}`,
        `- Error: ${String(f.error).split('\n')[0] || 'n/a'}`,
        ...(f.attachments?.length
          ? [`- Attachments: ${f.attachments.map(a => a.name).join(', ')}`]
          : []),
        '',
      ])),
  ...(failures.length > 20 ? ['', `_... ${failures.length - 20} more failures truncated_`] : []),
].join('\n');

fs.writeFileSync(OUT_PACKET, md);
console.log(`[extract-failures] ${failures.length} failures → ${OUT_FAILED} + ${OUT_PACKET}`);
