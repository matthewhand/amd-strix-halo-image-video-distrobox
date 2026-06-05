# slopfinity hardening docs

Reference documentation for the queue-concurrency + orchestration hardening work
(commit range `ec7b849^..HEAD`, sweep rounds 1–5). Each doc is written to be
diffed against the source: exact `file:line` references and verbatim excerpts.

## Subsystem references

| Doc | Covers |
|---|---|
| [queue-concurrency.md](queue-concurrency.md) | The lost-update race; `queue_lock()` / `mutate_queue()`; every converted read-modify-write call site; the non-recursive lock contract; the event-loop `asyncio.to_thread` wrapping; the cross-process no-lost-update proof. |
| [queue-persistence-schema.md](queue-persistence-schema.md) | `QueueItem` model + the `extra` JSON catch-all; `_split_queue_item` / `_flatten_queue_item`; the fields that were silently dropped; dual SQLite(authoritative)+JSON(backup) store; `db.py` migrations; `migrate_legacy` + `schema_version`; stable-id stamping. |
| [comfyui-polling.md](comfyui-polling.md) | `_poll_comfy_history` (deadline + socket timeout + error cap + empty-images guard); `_encode_frames_to_mp4`; the five pollers it unified; the gfx1151 GPU-hang rationale; node-id output contract. |
| [run_fleet-iter-lifecycle.md](run_fleet-iter-lifecycle.md) | The per-iteration orchestration: `_config_snapshot` vs `_eff_snap` vs `_CURRENT_ITER_CONFIG`; Fast Track; matrix mode; tier/frames/size derivation + clamps; seed modes (per-task / per-chain FLF2V); `stage_prompts` application; chain handoff + cleanup; iter failure + `_IterCancelled`. |
| [flag-ipc-protocol.md](flag-ipc-protocol.md) | The `terminate.flag` / `pause.flag` / `cancel.flag` file IPC: who writes/reads/deletes each, ordering guarantees, the startup terminate cleanup, the mtime-gated cancel-at-chain-boundary abort. |
| [config-settings-locking.md](config-settings-locking.md) | `load_config`/`save_config`; `config_lock` + `mutate_config` (opt-in); the `llm_cpu_mode` namespace fix; disk-guard POST; the SSRF guard on `base_url` + `tts_worker_url`/`comfy_url`; `api_key` redaction. |
| [scheduler-gpu-serialization.md](scheduler-gpu-serialization.md) | The GPU serialization invariant; `acquire_gpu` admission + release; why gfx1151 mandates it; the "never hold the lock across an unbounded op" principle (`/music` timeout, `/tts` validation); the `resident_models` residual. |

## Cross-cutting

| Doc | Covers |
|---|---|
| [CHANGELOG-hardening.md](CHANGELOG-hardening.md) | Pedantic per-fix changelog for the whole effort: id, severity, sweep round, root cause, mechanism, fix, files, verification. Plus the deferred/residual register. |
| [TODO-followups.md](TODO-followups.md) | Live register of remaining follow-ups (deferred items, residuals, future work) with severity + rationale. |

## How this was produced

Five adversarial issue-sweep rounds (find → two-lens verify → fix → test →
commit), then a documentation pass (draft → fact-check each `file:line` against
source), then a QA pass across build/run/test. See `CHANGELOG-hardening.md` for
the full provenance.
