# Tests — CI Rule: No Real AI Calls

This project's CI runs on free-tier GitHub runners with no GPU and no
model weights. **Every AI surface MUST be mocked when tested.** Real model
inference is forbidden in CI, even via `workflow_dispatch`.

| Surface | Mock |
|---|---|
| Qwen-Image worker (`/api/generate`) | `tests/mock_qwen_server.py` |
| LLM provider (LM Studio / Ollama / OpenAI-compat `/v1/chat/completions`) | `tests/mock_llm_server.py` |
| TTS worker (`/tts`) | `tests/mock_tts_server.py` |
| ComfyUI (`/free`, image gen) | not yet — add when needed |

## Adding a new test that touches an AI surface

1. If a mock for that surface doesn't exist yet, add
   `tests/mock_<surface>_server.py` following the stdlib `http.server`
   pattern of the existing mocks (no extra pip deps).
2. Spawn the mock via `subprocess.Popen` from your test, point the
   relevant env var (or seeded `config.json`) at it, and tear down on
   exit (`atexit` + `try/finally`).
3. A pytest test under `tests/` is picked up automatically by the
   gating `.github/workflows/python-tests.yml` job (it runs all of
   `tests/` except `e2e_qwen_web_test.py`). The lightweight
   `.github/workflows/e2e-qwen-web-test.yml::e2e-qwen-web-light` job
   also runs the mock-backed surface.
4. **NEVER call a real model in CI** — even via `workflow_dispatch`. The
   heavy `e2e-qwen-web` job is already locked to a self-hosted runner
   that doesn't exist; keep it that way.

## The mock backends

Three stdlib-only (`http.server`, no extra pip deps) mock servers stand
in for the AI surfaces. Each is spawnable as a `subprocess` and binds to
loopback. Verified against the module docstrings + handlers:

| Mock | Default bind | Provides |
|---|---|---|
| `tests/mock_llm_server.py` | `127.0.0.1:${LLM_MOCK_PORT:-11434}` | OpenAI-compat LLM. `GET /v1/models` (one mock model), `POST /v1/chat/completions`, `POST /v1/completions`. The completion body is shaped from the request: a system prompt mentioning "STRICT JSON" / "image, video, music, tts" returns a distribute-shaped JSON dict; a user prompt mentioning "Suggest" / "subject ideas" returns a JSON array of subject strings; otherwise a one-sentence rewrite. |
| `tests/mock_qwen_server.py` | `127.0.0.1:${QWEN_MOCK_PORT}` (set by the e2e test when `MOCK_QWEN_WEB=1`) | Qwen-Image web worker. `GET /` -> `{"status":"ok"}`, `POST /api/generate` -> `{"job_id":...}` and writes a tiny valid PNG to `~/.qwen-image-studio/`, `GET /api/job/<id>` -> `{"status":"completed"}`. |
| `tests/mock_tts_server.py` | `127.0.0.1:${TTS_MOCK_PORT:-8010}` | TTS worker proxied by `/tts`. `GET /health`, `POST /tts` -> JSON + writes a tiny valid 1 s 16 kHz mono WAV (44-byte RIFF header + 32000 silence bytes) to `/tmp/mock-tts/`. |

> The `mock_llm_server.py` default port is **11434** (the Ollama
> default) so probe code that targets a local Ollama also hits the mock.
> When you run a host Ollama on `:11434`, point the mock elsewhere
> (`LLM_MOCK_PORT=11500`) — CI uses 11500 for exactly this reason.

## Integration test

`tests/test_ai_mock_integration.py` spawns the LLM + TTS mocks plus a
real `uvicorn slopfinity.server:app` (env: `TTS_WORKER_URL` and
`LLM_PROVIDER_BASE_URL` pointed at the mocks; ports default to free
ports, overridable via `LLM_MOCK_PORT` / `TTS_MOCK_PORT`) and exercises
`/enhance`, `/enhance?distribute`, `/subjects/suggest`, and `/tts`. Run via:

```bash
python -m pytest tests/test_ai_mock_integration.py -v
# or directly:
python tests/test_ai_mock_integration.py
```

> **Host Ollama caveat.** This test inherits your environment
> (`os.environ.copy()`), and the dashboard's LLM pool defaults its CPU
> slot to `http://localhost:11434/v1`
> (`slopfinity/llm/pool.py:get_env_pool_config`). If a real Ollama is
> listening on `:11434` it can be probed instead of the mock and skew
> the result. Blank it with `SLOPFINITY_LLM_CPU_URL=""` (or point
> `SLOPFINITY_LLM_PRIMARY_URL` at the mock) before running.

## Running the full Python suite locally

CI runs the whole suite as a gate (see below). To reproduce locally:

```bash
# A writable experiment/state dir is REQUIRED — paths.py otherwise
# falls back to a path that may be read-only (e.g. /workspace).
export SLOPFINITY_EXP_DIR="$(mktemp -d)"
export SLOPFINITY_STATE_DIR="$SLOPFINITY_EXP_DIR"
# Avoid a host Ollama on :11434 interfering with test_ai_mock_integration:
export SLOPFINITY_LLM_CPU_URL=""        # or point SLOPFINITY_LLM_PRIMARY_URL at the mock
# No ComfyUI locally -> let scheduler.free_between() fail fast:
export SLOPFINITY_COMFY_URL="http://127.0.0.1:1"

python -m pytest tests/ --ignore=tests/e2e_qwen_web_test.py -q
```

`tests/e2e_qwen_web_test.py` is excluded because it needs a real AMD
GPU / ROCm (covered by the self-hosted job in `e2e-qwen-web-test.yml`).

Test deps beyond the runtime requirements: `pytest`, `pytest-asyncio`,
`requests`, plus `numpy` + `pillow` (so `tests/test_vae_grid.py`'s
`pytest.importorskip("numpy")` / `importorskip("PIL.Image")` actually
runs instead of skipping).

## CI gate — `.github/workflows/python-tests.yml`

Runs on every push / PR to `main` (and `workflow_dispatch`). It installs
the runtime + test deps (`fastapi`, `uvicorn[standard]`, `jinja2`,
`python-multipart`, `httpx`, `psutil`, `pyyaml`, `python-dotenv`,
`sqlmodel`, `tiktoken`, `numpy`, `pillow`, `pytest`, `pytest-asyncio`,
`requests`), sets `SLOPFINITY_EXP_DIR` / `SLOPFINITY_STATE_DIR` to a
writable workspace path, `SLOPFINITY_LLM_CPU_URL=""`, and
`SLOPFINITY_COMFY_URL="http://127.0.0.1:1"`, then runs:

```bash
python -m pytest tests/ --ignore=tests/e2e_qwen_web_test.py -q
```

This job **gates** (no `continue-on-error`) — a failing test fails the
build. `tests/test_ai_mock_integration.py` starts its own mock LLM/TTS
servers from its fixture, so the job does not start mocks manually.

## Current test inventory

The `tests/test_*.py` suite is **31 files / 382 test functions**
(`grep -rhE '^\s*(async )?def test_' tests/test_*.py | wc -l`). Highlights:

| File | Area |
|---|---|
| `test_ai_mock_integration.py` | full mock-backed `/enhance` + `/tts` round-trip |
| `test_llm_pool.py`, `test_llm_pool_dedup.py` | LLM pool config + endpoint de-duplication |
| `test_llm_probe.py`, `test_llm_providers.py` | provider probing + OpenAI-compat client |
| `test_router_smoke.py` | FastAPI route smoke |
| `test_config_internals.py` | config load/merge internals |
| `test_branding.py`, `test_branding_loading.py` | branding asset loading |
| `test_stats.py` | system stats endpoint |
| `test_vae_grid.py` | VAE grid helper (numpy/PIL, importorskip-gated) |
| `test_scheduler*.py`, `test_memory_planner.py` | GPU scheduler + memory planning |
| `test_worker_*.py` | per-stage workers (image, video, audio, tts, merge, post, base) |
| `test_server_*.py`, `test_queue_schema.py` | server config/assets/queue + SQLite schema |
| `test_ssrf_guard.py` | SSRF guard on outbound probes |

The `tests/run_*.py` files (`run_smoke.py`, `run_matrix.py`,
`run_all_permutations.py`, the `run_*_wave.py` family) are manual
pipeline drivers, not pytest cases.

---

# E2E Testing for AMD Strix Halo Image & Video Toolbox

## Overview

End-to-end testing for the Qwen Image Web UI that validates the complete workflow from service startup to image generation and persistence.

## Features

- **Service Startup Validation**: Tests Docker Compose service initialization
- **Health Checks**: Validates web UI responsiveness
- **GPU Utilization**: Confirms ROCm GPU acceleration is working
- **API Testing**: Tests image generation via REST API
- **Persistence Validation**: Confirms images persist to host filesystem
- **Ultra-fast Testing**: Uses 4-step generation for speed (2-3x faster)

## Test Coverage

| Test | Description | Duration |
|------|-------------|----------|
| Docker Compose Startup | Validates service deployment | ~30s |
| Service Health | Checks web UI responsiveness | ~30s |
| GPU Utilization | Confirms ROCm GPU access | ~5s |
| Image Generation | Tests API-based generation | ~60s |
| Image Persistence | Validates file system persistence | ~10s |
| **Total** | **Complete E2E workflow** | **~2-3 minutes** |

## Running Tests Locally

### Prerequisites
- Docker and Docker Compose installed
- ROCm-compatible GPU (for full testing)
- Python 3.13+
- Required Python packages: `requests`

### Quick Start

```bash
# Install test dependencies
python -m pip install requests

# Run the complete E2E test
python tests/e2e_qwen_web_test.py

# Run with Docker Compose
docker compose --profile qwen-image up --build -d
python tests/e2e_qwen_web_test.py
docker compose down
```

### Test Configuration

The test uses ultra-fast settings by default:
- **Steps**: 4 (vs 50 normal)
- **CFG Scale**: 1.0
- **Size**: 1:1 aspect ratio
- **Timeout**: 3 minutes total

### Environment Variables

```bash
# Optional: Override test settings
export QWEN_TEST_STEPS=4
export QWEN_TEST_SEED=12345
export QWEN_TEST_PROMPT="Custom test prompt"
```

## Test Workflow

1. **Service Deployment**
   - Starts Docker Compose with Qwen-only profile
   - Waits for services to initialize
   - Validates container health

2. **Service Health Check**
   - Tests HTTP connectivity to localhost:8000
   - Validates web UI responsiveness
   - Retries with exponential backoff

3. **GPU Utilization Test**
   - Checks PyTorch ROCm availability inside container
   - Validates GPU device detection
   - Confirms GPU acceleration is enabled

4. **Image Generation API Test**
   - Submits generation job via REST API
   - Uses ultra-fast settings (4 steps)
   - Polls for completion status
   - Validates successful generation

5. **Persistence Test**
   - Checks for generated images in `~/.qwen-image-studio/`
   - Validates file size (>1MB indicates real image)
   - Confirms host filesystem persistence

6. **Cleanup**
   - Stops Docker Compose services
   - Removes temporary test artifacts

## API Endpoints Tested

### Submit Generation
```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "E2E test: bright blue star on dark background",
  "num_images": 1,
  "steps": 4,
  "size": "1:1",
  "seed": 12345,
  "use_fast": true
}
```

### Check Job Status
```http
GET /api/job/{job_id}
```

## Expected Output

```
🧪 Starting Qwen Image Web UI E2E Test
==================================================
🚀 Testing Docker Compose startup...
✅ Docker Compose Startup
🏥 Testing service health...
✅ Service Health
🔥 Testing GPU utilization...
✅ GPU Utilization
🎨 Testing image generation via API...
✅ Image Generation
💾 Testing image persistence...
✅ Image Persistence
==================================================
📊 TEST RESULTS
==================================================
Docker Compose Startup      ✅ PASS
Service Health              ✅ PASS
GPU Utilization             ✅ PASS
Image Generation           ✅ PASS
Image Persistence           ✅ PASS

Summary: 5/5 tests passed
🎉 All tests passed!
```

## CI/CD Integration

The test integrates with GitHub Actions for automated testing:

- **Full E2E**: Runs on pushes to `main` and `e2e-testing` branches
- **Lightweight PR**: Syntax validation on pull requests
- **Manual Dispatch**: Available for on-demand testing
- **Artifact Upload**: Saves test outputs on failure

## Troubleshooting

### Common Issues

**Service Not Starting**
```bash
# Check Docker status
sudo systemctl status docker
# Check container logs
docker compose logs
```

**GPU Not Detected**
```bash
# Verify ROCm installation
rocm-smi
# Check GPU devices
ls -la /dev/dri /dev/kfd
```

**API Not Responding**
```bash
# Check if service is running
curl http://localhost:8000
# Check container logs
docker logs strix-halo-toolbox
```

**Images Not Persisting**
```bash
# Check output directory permissions
ls -la ~/.qwen-image-studio/
# Verify volume mounts
docker exec strix-halo-toolbox ls -la /root/.qwen-image-studio/
```

### Debug Mode

Run with verbose output:
```bash
python -v tests/e2e_qwen_web_test.py
```

Enable debug logging:
```bash
export E2E_DEBUG=1
python tests/e2e_qwen_web_test.py
```

## Performance Benchmarks

| Test | Typical Duration | Success Rate |
|------|-----------------|--------------|
| Service Startup | 30-45s | 95% |
| API Generation | 45-90s | 90% |
| Persistence | 5-15s | 100% |
| **Complete Test** | **2-3 minutes** | **~85%** |

*Performance depends on GPU specs and network speed for model downloads*

## Contributing

When adding new tests:

1. Follow the existing naming convention
2. Include timeout and error handling
3. Add cleanup procedures
4. Update documentation
5. Test locally before submitting PR

## Future Enhancements

- [ ] Multi-GPU testing support
- [ ] Video generation E2E tests
- [ ] Load testing scenarios
- [ ] Cross-platform compatibility tests
- [ ] Performance regression testing