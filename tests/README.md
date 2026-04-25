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
3. The lightweight CI job
   (`.github/workflows/e2e-qwen-web-test.yml::e2e-qwen-web-light`) MUST
   run your test.
4. **NEVER call a real model in CI** — even via `workflow_dispatch`. The
   heavy `e2e-qwen-web` job is already locked to a self-hosted runner
   that doesn't exist; keep it that way.

## Integration test

`tests/test_ai_mock_integration.py` spawns all three mocks plus a real
`uvicorn slopfinity.server:app` and exercises `/enhance`,
`/enhance?distribute`, `/subjects/suggest`, and `/tts`. Run via:

```bash
python -m pytest tests/test_ai_mock_integration.py -v
# or directly:
python tests/test_ai_mock_integration.py
```

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