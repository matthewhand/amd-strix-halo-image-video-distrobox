# tests/legacy/

Pre-LTX-2.3 prototypes. Kept for reference, not for extension.

These scripts were the iterative scaffolding used while bringing up the
Qwen image -> LTX-2 video pipeline on Strix Halo. They have been
superseded by the active wave-runner family in `tests/`:

- `tests/run_ironic_wave.py`, `run_chained_wave.py`, `run_tone_wave.py` - themed batch runners
- `tests/run_matrix.py` - parameter-sweep harness
- `tests/run_all_permutations.py` - full cartesian model sweep
- `tests/run_smoke.py` - quick pipeline smoke

## Files in this directory

| File | Replaced by |
|------|-------------|
| `test_qwen_to_ltx2.py` | the wave runners (`run_*_wave.py`) |
| `test_ltx2_audio_video.py` | wave runners (audio path now in workflow) |
| `test_ltx2_variations.py` | `run_matrix.py` |
| `test_qwen_variations.py` | `run_matrix.py` |
| `test_qwen_generation.py` | covered by wave runners' image phase |
| `test_waldo_birdseye.py` | one-off prompt; rolled into wave runners |
| `test_comfyui_ltx2.sh` | the wave runners (`run_*_wave.py`) |
| `run_pipeline.sh` | the wave runners (`run_*_wave.py`) |
| `run_pipeline_wave2.sh` | wave runners |
| `run_both_waves.sh` | wave runners + `run_matrix.py` |

## Policy

- Do NOT extend or refactor these.
- If you need behaviour from one of them, port it into the active
  LTX-2.3 wave family rather than reviving the legacy file.
- Safe to delete entirely once the new wave runners have been in
  production for a release cycle.
