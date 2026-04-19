# tests/legacy/

Pre-LTX-2.3 prototypes. Kept for reference, not for extension.

These scripts were the iterative scaffolding used while bringing up the
Qwen image -> LTX-2 video pipeline on Strix Halo. They have been
superseded by the LTX-2.3 wave runner family:

- `tests/test_qwen_to_ltx23.py` - canonical image -> video pipeline
- `tests/test_ironic_wave.py`, `test_chained_wave.py`, `test_tone_wave.py` - themed batch runners
- `tests/matrix_runner.py` - parameter-sweep harness

## Files in this directory

| File | Replaced by |
|------|-------------|
| `test_qwen_to_ltx2.py` | `test_qwen_to_ltx23.py` |
| `test_ltx2_audio_video.py` | wave runners (audio path now in workflow) |
| `test_ltx2_variations.py` | `matrix_runner.py` |
| `test_qwen_variations.py` | `matrix_runner.py` |
| `test_qwen_generation.py` | covered by wave runners' image phase |
| `test_waldo_birdseye.py` | one-off prompt; rolled into wave runners |
| `test_comfyui_ltx2.sh` | `test_qwen_to_ltx23.py` |
| `run_pipeline.sh` | `test_qwen_to_ltx23.py` |
| `run_pipeline_wave2.sh` | wave runners |
| `run_both_waves.sh` | wave runners + `matrix_runner.py` |

## Policy

- Do NOT extend or refactor these.
- If you need behaviour from one of them, port it into the active
  LTX-2.3 wave family rather than reviving the legacy file.
- Safe to delete entirely once the new wave runners have been in
  production for a release cycle.
