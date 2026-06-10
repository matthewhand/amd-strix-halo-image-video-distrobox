# scripts/legacy/

Superseded helper scripts. Kept for reference, not for extension.

## Files in this directory

| File | Replaced by |
|------|-------------|
| `generate_ltx_workflow.py` | `scripts/generate_ltx23_workflow.py` |

`generate_ltx_workflow.py` produced ComfyUI API workflow JSON for the
original LTX-2 19B node graph. The active version,
`scripts/generate_ltx23_workflow.py`, targets the LTX-2.3 node graph
(distilled checkpoints, audio VAE, STG guider, separate AV latents)
and is the one wired up to the wave runners under `tests/`.

## Policy

- Do NOT extend or refactor this script.
- If you need behaviour from it, port it into
  `generate_ltx23_workflow.py` instead.
- Safe to delete once the LTX-2.3 workflow generator has been in
  production for a release cycle.
