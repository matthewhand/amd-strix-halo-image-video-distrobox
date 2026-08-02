#!/usr/bin/env bash
# Detect and export ROCm toolchain paths from the _rocm_sdk_core package

# Query Python for the embedded ROCm paths using the toolbox venv first
_python_bin="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python3}"
_python_bin="${_python_bin:-/opt/venv/bin/python3}"

if command -v "$_python_bin" >/dev/null 2>&1; then
  _rocm_exports="$($_python_bin - <<'PY'
import pathlib
try:
    import _rocm_sdk_core as r
except ModuleNotFoundError:
    raise SystemExit(1)
base = pathlib.Path(r.__file__).parent / "lib" / "llvm" / "bin"
lib = pathlib.Path(r.__file__).parent / "lib"
print(f'export TRITON_HIP_LLD_PATH="{base / "ld.lld"}"')
print(f'export TRITON_HIP_CLANG_PATH="{base / "clang++"}"')
print(f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"')
PY
)" || _rocm_exports=""

  if [[ -n "$_rocm_exports" ]]; then
    eval "$_rocm_exports"
  else
    printf 'warning: _rocm_sdk_core not available; ROCm toolchain paths not exported\n' >&2
  fi
else
  printf 'warning: python interpreter for ROCm env not found (%s)\n' "$_python_bin" >&2
fi

# Enable Triton AMD backend for flash-attn
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
