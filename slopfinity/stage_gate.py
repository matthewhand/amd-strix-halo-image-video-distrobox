"""Host-safe GPU stage gate — hard floor before any model load.

Belt-and-braces: refuse to start a stage unless
``MemAvailable >= need_gb + safety_gb`` *after* a real reclaim pass.

Pure planner (Belady) answers "what to keep"; this module **enforces**
park/start so UMA cannot stack Qwen + HeartMuLa + LTX + TTS the way
``run_fleet`` did during the OOM campaign.

Designed for injection (mem_reader, registry, reclaim_fn) so unit tests
prove OOM refusal without a GPU.
"""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

log = logging.getLogger("slopfinity.stage_gate")

# Conservative peaks (GB). Prefer over-estimate over OOM.
# Mirrors scheduler.STAGE_BUDGETS + DramaBox (missing there historically).
STAGE_BUDGETS: Dict[Tuple[str, str], float] = {
    ("image", "qwen"): 28,
    ("image", "ernie"): 18,
    ("image", "ltx-2.3"): 38,
    ("video", "ltx-2.3"): 48,
    ("video", "wan2.2"): 84,
    ("video", "wan2.5"): 96,
    ("audio", "heartmula"): 14,
    ("tts", "qwen-tts"): 10,
    ("tts", "qwen"): 10,
    ("tts", "kokoro"): 8,
    ("tts", "dramabox"): 18,  # Gemma encoder + DiT — must not be 0
    ("upscale", "ltx-spatial"): 30,
}

DEFAULT_SAFETY_GB = 10.0
UMA_HEAVY_DEFAULT = ("qwen-image", "heartmula", "comfyui")


class InsufficientMemoryError(RuntimeError):
    """Raised when headroom is still too low after reclaim — do not start."""


def need_gb(role: str, model: str) -> float:
    """Peak GB expected for (role, model). Unknown pairs return a safe default."""
    if not model or model in ("none", ""):
        return 0.0
    key = (str(role).lower(), str(model).lower())
    if key in STAGE_BUDGETS:
        return float(STAGE_BUDGETS[key])
    # alias map
    aliases = {
        ("tts", "qwen3-tts"): 10.0,
        ("image", "qwen-image"): 28.0,
    }
    if key in aliases:
        return aliases[key]
    # Unknown model — refuse to pretend it is free
    return 24.0


def remaining_after_load(available_gb: float, need_gb: float) -> float:
    """Estimate MemAvailable after loading ``need_gb`` of weights/activations."""
    return float(available_gb) - float(need_gb)


def has_safety_after_load(
    available_gb: float,
    need_gb: float = 0.0,
    safety_gb: float = DEFAULT_SAFETY_GB,
    *,
    already_loaded: bool = False,
) -> bool:
    """True iff we keep ≥ safety_gb free after the model is resident.

    - Pre-load (already_loaded=False): require available - need ≥ safety
      i.e. available ≥ need + safety.
    - Post-load (already_loaded=True): require available ≥ safety
      (need already consumed; only the free floor matters).
    """
    safety = float(safety_gb)
    if already_loaded:
        return float(available_gb) + 1e-9 >= safety
    return remaining_after_load(available_gb, need_gb) + 1e-9 >= safety


def can_start(available_gb: float, need_gb: float, safety_gb: float = DEFAULT_SAFETY_GB) -> bool:
    """True iff estimated free RAM after load stays ≥ safety_gb (default 10GB)."""
    return has_safety_after_load(available_gb, need_gb, safety_gb, already_loaded=False)


def headroom_gb(mem_reader: Optional[Callable[[], float]] = None) -> float:
    """Current MemAvailable in GB."""
    if mem_reader is not None:
        return float(mem_reader())
    return _read_mem_available_gb()


def _read_mem_available_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # kB
                    kb = float(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def reclaim_all(
    reason: str = "",
    *,
    mem_reader: Optional[Callable[[], float]] = None,
    registry: Any = None,
    free_comfy_fn: Optional[Callable[[], dict]] = None,
    kill_launchers_fn: Optional[Callable[[], List[str]]] = None,
    park_ids: Sequence[str] = ("heartmula", "qwen-tts", "qwen-image", "comfyui"),
) -> Dict[str, Any]:
    """Ordered reclaim ladder; always re-reads MemAvailable at the end."""
    read = mem_reader or _read_mem_available_gb
    before = float(read())
    actions: List[str] = []

    # R1 — Comfy unload weights (cheap)
    if free_comfy_fn is not None:
        try:
            free_comfy_fn()
            actions.append("comfy_free")
        except Exception as e:
            actions.append(f"comfy_free_err:{e}")
    else:
        try:
            from . import scheduler as _sched

            # free_between is async — run best-effort sync via new loop if needed
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Cannot block; skip async free in nested context
                    actions.append("comfy_free_skipped_running_loop")
                else:
                    loop.run_until_complete(_sched.free_between())
                    actions.append("comfy_free")
            except RuntimeError:
                asyncio.run(_sched.free_between())
                actions.append("comfy_free")
        except Exception as e:
            actions.append(f"comfy_free_err:{type(e).__name__}")

    # R2/R3 — park known services
    reg = registry
    if reg is None:
        try:
            from . import service_registry as reg  # type: ignore
        except Exception:
            reg = None

    if reg is not None:
        for sid in park_ids:
            try:
                reg.ensure_down(sid)
                actions.append(f"ensure_down:{sid}")
            except Exception as e:
                actions.append(f"ensure_down_err:{sid}:{type(e).__name__}")
        try:
            reg.ensure_down_group("uma-heavy")
            actions.append("ensure_down_group:uma-heavy")
        except Exception as e:
            actions.append(f"ensure_down_group_err:{type(e).__name__}")

    # R4 — orphan launchers
    if kill_launchers_fn is not None:
        try:
            killed = kill_launchers_fn() or []
            for k in killed:
                actions.append(f"kill:{k}")
        except Exception as e:
            actions.append(f"kill_err:{type(e).__name__}")
    else:
        for pat in (
            "qwen_launcher.py",
            "ernie_launcher.py",
            "dramabox_launcher.py",
            "heartmula_launcher.py",
            "wan_launcher.py",
        ):
            try:
                subprocess.run(
                    ["pkill", "-f", pat],
                    check=False,
                    capture_output=True,
                    timeout=5,
                )
                actions.append(f"pkill:{pat}")
            except Exception:
                pass

    time.sleep(0.5)
    after = float(read())
    log.info(
        "reclaim_all reason=%s before=%.1f after=%.1f actions=%s",
        reason,
        before,
        after,
        actions,
    )
    return {
        "ok": True,
        "reason": reason,
        "before_gb": before,
        "after_gb": after,
        "freed_gb": round(max(0.0, after - before), 2),
        "actions": actions,
    }


@contextlib.contextmanager
def stage_gate(
    role: str,
    model: str,
    *,
    need: Optional[float] = None,
    safety_gb: float = DEFAULT_SAFETY_GB,
    mem_reader: Optional[Callable[[], float]] = None,
    registry: Any = None,
    reclaim_fn: Optional[Callable[[str], dict]] = None,
    service_id: Optional[str] = None,
    exclusive_group: Optional[str] = "uma-heavy",
    exclusive_members: Sequence[str] = UMA_HEAVY_DEFAULT,
    keep_after: bool = False,
    ensure_up: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Enter only when headroom is proven; park peers; park self on exit.

    Raises ``InsufficientMemoryError`` instead of starting a load that would
    OOM the host.
    """
    read = mem_reader or _read_mem_available_gb
    need_v = float(need if need is not None else need_gb(role, model))
    safety = float(safety_gb)
    sid = service_id or _default_service_id(role, model)

    reg = registry
    if reg is None and ensure_up:
        try:
            from . import service_registry as reg  # type: ignore
        except Exception:
            reg = None

    reclaim = reclaim_fn or (
        lambda reason: reclaim_all(reason, mem_reader=read, registry=reg)
    )

    available = float(read())
    reclaimed = False
    # Pre-load: estimate free AFTER model = available - need; must be ≥ safety (10GB).
    if not has_safety_after_load(available, need_v, safety, already_loaded=False):
        reclaim(
            f"pre-start {role}/{model} avail={available:.1f} "
            f"est_after_load={remaining_after_load(available, need_v):.1f} "
            f"need={need_v}+keep_free={safety}"
        )
        reclaimed = True
        available = float(read())

    if not has_safety_after_load(available, need_v, safety, already_loaded=False):
        est = remaining_after_load(available, need_v)
        raise InsufficientMemoryError(
            f"insufficient memory for {role}/{model}: "
            f"available={available:.1f}GB need={need_v:.1f}GB "
            f"est_free_after_load={est:.1f}GB require_free_after>={safety:.1f}GB "
            f"(after_reclaim={reclaimed})"
        )

    # Park exclusive peers before ensure_up
    if reg is not None and exclusive_group and sid:
        for peer in exclusive_members:
            if peer == sid:
                continue
            try:
                reg.ensure_down(peer)
            except Exception:
                pass
        # Also park TTS if starting a heavy non-tts stage
        if role not in ("tts",) and sid != "qwen-tts":
            try:
                reg.ensure_down("qwen-tts")
            except Exception:
                pass

    if ensure_up and reg is not None and sid:
        ens = reg.ensure_up(sid)
        if isinstance(ens, dict) and ens.get("ok") is False:
            raise RuntimeError(f"ensure_up({sid}) failed: {ens}")

    # Post-load: measured free must still be ≥ safety (model already resident).
    available = float(read())
    if not has_safety_after_load(available, 0.0, safety, already_loaded=True):
        if reg is not None and sid:
            try:
                reg.ensure_down(sid)
            except Exception:
                pass
        raise InsufficientMemoryError(
            f"free RAM after load below floor for {role}/{model}: "
            f"available={available:.1f}GB require_free>={safety:.1f}GB "
            f"(service={sid})"
        )

    info = {
        "role": role,
        "model": model,
        "need_gb": need_v,
        "available_gb": available,
        "est_free_after_load_gb": remaining_after_load(available + need_v, need_v)
        if not ensure_up
        else available,
        "require_free_after_gb": safety,
        "service_id": sid,
        "reclaimed": reclaimed,
    }
    try:
        yield info
    finally:
        if not keep_after and reg is not None and sid and ensure_up:
            try:
                reg.ensure_down(sid)
            except Exception:
                pass
        # If still under safety, last-ditch reclaim
        try:
            if float(read()) < safety:
                reclaim(f"post-stage low water {role}/{model}")
        except Exception:
            pass


def _default_service_id(role: str, model: str) -> Optional[str]:
    r = (role or "").lower()
    m = (model or "").lower()
    if r == "image" and m == "qwen":
        return "qwen-image"
    if r in ("image", "video", "upscale") and "ltx" in m:
        return "comfyui"
    if r == "audio" and "heart" in m:
        return "heartmula"
    if r == "tts":
        return "qwen-tts"
    if r == "video" and m.startswith("wan"):
        return None  # one-shot
    return None


# Sync helper for run_fleet-style code
def run_with_gate(role: str, model: str, fn: Callable[[], Any], **kwargs: Any) -> Any:
    with stage_gate(role, model, **kwargs):
        return fn()
