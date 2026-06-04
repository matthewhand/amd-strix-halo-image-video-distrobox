import os
import json
import time
import re
import glob
import asyncio

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse

from slopfinity.paths import EXP_DIR
import slopfinity.config as cfg
from slopfinity.stats import get_output_counts


router = APIRouter()

_ASSET_EXTS = ('.mp4', '.png', '.wav', '.webm', '.mov', '.mp3', '.ogg', '.flac')
_SEED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_SEED_MAX_BYTES = 25 * 1024 * 1024  # 25MB per file

# --- Evidence gallery -------------------------------------------------------
# One canonical example per config permutation lives under evidence/{image,
# video,music,tts}/ (see docs/asset-evidence-matrix.md). On startup we symlink
# each into EXP_DIR as `evidence_<cat>_<name>` so the existing gallery scan +
# file-serving surface them with no other changes. Idempotent, best-effort, and
# a no-op when the evidence dir doesn't exist ("...load on startup, if they exist").
EVIDENCE_DIR = os.environ.get("SLOPFINITY_EVIDENCE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evidence",
)


def seed_evidence_into_gallery() -> int:
    """Symlink evidence/{cat}/* into EXP_DIR as evidence_<cat>_<name>. Returns
    the number of new links created. Safe to call repeatedly."""
    linked = 0
    try:
        if not os.path.isdir(EVIDENCE_DIR):
            return 0
        for cat in ("image", "video", "music", "tts"):
            d = os.path.join(EVIDENCE_DIR, cat)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.lower().endswith(_ASSET_EXTS):
                    continue
                link = os.path.join(EXP_DIR, f"evidence_{cat}_{fn}")
                if os.path.lexists(link):
                    continue
                try:
                    os.symlink(os.path.join(d, fn), link)
                    linked += 1
                except OSError:
                    pass
    except Exception:
        pass
    return linked


# Run once at import (server startup imports this router).
seed_evidence_into_gallery()

def _kind_of(f: str) -> str:
    """Classify an asset filename into 'video', 'audio', or 'image'."""
    fl = f.lower()
    if fl.endswith(('.mp4', '.webm', '.mov')):
        return 'video'
    if fl.endswith(('.wav', '.mp3', '.ogg', '.flac')):
        return 'audio'
    return 'image'

def _list_outputs():
    """Return four lists sorted newest-first:
        finals  — FINAL_*.mp4 (curated keepers → Completed Gallery)
        live    — everything else (chain mp4s, base pngs, bridges, test images)
                  mixed and sorted by mtime → Live Gallery
        legacy_pngs — all pngs (for back-compat templates that still branch on imgs)
        mixed   — finals + live interleaved by mtime
    """
    try:
        entries = [
            (f, os.path.getmtime(os.path.join(EXP_DIR, f)))
            for f in os.listdir(EXP_DIR)
            if f.endswith('.mp4') or f.endswith('.png')
        ]
    except Exception:
        entries = []
    entries.sort(key=lambda x: x[1], reverse=True)
    finals = [f for f, _ in entries if f.endswith('.mp4') and f.startswith('FINAL_')]
    live = [f for f, _ in entries if not (f.endswith('.mp4') and f.startswith('FINAL_'))]
    legacy_pngs = [f for f, _ in entries if f.endswith('.png')]
    mixed = [f for f, _ in entries]
    return finals, live, legacy_pngs, mixed

@router.get("/assets")
async def assets(offset: int = 0, limit: int = 48):
    try:
        names = [
            f for f in os.listdir(EXP_DIR)
            if f.lower().endswith(_ASSET_EXTS)
        ]
    except OSError:
        names = []
    pairs = []
    for f in names:
        try:
            pairs.append((f, os.path.getmtime(os.path.join(EXP_DIR, f))))
        except OSError:
            continue
    pairs.sort(key=lambda x: x[1], reverse=True)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 256))
    page = pairs[offset:offset + limit]
    return {
        "items": [
            {"file": f, "mtime": ts, "kind": _kind_of(f)}
            for f, ts in page
        ],
        "offset": offset,
        "limit": limit,
        "total": len(pairs),
        "has_more": offset + limit < len(pairs),
    }

@router.get("/assets/by-vidx")
async def assets_by_vidx(v_idx: int):
    try:
        files = os.listdir(EXP_DIR)
    except OSError:
        files = []
    result: dict = {}
    legacy_prefix = f"v{v_idx}_"
    current_prefix = f"slop_{v_idx}_"
    prefix = current_prefix
    best_mtime: dict[str, float] = {}

    def _consider(role: str, name: str) -> None:
        try:
            mt = os.path.getmtime(os.path.join(EXP_DIR, name))
        except OSError:
            return
        if role not in best_mtime or mt > best_mtime[role]:
            best_mtime[role] = mt
            result[role] = name

    bridge_re = re.compile(rf"^(?:slop_{v_idx}|v{v_idx})(?:_.+)?_f(\d+)\.png$")

    for f in files:
        if f.startswith(prefix) or f.startswith(legacy_prefix):
            mb = bridge_re.match(f)
            if mb:
                idx = int(mb.group(1))
                bridges = result.setdefault("bridges", {})
                try:
                    mt = os.path.getmtime(os.path.join(EXP_DIR, f))
                except OSError:
                    mt = 0
                prev = bridges.get(idx)
                if prev is None:
                    bridges[idx] = f
                else:
                    try:
                        prev_mt = os.path.getmtime(os.path.join(EXP_DIR, prev))
                    except OSError:
                        prev_mt = 0
                    if mt > prev_mt:
                        bridges[idx] = f
                continue
            if f.endswith("_base.png"):
                _consider("base", f)
            elif f.endswith(".mp4"):
                _consider("video", f)
            elif f.endswith(".wav"):
                if "tts" in f.lower():
                    _consider("tts", f)
                else:
                    _consider("audio", f)
        if f == f"FINAL_{v_idx}.mp4" or f.startswith(f"FINAL_{v_idx}.") or f.startswith(f"FINAL_{v_idx}_"):
            if f.endswith(".mp4"):
                _consider("final", f)
    return {"v_idx": v_idx, "assets": result}

@router.get("/asset/{filename}")
async def asset_info(filename: str):
    if "/" in filename or ".." in filename or filename.startswith("."):
        return JSONResponse({"ok": False, "error": "invalid filename"}, status_code=400)
    path = os.path.join(EXP_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    st = os.stat(path)
    if filename.endswith(".mp4"):
        kind = "final" if filename.startswith("FINAL_") else "chain"
    elif filename.endswith(".wav"):
        kind = "audio"
    elif filename.endswith(".png") or filename.endswith(".jpg"):
        kind = "image"
    else:
        kind = "other"
    model = None
    m = re.match(r"^test_([A-Za-z0-9.-]+)_", filename)
    if m:
        model = m.group(1)
    elif filename.startswith("ltx_base_") or filename.endswith(".mp4"):
        model = "ltx-2.3"
    elif re.match(r"^v\d+_f\d+\.png$", filename):
        model = "ltx-bridge"
    prompt = None
    sidecar = os.path.join(EXP_DIR, filename + ".json")
    if os.path.isfile(sidecar):
        try:
            with open(sidecar) as f:
                prompt = json.load(f).get("prompt")
        except Exception:
            pass
    if not prompt:
        state = cfg.get_state()
        if state.get("ts") and abs(st.st_mtime - state["ts"]) < 60:
            prompt = state.get("current_prompt")
    return {
        "ok": True,
        "filename": filename,
        "kind": kind,
        "model": model,
        "size_bytes": st.st_size,
        "mtime": st.st_mtime,
        "prompt": prompt,
        "url": f"/files/{filename}",
    }

@router.get("/asset/components/{filename}")
async def asset_components(filename: str):
    if "/" in filename or ".." in filename or filename.startswith("."):
        return JSONResponse({"ok": False, "error": "invalid filename"}, status_code=400)
    if not filename.startswith("FINAL_") or not filename.endswith(".mp4"):
        return JSONResponse({"ok": False, "error": "components only apply to FINAL_*.mp4"}, status_code=400)
    m = re.match(r"^FINAL_(\d+)_(.+?)(?:_audio)?\.mp4$", filename)
    if not m:
        return JSONResponse({"ok": False, "error": "could not parse v_idx/slug from filename"}, status_code=400)
    v_idx = int(m.group(1))
    slug = m.group(2)

    def _component_row(name: str) -> dict:
        p = os.path.join(EXP_DIR, name)
        try:
            s = os.stat(p)
            size = s.st_size
            mt = s.st_mtime
        except OSError:
            size, mt = 0, 0.0
        side = {}
        sidecar = os.path.join(EXP_DIR, name + ".json")
        if os.path.isfile(sidecar):
            try:
                with open(sidecar) as f: side = json.load(f) or {}
            except Exception: pass
        if name.endswith(".wav"): inferred_kind = "audio"
        elif name.endswith(".png") or name.endswith(".jpg"): inferred_kind = "image"
        elif re.search(r"_c\d+\.mp4$", name): inferred_kind = "chain"
        elif name.endswith(".mp4"): inferred_kind = "video"
        else: inferred_kind = "other"
        part = side.get("part")
        if part is None:
            mc = re.search(r"_c(\d+)\.mp4$", name)
            if mc: part = int(mc.group(1))
        return {
            "file": name, "url": f"/files/{name}", "kind": side.get("kind") or inferred_kind,
            "model": side.get("model"), "prompt": side.get("prompt"), "part": part,
            "size_bytes": size, "mtime": mt,
        }

    components: list = []
    seen: set = set()
    prefix = f"slop_{v_idx}_"
    candidates = []
    for p in glob.glob(os.path.join(EXP_DIR, f"{prefix}*")):
        nm = os.path.basename(p)
        if nm.endswith(".json") or not nm.startswith(f"{prefix}{slug}") or re.search(r"_f\d+\.png$", nm):
            continue
        candidates.append(nm)

    def _sort_key(nm: str):
        mc = re.search(r"_c(\d+)\.mp4$", nm)
        if mc: return (0, int(mc.group(1)))
        if nm.endswith("_base.png"): return (1, 0)
        if nm.endswith(".wav"): return (2, nm)
        return (3, nm)

    for nm in sorted(candidates, key=_sort_key):
        if nm not in seen:
            components.append(_component_row(nm))
            seen.add(nm)
    return {"ok": True, "filename": filename, "v_idx": v_idx, "components": components}

@router.get("/seeds/list")
async def seeds_list():
    items = []
    try:
        for f in os.listdir(EXP_DIR):
            if f.startswith("seed_") and f.lower().endswith(_SEED_IMAGE_EXTS):
                try:
                    items.append({"file": f, "mtime": os.path.getmtime(os.path.join(EXP_DIR, f))})
                except OSError: pass
    except OSError: pass
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"items": items}

@router.post("/upload")
async def upload_seed_assets(files: list[UploadFile] = File(...)):
    saved, skipped = [], []
    ts = int(time.time())
    for idx, uf in enumerate(files or []):
        original = (uf.filename or "upload").strip()
        ext = os.path.splitext(original)[1].lower()
        if ext not in _SEED_IMAGE_EXTS:
            skipped.append({"name": original, "reason": "non-image extension"})
            continue
        stem = os.path.splitext(os.path.basename(original))[0] or "upload"
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:64].strip("_") or "upload"
        out_name = f"seed_{ts}_{idx:02d}_{slug}{ext}"
        out_path = os.path.join(EXP_DIR, out_name)
        size = 0
        oversize = False
        try:
            with open(out_path, "wb") as fh:
                while True:
                    chunk = await uf.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > _SEED_MAX_BYTES:
                        oversize = True
                        break
                    fh.write(chunk)
            if oversize:
                try: os.remove(out_path)
                except OSError: pass
                skipped.append({"name": original, "reason": "exceeds 25MB cap"})
            else:
                saved.append(out_name)
        except Exception as exc:
            skipped.append({"name": original, "reason": str(exc)})
    return {"ok": True, "saved": saved, "skipped": skipped}

@router.get("/vae_grid")
async def vae_grid_check(file: str):
    if not file or ".." in file or file.startswith("/"):
        return JSONResponse({"ok": False, "error": "bad_path"}, status_code=400)
    abs_path = os.path.join(EXP_DIR, file)
    if not os.path.isfile(abs_path):
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    from slopfinity import vae_grid as _vg
    cached = _vg.read_sidecar(abs_path)
    if cached: return {"ok": True, "cached": True, **cached}
    result = await asyncio.to_thread(_vg.detect_grid, abs_path)
    _vg.write_sidecar(abs_path, result)
    return {"ok": True, "cached": False, **result}

@router.get("/outputs")
async def outputs():
    return get_output_counts(EXP_DIR)
