import os
import json
import time
import re
import glob

from fastapi import APIRouter
from fastapi.responses import JSONResponse, FileResponse

from slopfinity.paths import EXP_DIR
import slopfinity.config as cfg
from slopfinity.stats import get_output_counts


router = APIRouter()

_ASSET_EXTS = ('.mp4', '.png', '.wav', '.webm', '.mov', '.mp3', '.ogg', '.flac')

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
        mixed   — finals + live interleaved by mtime so a FINAL's component
                  pieces (its base.png, chain mp4s) sit right next to it in
                  the gallery instead of being banished below all finals.
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
    """Return assets ordered by mtime desc, paginated.

    Used by the client-side infinite-scroll loader on the slop view to fetch
    older content as the user scrolls toward the bottom of the lower pane.
    The initial 64 cards are server-side rendered by `index()`; this endpoint
    serves offset >= 64 typically.

    EXP_DIR may mutate mid-request (the fleet writes new files every few
    minutes); we tolerate that by guarding os.listdir / getmtime with
    try/except so a vanished file at stat-time doesn't 500 the page.
    """
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
            # File vanished between listdir and stat; skip it.
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
    """Resolve actual on-disk filenames for a given video index.

    The fleet runner uses slug-based filenames
    (e.g. ``slop_1_sterile_chrome_corridors_algorithms_shep_base.png``)
    rather than the legacy ``v{N}_base.png`` shape that the dashboard
    used to synthesize. This endpoint maps a v_idx to whatever real
    filenames currently exist on disk so the client can build correct
    `/files/<name>` links instead of guessing — the previous synthesis
    would 404 against fresh slugged outputs, or worse, match a stale
    file from a previous run that happens to still be on disk under
    the old un-slugged name. Both the legacy ``v<idx>_`` and current
    ``slop_<idx>_`` prefixes are matched so historic outputs keep
    showing in the slop feed after the rename landed.
    """
    try:
        files = os.listdir(EXP_DIR)
    except OSError:
        files = []
    result: dict = {}
    legacy_prefix = f"v{v_idx}_"
    current_prefix = f"slop_{v_idx}_"
    prefix = current_prefix  # primary; legacy_prefix tested as fallback below
    # Track newest mtime per role so we prefer the most recent file when
    # the directory contains multiple matches (e.g. several video chains
    # for the same v_idx — keep the latest one for the `video` slot).
    best_mtime: dict[str, float] = {}

    def _consider(role: str, name: str) -> None:
        try:
            mt = os.path.getmtime(os.path.join(EXP_DIR, name))
        except OSError:
            return
        if role not in best_mtime or mt > best_mtime[role]:
            best_mtime[role] = mt
            result[role] = name

    # ffmpeg bridge frames: slop_{N}_<slug>_f{M}.png (or legacy
    # v{N}_<slug>_f{M}.png). Surfaced as a "bridges" {idx: filename}
    # sub-map so the dashboard can render the per-chain last-frame
    # extracts inline. The regex matches both prefixes for back-compat.
    bridge_re = re.compile(rf"^(?:slop_{v_idx}|v{v_idx})(?:_.+)?_f(\d+)\.png$")

    for f in files:
        if f.startswith(prefix) or f.startswith(legacy_prefix):
            mb = bridge_re.match(f)
            if mb:
                idx = int(mb.group(1))
                bridges = result.setdefault("bridges", {})
                # Prefer the slugged form if both forms ever exist (matches
                # whatever the runner currently writes); otherwise newest mtime.
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
                # Chain segments: v{N}_c{M}.mp4 (also covers slugged
                # variants like v{N}_<slug>_c{M}.mp4 if they appear).
                _consider("video", f)
            elif f.endswith(".wav"):
                # Heuristic: TTS lines often live alongside chain audio.
                if "tts" in f.lower():
                    _consider("tts", f)
                else:
                    _consider("audio", f)
        # Final merge has its own naming convention (FINAL_{N}*.mp4)
        # and isn't prefixed with v{N}_.
        if f == f"FINAL_{v_idx}.mp4":
            _consider("final", f)
        elif f.startswith(f"FINAL_{v_idx}.") and f.endswith(".mp4"):
            _consider("final", f)
        elif f.startswith(f"FINAL_{v_idx}_") and f.endswith(".mp4"):
            _consider("final", f)
    return {"v_idx": v_idx, "assets": result}

@router.get("/asset/{filename}")
async def asset_info(filename: str):
    """Return metadata about a single asset file: kind, model, size, mtime,
    and best-effort prompt (if a sidecar .json exists or state matches).

    Filename should be the leaf name only (no path), living under EXP_DIR.
    """
    import re
    # basic name safety — no path traversal
    if "/" in filename or ".." in filename or filename.startswith("."):
        return JSONResponse({"ok": False, "error": "invalid filename"}, status_code=400)
    path = os.path.join(EXP_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    st = os.stat(path)
    # kind
    if filename.endswith(".mp4"):
        kind = "final" if filename.startswith("FINAL_") else "chain"
    elif filename.endswith(".wav"):
        kind = "audio"
    elif filename.endswith(".png") or filename.endswith(".jpg"):
        kind = "image"
    else:
        kind = "other"
    # model (mirror template / app.js logic)
    model = None
    m = re.match(r"^test_([A-Za-z0-9.-]+)_", filename)
    if m:
        model = m.group(1)
    elif filename.startswith("ltx_base_") or filename.endswith(".mp4"):
        model = "ltx-2.3"
    elif re.match(r"^v\d+_f\d+\.png$", filename):
        model = "ltx-bridge"
    # best-effort prompt — look for a sibling JSON sidecar OR the running state.json
    prompt = None
    sidecar = os.path.join(EXP_DIR, filename + ".json")
    if os.path.isfile(sidecar):
        try:
            with open(sidecar) as f:
                prompt = json.load(f).get("prompt")
        except Exception:
            pass
    if not prompt:
        # fallback: if this file's mtime is within 60 s of the current state, use current prompt
        state = cfg.get_state()
        if state.get("ts") and abs(st.st_mtime - state["ts"]) < 60:
            prompt = state.get("current_prompt")
    return {
        "ok": True,
        "filename": filename,
        "kind": kind,
        "model": model,
        "size_bytes": st.st_size,
        "size_human": (
            f"{st.st_size/1e9:.2f} GB" if st.st_size > 1e9
            else f"{st.st_size/1e6:.1f} MB" if st.st_size > 1e6
            else f"{st.st_size/1e3:.0f} KB"
        ),
        "mtime": st.st_mtime,
        "mtime_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "age_seconds": int(time.time() - st.st_mtime),
        "prompt": prompt,
        "url": f"/files/{filename}",
    }

@router.get("/asset/components/{filename}")
async def asset_components(filename: str):
    """For a FINAL_*.mp4, list the source components that were concatenated to
    produce it (chain mp4s, base png, music wav, optional tts wav).

    Lookup strategy:
      1. Pattern-match `FINAL_<v_idx>_<slug>.mp4` (and `_audio` variant).
      2. Prefer the concat list `_concat_<v_idx>.txt` if still on disk
         (run_fleet removes it post-mux, so usually missing).
      3. Otherwise glob `slop_<v_idx>_*` to reconstruct components by sidecar
         + filename pattern (`_c\\d+.mp4`, `_base.png`, `.wav`).

    Each component row includes filename, kind, model, part/of (if known),
    size + mtime, and a `/files/<name>` URL for direct linking. Sidecar
    fields are merged in best-effort.

    Returns `{ok: True, v_idx: int, components: [...]}` on success.
    Graceful: missing sidecars or missing concat list are non-fatal — the
    endpoint returns whatever components it could reconstruct.
    """
    import re
    import glob
    if "/" in filename or ".." in filename or filename.startswith("."):
        return JSONResponse({"ok": False, "error": "invalid filename"}, status_code=400)
    if not filename.startswith("FINAL_") or not filename.endswith(".mp4"):
        return JSONResponse(
            {"ok": False, "error": "components only apply to FINAL_*.mp4"},
            status_code=400,
        )
    m = re.match(r"^FINAL_(\d+)_(.+?)(?:_audio)?\.mp4$", filename)
    if not m:
        return JSONResponse(
            {"ok": False, "error": "could not parse v_idx/slug from filename"},
            status_code=400,
        )
    v_idx = int(m.group(1))
    slug = m.group(2)

    def _component_row(name: str) -> dict:
        """Build a single component descriptor from a leaf filename."""
        p = os.path.join(EXP_DIR, name)
        try:
            s = os.stat(p)
            size = s.st_size
            mt = s.st_mtime
        except OSError:
            size = 0
            mt = 0.0
        # Best-effort sidecar merge — sidecars carry kind/model/part/of plus
        # FLF2V/cont fields like kf_start/kf_end/handoff_k.
        side = {}
        sidecar = os.path.join(EXP_DIR, name + ".json")
        if os.path.isfile(sidecar):
            try:
                with open(sidecar) as f:
                    side = json.load(f) or {}
            except Exception:
                side = {}
        # Derive kind from filename if sidecar didn't say.
        if name.endswith(".wav"):
            inferred_kind = "audio"
        elif name.endswith(".png") or name.endswith(".jpg"):
            inferred_kind = "image"
        elif re.search(r"_c\d+\.mp4$", name):
            inferred_kind = "chain"
        elif name.endswith(".mp4"):
            inferred_kind = "video"
        else:
            inferred_kind = "other"
        # Pull part index from filename when sidecar omitted it.
        part = side.get("part")
        if part is None:
            mc = re.search(r"_c(\d+)\.mp4$", name)
            if mc:
                part = int(mc.group(1))
        return {
            "file": name,
            "url": f"/files/{name}",
            "kind": side.get("kind") or inferred_kind,
            "model": side.get("model"),
            "prompt": side.get("prompt"),
            "part": part,
            "of": side.get("of"),
            "kf_start": side.get("kf_start"),
            "kf_end": side.get("kf_end"),
            "handoff_k": side.get("handoff_k"),
            "size_bytes": size,
            "mtime": mt,
        }

    components: list = []
    seen: set = set()

    # 1. Prefer concat list if still on disk (rare — run_fleet rm's it).
    concat_path = os.path.join(EXP_DIR, f"_concat_{v_idx}.txt")
    if os.path.isfile(concat_path):
        try:
            with open(concat_path) as f:
                for line in f:
                    line = line.strip()
                    cm = re.match(r"^file\s+'(.+)'\s*$", line)
                    if not cm:
                        continue
                    nm = os.path.basename(cm.group(1))
                    if nm in seen:
                        continue
                    if not os.path.isfile(os.path.join(EXP_DIR, nm)):
                        continue
                    components.append(_component_row(nm))
                    seen.add(nm)
        except Exception:
            pass

    # 2. Glob fallback — picks up chain mp4s in numeric order, plus base/wav.
    prefix = f"slop_{v_idx}_"
    candidates = []
    for p in glob.glob(os.path.join(EXP_DIR, f"{prefix}*")):
        nm = os.path.basename(p)
        if nm.endswith(".json"):
            continue  # sidecars are handled inline
        # Only count this component if it shares the slug — guards against
        # accidental v_idx collisions across different prompts.
        if not nm.startswith(f"{prefix}{slug}"):
            continue
        # Skip handoff/bridge frames `_f\d+.png` — they're FLF2V intermediates
        # consumed during chain assembly, not part of the final concat.
        if re.search(r"_f\d+\.png$", nm):
            continue
        candidates.append(nm)

    # Order: chain segments first (by part), then base.png, then any wavs.
    def _sort_key(nm: str):
        mc = re.search(r"_c(\d+)\.mp4$", nm)
        if mc:
            return (0, int(mc.group(1)))
        if nm.endswith("_base.png"):
            return (1, 0)
        if nm.endswith(".wav"):
            return (2, nm)
        return (3, nm)

    for nm in sorted(candidates, key=_sort_key):
        if nm in seen:
            continue
        components.append(_component_row(nm))
        seen.add(nm)

    return {
        "ok": True,
        "filename": filename,
        "v_idx": v_idx,
        "slug": slug,
        "concat_list_present": os.path.isfile(concat_path),
        "components": components,
    }

@router.get("/outputs")
async def outputs():
    """Return counters for produced artifacts: final mp4s, chain clips, base images."""
    return get_output_counts(EXP_DIR)
