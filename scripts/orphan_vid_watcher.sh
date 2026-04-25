#!/usr/bin/env bash
# Polls comfy-outputs/ every 30 s for vid_<ts>_*.png batches that lack a
# matching vid_<ts>.mp4, encodes them with libopenh264 (the only H.264
# encoder the container's ffmpeg-free supports), and removes the frames.
# Acts as a back-stop for any orchestrator running pre-fix code that
# leaves orphan PNGs behind. Once the patched run_fleet.py is live the
# helper picks the right encoder via the ffmpeg auto-heal path; this
# watcher then has nothing to do and is harmless.
set -u
cd "$(dirname "$0")/.."
LOG=logs/orphan_vid_watcher.log
mkdir -p logs
echo "$(date -Is) watcher started" >> "$LOG"

encode_one() {
    local ts=$1
    local out="comfy-outputs/vid_${ts}.mp4"
    [ -f "$out" ] && return 0
    local count
    count=$(ls "comfy-outputs/vid_${ts}_"*.png 2>/dev/null | wc -l)
    [ "$count" -lt 1 ] && return 1
    echo "$(date -Is) encoding ts=$ts frames=$count" >> "$LOG"
    if docker exec strix-halo-comfyui ffmpeg -hide_banner -loglevel error -y \
        -framerate 24 -start_number 1 \
        -i "/opt/ComfyUI/output/vid_${ts}_%05d_.png" \
        -c:v libopenh264 -pix_fmt yuv420p -b:v 8M \
        "/opt/ComfyUI/output/vid_${ts}.mp4" >> "$LOG" 2>&1
    then
        if [ -f "$out" ]; then
            echo "$(date -Is) ok ts=$ts $(stat -c%s "$out") bytes" >> "$LOG"
            rm -f "comfy-outputs/vid_${ts}_"*.png
        else
            echo "$(date -Is) FAIL no output ts=$ts" >> "$LOG"
        fi
    else
        echo "$(date -Is) FAIL ffmpeg-rc=$? ts=$ts" >> "$LOG"
    fi
}

while true; do
    for ts in $(ls comfy-outputs/vid_*_*.png 2>/dev/null \
                | sed 's/.*vid_\([0-9]*\)_.*/\1/' | sort -u); do
        latest_mtime=$(stat -c%Y "comfy-outputs/vid_${ts}_"*.png 2>/dev/null \
                       | sort -n | tail -1)
        [ -z "$latest_mtime" ] && continue
        age=$(( $(date +%s) - latest_mtime ))
        [ "$age" -lt 60 ] && continue
        encode_one "$ts"
    done
    sleep 30
done
