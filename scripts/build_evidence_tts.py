#!/usr/bin/env python3
"""Fill the TTS evidence matrix: one canonical sample per (engine, voice).

Idempotent / dedup-aware — skips any evidence/tts/{engine}_{voice}.wav that
already exists, so re-runs only generate what's missing. Ordered fastest-first
(kokoro → qwen → dramabox). Writes evidence/tts/manifest.json and prints rolling
progress so a watcher can report per-asset.
"""
import json, os, shutil, time, urllib.request, urllib.error, wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVID = os.path.join(ROOT, "evidence", "tts")
os.makedirs(EVID, exist_ok=True)
SERVER_TTS = "/home/matthewh/amd-strix-halo-image-video-toolboxes/comfy-outputs/experiments/tts"
TEXT = "Every voice tells a story, and this one is a sample."

VOICES = {
    "kokoro": ["af_alloy","af_aoede","af_bella","af_heart","af_jessica","af_kore","af_nicole","af_nova","af_river","af_sarah","af_sky","am_adam","am_echo","am_eric","am_fenrir","am_liam","am_michael","am_onyx","am_puck","am_santa","bf_alice","bf_emma","bf_isabella","bf_lily","bm_daniel","bm_fable","bm_george","bm_lewis","ef_dora","em_alex","em_santa","ff_siwis","hf_alpha","hf_beta","hm_omega","hm_psi","if_sara","im_nicola","jf_alpha","jf_gongitsune","jf_nezumi","jf_tebukuro","jm_kumo","pf_dora","pm_alex","pm_santa","zf_xiaobei","zf_xiaoni","zf_xiaoxiao","zf_xiaoyi","zm_yunjian","zm_yunxi","zm_yunxia","zm_yunyang"],
    "qwen": ["aiden","dylan","eric","ono_anna","ryan","serena","sohee","uncle_fu","vivian"],
    "dramabox": ["kid","narrator-female","narrator-male"],
}

def synth(engine, voice):
    payload = json.dumps({"text": TEXT, "voice": voice, "engine": engine}).encode()
    req = urllib.request.Request("http://localhost:8010/tts", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        d = json.loads(e.read().decode())
    return d, int(time.time() - t0)

def wav_meta(path):
    try:
        with wave.open(path) as w:
            return round(w.getnframes()/float(w.getframerate()), 1), w.getframerate()
    except Exception:
        return None, None

manifest = {}
total = sum(len(v) for v in VOICES.values())
done = skipped = failed = 0
print(f"[evidence-tts] target={total} permutations -> {EVID}", flush=True)
for engine in ("kokoro", "qwen", "dramabox"):
    for voice in VOICES[engine]:
        dst = os.path.join(EVID, f"{engine}_{voice}.wav")
        if os.path.isfile(dst):
            dur, sr = wav_meta(dst)
            manifest[f"{engine}_{voice}"] = {"engine": engine, "voice": voice, "dur_s": dur, "bytes": os.path.getsize(dst), "status": "have"}
            skipped += 1
            print(f"[skip] {engine}/{voice} (exists {dur}s)", flush=True)
            continue
        d, dt = synth(engine, voice)
        if not (d.get("ok") and d.get("url")):
            failed += 1
            print(f"[FAIL] {engine}/{voice} gen={dt}s: {str(d.get('error'))[:120]}", flush=True)
            manifest[f"{engine}_{voice}"] = {"engine": engine, "voice": voice, "status": "fail", "error": str(d.get("error"))[:160]}
            continue
        fname = d["url"].split("/files/tts/", 1)[-1]
        try:
            shutil.copyfile(os.path.join(SERVER_TTS, fname), dst)
        except Exception as e:
            failed += 1
            print(f"[COPYFAIL] {engine}/{voice}: {e}", flush=True); continue
        dur, sr = wav_meta(dst)
        sz = os.path.getsize(dst)
        done += 1
        manifest[f"{engine}_{voice}"] = {"engine": engine, "voice": voice, "dur_s": dur, "bytes": sz, "status": "new", "gen_s": dt}
        print(f"[ok] {engine}/{voice}  {dur}s  {sz/1024:.0f}KB  gen={dt}s  ({done} new, {skipped} had)", flush=True)
    print(f"[engine-done] {engine}: {sum(1 for k in manifest if manifest[k]['engine']==engine and manifest[k]['status'] in ('have','new'))}/{len(VOICES[engine])}", flush=True)

with open(os.path.join(EVID, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n[evidence-tts] COMPLETE new={done} had={skipped} failed={failed} total_ok={done+skipped}/{total}", flush=True)
