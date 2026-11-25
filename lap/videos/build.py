#!/usr/bin/env python3
"""Build Sorti's educational video library from Pixabay (free to use, no attribution required;
we still credit 'Pixabay / <author>' on screen).

    PY="env -u PYTHONPATH $HOME/ps2_et/bin/python"
    $PY build.py search            # candidates.json + contact sheets per class (for review)
    # reviewers write <class>/manifest.json: chosen clips + voice-over text in en/hi/ta
    $PY build.py build [class...]  # download 720p, synthesize voice-over, mux -> <class>/<id>_<lang>.mp4
    $PY build.py index             # videos/index.json used by the app

Layout:  videos/<class>/raw/<id>.mp4 (source)  sheets/<id>.jpg (3-frame review sheet)
         videos/<class>/<id>.mp4 (silent 1280x720 H.264, looped muted by the browser)
         videos/<class>/<id>_{en,hi,ta}.wav (voice-over, played by the server on the speaker)  <id>.jpg (poster)
"""
import json, os, subprocess, sys, time, urllib.parse, urllib.request, wave

HERE = os.path.dirname(os.path.abspath(__file__))
LAP = os.path.dirname(HERE)
ROOT = os.path.dirname(LAP)
VOICE = os.path.join(ROOT, "voice")
KEY = open(os.path.join(LAP, ".pixabay_key")).read().strip()

CLASSES = ["battery_waste", "clothes", "e_waste", "glass_waste", "light_bulbs",
           "metal_waste", "organic_waste", "paper_waste", "plastic_waste"]
QUERIES = {
    "battery_waste": ["batteries", "battery recycling", "used batteries", "battery", "aa battery"],
    "clothes": ["textile recycling", "second hand clothes", "old clothes", "clothes donation", "fabric", "folding clothes"],
    "e_waste": ["electronic waste", "e-waste", "circuit board", "old computer", "recycling electronics", "broken phone"],
    "glass_waste": ["glass recycling", "glass bottles", "broken glass", "glass bottle", "glass jar"],
    "light_bulbs": ["light bulb", "fluorescent lamp", "led bulb", "bulb", "tube light"],
    "metal_waste": ["aluminum cans", "scrap metal", "metal recycling", "tin cans", "cans recycling"],
    "organic_waste": ["compost", "food waste", "composting", "vegetable peels", "organic waste", "fruit peel"],
    "paper_waste": ["paper recycling", "cardboard", "newspaper", "waste paper", "cardboard boxes"],
    "plastic_waste": ["plastic bottles", "plastic pollution", "plastic waste", "plastic recycling", "plastic bag"],
}
EVERYDAY = {   # second pass: the household items visitors actually bring (user priority)
    "battery_waste": ["aa batteries", "button cell", "power bank"],
    "clothes": ["donate clothes", "jeans", "saree", "laundry clothes pile"],
    "e_waste": ["mobile phone", "phone charger", "cables", "earphones", "remote control", "broken laptop"],
    "glass_waste": ["glass jar", "empty bottles", "beer bottles"],
    "light_bulbs": ["cfl", "changing light bulb", "light bulbs"],
    "metal_waste": ["soda can", "crushed cans", "aluminium foil", "tin can"],
    "organic_waste": ["vegetable peeling", "kitchen waste", "leftover food", "banana peel", "tea leaves", "compost bin"],
    "paper_waste": ["newspapers", "delivery box", "cardboard box", "paper bag", "notebook"],
    "plastic_waste": ["water bottle", "plastic bags", "food packaging", "takeaway container", "plastic cup", "crushing plastic bottle"],
}
BAD_TAGS = {"animation", "animated", "3d", "render", "rendering", "cgi", "cartoon", "ai", "ai generated",
            "generative", "illustration", "loop", "motion graphics", "background", "abstract", "particles"}
PY = sys.executable


def api(q, page=1):
    url = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode(
        dict(key=KEY, q=q, safesearch="true", per_page=50, page=page, order="popular"))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
                return json.load(r)
        except Exception as e:  # 429 rate limit / transient
            print("  retry", q, e); time.sleep(3 + 5 * attempt)
    return {"hits": []}


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return path
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 16)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)
    return path


def sh(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------ stage 1: search + review sheets
def search(per_class=24, classes=None, queries=None, merge=False):
    for cls in classes or CLASSES:
        d = os.path.join(HERE, cls)
        os.makedirs(os.path.join(d, "sheets"), exist_ok=True)
        os.makedirs(os.path.join(d, "raw"), exist_ok=True)
        old = json.load(open(os.path.join(d, "candidates.json"))) if merge and os.path.exists(os.path.join(d, "candidates.json")) else []
        seen, cands = {c["id"] for c in old}, []
        for q in (queries or QUERIES)[cls]:
            for rank, h in enumerate(api(q)["hits"]):
                tags = {t.strip().lower() for t in h["tags"].split(",")}
                if h["id"] in seen or tags & BAD_TAGS or not (8 <= h["duration"] <= 60):
                    continue
                v = h["videos"]
                hd = next((v[k] for k in ("medium", "small", "large") if v.get(k, {}).get("url") and v[k]["height"] >= 720), None)
                if not hd or not v.get("tiny", {}).get("url"):
                    continue
                seen.add(h["id"])
                cands.append(dict(id=h["id"], query=q, rank=rank, tags=h["tags"], duration=h["duration"],
                                  page=h["pageURL"], user=h["user"], downloads=h["downloads"], likes=h["likes"],
                                  hd_url=hd["url"], hd_size=[hd["width"], hd["height"]], tiny_url=v["tiny"]["url"]))
            time.sleep(0.4)
        # relevance (search rank) first, popularity as tie-break
        cands.sort(key=lambda c: (c["rank"], -c["downloads"]))
        cands = cands[:per_class]
        for c in cands:
            c["pass"] = "everyday" if merge else "first"
        def make_sheet(c):
            sheet = os.path.join(d, "sheets", f"{c['id']}.jpg")
            if os.path.exists(sheet):
                return
            tiny = os.path.join(d, "raw", f"{c['id']}_tiny.mp4")
            try:
                fetch(c["tiny_url"], tiny)
                dur = max(c["duration"], 3)
                # 3 frames spread over the clip, side by side
                sh("ffmpeg", "-y", "-loglevel", "error", "-i", tiny, "-vf",
                   f"fps=3/{dur},scale=426:-2,tile=3x1:padding=4:color=black", "-frames:v", "1", "-q:v", "4", sheet)
            except Exception as e:
                print("  sheet failed", c["id"], e)
            finally:
                if os.path.exists(tiny):
                    os.remove(tiny)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(make_sheet, cands))
        json.dump(old + cands, open(os.path.join(d, "candidates.json"), "w"), indent=1, ensure_ascii=False)
        print(f"{cls}: {len(cands)} candidates")


# ------------------------------------------------------------------ stage 2: voice-over + mux
def tts_voices():
    from piper import PiperVoice
    return {"en": PiperVoice.load(os.path.join(VOICE, "tts_piper_en/en_US-lessac-medium.onnx")),
            "hi": PiperVoice.load(os.path.join(VOICE, "tts_piper_hi/hi_IN-pratham-medium.onnx")),
            "ta": PiperVoice.load(os.path.join(VOICE, "tts_piper_ta_female/ta_IN-rasa_female-medium.onnx"))}


def synth(voice, text, path, lang):
    import numpy as np
    sys.path.insert(0, LAP)
    from tts_utils import tighten
    with wave.open(path, "wb") as w:
        voice.synthesize_wav(text, w)
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        y = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    y = tighten(y, sr, word_gap_ms=110, sentence_gap_ms=320) if lang == "ta" else tighten(y, sr)
    y = (np.clip(y / max(1e-6, np.abs(y).max()) * 0.9, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes())
    return len(y) / sr


def build(classes):
    voices = tts_voices()
    for cls in classes:
        d = os.path.join(HERE, cls)
        mf = os.path.join(d, "manifest.json")
        if not os.path.exists(mf):
            print(f"{cls}: no manifest.json yet, skipped"); continue
        man = json.load(open(mf))
        for clip in man["clips"]:
            cid = clip["id"]
            raw = fetch(clip["hd_url"], os.path.join(d, "raw", f"{cid}.mp4"))
            poster = os.path.join(d, f"{cid}.jpg")
            if not os.path.exists(poster):
                sh("ffmpeg", "-y", "-loglevel", "error", "-ss", str(min(2, clip["duration"] / 3)), "-i", raw,
                   "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720", "-frames:v", "1", "-q:v", "4", poster)
            out = os.path.join(d, f"{cid}.mp4")       # silent 720p30 clip; the browser loops it, muted
            if not os.path.exists(out):
                sh("ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-t", "30", "-an", "-vf",
                   "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=30",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", out)
            for lang in ("en", "hi", "ta"):                 # voice-over, played by the server on the speaker
                wav = os.path.join(d, f"{cid}_{lang}.wav")
                if os.path.exists(wav) and os.path.getmtime(wav) > os.path.getmtime(mf):
                    continue
                secs = synth(voices[lang], clip["vo"][lang], wav, lang)
                print(f"  {cls}/{cid}_{lang}.wav  {secs:.1f}s")
    index()


def index():
    idx = {}
    for cls in CLASSES:
        mf = os.path.join(HERE, cls, "manifest.json")
        if not os.path.exists(mf):
            continue
        clips = []
        for c in json.load(open(mf))["clips"]:
            vo_files = {l: f"{cls}/{c['id']}_{l}.wav" for l in ("en", "hi", "ta")
                        if os.path.exists(os.path.join(HERE, cls, f"{c['id']}_{l}.wav"))}
            if len(vo_files) == 3 and os.path.exists(os.path.join(HERE, cls, f"{c['id']}.mp4")):
                clips.append(dict(id=c["id"], title=c.get("title", {}), video=f"{cls}/{c['id']}.mp4",
                                  vo_files=vo_files, poster=f"{cls}/{c['id']}.jpg",
                                  credit=f"Pixabay / {c['user']}", page=c["page"], vo=c["vo"]))
        idx[cls] = clips
    json.dump(idx, open(os.path.join(HERE, "index.json"), "w"), indent=1, ensure_ascii=False)
    print("index:", {k: len(v) for k, v in idx.items()})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "search"
    if cmd == "search":
        search(classes=sys.argv[2:] or None)
    elif cmd == "everyday":                       # add item-specific candidates (sheets too) to candidates.json
        search(per_class=18, classes=sys.argv[2:] or CLASSES, queries=EVERYDAY, merge=True)
    elif cmd == "build":
        build(sys.argv[2:] or CLASSES)
    elif cmd == "index":
        index()
