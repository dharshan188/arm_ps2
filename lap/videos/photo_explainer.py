#!/usr/bin/env python3
"""Documentary-style explainer from REAL photos (Wikimedia Commons, free licences) + voice-over.

    env -u PYTHONPATH ~/ps2_et/bin/python photo_explainer.py specs/photo/plastic_waste.json [--preview]

Each beat shows 1-3 photos with a slow Ken Burns pan/zoom, cross-fades, a colour grade + vignette,
and minimal overlays (animated stat counters, 1-2-3 disposal steps, bin badge). Every frame carries
the photo credit (CC BY / BY-SA need attribution). Photos + credits.json live in photos/<cls>/.

Writes <cls>/<slug>.mp4 (silent, like the other videos) + <slug>_{en,hi,ta}.wav + <slug>.jpg poster.
--preview also writes <cls>/raw/<slug>_preview.mp4 with en/hi/ta as three audio tracks (for VLC review).
Not registered in index.json until `--register`.
"""
import json, math, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W, H, FPS = 1280, 720, 24
LANGS = ("en", "hi", "ta")
BEATS = ["intro", "harm", "who", "dispose", "prevent"]
XF = 0.7                                   # cross-fade seconds between shots
LATO = "/usr/share/fonts/truetype/lato/Lato-%s.ttf"
BIN_RGB = {"blue": (52, 128, 235), "green": (46, 170, 92), "red": (226, 72, 60), "ewaste": (140, 90, 200)}
ACCENT = (255, 196, 61)
_F = {}


def font(w, n):
    k = (w, n)
    if k not in _F:
        _F[k] = ImageFont.truetype(LATO % w, n)
    return _F[k]


def ease(x):
    x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ photos
def load_photo(path, scale=1.35):
    """cover-fit to scale*(W,H) so Ken Burns has room to move; light documentary grade."""
    im = Image.open(path).convert("RGB")
    tw, th = int(W * scale), int(H * scale)
    r = max(tw / im.width, th / im.height)
    im = im.resize((math.ceil(im.width * r), math.ceil(im.height * r)), Image.LANCZOS)
    x0, y0 = (im.width - tw) // 2, (im.height - th) // 2
    im = im.crop((x0, y0, x0 + tw, y0 + th))
    a = np.asarray(im).astype(np.float32) / 255
    a = np.clip((a - 0.5) * 1.08 + 0.5, 0, 1)                     # a touch of contrast
    g = a.mean(axis=2, keepdims=True); a = g + (a - g) * 0.88      # slightly desaturated
    a[..., 2] *= 1.03; a[..., 0] *= 0.98                           # cool shadows
    return np.clip(a * 255, 0, 255).astype(np.uint8)


def kenburns(arr, p, z, c):
    """p: 0..1 progress; z=(z0,z1) zoom (1 = full scaled photo); c=((cx0,cy0),(cx1,cy1)) centre in 0..1."""
    e = p * p * (3 - 2 * p) * 0.6 + p * 0.4
    zz = z[0] + (z[1] - z[0]) * e
    cx = c[0][0] + (c[1][0] - c[0][0]) * e; cy = c[0][1] + (c[1][1] - c[0][1]) * e
    hh, ww = arr.shape[:2]
    cw, ch = ww / zz, hh / zz
    x0 = min(max(cx * ww - cw / 2, 0), ww - cw); y0 = min(max(cy * hh - ch / 2, 0), hh - ch)
    im = Image.fromarray(arr).resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
    return np.asarray(im).astype(np.float32)


def make_vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    v = 1 - 0.38 * np.clip(d - 0.55, 0, 1) ** 1.4
    bottom = 1 - 0.55 * np.clip((yy - H * 0.55) / (H * 0.45), 0, 1) ** 1.6   # room for overlays
    return (v * bottom)[..., None]


# ------------------------------------------------------------------ overlays
def shadow_text(d, xy, txt, f, fill=(255, 255, 255), anchor="ls"):
    x, y = xy
    for dx, dy in ((2, 2), (1, 3), (3, 1)):
        d.text((x + dx, y + dy), txt, font=f, fill=(0, 0, 0, 120), anchor=anchor)
    d.text((x, y), txt, font=f, fill=fill, anchor=anchor)


def ov_stat(d, o, q):
    """big number counting up, unit beside it, accent bar on the left. q = 0..1 local progress."""
    a = ease(q / 0.12)                                  # fade/slide in
    if a <= 0:
        return
    v = o["value"] * ease(min(1, q / 0.45)) if o.get("count", True) else o["value"]
    txt = o.get("fmt", "{:,.0f}").format(v) if isinstance(o["value"], (int, float)) else str(o["value"])
    x, y = o.get("x", 80), o.get("y", 600)
    x += int((1 - a) * -30)
    al = int(255 * a)
    big = font("Black", o.get("size", 150))
    d.rectangle([x - 26, y - o.get("size", 150) * 0.72, x - 18, y + 6], fill=ACCENT + (al,))
    for dx, dy in ((3, 3),):
        d.text((x + dx, y + dy), txt, font=big, fill=(0, 0, 0, int(110 * a)), anchor="ls")
    d.text((x, y), txt, font=big, fill=(255, 255, 255, al), anchor="ls")
    if o.get("unit"):
        bw = d.textlength(txt, font=big)
        n = 44                                           # shrink long labels to fit the frame
        while n > 20 and x + bw + 18 + d.textlength(o["unit"], font=font("Bold", n)) > W - 40:
            n -= 2
        d.text((x + bw + 18, y - 12), o["unit"], font=font("Bold", n), fill=ACCENT + (al,), anchor="ls")


def bin_icon(d, cx, cy, s, color, al=255):
    w, h = 70 * s, 86 * s
    d.rounded_rectangle([cx - w / 2 - 6 * s, cy - h / 2 - 14 * s, cx + w / 2 + 6 * s, cy - h / 2], 4 * s, fill=color + (al,))
    d.rectangle([cx - 12 * s, cy - h / 2 - 22 * s, cx + 12 * s, cy - h / 2 - 14 * s], fill=color + (al,))
    d.polygon([(cx - w / 2, cy - h / 2 + 4 * s), (cx + w / 2, cy - h / 2 + 4 * s),
               (cx + w / 2 - 8 * s, cy + h / 2), (cx - w / 2 + 8 * s, cy + h / 2)], fill=color + (al,))
    for k in (-1, 0, 1):
        d.line([(cx + k * 18 * s, cy - h / 2 + 18 * s), (cx + k * 15 * s, cy + h / 2 - 12 * s)],
               fill=(255, 255, 255, int(al * 0.55)), width=max(2, int(5 * s)))


def ov_steps(d, o, q):
    """numbered steps light up one by one, then the bin badge. o: n steps, at=[fractions], bin."""
    n = o.get("n", 3); ats = o.get("steps_at", [0.05, 0.2, 0.35]); bin_at = o.get("bin_at", 0.5)
    x0, y = 80, 610
    for i in range(n):
        a = ease((q - ats[i]) / 0.06)
        if a <= 0:
            continue
        cx = x0 + 40 + i * 110
        r = 38 * (0.8 + 0.2 * a)
        d.ellipse([cx - r, y - r, cx + r, y + r], fill=(255, 255, 255, int(235 * a)))
        d.text((cx, y + 2), str(i + 1), font=font("Black", 44), fill=(30, 30, 30, int(255 * a)), anchor="mm")
        if i < n - 1 and q > ats[i + 1]:
            d.line([(cx + 44, y), (cx + 66, y)], fill=(255, 255, 255, 200), width=4)
    a = ease((q - bin_at) / 0.08)
    if a > 0:
        col = BIN_RGB[o.get("bin", "blue")]
        bx = x0 + 40 + n * 110 + 40
        d.line([(bx - 60, y), (bx - 22, y)], fill=(255, 255, 255, int(200 * a)), width=4)
        d.polygon([(bx - 22, y - 10), (bx - 8, y), (bx - 22, y + 10)], fill=(255, 255, 255, int(200 * a)))
        s = 0.9 + 0.12 * math.sin(min(1, (q - bin_at) / 0.12) * math.pi)
        d.rounded_rectangle([bx, y - 62 * s, bx + 124 * s, y + 62 * s], 22, fill=(255, 255, 255, int(235 * a)))
        bin_icon(d, bx + 62 * s, y + 10 * s, 0.95 * s, col, int(255 * a))


OVERLAYS = {"stat": ov_stat, "steps": ov_steps}


# ------------------------------------------------------------------ render
def render(spec_path, preview=False, register=False):
    from build import tts_voices, synth
    sp = json.load(open(spec_path))
    cls, slug = sp["cls"], sp["slug"]
    pdir = os.path.join(HERE, "photos", cls)
    credits = json.load(open(os.path.join(pdir, "credits.json")))
    out = os.path.join(HERE, cls); raw = os.path.join(out, "raw"); os.makedirs(raw, exist_ok=True)

    voices = tts_voices()
    seg = {}
    for b in BEATS:
        for lang in LANGS:
            p = os.path.join(raw, f"{slug}_{b}_{lang}.wav")
            seg[b, lang] = (p, synth(voices[lang], sp["beats"][b][lang], p, lang))
    dur = {b: max(seg[b, l][1] for l in LANGS) + 0.8 for b in BEATS}
    starts, acc = {}, 0.5
    for b in BEATS:
        starts[b] = acc; acc += dur[b]
    total = acc + 1.2
    print("beats:", {b: round(dur[b], 1) for b in BEATS}, "total", round(total, 1), flush=True)
    wavs = {}
    for lang in LANGS:
        buf, sr = None, None
        for b in BEATS:
            with wave.open(seg[b, lang][0]) as w:
                sr = sr or w.getframerate(); y = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            if buf is None:
                buf = np.zeros(int(total * sr), np.int16)
            i = int(starts[b] * sr); buf[i:i + len(y)] = y[:len(buf) - i]
            os.remove(seg[b, lang][0])
        wavs[lang] = os.path.join(out, f"{slug}_{lang}.wav")
        with wave.open(wavs[lang], "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(buf.tobytes())

    # timeline of shots: each beat's time split across its shots (weights optional)
    shots = []
    for b in BEATS:
        lst = sp["shots"][b]; wsum = sum(s.get("w", 1) for s in lst); t0 = starts[b]
        for s in lst:
            d_ = dur[b] * s.get("w", 1) / wsum
            shots.append(dict(s, beat=b, t0=t0, t1=t0 + d_)); t0 += d_
    shots[0]["t0"] = 0.0; shots[-1]["t1"] = total
    cache = {}

    def frame_of(s, t):
        if s["img"] not in cache:
            cache[s["img"]] = load_photo(os.path.join(pdir, s["img"]), s.get("scale", 1.35))
        span = (s["t1"] + XF) - (s["t0"] - XF if s is not shots[0] else s["t0"])
        p = (t - (s["t0"] - (XF if s is not shots[0] else 0))) / span
        return kenburns(cache[s["img"]], min(1, max(0, p)), s.get("z", [1.0, 1.12]),
                        s.get("c", [[0.5, 0.5], [0.5, 0.5]]))

    vign = make_vignette()
    mp4 = os.path.join(out, f"{slug}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    poster = False
    for f in range(int(total * FPS)):
        t = f / FPS
        i = max(k for k, s in enumerate(shots) if s["t0"] <= t)
        s = shots[i]
        a = frame_of(s, t)
        if i + 1 < len(shots) and t > s["t1"] - XF / 2:            # cross-fade into next shot
            k = ease((t - (s["t1"] - XF / 2)) / XF)
            a = a * (1 - k) + frame_of(shots[i + 1], t) * k
        elif i > 0 and t < s["t0"] + XF / 2:                        # finishing the fade from previous
            k = ease((t - (s["t0"] - XF / 2)) / XF)
            a = frame_of(shots[i - 1], t) * (1 - k) + a * k
        a = a * vign * 0.94
        fade = min(1.0, t / 0.6, (total - t) / 0.8)
        a = np.clip(a * fade, 0, 255).astype(np.uint8)
        img = Image.fromarray(a).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        beat = s["beat"]; q = (t - starts[beat]) / dur[beat]
        for o in sp.get("overlays", {}).get(beat, []):
            if o.get("at", 0) <= q <= o.get("until", 1.0):
                OVERLAYS[o["type"]](d, o, (q - o.get("at", 0)) / (o.get("until", 1.0) - o.get("at", 0)))
        # progress: 5 thin segments, top-left
        for j, b in enumerate(BEATS):
            x = 80 + j * 64
            d.rectangle([x, 44, x + 54, 48], fill=(255, 255, 255, 70))
            if b == beat:
                d.rectangle([x, 44, x + 54 * min(1, max(0, q)), 48], fill=ACCENT + (255,))
            elif BEATS.index(b) < BEATS.index(beat):
                d.rectangle([x, 44, x + 54, 48], fill=(255, 255, 255, 200))
        # photo credit
        cr = credits.get(s["img"], {})
        if cr and fade > 0.5:
            txt = f"Photo: {cr.get('artist') or 'Wikimedia Commons'} · {cr.get('license', '')} · Wikimedia Commons"
            txt = txt if len(txt) < 90 else txt[:87] + "…"
            d.text((W - 24, H - 18), txt, font=font("Regular", 15), fill=(255, 255, 255, 150), anchor="rs")
        img = Image.alpha_composite(img, ov).convert("RGB")
        ff.stdin.write(img.tobytes())
        if not poster and beat == "harm" and q > 0.5:
            img.save(os.path.join(out, f"{slug}.jpg"), quality=88); poster = True
        if f % (FPS * 10) == 0:
            print(f"  {t:5.1f}/{total:.1f}s", flush=True)
    ff.stdin.close(); ff.wait()
    print("video:", mp4, flush=True)

    if preview:
        pv = os.path.join(raw, f"{slug}_preview.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4] + sum([["-i", wavs[l]] for l in LANGS], [])
        cmd += ["-map", "0:v"] + sum([["-map", f"{k + 1}:a"] for k in range(3)], [])
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "128k"]
        for k, (l, name) in enumerate(zip(LANGS, ("eng", "hin", "tam"))):
            cmd += [f"-metadata:s:a:{k}", f"language={name}", f"-metadata:s:a:{k}", f"title={l}"]
        subprocess.run(cmd + [pv], check=True)
        print("preview:", pv, flush=True)
    if register:
        do_register(sp, credits)


def do_register(sp, credits):
    cls, slug = sp["cls"], sp["slug"]
    p = os.path.join(HERE, "index.json"); idx = json.load(open(p))
    used = sorted({s["img"] for b in BEATS for s in sp["shots"][b]})
    credit = "; ".join(f"{credits[u].get('artist', '?')} ({credits[u].get('license', '')})" for u in used if u in credits)
    entry = dict(id=slug, title=sp.get("title", {}), video=f"{cls}/{slug}.mp4",
                 vo_files={l: f"{cls}/{slug}_{l}.wav" for l in LANGS}, poster=f"{cls}/{slug}.jpg",
                 credit="Photos via Wikimedia Commons: " + credit, page="",
                 vo={l: " ".join(sp["beats"][b][l] for b in BEATS) for l in LANGS}, sources=sp.get("sources", []),
                 photo_pages=[credits[u]["page"] for u in used if u in credits])
    idx[cls] = [entry] + [c for c in idx.get(cls, []) if c["id"] != slug]
    json.dump(idx, open(p, "w"), indent=1, ensure_ascii=False)
    print("registered", cls, slug)


if __name__ == "__main__":
    render(sys.argv[1], preview="--preview" in sys.argv, register="--register" in sys.argv)
