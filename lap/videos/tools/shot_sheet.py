#!/usr/bin/env python3
"""For a photo spec with {"need","query"} shots: fetch candidates per shot (serial, Commons) and draw a contact
sheet: one row per shot (need text) + extra rows with prefetched candidates.
usage: shot_sheet.py specs/photo/<cls>.json OUT.jpg [--nofetch]"""
import glob, json, math, os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(HERE, "tools", "commons_fetch.py")
sp = json.load(open(sys.argv[1])); cls = sp["cls"]; out = sys.argv[2]
cand = os.path.join(HERE, "photos", cls, "cand"); os.makedirs(cand, exist_ok=True)
BEATS = ["intro", "harm", "who", "dispose", "prevent"]
rows = []
for b in BEATS:
    for i, s in enumerate(sp["shots"][b]):
        tag = f"{b}{i}"
        if "--nofetch" not in sys.argv and not glob.glob(f"{cand}/{tag}_*.jpg"):
            for q in (s.get("query"), s.get("query2")):
                if q:
                    subprocess.run(["timeout", "150", "python3", "-I", F, cand, tag, q, "3"])
                    if glob.glob(f"{cand}/{tag}_*.jpg"):
                        break
        rows.append((f"{tag}: {s.get('need', '')[:80]}", sorted(glob.glob(f"{cand}/{tag}_*.jpg"))))
extra = sorted(f for f in glob.glob(f"{cand}/*.jpg") if not any(os.path.basename(f).startswith(b) for b in BEATS))
for k in range(0, len(extra), 6):
    rows.append(("prefetched", extra[k:k + 6]))
tw, th, lh = 300, 169, 26
f1 = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Bold.ttf", 20)
sheet = Image.new("RGB", (6 * tw, len(rows) * (th + lh)), "black"); d = ImageDraw.Draw(sheet)
for r, (label, fs) in enumerate(rows):
    y = r * (th + lh); d.text((6, y + 2), label, font=f1, fill="cyan")
    for c, p in enumerate(fs[:6]):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            os.remove(p); continue
        im.thumbnail((tw - 4, th)); x = c * tw
        sheet.paste(im, (x, y + lh)); d.rectangle([x, y + lh, x + 130, y + lh + 22], fill="black")
        d.text((x + 3, y + lh), os.path.basename(p)[:-4], font=f1, fill="yellow")
sheet.save(out, quality=78); print("sheet", out, len(rows), "rows")
