#!/usr/bin/env python3
"""apply_picks.py <cls> 'intro=a,b;harm=c;...'  -> fills spec shots with img + Ken Burns, copies photos + credits."""
import json, os, shutil, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cls, picks = sys.argv[1], sys.argv[2]
sp_p = os.path.join(HERE, "specs", "photo", cls + ".json"); sp = json.load(open(sp_p))
pd = os.path.join(HERE, "photos", cls); cc = json.load(open(os.path.join(pd, "cand", "credits.json")))
cr_p = os.path.join(pd, "credits.json"); cr = json.load(open(cr_p)) if os.path.exists(cr_p) else {}
MOVES = [([1.0, 1.15], [[0.5, 0.5], [0.45, 0.55]]), ([1.15, 1.0], [[0.5, 0.5], [0.55, 0.45]]),
         ([1.0, 1.18], [[0.45, 0.45], [0.55, 0.5]]), ([1.12, 1.0], [[0.55, 0.5], [0.45, 0.5]])]
k = 0
for part in picks.split(";"):
    b, imgs = part.split("="); shots = []
    for im in imgs.split(","):
        f = im.strip() + ".jpg"
        if f not in cc:
            sys.exit(f"no credit for {f}")
        shutil.copy(os.path.join(pd, "cand", f), os.path.join(pd, f)); cr[f] = cc[f]
        z, c = MOVES[k % 4]; k += 1
        shots.append({"img": f, "z": z, "c": c})
    sp["shots"][b.strip()] = shots
json.dump(sp, open(sp_p, "w"), indent=1, ensure_ascii=False); json.dump(cr, open(cr_p, "w"), indent=1, ensure_ascii=False)
print(cls, "ok", {b: len(v) for b, v in sp["shots"].items()})
