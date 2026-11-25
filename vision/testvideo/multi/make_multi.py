"""Multi-object exam: 60 scenes of 2-4 NEW Commons objects (never in training) on room backgrounds, exact GT boxes."""
import json, math, random, sys, importlib.util
from pathlib import Path
import cv2, numpy as np
from PIL import Image
R = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('mv', R / 'make_video.py'); mv = importlib.util.module_from_spec(spec)
sys.argv = ['x']; spec.loader.exec_module(mv)
sel = json.load(open(R / 'selection.json')); guides = json.load(open(R / 'cutout_guides.json')) if (R / 'cutout_guides.json').exists() else {}
objs, bgs = {}, []
for rec in sel:
    if rec['class'] == 'background':
        bgs.append(mv.cover(Image.open(R / rec['file']).convert('RGB')).astype(np.float32) / 255); continue
    try:
        g = guides.get(Path(rec['file']).name) or guides.get(rec['file'])
        objs.setdefault(rec['class'], []).append(mv.patch(mv.load(rec), g))
    except Exception as e:
        print('skip', rec['file'], e)
rng = random.Random(7); out = Path(__file__).resolve().parent / 'scenes'; out.mkdir(exist_ok=True); truth = []
classes = sorted(objs)
for k in range(60):
    canvas = bgs[k % len(bgs)].copy(); n = rng.randint(2, 4); picks = rng.sample(classes, n); boxes = []
    for cls in picks:
        rgba = rng.choice(objs[cls]); h, w = rgba.shape[:2]
        for _ in range(60):
            side = rng.uniform(150, 330); s = side / max(h, w); ang = rng.uniform(0, 360)
            M = cv2.getRotationMatrix2D((w * s / 2, h * s / 2), ang, 1.0)
            sw, sh = int(w * s), int(h * s); cw = int(abs(sw * math.cos(math.radians(ang))) + abs(sh * math.sin(math.radians(ang)))) + 2
            ch = int(abs(sw * math.sin(math.radians(ang))) + abs(sh * math.cos(math.radians(ang)))) + 2
            M[0, 2] += (cw - sw) / 2; M[1, 2] += (ch - sh) / 2
            piece = cv2.warpAffine(cv2.resize(rgba, (sw, sh), interpolation=cv2.INTER_AREA), M, (cw, ch))
            x, y = rng.randint(20, 1280 - cw - 20), rng.randint(20, 720 - ch - 20)
            bx = mv.object_bbox(piece[:, :, 3]); box = [x + bx[0], y + bx[1], x + bx[2], y + bx[3]]
            if all(max(0, min(box[2], b[2]) - max(box[0], b[0])) * max(0, min(box[3], b[3]) - max(box[1], b[1])) == 0 for b, _ in boxes):
                a = piece[:, :, 3:4]; canvas[y:y + ch, x:x + cw] = piece[:, :, :3] + canvas[y:y + ch, x:x + cw] * (1 - a)
                boxes.append((box, cls)); break
    img = (canvas.clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    p = out / f'scene_{k:02d}.jpg'; cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    truth.append(dict(file=p.name, objects=[dict(box=b, cls=c) for b, c in boxes]))
json.dump(truth, open(out.parent / 'truth_multi.json', 'w'), indent=1)
print('made', len(truth), 'scenes,', sum(len(t['objects']) for t in truth), 'objects; classes', classes)
