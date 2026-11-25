"""Detect -> padded square crops -> SortiVision -> IoU tracks and consecutive agreement."""
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from .detector import Detector, DEFAULT_PTE, iou
    from .sorti_model import SortiVision
except ImportError:
    from detector import Detector, DEFAULT_PTE, iou
    from sorti_model import SortiVision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lap'))
import content as C


def square_crop(rgb, box, pad=.10):
    """Preserve the full object, including at image boundaries, using replicated edges."""
    x0, y0, x1, y1 = box[:4]
    side = max(1, int(np.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * pad))))
    left = int(round((x0 + x1 - side) / 2)); top = int(round((y0 + y1 - side) / 2))
    h, w = rgb.shape[:2]
    crop = rgb[max(0, top):min(h, top + side), max(0, left):min(w, left + side)]
    return cv2.copyMakeBorder(crop, max(0, -top), max(0, top + side - h),
                             max(0, -left), max(0, left + side - w), cv2.BORDER_REPLICATE)


class MultiVision:
    def __init__(self, detector_pte=DEFAULT_PTE, folder=HERE / 'v2_taught', *, detector=None,
                 classifier=None, confirm_frames=6, confirm_conf=None, exclude=None, centre_fallback=False,
                 hold_frames=2, smooth=.5, extra_detectors='coco', confirm_floor=.15,
                 class_bias=None, extra_every=1, pre_dedupe=None, max_area=None):  # 6@0.65: ~2x coverage, 0 wrong/hallucinated on test video
        self.detector = detector if detector is not None else Detector(detector_pte, exclude=exclude)
        self.classifier = classifier if classifier is not None else SortiVision(str(Path(folder).expanduser()))
        self.meta = self.classifier.meta
        self.confirm_frames, self.confirm_conf = confirm_frames, confirm_conf
        # per-class confirmation bar: high where the model is over-confident (e-waste), lower where it is naturally unsure
        self.class_conf = {'e_waste': .75, 'battery_waste': .50, 'light_bulbs': .75, 'general_waste': .70, 'metal_waste': .65,
                           'clothes': .65, 'paper_waste': .55, 'plastic_waste': .45, 'glass_waste': .60, 'organic_waste': .60}
        if isinstance(confirm_conf, dict):
            self.class_conf.update(confirm_conf); self.confirm_conf = .70
        elif confirm_conf is None:          # default: per-class table above
            self.confirm_conf = .70
        else:                               # an explicit single value (tests / CLI) overrides every class
            self.class_conf = {}
        self.centre_fallback = centre_fallback   # False: scan the whole frame, boxes only where waste is detected
        self.hold_frames = hold_frames   # a confirmed item stays shown through this many missed/unsure frames
        self.smooth = smooth             # box EMA weight of the previous position (0 = no smoothing)
        self.floor = confirm_floor        # no frame in a confirming streak may drop below confirm_conf - floor
        if extra_detectors == 'coco':      # hybrid default: waste YOLO + COCO YOLO11n (mouse, laptop, bottle...) + YOLO-World (paper, sponge...)
            extra_detectors = []
            if detector is None:
                for f, ex in (('det/yolo11n_320.pte', [0]), ('det/world_320.pte', [])):
                    if (HERE / f).exists():
                        extra_detectors.append(Detector(HERE / f, threshold=.15, exclude=ex))
        self.extra = list(extra_detectors or [])
        # Pi latency: run the extra detectors every Nth frame (reusing their last boxes in between) and
        # classify near-identical boxes from different detectors once (IoU >= pre_dedupe). 1 / None = off.
        self.extra_every, self.pre_dedupe = max(1, int(extra_every)), pre_dedupe
        # boxes covering more than this fraction of the frame are the base/table/background, never an item
        self.max_area = max_area
        self.zone = None                 # optional box -> bool (e.g. BaseZone.keep: only items on the cardboard base)
        self._frame = 0; self._extra_boxes = [[] for _ in self.extra]
        # prior correction for the classifier's e-waste bias (electronics-heavy desks); 1.0 = none
        self.bias = np.ones(len(self.classifier.classes), np.float32)
        for c, v in (class_bias or {}).items():
            if c in self.classifier.classes:
                self.bias[self.classifier.classes.index(c)] = v   # hybrid: e.g. COCO yolo11n (mouse, laptop, phone, bottle...) + waste YOLO
        self.tracks = {}; self.next_id = 1

    def process(self, rgb):
        boxes = list(self.detector.detect(rgb))
        for i, d in enumerate(self.extra):
            if self._frame % self.extra_every == 0 or len(self._extra_boxes) <= i:
                self._extra_boxes[i:i + 1] = [list(d.detect(rgb))]
            boxes += self._extra_boxes[i]
        self._frame += 1
        if self.zone is not None:
            boxes = [b for b in boxes if self.zone(b)]
        if self.max_area:
            h, w = rgb.shape[:2]
            boxes = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) <= self.max_area * w * h]
        if self.pre_dedupe and len(boxes) > 1:
            boxes.sort(key=lambda b: b[4], reverse=True)
            uniq = []
            for b in boxes:
                if not uniq or float(iou(np.array(b[:4]), [u[:4] for u in uniq]).max()) < self.pre_dedupe:
                    uniq.append(b)
            boxes = uniq
        candidates = []
        for box in boxes:
            p = self.classifier.probs(square_crop(rgb, box, pad=.10)) * self.bias; p = p / p.sum()
            idx = int(p.argmax()); cls, conf = self.classifier.classes[idx], float(p[idx])
            if cls != 'background':
                candidates.append((box, cls, conf, p))
        # Background detections (e.g. room furniture) must not suppress the
        # single-object fallback when the actual waste has no COCO category.
        fallback = not candidates and self.centre_fallback
        if fallback:
            h, w = rgb.shape[:2]; side = int(min(h, w) * .6)
            left, top = (w - side) // 2, (h - side) // 2
            box = (left, top, left + side, top + side, 0.)
            p = self.classifier.probs(square_crop(rgb, box, pad=0))
            idx = int(p.argmax())
            candidates = [(box, self.classifier.classes[idx], float(p[idx]), p)]
        # duplicate removal: one box per physical item (overlapping or mostly-contained boxes -> keep the surer one)
        candidates.sort(key=lambda c: c[2] * max(float(c[0][4]), .05), reverse=True)
        kept = []
        for c in candidates:
            b = np.array(c[0][:4], float); area = max((b[2] - b[0]) * (b[3] - b[1]), 1.)
            dup = False
            for k in kept:
                q = np.array(k[0][:4], float)
                ix = max(0., min(b[2], q[2]) - max(b[0], q[0])) * max(0., min(b[3], q[3]) - max(b[1], q[1]))
                qa = max((q[2] - q[0]) * (q[3] - q[1]), 1.)
                if ix / (area + qa - ix) > .45 or ix / min(area, qa) > .7:
                    dup = True; break
            if not dup:
                kept.append(c)
        candidates = kept
        boxes = [c[0] for c in candidates]
        # Greedy global IoU matching, each track/detection used at most once.
        pairs = sorted(((float(iou(np.array(b[:4]), [t['box']])[0]), j, tid)
                        for j, b in enumerate(boxes) for tid, t in self.tracks.items()
                        if t['fallback'] == fallback), reverse=True)
        matched = {}; used = set()
        for overlap, j, tid in pairs:
            if overlap >= .3 and j not in matched and tid not in used:
                matched[j] = tid; used.add(tid)
        objects = []
        for tid in list(self.tracks):          # unmatched tracks: confirmed ones are held (no flicker), others dropped
            if tid in used:
                continue
            t = self.tracks[tid]; t['missed'] += 1; t['streak'] = 0
            if t.get('confirmed_cls') and t['missed'] <= self.hold_frames:
                objects.append(self._out(tid, t, fallback))
            elif t['missed'] > (self.hold_frames if t.get('confirmed_cls') else 3):
                del self.tracks[tid]
            else:
                t['confirmed'] = False
        for j, (box, cls, conf, probs) in enumerate(candidates):
            tid = matched.get(j)
            if tid is None:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = dict(cls=None, streak=0, fallback=fallback, confirmed_cls=None, conf=0., box=list(map(float, box[:4])))
            t = self.tracks[tid]
            # steady rule: same class for N frames, mean conf >= confirm_conf, no frame below confirm_conf - 0.2
            same = t['cls'] == cls and t['streak'] > 0
            streak = t['streak'] + 1 if same else 1
            t['csum'] = (t.get('csum', 0.) if same else 0.) + conf
            need = self.class_conf.get(cls, self.confirm_conf)
            if conf < need - self.floor or cls == 'background':
                streak = 0; t['csum'] = 0.
            a = self.smooth if t.get('confirmed_cls') or t['streak'] else 0.
            nb = [a * o + (1 - a) * n for o, n in zip(t['box'], map(float, box[:4]))]
            t.update(cls=cls, box=nb, streak=streak, missed=0, score=float(box[4]))
            if streak >= self.confirm_frames and t['csum'] / streak >= need:   # (re)confirm needs a full steady streak
                t['confirmed_cls'] = cls
            if t.get('confirmed_cls') == cls:
                t['conf'] = .7 * t['conf'] + .3 * conf if t['conf'] else conf
            t['confirmed'] = bool(t.get('confirmed_cls')) and (self.hold_frames > 0 or streak >= self.confirm_frames)
            t['seen'] = t.get('seen', 0) + 1; t['dsum'] = t.get('dsum', 0.) + float(box[4])
            t['unsure'] = (not t['confirmed'] and cls != 'background' and t['seen'] >= self.confirm_frames
                           and t['dsum'] / t['seen'] >= .45)
            if not t['confirmed'] and cls == 'background':
                continue
            objects.append(self._out(tid, t, fallback, cls, conf))
        # track-level merge: one box per physical item (a held track and a new track on the same object)
        objects.sort(key=lambda o: (not o['confirmed'], o['id']))
        keep = []
        for o in objects:
            b = o['box']; dup = None
            for k in keep:
                q = k['box']; ix = max(0., min(b[2], q[2]) - max(b[0], q[0])) * max(0., min(b[3], q[3]) - max(b[1], q[1]))
                a1 = (b[2] - b[0]) * (b[3] - b[1]); a2 = (q[2] - q[0]) * (q[3] - q[1])
                if ix / max(a1 + a2 - ix, 1.) > .5 or ix / max(min(a1, a2), 1.) > .8:
                    dup = k; break
            if dup is None:
                keep.append(o)
            elif o['id'] in self.tracks and o['id'] != dup['id']:
                del self.tracks[o['id']]
        return keep

    def _out(self, tid, t, fallback, cls=None, conf=None):
        shown = t['confirmed_cls'] if t.get('confirmed') and t.get('confirmed_cls') else (cls or t['cls'])
        stream = self.meta['stream_map'].get(shown)
        return dict(id=tid, box=t['box'], cls=shown, conf=round(float(t['conf'] if t.get('confirmed') else (conf or 0)), 4),
                    score=round(float(t.get('score', 0)), 4), stream=stream,
                    color=C.BINS.get(stream, {}).get('color', 'grey'),
                    confirmed=bool(t.get('confirmed')), unsure=bool(t.get('unsure')) and not t.get('confirmed'),
                    streak=t['streak'], need=self.confirm_frames, fallback=fallback)


def signature(objects):
    """A multiset: replacing track IDs does not reannounce the same classes/counts."""
    return tuple(sorted(o['cls'] for o in objects if o['confirmed']))


def draw_objects(frame, objects, ms=0., fps=0.):
    """Clean overlay: one box per detected item. Confirmed = solid bin-coloured box + corner accents + filled tag;
    still checking = thin faint box + small tag. Header shows the item count."""
    colors = {'yellow': (0, 210, 255), 'blue': (235, 140, 40), 'green': (60, 190, 70), 'red': (50, 50, 230), 'grey': (150, 150, 150),
              'purple': (190, 80, 170), 'black': (90, 90, 90)}
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    def tag(text, x, y, bg, scale=.55, th=1):
        (tw, tht), base = cv2.getTextSize(text, font, scale, th)
        x = min(max(0, x), max(0, w - tw - 10)); y0 = y - tht - 10
        if y0 < 0:                       # no room above the box -> put the tag inside it
            y0 = y; y = y + tht + 10
        cv2.rectangle(frame, (x, y0), (x + tw + 10, y), bg, -1)
        cv2.putText(frame, text, (x + 5, y - 6), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    shown = 0; tags = []
    for o in objects:                                   # pass 1: boxes
        if o.get('unsure'):                             # detected steadily, class unknown -> neutral box, no bin advice
            x0, y0, x1, y1 = map(int, o['box']); shown += 1
            cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 2, cv2.LINE_AA)
            tags.append(("Item - not sure", x0, y0, (110, 110, 110))); continue
        if not o['confirmed']:
            continue
        x0, y0, x1, y1 = map(int, o['box'])
        color = colors.get(o.get('color'), (170, 170, 170))
        name = {'e_waste': 'E-waste', 'battery_waste': 'Battery', 'light_bulbs': 'Light bulb', 'general_waste': 'General waste'}.get(
            o['cls'], o['cls'].replace('_waste', '').replace('_', ' ').capitalize())
        dest = C.BIN_SHORT.get(o['stream'], {}).get('en', 'check')
        shown += 1
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        L = max(12, min(x1 - x0, y1 - y0) // 5)        # corner accents
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, 5, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, 5, cv2.LINE_AA)
        tags.append((f"{name} - {dest}", x0, y0, color))      # owner: no confidence %
    for text, x, y, color in tags:                      # pass 2: labels on top of every box
        tag(text, x, y, color, .6, 2)
    head = f"{shown} item{'s' if shown != 1 else ''}" if shown else "show waste to the camera"
    tag(head, 10, 34, (40, 40, 40), .65, 2)
    cv2.putText(frame, f'{fps:.1f} FPS', (w - 95, h - 12), font, .5, (120, 230, 120), 1, cv2.LINE_AA)
    return frame
