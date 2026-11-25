"""ExecuTorch raw YOLO detector. Runtime postprocessing uses NumPy, no Ultralytics."""
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PTE = Path(__file__).resolve().parent / 'det/waste_320.pte'  # waste-trained YOLO11n; yolo11n_320/world_320 are options


def iou(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:4], boxes[:, 2:4])
    intersection = np.maximum(rb - lt, 0).prod(axis=1)
    area = np.maximum(box[2:4] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:4] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def decode(raw, shape, resized, offset, imgsz=320, threshold=.25, nms=.45,
           exclude=(0,), max_objects=300):
    """Decode [1,4+nc,N] (xywh pixels, sigmoid scores); class-agnostic NMS."""
    a = np.asarray(raw)[0].T
    if a.ndim != 2 or a.shape[1] < 5:
        raise ValueError(f'Expected raw YOLO [1,4+nc,N], got {np.shape(raw)}')
    cls = a[:, 4:].argmax(axis=1)
    scores = a[:, 4:].max(axis=1)
    keep = (scores >= threshold) & ~np.isin(cls, exclude)
    a, scores = a[keep], scores[keep]
    boxes = np.concatenate((a[:, :2] - a[:, 2:4] / 2, a[:, :2] + a[:, 2:4] / 2), axis=1)
    order = scores.argsort()[::-1]
    selected = []
    while order.size and len(selected) < max_objects:
        k = order[0]; selected.append(k)
        order = order[1:][iou(boxes[k], boxes[order[1:]]) < nms]
    h, w = shape[:2]; rw, rh = resized; left, top = offset
    result = []
    for k in selected:
        b = boxes[k].copy()
        b[[0, 2]] = np.clip((b[[0, 2]] - left) * w / rw, 0, w)
        b[[1, 3]] = np.clip((b[[1, 3]] - top) * h / rh, 0, h)
        if b[2] > b[0] and b[3] > b[1]:
            result.append((*map(float, b), float(scores[k])))
    return result


class Detector:
    def __init__(self, pte=DEFAULT_PTE, imgsz=320, threshold=.25, exclude=None):
        import torch
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        self.torch = torch
        path = Path(pte).expanduser()
        sidecar = path.with_suffix('.json')
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        self.imgsz = meta.get('imgsz', imgsz)
        # Waste-specific checkpoints must set exclude_classes=[] in their sidecar,
        # or pass --detector-exclude '' (their class 0 need not be a person).
        self.exclude = meta.get('exclude_classes', [0]) if exclude is None else exclude
        self.threshold = threshold
        self.model = _load_for_executorch(str(path))
        self.model.forward((torch.zeros(1, 3, self.imgsz, self.imgsz),))

    def detect(self, rgb):
        h, w = rgb.shape[:2]; size = self.imgsz
        scale = min(size / w, size / h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (size - rw) // 2, (size - rh) // 2
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[top:top + rh, left:left + rw] = cv2.resize(rgb, (rw, rh))
        x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        raw = self.model.forward((self.torch.from_numpy(x),))[0].numpy()
        return decode(raw, rgb.shape, (rw, rh), (left, top), size, self.threshold, exclude=self.exclude)
