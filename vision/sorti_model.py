#!/usr/bin/env python3
"""Sorti vision v2 = the proven v1 9-class model + an ImageNet feature model + a small trained head (11 classes).

    from sorti_model import SortiVision
    sv = SortiVision()                 # loads labels_v2.json, both .pte files and head_v2.npz from vision/
    probs = sv.probs(rgb_crop)         # numpy RGB uint8 (H,W,3) -> probabilities over sv.classes (11)

Classes: the 9 v1 waste types + general_waste (duster, sponge, tissue, ...) + background (nothing in view).
The head sees [ImageNet MobileNetV3 features (960), v1 logits (9)], so it keeps v1's accuracy on the 9 types
and learns the two new classes. Falls back to v1 alone if the v2 files are missing.
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def preprocess(rgb, spec):
    """Resize(short=256, bilinear) -> CenterCrop(224) -> ImageNet normalise -> NCHW float32 (same as training)."""
    from PIL import Image
    img = Image.fromarray(rgb); w, h = img.size; r = spec["resize_short"]
    nw, nh = (r, int(r * h / w)) if w <= h else (int(r * w / h), r)
    img = img.resize((nw, nh), Image.BILINEAR); c = spec["center_crop"]; l, t = (nw - c) // 2, (nh - c) // 2
    a = (np.asarray(img.crop((l, t, l + c, t + c)), np.float32) / 255.0 - np.array(spec["mean"], np.float32)) \
        / np.array(spec["std"], np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


class SortiVision:
    def __init__(self, folder=HERE):
        import torch
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        self.torch = torch
        v2 = os.path.join(folder, "labels_v2.json")
        self.v2 = os.path.exists(v2) and os.path.exists(os.path.join(folder, "head_v2.npz")) and \
            os.path.exists(os.path.join(folder, json.load(open(v2)).get("feature_pte", "-")))
        self.meta = json.load(open(v2 if self.v2 else os.path.join(folder, "labels.json")))
        self.classes = self.meta["classes"]; self.spec = self.meta["input"]
        self.ens = [_load_for_executorch(os.path.join(folder, f)) for f in self.meta.get("ensemble", [])]
        self.old = self.ens[0] if self.ens else _load_for_executorch(
            os.path.join(folder, self.meta.get("v1_pte", "mobilenet_v3_large_int8_xnnpack.pte")))
        if self.v2:
            self.feat = _load_for_executorch(os.path.join(folder, self.meta["feature_pte"]))
            h = np.load(os.path.join(folder, "head_v2.npz"))
            self.mu, self.sd = h["mu"], h["sd"]
            self.W = [h[k] for k in sorted(k for k in h.files if k.startswith("W"))]
            self.b = [h[k] for k in sorted(k for k in h.files if k.startswith("b"))]
        x = torch.zeros(self.spec["shape"])
        for m in (self.ens or [self.old]):
            m.forward((x,))
        if self.v2:
            self.feat.forward((x,))

    def features(self, x):
        """x: preprocessed NCHW numpy -> (head input vector, v1 logits)"""
        t = self.torch.from_numpy(x)
        lo = self.old.forward((t,))[0].numpy()[0].astype(np.float32)
        if not self.v2:
            return None, lo
        f = self.feat.forward((t,))[0].numpy()[0].astype(np.float32)
        return np.concatenate([f, lo]), lo

    def head(self, v):
        h = (v - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
        return h

    def probs(self, rgb):
        if self.ens:      # v2: average the softmax of several .pte models (MobileNetV3-L + MobileNetV4-Conv-M)
            t = self.torch.from_numpy(preprocess(rgb, self.spec))
            return np.mean([softmax(m.forward((t,))[0].numpy()[0].astype(np.float32)) for m in self.ens], axis=0)
        v, lo = self.features(preprocess(rgb, self.spec))
        return softmax(self.head(v) if self.v2 else lo)
