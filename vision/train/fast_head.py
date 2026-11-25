#!/usr/bin/env python3
"""Fast (~10 min, CPU) Sorti v2: train a small head on top of [ImageNet MobileNetV3-L INT8 features, v1 logits].

    env -u PYTHONPATH ~/ps2_et/bin/python fast_head.py            # writes vision/{labels_v2.json, head_v2.npz, ...}

Steps: export ImageNet MobileNetV3-L feature model (960-d) to INT8 .pte -> run BOTH deployed .pte files on every
image (clean + augmented camera-style views) -> train a 2-layer MLP head (seconds) -> score on held-out data
(v1 model alone vs v2, including vision/pi_testset) -> save. Features come from the exact .pte files the Pi runs.
"""
import glob, io, json, os, random, sys, time, zipfile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__)); VIS = os.path.dirname(HERE)
sys.path.insert(0, VIS)
from sorti_model import preprocess, softmax  # noqa: E402

DATA = os.path.expanduser("~/sorti_data")
CLASSES = ["battery_waste", "clothes", "e_waste", "glass_waste", "light_bulbs", "metal_waste",
           "organic_waste", "paper_waste", "plastic_waste", "general_waste", "background"]
CI = {c: i for i, c in enumerate(CLASSES)}
RW = {"Plastic": "plastic_waste", "Metal": "metal_waste", "Paper": "paper_waste", "Cardboard": "paper_waste",
      "Vegetation": "organic_waste", "Food Organics": "organic_waste", "Glass": "glass_waste",
      "Textile Trash": "clothes", "Miscellaneous Trash": "general_waste"}
TN = {"cardboard": "paper_waste", "glass": "glass_waste", "metal": "metal_waste", "paper": "paper_waste",
      "plastic": "plastic_waste", "trash": "general_waste"}
SPEC = json.load(open(os.path.join(VIS, "labels.json")))["input"]
FEAT_PTE = "mnv3l_imagenet_feat_int8_xnnpack.pte"
rng = random.Random(0)


def unzip(zp, dst):
    if not os.path.isdir(dst) and os.path.exists(zp):
        zipfile.ZipFile(zp).extractall(dst)
    return dst


def collect():
    """-> list of (path, class, source, split)"""
    items = []

    def add(paths, cls, src, fr=(0.7, 0.15), cap=None):
        paths = sorted(paths); rng.shuffle(paths); paths = paths[:cap] if cap else paths
        n = len(paths); a, b = int(n * fr[0]), int(n * (fr[0] + fr[1]))
        for i, p in enumerate(paths):
            items.append((p, cls, src, "train" if i < a else "val" if i < b else "test"))

    for d in glob.glob(f"{DATA}/small/base/*"):          # 80/class from the v1 dataset; test = pi_testset
        add(glob.glob(d + "/*.jpg"), os.path.basename(d), "base", (0.85, 0.15))
    for d in glob.glob(f"{DATA}/small/realwaste/*"):
        add(glob.glob(d + "/*.jpg"), RW[os.path.basename(d)], "realwaste")
    tn = unzip(f"{DATA}/raw/trashnet.zip", f"{DATA}/trashnet")
    for k, c in TN.items():
        add(glob.glob(f"{tn}/**/{k}/*.jpg", recursive=True), c, "trashnet", cap=None if k == "trash" else 150)
    ind = unzip(f"{DATA}/raw/indoor_valid.zip", f"{DATA}/indoor")
    add([p for p in glob.glob(f"{ind}/**/*", recursive=True) if p.lower().endswith((".jpg", ".jpeg", ".png"))],
        "background", "indoor", cap=1400)
    add(glob.glob(os.path.join(VIS, "extra/general_waste/*.jpg")), "general_waste", "commons")
    for d in glob.glob(os.path.join(VIS, "pi_testset/*")):
        for p in glob.glob(d + "/*"):
            items.append((p, os.path.basename(d), "pitest", "test"))
    # drop exact duplicates (same bytes) across ALL sources/splits: test copies win, then val, then train
    import hashlib
    rank = {"test": 0, "val": 1, "train": 2}; seen, out, dup = set(), [], 0
    for it in sorted(items, key=lambda it: rank[it[3]]):
        try:
            h = hashlib.md5(open(it[0], "rb").read()).hexdigest()
        except OSError:
            continue
        if h in seen:
            dup += 1; continue
        seen.add(h); out.append(it)
    print("duplicates dropped:", dup, flush=True)
    return out


def plain_surface(s=256):
    c = np.array([rng.randint(30, 230) for _ in range(3)], np.float32)
    g = np.linspace(-1, 1, s)[None, :, None] * rng.uniform(0, 40)
    a = c + (g if rng.random() < 0.5 else g.transpose(1, 0, 2)) + np.random.normal(0, rng.uniform(2, 12), (s, s, 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def augment(im, cls, bgs):
    """camera-style view: item pasted on a room/desk, random crop/zoom, colour, blur, JPEG."""
    from torchvision import transforms as T
    if cls == "background":
        im = plain_surface() if rng.random() < 0.15 else im
    elif rng.random() < 0.5 and bgs:
        bg = plain_surface() if rng.random() < 0.25 else Image.open(rng.choice(bgs)).convert("RGB")
        s = 300; bg = T.RandomResizedCrop(s, scale=(0.3, 1.0))(bg)
        k = rng.uniform(0.45, 0.9); w, h = im.size; r = k * s / max(w, h)
        it = im.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
        x = int(s / 2 + rng.uniform(-0.1, 0.1) * s - it.size[0] / 2); y = int(s / 2 + rng.uniform(-0.1, 0.1) * s - it.size[1] / 2)
        bg.paste(it, (max(0, x), max(0, y))); im = bg
    im = T.RandomResizedCrop(256, scale=(0.45, 1.0), ratio=(0.8, 1.25))(im)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = T.ColorJitter(0.4, 0.4, 0.35, 0.05)(im)
    if rng.random() < 0.25:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.8)))
    if rng.random() < 0.3:
        b = io.BytesIO(); im.save(b, "JPEG", quality=rng.randint(30, 70)); b.seek(0); im = Image.open(b).convert("RGB")
    return im


def export_feature_pte(path, calib_paths):
    import torchvision
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    class Feat(nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")
            self.f, self.p = m.features, m.avgpool
        def forward(self, x):
            return torch.flatten(self.p(self.f(x)), 1)

    m = Feat().eval(); x0 = torch.zeros(1, 3, 224, 224)
    q = XNNPACKQuantizer(); q.set_global(get_symmetric_quantization_config(is_per_channel=True))
    gm = prepare_pt2e(torch.export.export(m, (x0,)).module(), q)
    with torch.no_grad():
        for p in calib_paths:
            gm(torch.from_numpy(preprocess(np.asarray(Image.open(p).convert("RGB")), SPEC)))
    prog = to_edge_transform_and_lower(torch.export.export(convert_pt2e(gm), (x0,)),
                                       partitioner=[XnnpackPartitioner()]).to_executorch()
    open(path, "wb").write(prog.buffer)


def macro(cm):
    tp = np.diag(cm).astype(float); p = tp / np.maximum(cm.sum(0), 1); r = tp / np.maximum(cm.sum(1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-9); return f


def main():
    t_all = time.time()
    items = collect()
    from collections import Counter
    print("items:", dict(Counter((s, sp) for _, _, s, sp in items)), flush=True)
    feat_path = os.path.join(VIS, FEAT_PTE)
    if not os.path.exists(feat_path):
        tr = [p for p, c, s, sp in items if sp == "train"]; rng.shuffle(tr)
        t = time.time(); export_feature_pte(feat_path, tr[:200]); print(f"feature pte exported ({time.time() - t:.0f}s)", flush=True)
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    old = _load_for_executorch(os.path.join(VIS, "mobilenet_v3_large_int8_xnnpack.pte"))
    feat = _load_for_executorch(feat_path)
    bgs = [p for p, c, s, sp in items if c == "background" and sp == "train"]

    def run(im):
        x = torch.from_numpy(preprocess(np.asarray(im.convert("RGB")), SPEC))
        lo = old.forward((x,))[0].numpy()[0]; f = feat.forward((x,))[0].numpy()[0]
        return np.concatenate([f, lo]).astype(np.float32)

    X, Y, S, SP = [], [], [], []
    t = time.time(); views = {"train": 3, "val": 1, "test": 1}
    for k, (p, c, s, sp) in enumerate(items):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        for v in range(views[sp]):
            X.append(run(im if v == 0 else augment(im, c, bgs))); Y.append(CI[c]); S.append(s); SP.append(sp)
        if k % 500 == 0:
            print(f"  features {k}/{len(items)}  {time.time() - t:.0f}s", flush=True)
    X, Y, S, SP = np.stack(X), np.array(Y), np.array(S), np.array(SP)
    print(f"features done: {X.shape} in {time.time() - t:.0f}s", flush=True)

    tr, va, te = SP == "train", SP == "val", SP == "test"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5
    Xt = torch.from_numpy((X - mu) / sd).float(); Yt = torch.from_numpy(Y)
    cnt = np.bincount(Y[tr], minlength=len(CLASSES)).astype(float)
    w = torch.tensor(1 / np.sqrt(np.maximum(cnt, 1)) / np.mean(1 / np.sqrt(np.maximum(cnt, 1))), dtype=torch.float32)
    torch.manual_seed(0)
    best = None
    for hid in (256, 512):
        net = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, len(CLASSES)))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
        itr = torch.from_numpy(np.where(tr)[0])
        for ep in range(60):
            net.train(); perm = itr[torch.randperm(len(itr))]
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]; opt.zero_grad(); lossf(net(Xt[b]), Yt[b]).backward(); opt.step()
            if ep % 5 == 4:
                net.eval()
                with torch.no_grad():
                    pv = net(Xt[va]).argmax(1).numpy()
                cm = np.zeros((len(CLASSES),) * 2, int)
                for a_, b_ in zip(Y[va], pv):
                    cm[a_, b_] += 1
                mf = macro(cm).mean()
                if best is None or mf > best[0]:
                    best = (mf, hid, ep + 1, {k: v.clone() for k, v in net.state_dict().items()})
    print(f"best head: hidden {best[1]}, epoch {best[2]}, val macroF1 {best[0]:.4f}", flush=True)
    net = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], best[1]), nn.ReLU(), nn.Dropout(0.3), nn.Linear(best[1], len(CLASSES)))
    net.load_state_dict(best[3]); net.eval()
    with torch.no_grad():
        pred = net(Xt).argmax(1).numpy()
    old_pred = X[:, 960:].argmax(1)      # v1 logits, 9 classes: index matches CLASSES[:9]

    res = {"classes": CLASSES, "hidden": best[1], "val_macro_f1": float(best[0])}
    for src in sorted(set(S[te])):
        k = te & (S == src)
        res[f"test_{src}"] = dict(n=int(k.sum()), v2_acc=float((pred[k] == Y[k]).mean()),
                                  v1_acc=float((old_pred[k] == Y[k]).mean()))
    cm = np.zeros((len(CLASSES),) * 2, int)
    for a_, b_ in zip(Y[te], pred[te]):
        cm[a_, b_] += 1
    f = macro(cm)
    res["test_all"] = dict(v2_acc=float(np.trace(cm) / cm.sum()), v2_macro_f1=float(f.mean()),
                           per_class_f1={c: round(float(f[i]), 3) for i, c in enumerate(CLASSES)}, confusion=cm.tolist())
    for kk, vv in res.items():
        if kk.startswith("test_") and kk != "test_all":
            print(f"{kk:18s} n={vv['n']:4d}  v1 {vv['v1_acc']:.3f}  ->  v2 {vv['v2_acc']:.3f}")
    print("test all: v2 acc", round(res["test_all"]["v2_acc"], 4), "macroF1", round(res["test_all"]["v2_macro_f1"], 4))
    print("per-class F1:", res["test_all"]["per_class_f1"])

    sdict = best[3]
    np.savez(os.path.join(VIS, "head_v2.npz"), mu=mu, sd=sd,
             W0=sdict["1.weight"].numpy().T.copy(), b0=sdict["1.bias"].numpy(),
             W1=sdict["4.weight"].numpy().T.copy(), b1=sdict["4.bias"].numpy())
    meta = json.load(open(os.path.join(VIS, "labels.json")))
    meta.update(version="v2", classes=CLASSES, feature_pte=FEAT_PTE, v1_pte="mobilenet_v3_large_int8_xnnpack.pte",
                v1_classes=meta["classes"])
    meta["display_names"].update(general_waste="general waste", background="nothing in view")
    meta["stream_map"].update(general_waste="general", background="none")
    json.dump(meta, open(os.path.join(VIS, "labels_v2.json"), "w"), indent=2)
    json.dump(res, open(os.path.join(VIS, "metrics_v2.json"), "w"), indent=1)
    print(f"saved labels_v2.json head_v2.npz metrics_v2.json  (total {time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
