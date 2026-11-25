#!/usr/bin/env python3
"""Fine-tune an ImageNet backbone on the Sorti v2 folders (11 classes, see prep_data.py).

    python train_sorti.py --data /kaggle/working/data --arch mobilenet_v3_large --epochs 18 --out runs/mnv3l

Real-camera robustness (the v1 model was trained on clean web photos and had no "nothing" class):
  * background class (empty rooms/desks) so an empty view is not forced into a waste class
  * paste-on-background: an item photo shrunk onto a random room photo (like holding it up to a webcam)
  * strong photometric aug: colour/brightness, blur, JPEG artefacts, noise, grayscale, random erasing
  * class-balanced sampling (clothes has ~7x more photos than e-waste), label smoothing, EMA weights
Saves best.pt (EMA weights, best val macro-F1) + metrics.json (per-class P/R/F1 on val and test).
"""
import argparse, io, json, math, os, random, time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def list_split(root, split, classes):
    items = []
    for ci, c in enumerate(classes):
        d = os.path.join(root, split, c)
        if os.path.isdir(d):
            items += [(os.path.join(d, f), ci) for f in sorted(os.listdir(d)) if f.endswith(".jpg")]
    return items


class JPEG:
    def __init__(self, p=0.3): self.p = p
    def __call__(self, im):
        if random.random() > self.p:
            return im
        b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(25, 70)); b.seek(0)
        return Image.open(b).convert("RGB")


class Noise:
    def __init__(self, p=0.25): self.p = p
    def __call__(self, t):
        return (t + torch.randn_like(t) * random.uniform(0.01, 0.06)).clamp(0, 1) if random.random() < self.p else t


def plain_surface(size=256):
    """synthetic empty table/wall: flat colour + gradient + grain (extra 'background' variety)."""
    c = np.array([random.randint(30, 230) for _ in range(3)], np.float32)
    g = np.linspace(-1, 1, size)[None, :, None] * random.uniform(0, 40)
    if random.random() < 0.5:
        g = g.transpose(1, 0, 2)
    a = c + g + np.random.normal(0, random.uniform(2, 12), (size, size, 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


class Folder(Dataset):
    def __init__(self, items, classes, train, bg_paths=(), paste_p=0.35):
        self.items, self.classes, self.train = items, classes, train
        self.bg = [p for p, _ in bg_paths]; self.paste_p = paste_p
        self.bg_idx = classes.index("background") if "background" in classes else -1
        if train:
            self.geo = T.Compose([T.RandomResizedCrop(224, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
                                  T.RandomHorizontalFlip(), T.RandomApply([T.RandomRotation(20)], p=0.3)])
            self.pix = T.Compose([T.RandomApply([T.ColorJitter(0.45, 0.45, 0.4, 0.06)], p=0.85),
                                  T.RandomGrayscale(0.05),
                                  T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25), JPEG(0.3)])
            self.post = T.Compose([T.ToTensor(), Noise(0.25), T.Normalize(MEAN, STD),
                                   T.RandomErasing(p=0.2, scale=(0.02, 0.15))])
        else:   # identical to the deployed preprocessing (labels.json): short side 256 -> centre crop 224
            self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def paste(self, im):
        """put the item on a random room photo, 45-90% of the frame, roughly centred (as in the camera box)."""
        bg = Image.open(random.choice(self.bg)).convert("RGB") if self.bg and random.random() < 0.8 else plain_surface()
        s = 288; bg = T.RandomResizedCrop(s, scale=(0.3, 1.0))(bg)
        k = random.uniform(0.45, 0.9); w, h = im.size; r = k * s / max(w, h)
        im = im.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
        cx = int(s / 2 + random.uniform(-0.12, 0.12) * s - im.size[0] / 2)
        cy = int(s / 2 + random.uniform(-0.12, 0.12) * s - im.size[1] / 2)
        mask = Image.new("L", im.size, 255).filter(ImageFilter.GaussianBlur(2)) if random.random() < 0.5 else None
        bg.paste(im, (max(0, cx), max(0, cy)), mask)
        return bg

    def __getitem__(self, i):
        p, y = self.items[i]
        im = Image.open(p).convert("RGB")
        if not self.train:
            return self.tf(im), y
        if y == self.bg_idx and random.random() < 0.15:
            im = plain_surface()
        elif y != self.bg_idx and random.random() < self.paste_p:
            im = self.paste(im)
        return self.post(self.pix(self.geo(im))), y


def build(arch, n):
    if arch == "mobilenet_v3_large":
        m = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    elif arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif arch == "efficientnet_b2":
        m = torchvision.models.efficientnet_b2(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else:
        raise SystemExit(f"unknown arch {arch}")
    return m


def prf(cm):
    tp = np.diag(cm).astype(float); p = tp / np.maximum(cm.sum(0), 1); r = tp / np.maximum(cm.sum(1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    return p, r, f


@torch.no_grad()
def evaluate(model, loader, n, dev):
    model.eval(); cm = np.zeros((n, n), np.int64)
    for x, y in loader:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            pr = model(x.to(dev, non_blocking=True)).argmax(1).cpu().numpy()
        for t, q in zip(y.numpy(), pr):
            cm[t, q] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N steps (pipeline test)")
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    os.makedirs(a.out, exist_ok=True)
    classes = json.load(open(os.path.join(a.data, "manifest.json")))["classes"]
    n = len(classes)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = (list_split(a.data, s, classes) for s in ("train", "val", "test"))
    bg = [it for it in tr if it[1] == classes.index("background")]
    print(f"{a.arch}: train {len(tr)} val {len(va)} test {len(te)} on {dev}", flush=True)

    counts = np.bincount([y for _, y in tr], minlength=n).astype(float)
    w_cls = 1.0 / np.sqrt(np.maximum(counts, 1)); w_cls /= w_cls.mean()      # sqrt balancing: fair but not extreme
    sampler = WeightedRandomSampler([w_cls[y] for _, y in tr], num_samples=len(tr), replacement=True)
    dl = DataLoader(Folder(tr, classes, True, bg), batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    mk = lambda it: DataLoader(Folder(it, classes, False), batch_size=128, num_workers=a.workers, pin_memory=True)
    dva, dte = mk(va), mk(te)

    model = build(a.arch, n).to(dev)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
                                              use_buffers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (a.max_steps or len(dl)); warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, hist = -1, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tl, k = 0.0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step(); ema.update_parameters(model); tl += loss.item(); k += 1
            if a.max_steps and k >= a.max_steps:
                break
        cm = evaluate(ema.module, dva, n, dev); _, _, f = prf(cm)
        mf = float(f.mean()); hist.append(dict(epoch=ep + 1, loss=tl / max(k, 1), val_macro_f1=mf, sec=time.time() - t0))
        print(f"ep {ep + 1:2d} loss {tl / max(k, 1):.3f} val macroF1 {mf:.4f} "
              f"bgF1 {f[classes.index('background')]:.3f} genF1 {f[classes.index('general_waste')]:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if mf > best:
            best = mf; torch.save(ema.module.state_dict(), os.path.join(a.out, "best.pt"))
    ema.module.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), map_location=dev))
    res = {"arch": a.arch, "classes": classes, "history": hist, "best_val_macro_f1": best}
    for name, dld in (("val", dva), ("test", dte)):
        cm = evaluate(ema.module, dld, n, dev); p, r, f = prf(cm)
        res[name] = dict(acc=float(np.trace(cm) / max(cm.sum(), 1)), macro_f1=float(f.mean()),
                         per_class={c: dict(p=float(p[i]), r=float(r[i]), f1=float(f[i]), n=int(cm[i].sum()))
                                    for i, c in enumerate(classes)}, confusion=cm.tolist())
        print(name, "acc", round(res[name]["acc"], 4), "macroF1", round(res[name]["macro_f1"], 4), flush=True)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
