#!/usr/bin/env python3
"""Build the Sorti v2 training set (11 classes) from public sources, as plain JPEG folders.

    python prep_data.py --out /kaggle/working/data [--extra /kaggle/input/sorti-kit/general_waste] [--quick 40]

Sources (all public, downloaded from Hugging Face):
  4w4kt/waste-dataset-image   the 9 original classes (train -> train/val, test -> test)
  shahzaibvohra/realwaste     real landfill photos, CC-BY-4.0 (mapped below; "Miscellaneous Trash" -> general_waste)
  garythung/trashnet          MIT ("trash" -> general_waste)
  keremberke/indoor-scene-classification   rooms/offices/kitchens -> background ("no item in view")
  --extra folder              Wikimedia Commons photos of dusters, sponges, tissues, ... -> general_waste
Output: OUT/{train,val,test}/<class>/<source>_<n>.jpg (short side 288) + OUT/manifest.json.
Near-duplicates (16x16 average hash) in val/test that also appear in train are dropped (no leakage).
"""
import argparse, glob, hashlib, io, json, os, random, re, shutil, zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

CLASSES = ["battery_waste", "clothes", "e_waste", "glass_waste", "light_bulbs", "metal_waste",
           "organic_waste", "paper_waste", "plastic_waste", "general_waste", "background"]
BASE_MAP = {"E-waste": "e_waste", "battery waste": "battery_waste", "clothes": "clothes", "glass waste": "glass_waste",
            "light bulbs": "light_bulbs", "metal waste": "metal_waste", "organic waste": "organic_waste",
            "paper waste": "paper_waste", "plastic waste": "plastic_waste"}
REALWASTE_MAP = {"Plastic": "plastic_waste", "Metal": "metal_waste", "Paper": "paper_waste", "Cardboard": "paper_waste",
                 "Vegetation": "organic_waste", "Food Organics": "organic_waste", "Glass": "glass_waste",
                 "Textile Trash": "clothes", "Miscellaneous Trash": "general_waste"}
TRASHNET_MAP = {"cardboard": "paper_waste", "glass": "glass_waste", "metal": "metal_waste", "paper": "paper_waste",
                "plastic": "plastic_waste", "trash": "general_waste"}
SHORT = 288


def ahash(im):
    a = np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), np.float32)
    return hashlib.md5((a > a.mean()).tobytes()).hexdigest()


def save(im, path):
    im = im.convert("RGB"); w, h = im.size; r = SHORT / min(w, h)
    if r < 1:
        im = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BILINEAR)
    im.save(path, quality=90)


def split3(items, rng, fr=(0.75, 0.125)):
    items = list(items); rng.shuffle(items); n = len(items)
    a, b = int(n * fr[0]), int(n * (fr[0] + fr[1]))
    return {"train": items[:a], "val": items[a:b], "test": items[b:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default=None, help="folder of extra general_waste jpgs")
    ap.add_argument("--quick", type=int, default=0, help="cap images per class per source (pipeline test)")
    ap.add_argument("--background", type=int, default=4000, help="indoor images used for 'background'")
    a = ap.parse_args()
    rng = random.Random(0)
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download

    # records: (split, cls, source, loader) ; loader() -> PIL image
    recs = []
    cap = defaultdict(int)

    def take(split, cls, src, loader):
        k = (split, cls, src)
        if a.quick and cap[k] >= a.quick:
            return
        cap[k] += 1; recs.append((split, cls, src, loader))

    # 1) base dataset
    if a.quick:   # pipeline test: one small shard each
        ds = load_dataset("4w4kt/waste-dataset-image", data_files={"train": "data/train-00002-of-00033.parquet",
                                                                    "test": "data/test-00000-of-00003.parquet"})
    else:
        ds = load_dataset("4w4kt/waste-dataset-image")
    names = ds["train"].features["label"].names
    trl, tel = ds["train"]["label"], ds["test"]["label"]          # label columns only (no image decode)
    tr = list(range(len(trl))); rng.shuffle(tr); nval = len(tr) // 10
    for j, i in enumerate(tr):
        take("val" if j < nval else "train", BASE_MAP[names[trl[i]]], "base", (lambda i=i: ds["train"][i]["image"]))
    for i in range(len(tel)):
        take("test", BASE_MAP[names[tel[i]]], "base", (lambda i=i: ds["test"][i]["image"]))
    print("base done", len(recs), flush=True)

    # 2) RealWaste (files named "<Class>_<n>.jpg")
    rw = snapshot_download("shahzaibvohra/realwaste", repo_type="dataset",
                           allow_patterns=["*_1[0-9].jpg"] if a.quick else ["*.jpg"])
    byc = defaultdict(list)
    for f in glob.glob(os.path.join(rw, "*.jpg")):
        byc[re.sub(r"_\d+\.jpg$", "", os.path.basename(f))].append(f)
    for k, fs in byc.items():
        for sp, part in split3(sorted(fs), rng).items():
            for f in part:
                take(sp, REALWASTE_MAP[k], "realwaste", (lambda f=f: Image.open(f)))
    print("realwaste done", len(recs), flush=True)

    # 3) TrashNet (the 43 MB resized zip: <root>/<class>/<file>.jpg)
    tz = zipfile.ZipFile(hf_hub_download("garythung/trashnet", "dataset-resized.zip", repo_type="dataset"))
    idx = defaultdict(list)
    for nm in tz.namelist():
        parts = nm.split("/")
        if nm.lower().endswith(".jpg") and len(parts) >= 2 and parts[-2] in TRASHNET_MAP and "__MACOSX" not in nm:
            idx[parts[-2]].append(nm)
    for k, ii in sorted(idx.items()):
        for sp, part in split3(sorted(ii), rng).items():
            for nm in part:
                take(sp, TRASHNET_MAP[k], "trashnet", (lambda nm=nm: Image.open(io.BytesIO(tz.read(nm)))))
    print("trashnet done", len(recs), flush=True)

    # 4) background: indoor scenes (zip of jpgs)
    z = zipfile.ZipFile(hf_hub_download("keremberke/indoor-scene-classification",
                                         "data/valid-mini.zip" if a.quick else "data/train.zip", repo_type="dataset"))
    jp = sorted(n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")))
    rng.shuffle(jp); jp = jp[:a.background]
    for sp, part in split3(jp, rng, (0.8, 0.1)).items():
        for n in part:
            take(sp, "background", "indoor", (lambda n=n: Image.open(io.BytesIO(z.read(n)))))
    print("indoor done", len(recs), flush=True)

    # 5) extra general waste (Wikimedia Commons)
    if a.extra and os.path.isdir(a.extra):
        fs = sorted(glob.glob(os.path.join(a.extra, "*.jpg")))
        for sp, part in split3(fs, rng, (0.7, 0.15)).items():
            for f in part:
                take(sp, "general_waste", "commons", (lambda f=f: Image.open(f)))
        print("extra done", len(recs), flush=True)

    # write: train first, so its hashes are known before val/test
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    for sp in ("train", "val", "test"):
        for c in CLASSES:
            os.makedirs(os.path.join(a.out, sp, c), exist_ok=True)
    seen, cnt, dropped, n = set(), Counter(), Counter(), 0
    order = {"train": 0, "val": 1, "test": 2}
    for sp, cls, src, ld in sorted(recs, key=lambda r: order[r[0]]):
        try:
            im = ld(); im.load()
        except Exception:
            dropped["unreadable"] += 1; continue
        h = ahash(im)
        if sp == "train":
            seen.add(h)
        elif h in seen:
            dropped[f"{sp}_dup_of_train"] += 1; continue
        n += 1
        save(im, os.path.join(a.out, sp, cls, f"{src}_{n:06d}.jpg")); cnt[(sp, cls, src)] += 1
        if n % 5000 == 0:
            print("written", n, flush=True)
    man = {"classes": CLASSES, "counts": {f"{s}/{c}/{r}": v for (s, c, r), v in sorted(cnt.items())},
           "dropped": dict(dropped)}
    for sp in ("train", "val", "test"):
        man[sp] = {c: sum(v for (s, cc, _), v in cnt.items() if s == sp and cc == c) for c in CLASSES}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("train", "val", "test", "dropped")}, indent=1))


if __name__ == "__main__":
    main()
