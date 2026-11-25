#!/usr/bin/env python3
"""Export a trained Sorti model to ExecuTorch (.pte, XNNPACK), FP32 and INT8, and score the .pte files themselves.

Run with torch 2.12.x + executorch 1.3.1 (the versions on the Pi), CPU is fine:
    python export_sorti.py --run runs/mnv3l --data /kaggle/working/data [--calib 400]
Writes into the run folder: <arch>_v2_fp32_xnnpack.pte, <arch>_v2_int8_xnnpack.pte, labels.json, pte_metrics.json
Scoring uses the exact deployed preprocessing (vision/webcam_demo.py: PIL resize short side 256, centre crop 224).
"""
import argparse, json, os, random, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sorti import build, list_split, prf  # noqa: E402

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
DISPLAY = {"e_waste": "E-waste", "battery_waste": "battery waste", "clothes": "clothes", "glass_waste": "glass waste",
           "light_bulbs": "light bulbs", "metal_waste": "metal waste", "organic_waste": "organic waste",
           "paper_waste": "paper waste", "plastic_waste": "plastic waste", "general_waste": "general waste",
           "background": "nothing in view"}
STREAM = {"e_waste": "e_waste", "battery_waste": "hazardous", "clothes": "recyclable", "glass_waste": "recyclable",
          "light_bulbs": "hazardous", "metal_waste": "recyclable", "organic_waste": "biodegradable",
          "paper_waste": "recyclable", "plastic_waste": "recyclable", "general_waste": "general", "background": "none"}


def prep(path):
    from PIL import Image
    img = Image.open(path).convert("RGB"); w, h = img.size; r = 256
    nw, nh = (r, int(r * h / w)) if w <= h else (int(r * w / h), r)
    img = img.resize((nw, nh), Image.BILINEAR); l, t = (nw - 224) // 2, (nh - 224) // 2
    a = (np.asarray(img.crop((l, t, l + 224, t + 224)), np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)[None]))


def lower(ep, path):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(path, "wb") as f:
        f.write(prog.buffer)
    return os.path.getsize(path)


def score(pte, items, n):
    """run the .pte on every item once; returns predictions + median latency (ms)."""
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    m = _load_for_executorch(pte); preds, ts = [], []
    for p, _ in items:
        x = prep(p); t = time.perf_counter(); o = m.forward((x,))[0]; ts.append(time.perf_counter() - t)
        preds.append(int(o.argmax()))
    return np.array(preds), float(np.median(ts) * 1e3)


def confusion(ys, ps, n):
    cm = np.zeros((n, n), np.int64)
    for y, q in zip(ys, ps):
        cm[y, q] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--calib", type=int, default=400)
    ap.add_argument("--max-test", type=int, default=0, help="score only N test images (pipeline test)")
    a = ap.parse_args()
    met = json.load(open(os.path.join(a.run, "metrics.json"))); arch, classes = met["arch"], met["classes"]; n = len(classes)
    model = build(arch, n); model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location="cpu")); model.eval()
    x0 = torch.zeros(1, 3, 224, 224)
    out = {}
    p32 = os.path.join(a.run, f"{arch}_v2_fp32_xnnpack.pte")
    out["fp32_bytes"] = lower(torch.export.export(model, (x0,)), p32); print("fp32 exported", flush=True)

    # INT8 post-training quantisation (per-channel, PT2E), calibrated on training images (val preprocessing)
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    q = XNNPACKQuantizer(); q.set_global(get_symmetric_quantization_config(is_per_channel=True))
    gm = prepare_pt2e(torch.export.export(model, (x0,)).module(), q)
    tr = list_split(a.data, "train", classes); random.Random(0).shuffle(tr)
    per = max(1, a.calib // n); calib = []
    for c in range(n):
        calib += [it for it in tr if it[1] == c][:per]
    with torch.no_grad():
        for p, _ in calib:
            gm(prep(p))
    qm = convert_pt2e(gm)
    p8 = os.path.join(a.run, f"{arch}_v2_int8_xnnpack.pte")
    out["int8_bytes"] = lower(torch.export.export(qm, (x0,)), p8); print("int8 exported", flush=True)

    te = list_split(a.data, "test", classes)
    if a.max_test:
        random.Random(1).shuffle(te); te = te[:a.max_test]
    srcs = sorted({os.path.basename(p).split("_")[0] for p, _ in te})
    ys = np.array([y for _, y in te]); src = np.array([os.path.basename(p).split("_")[0] for p, _ in te])
    for tag, pte in (("fp32", p32), ("int8", p8)):
        ps, ms = score(pte, te, n); cm = confusion(ys, ps, n); pr, rc, f = prf(cm)
        r = dict(acc=float(np.trace(cm) / cm.sum()), macro_f1=float(f.mean()), ms_p50=ms,
                 macro_f1_9_original=float(f[:9].mean()),
                 per_class={c: dict(p=float(pr[i]), r=float(rc[i]), f1=float(f[i]), n=int(cm[i].sum())) for i, c in enumerate(classes)},
                 confusion=cm.tolist())
        # accuracy per data source: base = original dataset, realwaste/trashnet = messy real photos, indoor = background
        for s_ in srcs:
            k = src == s_
            r[f"acc_{s_}"] = float((ps[k] == ys[k]).mean())
        out[tag] = r
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("per_class", "confusion")}, flush=True)
    json.dump(out, open(os.path.join(a.run, "pte_metrics.json"), "w"), indent=1)
    labels = {"arch": arch, "version": "v2", "classes": classes, "display_names": {c: DISPLAY[c] for c in classes},
              "stream_map": {c: STREAM[c] for c in classes},
              "input": {"shape": [1, 3, 224, 224], "resize_short": 256, "center_crop": 224, "mean": MEAN.tolist(),
                        "std": STD.tolist(), "layout": "NCHW RGB float32"}}
    json.dump(labels, open(os.path.join(a.run, "labels.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
