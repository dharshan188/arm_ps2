#!/usr/bin/env python3
"""Benchmark the waste classifier .pte on the Raspberry Pi 5 (or any Arm Linux box).

Measures, for the real deployed ExecuTorch + XNNPACK artifact:
  * model load time, RAM added by the model (VmRSS delta) and peak RSS (VmHWM)
  * forward-only latency p50/p90/p99/mean + FPS (warm-up excluded)
  * end-to-end latency: JPEG decode + resize + normalise + forward
  * CPU usage during the timed loop (% of one core, and % of the whole 4-core CPU)
  * SoC temperature before/after and the Pi throttle flags
  * accuracy / macro-F1 (9 classes and 4 PS2 streams) on a folder of test images

Usage (on the Pi, inside a venv with:  pip install executorch==1.3.1 pillow numpy):
  python3 pi_benchmark.py --pte mobilenet_v3_large_int8_xnnpack.pte \
      --labels labels.json --testdir pi_testset --runs 300
Run each artifact separately; close other heavy apps; use the official 27 W PSU and a cooler.
"""
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
from PIL import Image


def status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1]) / 1024.0  # MB
    return float("nan")


def soc_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except OSError:
        return None


def throttled():
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "vcgencmd not available"


def preprocess(path, spec):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    r = spec["resize_short"]  # same size rule as torchvision.transforms.Resize(int)
    nw, nh = (r, int(r * h / w)) if w <= h else (int(r * w / h), r)
    img = img.resize((nw, nh), Image.BILINEAR)
    w, h = img.size
    c = spec["center_crop"]
    l, t = (w - c) // 2, (h - c) // 2
    img = img.crop((l, t, l + c, t + c))
    a = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(spec["mean"], np.float32)) / np.array(spec["std"], np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def macro_f1(y, p, n):
    f1s = []
    for k in range(n):
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def pct(ts, q):
    return float(np.percentile(ts, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--testdir", default=None, help="folder with <class>/*.jpg subfolders")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.load(open(a.labels))
    spec, classes = meta["input"], meta["classes"]
    streams = sorted(set(meta["stream_map"].values()))
    to_s = np.array([streams.index(meta["stream_map"][c]) for c in classes])

    import torch  # pulled in by the executorch wheel
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    rss0 = status("VmRSS"); temp0 = soc_temp()
    t = time.perf_counter(); mod = _load_for_executorch(a.pte); load_ms = (time.perf_counter() - t) * 1e3

    x = torch.from_numpy(np.random.default_rng(0).standard_normal(spec["shape"]).astype(np.float32))
    t = time.perf_counter(); mod.forward((x,)); first_ms = (time.perf_counter() - t) * 1e3
    for _ in range(a.warmup):
        mod.forward((x,))

    ts = []
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(a.runs):
        s = time.perf_counter(); mod.forward((x,)); ts.append((time.perf_counter() - s) * 1e3)
    wall = time.perf_counter() - w0
    cpu_one = (time.process_time() - c0) / wall * 100
    rss_model = status("VmRSS") - rss0
    temp1 = soc_temp()

    res = {
        "pte": os.path.basename(a.pte), "pte_mb": os.path.getsize(a.pte) / 1e6,
        "device": platform.machine() + " | " + (open("/proc/device-tree/model").read().strip("\x00\n") if os.path.exists("/proc/device-tree/model") else platform.platform()),
        "cpu_cores": os.cpu_count(),
        "load_ms": load_ms, "first_inference_ms": first_ms,
        "forward_ms": {"p50": pct(ts, 50), "p90": pct(ts, 90), "p99": pct(ts, 99), "mean": float(np.mean(ts)), "min": float(np.min(ts))},
        "fps_forward": 1000.0 / float(np.mean(ts)),
        "cpu_pct_of_one_core": cpu_one, "cpu_pct_of_all_cores": cpu_one / (os.cpu_count() or 1),
        "ram_added_by_model_mb": rss_model, "peak_rss_mb": status("VmHWM"),
        "soc_temp_c": {"before": temp0, "after": temp1}, "throttled": throttled(),
    }

    if a.testdir:
        files = [(p, classes.index(c.name)) for c in sorted(Path(a.testdir).iterdir()) if c.is_dir() and c.name in classes
                 for p in sorted(c.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        ys, ps, e2e = [], [], []
        for p, y in files:
            s = time.perf_counter()
            out = mod.forward((torch.from_numpy(preprocess(p, spec)),))[0]
            e2e.append((time.perf_counter() - s) * 1e3)
            ps.append(int(out.argmax())); ys.append(y)
        ys, ps = np.array(ys), np.array(ps)
        res["test"] = {"n_images": len(ys), "acc": float((ys == ps).mean()), "macro_f1_9class": macro_f1(ys, ps, len(classes)),
                       "stream_acc": float((to_s[ys] == to_s[ps]).mean()),
                       "macro_f1_4stream": macro_f1(to_s[ys], to_s[ps], len(streams))}
        res["end_to_end_ms_incl_jpeg_decode"] = {"p50": pct(e2e, 50), "p90": pct(e2e, 90), "mean": float(np.mean(e2e))}

    res["peak_rss_mb"] = status("VmHWM")
    out = a.out or Path(a.pte).with_suffix(".pi_bench.json")
    json.dump(res, open(out, "w"), indent=2)
    f = res["forward_ms"]
    print(f"\n=== {res['pte']} on {res['device']} ===")
    print(f"size {res['pte_mb']:.2f} MB | load {load_ms:.0f} ms | first run {first_ms:.1f} ms")
    print(f"forward  p50 {f['p50']:.2f} ms  p90 {f['p90']:.2f}  p99 {f['p99']:.2f}  -> {res['fps_forward']:.0f} FPS")
    print(f"CPU {cpu_one:.0f}% of one core ({res['cpu_pct_of_all_cores']:.0f}% of all {res['cpu_cores']} cores)")
    print(f"RAM: model adds {rss_model:.1f} MB, process peak {res['peak_rss_mb']:.1f} MB")
    print(f"SoC temp {temp0} -> {temp1} C | throttled: {res['throttled']}")
    if "test" in res:
        tt, e = res["test"], res["end_to_end_ms_incl_jpeg_decode"]
        print(f"test n={tt['n_images']}: acc {tt['acc']:.4f}  macro-F1(9) {tt['macro_f1_9class']:.4f}  macro-F1(4 streams) {tt['macro_f1_4stream']:.4f}")
        print(f"end-to-end incl. JPEG decode+resize: p50 {e['p50']:.1f} ms")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
