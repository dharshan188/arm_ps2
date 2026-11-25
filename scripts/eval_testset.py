#!/usr/bin/env python3
"""Score the shipped vision classifier ensemble on vision/pi_testset (540 images, 9 waste types).

Reports type accuracy, bin (stream) accuracy and median per-image classification time for the
raw ensemble (MobileNetV3-L + MobileNetV4-Conv-M, dynamic INT8, ExecuTorch + XNNPACK).

    env -u PYTHONPATH ~/ps2_et/bin/python scripts/eval_testset.py
    env -u PYTHONPATH ~/ps2_et/bin/python scripts/eval_testset.py --model-dir vision/v2_taught --testdir vision/pi_testset
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))
from sorti_model import SortiVision  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=str(ROOT / "vision" / "v2_taught"))
    ap.add_argument("--testdir", default=str(ROOT / "vision" / "pi_testset"))
    a = ap.parse_args()

    sv = SortiVision(a.model_dir)
    sm, classes = sv.meta["stream_map"], list(sv.classes)
    n = t_ok = b_ok = 0
    per = Counter(); per_ok = Counter(); times = []
    for d in sorted(p for p in Path(a.testdir).iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            rgb = np.asarray(Image.open(f).convert("RGB"))
            t = time.perf_counter()
            p = sv.probs(rgb)
            times.append(time.perf_counter() - t)
            k = classes[int(np.argmax(p))]
            n += 1; per[d.name] += 1
            t_ok += k == d.name; per_ok[d.name] += k == d.name
            b_ok += sm.get(k) == sm[d.name]
    for c in sorted(per):
        print(f"  {c:15s} {100 * per_ok[c] / per[c]:5.1f}%  ({per[c]} images)")
    print(f"{n} images | type {100 * t_ok / n:.1f}% | bin {100 * b_ok / n:.1f}% | "
          f"median {1e3 * np.median(times):.0f} ms/image")


if __name__ == "__main__":
    main()
