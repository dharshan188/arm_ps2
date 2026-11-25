#!/usr/bin/env python3
"""Export a raw YOLO11 detection graph, FP32 XNNPACK; never trains.

Run with the ps2_et interpreter. Ultralytics is needed only during export.
"""
import argparse
import json
import sys
from pathlib import Path

# Local export-only installation when the shared interpreter is read-only.
sys.path.insert(0, str(Path(__file__).parent / '_export_deps'))

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default='yolo11n.pt')
    ap.add_argument('--output', default=str(Path(__file__).with_name('yolo11n_320.pte')))
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--exclude', default='0', help='comma-separated class IDs; empty for waste YOLO')
    a = ap.parse_args()
    from ultralytics import YOLO
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = YOLO(a.weights).model.cpu().float().eval()
    model.fuse()
    for m in model.modules():
        if m.__class__.__name__ == 'Detect':
            m.export = True; m.format = 'torchscript'; m.dynamic = False
    x = torch.rand(1, 3, a.imgsz, a.imgsz)
    with torch.no_grad():
        expected = model(x).numpy()
        ep = torch.export.export(model, (x,), strict=False)
        program = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(program.buffer)
    actual = _load_for_executorch(str(out)).forward((x,))[0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    meta = dict(imgsz=a.imgsz, layout='1,4+nc,N; xywh pixels + class probabilities',
                names=model.names, exclude_classes=[int(c) for c in a.exclude.split(',') if c],
                torch=torch.__version__, max_abs_error=float(np.abs(actual - expected).max()))
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Exported {out}: {out.stat().st_size / 1e6:.2f} MB, shape {actual.shape}; parity OK', flush=True)


if __name__ == '__main__':
    main()
