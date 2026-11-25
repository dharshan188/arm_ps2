# vision/det

Raw YOLO11 / YOLO-World detectors (FP32 XNNPACK, 320 square input). At runtime they need only the
existing torch, ExecuTorch, NumPy and OpenCV. Ultralytics is needed for export only.

## Export

No training and no changes to torch or ExecuTorch. Run from the repository root:

```sh
~/ps2_et/bin/pip install --no-deps ultralytics ultralytics-thop py-cpuinfo
PATH=$HOME/ps2_et/bin:$PATH env -u PYTHONPATH YOLO_CONFIG_DIR=/tmp/sorti_yolo MPLCONFIGDIR=/tmp/sorti_mpl ~/ps2_et/bin/python vision/det/export_yolo.py --weights vision/det/yolo11n.pt
```

If site-packages is read-only, append `--target vision/det/_export_deps` to the pip command.
The COCO weights are at
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt .
The exporter compares the saved PTE output with the original PyTorch model.

## Waste-trained detector (Track B)

Export it with `--weights waste_best.pt --output vision/det/waste_320.pte
--exclude ''`. Pass `--detector-pte vision/det/waste_320.pte` to
`pi_stream.py` or `lap/server.py`. The `.json` file next to each `.pte` gives imgsz and the
excluded class IDs. Without that file, pass `--detector-exclude ''` for a waste detector:
class 0 (person) is excluded by default for COCO, but may be a waste class in Track B.

## Output format

Raw output: `[1, 4+num_classes, N]`, xywh in input pixels followed by
class probabilities. There is no objectness column and no embedded NMS. Do not substitute
an end-to-end/NMS model. The Sorti classifier and bin rules do not depend on which detector is used.

## Tracking in the earlier COCO-only build

This section describes an earlier build that used the COCO detector alone. The current
tracker settings are in `vision/multi.py` and [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

COCO localises common objects, not all waste. In that build, unknown waste used the central
60% square fallback only when there were no retained foreground detections (including when all detected
crops classified as background). Confirmation
required 8 consecutive matched frames with confidence >=0.70. No detections or
low confidence reset that track's confirmation at once; IDs survived up to
3 missed frames. Two extra unchanged frames settled the set before automatic speech.

## Validation

During development the detector pipeline was checked with a full 1,025-frame laptop run, including
inference, overlay and JPEG encoding (no playback sleep), and with a two-bottle smoke test
built from the Commons assets credited in `../testvideo/credits.json`, where both tracks had to
confirm after eight frames. The script and output files of those runs (`smoke_multi.py`,
`benchmark_laptop.json` and the saved JPEG and JSON) are not in this repository. The smoke test's English reply
was "2 plastic items: Blue bin."; that build still called the recyclable bin blue, and the shipped app
says yellow. Unit/integration checks: `python -m unittest vision.test_multi
lap.test_multi_flow`. Audio devices and Pi throughput were not exercised in these checks.
