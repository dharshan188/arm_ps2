#!/usr/bin/env bash
# One-time setup of the VISION part of Sorti on a Raspberry Pi 5 (64-bit Raspberry Pi OS).
#   bash pi/setup_vision.sh   (from the repo root)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="$HOME/ps2_et"

echo "== [1/4] system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev v4l-utils curl

echo "== [2/4] python environment at $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
# torch from the CPU index (no CUDA); 2.12.x is what executorch 1.3.1 was built against
"$VENV/bin/pip" install -q torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q executorch==1.3.1 opencv-python-headless pillow numpy

echo "== [3/4] check the model loads and runs"
cd vision
env -u PYTHONPATH "$VENV/bin/python" - <<'PY'
import json, time, torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
m = _load_for_executorch("mobilenet_v3_large_int8_xnnpack.pte")
x = torch.zeros(json.load(open("labels.json"))["input"]["shape"])
m.forward((x,)); t = time.perf_counter()
for _ in range(20): m.forward((x,))
print(f"model OK: {(time.perf_counter()-t)/20*1e3:.1f} ms per image")
PY

echo "== [4/4] cameras found"
v4l2-ctl --list-devices 2>/dev/null | grep -A3 -iE "usb|webcam|camera" || echo "  no USB camera found yet: plug one in"
echo
echo "Setup done. Start with:   bash pi/start_vision.sh"
