#!/usr/bin/env python3
"""Live webcam test of the PS2 waste classifier (ExecuTorch INT8 .pte).

Runs the SAME .pte file that goes on the Raspberry Pi, on your laptop/PC webcam.
Hold one item inside the centre box. Overlay shows: class, PS2 stream + bin colour,
top-3 probabilities, model latency and FPS.

Install (Windows / macOS / Linux, Python 3.10-3.13):
    pip install executorch==1.3.1 opencv-python pillow numpy
Run:
    python webcam_demo.py --pte mobilenet_v3_large_int8_xnnpack.pte --labels labels.json
Keys:  q = quit   s = save snapshot   space = freeze/unfreeze   f = toggle full-frame vs centre box
Test without a camera:  --image some.jpg   or   --video clip.mp4
"""
import argparse, json, os, sys, time
from collections import deque

import cv2
import numpy as np
from PIL import Image

# Bin colours follow the common Indian municipal scheme (green = wet, blue = dry,
# red = domestic hazardous); e-waste goes to an authorised e-waste collector.
STREAM_STYLE = {  # BGR colours for OpenCV
    "biodegradable": ((40, 160, 40), "GREEN bin (wet / compost)"),
    "recyclable": ((200, 110, 20), "BLUE bin (dry / recyclable)"),
    "hazardous": ((40, 40, 210), "RED bin (domestic hazardous) - do not mix"),
    "e_waste": ((60, 60, 60), "E-WASTE: give to authorised collector"),
    "general": ((110, 110, 110), "GREY/BLACK bin (general / reject waste)"),
    "none": ((90, 90, 90), "show one item in the box"),
}


def preprocess(rgb: np.ndarray, spec: dict) -> np.ndarray:
    """Identical to training eval transform: Resize(short=256) -> CenterCrop(224) -> normalise."""
    img = Image.fromarray(rgb)
    w, h = img.size
    r = spec["resize_short"]
    nw, nh = (r, int(r * h / w)) if w <= h else (int(r * w / h), r)
    img = img.resize((nw, nh), Image.BILINEAR)
    c = spec["center_crop"]
    l, t = (nw - c) // 2, (nh - c) // 2
    img = img.crop((l, t, l + c, t + c))
    a = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(spec["mean"], np.float32)) / np.array(spec["std"], np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def centre_box(frame, frac=0.6):
    h, w = frame.shape[:2]
    s = int(min(h - 150, w) * frac)          # keep clear of the 150 px info panel
    x0, y0 = (w - s) // 2, max(0, (h - 150 - s) // 2)
    return x0, y0, s


def draw(frame, meta, probs, ms, fps, roi, frozen, threshold):
    classes = meta["classes"]
    names = meta.get("display_names", {})
    order = np.argsort(probs)[::-1]
    top = order[0]
    cls = classes[top]
    stream = meta["stream_map"][cls]
    colour, advice = STREAM_STYLE[stream]
    conf = float(probs[top])
    uncertain = conf < threshold

    if roi is not None:
        x0, y0, s = roi
        cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (255, 255, 255) if uncertain else colour, 3)

    h, w = frame.shape[:2]
    panel_h = 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    y = h - panel_h + 34
    if uncertain:
        cv2.putText(frame, f"Not sure ({conf*100:.0f}%) - hold one item closer, plain background",
                    (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        title = f"{names.get(cls, cls)}  {conf*100:.0f}%"
        cv2.putText(frame, title, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (14, y + 12), (34, y + 32), colour, -1)
        cv2.putText(frame, f"{stream.upper()}: {advice}", (44, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    for k, i in enumerate(order[:3]):
        cv2.putText(frame, f"{names.get(classes[i], classes[i])[:18]:18s} {probs[i]*100:5.1f}%",
                    (14, y + 62 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"model {ms:.1f} ms | {fps:.0f} FPS | ExecuTorch INT8 (XNNPACK)", (w - 470, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 255, 160), 1, cv2.LINE_AA)
    if frozen:
        cv2.putText(frame, "FROZEN (space)", (w - 190, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return cls, stream, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--smooth", type=int, default=5, help="average probabilities over N frames to stop flicker")
    ap.add_argument("--threshold", type=float, default=0.55, help="below this confidence show 'not sure'")
    ap.add_argument("--box", type=float, default=0.6, help="centre box size as fraction of frame")
    ap.add_argument("--headless", action="store_true", help="no window; print predictions (for testing)")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    meta = json.load(open(a.labels))
    spec = meta["input"]
    import torch  # installed with executorch
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    model = _load_for_executorch(a.pte)
    model.forward((torch.zeros(spec["shape"]),))  # warm-up

    if a.image:
        frames = iter([cv2.imread(a.image)])
        cap = None
    else:
        src = a.video if a.video else a.camera
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if (os.name == "nt" and a.video is None) else cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"Could not open camera/video {src}. Try --camera 1, or close apps using the webcam.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    return
                yield f
        frames = gen()

    hist = deque(maxlen=max(1, a.smooth))
    use_box, frozen, last, n = True, False, None, 0
    t_prev = time.perf_counter(); fps = 0.0
    for frame in frames:
        if frame is None:
            sys.exit("Could not read the image.")
        if a.video is None and a.image is None:
            frame = cv2.flip(frame, 1)  # mirror view feels natural for a webcam
        if frozen and last is not None:
            frame = last.copy()
        roi = centre_box(frame, a.box) if use_box else None
        crop = frame[roi[1]:roi[1] + roi[2], roi[0]:roi[0] + roi[2]] if roi else frame
        x = torch.from_numpy(preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), spec))
        t = time.perf_counter()
        logits = model.forward((x,))[0].numpy()[0]
        ms = (time.perf_counter() - t) * 1e3
        hist.append(softmax(logits))
        probs = np.mean(hist, axis=0)
        now = time.perf_counter(); fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev)); t_prev = now
        view = frame.copy()
        cls, stream, conf = draw(view, meta, probs, ms, fps, roi, frozen, a.threshold)
        last = frame
        n += 1
        if a.headless or a.image:
            print(f"frame {n}: {cls} ({conf*100:.1f}%) -> {stream} | model {ms:.1f} ms")
            if a.image:
                out = os.path.splitext(a.image)[0] + "_pred.jpg"
                cv2.imwrite(out, view); print("saved", out)
                break
            if a.max_frames and n >= a.max_frames:
                break
            continue
        cv2.imshow("PS2 waste classifier - ExecuTorch INT8", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            p = f"snapshot_{int(time.time())}.jpg"; cv2.imwrite(p, view); print("saved", p)
        if k == ord(" "):
            frozen = not frozen
        if k == ord("f"):
            use_box = not use_box; hist.clear()
    if cap is not None:
        cap.release()
    if not (a.headless or a.image):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
