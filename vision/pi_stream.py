#!/usr/bin/env python3
"""Multi-object Sorti MJPEG server; --video FILE --benchmark processes one pass."""
import argparse
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from multi import MultiVision, draw_objects
from detector import DEFAULT_PTE

PAGE = b'''<!doctype html><title>Sorti multi-object</title><style>body{background:#111;color:white;font-family:sans-serif}img{max-width:100%}</style><img src="/stream"><pre id="p"></pre><script>setInterval(async()=>{let a=await(await fetch('/pred')).json();document.getElementById('p').textContent=a.map(o=>`${o.cls}: ${(o.conf*100).toFixed(0)}% -> ${o.stream} ${o.confirmed?'CONFIRMED':'checking '+o.streak+'/'+o.need}`).join('\\n')},500)</script>'''
LATEST = {'jpg': None, 'pred': []}
COND = threading.Condition()


def camera_loop(a, model):
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {a.video or a.camera}')
    if not a.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    n = 0; total_ms = 0.; start = time.perf_counter(); confirmed_frames = 0; max_objects = 0
    detection_frames = 0; classes = set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if a.video and not a.benchmark:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); model.tracks.clear(); continue
                break
            if not a.video:
                frame = cv2.flip(frame, 1)
            t = time.perf_counter()
            objects = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ms = (time.perf_counter() - t) * 1000; total_ms += ms; n += 1
            fps = n / (time.perf_counter() - start)
            confirmed_frames += any(o['confirmed'] for o in objects)
            detection_frames += any(not o['fallback'] for o in objects)
            max_objects = max(max_objects, len(objects))
            classes.update(o['cls'] for o in objects if o['confirmed'])
            draw_objects(frame, objects, ms, fps)
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with COND:
                    LATEST.update(jpg=jpg.tobytes(), pred=objects); COND.notify_all()
            if a.print_every and n % a.print_every == 0:
                print(f'{n} frames | {fps:.2f} FPS | {ms:.1f} ms | {len(objects)} objects', flush=True)
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        cap.release()
    result = dict(frames=n, elapsed_s=time.perf_counter() - start, model_ms=total_ms / max(n, 1),
                  confirmed_frames=confirmed_frames, detection_frames=detection_frames,
                  max_objects=max_objects, confirmed_classes=sorted(classes))
    result['fps'] = n / max(result['elapsed_s'], 1e-9)
    print(json.dumps(result, indent=2), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(result, indent=2) + '\n')
    return result


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in ('/', '/pred'):
            with COND:
                b = PAGE if self.path == '/' else json.dumps(LATEST['pred']).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html' if self.path == '/' else 'application/json')
            self.end_headers(); self.wfile.write(b)
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame'); self.end_headers()
            try:
                while True:
                    with COND:
                        COND.wait(timeout=2); jpg = LATEST['jpg']
                    if jpg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detector-pte', default=str(DEFAULT_PTE))
    ap.add_argument('--detector-exclude', default=None, help="COCO IDs to skip; '' for waste-specific model")
    ap.add_argument('--model-dir', default=str(Path(__file__).resolve().parent / 'v2_taught'))
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--video')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--confirm-frames', type=int, default=6)
    ap.add_argument('--centre-fallback', action='store_true', help='classify the centre square when no box is detected')
    ap.add_argument('--confirm-conf', type=float, default=None, help='one bar for all classes (default: per-class table)')
    ap.add_argument('--no-hybrid', action='store_true', help='waste detector only (no COCO YOLO11n)')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--print-every', type=int, default=100)
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--report')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    exclude = None if a.detector_exclude is None else [int(c) for c in a.detector_exclude.split(',') if c]
    model = MultiVision(a.detector_pte, a.model_dir, confirm_frames=a.confirm_frames,
                        confirm_conf=a.confirm_conf, exclude=exclude, centre_fallback=a.centre_fallback,
                        extra_detectors=[] if a.no_hybrid else 'coco')
    if a.benchmark:
        camera_loop(a, model); return
    threading.Thread(target=camera_loop, args=(a, model), daemon=True).start()
    print(f'Open http://localhost:{a.port}', flush=True)
    try:
        ThreadingHTTPServer((a.host, a.port), H).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
