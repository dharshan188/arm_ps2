#!/usr/bin/env python3
"""Contact sheet of every rendered source, with review-only ROI/bbox overlays."""
import json
import math
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def main():
    timeline = json.loads((ROOT/'timeline.json').read_text())
    audit = json.loads((ROOT/'object_bboxes.json').read_text())
    bounds = {(r['frame'], r['class_']): r for r in audit['frames']}
    samples = []
    for seg in timeline['segments']:
        offsets = [seg['frames']//2] if seg['class'] == 'background' else [15, 47, 79]
        samples.extend((seg['start_frame']+offset, seg['class']) for offset in offsets)
    sheet = Image.new('RGB', (1280, math.ceil(len(samples)/4)*205), 'white')
    draw = ImageDraw.Draw(sheet)
    cap = cv2.VideoCapture(str(ROOT/'sorti_test_40s.mp4'))
    for i, (idx, cls) in enumerate(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode review frame {idx}')
        if cls == 'battery_waste' and i % 3 == 0:
            cv2.imwrite(str(ROOT/'sample_frame.jpg'), frame)
        x, y, side = audit['roi']
        cv2.rectangle(frame, (x,y), (x+side,y+side), (0,255,255), 2)
        row = bounds.get((idx, cls))
        suffix = ''
        if row:
            l,t,r,b = row['bbox']
            cv2.rectangle(frame, (l,t), (r-1,b-1), (0,255,0), 2)
            suffix = f" #{row['source_index']+1} {row['longest_px']}px"
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((320,180))
        x, y = i%4*320, i//4*205
        sheet.paste(im, (x,y))
        draw.text((x+2,y+183), f'{idx/timeline["fps"]:.2f}s {cls}{suffix}', fill='black')
    cap.release()
    sheet.save(ROOT/'video_review.jpg', quality=93)
    print(f'Saved video_review.jpg: {len(samples)} samples, all 30 object sources')


if __name__ == '__main__':
    main()
