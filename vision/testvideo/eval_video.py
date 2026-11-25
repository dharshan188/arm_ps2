#!/usr/bin/env python3
"""Frame-level ExecuTorch evaluation and 8-frame/0.70 confirmation events.

Crossfades are inferred from truth gaps and excluded. Confirmation state resets
at every labelled segment, so a transition cannot supply evidence for an item.
A confirmation event fires on the eighth qualifying frame; it fires again only
following a broken streak or a new class. Segment success means the FIRST
confirmation is correct. Also report whether any correct confirmation occurred.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import time
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parent
# Import the exact app functions without writing outside this directory.
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('sorti_webcam', ROOT.parent/'webcam_demo.py')
webcam = importlib.util.module_from_spec(spec); spec.loader.exec_module(webcam)
preprocess, centre_box, softmax = webcam.preprocess, webcam.centre_box, webcam.softmax

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pte',required=True); ap.add_argument('--labels',required=True)
    ap.add_argument('--video',default=str(ROOT/'sorti_test_40s.mp4'))
    ap.add_argument('--truth',default=str(ROOT/'truth.json'))
    a=ap.parse_args()
    import torch
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    torch.set_num_threads(2)
    meta=json.loads(Path(a.labels).read_text()); classes=meta['classes']; inp=meta['input']
    truth=json.loads(Path(a.truth).read_text())
    for i,s in enumerate(truth):
        if s['t0']<0 or s['t1']<=s['t0'] or (i and s['t0']<truth[i-1]['t1']):
            raise ValueError('Truth intervals must be sorted, positive, and nonoverlapping')
    model=_load_for_executorch(str(Path(a.pte).resolve()))
    model.forward((torch.zeros(inp['shape']),))
    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {a.video}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0: raise ValueError('Video must have a valid frame rate')
    segments=[dict(**s, supported=s['class'] in classes, frames=0, correct_frames=0, confirmations=[]) for s in truth]
    frames=[]; idx=0; prev_segment=None; streak_class=None; streak=0; start=time.perf_counter()
    while True:
        ok,frame=cap.read()
        if not ok: break
        t=idx/fps
        x,y,size=centre_box(frame,0.6)
        rgb=cv2.cvtColor(frame[y:y+size,x:x+size],cv2.COLOR_BGR2RGB)
        tensor=torch.from_numpy(preprocess(rgb,inp))
        logits=model.forward((tensor,))[0].detach().cpu().numpy().reshape(-1)
        if len(logits)!=len(classes): raise ValueError('Model output and labels class count differ')
        probs=softmax(logits); k=int(probs.argmax()); cls=classes[k]; conf=float(probs[k])
        si=next((j for j,s in enumerate(truth) if s['t0']<=t<s['t1']),None)
        event=False
        if si != prev_segment:
            streak_class=None; streak=0
        if si is not None:
            seg=segments[si]; seg['frames']+=1; seg['correct_frames']+=int(cls==seg['class'])
            if conf>=0.70:
                streak=streak+1 if cls==streak_class else 1
                streak_class=cls
                if streak==8:
                    event=True
                    seg['confirmations'].append({'frame':idx,'time':t,'seconds_from_start':round(t-seg['t0'],6),'class':cls,'confidence':conf,'correct':cls==seg['class']})
            else: streak_class=None; streak=0
        else: streak_class=None; streak=0
        frames.append({'frame':idx,'time':t,'class':cls,'confidence':conf,'segment':si,'confirmed':event})
        prev_segment=si; idx+=1
        if idx%200==0: print(f'Evaluated {idx} frames',file=sys.stderr,flush=True)
    cap.release()
    if not frames: raise ValueError('Video contains no readable frames')
    if truth[-1]['t1']>len(frames)/fps+1e-6: raise ValueError('Truth extends beyond video')
    print(f"{'segment':<4} {'class':<17} {'frames':>6} {'accuracy':>9} {'first confirmed':<18} {'delay(s)':>9} {'result':<7} {'wrong':>6}")
    for i,s in enumerate(segments):
        events=s['confirmations']; first=events[0] if events else None
        s.update(frame_accuracy=s['correct_frames']/s['frames'] if s['frames'] else None,
                 confirmed_class=first['class'] if first else None,
                 confirmed_correct=bool(first and first['correct']),
                 seconds_to_first_confirmation=first['seconds_from_start'] if first else None,
                 any_correct_confirmation=any(e['correct'] for e in events),
                 seconds_to_first_correct_confirmation=next((e['seconds_from_start'] for e in events if e['correct']),None),
                 wrong_confirmations=[e for e in events if not e['correct']])
        delay=f"{s['seconds_to_first_confirmation']:.2f}" if first else '-'
        print(f"{i:<4} {s['class']:<17} {s['frames']:>6} {s['frame_accuracy']:>8.1%} {(s['confirmed_class'] or '-'): <18} {delay:>9} {('correct' if s['confirmed_correct'] else 'wrong' if first else 'none'):<7} {len(s['wrong_confirmations']):>6}" + ('' if s['supported'] else ' (absent from labels)'))
    total=sum(s['frames'] for s in segments); supported=[s for s in segments if s['supported']]
    support_n=sum(s['frames'] for s in supported)
    summary=dict(overall_frame_accuracy=sum(s['correct_frames'] for s in segments)/total,
                 segment_accuracy=sum(s['confirmed_correct'] for s in segments)/len(segments),
                 supported_frame_accuracy=sum(s['correct_frames'] for s in supported)/support_n if support_n else None,
                 supported_segment_accuracy=sum(s['confirmed_correct'] for s in supported)/len(supported) if supported else None,
                 scored_frames=total, crossfade_frames=len(frames)-total,
                 wrong_confirmations=sum(len(s['wrong_confirmations']) for s in segments))
    print(f"Overall frame accuracy: {summary['overall_frame_accuracy']:.2%}; segment accuracy (first confirmation): {summary['segment_accuracy']:.2%}")
    print(f"Supported-class frame accuracy: {summary['supported_frame_accuracy']:.2%}; segment accuracy: {summary['supported_segment_accuracy']:.2%}")
    out=ROOT/f'eval_{Path(a.pte).stem}.json'
    out.write_text(json.dumps(dict(model=str(Path(a.pte).resolve()),labels=str(Path(a.labels).resolve()),video=str(Path(a.video).resolve()),fps=fps,classes=classes,confirmation_rule={'frames':8,'confidence':0.70,'reset_at_segment':True,'segment_success':'first confirmation correct'},summary=summary,segments=segments,frames=frames,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')
    print(f'Saved {out}')
if __name__=='__main__': main()
