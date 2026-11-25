#!/usr/bin/env python3
"""Reproducible Commons photo composite stress test (no burned-in labels).

Each of 11 classes has a 3.8s segment; two extra backgrounds have 2s segments.
Adjacent segments overlap by 0.4s: 41 seconds at 25fps. Truth intervals omit
both sides of each crossfade. All three source photos appear within each waste
segment. Crops in credits.json are normalized [left, top, right, bottom].
"""
import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('sorti_webcam',ROOT.parent/'webcam_demo.py')
webcam=importlib.util.module_from_spec(spec); spec.loader.exec_module(webcam)
W,H,FPS=1280,720,25
FADE=10

def load(record):
    im=Image.open(ROOT/record['file']).convert('RGB')
    l,t,r,b=record.get('crop',[0,0,1,1]); w,h=im.size
    return im.crop((int(l*w),int(t*h),int(r*w),int(b*h)))
def cover(im):
    w,h=im.size; factor=max(W/w,H/h)
    im=im.resize((math.ceil(w*factor),math.ceil(h*factor)),Image.Resampling.LANCZOS)
    x=(im.width-W)//2; y=(im.height-H)//2
    return np.asarray(im.crop((x,y,x+W,y+H))).copy()
def patch(im, guide=None):
    """Segment the reviewed crop, trim to foreground, and feather its silhouette."""
    im=im.copy(); im.thumbnail((500,500),Image.Resampling.LANCZOS)
    rgb=np.asarray(im).copy(); h,w=rgb.shape[:2]
    # Pad with the crop's edge colour; the two-pixel inner rim supplies real
    # background samples, including white cards and grey pavement.
    edge=np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    padded=cv2.copyMakeBorder(rgb,10,10,10,10,cv2.BORDER_CONSTANT,value=tuple(int(v) for v in np.median(edge,axis=0)))
    mask=np.zeros(padded.shape[:2],np.uint8)
    cv2.setRNGSeed(42)
    cv2.grabCut(padded,mask,(12,12,w-4,h-4),None,None,7,cv2.GC_INIT_WITH_RECT)
    foreground=np.isin(mask[10:10+h,10:10+w],[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
    if guide:
        silhouette=np.zeros((h,w),np.uint8)
        def points(values):
            return np.rint(np.array(values)*[w-1,h-1]).astype(np.int32)
        for polygon in guide.get('polygons',[]):
            cv2.fillPoly(silhouette,[points(polygon)],255)
        for stroke in guide.get('strokes',[]):
            cv2.polylines(silhouette,[points(stroke['points'])],False,255,max(2,round(stroke['width']*min(w,h))),cv2.LINE_AA)
        # Cable guides explicitly restore thin wires missed by colour segmentation.
        foreground=silhouette if guide.get('strokes') else cv2.bitwise_and(foreground,silhouette)
    if not np.any(foreground): raise ValueError('GrabCut found no object')
    alpha=np.asarray(Image.fromarray(foreground).filter(ImageFilter.GaussianBlur(1.0)))
    yy,xx=np.where(alpha>=128)
    left=max(0,xx.min()-4); right=min(w,xx.max()+5)
    top=max(0,yy.min()-4); bottom=min(h,yy.max()+5)
    # Premultiplied RGB avoids dark/white fringes when warping the silhouette.
    rgba=np.dstack([rgb,alpha]).astype(np.float32)/255
    rgba[:,:,:3]*=rgba[:,:,3:4]
    return rgba[top:bottom,left:right]

def object_bbox(alpha):
    """Integer half-open bounds of the visible (>=50% opacity) object."""
    yy,xx=np.where(alpha>=.5)
    if not len(xx): raise ValueError('Object has no visible alpha support')
    return int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)

def render_segment(seg, backgrounds, sources, roi, seed, audit):
    rng=np.random.default_rng(seed); bg=backgrounds[seg['background']].astype(np.float32)/255
    n=seg['frames']; x,y,s=roi; phase=rng.uniform(0,2*np.pi,4)
    # The spec uses the geometric centre square, not the panel-adjusted app ROI.
    # Centres also stay inside the app ROI to keep evaluation meaningful.
    ax,ay,app_side=webcam.centre_box(np.empty((H,W,3)),.6)
    cx=x+s/2; cy=(max(y,ay)+min(y+s,ay+app_side))/2
    for i in range(n):
        u=i/max(n-1,1)
        frame=bg.copy()
        if seg['class']!='background':
            source_index=min(2,int(u*3))
            photo=sources[seg['class']][source_index]
            ph,pw=photo.shape[:2]
            scale_frac=.575+.225*math.sin(2*math.pi*u+phase[0])
            size=s*scale_frac
            angle=2*math.pi*u
            c,sn=math.cos(angle), math.sin(angle)
            corners=np.array([[-pw/2,-ph/2],[pw/2,-ph/2],[pw/2,ph/2],[-pw/2,ph/2]],np.float32)
            corners*=size/max(pw,ph)
            corners[:,0]*=1-.12*math.sin(2*math.pi*u+phase[1])**2
            corners[:,1]*=1-.10*math.cos(2*math.pi*u+phase[2])**2
            corners=corners@np.array([[c,sn],[-sn,c]],np.float32)
            corners+=np.array([[1,1],[-1,-1],[1,-1],[-1,1]],np.float32)*size*.025*math.sin(2*math.pi*u)
            src=np.array([[0,0],[pw-1,0],[pw-1,ph-1],[0,ph-1]],np.float32)
            corners-=np.floor(corners.min(axis=0))-4
            transform=cv2.getPerspectiveTransform(src,corners.astype(np.float32))
            obj=cv2.warpPerspective(photo,transform,tuple(np.ceil(corners.max(axis=0)+5).astype(int)),flags=cv2.INTER_LINEAR)
            # Normalize AFTER rotation/perspective using foreground, never crop size.
            l,t,r,b=object_bbox(obj[:,:,3]); extent=max(r-l,b-t)
            obj=obj[max(0,t-3):b+3,max(0,l-3):r+3]
            # One pixel of headroom absorbs alpha resampling/blur rounding.
            target=int(np.clip(round(size),math.ceil(.35*s)+1,math.floor(.80*s)-1))
            oh,ow=obj.shape[:2]
            obj=cv2.resize(obj,(max(1,round(ow*target/extent)),max(1,round(oh*target/extent))),interpolation=cv2.INTER_LINEAR)
            # Restore transparent padding for blur and feathering at the boundary.
            obj=cv2.copyMakeBorder(obj,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
            if i%5 in (1,2): obj=cv2.blur(obj,(3,1))
            l,t,r,b=object_bbox(obj[:,:,3])
            # Thin tips can cross the alpha threshold during interpolation.
            # Correct the measured result, including blur, before placement.
            for _ in range(5):
                measured=max(r-l,b-t)
                if .35*s<=measured<=.80*s: break
                corrected=np.clip(measured,math.ceil(.35*s)+1,math.floor(.80*s)-1)
                oh,ow=obj.shape[:2]
                obj=cv2.resize(obj,(max(1,round(ow*corrected/measured)),max(1,round(oh*corrected/measured))),interpolation=cv2.INTER_LINEAR)
                l,t,r,b=object_bbox(obj[:,:,3])
            center_x=cx+.12*s*math.sin(2*math.pi*u+phase[1])
            center_y=cy+.12*s*math.sin(4*math.pi*u+phase[2])
            left=round(center_x-(l+r)/2); top=round(center_y-(t+b)/2)
            oh,ow=obj.shape[:2]; right=left+ow; bottom=top+oh
            bbox=[left+l,top+t,left+r,top+b]
            longest=max(r-l,b-t); actual_cx=(bbox[0]+bbox[2])/2; actual_cy=(bbox[1]+bbox[3])/2
            assert .35*s<=longest<=.80*s, (seg['class'],i,bbox,longest)
            assert x<=actual_cx<=x+s and y<=actual_cy<=y+s, bbox
            assert ax<=actual_cx<=ax+app_side and ay<=actual_cy<=ay+app_side, bbox
            assert 0<=left<right<=W and 0<=top<bottom<=H, bbox
            audit.append(dict(frame=seg['start_frame']+i,segment_frame=i,class_=seg['class'],source_index=source_index,bbox=bbox,longest_px=longest,fraction=longest/s,center=[actual_cx,actual_cy]))
            alpha=obj[:,:,3:4]
            bright=.04*math.sin(2*math.pi*u+phase[3]); contrast=1+.10*math.sin(4*math.pi*u)
            color=np.clip((obj[:,:,:3]/np.maximum(alpha,1e-6)-.5)*contrast+.5+bright,0,1)
            region=frame[top:bottom,left:right]
            frame[top:bottom,left:right]=region*(1-alpha)+color*alpha
        frame=np.clip(frame*(1+.025*math.sin(2*math.pi*u)),0,1)
        noise=rng.normal(0,.0025,(H,W,1)).astype(np.float32)
        frame=np.clip(frame+noise,0,1)
        yield (frame*255+.5).astype(np.uint8)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--seed',type=int,default=20261118); a=ap.parse_args()
    cv2.setNumThreads(2)
    credits=json.loads((ROOT/'credits.json').read_text())
    groups={}
    for r in credits: groups.setdefault(r['class'],[]).append(r)
    expected={'battery_waste','clothes','e_waste','glass_waste','light_bulbs','metal_waste','organic_waste','paper_waste','plastic_waste','general_waste','background'}
    if set(groups)!=expected or any(len(v)!=3 for v in groups.values()): raise ValueError('Need exactly three reviewed photos per class')
    guides=json.loads((ROOT/'cutout_guides.json').read_text())
    sources={}
    for k,v in groups.items():
        if k=='background': continue
        print(f'Preparing cutouts: {k}',flush=True)
        sources[k]=[patch(load(r),guides.get(Path(r['file']).stem)) for r in v]
    backgrounds=[cover(load(r)) for r in groups['background']]
    rng=np.random.default_rng(a.seed)
    order=list(groups); rng.shuffle(order)
    # Shuffle extra background segments alongside the class segments.
    segments=[dict(class_=c,frames=95) for c in order]+[dict(class_='background',frames=50) for _ in range(2)]
    rng.shuffle(segments)
    timeline=[]; truth=[]; start=0
    for j,seg in enumerate(segments):
        seg['class']=seg.pop('class_'); seg['background']=j%3
        seg['start_frame']=start
        end=start+seg['frames']
        truth.append({'t0':(start+(FADE if j else 0))/FPS,'t1':(end-(FADE if j<len(segments)-1 else 0))/FPS,'class':seg['class']})
        timeline.append(dict(seg,photos=[r['file'] for r in groups[seg['class']]]))
        start=end-FADE
    (ROOT/'truth.json').write_text(json.dumps(truth,indent=2)+'\n')
    (ROOT/'timeline.json').write_text(json.dumps(dict(seed=a.seed,fps=FPS,resolution=[W,H],crossfade_frames=FADE,segments=timeline,roi=list(webcam.centre_box(np.empty((H,W,3)),.6))),indent=2)+'\n')
    outfile=ROOT/'sorti_test_40s.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(outfile)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    tail=[]; count=0
    def write(frame):
        nonlocal count
        proc.stdin.write(frame.tobytes()); count+=1
    side=int(.6*min(H,W)); roi=((W-side)//2,(H-side)//2,side)
    audit=[]
    try:
        for j,seg in enumerate(segments):
            print(f"Rendering {j+1}/{len(segments)}: {seg['class']}",flush=True)
            generated=list(render_segment(seg,backgrounds,sources,roi,a.seed+j,audit))
            if tail:
                for k in range(FADE):
                    alpha=(k+1)/(FADE+1)
                    write(cv2.addWeighted(tail[k],1-alpha,generated[k],alpha,0))
            first=FADE if j else 0
            last=len(generated)-FADE if j<len(segments)-1 else len(generated)
            for f in generated[first:last]: write(f)
            tail=generated[-FADE:]
        proc.stdin.close()
        if proc.wait(): raise RuntimeError('ffmpeg failed')
    except BaseException:
        proc.kill(); proc.wait(); raise
    (ROOT/'object_bboxes.json').write_text(json.dumps(dict(roi=list(roi),alpha_threshold=.5,frames=audit),indent=2)+'\n')
    sizes=[r['longest_px'] for r in audit]
    print(f'Object bbox longest side: {min(sizes)}–{max(sizes)} px across {len(sizes)} object frames; see object_bboxes.json')
    print(f'Saved {outfile.name}: {count} frames, {count/FPS:.2f}s')
if __name__=='__main__': main()
