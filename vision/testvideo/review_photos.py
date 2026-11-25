#!/usr/bin/env python3
"""Make labelled contact sheets for source/crop review (never used in the video)."""
from pathlib import Path
import json
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
records=json.loads((ROOT/'credits.json').read_text())
for cropped in (False,True):
    sheet=Image.new('RGB',(1000,180*((len(records)+4)//5)),'white'); d=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        im=Image.open(ROOT/r['file']).convert('RGB')
        if cropped:
            l,t,rr,b=r.get('crop',[0,0,1,1]);w,h=im.size
            im=im.crop((int(l*w),int(t*h),int(rr*w),int(b*h)))
        im.thumbnail((190,145));x=i%5*200;y=i//5*180
        sheet.paste(im,(x+(190-im.width)//2,y));d.text((x,y+148),Path(r['file']).stem,fill='black')
    sheet.save(ROOT/('crop_review.jpg' if cropped else 'photo_review.jpg'),quality=92)
