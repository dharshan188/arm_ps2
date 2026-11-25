#!/usr/bin/env python3
"""Serial Commons-only photo acquisition; restartable, with licence/provenance records."""
import hashlib, html, io, json, re, time
from pathlib import Path
import requests
from PIL import Image
ROOT = Path(__file__).resolve().parent
def clean(s): return html.unescape(re.sub('<[^>]*>', '', s or '')).strip()
session = requests.Session()
session.headers['User-Agent'] = 'SortiPhotoTest/1.0 (educational classifier evaluation; Commons attribution)'
def get(url, **kwargs):
    for attempt in range(8):
        try:
            time.sleep(1.5)
            r = session.get(url, timeout=(20, 90), **kwargs)
            if r.status_code in (429, 502, 503, 504):
                delay = max(float(r.headers.get('Retry-After', 0)), min(120, 5 * 2**attempt))
                print(f'HTTP {r.status_code}; retry in {delay}s', flush=True); time.sleep(delay); continue
            r.raise_for_status(); return r
        except requests.RequestException as e:
            print(f'Retry {attempt+1}: {e}', flush=True); time.sleep(min(60, 3*2**attempt))
    raise RuntimeError(f'Failed after retries: {url}')
def main():
    dest=ROOT/'photos'; dest.mkdir(exist_ok=True)
    cp=ROOT/'credits.json'
    old=json.loads(cp.read_text()) if cp.exists() else []
    selected=json.loads((ROOT/'selection.json').read_text())
    excluded=set()
    def collect(v):
        if isinstance(v,dict):
            for x in v.values(): collect(x)
        elif isinstance(v,list):
            for x in v: collect(x)
        elif isinstance(v,str): excluded.add(v)
    for p in (ROOT.parent/'extra').rglob('*.json'):
        collect(json.loads(p.read_text()))
    pages={}
    for p in (ROOT/'api_cache').glob('*.json'):
        for r in json.loads(p.read_text()).get('query',{}).get('pages',{}).values(): pages[r['title']]=r
    result=[]
    for sel in selected:
        title=sel['title']; cls=sel['class']
        if title not in pages:
            data=get('https://commons.wikimedia.org/w/api.php',params=dict(action='query',generator='search',gsrsearch='intitle:"'+title[5:]+'"',gsrnamespace=6,gsrlimit=10,prop='imageinfo',iiprop='url|extmetadata|sha1',iiurlwidth=640,format='json')).json()
            (ROOT/'api_cache'/('selected_'+Path(sel['file']).stem+'.json')).write_text(json.dumps(data,indent=2))
            for r in data.get('query',{}).get('pages',{}).values(): pages[r['title']]=r
        page=pages[title]; info=page['imageinfo'][0]; meta=info['extmetadata']
        val=lambda k: clean(meta.get(k,{}).get('value',''))
        licence=val('LicenseShortName')
        if not re.match(r'^(CC BY(?:-SA)?(?: |$)|CC0|Public domain)',licence): raise ValueError(f'Unapproved licence: {licence}')
        if title in excluded or info['descriptionurl'] in excluded: raise ValueError(f'Training source excluded: {title}')
        previous=next((r for r in old if r['file']==sel['file'] and r['title']==title),None)
        if previous and (ROOT/sel['file']).exists():
            record=dict(previous,crop=sel['crop'])
        else:
            print(f"Download {cls}: {title} [{licence}]",flush=True)
            response=get(info['thumburl'])
            im=Image.open(io.BytesIO(response.content)).convert('RGB'); im.thumbnail((640,640))
            im.save(ROOT/sel['file'],quality=94)
            record=dict(sel,author=val('Artist'),licence=licence,licence_url=val('LicenseUrl'),page=info['descriptionurl'],thumbnail_url=info['thumburl'],commons_sha1=info.get('sha1'),download_sha256=hashlib.sha256(response.content).hexdigest(),description=val('ImageDescription'))
        result.append(record)
        # Save progress with pending old rows retained for restartability.
        cp.write_text(json.dumps(result+[r for r in old if r['file'] not in {v['file'] for v in result}],indent=2,ensure_ascii=False)+'\n')
    cp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(f'Complete: {len(result)} curated Commons photos',flush=True)
if __name__=='__main__': main()
