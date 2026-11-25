#!/usr/bin/env python3
"""Search Wikimedia Commons for freely-licensed photos and download candidates + credits.
usage: commons_fetch.py OUTDIR TAG "query" [N]"""
import json, os, re, sys, urllib.error, urllib.parse, urllib.request
UA = {"User-Agent": "SortiEduVideo/1.0 (offline school project)"}
OK = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? [1-4]\.0)", re.I)
import time
def get(url):
    for i in range(6):
        try:
            time.sleep(2.0)
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(5 * (i + 1))
    raise RuntimeError("rate limited")
out, tag, q = sys.argv[1], sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
api = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(dict(
    action="query", format="json", generator="search", gsrsearch=f"{q} filetype:bitmap", gsrnamespace=6,
    gsrlimit=30, prop="imageinfo", iiprop="url|size|extmetadata|mime", iiurlwidth=1280))
pages = json.loads(get(api)).get("query", {}).get("pages", {})
creds = json.load(open(os.path.join(out, "credits.json"))) if os.path.exists(os.path.join(out, "credits.json")) else {}
k = 0
for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
    ii = p["imageinfo"][0]; md = ii.get("extmetadata", {})
    lic = md.get("LicenseShortName", {}).get("value", "")
    if ii.get("mime") != "image/jpeg" or ii["width"] < 1200 or ii["width"] < ii["height"] or not OK.match(lic):
        continue
    fn = f"{tag}_{k:02d}.jpg"
    open(os.path.join(out, fn), "wb").write(get(ii["thumburl"]))
    artist = re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", "")).strip()
    creds[fn] = dict(title=p["title"], page=ii["descriptionurl"], license=lic, artist=artist)
    k += 1
    if k >= n: break
json.dump(creds, open(os.path.join(out, "credits.json"), "w"), indent=1, ensure_ascii=False)
print(tag, q, "->", k)
