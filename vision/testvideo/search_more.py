from fetch_photos import get,ROOT
import json
for name,query in [('paper_stack','stack newspapers'),('duster','feather duster'),('dishcloth','dishcloth'),('glass_extra','empty glass bottle')]:
 p=ROOT/'api_cache'/f'{name}.json'
 if not p.exists():
  data=get('https://commons.wikimedia.org/w/api.php',params=dict(action='query',generator='search',gsrsearch=query+' filetype:bitmap -drawing -diagram -icon',gsrnamespace=6,gsrlimit=20,prop='imageinfo',iiprop='url|extmetadata|sha1',iiurlwidth=640,format='json')).json()
  p.write_text(json.dumps(data,indent=2))
 print('\n'+name,flush=True)
 for r in sorted(json.loads(p.read_text()).get('query',{}).get('pages',{}).values(),key=lambda x:x.get('index',0)):
  m=r['imageinfo'][0].get('extmetadata',{});print(r['title'],m.get('LicenseShortName',{}).get('value'),flush=True)
