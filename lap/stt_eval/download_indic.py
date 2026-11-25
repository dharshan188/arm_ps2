"""Pinned, size-limited public model downloads; no Hugging Face cache duplication."""
import hashlib
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
HTTP = requests.Session()
HTTP.mount('https://', HTTPAdapter(max_retries=Retry(
    total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

def download(repo, revision, files, name):
    dest = ROOT / 'voice/_dl' / name
    dest.mkdir(parents=True, exist_ok=True)
    meta = HTTP.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=30)
    meta.raise_for_status()
    sizes = {s['rfilename']: s.get('size', 0) for s in meta.json()['siblings']}
    expected_hashes = {s['rfilename']: s.get('lfs', {}).get('sha256') for s in meta.json()['siblings']}
    assert sum(sizes[f] for f in files) < 2_000_000_000, 'Download exceeds 2 GB'
    manifest = dict(repo=repo, revision=revision, files=[])
    for remote, local in files.items():
        path = dest / local
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{remote}'
        if not path.exists() or path.stat().st_size != sizes[remote]:
            print('Downloading', url, sizes[remote], flush=True)
            r = HTTP.get(url, stream=True, timeout=(30, 120))
            r.raise_for_status()
            part = path.with_suffix(path.suffix + '.part')
            with part.open('wb') as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk)
            assert part.stat().st_size == sizes[remote]
            part.rename(path)
        sha=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                sha.update(chunk)
        digest=sha.hexdigest()
        if expected_hashes.get(remote) and digest != expected_hashes[remote]:
            raise ValueError(f'SHA-256 mismatch for {remote}')
        manifest['files'].append(dict(file=local, source=url, bytes=path.stat().st_size, sha256=digest))
    (dest/'download.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(name, 'complete', flush=True)

if __name__ == '__main__':
    for lang in ('hi', 'ta'):
        download('parismitaglobalsolutions/indicconformer-sherpa-onnx',
                 '9721eb71eea141fae0982cfcdb9dd2e3d4953c4a',
                 {f'{lang}/model.int8.onnx':'model.int8.onnx', 'tokens.txt':'tokens.txt', 'README.md':'MODEL_CARD.md'},
                 f'indicconformer-{lang}')
    download('ukta-app/whisper-tamil-ct2','fd6de42af4f590e063ec0fc72bc9bfd6c13a4595',
             {f:f for f in ['model.bin','config.json','tokenizer.json','preprocessor_config.json',
                            'vocabulary.json','README.md']}, 'whisper-tamil-small-ct2')
