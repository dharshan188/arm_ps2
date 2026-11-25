#!/usr/bin/env python3
"""Evaluate one offline CPU STT at a time (4 threads by default).

  env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/eval.py en voice/stt_whisper_base_en
  ... eval.py hi voice/stt_shruti_hi
  ... eval.py --rescore results/NAME.json  # reuse saved hypotheses after text-rule changes

Default: every WAV, raw and preprocessed, corrected and uncorrected; no TTS loaded.
"""
import argparse
import fcntl
import hashlib
import json
import re
import resource
import statistics
import sys
import os
import subprocess
import time
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'lap'),str(ROOT/'voice')]
from stt_fix import correct, preprocess
from domain_snap import snap

# Exact lexical concept survival (not fuzzy matching, and never correct the ref).
ALIASES = {
'waste': ['waste','वेस्ट','कचरा','வேஸ்ட்','குப்பை'],
'bin':['bin','dustbin','बिन','डिब्ब','பின்','தொட்டி'],
'wet':['wet','वेट','गील','வெட்','ஈர'], 'dry':['dry','ड्राई','ட்ரை'],
'recycle':['recycle','recyclable','recycling','रीसायकल','रिसायकल','रीसाइकिल','ரீசைக்கிள்'],
'plastic':['plastic','plastics','प्लास्टिक','பிளாஸ்டிக்'],
'battery':['battery','batteries','बैटर','பேட்டரி'],
'bulb':['bulb','बल्ब','பல்ப'],
'e-waste':['e-waste','e waste','ewaste','ई-वेस्ट','ई वेस्ट','ஈ வேஸ்ட்','ஈ வேஸ்ட'],
'compost':['compost','कम्पोस्ट','कंपोस्ट','கம்போஸ்ட்'],
'glass':['glass','कांच','काँच','கண்ணாடி'],
'metal':['metal','मेटल','धातु','மெட்டல்'],
'paper':['paper','पेपर','कागज','कागज़','பேப்பர்','காகித']}


def norm(s):
    s=unicodedata.normalize('NFC',s).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c)[0] in 'PS' else c for c in s).split())


# Symmetric text normalization only: never infer an intent, repair an ASR error,
# or collapse distinct waste concepts. Keep strict WER/CER beside these scores.
SPELLINGS = {
    'hi': {
        'रीसाइकिल': 'रीसायकल', 'रीसाइकल': 'रीसायकल', 'रिसायकल': 'रीसायकल',
        'रीसाइक्ल': 'रीसायकल', 'रिसाइकल': 'रीसायकल',
        'कंपोस्ट': 'कम्पोस्ट', 'काँच': 'कांच', 'कहां': 'कहाँ',
        'डालूं': 'डालूँ', 'दूं': 'दूँ', 'बैटरियां': 'बैटरियाँ',
        'यह': 'ये', 'वह': 'वो', 'प्लीज': 'प्लीज़',
    },
    'ta': {
        'எங்கே': 'எங்க', 'எங்கு': 'எங்க', 'போடவேண்டும்': 'போடணும்',
        'போடவேணும்': 'போடணும்', 'கொடுக்கணும்': 'குடுக்கணும்',
        'கொடுக்கவேண்டும்': 'குடுக்கணும்', 'கொடுக்கலாம்': 'குடுக்கலாம்',
        'வீட்டில்': 'வீட்டுல', 'உடைந்த': 'உடைஞ்ச', 'இருக்கிறது': 'இருக்கு',
        'பின்னில்': 'பின்ல', 'பினில்': 'பின்ல', 'பின்ல': 'பின்ல',
        'செய்யலாமா': 'பண்ணலாமா', 'செய்வது': 'பண்றது',
        'முன்னால்': 'முன்னாடி', 'தடவை': 'தடவ', 'சொல்லுங்கள்': 'சொல்லுங்க',
        'புளூ': 'ப்ளூ', 'ப்ளு': 'ப்ளூ', 'ரிசைக்கிள்': 'ரீசைக்கிள்',
        'ரீசைகிள்': 'ரீசைக்கிள்', 'ரீசைக்கில்': 'ரீசைக்கிள்',
        'கம்போஸ்டு': 'கம்போஸ்ட்', 'பிளாஸ்டிக்கு': 'பிளாஸ்டிக்',
    },
}
PHRASES = {'ta': {'போட வேண்டும்': 'போடணும்', 'போட வேணும்': 'போடணும்',
                   'கொடுக்க வேண்டும்': 'குடுக்கணும்', 'என்னுடைய': 'என்னோட'}}


def normalized_text(s, lang):
    s = norm(s)
    for a,b in PHRASES.get(lang, {}).items():
        s = re.sub(r'(?<!\S)'+re.escape(a)+r'(?!\S)', b, s)
    if lang == 'hi':
        # Nukta often varies in borrowed words; retain all vowel/nasal marks.
        s = unicodedata.normalize('NFC', unicodedata.normalize('NFD', s).replace('\u093c', ''))
    mapping = {norm(k): norm(v) for k, v in SPELLINGS.get(lang, {}).items()}
    if lang == 'hi':
        mapping = {unicodedata.normalize('NFD', k).replace('\u093c', ''):
                   unicodedata.normalize('NFD', v).replace('\u093c', '') for k,v in mapping.items()}
    return ' '.join(mapping.get(w, w) for w in s.split())


def keywords(s):
    s=norm(s); found=set()
    for concept,aliases in ALIASES.items():
        for alias in aliases:
            a=norm(alias)
            # Native suffixes are allowed; Latin aliases must match complete words.
            pattern=r'(?<!\w)'+re.escape(a)+(r'(?!\w)' if a.isascii() else '')
            if re.search(pattern,s): found.add(concept); break
    return found


def distance(a,b):
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a,b)


def recognizer(lang,d,backend='auto',threads=4):
    d=Path(d)
    if backend == 'auto':
        backend = 'faster-whisper' if (d/'model.bin').exists() else 'sherpa'
    if backend == 'faster-whisper':
        # Workspace-local wheels keep the shared venv and pinned torch untouched.
        sys.path.insert(0, str(Path(__file__).parent/'_deps'))
        from faster_whisper import WhisperModel
        return WhisperModel(str(d), device='cpu', compute_type='int8', cpu_threads=threads,
                            num_workers=1, local_files_only=True)
    import sherpa_onnx
    if (d/'model.int8.onnx').exists():
        return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(d/'model.int8.onnx'),tokens=str(d/'tokens.txt'),num_threads=threads,
            decoding_method='greedy_search',provider='cpu')
    if list(d.glob('*encoder*.onnx')):
        enc=sorted(d.glob('*encoder*.int8.onnx'))
        dec=sorted(d.glob('*decoder*.int8.onnx'))
        tok=sorted(d.glob('*tokens.txt'))
        if not enc or not dec or not tok: raise ValueError('Whisper requires int8 encoder/decoder and tokens')
        return sherpa_onnx.OfflineRecognizer.from_whisper(encoder=str(enc[0]),decoder=str(dec[0]),tokens=str(tok[0]),language=lang,task='transcribe',num_threads=threads)
    # Shruti names match Speech.load exactly; upstream English moonshine uses *_int8.
    def pick(*names):
        for name in names:
            if (d/name).exists(): return str(d/name)
        raise FileNotFoundError(f'{d}: expected one of {names}')
    return sherpa_onnx.OfflineRecognizer.from_moonshine(
        preprocessor=pick('preprocess.onnx'),encoder=pick('encode.int8.onnx','encode.onnx'),
        uncached_decoder=pick('uncached_decode.int8.onnx','uncached_decode.onnx'),
        cached_decoder=pick('cached_decode.int8.onnx','cached_decode.onnx'),tokens=pick('tokens.txt'),num_threads=threads)


def transcribe(rec, audio, lang, backend):
    if backend == 'faster-whisper':
        segments, _ = rec.transcribe(audio, language=lang, task='transcribe', beam_size=1,
            best_of=1, temperature=0, condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True)
        return ''.join(s.text for s in segments).strip()  # generator must be consumed inside timer
    stream=rec.create_stream(); stream.accept_waveform(16000,audio); rec.decode_stream(stream)
    return stream.result.text.strip()


def evaluation_lock():
    result_dir=Path(__file__).parent/'results'
    result_dir.mkdir(parents=True,exist_ok=True)
    lock=open(result_dir/'.eval.lock','a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    # pgrep also catches evaluators launched by other jobs before this lock existed.
    while True:
        found = subprocess.run(['pgrep', '-f', 'stt_eval/eval.py'], capture_output=True, text=True)
        active=[]
        for pid in found.stdout.split():
            if int(pid) == os.getpid(): continue
            try:
                argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
                # Newer evaluators are waiting on our flock, so don't wait back.
                mine=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
                theirs=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                if int(theirs) <= int(mine) and 'python' in Path(argv[0]).name and any(a.endswith('stt_eval/eval.py') for a in argv[1:]):
                    active.append(pid)
            except (OSError, UnicodeError): pass
        if not active: break
        print('Waiting for other eval.py PID(s):', ', '.join(active), flush=True)
        time.sleep(10)
    return lock


def score(result):
    cached_snap = lru_cache(maxsize=10000)(snap)
    groups=defaultdict(list)
    correct_times=[]
    # Ref snap is reported as a ceiling diagnostic, never used as semantic gold.
    for row in result['utterances']:
        for mode in ('raw','pre'):
            if mode not in row: continue
            hyp=row[mode]['text']; t=time.perf_counter(); fixed=correct(hyp,row['lang']); correct_times.append(time.perf_counter()-t)
            row[mode]['corrected']=fixed
            for variant,text in [('asr',hyp),('fixed',fixed)]:
                ref=norm(row['text']); h=norm(text)
                nr=normalized_text(row['text'],row['lang']); nh=normalized_text(text,row['lang'])
                expected=keywords(row['text']); survived=expected & keywords(text)
                parsed=cached_snap(text,row['lang'])
                row[mode][variant+'_snap']=parsed
                gold=(row['intent'],row['item'])
                item=(parsed['intent'],parsed['item'])
                groups[(row['condition'],mode,variant)].append(dict(
                    we=distance(ref.split(),h.split()),nw=len(ref.split()),ce=distance(ref.replace(' ',''),h.replace(' ','')),nc=len(ref.replace(' ','')),
                    nwe=distance(nr.split(),nh.split()),nnw=len(nr.split()),
                    kw=len(survived),nkw=len(expected),semantic=item==gold,seconds=row[mode]['seconds'],duration=row.get('duration')))
    stats=[]
    for (condition,mode,variant),rs in sorted(groups.items()):
        total=lambda k:sum(r[k] for r in rs)
        stats.append(dict(condition=condition,audio=mode,text=variant,n=len(rs),wer=total('we')/max(1,total('nw')),cer=total('ce')/max(1,total('nc')),
                          normalized_wer=total('nwe')/max(1,total('nnw')),
                          keyword_accuracy=total('kw')/max(1,total('nkw')),snap_accuracy=total('semantic')/len(rs),latency_mean_s=statistics.mean(r['seconds'] for r in rs),latency_p95_s=sorted(r['seconds'] for r in rs)[int(.95*(len(rs)-1))]))
        valid=[r for r in rs if r['duration']]
        if valid:
            stats[-1]['rtf']=sum(r['seconds'] for r in valid)/sum(r['duration'] for r in valid)
            near3=[r['seconds'] for r in valid if 2.5 <= r['duration'] <= 3.5]
            stats[-1]['near_3s_n']=len(near3)
            stats[-1]['near_3s_mean_s']=statistics.mean(near3) if near3 else None
    result['summary']=stats
    result['corrector_sha256']=hashlib.sha256((ROOT/'lap/stt_fix.py').read_bytes()).hexdigest()
    refs={(r['question_id'],r['lang'],r['text'],r['intent'],r['item']) for r in result['utterances']}
    result['reference_snap_accuracy']=sum((p:=cached_snap(t,l))['intent']==intent and p['item']==item for _,l,t,intent,item in refs)/max(1,len(refs))
    result['corrector_timing_ms']={'mean':statistics.mean(correct_times)*1000,
                                 'p95':sorted(correct_times)[int(.95*(len(correct_times)-1))]*1000,
                                 'max':max(correct_times)*1000}
    result['normalization_rules']=SPELLINGS
    result['normalization_phrases']=PHRASES
    result['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('condition audio text     n    WER    CER   nWER keyword   snap  mean_s  p95_s')
    for r in stats:
        print(f"{r['condition']:9} {r['audio']:5} {r['text']:5} {r['n']:4} "+' '.join(f'{100*r[k]:6.1f}' for k in ('wer','cer','normalized_wer','keyword_accuracy','snap_accuracy'))+f" {r['latency_mean_s']:7.3f} {r['latency_p95_s']:6.3f}")
    print('Peak RSS MiB:',result.get('peak_rss_mib'),'text-only gold snap:',result['reference_snap_accuracy'],'corrector ms:',result['corrector_timing_ms'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('lang',nargs='?',choices=['en','hi','ta']); ap.add_argument('model',nargs='?',type=Path)
    ap.add_argument('--refs',type=Path,default=Path(__file__).parent/'data/refs.jsonl')
    ap.add_argument('--out',type=Path); ap.add_argument('--rescore',type=Path)
    ap.add_argument('--limit',type=int); ap.add_argument('--raw-only',action='store_true')
    ap.add_argument('--backend',choices=['auto','sherpa','faster-whisper'],default='auto')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--condition',choices=['clean','snr15','snr5'])
    args=ap.parse_args()
    lock=evaluation_lock()
    if args.rescore:
        result=json.loads(args.rescore.read_text()); score(result); (args.out or args.rescore).write_text(json.dumps(result,ensure_ascii=False,indent=2)); return
    if not args.lang or not args.model: ap.error('lang and model are required')
    import soundfile as sf
    rows=[json.loads(s) for s in args.refs.read_text().splitlines() if s.strip()]
    rows=[r for r in rows if r['question_id'].startswith(args.lang+'_')]
    if args.condition: rows=[r for r in rows if r['condition']==args.condition]
    if args.limit: rows=rows[:args.limit]
    out=args.out or Path(__file__).parent/'results'/f'{args.lang}_{args.model.name}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    backend=args.backend if args.backend!='auto' else ('faster-whisper' if (args.model/'model.bin').exists() else 'sherpa')
    result=dict(lang=args.lang,model=str(args.model),backend=backend,num_threads=args.threads,utterances=[],refs=str(args.refs),source='synthetic Piper; not real microphone recordings')
    result['refs_sha256']=hashlib.sha256(args.refs.read_bytes()).hexdigest()
    checkpoint=out.with_suffix('.jsonl')
    config_path=out.with_suffix('.config.json')
    manifest=args.model/'download.json'
    config=dict(lang=args.lang, model=str(args.model.resolve()), backend=backend,
                threads=args.threads, refs_sha256=result['refs_sha256'],
                model_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None)
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Checkpoint config differs; choose a new --out path for this run')
    if checkpoint.exists() and not config_path.exists():
        raise ValueError('Legacy checkpoint: use --rescore or choose a new --out path')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    result['config']=config
    done={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            r=json.loads(line); done[r['wav']]=r
    start=time.perf_counter(); rec=recognizer(args.lang,args.model,backend,args.threads); result['load_seconds']=time.perf_counter()-start
    result['resumed_utterances']=len(done)
    with checkpoint.open('a') as f:
        for i,row in enumerate(rows):
            if row['wav'] in done and (args.raw_only or 'pre' in done[row['wav']]):
                result['utterances'].append(done[row['wav']]); continue
            x,sr=sf.read(args.refs.parent/row['wav'],dtype='float32'); assert sr==16000 and x.ndim==1
            row=dict(row,lang=args.lang)
            for mode in (('raw',) if args.raw_only else ('raw','pre')):
                t=time.perf_counter(); audio=preprocess(x) if mode=='pre' else x
                text=transcribe(rec,audio,args.lang,backend)
                row[mode]=dict(text=text,seconds=time.perf_counter()-t)
            result['utterances'].append(row); f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
            if (i+1)%10==0: print(f'{i+1}/{len(rows)}',flush=True)
    result['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    score(result); out.write_text(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
