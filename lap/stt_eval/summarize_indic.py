"""Insert tables from completed same-set results into INDIC_REPORT.md (no inference)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
FILES = [
    ('hi', 'Shruti', 'hi_shruti_normalized.json'),
    ('hi', 'IndicConformer', 'hi_indicconformer.json'),
    ('ta', 'Shruti', 'ta_shruti_normalized.json'),
    ('ta', 'IndicConformer', 'ta_indicconformer.json'),
    ('ta', 'Whisper-small CT2', 'ta_whisper_small.json'),
]

def main():
    results=[]
    for lang, name, filename in FILES:
        p=BASE/'results'/filename
        if not p.exists():
            raise FileNotFoundError(f'Wait for the full evaluation: {p}')
        j=json.loads(p.read_text())
        assert len(j['utterances']) == (240 if lang=='hi' else 480), filename
        results.append((lang,name,j))
    pct=lambda s,k: f'{100*s[k]:.1f}'
    summary=lambda j,c='clean',v='asr',a='raw': next(s for s in j['summary'] if s['condition']==c and s['audio']==a and s['text']==v)
    out=['## Same-set results','',
         'All accuracy columns are percentages. Clean raw audio, unmodified ASR text:', '',
         '| Language | Model | n | WER | Normalized WER | CER | Keyword recall | Snap | Mean s | p95 s | Peak MiB |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for lang,name,j in results:
        s=summary(j)
        out.append(f"| {lang} | {name} | {s['n']} | "+' | '.join(pct(s,k) for k in ['wer','normalized_wer','cer','keyword_accuracy','snap_accuracy'])+f" | {s['latency_mean_s']:.3f} | {s['latency_p95_s']:.3f} | {j['peak_rss_mib']:.1f} |")
    out += ['', 'The same clean raw hypotheses with the **existing** text corrector:', '',
            '| Language | Model | WER | Normalized WER | CER | Keyword recall | Snap |',
            '|---|---|---:|---:|---:|---:|---:|']
    for lang,name,j in results:
        s=summary(j,v='fixed')
        out.append(f'| {lang} | {name} | '+' | '.join(pct(s,k) for k in ['wer','normalized_wer','cer','keyword_accuracy','snap_accuracy'])+' |')
    out += ['', 'Noisy raw audio, unmodified ASR text:', '',
            '| Language | Model | Noise | n | WER | Normalized WER | CER | Keyword recall | Snap | Mean s |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for lang,name,j in results:
        for c in ['snr15','snr5']:
            s=summary(j,c=c)
            out.append(f"| {lang} | {name} | {c} | {s['n']} | "+' | '.join(pct(s,k) for k in ['wer','normalized_wer','cer','keyword_accuracy','snap_accuracy'])+f" | {s['latency_mean_s']:.3f} |")
    out += ['', 'Latency on actual clean clips lasting 2.5–3.5 seconds:', '',
            '| Language | Model | Clips | Measured laptop mean s | Pi estimate at 2–3× s | Load s |',
            '|---|---|---:|---:|---:|---:|']
    for lang,name,j in results:
        s=summary(j); t=s['near_3s_mean_s']
        out.append(f"| {lang} | {name} | {s['near_3s_n']} | {t:.3f} | {2*t:.2f}–{3*t:.2f} | {j['load_seconds']:.3f} |")
    out += ['', 'Audio preprocessing comparison for IndicConformer (ASR text, all scores in %):', '',
            '| Language | Condition | Raw WER | Pre WER | Raw keyword | Pre keyword | Raw snap | Pre snap |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for lang,name,j in results:
        if name!='IndicConformer':continue
        for c in ['clean','snr15','snr5']:
            a=summary(j,c=c);b=summary(j,c=c,a='pre')
            out.append(f'| {lang} | {c} | '+' | '.join(pct(s,k) for k in ['wer','keyword_accuracy','snap_accuracy'] for s in [a,b])+' |')
    out += ['', 'Whisper was evaluated on all 480 Tamil clips using raw audio only. IndicConformer',
            'was evaluated on raw **and** preprocessed audio (240 Hindi / 480 Tamil clips).',
            'All candidate runs started fresh (`resumed_utterances=0`). The baseline RSS/timings',
            'are historical saved measurements; baseline hypotheses are rescored without re-decoding.', '',
            'Result files: '+', '.join(f'[`{f}`](results/{f})' for _,_,f in FILES)+'.', '']
    block='\n'.join(out)
    p=BASE/'INDIC_REPORT.md'; s=p.read_text()
    begin='<!-- generated-results:start -->';end='<!-- generated-results:end -->'
    if begin in s:
        s=s[:s.index(begin)]+begin+'\n'+block+end+s[s.index(end)+len(end):]
    else:
        s=s.replace('## Measurement method',begin+'\n'+block+end+'\n\n## Measurement method')
    p.write_text(s)
    print(block)

if __name__=='__main__':main()
