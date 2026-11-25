#!/usr/bin/env python3
"""Seeded-noise Piper set: 40 questions/language, 2 rates, 3 mic conditions.
Run: env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/make_set.py
Voices are loaded serially; cached clean WAVs allow resumable generation.
"""
import argparse
import gc
import io
import json
from pathlib import Path
import wave
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly
from questions import QUESTIONS

ROOT = Path(__file__).resolve().parents[2]
VOICES = {'en':['en'], 'hi':['hi'], 'ta':['ta_female','ta_male']}
FILES = {'en':'en_US-lessac-medium', 'hi':'hi_IN-pratham-medium', 'ta_female':'ta_IN-rasa_female-medium', 'ta_male':'ta_IN-rasa_male-medium'}


def rms(x):
    return np.sqrt(np.mean(x.astype(np.float64)**2)) + 1e-12


def microphone(x):
    # Low-pass + two quiet room reflections; applied to noise conditions only.
    y = sosfilt(butter(2, 4800, fs=16000, output='sos'), x)
    for delay, gain in [(347,.13),(731,.07)]:
        y[delay:] += gain*x[:-delay]
    return y


def noise_for(x, others, rng):
    n = len(x)
    white = rng.standard_normal(n)
    freq = np.fft.rfftfreq(n)
    pink = np.fft.irfft(np.fft.rfft(white)/np.sqrt(np.maximum(freq,1/n)),n)
    babble = np.zeros(n)
    for other in others:
        offset = int(rng.integers(len(other)))
        babble += np.resize(np.roll(other,offset),n)/rms(other)
    return .5*pink/rms(pink) + .5*babble/rms(babble)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path(__file__).parent/'data')
    ap.add_argument('--langs',nargs='+',default=['en','hi','ta'],choices=['en','hi','ta'])
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    from piper import PiperVoice, SynthesisConfig
    rows=[]
    for lang in args.langs:
        for voice in VOICES[lang]:
            v=None; clean=[]; meta=[]
            for q in QUESTIONS[lang]:
                for scale in (.85,1.15):
                    uid=f"{q['question_id']}_{voice}_{scale}"
                    path=args.out/f'{uid}_clean.wav'
                    if not path.exists():
                        if v is None:
                            v=PiperVoice.load(ROOT/'voice'/f'tts_piper_{voice}'/(FILES[voice]+'.onnx'))
                        buf=io.BytesIO()
                        with wave.open(buf,'wb') as w:
                            v.synthesize_wav(q['text'],w,SynthesisConfig(length_scale=scale))
                        buf.seek(0); x,sr=sf.read(buf,dtype='float32')
                        import math
                        divisor=math.gcd(sr,16000)
                        x=resample_poly(x,16000//divisor,sr//divisor)
                        sf.write(path,x,16000,subtype='PCM_16')
                    x,sr=sf.read(path,dtype='float32'); assert sr==16000
                    clean.append(x); meta.append(dict(q,lang=lang,voice=voice,length_scale=scale,id=uid))
            # Free the voice before augmentation or next voice.
            del v; gc.collect()
            for i,(x,m) in enumerate(zip(clean,meta)):
                rows.append(dict(m,condition='clean',wav=f"{m['id']}_clean.wav",duration=len(x)/16000))
                rng=np.random.default_rng(20261117+i+{'en':0,'hi':1000,'ta_female':2000,'ta_male':3000}[voice])
                # Different questions, not the alternate rate of this same question.
                choices=[j for j in range(len(clean)) if j//2 != i//2]
                others=[clean[j] for j in rng.choice(choices,4,replace=False)]
                speech=microphone(x); noise=microphone(noise_for(x,others,rng))
                for snr in (15,5):
                    y=speech+noise*(rms(speech)/rms(noise)*10**(-snr/20))
                    # Common gain on speech and noise preserves calibrated SNR.
                    y *= min(1.,.98/(np.max(np.abs(y))+1e-12))
                    name=f"{m['id']}_snr{snr}.wav"
                    sf.write(args.out/name,y,16000,subtype='PCM_16')
                    rows.append(dict(m,condition=f'snr{snr}',wav=name,duration=len(y)/16000,snr_db=snr))
            print(f'{voice}: {len(clean)*3} WAVs',flush=True)
            del clean; gc.collect()
    with (args.out/'refs.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(f'{len(rows)} refs -> {args.out}',flush=True)

if __name__=='__main__': main()
