"""Run downloaded candidates sequentially; each child performs pgrep + flock checks."""
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'lap/stt_eval'
env=dict(os.environ); env.pop('PYTHONPATH',None)
env['OMP_NUM_THREADS']='4'; env['OPENBLAS_NUM_THREADS']='1'

def run(args,log):
    print('Running',args,flush=True)
    with (BASE/'results'/log).open('w') as f:
        p=subprocess.run([sys.executable,str(BASE/'eval.py'),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(f'{log}: exit {p.returncode}')
    print('Done',log,flush=True)

if __name__=='__main__':
    for lang in ('hi','ta'):
        run(['--rescore',f'lap/stt_eval/results/{lang}_stt_shruti_{lang}.json','--out',f'lap/stt_eval/results/{lang}_shruti_normalized.json'],f'{lang}_shruti_normalized.log')
    pending=[('hi','indicconformer-hi','hi_indicconformer',[]),
             ('ta','indicconformer-ta','ta_indicconformer',[]),
             ('ta','whisper-tamil-small-ct2','ta_whisper_small',['--raw-only'])]
    while pending:
        for entry in pending[:]:
            lang,name,stem,options=entry
            d=ROOT/'voice/_dl'/name
            if not (d/'download.json').exists(): continue
            out=BASE/f'results/{stem}.json'
            if out.exists():
                print('Already completed',stem,flush=True)
            else:
                run([lang,str(d),*options,'--out',str(out)],stem+'.log')
            pending.remove(entry)
            break
        else: time.sleep(5)
