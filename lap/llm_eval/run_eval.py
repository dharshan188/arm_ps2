#!/usr/bin/env python3
"""Run the free-form eval questions through Sorti's real answer path (English) and save the answers.

    cd lap/llm_eval && env -u PYTHONPATH ~/ps2_et/bin/python run_eval.py --mode final

--mode baseline  the pre-change path: llama_context + "Visitor: ... Sorti:" + LL.ask(max_new_tokens=60)
--mode final     server.free_form_en(): answer bank -> grounded Llama -> quality/safety guard
--mode prompt    new prompt/few-shots/early stop only (bank, off-topic guard and answer guard off)
--mode ground    prompt + bank facts given to Llama as a "Fact:" line
--mode llama     final path with the bank's direct answers switched off (Llama + grounding + both guards)

Loads ONE Llama worker (~2.2 GB). Check `free -m` first; never run two evals at once.
"""
import argparse, json, os, sys, time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
LAP = os.path.dirname(HERE)
sys.path.insert(0, LAP)
import server as s  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["baseline", "prompt", "ground", "llama", "final"], required=True)
ap.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
ap.add_argument("--out")
ap.add_argument("--only", nargs="*")
a = ap.parse_args()
out_path = a.out or os.path.join(HERE, f"results_{a.mode}.json")

meta = json.load(open(os.path.join(s.VISION, "v2_taught", "labels.json")))
s.VIS = SimpleNamespace(meta=meta)
smap = meta["stream_map"]
from pytorch_tokenizers import get_tokenizer  # noqa: E402
tok = get_tokenizer(os.path.join(s.LLAMA, "checkpoint", "tokenizer.model"))
ntok = lambda t: len(tok.encode(t, bos=False, eos=False))

FLAGS = {"prompt": dict(FAQ_DIRECT=False, FAQ_GROUND=False, OFF_TOPIC=False, ANSWER_GUARD=False),
         "ground": dict(FAQ_DIRECT=False, FAQ_GROUND=True, OFF_TOPIC=False, ANSWER_GUARD=False),
         "llama": dict(FAQ_DIRECT=False, FAQ_GROUND=True, OFF_TOPIC=True, ANSWER_GUARD=True)}
for k, v in FLAGS.get(a.mode, {}).items():
    setattr(s, k, v)
qs = json.load(open(a.questions))
if a.only:
    qs = [q for q in qs if q["id"] in a.only]

rows = []
for q in qs:
    s.S = s.State(); s.S.chosen = True; s.S.lang = "en"
    scene = [dict(cls=c, stream=smap[c], confirmed=True) for c in q["scene"]]
    s.S.objects = scene
    if q.get("prev"):
        s.S.add("user", "question", q["prev"]["q"])
        s.S.add("assistant", "answer", q["prev"]["a"], source="llama")
    um = s.S.add("user", "question", q["q"])
    r = s.snap(q["q"], "en")
    named = r["item"]
    # make sure the system prompt is primed (as it is after idle time in the kiosk) before timing
    w = s.LL.get()
    if hasattr(w, "prime"):
        w.prime(s.FRIEND); w._pending_prime = False; w._receive()
    t0 = time.time()
    if a.mode == "baseline":
        context = s.llama_context(scene, named, smap, um["id"])
        prompt = context + "\nVisitor: " + q["q"] + "\nSorti:"
        ans = s.LL.ask(prompt, max_new_tokens=60)
        res = dict(source="llama", text_en=ans, prompt=prompt)
    else:
        res = s.free_form_en(q["q"], scene, named, smap, um["id"])
    dt = time.time() - t0
    row = dict(id=q["id"], q=q["q"], scene=q["scene"], answer=res["text_en"], source=res["source"],
               latency_s=round(dt, 2), prompt_tokens=ntok(res.get("prompt") or ""),
               gen_tokens=ntok(res.get("raw", res["text_en"])) if res["source"] == "llama" else 0,
               raw=res.get("raw"), match=res.get("match"), guard=res.get("guard"))
    rows.append(row)
    print(f"{q['id']:4} {row['source']:8} {dt:5.2f}s  {row['answer']}", flush=True)

s.LL.unload()
json.dump(dict(mode=a.mode, questions=a.questions, rows=rows), open(out_path, "w"), indent=1, ensure_ascii=False)
print("saved", out_path)
