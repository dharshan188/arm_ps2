#!/usr/bin/env python3
"""Hindi/Tamil: how often the answer bank answers with verified native text (no Llama, no NLLB answer translation).

    cd lap/llm_eval && env -u PYTHONPATH ~/ps2_et/bin/python run_hi_ta.py

Real path up to the bank: domain_snap on the native text -> NLLB question->English (the same Translator the server
uses) -> faq.match. Loads only NLLB (~1 GB), not Llama. Misses would go to Llama + NLLB back (not run here).
"""
import json, os, sys, time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import server as s  # noqa: E402

meta = json.load(open(os.path.join(s.VISION, "v2_taught", "labels.json")))
s.VIS = SimpleNamespace(meta=meta)
# translate/nllb_vmap_* holds symlinks to another machine (another user's home directory) here; the question direction
# (hi/ta -> en) never uses the vocabulary map, so load the plain model directory.
s.TR.vmap_dir = lambda lang: s.TR.DIR
en = {q["id"]: q for q in json.load(open(os.path.join(HERE, "questions.json")))}
final = {r["id"]: r for r in json.load(open(os.path.join(HERE, "results_final.json")))["rows"]}
rows = []
for lang in ("hi", "ta"):
    s.TR.unload()
    for q in json.load(open(os.path.join(HERE, "questions_hi_ta.json"))):
        text, scene = q[lang], en[q["id"]]["scene"]
        s.S = s.State(); s.S.chosen = True; s.S.lang = lang
        r = s.snap(text, lang)
        t0 = time.time()
        q_en = s.TR(text, lang, "en")
        t_tr = time.time() - t0
        item = r["item"] or (scene[0] if len(set(scene)) == 1 else None)
        e, score, direct = s.faq.match(q_en, item, scene)
        row = dict(id=q["id"], lang=lang, q=text, q_en=q_en, intent=r["intent"], item=r["item"], tr_s=round(t_tr, 2),
                   match=e["id"] if direct else None, score=round(score, 2),
                   same_as_en=(e["id"] if direct else None) == (final[q["id"]]["match"] if final[q["id"]]["source"] == "faq" else None),
                   answer=e[lang] if direct else None)
        rows.append(row)
        print(f"{lang} {q['id']:4} {r['intent']:10} {t_tr:4.2f}s {row['match']} {q_en!r}", flush=True)
s.TR.unload()
json.dump(rows, open(os.path.join(HERE, "results_hi_ta.json"), "w"), indent=1, ensure_ascii=False)
for lang in ("hi", "ta"):
    rs = [r for r in rows if r["lang"] == lang]
    routed = [r for r in rs if r["intent"] == "other"]
    print(lang, f"reach free-form {len(routed)}/{len(rs)}, bank hits {sum(bool(r['match']) for r in routed)}/{len(routed)}, "
          f"same entry as English {sum(r['same_as_en'] and bool(r['match']) for r in routed)}, "
          f"avg question NLLB {sum(r['tr_s'] for r in rs) / len(rs):.2f}s")
