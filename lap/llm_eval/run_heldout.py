#!/usr/bin/env python3
"""Held-out checks for the answer bank matcher and the off-topic guard (no models; < 1 s).

    cd lap/llm_eval && env -u PYTHONPATH ~/ps2_et/bin/python run_heldout.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(os.path.dirname(HERE)), "voice")]
import faq  # noqa: E402
from domain_snap import snap  # noqa: E402

h = json.load(open(os.path.join(HERE, "heldout_match.json")))
rows = []
for m in h["match"]:
    r = snap(m["q"], "en")
    sc = m["scene"]
    item = r["item"] or (sc[0] if len(set(sc)) == 1 else None)
    e, score, direct = faq.match(m["q"], item, sc)
    got = e["id"] if direct else None
    rows.append(dict(q=m["q"], intent=r["intent"], expect=m["expect"], got=got, score=round(score, 2), ok=got == m["expect"]))
pos = [x for x in rows if x["expect"]]
neg = [x for x in rows if not x["expect"]]
hits = [x for x in rows if x["got"]]
summary = dict(
    n=len(rows), correct=sum(x["ok"] for x in rows),
    should_answer=len(pos), answered_right=sum(x["got"] == x["expect"] for x in pos),
    answered_wrong_entry=sum(bool(x["got"]) and x["got"] != x["expect"] for x in rows),
    should_not_answer=len(neg), false_answers=sum(bool(x["got"]) for x in neg),
    precision=round(sum(x["ok"] for x in hits) / max(1, len(hits)), 3),
    offtopic_legit_n=len(h["offtopic_legit"]),
    offtopic_false_declines=[q for q in h["offtopic_legit"] if faq.off_topic(q, True, False)],
    offtopic_bad_n=len(h["offtopic_bad"]),
    offtopic_caught=sum(faq.off_topic(q, True, False) for q in h["offtopic_bad"]),
    offtopic_caught_no_scene=sum(faq.off_topic(q, False, False) for q in h["offtopic_bad"]),
)
for x in rows:
    print("OK " if x["ok"] else "BAD", f"{x['score']:.2f}", x["got"], "expected", x["expect"], "|", x["q"])
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, "results_heldout.json"), "w"), indent=1)
