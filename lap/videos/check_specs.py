#!/usr/bin/env python3
"""Validate explainer specs (specs/**/*.json) against specs/README.md.
Prints one line per problem; 'OK n specs' when clean."""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = sys.argv[:1]
import render  # noqa: E402

BEATS = ["intro", "harm", "who", "dispose", "prevent"]
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9०-९௦-௯]")
probs, n = [], 0
for f in sorted(glob.glob(os.path.join(HERE, "specs", "**", "*.json"), recursive=True)):
    name = os.path.relpath(f, HERE); n += 1
    try:
        sp = json.load(open(f))
    except Exception as e:
        probs.append(f"{name}: bad JSON {e}"); continue
    exp = os.path.basename(f)[:-5]
    if f"{sp.get('cls')}__{sp.get('slug')}" != exp:
        probs.append(f"{name}: cls/slug don't match file name")
    keys = [sp.get("item")] + [sp.get("beats", {}).get("prevent", {}).get(k) for k in ("swap_from", "swap_to", "swap_to2")]
    for k in keys:
        if k and k not in render.ITEMS:
            probs.append(f"{name}: unknown item key '{k}'")
    for v in sp.get("beats", {}).get("who", {}).get("victims", []):
        if v not in render.VICTIMS:
            probs.append(f"{name}: unknown victim '{v}'")
    if sp.get("beats", {}).get("dispose", {}).get("bin") not in ("blue", "green", "red", "ewaste"):
        probs.append(f"{name}: bad bin")
    if not sp.get("sources"):
        probs.append(f"{name}: no sources")
    for b in BEATS:
        beat = sp.get("beats", {}).get(b, {})
        for l in ("en", "hi", "ta"):
            t = beat.get(l, "")
            if not t:
                probs.append(f"{name}: {b}.{l} missing"); continue
            if l != "en" and LATIN.search(t):
                probs.append(f"{name}: {b}.{l} has Latin letters: {LATIN.findall(t)[:5]}")
            if DIGIT.search(t):
                probs.append(f"{name}: {b}.{l} has digits (write numbers as words)")
        wc = len(beat.get("en", "").split())
        if not 10 <= wc <= 32:
            probs.append(f"{name}: {b}.en has {wc} words (want 15-25)")
print("\n".join(probs) if probs else f"OK {n} specs")
