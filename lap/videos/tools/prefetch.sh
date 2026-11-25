#!/bin/bash
# serial Commons prefetch: lines "cls|tag|query|n"
F=~/arm2026/lap/videos/tools/commons_fetch.py
while IFS='|' read cls tag q n; do
  [ -z "$cls" ] && continue
  d=~/arm2026/lap/videos/photos/$cls/cand; mkdir -p $d
  ls $d/${tag}_00.jpg >/dev/null 2>&1 && continue
  timeout 150 python3 -I $F $d "$tag" "$q" $n 2>&1 | tail -1
done
