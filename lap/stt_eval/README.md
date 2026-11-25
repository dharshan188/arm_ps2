Run from the project root, always with `PYTHONPATH` unset:

```sh
env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/make_set.py
env -u PYTHONPATH ~/ps2_et/bin/python -m unittest discover -s lap/stt_eval -p 'test_*.py'
env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/eval.py en voice/stt_whisper_base_en
env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/eval.py hi voice/stt_shruti_hi
env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/eval.py ta voice/stt_shruti_ta
```

Run model commands sequentially. `--out FILE.json` selects the result path;
`--refs PATH/refs.jsonl` selects another set. Both raw and preprocessed audio are
measured by default; `--raw-only` and `--limit N` support smoke checks.
`--rescore FILE.json` updates correction and semantic scores from saved transcripts
without loading any model. A sibling `.jsonl` checkpoints each completed WAV;
remove it before a fresh timing run or after changing audio/model/decoder settings.

`questions.py` contains 40 manually labelled questions per language. Labels express
what the speaker asks, independently of what `domain_snap` currently recognises.
For example, pizza box means paper and “kitchen waste” means organic. Procedural
questions (“how to clean ... before recycling”) are `other`, not `recyclable`.

The set uses a fixed noise seed and contains English lessac, Hindi pratham, and Tamil rasa female
and male, length scales .85 and 1.15. WAVs are PCM16, mono, 16 kHz. Each clean WAV
has SNR 15 dB and 5 dB versions: equal-RMS pink noise and four other same-voice
questions form the noise, then a 4.8 kHz low-pass plus two quiet echoes colour the
speech and noise. SNR is calibrated by whole-utterance RMS after colouration;
a common final gain prevents clipping. These conditions simulate interference,
not a calibrated physical microphone or real human accents.

WER is total word edit distance / reference words; CER excludes whitespace.
Both use NFC, lowercase and punctuation removal, preserving native vowel marks.
Keyword accuracy is micro recall over reference domain concepts, with explicit
loanword/inflection aliases, no fuzzy scoring. Semantic accuracy requires the
exact manually labelled intent **and** item (including `None`). Ref-text snap
accuracy is reported to expose matcher limitations. Rates are fractions in JSON,
percentages in tables. Decode latency includes stream creation and preprocessing
where enabled, excluding model load and text scoring; first decode is included.
RSS is process peak resident memory in MiB (not model-file size or Pi memory).

Correction rules derived from this synthetic set are development results, not an
independent held-out estimate. Negative sentence tests protect common ordinary
uses; validate on real kiosk recordings before deciding production thresholds.

## Hindi / Tamil candidate comparison (2026-11-17)

See `INDIC_CANDIDATES.md` for model selection and `INDIC_REPORT.md` for results.
`download_indic.py` retrieves all three pinned public exports and saves exact
sizes and hashes in each model directory. `run_indic.py` runs baselines and
candidates sequentially, waiting for download manifests. Every `eval.py` invocation
runs `pgrep -f stt_eval/eval.py` and waits for an older evaluator, holding a file lock
to serialize new jobs. Shell and sandbox wrapper matches are excluded by checking
Python argv; only one model is loaded at a time.

Additional options:

```sh
env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/eval.py hi voice/_dl/indicconformer-hi --out lap/stt_eval/results/hi_indicconformer.json
env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/eval.py ta voice/_dl/indicconformer-ta --out lap/stt_eval/results/ta_indicconformer.json
env -u PYTHONPATH ~/ps2_et/bin/python lap/stt_eval/eval.py ta voice/_dl/whisper-tamil-small-ct2 --backend faster-whisper --raw-only --out lap/stt_eval/results/ta_whisper_small.json
```

`--threads N` changes CPU threads. Auto backend selection recognizes CT2's
`model.bin`, NeMo CTC's `model.int8.onnx`, or existing sherpa Whisper/Moonshine
layouts. Whisper language now follows the requested language instead of being
hard-coded to English. Faster-whisper uses local files, CPU int8, greedy beam 1,
zero temperature, no previous-text conditioning or VAD. Its segment generator is
fully consumed inside the latency measurement. Model download is never timed as
inference. Workspace-local `_deps` provides faster-whisper/PyAV without modifying
shared torch or other venv packages.

`wer` remains the original punctuation-insensitive metric. `normalized_wer` also
applies the explicit, symmetric `SPELLINGS` / `PHRASES` rules in `eval.py`, including
Hindi nukta spelling variation and common Tamil formal/casual equivalents.
It does not snap words to waste categories, infer an intent, remove negation, or
apply the hypothesis corrector to references. It is a development diagnostic;
inspect strict WER/CER and keyword/snap alongside it. Rules are serialized in each
result. WER token boundaries still matter: arbitrary Tamil compounds are not
silently joined/split. Rates are micro-aggregated over references.

Summary JSON also includes audio-duration-weighted RTF and mean latency on real
clips lasting 2.5–3.5 seconds (`near_3s_n`, `near_3s_mean_s`). These are measured
clip times, not 3-second latency extrapolated from the full-set mean. `peak_rss_mib`
is whole-process peak RSS on this x86 laptop, including load/first decode; no Pi
measurement is implied. Reusing checkpoints records `resumed_utterances`; it does
not produce a fresh timing experiment. A `.config.json` sidecar rejects changes to
backend, threads, reference hash, model path or download manifest. Use a new output
path for a fresh run. Old checkpoints can be rescored with `--rescore`.
