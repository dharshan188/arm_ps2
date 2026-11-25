# Hindi / Tamil offline STT evaluation

Date: 2026-11-17. Evaluation host: Intel Core i3-1115G4 laptop, two physical cores /
four logical CPUs, 8 GB RAM, x86-64 Linux. Target: Raspberry Pi 5, four Cortex-A76
cores, 8 GB, CPU only. **Pi inference has not been measured.**

## Decision

Hindi: replace Shruti with the downloaded per-language IndicConformer 120M CTC,
MatMul-int8 sherpa-onnx export in `voice/_dl/indicconformer-hi`.
Tamil: evaluation in progress; results and decision will be filled after completion.

## Measurement method

Same `lap/stt_eval/data/refs.jsonl` and WAVs as the saved Shruti baseline: 40 question
texts per language; Hindi has one Piper voice × two rates (80 clean clips), Tamil
two voices × two rates (160 clean clips). Each clean clip also has SNR15 and SNR5
versions. Total: 240 Hindi / 480 Tamil clips. These are **synthetic development
examples, not human recordings or an independent test set**. Loanwords are written
in native script. The saved baseline texts and gold labels were checked against
the current reference file before comparison.

Raw and `stt_fix.preprocess` audio are decoded independently for IndicConformer.
`asr` scores unmodified hypotheses; `fixed` applies the existing `stt_fix.correct`.
No corrector, domain_snap, server, web or video changes were made for this work.
Baseline inference timings/RSS come from the existing saved runs; text metrics are
rescored with the same current scorer as the candidates. Therefore snap numbers
can differ from the older `hi.log` / `ta.log`.

Strict WER/CER retain the harness's NFC/case/punctuation treatment. Normalized WER
also uses explicit symmetric spelling rules, including Hindi nukta, recycle/compost
spellings and common Tamil formal/casual equivalents. It is not semantic scoring:
negation, distinct waste categories and arbitrary compound word boundaries remain.
Rules are embedded in result JSON and are visible in `eval.py`; both strict and
normalized metrics are reported. No references pass through the ASR corrector.
Keyword accuracy is micro recall of reference waste concepts with the existing
alias matcher. Snap requires exact gold intent **and** item; it is also limited by
the current matcher even for perfect transcripts.

Four CPU threads, one utterance at a time, no GPU, no TTS. Decode latency includes
stream creation/features and preprocessing where enabled; model load is separate.
First decode is included. RSS is whole-process Linux peak RSS, including load,
reported in MiB. Mean near-3-second time uses actual 2.5–3.5 s clips; the sample is
small (6 Hindi / 4 Tamil clean clips), so report its count. Pi times inferred using
the user's 2–3× factor are estimates, not benchmarks.

Before each evaluator, `pgrep -f stt_eval/eval.py` checks for older Python evaluators
and waits; a shared `results/.eval.lock` serializes the updated harness. All child
evaluations in this task run sequentially. This tool's process namespaces restrict
`pgrep` visibility to its own session, so a pre-existing external job in another
namespace cannot be ruled out by that check. The shared lock works across sessions
for jobs using this harness. Background downloads do not run model inference.

## Integration code

The following replaces only the current Hindi/Tamil Moonshine `else` branch inside
`Speech.load`, after the existing English branch. `VOICE`, `os`, `sherpa_onnx` and
`self.stt` are already defined by the server. The models stay in their downloaded
locations; no copy is needed. Load one selected language at a time.

```python
# Speech.load(lang), after the existing `if lang == "en": ...` branch:
elif lang in ("hi", "ta"):
    d = os.path.join(VOICE, "_dl", f"indicconformer-{lang}")
    self.stt = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=os.path.join(d, "model.int8.onnx"),
        tokens=os.path.join(d, "tokens.txt"),
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )
```

The existing sherpa transcription path works for both CTC and English Moonshine:

```python
def transcribe(self, audio):
    # audio: mono float32 at SR == 16000, nominally in [-1, 1]
    s = self.stt.create_stream()
    s.accept_waveform(SR, audio)
    self.stt.decode_stream(s)
    return stt_fix.correct(s.result.text.strip(), self.lang)
```

Keep the production raw-audio path for the direct comparison shown here. Scoring
normalization is for metrics only; do not apply it to production transcripts.
The existing welcome language detector and TTS load path are separate from this
replacement. No production files were edited.

## Reproducibility and licensing

Read [INDIC_CANDIDATES.md](INDIC_CANDIDATES.md) for alternative models and sources.
The IndicConformer export revision is
`9721eb71eea141fae0982cfcdb9dd2e3d4953c4a`; each language weighs 197.6 MB plus
67.6 KB tokens. Underlying AI4Bharat weights are MIT; the export repository metadata
is Apache-2.0 and its credit section retains source model licences. Both permit
commercial use with the relevant notices. Preserve AI4Bharat attribution and the
saved model cards. The Tamil Whisper-small CT2 export is Apache-2.0 and traces to
`vasista22/whisper-tamil-small`; pinned revision
`fd6de42af4f590e063ec0fc72bc9bfd6c13a4595`. Every download has exact byte size, source
URL and SHA-256 in `voice/_dl/<name>/download.json`; none exceeds 2 GB.

Runtime: existing sherpa-onnx 2.0.1; faster-whisper 1.2.1, CTranslate2 4.9.0,
PyAV 17.1.0 for the Whisper challenger. Only faster-whisper/PyAV were installed,
with `pip --no-deps --target lap/stt_eval/_deps`; no shared environment packages
were replaced. **torch is still 2.12.1+cpu.** CTC inference needs no added dependency.
See README commands and `run_indic.py`. Fresh result files record zero resumed
utterances; use a new `--out` for new timings.

Validation: all 13 existing/new `test_*.py` tests passed. These cover existing text
rules plus Unicode mark preservation, symmetric spelling normalization, semantic
non-collapse and consumption of faster-whisper's lazy generator inside the timer.

## Remaining routing limits

Tamil's much lower WER/CER does **not** establish a clean-audio snap improvement.
The unchanged matcher can prefer Shruti's Latin loanwords over IndicConformer's
native-script forms. For example, one clean `பிளாஸ்டிக் ரீசைக்கிள் பண்ணலாமா?`
clip becomes `பிளாஸ்டிக் ரீசிகிள் பண்ணலாமா`: plastic survives, but the current
matcher returns `other` instead of `recyclable`. Some misses are real recognition
errors, such as `துணிகளை` becoming `துமிகளை` (clothes item lost).

There are also matcher errors despite a near-perfect transcript: the question
`ரீசைக்கிள் பண்றதுக்கு முன்னாடி பிளாஸ்டிக் பாட்டிலை எப்படி கழுவணும்?`
asks how to wash a bottle before recycling (gold intent `other`), but its well
recognized IndicConformer transcript triggers `recyclable`. This explains why
routing is a separate measure and why a model replacement alone is insufficient
to claim better end-to-end routing. No matcher or correction edits were made.

Hindi still has joined-word CTC errors (`ब्लूबिन`, `ड्राई वेस्टकसबिन`) and variable
recycle spellings. Tamil has similar compound boundaries and loanword errors.
CER is particularly useful here; normalized WER intentionally does not erase
arbitrary boundaries or turn broad semantic paraphrases into exact matches.
