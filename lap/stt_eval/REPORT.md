# STT evaluation — 17 November 2026

960 synthetic Piper WAVs: 40 hand-labelled questions/language, two rates, clean/15 dB/5 dB; both Tamil voices. English and Hindi: 240 WAVs/model; Tamil: 480. Models ran serially with the server’s four-thread sherpa loader/decode settings. No server changes. Arrows mean **ASR → corrected**; accuracy/error values are percentages.

Current models, raw audio:

| Model | Condition | WER | CER | Keywords | Intent+item |
|---|---|---:|---:|---:|---:|
| Whisper base | clean | 5.7→2.8 | 2.1→1.3 | 85.8→99.1 | 71.2→71.2 |
| Whisper base | snr15 | 10.8→8.5 | 5.0→4.2 | 81.1→91.5 | 68.8→68.8 |
| Whisper base | snr5 | 22.8→21.1 | 14.0→13.3 | 67.0→74.5 | 51.2→52.5 |
| Hindi Shruti | clean | 35.4→28.3 | 30.1→26.5 | 58.0→87.5 | 63.7→66.2 |
| Hindi Shruti | snr15 | 47.2→42.9 | 40.8→37.8 | 46.4→63.4 | 51.2→52.5 |
| Hindi Shruti | snr5 | 61.6→57.1 | 50.5→47.5 | 26.8→40.2 | 36.2→36.2 |
| Tamil Shruti | clean | 59.9→55.7 | 37.6→35.7 | 68.1→77.8 | 63.7→64.4 |
| Tamil Shruti | snr15 | 63.3→60.8 | 41.7→40.5 | 57.4→63.9 | 56.2→56.9 |
| Tamil Shruti | snr5 | 77.4→76.3 | 52.6→52.1 | 36.1→39.4 | 36.2→36.2 |

English comparison, pooled equally across all three conditions; raw audio:

| Model | WER | Keywords | Intent+item | Mean / p95 s | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Whisper base | 13.1→10.8 | 78.0→88.4 | 63.7→64.2 | 0.640 / 0.960 | 414 |
| Moonshine base int8 | 10.6→9.3 | 83.6→89.6 | 67.1→67.1 | 0.276 / 0.431 | 398 |
| Whisper small.en int8 | 7.5→6.2 | 86.2→92.5 | 70.0→70.0 | 1.472 / 2.200 | 882 |

Preprocessing comparison (arrows here mean **raw → preprocessed**, all conditions pooled):

| Model | ASR WER | Corrected intent+item |
|---|---:|---:|
| Whisper base | 13.1→12.1 | 64.2→62.9 |
| Moonshine base int8 | 10.6→9.0 | 67.1→65.4 |
| Whisper small.en int8 | 7.5→5.7 | 70.0→70.0 |
| Hindi Shruti | 48.0→50.4 | 51.7→51.2 |
| Tamil Shruti | 66.9→71.8 | 52.5→51.5 |

Recommend **English Moonshine base int8 + correction, raw audio** for this RAM-limited kiosk. Small.en offers the best accuracy (corrected WER 6.2% versus 9.3%) but uses 2.2× the RSS and 5.3× the decode time. Keep preprocessing **off by default**: it hurts aggregate intent+item accuracy for base/Moonshine/Shruti and raises native-language WER, especially at 5 dB. For an accuracy-first small.en deployment, preprocessing improves WER while leaving routing accuracy unchanged.

Correction improves domain-word survival substantially; routing gains are modest because the matcher already fuzzily tolerates many spellings. Seven unit-test methods pass, including ordinary-sentence safeguards and audio edge cases. Isolated correction timing: mean 0.033 ms, p95 0.063 ms, max 0.656 ms over 8,640 calls.

Keyword accuracy is exact concept recall, including explicit loanword aliases; intent+item requires both manual labels, including `None`. Native WER/CER also penalise valid Latin-script loanwords; those are preserved. Reference-text matcher accuracy: Whisper base 72.5%, Hindi Shruti 85.0%, Tamil Shruti 75.0%. `domain_snap("waste", "en")` incorrectly returns `e_waste` (83.3 score); matcher false positives and missing phrasings limit routing gains even after accurate correction. The matcher was unchanged.

Rules were developed on these outputs; this is not held-out human accuracy. Validate with real kiosk recordings. Latency excludes load/text scoring, includes first decode and preprocessing where enabled. RSS is STT-process peak, excluding the app’s TTS/vision/Llama. Full per-condition metrics, transcripts and correction timings are in [results](results/); generation/scoring details and rerun commands are in [README.md](README.md).

After the existing `text = ... SP.transcribe(audio)` assignment, before `snap` or Llama (import at module scope):

```python
from stt_fix import correct
if typed is None: text = correct(text, lang)
```
