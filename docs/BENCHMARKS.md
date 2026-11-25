# Sorti benchmarks

Every number here was measured, and every number names its source. Estimates are left out on purpose.

| Source key | Document | What it measures |
|---|---|---|
| **PI** | [`reports/Sorti_RaspberryPi5_Test_Report.pdf`](reports/Sorti_RaspberryPi5_Test_Report.pdf) | Raspberry Pi 5 (8 GB) on 19 Nov 2026, with the full app live: camera, 3 detectors, 2 classifiers, always-on mic, Bluetooth speaker, Llama 3.2 1B, NLLB and the Arduino sorter. This is the shipped configuration. |
| **ACC** | [`reports/Sorti_Model_Accuracy_Report.pdf`](reports/Sorti_Model_Accuracy_Report.pdf) | Model selection on a 2-thread Colab CPU stand-in (16–17 Nov 2026). It covers the v1 single-model stack: one centre-crop classifier, Shruti STT and MMS Tamil TTS. Several of these components have since been replaced; see the notes below. |
| **DEV** | the team's development notes; these are not in the repository and were not re-measured for it | x86 Ubuntu laptop, Python 3.10, used as a Pi stand-in during development. Treat them as indicative laptop figures, not Pi results. |
| **STT** | [`../lap/stt_eval/REPORT.md`](../lap/stt_eval/REPORT.md), [`../lap/stt_eval/INDIC_REPORT.md`](../lap/stt_eval/INDIC_REPORT.md) | the speech-to-text evaluation set, scripts and raw results in `lap/stt_eval/results/` |
| **REPO** | `scripts/eval_testset.py`, run while preparing this repository (25 Nov 2026) | x86 laptop, same models and test set as PI §2 |

---

## 1. Raspberry Pi 5: full system (source: PI)

### Memory, worst case (everything loaded)

| Process / state | Resident memory |
|---|---|
| Sorti server: camera, detectors, classifiers, STT, TTS, language ID, NLLB | 2.32 GB |
| Llama 3.2 1B worker (SpinQuant INT4, ExecuTorch) | 1.57 GB |
| **App total** (server + Llama), reported by `/api/state` | **3.89 GB** |
| Whole Pi, including desktop and audio services | 5.47–5.59 GB used of 8.06 GB, swap 0 MB |

### Vision accuracy on `vision/pi_testset/` (540 images, 60 per type)

These are 540 held-out images from the test split of `4w4kt/waste-dataset-image`, evaluated on the Pi 5. They are not Pi-camera photos. The headline pair used everywhere is 93.9 % type / 95.2 % bin (live settings).

The classifier is the ensemble of MobileNetV3-L and MobileNetV4-M, dynamic INT8.

| Waste type | Correct type | Correct bin |
|---|---|---|
| Battery | 95.0 % | 95.0 % |
| Clothes | 100.0 % | 100.0 % |
| E-waste | 90.0 % | 90.0 % |
| Glass | 98.3 % | 98.3 % |
| Light bulbs | 98.3 % | 98.3 % |
| Metal | 81.7 % | 88.3 % |
| Organic | 93.3 % | 95.0 % |
| Paper | 93.3 % | 96.7 % |
| Plastic | 95.0 % | 95.0 % |
| **All (live paper/clothes weighting)** | **93.9 %** | **95.2 %** |
| **All (raw ensemble)** | **94.1 %** | **95.6 %** |

The summary line of the PI report quotes "94.1 % type, 95.2 % bin", which mixes the raw type figure with the live bin figure. This repository uses the live pair (93.9 / 95.2) as the headline and the raw pair (94.1 / 95.6) as a footnote.

- Classification time per item: 78 ms median, 92 ms p90, measured while the live app was running.
- Live camera with all three detectors: 2.8–4.8 FPS.
- Weakest type: metal (81.7 %), mostly confused with plastic or e-waste.

**Reproduction (REPO):** `scripts/eval_testset.py` on an x86 laptop gives 94.3 % type / 95.7 % bin, with a median of 25–29 ms per image. The same `.pte` files score 94.1 % / 95.6 % on the Pi. The two runs differ in 1 image.

### Hazard recall

Source: the per-type "correct bin" column of PI §2. Each type has 60 images.

| Type | Correct bin | Images correct |
|---|---|---|
| Battery (hazardous) | 95.0 % | 57 / 60 |
| Light bulbs (hazardous) | 98.3 % | 59 / 60 |
| E-waste | 90.0 % | 54 / 60 |
| **Hazardous only** | | **116 / 120 = 96.7 %** |
| **Hazardous + e-waste** | | **170 / 180 = 94.4 %** |

The image counts are the percentages × 60, which are exact for these values. The figures are for the live settings (paper/clothes weighting on).

v1 single model on 5,460 held-out images (`vision/mobilenet_v3_large_metrics.json`, `test_int8_pte.stream_confusion`): hazardous-stream recall 572 / 601 = 95.2 %, e-waste-stream recall 274 / 299 = 91.6 %, overall stream accuracy 97.0 %. The confusion matrix is in `docs/images/confusion_matrix.png`.

### Response time (live app, everything running)

| Step | Typical | Worst seen |
|---|---|---|
| Rule answer (which bin / recyclable / hazardous) | 0.13 s | 0.20 s |
| Fixed demo answer (recycling steps etc.) | 0.12 s | 0.13 s |
| Speech-to-text, English (Moonshine) | 0.71 s | 0.91 s |
| Speech-to-text, Tamil (IndicConformer) | 1.14 s | 1.48 s |
| Speech-to-text, Hindi (IndicConformer) | 1.85 s | 1.85 s |
| Language detection (runs alongside STT) | 1.04 s | 1.42 s |
| English Llama, first words on screen | 1.7 s | 2.0 s |
| English Llama, complete answer | 6.3 s | 6.5 s |
| Tamil / Hindi Llama answer (NLLB → Llama → NLLB) | ~19 s | 23.0 s |
| Arduino sort cycle (servo move + return) | 5.8 s | 5.9 s |

### Optimisations, measured idle on the Pi

| Optimisation | Before | After |
|---|---|---|
| Language ID encodes only the speech segment | 1.12 s | 0.23 s |
| Llama system prompt prefilled into the KV cache while idle (time to first token) | 7.3 s | 0.85 s |
| NLLB per-language vocabulary map + 2 threads | 9.2 s | 3.5 s |

### Language understanding (`voice/domain_snap.py` + fixed answers)

| Test | Result |
|---|---|
| Intent + item, English / Hindi / Tamil (72 questions each) | 72/72, 72/72, 72/72 (**216/216**) |
| Held-out questions (new phrasings, written after the rules) | 27/27 |
| Tanglish / Hinglish code-mixed (native script and romanised) | 13/13 |
| Fixed demo questions (3 questions × 3 phrasings) | 9/9, and 4/4 normal questions correctly not matched |
| Language detection on en/hi/ta spoken questions (synthetic voices) | 15/15 |

### Sorter, power and thermal

- Arduino: 5 live sorts (e-waste, battery, paper, plastic), each acknowledged in 5.8–5.9 s.
- Supply 5.07–5.18 V with no under-voltage or throttling flags. 70.8 °C under full load.

---

## 2. Model selection: v1 single-model stack (source: ACC)

Measured on a 2-thread Colab CPU with ExecuTorch 1.3.1. The test set has 5,460 held-out images from `4w4kt/waste-dataset-image` (9 classes). Duplicates were removed: 311 training images and 116 test images that copied training images.

| Model | INT8 .pte | FP32 .pte | INT8 latency p50 (Colab CPU) | Accuracy (INT8) | Macro-F1, 9 classes | Macro-F1, 4 streams |
|---|---|---|---|---|---|---|
| **MobileNetV3-Large** (shipped as v1) | 4.52 MB | 16.87 MB | 16.1 ms | 94.0 % (0.9405 in the metrics file) | 0.944 FP32 → **0.929 INT8** | 0.951 → **0.939** |
| EfficientNet-B0 (head kept FP32) | 4.51 MB | 16.07 MB | 24.8 ms | 90.9 % | 0.937 → 0.893 | 0.944 → 0.904 |
| MobileNetV3-Small | 1.73 MB | 6.14 MB | 3.2 ms | 31.8 % | 0.910 → **0.288** (collapsed) | 0.920 → 0.336 |

- Quantisation-aware training (3 epochs) on MobileNetV3-L scored 0.915 and was rejected in favour of PTQ.
- Every model was fully delegated to XNNPACK (1 delegate call).
- On the x86 Colab host, INT8 was slower than FP32 for MobileNetV3-L (16.1 vs 7.5 ms p50; model RSS 10.7 vs 25.5 MB; `host_bench_NOT_PI` in `vision/mobilenet_v3_large_metrics.json`). INT8 was chosen for size and RAM. A latency profile on the Pi itself is Phase 2.
- The `.pte` files and `*_metrics.json` are in `vision/`.

**Intent matcher (ACC, on saved Shruti / Whisper transcripts):** question-type accuracy 96–100 %. A wrong item was named in only 3 of 243 questions. Item F1 was 0.893 (ta), 0.942 (hi) and 0.987 (en).

> Components that have changed since ACC: STT is now Moonshine (en) and IndicConformer (hi, ta) instead of Shruti and Whisper. Tamil TTS is now Piper Rasa instead of MMS. Vision is now the multi-object detector + ensemble pipeline. The Llama section of ACC contains estimates only; use section 1 for measured Llama numbers.

---

## 3. Speech-to-text evaluation (source: STT, laptop)

The set was built from Piper TTS speech plus added noise. See `lap/stt_eval/README.md` to re-run it.

| Language | Before → after | Metric |
|---|---|---|
| English: Whisper base.en → **Moonshine base INT8** | 5.7 % → 2.8 % clean, 10.8 % → 4.5 % at 15 dB SNR | WER |
| English: same change | 86 % → 99 % | keyword accuracy |
| English: same change | 0.64 s → 0.28 s | per question |
| Hindi: Shruti → **IndicConformer 120M CTC INT8** | 35 % → 14 % (CER 30 % → 4 %) | WER, clean |
| Tamil: Shruti → **IndicConformer 120M CTC INT8** | 60 % → 26 % (CER 38 % → 6 %) | WER, clean |

Language ID (`lap/lid.py`, Whisper base logits pooled to en/hi/ta): 159/160 correct on the evaluation set (DEV, laptop, development notes). The Pi figure is 15/15 (PI §4).

---

## Development laptop measurements

Source: DEV (laptop, development notes; not in the repository). x86 laptop, not the Pi.

| What | Result |
|---|---|
| Streaming TTS by sentence: first sound on a 460-character reply | 5.4 s → 0.4 s |
| Tamil TTS, 4 casual lines: Meta MMS vs Piper Rasa female | 14.5 s for 50 s of audio (27 long pauses) vs 2.4 s for 36 s (16 pauses); about 400 MB less RAM |
| Llama 3.2 1B QLoRA INT4, ExecuTorch (**laptop, not Pi**; no decode tok/s was measured on the Pi, where first-words and total answer time are the measured metrics) | about 32 tok/s decode, about 160 tok/s prefill, about 2.2 GB RSS |
| Multi-object vision pipeline | about 560 MB RAM, 5.5 FPS; 4.4–4.5 FPS in the 4-item scenes shown in the README |
