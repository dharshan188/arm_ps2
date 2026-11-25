# Sorti architecture

Sorti is one Python process, `lap/server.py` (FastAPI + uvicorn), plus one optional child process for Llama. The browser is only a screen. The server owns the camera, the microphone, the speaker and the Arduino serial port, as a kiosk would.

```
lap/server.py
 ├─ Vision thread      camera → 3 detectors → tracker → classifier ensemble → confirmed objects
 ├─ Recorder thread    arecord 16 kHz mono → Silero VAD → utterances
 ├─ Answer path        LID ∥ STT → stt_fix → domain_snap → rules | fixed answers | NLLB → Llama → NLLB
 ├─ Speech             Piper TTS, sentence by sentence → aplay / pw-play
 ├─ Sorter thread      one serial command per confirmed item → Arduino
 ├─ Heavy-model governor (Llama, NLLB: lazy load, idle unload, MemAvailable checks)
 └─ HTTP               /, /static (lap/web), /media (lap/videos), /video (MJPEG), /api/*
llama/chat.py --worker (child process): ExecuTorch TextLLMRunner, KV-cache-primed system prompt
```

---

## 1. Vision pipeline (`vision/`)

| Stage | File | Details |
|---|---|---|
| Capture | `lap/server.py` `Vision` | OpenCV `VideoCapture`, whole frame. It retries while the USB camera enumerates late at boot. |
| Base masking | `vision/base_zone.py` | The cardboard sorting base is found by colour (`pi/base_zone.json` is generated on the device) and never counted as an item. `SORTI_BASE=0` disables this. |
| Detection | `vision/detector.py`, `vision/det/*.pte` | Three 320 px ExecuTorch / XNNPACK detectors run together: `waste_320.pte` (YOLO11n trained on waste), `yolo11n_320.pte` (COCO) and `world_320.pte` (YOLO-World, 23 waste prompts). The raw output is `[1, 4+C, N]` and NMS runs in Python. The export script is `vision/det/export_yolo.py`, documented in `vision/det/README.md`. |
| Tracking | `vision/multi.py` | IoU tracker with duplicate-box merge, box smoothing and a 2-frame hold. An item is confirmed after 6 steady frames that clear a per-class confidence bar. Boxes larger than `SORTI_MAX_ITEM_AREA` (default 0.45 of the frame) are ignored. |
| Classification | `vision/sorti_model.py`, `vision/v2_taught/` | Ensemble of MobileNetV3-Large and MobileNetV4-Conv-Medium, dynamic INT8, 224 px. It averages the softmax over 11 classes: 9 waste types, general waste and background. Preprocessing: resize the short side to 256, centre-crop 224, ImageNet mean/std. |
| Unknown items | `SORTI_UNKNOWN_AS` | A steady item that no class claims is treated as plastic by default. Set it to `""` to show a grey "not sure" box instead. |
| Bin | `vision/v2_taught/labels.json` `stream_map` | biodegradable (organic, paper) → green, recyclable (plastic, glass, metal, clothes) → yellow, hazardous (battery, bulbs) → red, e_waste → authorised collector, general → local general-waste collection. |

The vision thread pauses while an answer is being computed, so STT, Llama and NLLB get all four cores.

The vision-only preview (`vision/pi_stream.py`, started by `pi/start_vision.sh`) runs the same `MultiVision` pipeline without audio and serves an MJPEG page on port 8080.

### v1 single-model classifier

`vision/mobilenet_v3_large_int8_xnnpack.pte` (4.5 MB, full INT8 PT2E PTQ) is the model scored in the model accuracy report. `vision/webcam_demo.py` and `vision/pi_benchmark.py` use it. `vision/train/` holds its training and export scripts.

---

## 2. Voice pipeline (`lap/`, `voice/`)

| Stage | Model / file | Details |
|---|---|---|
| Capture | `arecord`, 16 kHz mono | `pick_mic()` chooses the source (`--mic <substring>`) and unmutes it. ALSA devices can be forced with `SORTI_MIC_DEV` / `SORTI_SPK_DEV`. |
| VAD | Silero VAD (`voice/vad/silero_vad.onnx`) | Always-on listening. An echo guard ignores Sorti's own voice. |
| Language ID | `lap/lid.py`, Whisper base INT8 | Language-token logits pooled to en / hi / ta (ml/te/kn → ta, ur/mr → hi). English must win by a margin of 0.2. It runs in parallel with STT. If the speaker uses another language (p ≥ 0.8), the server loads that language's STT and TTS and re-transcribes. |
| STT | sherpa-onnx INT8 | English: Moonshine base. Hindi / Tamil: AI4Bharat IndicConformer 120M CTC. Only the active language is loaded. |
| ASR repair | `lap/stt_fix.py` `correct()` | Context-gated fixes: west → waste, bean → bin, wait → wet. "West Bengal" is left alone. |
| Understanding | `voice/domain_snap.py` `snap(text, lang)` | Fuzzy (rapidfuzz) and phonetic keyword matching that tolerates ASR slips, Tamil/Hindi case endings and code-mixing (native-script and romanised Tanglish / Hinglish). It returns `item`, `intent`, scores and `confident`. `item=None` means the camera's objects are used. |
| Routing | `lap/server.py` | `which_bin`, `recyclable`, `hazardous`, `where_to_give` and `repeat` get rule answers from `lap/content.py`. Fixed demo questions get fixed en/hi/ta answers. `other` goes to Llama. |
| TTS | Piper (`voice/tts_piper_*`) | en: lessac, hi: pratham, ta: AI4Bharat Rasa female (`--ta-voice male|mms` to switch). The reply is streamed sentence by sentence, and `lap/tts_utils.py` tightens the pauses. |
| Playback | `aplay` / `pw-play` | Bluetooth A2DP via `pi/bt_speaker.sh`. `--no-speaker` shows replies as text and plays videos muted. |

All spoken and UI text lives in `lap/content.py`: en, casual Hinglish and spoken Chennai Tamil, with English loan words written in native script.

---

## 3. Llama and translation

| Component | Details |
|---|---|
| Model | Llama 3.2 1B Instruct. The Pi uses `llama/llama3_2_1b_spinquant_int4_xnnpack.pte`, the public ExecuTorch SpinQuant INT4_EO8 export. `llama/setup_llama.sh` can export the QLoRA INT4 checkpoint instead. Override with `SORTI_LLAMA_PTE`. |
| Runtime | ExecuTorch `TextLLMRunner` (XNNPACK). The ExecuTorch LLM custom ops and quantized kernels are imported first. |
| Isolation | `llama/chat.py --worker`: a persistent child process speaking JSON lines. An in-process crash (SIGSEGV) seen on the Pi no longer takes the app down. A worker failure produces a rule answer. |
| KV-cache priming | After loading, and again after every answer, the worker prefills the system prompt (the `FRIEND` persona, grounded with few-shot examples and item facts) into the KV cache while idle. A question then prefills only its own tokens. First token went from 7.3 s to 0.85 s on the Pi. |
| Decoding | Greedy, Llama 3 chat template, short replies (`max_new_tokens` about 56), streamed to the UI. |
| Safety | Llama answers only intent `other`. Bin words in its output are replaced with the rule bin. For scene-based reuse questions it is consulted only for confirmed recyclable materials, and the fixed rules are appended for hazardous and organic items. |
| Translation | NLLB-200 distilled 600M, CTranslate2 INT8 (`translate/nllb_600m_ct2_int8/`), for hi/ta → en before Llama and en → hi/ta after it. A per-language vocabulary map (`translate/nllb_vmap_<lang>/vmap.txt`, built once on first use) limits the 256k-token output vocabulary to the target script. With beam 1 and 2 threads, translation went from 9.2 s to 1.1 s (Pi re-test, 25 Nov 2026). `content.casual()` swaps formal words for spoken ones. |

---

## 4. RAM governor

`Heavy` in `lap/server.py` manages the two large models.

| Model | `need_mb` | Policy |
|---|---|---|
| Llama | 1900 | Loaded lazily (and pre-warmed in the background once a language is chosen). Unloaded after `IDLE_UNLOAD_S` = 180 s idle when RAM is short, and kept for up to 15 min while more than 1.5 GB is free. |
| NLLB | 1000 | Same policy. Loaded in the background when Hindi or Tamil is chosen. |

Before each load the server checks `MemAvailable` plus half of free swap against `need_mb + MARGIN_MB` (150 MB). If that is short it unloads the other heavy model first. If it is still short it gives the rule answer for the current item, or a general bin guide, instead of freezing. Only the active language's STT and TTS are resident. `/api/state` reports `ram_mb` and `ram_avail_mb`.

Measured worst case on the Pi: server 2.32 GB + Llama worker 1.57 GB = **3.89 GB**, with 0 MB swap.

---

## 5. Arduino sorter

`Sorter` in `lap/server.py` drives the Arduino 2-servo sorter over USB serial (commands bio / re / hazard / ewaste).

- **Port:** `SORTI_SERIAL`, otherwise the first of `/dev/serial/by-id/*`, `/dev/ttyACM*` or `/dev/ttyUSB*`. It reconnects if the cable is unplugged.
- **Link:** 9600 baud. Opening the port resets the board.
- **Protocol:** one line per confirmed item:

  | Stream | Command |
  |---|---|
  | biodegradable (organic, paper) | `bio` |
  | recyclable (plastic, glass, metal, clothes) | `re` |
  | hazardous (battery, light bulbs) | `hazard` |
  | e-waste | `ewaste` |

  The server waits for the sketch's `Ready for next item.` before it sends the next command. General waste is not sent.
- **Manual trigger:** `POST /api/sort {"stream": "re"}`.
- The sketch itself is not part of this repository.

---

## 6. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Kiosk UI (`lap/web/index.html`). `?mock=1&lang=ta` gives a mock without devices. |
| GET | `/static/*` | `lap/web/` assets: `sorti.css`, `sorti.js` and local fonts. There are no CDNs. |
| GET | `/media/*` | Documentary videos and posters (Range requests supported) |
| GET | `/video` | Annotated MJPEG camera stream |
| GET | `/pred` | Latest vision prediction |
| GET | `/api/state` | Full UI state: objects, chat, status, language, RAM |
| POST | `/api/lang` | Set the language (`en`, `hi`, `ta`) |
| POST | `/api/welcome` | Restart the welcome flow |
| POST | `/api/ask/start`, `/api/ask/stop` | Push-to-talk recording |
| POST | `/api/ask/text` | Typed question: `{"text": "..."}` |
| POST | `/api/explain` | Re-announce the current items |
| POST | `/api/auto` | Toggle automatic announcements: `{"on": true}` |
| POST | `/api/video/ready`, `/api/video/done`, `/api/video/skip` | Video / voice-over synchronisation |
| POST | `/api/sort` | Manual sorter command: `{"stream": "..."}` |
| POST | `/api/stop`, `/api/clear` | Stop speech, clear the conversation |

---

## 7. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SORTI_LLAMA_PTE` | (auto) | Path to the Llama `.pte` |
| `SORTI_SERIAL` | (auto) | Arduino serial port |
| `SORTI_MIC_DEV`, `SORTI_SPK_DEV` | unset | Force ALSA capture / playback devices |
| `SORTI_MIC_GAIN` | `0` | USB mic gain to set |
| `SORTI_MIC_BUTTON` | `PnP_Sound` | Input device name of a physical mic button |
| `SORTI_BASE` | `1` | Mask the cardboard base |
| `SORTI_MAX_ITEM_AREA` | `.45` | Ignore boxes larger than this fraction of the frame |
| `SORTI_UNKNOWN_AS` | `plastic_waste` | Label for steady but unclassified items (`""` gives a grey box) |
| `SORTI_AV_DELAY` | unset | Extra audio delay for video sync (Bluetooth) |

Run `python lap/server.py --help` for the command-line flags: `--lang`, `--camera`, `--mic`, `--host`, `--port`, `--no-speaker`, `--welcome`, `--push-to-talk`, `--ta-voice`, `--ta-slow`, `--detector-pte`, `--model-dir` and `--video`.
