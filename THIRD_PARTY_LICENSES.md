# Third-party licences

Sorti's own source code is MIT-licensed (see [`LICENSE`](LICENSE)). Everything listed below is third-party material. It is **not** covered by that MIT licence and keeps its own licence. Check each upstream page before redistributing or making commercial use. Where this file and an upstream licence disagree, the upstream licence wins.

> **Built with Llama.** Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

> **Use restriction of the complete system.** NLLB-200 is CC-BY-NC-4.0 (non-commercial), and the YOLO detectors are derived from Ultralytics models under AGPL-3.0. Sorti as shipped is therefore intended for **non-commercial, educational and research use**.

## Models in this repository

| File(s) | Origin | Licence |
|---|---|---|
| `vision/det/yolo11n_320.pte` | Ultralytics YOLO11n (COCO), exported to ExecuTorch | AGPL-3.0 (Ultralytics), or an Ultralytics Enterprise licence |
| `vision/det/waste_320.pte` | YOLO11n fine-tuned by us on waste images, exported to ExecuTorch | derived from Ultralytics YOLO11n: AGPL-3.0 |
| `vision/det/world_320.pte` | Ultralytics YOLO-World with a fixed vocabulary of 23 waste prompts, exported to ExecuTorch | AGPL-3.0 (Ultralytics) |
| `vision/v2_taught/*.pte` | MobileNetV3-Large and MobileNetV4-Conv-Medium fine-tuned by us; ImageNet-pretrained backbones | backbone weights under their upstream licences (torchvision: BSD-3-Clause; MobileNetV4: see the upstream model card) |
| `vision/*.pte` (v1) | MobileNetV3-Large/Small and EfficientNet-B0 fine-tuned by us from torchvision ImageNet weights | torchvision: BSD-3-Clause |

## Models downloaded by `download_models.sh` or the Llama setup (not committed)

| Model | Source | Licence |
|---|---|---|
| Moonshine base (English STT), sherpa-onnx INT8 | Useful Sensors, via the k2-fsa/sherpa-onnx releases | MIT |
| Whisper base (language ID), sherpa-onnx INT8 | OpenAI, via the k2-fsa/sherpa-onnx releases | MIT |
| IndicConformer 120M CTC hi / ta (STT), INT8 ONNX | AI4Bharat, via the Hugging Face repo `parismitaglobalsolutions/indicconformer-sherpa-onnx` | MIT (AI4Bharat IndicConformer); see the converted repo's card |
| Silero VAD | snakers4/silero-vad, via the sherpa-onnx releases | MIT |
| Piper voice `en_US-lessac-medium` | rhasspy/piper-voices | see the voice's `MODEL_CARD` (trained on the Blizzard 2013 Lessac dataset, which has its own terms) |
| Piper voice `hi_IN-pratham-medium` | rhasspy/piper-voices | see the voice's `MODEL_CARD` |
| Piper voice `ta_IN-rasa_female-medium` | `tinisoft/piper-ta_IN-rasa_female-medium`, trained on the AI4Bharat Rasa dataset | CC-BY-4.0 |
| Meta MMS Tamil TTS (optional fallback, `--ta-voice mms`; not downloaded by default) | facebook/mms-tts-tam | **CC-BY-NC-4.0, non-commercial** |
| NLLB-200 distilled 600M, CTranslate2 INT8 | Meta, via `JustFrederik/nllb-200-distilled-600M-ct2-int8` | **CC-BY-NC-4.0, non-commercial** |
| Llama 3.2 1B Instruct SpinQuant INT4 (ExecuTorch `.pte`) | Meta, via `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` | **Llama 3.2 Community License**, Acceptable Use Policy; "Built with Llama" attribution required |
| Llama 3.2 1B Instruct QLoRA INT4 (alternative) | `meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8` | **Llama 3.2 Community License** |

## Data and media in this repository

| Material | Licence |
|---|---|
| Photos in the documentary videos (`lap/videos/photos/`, `lap/videos/*/general_photo.*`) | Wikimedia Commons: CC0 / public domain / CC BY / CC BY-SA. The author and licence of each photo are in `lap/videos/photos/<type>/credits.json` and are shown on every frame of the videos. |
| Stress-test video photos (`vision/testvideo/`) | Wikimedia Commons; see `vision/testvideo/credits.json` and `CREDITS.md` |
| Extra "general waste" training photos (`vision/extra/general_waste/`) | Wikimedia Commons; see `vision/extra/general_waste/credits.json` |
| Vision test set (`vision/pi_testset/`) | held-out images from the training dataset `4w4kt/waste-dataset-image`; subject to that dataset's terms |
| Fonts in `lap/web/fonts/` (Lato, Lohit Devanagari, Lohit Tamil; renamed `Sorti*`) | SIL Open Font License 1.1; the licence texts are next to the fonts |

## Python runtime dependencies (installed via `requirements.txt`, not bundled)

| Package | Licence |
|---|---|
| PyTorch, ExecuTorch, torchao, pytorch_tokenizers | BSD-3-Clause |
| sherpa-onnx | Apache-2.0 |
| onnxruntime | MIT |
| **piper-tts 1.8.0** (OHF-Voice/piper1-gpl) | **GPL-3.0-or-later** |
| CTranslate2 | MIT |
| sentencepiece | Apache-2.0 |
| transformers, tokenizers | Apache-2.0 |
| OpenCV (opencv-python) | Apache-2.0 |
| rapidfuzz | MIT |
| FastAPI | MIT |
| uvicorn | BSD-3-Clause |
| soundfile | BSD-3-Clause |
| numpy, pillow | BSD-3-Clause / MIT-CMU (HPND) |
| huggingface_hub | Apache-2.0 |
| pyserial | BSD-3-Clause |
| Ultralytics (export only, `vision/det/export_yolo.py`; not needed at runtime) | AGPL-3.0 |
