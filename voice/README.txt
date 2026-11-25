PS2 waste assistant: offline voice models (Tamil / Hindi / English) for Raspberry Pi 5

This file documents the first-version voice stack scored in the model accuracy report
(Shruti STT for Tamil and Hindi, Whisper base for English, Meta MMS for Tamil TTS).
The shipped app uses Moonshine (en), IndicConformer (hi, ta) and Piper voices instead;
see docs/ARCHITECTURE.md and download_models.sh. The model folders listed below are
not in this repository and are not fetched by download_models.sh, except
stt_whisper_base_en/ (language ID) and the Piper hi / en voices.

FOLDERS
  stt_shruti_ta/        Tamil speech-to-text (Shruti, sherpa-onnx)        Apache-2.0, keep NOTICE
  stt_shruti_hi/        Hindi speech-to-text (Shruti, sherpa-onnx)        Apache-2.0, keep NOTICE
  stt_whisper_base_en/  English speech-to-text (Whisper base, INT8)       MIT
  tts_piper_hi/         Hindi voice (Piper hi_IN pratham medium)          see MODEL_CARD in folder
  tts_piper_en/         English voice (Piper en_US lessac medium)         see MODEL_CARD in folder
  tts_mms_ta_onnx/      Tamil voice (Meta MMS, exported to ONNX fp32)     CC-BY-NC-4.0 = NON-COMMERCIAL ONLY
  domain_snap.py        maps a noisy transcript to (item, intent)         needs: pip install rapidfuzz
  speech_bench.py       the STT / TTS benchmark harness

INSTALL (Pi or laptop, Python 3.10-3.13)
  pip install sherpa-onnx piper-tts onnxruntime transformers rapidfuzz soundfile numpy

SPEECH-TO-TEXT (16 kHz mono float32 audio)
  import sherpa_onnx
  d = "stt_shruti_ta/"
  rec = sherpa_onnx.OfflineRecognizer.from_moonshine(preprocessor=d+"preprocess.onnx", encoder=d+"encode.onnx",
        uncached_decoder=d+"uncached_decode.onnx", cached_decoder=d+"cached_decode.onnx", tokens=d+"tokens.txt", num_threads=4)
  d = "stt_whisper_base_en/"
  rec_en = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=d+"base-encoder.int8.onnx", decoder=d+"base-decoder.int8.onnx",
        tokens=d+"base-tokens.txt", language="en", task="transcribe", num_threads=4)
  s = rec.create_stream(); s.accept_waveform(16000, audio); rec.decode_stream(s); text = s.result.text

SNAP TO INTENT
  from domain_snap import snap
  snap(text, "ta")   # {'item': 'battery_waste' or None (use camera class), 'intent': 'which_bin' | ... | 'other' (send to Llama)}

TEXT-TO-SPEECH
  Piper:  from piper import PiperVoice; v = PiperVoice.load("tts_piper_hi/hi_IN-pratham-medium.onnx")
          import wave; v.synthesize_wav("namaste", wave.open("out.wav", "wb"))
  Tamil:  import onnxruntime as ort; from transformers import AutoTokenizer
          tok = AutoTokenizer.from_pretrained("tts_mms_ta_onnx"); sess = ort.InferenceSession("tts_mms_ta_onnx/model.onnx")
          wav = sess.run(None, {"input_ids": tok(text, return_tensors="np")["input_ids"].astype("int64")})[0][0]   # 16 kHz

MEASURED (Colab CPU, 2 threads, not the Pi; a Pi 5 is expected to be somewhat slower)
  STT  Tamil Shruti ~455 MB, ~1.1 s per 3 s question | Hindi Shruti ~420 MB, ~1.0 s | English Whisper-base ~740 MB, ~1.1 s
  TTS  Tamil MMS-ONNX ~615 MB, ~2.2 s per 3 s speech | Hindi/English Piper ~200-230 MB, ~0.4 s
  Domain test (synthetic voices): intent 96-100%, item 83% ta / 92% hi / 99% en. Not yet re-tested with real recordings.
