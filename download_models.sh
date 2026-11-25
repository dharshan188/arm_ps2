#!/usr/bin/env bash
# Download every speech / translation model Sorti needs (~2.3 GB) from their original public sources.
# Safe to re-run: files that already exist are skipped.   Usage:  ./download_models.sh
set -euo pipefail
cd "$(dirname "$0")"
V=voice
GH=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
HF=https://huggingface.co
IC=$HF/parismitaglobalsolutions/indicconformer-sherpa-onnx/resolve/9721eb71eea141fae0982cfcdb9dd2e3d4953c4a

get() {  # get URL DEST
  if [ -s "$2" ]; then echo "  ok   $2"; return; fi
  mkdir -p "$(dirname "$2")"; echo "  get  $2"
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

echo "[1/6] English speech-to-text (Moonshine base, int8)"
if [ ! -s $V/stt_moonshine_base_en/encode.int8.onnx ]; then
  get $GH/sherpa-onnx-moonshine-base-en-int8.tar.bz2 /tmp/moonshine.tar.bz2
  mkdir -p $V/stt_moonshine_base_en
  tar xjf /tmp/moonshine.tar.bz2 -C $V/stt_moonshine_base_en --strip-components=1 && rm /tmp/moonshine.tar.bz2
fi

echo "[2/6] Language detection (Whisper base, int8)"
if [ ! -s $V/stt_whisper_base_en/base-encoder.int8.onnx ]; then
  get $GH/sherpa-onnx-whisper-base.tar.bz2 /tmp/whisper.tar.bz2
  mkdir -p $V/stt_whisper_base_en
  tar xjf /tmp/whisper.tar.bz2 -C $V/stt_whisper_base_en --strip-components=1 \
      sherpa-onnx-whisper-base/base-encoder.int8.onnx sherpa-onnx-whisper-base/base-decoder.int8.onnx \
      sherpa-onnx-whisper-base/base-tokens.txt && rm /tmp/whisper.tar.bz2
fi

echo "[3/6] Hindi + Tamil speech-to-text (AI4Bharat IndicConformer, int8)"
for l in hi ta; do
  get $IC/$l/model.int8.onnx $V/stt_indicconformer_$l/model.int8.onnx
  get $IC/tokens.txt         $V/stt_indicconformer_$l/tokens.txt
done

echo "[4/6] Voice activity detection (Silero VAD)"
get $GH/silero_vad.onnx $V/vad/silero_vad.onnx

echo "[5/6] Text-to-speech voices (Piper: English, Hindi, Tamil)"
P=$HF/rhasspy/piper-voices/resolve/main
for f in onnx onnx.json; do
  get $P/en/en_US/lessac/medium/en_US-lessac-medium.$f  $V/tts_piper_en/en_US-lessac-medium.$f
  get $P/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.$f $V/tts_piper_hi/hi_IN-pratham-medium.$f
  get $HF/tinisoft/piper-ta_IN-rasa_female-medium/resolve/main/ta_IN-rasa_female-medium.$f \
      $V/tts_piper_ta_female/ta_IN-rasa_female-medium.$f
done

echo "[6/6] Translator for Hindi/Tamil <-> English (NLLB-200 600M, CTranslate2 int8)"
N=$HF/JustFrederik/nllb-200-distilled-600M-ct2-int8/resolve/main
for f in config.json model.bin sentencepiece.bpe.model shared_vocabulary.txt special_tokens_map.json; do
  get $N/$f translate/nllb_600m_ct2_int8/$f
done

echo
echo "Done. Optional: the Llama model for free-form questions -> see README step 6 (llama/setup_llama.sh)."
