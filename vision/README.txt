PS2 waste classifier, first-version single model: mobilenet_v3_large

This is the single-model classifier scored in the model accuracy report. The shipped
app uses a classifier ensemble plus the det/ detectors; see docs/ARCHITECTURE.md.

LAPTOP WEBCAM TEST (Windows/macOS/Linux, Python 3.10-3.13):
  pip install executorch==1.3.1 opencv-python pillow numpy
  python webcam_demo.py --pte mobilenet_v3_large_int8_xnnpack.pte --labels labels.json
  keys: q quit | s snapshot | space freeze | f full frame / centre box

RASPBERRY PI 5 BENCHMARK (64-bit Raspberry Pi OS):
  python3 -m venv ~/ps2_et && source ~/ps2_et/bin/activate
  pip install executorch==1.3.1 pillow numpy
  python3 pi_benchmark.py --pte mobilenet_v3_large_int8_xnnpack.pte --labels labels.json --testdir pi_testset --runs 300

PI LIVE CAPTIONS (optional):
  Install ffmpeg and sherpa-onnx in the Pi setup (sherpa-onnx goes in ~/ps2_et).
  The Whisper base INT8 and Silero VAD files are listed in download_models.sh.
  For Hindi and Tamil captions, also install the IndicConformer files listed there.
  Use arecord -l to find the USB card name, then start the camera preview with:
    SORTI_MIC_DEV=plughw:CARD=Device,DEV=0 bash pi/start_vision.sh --bg
  The browser shows captions and the detected language for Hindi, Tamil, and English.
  The Pi decodes locally and holds the eight most recent captions in memory.
  The camera preview and captions are visible to anyone who can reach its LAN port.
