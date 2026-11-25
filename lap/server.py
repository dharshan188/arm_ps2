#!/usr/bin/env python3
"""Laptop-as-Raspberry-Pi demo: webcam -> waste classifier -> spoken explanation -> press-to-ask Q&A.

The laptop camera, microphone and speaker stand in for the Pi's. The browser is only the screen:
camera, mic and speaker are all driven by this process (OpenCV, arecord, aplay), as on the Pi.

    cd ~/arm2026/lap && env -u PYTHONPATH PATH=$HOME/ps2_et/bin:$PATH ~/ps2_et/bin/python server.py
    then open http://127.0.0.1:8000
"""
import argparse, collections, gc, json, os, re, signal, subprocess, sys, tempfile, threading, time, wave

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "lap")
sys.path[:0] = [os.path.join(ROOT, "voice"), os.path.join(ROOT, "llama"), HERE]

import content as C  # noqa: E402
import faq  # noqa: E402  (answer bank, off-topic guard, answer guard for free-form questions)
from domain_snap import snap  # noqa: E402

VISION = os.path.join(ROOT, "vision")
VOICE = os.path.join(ROOT, "voice")
LLAMA = os.path.join(ROOT, "llama")
TMP = tempfile.mkdtemp(prefix="lap_")
SR = 16000
# Raw ALSA devices (Pi without a desktop sound server), e.g. SORTI_MIC_DEV=plughw:CARD=Device,DEV=0.
# Unset = system default (PulseAudio/PipeWire on a desktop, as on the laptop).
_MIC_DEV, _SPK_DEV = os.environ.get("SORTI_MIC_DEV"), os.environ.get("SORTI_SPK_DEV")
# Speaker: with PipeWire (Pi desktop, Bluetooth speaker) use pw-play, which plays to the default sink; the Pi has no
# ALSA->PipeWire plugin, so plain `aplay` cannot reach a Bluetooth speaker. SORTI_SPK_DEV forces a raw ALSA device.
import shutil  # noqa: E402
APLAY = (["aplay", "-q", "-D", _SPK_DEV] if _SPK_DEV else
         ["pw-play"] if shutil.which("pw-play") and os.environ.get("XDG_RUNTIME_DIR") else ["aplay", "-q"])
ARECORD = ["arecord", "-q"] + (["-D", _MIC_DEV] if _MIC_DEV else [])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ shared state
class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.lang = "en"
        self.status = "loading"          # loading | watching | speaking | listening | thinking
        self.auto = True
        self.objects = []               # current multi-object tracks
        self.announced_signature = ()
        self.pred = None                 # live classifier output
        self.item = None                 # last announced item {cls, stream, ...}
        self.messages = []
        self.next_id = 1
        self.models = {"vision": "loading", "stt": "off", "tts": "off", "llama": "off", "translate": "off"}
        self.mic_level = 0.0
        self.speaking_id = None
        self.error = None
        self.video = None                # clip on screen: {id, cls, url, poster, credit, title, text, t}
        self.chosen = False              # False = welcome screen: asking which language the visitor wants
        self.welcome_phase = None        # asking | listening | detecting | waiting
        self.heard = None                # what the welcome mic heard
        self.next_prompt = 0.0
        self.tries = 0
        self.last_active = time.time()

    def add(self, role, kind, text, **extra):
        with self.lock:
            m = dict(id=self.next_id, role=role, kind=kind, text=text, lang=self.lang, t=time.time(), **extra)
            self.next_id += 1
            self.messages.append(m)
            self.messages = self.messages[-60:]
            return m

    def patch(self, mid, **kw):
        with self.lock:
            for m in self.messages:
                if m["id"] == mid:
                    m.update(kw)

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                if k == "status" and v != "listening" and recording():
                    continue             # a finishing speech/video must not hide a running recording (stuck-mic bug)
                setattr(self, k, v)


S = State()


def recording():
    rec = globals().get("REC")
    return rec is not None and rec.proc is not None


# ------------------------------------------------------------------ vision
UNKNOWN_AS = os.environ.get("SORTI_UNKNOWN_AS", "plastic_waste")   # label for steady-but-unknown items ("" = grey box)


class Vision(threading.Thread):
    def __init__(self, cam, pte, threshold, detector_pte=None, model_dir=None, video=None, exclude=None):
        super().__init__(daemon=True)
        self.cam, self.video = cam, video
        self.threshold = threshold
        model_dir = model_dir or os.path.join(VISION, "v2_taught")
        detector_pte = detector_pte or os.path.join(VISION, "det", "waste_320.pte")
        self.detector_pte, self.model_dir, self.exclude = detector_pte, model_dir, exclude
        self.jpeg = None
        self.jpeg_cond = threading.Condition()
        self.meta = json.load(open(os.path.join(model_dir, "labels.json")))
        self.last_seen = 0.

    def run(self):
        try:
            self._run()
        except Exception as e:
            log("vision failed:", repr(e))
            S.set(error=f"vision: {e}")
            S.models["vision"] = "error"

    def _run(self):
        import torch
        sys.path.insert(0, VISION)
        from multi import MultiVision, draw_objects, signature
        torch.set_num_threads(4)
        model = MultiVision(self.detector_pte, self.model_dir, exclude=self.exclude,
                            pre_dedupe=.85,
                            max_area=float(os.environ.get("SORTI_MAX_ITEM_AREA", ".45")),
                            # crumpled paper looks like cloth to the classifier (live: clothes .53 / paper .23)
                            class_bias={"paper_waste": 2.5, "clothes": .6},
                            confirm_conf={"paper_waste": .45})
        self.meta = model.meta
        zone = None
        if os.environ.get("SORTI_BASE", "1") != "0":   # only items sitting on the cardboard base are detected
            from base_zone import BaseZone
            zone = BaseZone(os.path.join(ROOT, "pi", "base_zone.json"))   # only its top edge is used: the crop line
        attempt = 0
        while True:                      # at boot the webcam can enumerate late (seen: 120 s) or be probed by WirePlumber
            cap = cv2.VideoCapture(self.video if self.video else self.cam)
            if cap.isOpened() or self.video:
                break
            cap.release()
            attempt += 1
            if attempt in (1, 10) or attempt % 30 == 0:
                log(f"camera {self.cam} not available yet, retrying every 2 s")
            S.models["vision"] = "loading"
            time.sleep(2)
        if not self.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError(f"camera/video {self.video or self.cam} could not be opened")
        S.models["vision"] = "ready"
        log("multi-object vision ready:", self.detector_pte)
        fps_t, fps, ms = time.perf_counter(), 0., 0.
        previous, set_frames = (), 0
        sorted_ids = set()
        while True:
            ok, frame = cap.read()
            if not ok:
                if self.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); model.tracks.clear()
                time.sleep(.05); continue
            if not self.video:
                frame = cv2.flip(frame, 1)
            if busy.locked() and S.status in ("thinking", "loading"):
                # Answering a question: give all 4 cores to STT / language-ID / NLLB / Llama (vision at 4 threads
                # made a Llama answer 22 s instead of ~4 s). Keep the live picture with the last boxes.
                draw_objects(frame, S.objects, ms, 0.)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with self.jpeg_cond:
                        self.jpeg = buf.tobytes(); self.jpeg_cond.notify_all()
                time.sleep(.15)
                fps_t = time.perf_counter()
                continue
            t0 = time.perf_counter()
            seen, board = frame, None
            if zone is not None:
                # the cardboard base: boxes that are mostly cardboard (the board itself) or not over the board
                # (people, monitor, table) are dropped; items lying on it are kept
                board = zone.mask(frame)
                model.zone = (lambda b, m=board, h=frame.shape[0]: zone.on_board(b, m, h))
            objects = model.process(cv2.cvtColor(seen, cv2.COLOR_BGR2RGB))
            if UNKNOWN_AS:                   # owner: a steady item the classifier can't name is treated as plastic
                smap = model.meta["stream_map"]
                for o in objects:
                    if o.get("unsure"):
                        # keep the classifier's best guess (a battery at 58% stays a battery); plastic only if none
                        guess = o["cls"] if o["cls"] in smap and smap[o["cls"]] in C.BINS else UNKNOWN_AS
                        st_ = smap.get(guess)
                        if st_ in C.BINS:
                            o.update(cls=guess, stream=st_, color=C.BINS[st_]["color"], confirmed=True, unsure=False)
            ms = (time.perf_counter() - t0) * 1000
            now = time.perf_counter(); fps = .9 * fps + .1 / max(now - fps_t, 1e-6); fps_t = now
            lang = S.lang
            for o in objects:
                o.update(title=C.TITLES[o['cls']][lang], bin=C.BIN_SHORT[o['stream']][lang])
            S.objects = objects
            for o in objects:                # Arduino: each item, once, as soon as the camera confirms it on the base
                if o["confirmed"] and o["id"] not in sorted_ids and o["cls"] != "background":
                    sorted_ids.add(o["id"])
                    SORTER.sort(o["cls"], o["stream"])
            sig = signature(objects)
            all_ready = bool(sig) and all(o['confirmed'] for o in objects)
            set_frames = set_frames + 1 if sig == previous and all_ready else 0
            previous = sig
            stable = all_ready and set_frames >= 1     # confirmed tracks are already 6-frame steady
            # Preserve the browser's single-card schema; expose all tracks alongside it.
            primary = next((o for o in objects if o['confirmed']), objects[0] if objects else None)
            S.pred = (dict(primary, stable=stable, ms=round(ms, 1), fps=round(fps, 1),
                           objects=objects, top3=[dict(cls=o['cls'], title=o['title'], p=o['conf'])
                                                  for o in objects]) if primary else None)
            if sig:
                self.last_seen = time.time()
            if stable and S.chosen and S.auto and S.status == "watching" and sig != S.announced_signature:
                threading.Thread(target=announce_objects, args=(list(objects),), daemon=True).start()
            elif not sig and time.time() - self.last_seen > 4 and S.status == "watching":
                S.item = None; S.announced_signature = ()
            draw_objects(frame, objects, ms, fps)
            if board is not None:               # outline the erased base on the live feed
                cs = cv2.findContours(board.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(frame, cs, -1, (184, 212, 0), 1)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.jpeg_cond:
                    self.jpeg = buf.tobytes(); self.jpeg_cond.notify_all()


# ------------------------------------------------------------------ speech
TA_SLOW = 1.0          # Tamil speaking pace: >1 = slower (1.1 = 10% slower, user asked "slightly slow")
TA_VOICE = "female"     # Tamil TTS: female | male (Piper, AI4Bharat Rasa, CC-BY-4.0) | mms (old, choppy)


from tts_utils import tighten  # noqa: E402
import stt_fix  # noqa: E402


class Speech:
    """Loads only the active language's STT + TTS (memory budget on the Pi)."""

    def __init__(self):
        self.lang = None
        self.no_speaker = False
        self.stt = self.tts = None
        self.lock = threading.Lock()
        self.player = None
        self._halt = False
        self.cache = {}                  # (lang, english, text) -> wav kept on disk: intros/prompts play instantly
        self.synth_lock = threading.Lock()

    def load(self, lang):
        with self.lock:
            if self.lang == lang:
                return
            S.models.update(stt="loading", tts="off" if self.no_speaker else "loading")
            self.stt = self.tts = None
            self.lang = None
            trim()
            import sherpa_onnx
            t0 = time.time()
            if lang == "detect":
                # welcome screen: multilingual Whisper with auto language-ID; no TTS (greetings are cached wavs)
                d = os.path.join(VOICE, "stt_whisper_base_en") + "/"
                self.stt = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=d + "base-encoder.int8.onnx", decoder=d + "base-decoder.int8.onnx",
                    tokens=d + "base-tokens.txt", language="", task="transcribe", num_threads=4)
                self.lang = lang
                S.models.update(stt="ready", tts="off")
                log(f"language-detect STT loaded in {time.time() - t0:.1f}s")
                return
            if lang == "en":
                # Moonshine base (int8): 2.3x faster than Whisper base.en and half the word errors on the kiosk
                # eval set (lap/stt_eval, 2026-11-17). Whisper base stays for the welcome language-ID.
                d = os.path.join(VOICE, "stt_moonshine_base_en") + "/"
                self.stt = sherpa_onnx.OfflineRecognizer.from_moonshine(
                    preprocessor=d + "preprocess.onnx", encoder=d + "encode.int8.onnx",
                    uncached_decoder=d + "uncached_decode.int8.onnx", cached_decoder=d + "cached_decode.int8.onnx",
                    tokens=d + "tokens.txt", num_threads=4)
            else:
                # AI4Bharat IndicConformer 120M CTC, int8 sherpa export (MIT/Apache-2.0). Kiosk eval set, clean:
                # WER hi 35->14%, ta 60->26% (CER 30->4%, 38->6%) vs Shruti, and 3-5x faster (lap/stt_eval/INDIC_REPORT.md)
                d = os.path.join(VOICE, f"stt_indicconformer_{lang}")
                self.stt = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                    model=os.path.join(d, "model.int8.onnx"), tokens=os.path.join(d, "tokens.txt"),
                    num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
            if self.no_speaker:
                self.lang = lang
                S.models.update(stt="ready", tts="off")
                log(f"STT for {lang} loaded in {time.time() - t0:.1f}s")
                return
            if lang == "ta" and TA_VOICE == "mms":   # old Meta MMS voice: reads word by word, kept as a fallback
                import onnxruntime as ort
                from transformers import AutoTokenizer
                d = os.path.join(VOICE, "tts_mms_ta_onnx")
                self.tts = ("mms", AutoTokenizer.from_pretrained(d), ort.InferenceSession(os.path.join(d, "model.onnx")))
            else:
                from piper import PiperVoice
                f = {"en": "tts_piper_en/en_US-lessac-medium.onnx", "hi": "tts_piper_hi/hi_IN-pratham-medium.onnx",
                     "ta": f"tts_piper_ta_{TA_VOICE}/ta_IN-rasa_{TA_VOICE}-medium.onnx"}[lang]
                self.tts = ("piper", PiperVoice.load(os.path.join(VOICE, f)))
            self.lang = lang
            S.models.update(stt="ready", tts="ready")
            log(f"speech models for {lang} loaded in {time.time() - t0:.1f}s")

    def english_voice(self):
        if not hasattr(self, "_en_voice"):
            from piper import PiperVoice
            self._en_voice = PiperVoice.load(os.path.join(VOICE, "tts_piper_en/en_US-lessac-medium.onnx"))
        return self._en_voice

    def synth(self, text, english=False):
        hit = self.cache.get((self.lang, english, text))
        if hit and os.path.exists(hit):
            return hit
        with self.synth_lock:
            return self._synth(text, english)

    def warm(self, texts):
        """Pre-synthesise short fixed lines (item intros, prompts) for the active language, in the background."""
        lang = self.lang
        for t in texts:
            if self.lang != lang or self.tts is None:
                return
            key = (lang, False, t)
            if key not in self.cache:
                try:
                    with self.synth_lock:
                        self.cache[key] = self._synth(t)
                except Exception as e:
                    log("tts warm-up failed:", repr(e)); return

    def _synth(self, text, english=False):
        path = os.path.join(TMP, f"say_{time.time_ns()}.wav")
        if english or self.tts[0] == "piper":
            v = self.english_voice() if english and self.lang != "en" else self.tts[1]
            with wave.open(path, "wb") as w:
                cfg = None
                if self.lang == "ta" and not english:
                    from piper import SynthesisConfig
                    cfg = SynthesisConfig(length_scale=TA_SLOW)
                v.synthesize_wav(text, w, syn_config=cfg)
            with wave.open(path, "rb") as w:
                sr = w.getframerate()
                y = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        else:
            _, tok, sess = self.tts
            ids = tok(text, return_tensors="np")["input_ids"].astype("int64")
            y, sr = sess.run(None, {"input_ids": ids})[0][0], SR
        if self.lang == "ta" and not english:
            y = tighten(y, sr, word_gap_ms=110, sentence_gap_ms=320)   # a touch more breathing room for Tamil
        else:
            y = tighten(y, sr)
        y = (np.clip(y / max(1e-6, np.abs(y).max()) * 0.9, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes())
        return path

    SENT = re.compile(r"(?<=[.!?\u0964])\s+")

    def say(self, text, mid=None, english=False):
        """Synthesize and play through the laptop speaker; blocks until done or interrupted.
        Streams sentence by sentence: sentence 1 starts playing while the rest is still being synthesized,
        so the first sound comes ~2-4x sooner on long replies."""
        if self.no_speaker:
            return
        parts, buf = [], ""
        for p in self.SENT.split(text.strip()):           # merge very short bits so prosody stays natural
            buf = (buf + " " + p).strip()
            if len(buf) >= 40:
                parts.append(buf); buf = ""
        if buf:
            if parts and len(buf) < 25:
                parts[-1] += " " + buf
            else:
                parts.append(buf)
        self._halt = False
        q = collections.deque()
        done = threading.Event()

        def produce():
            try:
                for p in parts[1:]:
                    if self._halt:
                        break
                    q.append(self.synth(p, english))
            finally:
                done.set()

        first = self.synth(parts[0], english) if parts else None
        if len(parts) > 1:
            threading.Thread(target=produce, daemon=True).start()
        else:
            done.set()
        S.set(status="speaking", speaking_id=mid)
        path = first
        while path:
            if not self._halt:
                self.player = subprocess.Popen(APLAY + [path])
                self.player.wait()
                self.player = None
            if path not in self.cache.values():
                os.remove(path)
            path = None
            while not path and not (done.is_set() and not q):
                if q:
                    path = q.popleft()
                else:
                    time.sleep(0.02)
        S.set(speaking_id=None)

    def stop(self):
        self._halt = True
        p = self.player
        if p and p.poll() is None:
            p.terminate()

    def play(self, path):
        """Play a cached wav (welcome prompts); blocks until done or interrupted."""
        if self.no_speaker:
            return
        self.player = subprocess.Popen(APLAY + [path])
        self.player.wait()
        self.player = None

    def detect(self, audio):
        """(transcript, whisper language code) - only valid while lang == 'detect'."""
        s = self.stt.create_stream()
        s.accept_waveform(SR, audio)
        self.stt.decode_stream(s)
        return s.result.text.strip(), getattr(s.result, "lang", "") or ""

    def transcribe(self, audio):
        # stt_fix.preprocess() lowers WER but lowered intent+item routing at 5 dB SNR (lap/stt_eval/REPORT.md) -> off
        s = self.stt.create_stream()
        s.accept_waveform(SR, audio)
        t0 = time.perf_counter()
        self.stt.decode_stream(s)
        log(f"STT {self.lang} {time.perf_counter() - t0:.3f}s")
        return stt_fix.correct(s.result.text.strip(), self.lang)   # west->waste, bean->bin, ... (context-gated)


SP = Speech()


# ------------------------------------------------------------------ RAM-managed heavy models
# Budget (laptop 7.7 GB, Pi 5 8 GB): vision ~0.2 GB + one language's STT/TTS 0.4-1.0 GB are always loaded.
# Llama (~2.2 GB) and the NLLB translator (~1.0 GB) load only when a free-form question needs them,
# and unload after IDLE_UNLOAD_S of no use. Before loading, MemAvailable is checked; if short, the
# other idle heavy model is unloaded first, and if still short the question is refused politely.
import ctypes

IDLE_UNLOAD_S = 180
MARGIN_MB = 150
MIC_VOL = "60%"


def mem_avail_mb():
    """MemAvailable plus half the free swap: the kernel can page idle browser/desktop memory out,
    so a load that fits this way is slow for a moment, not a freeze. (Pi: give it a 2 GB swapfile too.)"""
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0]) // 1024
    return m.get("MemAvailable", 0) + m.get("SwapFree", 0) // 2


def rss_mb(pid="self"):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except FileNotFoundError:
        pass
    return 0


def trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class LowMemory(Exception):
    pass


class Heavy:
    """A model that is loaded on demand and unloaded when idle."""
    name, need_mb = "?", 0

    def __init__(self):
        self.obj = None
        self.last = 0.0
        self.lock = threading.RLock()

    def _load(self):
        raise NotImplementedError

    def get(self):
        with self.lock:
            self.last = time.time()
            if self.obj is not None:
                return self.obj
            if mem_avail_mb() < self.need_mb + MARGIN_MB:
                for other in HEAVY:
                    if other is not self:
                        other.unload()
            if mem_avail_mb() < self.need_mb + MARGIN_MB:
                S.models[self.name] = "off"
                raise LowMemory(self.name)
            S.models[self.name] = "loading"
            t0 = time.time()
            try:
                self.obj = self._load()
            except Exception:
                S.models[self.name] = "error"
                raise
            S.models[self.name] = "ready"
            log(f"{self.name} loaded in {time.time() - t0:.1f}s, rss {rss_mb()} MB, avail {mem_avail_mb()} MB")
            self.last = time.time()
            return self.obj

    def unload(self):
        with self.lock:
            if self.obj is None:
                return
            self.obj = None
            trim()
            S.models[self.name] = "off"
            log(f"{self.name} unloaded, rss {rss_mb()} MB")

    def reap(self):
        # keep it warm (no reload delay on the next question) unless RAM is tight or nobody used it for 15 min
        idle = time.time() - self.last
        if self.obj is None or idle < IDLE_UNLOAD_S or (idle < 900 and mem_avail_mb() > 1500):
            return
        if self.lock.acquire(blocking=False):
            try:
                if time.time() - self.last > IDLE_UNLOAD_S:
                    self.unload()
            finally:
                self.lock.release()


# The system prompt is prefilled into the KV cache while idle (LlamaWorker.prime), so its length costs no answer
# latency; only the per-question context does. Bins are deliberately NOT listed: the 1B model mixed the colours up
# ("rinse the bottle and put it in the GREEN bin"); bin answers come only from content.py rules.
# The examples are real chat turns (not "Visitor:/Sorti:" text), so the model no longer echoes the example format.
# They are chosen NOT to overlap the eval questions in llm_eval/questions.json.
FRIEND_SYSTEM = (
    "You are Sorti, a friendly waste-sorting helper at a kiosk in India; a camera shows you the visitor's items. "
    "Reply to the Question in one or two short, simple sentences (under 30 words) with one specific, practical tip. "
    "Use the Fact lines when they help and never contradict them. Never name bins or bin colours; the kiosk tells the bin. "
    "Safety first: never suggest burning, breaking, opening or burying anything, pouring chemicals or oil down drains, "
    "or eating waste; batteries, bulbs and electronics go whole to proper collection. "
    "If the question is not about waste, say so kindly in one sentence and bring it back to waste. "
    "The question is speech-recognised: read west/waist as waste, bean as bin, wait as wet.")
FRIEND_SHOTS = [
    ("Items: Plastic.\nQuestion: what can I do with these plastic covers?",
     "Use them again for shopping or as small bin liners, and keep the rest clean and dry so they can be recycled."),
    ("Items: none in front of the camera.\nQuestion: how do I get my kids to care about waste?",
     "Make it a game: give them a small box for dry waste and let them check the family's sorting every evening."),
    ("Items: E-waste.\nQuestion: can I break open this old phone to take out the parts?",
     "Please don't open it, the battery inside can catch fire. Give the whole phone to an authorised e-waste collector."),
    ("Items: Light bulb.\nQuestion: my mercury thermometer broke, what now?",
     "Open the windows, don't touch the silver beads, and scoop them with stiff paper into a sealed jar for hazardous-waste collection."),
    ("Items: Food waste.\nQuestion: can I give these leftovers to stray dogs?",
     "Plain rice or chapati is fine, but never onion, chocolate, bones or spoilt food. The rest can go into compost."),
    ("Items: none in front of the camera.\nQuestion: who will win the IPL this year?",
     "I only know waste, sorry! But stadiums make lots of plastic waste, so carry your own bottle to the match."),
    ("Items: Paper.\nPrevious question: how do I reuse this?\nYour previous answer: Make paper bags or gift wrap from it."
     "\nQuestion: and after that?",
     "When it's too worn out, tear it into small pieces and add it to your compost."),
]
FRIEND = FRIEND_SYSTEM + "".join(   # system text + example turns, all inside the primed prefix (chat.system_part)
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + u +
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + a for u, a in FRIEND_SHOTS)
BIN_COLOUR = re.compile(r"\b(?:the\s+)?(green|blue|red|yellow|black)\s+(?:dust)?bin", re.I)


def clean_llama(text):
    """The few-shot examples make the 1B model echo 'Visitor: ... Sorti: ...': keep only Sorti's reply."""
    if re.search(r"\bSorti\s*:", text, re.I):
        text = re.split(r"\bSorti\s*:", text, flags=re.I)[-1]
    text = re.split(r"\n\s*(?:Visitor|Items|Fact|Question)\b", text)[0]
    text = re.sub(r"^\s*(?:Visitor\s*:.*?\?\s*)", "", text, flags=re.I | re.S)
    return drop_bin_claims(text.strip().strip('"\u201c\u201d').strip())


def drop_bin_claims(text):
    """Remove sentences where Llama still names a bin colour (it may be wrong); keep at least one sentence."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    keep = [p for p in parts if not BIN_COLOUR.search(p)]
    return " ".join(keep) if keep else re.sub(BIN_COLOUR, "the right bin", text)


class Llama(Heavy):
    """Only for intent 'other'; it never decides a bin. English in, English out."""
    name, need_mb = "llama", 1900   # measured 2026-11-17: RssAnon 1.8 GB warm (+0.34 GB file-backed, reclaimable)

    def _load(self):
        override = os.environ.get("SORTI_LLAMA_PTE")
        if not override and not os.path.isfile(os.path.join(LLAMA, "llama3_2_1b_qlora_int4_xnnpack.pte")):
            spin = os.path.join(LLAMA, "llama3_2_1b_spinquant_int4_xnnpack.pte")   # Pi: public SpinQuant export
            override = spin if os.path.isfile(spin) else None
        model = override or os.path.join(LLAMA, "llama3_2_1b_qlora_int4_xnnpack.pte")
        tokenizer = os.path.join(LLAMA, "checkpoint", "tokenizer.model")
        for path in (model, tokenizer):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        if override:
            from chat import LlamaWorker
            w = LlamaWorker(model, tokenizer)
            w.prime(FRIEND)      # system prompt prefilled while idle; the worker re-primes after every answer
            return w
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
        import executorch.kernels.quantized  # noqa: F401
        from executorch.extension.llm.runner import TextLLMRunner
        return TextLLMRunner(model, tokenizer)

    def ask(self, prompt, max_new_tokens=56, on_partial=None, stop_sentences=None):
        """Short answers: they are spoken aloud, and fewer tokens = less latency on the Pi.
        on_partial(text) gets the answer so far while it is generated (worker mode), for the live chat."""
        from chat import LlamaWorker, ask
        with self.lock:
            runner = self.get()
            if isinstance(runner, LlamaWorker):
                try:
                    out = runner.ask(prompt, max_new_tokens, FRIEND, on_partial=on_partial,
                                     stop_sentences=stop_sentences)
                except ChildProcessError:
                    self.unload()
                    S.models["llama"] = "error"
                    raise
            else:
                out = ask(runner, prompt, max_new_tokens=max_new_tokens, system=FRIEND, stop_sentences=stop_sentences)
        cut = max(out.rfind(". "), out.rfind("! "), out.rfind("? "))
        if not out.rstrip().endswith((".", "!", "?")) and cut > 20:
            out = out[:cut + 1]              # drop a sentence cut off by the token cap
        out = re.sub(r'^\s*(sorti\s*[:\-]\s*)', '', out, flags=re.I).strip().strip('"\u201c\u201d').strip()
        return clean_llama(out)

    def unload(self):
        from chat import LlamaWorker
        with self.lock:
            if isinstance(self.obj, LlamaWorker):
                self.obj.close()
            super().unload()


class Translator(Heavy):
    """NLLB-200 distilled 600M, INT8 CTranslate2 (CC-BY-NC-4.0). Hindi/Tamil <-> English.
    Pi 5 latency: CTranslate2's OpenMP is slower on 4 threads than on 2 there, and English -> hi/ta uses a
    vocabulary map (vmap.txt: English source -> only that language's script tokens), so the 256k-token output
    layer shrinks ~20x: a 45-token answer 9.2 s -> 3.5 s with the same output. One language's map is loaded."""
    name, need_mb = "translate", 1000
    CODES = {"en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml"}
    DIR = os.path.join(ROOT, "translate", "nllb_600m_ct2_int8")
    SCRIPT = {"hi": r"[\u0900-\u097F]", "ta": r"[\u0B80-\u0BFF]"}

    def __init__(self):
        super().__init__()
        self.vlang = self.want = None

    def vmap_dir(self, lang):
        """translate/nllb_vmap_<lang>/: symlinks to the model + vmap.txt (built once)."""
        out = os.path.join(ROOT, "translate", f"nllb_vmap_{lang}")
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(self.DIR):
            if not os.path.lexists(os.path.join(out, f)):
                os.symlink(os.path.join(self.DIR, f), os.path.join(out, f))
        vm = os.path.join(out, "vmap.txt")
        if not (os.path.isfile(vm) and os.path.getsize(vm) > 0):
            rx, punct = re.compile(self.SCRIPT[lang]), re.compile(r"[0-9\W_]+")
            common, native = [], []
            with open(os.path.join(self.DIR, "shared_vocabulary.txt"), encoding="utf-8") as f:
                for tok in (l.rstrip("\n") for l in f):
                    w = tok.replace("\u2581", "")
                    if tok.startswith("<") or w == "" or punct.fullmatch(w):
                        common.append(tok)
                    elif rx.search(w):
                        native.append(tok)
            with open(vm + ".tmp", "w", encoding="utf-8") as f:
                f.write("\t" + " ".join(common + ["eng_Latn", self.CODES[lang]]) + "\n")
                f.write("eng_Latn\t" + " ".join(native) + "\n")
            os.replace(vm + ".tmp", vm)
        return out

    def _load(self):
        model = os.path.join(self.DIR, "model.bin")
        if not os.path.isfile(model):
            raise FileNotFoundError(model)
        import ctranslate2, sentencepiece as spm
        lang = self.want if self.want in self.SCRIPT else S.lang if S.lang in self.SCRIPT else "hi"
        try:
            d, self.vlang = self.vmap_dir(lang), lang
        except OSError as e:
            log("translate vmap unavailable:", repr(e)); d, self.vlang = self.DIR, None
        return (ctranslate2.Translator(d, device="cpu", compute_type="int8", inter_threads=1, intra_threads=2),
                spm.SentencePieceProcessor(model_file=os.path.join(self.DIR, "sentencepiece.bpe.model")))

    def __call__(self, text, src, dst):
        with self.lock:
            other = dst if src == "en" else src
            if self.obj is not None and self.vlang not in (None, other):
                self.unload()                      # language switched: load that language's vocabulary map
            self.want = other
            tr, sp = self.get()
            toks = [self.CODES[src]] + sp.encode(text, out_type=str) + ["</s>"]
            vmap = src == "en" and self.vlang == dst
            r = tr.translate_batch([toks], target_prefix=[[self.CODES[dst]]], beam_size=1, max_decoding_length=160,
                                   use_vmap=vmap)
            return sp.decode(r[0].hypotheses[0][1:]).strip()


LL = Llama()
TR = Translator()


GREETING = re.compile(r"^\W*(hello|hi|hey|hai|good (morning|afternoon|evening)|namaste|namaskar|vanakkam|"
                      r"नमस्ते|नमस्कार|हेलो|हाय|வணக்கம்|ஹலோ|ஹாய்)\b", re.I)
GREET_REPLY = {"en": "Hi! Show me any item and ask me where it goes or how to reuse it.",
               "hi": "हाय! कोई भी चीज़ दिखाओ और पूछो — किस बिन में जाएगी या कैसे दोबारा यूज़ करें।",
               "ta": "ஹாய்! எந்த பொருளையாவது காட்டுங்க, எந்த பின்ல போடணும் இல்ல எப்படி மறுபடி யூஸ் பண்ணலாம்னு கேளுங்க."}


def is_echo(text):
    """True if a heard 'question' is mostly words Sorti itself said in the last 90 s (its speaker leaking into the mic)."""
    words = set(re.findall(r"\w+", text.lower()))
    if len(words) < 2:
        return False
    said = set()
    now = time.time()
    with S.lock:
        for m in S.messages:
            if m["role"] == "assistant" and now - m.get("t", 0) < 90:
                said |= set(re.findall(r"\w+", m["text"].lower()))
        if S.video:
            said |= set(re.findall(r"\w+", str(S.video.get("text", "")).lower()))
    return len(words & said) / len(words) >= .7


def llama_context(scene, named, smap, qid):
    """Ground Llama: the items in front of the camera (name, bin, one real fact from content.py) and the previous
    exchange, so answers are about *this* item and follow-ups make sense. Bin facts come from the rules."""
    lines = []
    items = {}
    for o in scene:
        items[o["cls"]] = items.get(o["cls"], 0) + 1
    if named and named not in items:
        items = {named: 1, **items}
    if items:
        lines.append("Items: " + ", ".join(f"{n} x {C.TITLES[c]['en']}" if n > 1 else C.TITLES[c]["en"]
                                           for c, n in items.items()) + ".")
        lines += [f"Fact about {C.TITLES[c]['en']} ({C.BIN_SHORT.get(smap.get(c), {}).get('en', '')}): {C.FACTS[c]['en']}"
                  for c in items]
    else:
        lines.append("Items: none in front of the camera.")
    prev_q, prev_a = previous_exchange(qid)
    if prev_q and prev_a:
        lines.append(f"Previous question: {prev_q}\nYour previous answer: {prev_a}")
    return "\n".join(lines)


def previous_exchange(qid):
    """(question, answer) in English just before message qid, or (None, None)."""
    prev_q = prev_a = None
    with S.lock:
        for m in S.messages:
            if m["id"] >= qid:
                break
            if m["role"] == "user" and m["kind"] == "question":
                prev_q, prev_a = m.get("translation") or m["text"], None
            elif m["role"] == "assistant" and m["kind"] == "answer" and prev_q:
                prev_a = m.get("en") or m.get("translation") or m["text"]
    return (prev_q, prev_a) if prev_q and prev_a else (None, None)


# Free-form pipeline switches (all on in the kiosk; the eval in llm_eval/ turns them off one by one for the ablation).
FAQ_DIRECT = True     # clear answer-bank match -> verified answer, no Llama
FAQ_GROUND = False    # partial match -> given to Llama as a "Fact:" line. Off: measured no gain on the eval
                      # (36 -> 35 of 46 useful) for +~40 prompt tokens per question (llm_eval/RESULTS.md)
OFF_TOPIC = True      # nothing about waste -> polite redirect, no Llama
ANSWER_GUARD = True   # unsafe / cut-off / echoed Llama answer -> bank answer or the item's fact instead
GROUND_MIN = 0.6


def free_form_en(q_en, scene, named, smap, qid, lang="en", on_partial=None):
    """Answer a free-form question (English text). Returns dict(source, text_en, native, prompt, raw, match, guard):
    native = {en, hi, ta} when the reply is verified text (bank / rules) that needs no translation, else None
    (then text_en is Llama's answer and the caller translates it for hi/ta). Never decides a bin."""
    classes = [o["cls"] for o in scene]
    item = named or (classes[0] if len(set(classes)) == 1 else None)
    prev_q, prev_a = previous_exchange(qid)
    entry, score, direct = faq.match(q_en, item, classes, prev_q, prev_a)
    info = dict(match=entry and entry["id"], score=round(score, 2), direct=direct)
    if direct and FAQ_DIRECT:
        return dict(source="faq", text_en=entry["en"], native=entry, **info)
    if OFF_TOPIC and faq.off_topic(q_en, bool(scene) or bool(named), bool(prev_q)):
        return dict(source="rules", text_en=faq.OFF_TOPIC_REPLY["en"], native=faq.OFF_TOPIC_REPLY, **info)
    context = llama_context(scene, named, smap, qid)
    if FAQ_GROUND and entry is not None and score >= GROUND_MIN:
        context += "\nFact: " + entry["en"]
    prompt = context + "\nQuestion: " + q_en + ("" if lang == "en" else " (answer in one short sentence)")
    raw = LL.ask(prompt, max_new_tokens=60 if lang == "en" else 40, on_partial=on_partial, stop_sentences=2)
    out = dict(source="llama", text_en=raw, native=None, prompt=prompt, raw=raw, **info)
    if ANSWER_GUARD:
        text, problem = faq.guard(raw)
        out.update(text_en=text, guard=problem)
        if problem:                           # fall back to verified text in the visitor's language
            log(f"llama answer rejected ({problem}): {raw!r}")
            if entry is not None and score >= GROUND_MIN:
                out.update(source="faq", text_en=entry["en"], native=entry)
            elif item:
                out.update(source="rules", text_en=C.FACTS[item]["en"], native=C.FACTS[item])
            else:
                out.update(source="rules", text_en=C.PHRASES["no_item"]["en"], native=C.PHRASES["no_item"])
    return out


def warm_translator(lang):
    """Hindi/Tamil chosen: load NLLB (with that language's vocabulary map) in the background right away,
    so the first free-form question doesn't pay the ~7 s cold load."""
    if lang not in ("hi", "ta"):
        return

    def go():
        try:
            with TR.lock:
                if TR.obj is not None and TR.vlang != lang:
                    TR.unload()
                TR.want = lang
                TR.get()
        except Exception as e:
            log("translate warm-up skipped:", repr(e))
    threading.Thread(target=go, daemon=True).start()
HEAVY = [LL, TR]


def reaper():
    while True:
        time.sleep(10)
        if not busy.locked():
            for h in HEAVY:
                h.reap()
            prewarm()


def prewarm():
    """Load Llama (and NLLB for hi/ta) in the background once a language is chosen and RAM is plentiful,
    so the first free-form question does not pay the ~3 s model load."""
    if not getattr(S, "chosen", False) or S.lang not in ("en", "hi", "ta"):
        return
    want = [LL] + ([TR] if S.lang != "en" else [])
    for h in want:
        if h.obj is None and mem_avail_mb() > h.need_mb + 1200 and not busy.locked():
            try:
                h.get(); h.last = time.time()
            except Exception as e:
                log("prewarm skipped:", repr(e))


# ------------------------------------------------------------------ assistant flow
busy = threading.Lock()


# ------------------------------------------------------------------ educational videos (lap/videos, built by videos/build.py)
VIDEOS = os.path.join(HERE, "videos")
_VIDX = {"mtime": 0, "data": {}}
_RECENT = collections.defaultdict(lambda: collections.deque(maxlen=3))


def video_index():
    f = os.path.join(VIDEOS, "index.json")
    try:
        m = os.path.getmtime(f)
    except OSError:
        return {}
    if m != _VIDX["mtime"]:                          # hot-reload as the library grows
        _VIDX.update(mtime=m, data=json.load(open(f)))
    return _VIDX["data"]


def pick_video(cls):
    clips = video_index().get(cls) or []
    if not clips:
        return None
    # camera knows only the type, not the exact item -> its one general video (plastic's is "explainer")
    for gid in ("general_photo", "general", "explainer"):     # real-photo documentary first (2026-11-17)
        gen = [c for c in clips if c["id"] == gid]
        if gen:
            return gen[0]
    fresh = [c for c in clips if c["id"] not in _RECENT[cls]] or clips
    c = fresh[int(time.time() * 1000) % len(fresh)]
    _RECENT[cls].append(c["id"])
    return c


_LID = {"m": None, "lock": threading.Lock()}


def lid():
    """Whisper-base language-ID restricted to en/hi/ta (lid.py); ~200 MB, loaded once."""
    with _LID["lock"]:
        if _LID["m"] is None:
            from lid import LID
            t0 = time.time()
            _LID["m"] = LID(os.path.join(VOICE, "stt_whisper_base_en"))
            log(f"language-ID loaded in {time.time() - t0:.1f}s")
        return _LID["m"]


def spoken_lang(audio, current):
    """Language actually spoken in a question, or None if it's the current one / not sure.
    Switching away needs p >= 0.8; switching TO English needs 0.9 (Tanglish/Hinglish is full of English words)."""
    if audio is None or len(audio) < SR * 0.8:
        return None
    t0 = time.perf_counter()
    try:
        g, p = lid().decide(audio)
    except Exception as e:
        log("language-ID failed:", repr(e)); return None
    log(f"language-ID {g} {time.perf_counter() - t0:.3f}s " +
        " ".join(f"{k}={v:.2f}" for k, v in p.items()) + f" (session {current})")
    if g != current and p[g] >= (0.9 if g == "en" else 0.8):
        return g
    return None


VIDEO_READY = threading.Event()
VIDEO_SKIP = threading.Event()
VIDEO_ENDED = threading.Event()
INTERRUPT = threading.Event()          # a new question / language switch cut the current speech or video short
VIDEO_SHOWN = {}                       # cls -> time its video last played to the end
VIDEO_AGAIN_S = 300                    # don't replay a finished video for the same type within 5 minutes


def maybe_video(classes, lang):
    """After a reply: play the first of these types whose video wasn't watched to the end recently."""
    for cls in dict.fromkeys(classes):
        clip = pick_video(cls)
        if clip and time.time() - VIDEO_SHOWN.get(cls, 0) > VIDEO_AGAIN_S:
            play_video(clip, cls, lang)
            return True
    return False


def play_video(clip, cls, lang):
    """Show the clip (muted, looped by the browser) while the server plays its voice-over on the speaker."""
    m = S.add("assistant", "video", clip["vo"][lang], item=cls, video_id=clip["id"])
    VIDEO_READY.clear(); VIDEO_SKIP.clear()
    vd = dict(id=clip["id"], cls=cls, url="/media/" + clip["video"], poster="/media/" + clip["poster"],
              credit=clip["credit"], title=(clip.get("title") or {}).get(lang, ""), text=clip["vo"][lang], t=None)
    VIDEO_ENDED.clear()
    S.set(status="video", speaking_id=m["id"], video=vd)
    if SP.no_speaker:
        VIDEO_SKIP.wait(90)  # browser signals completion (/api/video/done) or skip (ask button, skip)
        if VIDEO_ENDED.is_set():
            VIDEO_SHOWN[cls] = time.time()
        S.set(video=None, speaking_id=None)
        return
    # A/V sync: the voice-over starts only once the browser has the video buffered (POST /api/video/ready),
    # then video.t = the wall-clock start of the voice-over; the browser keeps currentTime = now - t.
    VIDEO_READY.wait(4.0)
    if S.status != "video" or VIDEO_SKIP.is_set():   # skipped / interrupted while loading
        S.set(video=None, speaking_id=None)
        return
    S.set(video=dict(vd, t=time.time() + 0.05))  # aplay needs ~50 ms to produce sound
    SP.play(os.path.join(VIDEOS, clip["vo_files"][lang]))
    if not VIDEO_SKIP.is_set() and not INTERRUPT.is_set() and S.status == "video":
        VIDEO_SHOWN[cls] = time.time()
    S.set(video=None, speaking_id=None)


# ------------------------------------------------------------------ Arduino 2-servo sorter (pyserial, 9600 baud)
class Sorter(threading.Thread):
    """Sends one command per confirmed item to the Arduino sketch (bio / hazard / ewaste / re) and waits for its
    'Ready for next item.' before the next. Finds the port itself (SORTI_SERIAL, else the first USB serial device),
    and reconnects if the cable is unplugged. Items of other types (general waste) are not sent."""
    CMD = {"biodegradable": "bio", "hazardous": "hazard", "e_waste": "ewaste", "recyclable": "re"}

    def __init__(self):
        super().__init__(daemon=True)
        import queue
        self.q = queue.Queue()
        self.state = {"state": "off", "port": None, "last": None}

    def sort(self, cls, stream):
        cmd = self.CMD.get(stream)
        if not cmd:
            return
        if self.state["state"] == "off":
            log(f"sorter: Arduino not connected, {cls} -> {cmd} not sent"); return
        self.q.put((cls, cmd, time.time()))

    def _port(self):
        import glob
        want = os.environ.get("SORTI_SERIAL")
        if want:
            return want if os.path.exists(want) else None
        ports = sorted(glob.glob("/dev/serial/by-id/*")) + sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
        return ports[0] if ports else None

    def _wait(self, ser, text, timeout):
        buf, t0 = "", time.time()
        while time.time() - t0 < timeout:
            buf += ser.read(256).decode(errors="replace")
            if text in buf:
                return buf
        return None

    def run(self):
        try:
            import serial
        except ImportError:
            log("sorter: pyserial is not installed"); return
        warned = False
        while True:
            port = self._port()
            if not port:
                if not warned:
                    log("sorter: no Arduino serial port found (will keep looking)"); warned = True
                self.state.update(state="off", port=None); time.sleep(3); continue
            try:
                with serial.Serial(port, 9600, timeout=0.2) as ser:      # opening resets the Arduino
                    ready = self._wait(ser, "Ready", 5)
                    self.state.update(state="ready", port=os.path.basename(port))
                    log(f"sorter ready on {port}" + ("" if ready else " (no greeting seen)"))
                    warned = False
                    while True:
                        cls, cmd, queued = self.q.get()
                        if time.time() - queued > 10:          # stale: that item is long gone from the platform
                            log(f"sorter: dropped stale {cls} -> {cmd}"); continue
                        self.state.update(state="sorting", last=cls)
                        t0 = time.time()
                        ser.write((cmd + "\n").encode())
                        done = self._wait(ser, "Ready for next item", 12)
                        log(f"sorter: {cls} -> {cmd} {'done' if done else 'NO reply'} in {time.time() - t0:.1f}s")
                        self.state.update(state="ready")
            except Exception as e:
                log("sorter: serial error, reconnecting:", repr(e))
                self.state.update(state="off"); time.sleep(2)


SORTER = Sorter()


def short_intro(cls, stream, lang):
    return f"{C.TITLES[cls][lang]}! {C.BIN_SHORT[stream][lang]}."


def warm_speech(lang):
    if SP.no_speaker or VIS is None:
        return
    smap = VIS.meta.get("stream_map", {})
    texts = [short_intro(c, smap[c], lang) for c in C.TITLES if c in smap and smap[c] in C.BIN_SHORT]
    texts += [C.PHRASES["ask"][lang], C.PHRASES["not_heard"][lang]]
    threading.Thread(target=SP.warm, args=(texts,), daemon=True).start()


def announce(cls, stream, force=False):
    if not S.chosen or not busy.acquire(blocking=False):
        return
    try:
        INTERRUPT.clear()
        S.last_active = time.time()
        lang = S.lang
        S.announced_signature = (cls,)
        S.item = dict(cls=cls, stream=stream, color=C.BINS[stream]["color"], title=C.TITLES[cls][lang],
                      bin=C.BIN_SHORT[stream][lang], fact=C.FACTS[cls][lang], t=time.time())
        clip = pick_video(cls)
        if clip:   # short spoken intro (item + bin), then the video teaches the rest
            text = C.PHRASES["intro"][lang].format(name=C.NAMES[cls][lang], do=C.BINS[stream][lang])
            ask = C.PHRASES["ask"][lang]
        else:
            text, ask = C.explain(cls, stream, lang)
        m = S.add("assistant", "explain", text, item=cls, stream=stream)
        if SP.no_speaker:
            S.set(status="speaking")
        # spoken: just "Plastic! Yellow bin." (~1.5 s, pre-synthesised); the chat shows the full line and the video explains
        SP.say(short_intro(cls, stream, lang) if clip else text, m["id"])
        if clip and S.status == "speaking":
            maybe_video([cls], lang)
        if S.status in ("speaking", "video") and not INTERRUPT.is_set():   # not cut short by a question
            m2 = S.add("assistant", "prompt", ask)
            SP.say(ask, m2["id"])
    finally:
        S.video = None
        if S.status in ("speaking", "video"):
            S.set(status="watching")
        busy.release()


def announce_objects(objects):
    objects = [o for o in objects if o['confirmed']]
    if not objects:
        return
    if len(objects) == 1:
        announce(objects[0]['cls'], objects[0]['stream'])
        return
    if not S.chosen or not busy.acquire(blocking=False):
        return
    try:
        INTERRUPT.clear()
        S.last_active = time.time()
        S.announced_signature = tuple(sorted(o['cls'] for o in objects))
        first = objects[0]
        text = C.grouped_reply(objects, VIS.meta['stream_map'], S.lang)
        S.item = dict(first, objects=objects, title=text, bin='', fact='', t=time.time())
        m = S.add('assistant', 'explain', text, objects=objects, source='rules')
        if SP.no_speaker:
            S.set(status='speaking')
        SP.say(text, m['id'])
        if S.status == 'speaking' and maybe_video([o['cls'] for o in objects], S.lang) \
                and S.status in ('speaking', 'video') and not INTERRUPT.is_set():
            m2 = S.add('assistant', 'prompt', C.PHRASES['ask'][S.lang])
            SP.say(C.PHRASES['ask'][S.lang], m2['id'])
    finally:
        S.video = None
        if S.status in ('speaking', 'video'):
            S.set(status='watching')
        busy.release()


def pick_mic(want=None):
    """PulseAudio source for arecord. Laptop: prefer the built-in digital mic (DMIC), since the default
    is often the empty headset jack. Pi: a USB mic is usually the default, so None = default."""
    try:
        out = subprocess.run(["pactl", "list", "sources", "short"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = [l.split("\t")[1] for l in out.splitlines() if "\t" in l and ".monitor" not in l]
    pick = next((n for n in names if want and want in n), None) or \
        next((n for n in names if "dmic" in n.lower() or n.endswith("_6__source")), None)
    if pick:
        subprocess.run(["pactl", "set-source-mute", pick, "0"], capture_output=True)
        # +7 dB over the DMIC's 0 dB (46%). At 100% (+20 dB) the room/fan noise floor was rms 0.18, as loud as
        # speech, which wrecked STT and language-ID; at 60% it is ~0.03.
        subprocess.run(["pactl", "set-source-volume", pick, MIC_VOL], capture_output=True)
        log("mic:", pick)
    return pick


def finish_capture(proc, sig):
    if proc.poll() is None:
        proc.send_signal(sig)
    proc.wait()
    if proc.stderr is not None:
        with proc.stderr:
            for line in proc.stderr.read().decode(errors="replace").splitlines():
                if "Interrupted system call" not in line:
                    log("mic:", line)


MAX_REC_S = 12          # a question longer than this is submitted automatically (safety net for a missed stop)
MIC_GAIN = os.environ.get("SORTI_MIC_GAIN", "0")   # USB mic (C-Media, AGC on): 16 = +23.8 dB clipped speech


def set_mic_gain():
    """Raw-ALSA USB mic: (re)apply the capture gain before every question, so a re-plugged mic is not left at 100%."""
    card = re.search(r"CARD=([^,]+)", _MIC_DEV or "")
    if card:
        r = subprocess.run(["amixer", "-q", "-c", card.group(1), "sset", "Mic", MIC_GAIN, "cap"], capture_output=True, text=True)
        if r.returncode:
            log("mic gain:", r.stderr.strip())


class Recorder:
    def __init__(self):
        self.source = None
        self.proc = None
        self.path = os.path.join(TMP, "question.raw")
        self.t0 = 0

    def start(self):
        SP.stop()
        if self.proc:
            return
        if os.path.exists(self.path):
            os.remove(self.path)
        set_mic_gain()
        env = dict(os.environ, **({"PULSE_SOURCE": self.source} if self.source else {}))
        self.proc = subprocess.Popen(
            ARECORD + ["-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw", self.path], env=env,
            stderr=subprocess.PIPE if SP.no_speaker else None)
        self.t0 = time.time()
        S.set(status="listening")
        threading.Thread(target=self._meter, daemon=True).start()

    def _meter(self):
        while self.proc:
            try:
                with open(self.path, "rb") as f:
                    f.seek(max(0, os.path.getsize(self.path) - 3200))
                    x = np.frombuffer(f.read(), np.int16).astype(np.float32) / 32768
                S.mic_level = float(min(1.0, np.sqrt(np.mean(x ** 2) + 1e-9) * 8)) if len(x) else 0.0
            except OSError:
                pass
            if self.proc and time.time() - self.t0 > MAX_REC_S:
                log(f"recording reached {MAX_REC_S}s: submitting it")
                threading.Thread(target=submit_recording, daemon=True).start()
                break
            time.sleep(0.06)
        S.mic_level = 0.0

    def stop(self):
        p, self.proc = self.proc, None
        if not p:
            return None
        time.sleep(0.25)                 # catch the tail of the last word
        finish_capture(p, signal.SIGINT)
        try:
            return np.fromfile(self.path, np.int16).astype(np.float32) / 32768
        except OSError:
            return None


REC = Recorder()


def answer_question(audio, typed=None):
    if not S.chosen:                     # button pressed on the welcome screen: this is the language reply
        return welcome_reply(audio)
    with busy:
        INTERRUPT.clear()
        S.last_active = time.time()
        lang = S.lang
        smap = VIS.meta["stream_map"]
        scene = [o for o in S.objects if o["confirmed"] and o["cls"] != "background" and o["cls"] in smap]
        S.set(status="thinking")
        try:
            if typed is None and (audio is None or len(audio) < SR * 0.4 or np.abs(audio).max() < 0.01):
                m = S.add("assistant", "answer", C.PHRASES["not_heard"][lang], source="rules")
                SP.say(C.PHRASES["not_heard"][lang], m["id"])
                return
            t0 = time.time()
            lid_out = {}
            lid_t = None
            if typed is None:      # language-ID runs alongside STT (separate cores)
                lid_t = threading.Thread(target=lambda: lid_out.update(lang=spoken_lang(audio, lang)), daemon=True)
                lid_t.start()
            text = typed.strip() if typed is not None else SP.transcribe(audio)
            if lid_t is not None:
                lid_t.join(5)
                new = lid_out.get("lang")
                if new:            # visitor is speaking another language: switch to it and re-transcribe
                    log(f"switching language {lang} -> {new} (heard {text!r})")
                    warm_translator(new)
                    S.set(status="loading")
                    SP.load(new)
                    S.lang = lang = new
                    warm_speech(new)
                    if S.item:
                        c, st_ = S.item["cls"], S.item["stream"]
                        if S.item.get('objects'):
                            S.item = dict(S.item, title=C.grouped_reply(S.item['objects'], VIS.meta['stream_map'], new),
                                          bin='', fact='')
                        else:
                            S.item = dict(S.item, title=C.TITLES[c][new], bin=C.BIN_SHORT[st_][new], fact=C.FACTS[c][new])
                    S.set(status="thinking")
                    text = SP.transcribe(audio)
            stt_s = time.time() - t0
            if not text:
                m = S.add("assistant", "answer", C.PHRASES["not_heard"][lang], source="rules")
                SP.say(C.PHRASES["not_heard"][lang], m["id"])
                return
            q = S.add("user", "question", text, stt_s=None if typed is not None else round(stt_s, 2),
                      audio_s=None if typed is not None else round(len(audio) / SR, 1), typed=typed is not None)
            if typed is None and is_echo(text):   # the mic heard Sorti's own speaker (video voice-over / reply)
                log(f"ignored echo of Sorti's own voice: {text!r}")
                with S.lock:
                    S.messages = [x for x in S.messages if x["id"] != q["id"]]
                return
            fixed = C.faq_answer(text, lang)
            if fixed:                        # owner's fixed answers: always the same, in the visitor's language
                S.patch(q["id"], intent="faq:" + fixed[0])
                m = S.add("assistant", "answer", fixed[1], source="rules")
                SP.say(fixed[1], m["id"])
                return
            r = snap(text, lang)
            if r["intent"] in ("which_bin", "repeat") and not r["item"]:
                reply = (C.rule_answer('which_bin', scene[0]['cls'], smap[scene[0]['cls']], lang)
                         if len(scene) == 1 else C.grouped_reply(scene, smap, lang))
                S.patch(q['id'], intent=r['intent'])
                m = S.add('assistant', 'answer', reply, source='rules', objects=scene)
                SP.say(reply, m['id'])
                S.set(status="speaking")
                maybe_video([o['cls'] for o in scene], lang)
                return
            if r["intent"] == "repeat" and S.item:
                r["intent"], r["item"] = "which_bin", S.item["cls"]
            cls = r["item"] or (S.item or {}).get("cls")
            S.patch(q["id"], intent=r["intent"], item=r["item"])
            if r["intent"] != "other":
                if cls is None:
                    reply = C.PHRASES["no_item"][lang]
                else:
                    reply = C.rule_answer(r["intent"], cls, smap[cls], lang)
                m = S.add("assistant", "answer", reply, source="rules", intent=r["intent"], item=cls)
                SP.say(reply, m["id"])
                if cls is not None:
                    S.set(status="speaking")
                    maybe_video([cls], lang)
                return
            if GREETING.search(text) and len(text.split()) <= 5 and not r["item"]:
                reply = GREET_REPLY[lang]
                m = S.add("assistant", "answer", reply, source="rules")
                SP.say(reply, m["id"])
                return
            # free-form question -> translate to English (NLLB) -> Llama -> translate back
            t0 = time.time()
            live, live_box = None, []
            try:
                q_en = TR(text, lang, "en") if lang != "en" else text
                t_q = time.time()
                if lang != "en":
                    S.patch(q["id"], translation=q_en)
                prompt_scene, separate = scene, []
                if scene and not r["item"] and re.search(r"\b(make|reuse|repurpose|upcycle|diy)\b", q_en, re.I):
                    prompt_scene = [o for o in scene if smap[o["cls"]] == "recyclable"]
                    separate = [o for o in scene if smap[o["cls"]] != "recyclable"]
                    if not prompt_scene:
                        reply = C.grouped_reply(separate, smap, lang)
                        m = S.add("assistant", "answer", reply, source="rules", objects=separate)
                        SP.say(reply, m["id"])
                        return
                def on_partial(text):  # English: stream Llama's answer into the chat as it is generated
                    text = clean_llama(text) if text else ""
                    if not text:
                        return
                    if not live_box:   # created lazily: bank / off-topic answers never open a streaming bubble
                        live_box.append(S.add("assistant", "answer", text, source="llama", streaming=True))
                    S.patch(live_box[0]["id"], text=text, t_upd=time.time())
                res = free_form_en(q_en, prompt_scene, r["item"], smap, q["id"], lang,
                                   on_partial=on_partial if lang == "en" else None)
                live = live_box[0] if live_box else None
                a_en = res["text_en"]
                source = "llama" if res["source"] == "llama" else "rules"   # the UI labels bank answers like rules
                if res["source"] == "faq":
                    S.patch(q["id"], intent="faq:" + res["match"])
                t_llm = time.time()
                if res["native"] is not None:   # verified text, already in every language: no NLLB round trip
                    reply = res["native"][lang]
                else:
                    reply = C.casual(TR(a_en, "en", lang), lang) if lang != "en" else a_en
                log(f"free-form {lang} [{source} {res.get('match')} {res.get('score')}]: question->en {t_q - t0:.2f}s, "
                    f"answer {t_llm - t_q:.2f}s ({len(a_en.split())} words), ->{lang} {time.time() - t_llm:.2f}s")
                if separate:
                    reply += "\n\n" + C.grouped_reply(separate, smap, lang)
            except (LowMemory, FileNotFoundError, ChildProcessError) as e:
                log("Free-form model unavailable:", str(e))
                live = live_box[0] if live_box else None
                if live:
                    with S.lock:
                        S.messages = [x for x in S.messages if x["id"] != live["id"]]
                if r["item"]:
                    cls = r["item"]
                    reply = C.rule_answer("which_bin", cls, smap[cls], lang) + " " + C.FACTS[cls][lang]
                elif scene:
                    reply = C.grouped_reply(scene, smap, lang)
                else:
                    reply = C.PHRASES["bins_tip"][lang]
                m = S.add("assistant", "answer", reply, source="rules")
                SP.say(reply, m["id"])
                return
            if live:
                S.patch(live["id"], text=reply, source=source, streaming=False, llm_s=round(time.time() - t0, 2),
                        t_upd=time.time(), en=a_en)
                m = live
            else:
                m = S.add("assistant", "answer", reply, source=source, translation=a_en if lang != "en" else None,
                          llm_s=round(time.time() - t0, 2), en=a_en)
            SP.say(reply, m["id"])
        except Exception as e:
            log("answer failed:", repr(e))
            S.add("assistant", "error", f"Something went wrong: {e}")
        finally:
            S.set(status="watching")


# ------------------------------------------------------------------ welcome: "which language?" answered by voice
AUDIO = os.path.join(HERE, "audio")
IDLE_WELCOME_S = 150                     # kiosk: back to the start after this long with nobody using it
WELCOME_SCREEN = False                   # --welcome: the old start screen that asks the language by voice
LISTEN = {"proc": None}


def welcome_wav(lang):
    return os.path.join(AUDIO, f"welcome_{lang}.wav")


def make_welcome_audio():
    """Synthesize the greetings once, in a child process, so no TTS needs to stay in RAM for them."""
    if all(os.path.exists(welcome_wav(l)) for l in C.LANGS):
        return
    log("making welcome audio (first run only)...")
    subprocess.run([sys.executable, os.path.abspath(__file__), "--make-welcome", "--ta-voice", TA_VOICE, "--ta-slow", str(TA_SLOW)], check=False)


_VAD = {}


def vad_listen(max_s=7.0):
    """Record from the mic until the visitor has said one utterance (Silero VAD). None if nobody spoke."""
    import sherpa_onnx
    if "cfg" not in _VAD:
        c = sherpa_onnx.VadModelConfig()
        c.silero_vad.model = os.path.join(VOICE, "vad", "silero_vad.onnx")
        c.silero_vad.min_silence_duration = 0.7     # end of the reply = 0.7 s of silence
        c.silero_vad.min_speech_duration = 0.25
        c.sample_rate = SR
        _VAD["cfg"] = c
    vad = sherpa_onnx.VoiceActivityDetector(_VAD["cfg"], buffer_size_in_seconds=max_s + 2)
    env = dict(os.environ, **({"PULSE_SOURCE": REC.source} if REC.source else {}))
    p = subprocess.Popen(ARECORD + ["-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"],
                         stdout=subprocess.PIPE, env=env, stderr=subprocess.PIPE if SP.no_speaker else None)
    LISTEN["proc"] = p
    segs, t0, peak = [], time.time(), 0.0
    try:
        while time.time() - t0 < max_s and not S.chosen:
            b = p.stdout.read(1024)                       # 512 samples = 32 ms, Silero's window
            if len(b) < 1024:
                break
            x = np.frombuffer(b, np.int16).astype(np.float32) / 32768
            e = float(np.sqrt(np.mean(x ** 2) + 1e-12)); peak = max(peak, e)
            S.mic_level = min(1.0, e * 8)
            vad.accept_waveform(x)
            while not vad.empty():
                segs.append(np.array(vad.front.samples, np.float32)); vad.pop()
            if segs and not vad.is_speech_detected():
                break                                     # one complete utterance captured
        if not segs:
            vad.flush()
            while not vad.empty():
                segs.append(np.array(vad.front.samples, np.float32)); vad.pop()
    finally:
        finish_capture(p, signal.SIGTERM)
        LISTEN["proc"] = None
        S.mic_level = 0.0
    speech = sum(len(x) for x in segs) / SR
    log(f"vad: {len(segs)} segment(s), {speech:.2f}s speech, peak {peak:.3f}")
    if speech < 0.3:
        return None
    pad = np.zeros(SR // 5, np.float32)
    return np.concatenate([pad] + segs + [pad])


ALWAYS_LISTEN = False       # main() turns it on by default: the mic is always on (the visitor uses the mic's own switch); --push-to-talk = old
DEAF = ("speaking", "video", "thinking", "loading")   # Sorti is talking/answering: ignore the mic (no self-hearing)


def always_listen():
    """Continuous mic + Silero VAD. Each utterance (ended by 0.7 s of silence, max 10 s) becomes a question.
    The mic is ignored while Sorti speaks, plays a video or answers, and for 0.6 s after, so the speaker can't
    trigger it. Restarts arecord if the mic is unplugged."""
    import sherpa_onnx
    c = sherpa_onnx.VadModelConfig()
    c.silero_vad.model = os.path.join(VOICE, "vad", "silero_vad.onnx")
    c.silero_vad.min_silence_duration = 0.7
    c.silero_vad.min_speech_duration = 0.3
    c.silero_vad.max_speech_duration = 10
    c.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=14)
    while True:
        set_mic_gain()
        env = dict(os.environ, **({"PULSE_SOURCE": REC.source} if REC.source else {}))
        p = subprocess.Popen(ARECORD + ["-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        log("always listening on the mic")
        deaf_until, was_deaf, talking = 0.0, False, False
        hist, fed = np.zeros(0, np.float32), 0          # audio since the last VAD reset (last 15 s), for pre-roll
        try:
            while True:
                b = p.stdout.read(1024)                       # 512 samples = 32 ms (Silero's window)
                if len(b) < 1024:
                    break
                x = np.frombuffer(b, np.int16).astype(np.float32) / 32768
                S.mic_level = float(min(1.0, np.sqrt(np.mean(x ** 2) + 1e-12) * 8))
                now = time.time()
                deaf = not S.chosen or S.status in DEAF or busy.locked() and S.status != "listening"
                if deaf:
                    deaf_until = now + 1.5       # Bluetooth plays ~0.3 s late + room echo: don't hear Sorti's own tail
                if deaf or now < deaf_until:
                    if not was_deaf:
                        vad.reset(); was_deaf = True; talking = False
                        hist, fed = np.zeros(0, np.float32), 0
                    continue
                was_deaf = False
                vad.accept_waveform(x)
                hist = np.concatenate([hist, x])[-SR * 15:]; fed += len(x)
                if vad.is_speech_detected() and not talking:
                    talking = True
                    S.set(status="listening"); S.last_active = now
                while not vad.empty():
                    seg = np.array(vad.front.samples, np.float32); start = vad.front.start; vad.pop()
                    talking = False
                    # Silero trims the first syllable: take the utterance again from the rolling buffer, -0.3/+0.2 s
                    a, b = start - SR * 3 // 10, start + len(seg) + SR // 5
                    base = fed - len(hist)
                    if a - base >= 0 or base == 0:
                        seg = hist[max(0, a - base):max(0, b - base)]
                    if len(seg) < SR * 0.45:                   # a cough / click
                        if S.status == "listening":
                            S.set(status="watching")
                        continue
                    pad = np.zeros(SR // 10, np.float32)
                    audio = np.concatenate([pad, seg, pad])
                    log(f"heard {len(seg) / SR:.1f}s peak {np.abs(seg).max():.2f} rms {np.sqrt(np.mean(seg ** 2)):.3f}")
                    S.set(status="thinking")
                    threading.Thread(target=lambda a=audio: (_wait_free(), answer_question(a)), daemon=True).start()
                    deaf_until = time.time() + 0.6
        except Exception as e:
            log("always-listen error:", repr(e))
        finally:
            finish_capture(p, signal.SIGTERM)
            S.mic_level = 0.0
            if S.status == "listening":
                S.set(status="watching")
        log("mic stream stopped; restarting in 2 s")
        time.sleep(2)


def stop_listening():
    p = LISTEN["proc"]
    if p and p.poll() is None:
        p.terminate()
        if SP.no_speaker:
            p.wait()


def welcome_reply(audio):
    """Decide the language from a spoken reply; returns True if chosen."""
    if audio is None or len(audio) < SR * 0.3 or SP.lang != "detect":
        return False
    S.set(welcome_phase="detecting", status="thinking")
    if not SP.no_speaker:
        import soundfile as sf
        sf.write(os.path.join(TMP, "welcome_last.wav"), audio, SR)  # for debugging what the mic heard
    text, wl = SP.detect(audio)
    try:
        wl, p = lid().decide(audio)        # en/hi/ta only, English must clearly win (lid.py)
        log("welcome language-ID " + " ".join(f"{k}={v:.2f}" for k, v in p.items()))
    except Exception as e:
        log("language-ID failed, using Whisper's guess:", repr(e))
    text = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", text).strip()   # drop Whisper non-speech tags: [Music], (laughs)
    lang = C.pick_lang(text, wl) if len(text) >= 2 else None
    S.heard = text
    log(f"welcome heard {text!r} whisper={wl!r} -> {lang}")
    if lang:
        threading.Thread(target=choose, args=(lang,), daemon=True).start()
        return True
    S.set(welcome_phase="waiting", status="watching", next_prompt=time.time() + 1.5)
    return False


def choose(lang):
    """Visitor picked a language (by voice or by tapping): load it and greet them in it."""
    with S.lock:
        if S.chosen and S.lang == lang and SP.lang == lang:
            return
        S.chosen = True
        S.welcome_phase = None
        S.lang, S.item = lang, None
        S.announced_signature = ()
        S.status = "loading"
    stop_listening(); SP.stop()
    warm_translator(lang)
    with busy:
        SP.load(lang)
        warm_speech(lang)
        S.last_active = time.time()
        m = S.add("assistant", "prompt", C.READY[lang])
        SP.say(C.READY[lang], m["id"])
        S.set(status="watching")


def reset_session():
    """No welcome screen: a fresh chat in English (language-ID still switches to Hindi/Tamil per question)."""
    stop_listening(); SP.stop(); REC.stop()
    if SP.no_speaker:
        VIDEO_SKIP.set()
    with busy:
        with S.lock:
            S.messages.clear(); S.item = None; S.announced_signature = (); S.heard = None
            S.chosen, S.welcome_phase, S.status = True, None, "loading"
        if SP.lang != "en":
            SP.load("en")
        S.lang = "en"
        S.last_active = time.time()
        S.add("assistant", "prompt", C.READY["en"])
        S.set(status="watching")


def go_welcome():
    """Back to the welcome screen (start-up, 'start over', or nobody used it for a while)."""
    if not WELCOME_SCREEN:
        return reset_session()
    stop_listening(); SP.stop(); REC.stop()
    with busy:
        with S.lock:
            S.chosen = False
            S.messages.clear(); S.item = None; S.announced_signature = (); S.heard = None
            S.status, S.welcome_phase, S.tries = "loading", "waiting", 0
        for h in HEAVY:
            h.unload()
        SP.load("detect")
        S.set(status="watching", next_prompt=time.time() + 0.5)


def welcome_loop():
    while True:
        time.sleep(0.2)
        if S.chosen:
            if (S.status == "watching" and not busy.locked() and REC.proc is None
                    and time.time() - S.last_active > IDLE_WELCOME_S):
                if WELCOME_SCREEN or S.messages:
                    log("idle: starting over")
                    go_welcome()
                else:
                    S.last_active = time.time()
            continue
        if SP.lang != "detect" or REC.proc is not None or busy.locked() or time.time() < S.next_prompt:
            continue
        if not SP.no_speaker:
            S.set(welcome_phase="asking", status="speaking")
            for l in ("en", "hi", "ta"):
                if S.chosen or REC.proc is not None:
                    break
                SP.play(welcome_wav(l))
        if S.chosen or REC.proc is not None:
            continue
        S.set(welcome_phase="listening", status="listening")
        audio = vad_listen()
        if S.chosen or REC.proc is not None:
            continue
        if audio is None or not welcome_reply(audio):
            S.tries += 1
            S.set(welcome_phase="waiting", status="watching",
                  next_prompt=time.time() + (6 if S.tries < 3 else 25))


# ------------------------------------------------------------------ web
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

app = FastAPI()
WEB = os.path.join(HERE, "web")
VIS = None


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/video")
def video():
    def gen():
        last = None
        while True:
            with VIS.jpeg_cond:
                VIS.jpeg_cond.wait(timeout=1)
                j = VIS.jpeg
            if j is None or j is last:
                continue
            last = j
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/pred")
def predictions():
    return S.objects


@app.get("/api/state")
def state(since: int = 0):
    with S.lock:
        worker = getattr(LL.obj, "process", None)
        used_mb = rss_mb() + (rss_mb(worker.pid) if worker else 0)
        return dict(now=time.time(), av_delay=AV_DELAY, sorter=SORTER.state, always_listening=ALWAYS_LISTEN, lang=S.lang, langs=C.LANGS, status=S.status, chosen=S.chosen,
                    no_speaker=SP.no_speaker, welcome_prompt=C.WELCOME if SP.no_speaker and not S.chosen else None,
                    welcome_phase=S.welcome_phase, heard=S.heard, video=S.video, auto=S.auto, pred=S.pred, item=S.item, objects=S.objects,
                    models=S.models, mic_level=round(S.mic_level, 3), speaking_id=S.speaking_id,
                    error=S.error, ram_mb=used_mb, ram_avail_mb=mem_avail_mb(),
                    messages=[m for m in S.messages if m["id"] > since or m.get("t_upd", 0) > time.time() - 2],
                    last_id=S.next_id - 1, rec_s=round(time.time() - REC.t0, 1) if REC.proc else 0)


@app.post("/api/lang")
async def set_lang(req: Request):
    """Tap fallback on the welcome screen, and the header language switch."""
    lang = (await req.json()).get("lang")
    if lang not in C.LANGS:
        return JSONResponse({"error": "bad lang"}, 400)
    if S.chosen and busy.locked() and S.status not in ("speaking", "video"):
        return JSONResponse({"error": "busy"}, 409)      # an answer is being computed
    if S.chosen:
        S.chosen = False                 # let choose() reload + greet in the new language
    # a speech/video in progress is interrupted (choose() waits for it to stop)
    threading.Thread(target=lambda: (_wait_free(), choose(lang)), daemon=True).start()
    return {"ok": True}


@app.post("/api/welcome")
def welcome():
    """Start over: ask the next visitor which language they want."""
    threading.Thread(target=go_welcome, daemon=True).start()
    return {"ok": True}


@app.post("/api/ask/start")
def ask_start():
    if ALWAYS_LISTEN:
        return JSONResponse({"error": "always listening: just speak"}, 409)
    if recording():
        return {"ok": True, "recording": True}      # already listening (idempotent)
    stop_listening()
    if S.models["stt"] != "ready":
        return JSONResponse({"error": "speech models still loading"}, 409)
    if S.status == "thinking":
        return JSONResponse({"error": "busy"}, 409)
    INTERRUPT.set()
    VIDEO_SKIP.set()
    REC.start()
    return {"ok": True}


# Speaker latency the video picture must wait for: Bluetooth (SBC, PipeWire) buffers ~0.25 s, pw-play starts ~50 ms.
def _av_delay():
    if os.environ.get("SORTI_AV_DELAY"):
        return float(os.environ["SORTI_AV_DELAY"])
    try:
        sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        sink = ""
    return 0.3 if "bluez" in sink else 0.05


AV_DELAY = 0.05


def mic_button():
    """USB mic's own button (C-Media HID: KEY_MUTE / VOLUMEDOWN / VOLUMEUP). Hold = push-to-talk (release stops),
    a quick press toggles (press again to stop). The device is grabbed so the desktop doesn't mute the speaker."""
    import fcntl, glob, struct
    EVIOCGRAB, fmt = 0x40044590, "llHHi"
    size = struct.calcsize(fmt)
    want = os.environ.get("SORTI_MIC_BUTTON", "PnP_Sound")
    warned = False
    while True:
        devs = [d for d in glob.glob("/dev/input/by-id/*-event-*") if want in d]
        if not devs:
            if not warned:
                log("mic button: no input device matching", repr(want)); warned = True
            time.sleep(3); continue
        try:
            with open(devs[0], "rb", buffering=0) as f:
                try:
                    fcntl.ioctl(f, EVIOCGRAB, 1)
                except OSError as e:
                    log("mic button: could not grab device:", e)
                log("mic button ready:", os.path.basename(devs[0]))
                warned, pressed_at, started_by_press = False, 0.0, False
                while True:
                    _, _, etype, code, value = struct.unpack(fmt, f.read(size))
                    if etype != 1 or code not in (113, 114, 115):
                        continue
                    if ALWAYS_LISTEN:                    # grabbed only so the desktop doesn't mute the speaker
                        continue
                    if value == 1:                       # pressed
                        pressed_at = time.time()
                        if recording():
                            started_by_press = False
                            submit_recording()
                        else:
                            r = ask_start()
                            started_by_press = isinstance(r, dict) and r.get("ok", False)
                            if not started_by_press:
                                log("mic button: not ready to record")
                    elif value == 0 and started_by_press:  # released after holding: push-to-talk ends
                        started_by_press = False
                        if time.time() - pressed_at > 0.6 and recording():
                            submit_recording()
        except OSError as e:
            log("mic button: device lost, waiting for it:", e)
            time.sleep(2)


def submit_recording():
    audio = REC.stop()
    if audio is None:
        return None
    if len(audio):
        log(f"question audio {len(audio) / SR:.1f}s peak {np.abs(audio).max():.2f} "
            f"rms {np.sqrt(np.mean(audio ** 2)):.3f} clipped {np.mean(np.abs(audio) > .99) * 100:.1f}%")
    S.set(status="thinking")
    threading.Thread(target=lambda: (_wait_free(), answer_question(audio)), daemon=True).start()
    return audio


@app.post("/api/ask/stop")
def ask_stop():
    audio = submit_recording()
    if audio is None:
        return JSONResponse({"error": "not recording"}, 409)
    return {"ok": True, "seconds": round(len(audio) / SR, 1)}


@app.post("/api/ask/text")
async def ask_text(req: Request):
    """Typed question (fallback when the room is too noisy, and for testing without a mic)."""
    text = ((await req.json()).get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, 400)
    threading.Thread(target=lambda: (_wait_free(), answer_question(None, typed=text)), daemon=True).start()
    return {"ok": True}


def _wait_free():
    INTERRUPT.set()
    VIDEO_SKIP.set()
    SP.stop()
    while busy.locked() and S.status in ("speaking", "video"):
        time.sleep(0.05)


@app.post("/api/explain")
def explain_now():
    objects = [o for o in S.objects if o['confirmed']]
    if not objects:
        return JSONResponse({"error": "no prediction"}, 409)
    SP.stop()
    threading.Thread(target=lambda: (_wait_free(), announce_objects(objects)), daemon=True).start()
    return {"ok": True}


@app.post("/api/auto")
async def set_auto(req: Request):
    S.set(auto=bool((await req.json()).get("on")))
    return {"ok": True, "auto": S.auto}


@app.post("/api/video/ready")
def video_ready():
    """The browser has buffered the current clip and is paused at 0 - start the voice-over."""
    VIDEO_READY.set()
    return {"ok": True}


@app.post("/api/sort")
async def sort_now(req: Request):
    """Manual sorter test: {"stream": "recyclable" | "biodegradable" | "hazardous" | "e_waste"}."""
    stream = (await req.json()).get("stream")
    if stream not in Sorter.CMD:
        return JSONResponse({"error": "stream must be one of " + ", ".join(Sorter.CMD)}, 400)
    SORTER.sort("manual", stream)
    return {"ok": True, "sorter": SORTER.state}


@app.post("/api/video/done")
def video_done():
    """The browser played the muted clip to the end (no-speaker mode)."""
    VIDEO_ENDED.set()
    VIDEO_SKIP.set()
    return {"ok": True}


@app.post("/api/video/skip")
def video_skip():
    """Skip the current video (stops its voice-over; the flow continues to 'any questions?')."""
    VIDEO_SKIP.set()
    VIDEO_READY.set()
    SP.stop()
    return {"ok": True}


@app.post("/api/stop")
def stop_speaking():
    INTERRUPT.set()
    VIDEO_SKIP.set()
    SP.stop()
    return {"ok": True}


@app.post("/api/clear")
def clear():
    with S.lock:
        S.messages.clear(); S.item = None; S.announced_signature = ()
    return {"ok": True}


app.mount("/static", StaticFiles(directory=WEB), name="static")
os.makedirs(VIDEOS, exist_ok=True)
app.mount("/media", StaticFiles(directory=VIDEOS), name="media")      # video clips + posters (Range requests OK)


def main():
    global VIS
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pte", default=os.path.join(VISION, "mobilenet_v3_large_int8_xnnpack.pte"))
    ap.add_argument("--threshold", type=float, default=0.70, help="per-frame confirmation confidence (minimum 0.70), for 8 consecutive frames")
    ap.add_argument("--detector-pte", default=os.path.join(VISION, "det", "waste_320.pte"))
    ap.add_argument("--detector-exclude", default=None, help="comma-separated IDs; empty for waste detector")
    ap.add_argument("--model-dir", default=os.path.join(VISION, "v2_taught"))
    ap.add_argument("--video", help="use a video instead of the camera")
    ap.add_argument("--lang", default=None, choices=list(C.LANGS),
                    help="skip the welcome screen and start in this language")
    ap.add_argument("--ta-slow", type=float, default=1.0, help="Tamil pace, >1 is slower")
    ap.add_argument("--ta-voice", default="female", choices=["female", "male", "mms"], help="Tamil voice")
    ap.add_argument("--make-welcome", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--push-to-talk", action="store_true", help="record only while the mic button / UI button is held")
    ap.add_argument("--welcome", action="store_true", help="start screen that asks the visitor's language by voice")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from other computers (Pi)")
    ap.add_argument("--mic", default=None, help="substring of the PulseAudio source name (default: built-in DMIC if present)")
    ap.add_argument("--no-speaker", action="store_true", help="show replies as text and play videos muted")
    a = ap.parse_args()
    global TA_VOICE, TA_SLOW
    TA_VOICE, TA_SLOW = a.ta_voice, a.ta_slow
    SP.no_speaker = a.no_speaker
    global WELCOME_SCREEN
    WELCOME_SCREEN = a.welcome
    if a.no_speaker and a.make_welcome:
        ap.error("--make-welcome requires speaker mode")
    if a.make_welcome:                   # child process: synthesize the greeting wavs, then exit
        import shutil
        os.makedirs(AUDIO, exist_ok=True)
        for l in C.LANGS:
            SP.load(l)
            shutil.move(SP.synth(C.WELCOME[l]), welcome_wav(l))
        return
    S.lang = a.lang or "en"
    if a.no_speaker and _MIC_DEV:
        log("ALSA mic:", _MIC_DEV)
    else:
        REC.source = pick_mic(a.mic)
    exclude = None if a.detector_exclude is None else [int(c) for c in a.detector_exclude.split(',') if c]
    VIS = Vision(a.camera, a.pte, a.threshold, a.detector_pte, os.path.expanduser(a.model_dir), a.video, exclude)
    VIS.start()

    def boot():
        if a.lang or not a.welcome:      # default: no welcome screen, start straight in a language
            SP.load(S.lang)
            S.add("assistant", "prompt", C.READY[S.lang])
            warm_speech(S.lang)
            S.set(chosen=True, status="watching")
        else:
            if not a.no_speaker:
                make_welcome_audio()
            go_welcome()
        threading.Thread(target=welcome_loop, daemon=True).start()
    threading.Thread(target=boot, daemon=True).start()
    threading.Thread(target=reaper, daemon=True).start()
    threading.Thread(target=mic_button, daemon=True).start()
    SORTER.start()
    global ALWAYS_LISTEN
    ALWAYS_LISTEN = not a.push_to_talk and not a.welcome and not a.no_speaker
    if ALWAYS_LISTEN:
        threading.Thread(target=always_listen, daemon=True).start()
    global AV_DELAY
    AV_DELAY = _av_delay()
    log(f"speaker A/V delay {AV_DELAY:.2f}s")
    threading.Thread(target=lid, daemon=True).start()          # language-ID ready before the first reply
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
