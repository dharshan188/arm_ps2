"""Benchmark offline STT and TTS candidates for Tamil / Hindi / English on CPU.

Every candidate runs in its own subprocess so RAM numbers are isolated.
Speed is measured on CPU with 2 threads (Colab CPU = rough stand-in for a Pi 5; the
Pi's Cortex-A76 cores are somewhat slower per core, so treat Colab RTF as optimistic).

  python speech_bench.py prep  --n 60                 # cache FLEURS test clips (ta, hi, en)
  python speech_bench.py stt   --cand whisper-base --lang ta
  python speech_bench.py tts   --cand mms --lang ta
  python speech_bench.py roundtrip --stt <cand> --lang ta --tts <cand>
"""
import argparse, glob, io, json, os, pickle, re, sys, time, unicodedata, wave
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SPEECH_ROOT", "/content/speech"))
CACHE, OUT = ROOT / "cache", ROOT / "results"
FLEURS_CFG = {"ta": "ta_in", "hi": "hi_in", "en": "en_us"}
THREADS = 2

# ---------- sentences for the waste assistant (TTS + round trip) ----------
SENTENCES = {
    "ta": [
        "இந்த பிளாஸ்டிக் பாட்டிலை நீல தொட்டியில் போடுங்கள்.",
        "பழைய பேட்டரியை சாதாரண குப்பையில் போடாதீர்கள்.",
        "உணவுக் கழிவுகளை பச்சை தொட்டியில் போடவும்.",
        "உடைந்த கைபேசியை மின்னணு கழிவு சேகரிப்பு மையத்தில் கொடுங்கள்.",
        "காகிதத்தை உலர்ந்த நிலையில் மறுசுழற்சிக்கு அனுப்புங்கள்.",
        "கண்ணாடி பாட்டில்களை கவனமாக தனியாக வைக்கவும்.",
        "மின் விளக்குகளில் பாதரசம் இருக்கலாம்.",
        "இது எந்த தொட்டியில் போக வேண்டும்?",
        "உலோக டப்பாக்களை கழுவி மறுசுழற்சி செய்யலாம்.",
        "பழைய துணிகளை தானமாக கொடுக்கலாம்.",
    ],
    "hi": [
        "इस प्लास्टिक की बोतल को नीले डिब्बे में डालें।",
        "पुरानी बैटरी को सामान्य कचरे में न फेंकें।",
        "खाने का कचरा हरे डिब्बे में डालें।",
        "टूटा हुआ मोबाइल ई-कचरा संग्रह केंद्र पर दें।",
        "कागज़ को सूखा रखकर रीसायकल करें।",
        "कांच की बोतलें सावधानी से अलग रखें।",
        "बल्ब में पारा हो सकता है।",
        "यह किस डिब्बे में जाएगा?",
        "धातु के डिब्बों को धोकर रीसायकल किया जा सकता है।",
        "पुराने कपड़े दान कर सकते हैं।",
    ],
    "en": [
        "Please put this plastic bottle in the blue bin.",
        "Do not throw old batteries in the normal dustbin.",
        "Food waste goes into the green bin.",
        "Give the broken phone to an e-waste collection centre.",
        "Keep paper dry before sending it for recycling.",
        "Keep glass bottles separate and handle them carefully.",
        "Tube lights can contain mercury.",
        "Which bin does this go in?",
        "Metal cans can be washed and recycled.",
        "Old clothes can be donated.",
    ],
}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def rss_mb(key="VmRSS"):
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1]) / 1024
    return float("nan")


def to16k(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if sr == 16000:
        return x
    from scipy.signal import resample_poly
    g = np.gcd(int(sr), 16000)
    return resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)


# ---------- data ----------
def prep(n):
    import soundfile as sf
    from datasets import load_dataset, Audio
    CACHE.mkdir(parents=True, exist_ok=True)
    for lang, cfg in FLEURS_CFG.items():
        dst = CACHE / f"fleurs_{lang}.pkl"
        if dst.exists():
            print("cached", dst); continue
        ds, err = None, []
        for how in ("convert", "config"):
            try:
                if how == "convert":
                    ds = load_dataset("parquet", split="test", data_files={
                        "test": f"hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet/{cfg}/test/*.parquet"})
                else:
                    ds = load_dataset("google/fleurs", cfg, split="test")
                break
            except Exception as e:  # noqa
                err.append(f"{how}: {str(e)[:200]}")
        if ds is None:
            raise RuntimeError("could not load FLEURS " + cfg + "\n" + "\n".join(err))
        ds = ds.cast_column("audio", Audio(decode=False))
        items, seen = [], set()
        for ex in ds:
            ref = ex.get("transcription") or ex.get("raw_transcription")
            if ref in seen:          # FLEURS repeats sentences with different speakers
                continue
            a = ex["audio"]
            x, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
            x = to16k(x, sr)
            if not (1.0 < len(x) / 16000 < 28.0):
                continue
            seen.add(ref)
            items.append((x, ref))
            if len(items) >= n:
                break
        pickle.dump(items, open(dst, "wb"))
        print(lang, "clips", len(items), "audio s", round(sum(len(x) for x, _ in items) / 16000))


# ---------- STT backends ----------
def _hf(repo, pattern):
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo, allow_patterns=[pattern])
    return d


def _pick(d, pat):
    f = sorted(glob.glob(os.path.join(d, "**", pat), recursive=True))
    if not f:
        raise FileNotFoundError(f"{pat} not found under {d}")
    return f[0]


def load_stt(cand, lang):
    import sherpa_onnx
    if cand in ("whisper-base", "whisper-tiny", "whisper-small-iitm"):
        if cand == "whisper-base":
            d = _hf("csukuangfj/sherpa-onnx-whisper-base", "base-*")
        elif cand == "whisper-tiny":
            d = _hf("ippocode/indic-asr-onnx", "*whisper-tiny/*")
        else:
            if lang not in ("ta", "hi"):
                raise ValueError("IITM fine-tunes exist for ta/hi only")
            d = _hf("ippocode/indic-asr-onnx", f"*whisper-small-{lang}/*")
        enc, dec = _pick(d, "*encoder.int8.onnx"), _pick(d, "*decoder.int8.onnx")
        tok = _pick(d, "*tokens.txt")
        rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=enc, decoder=dec, tokens=tok, language=lang, task="transcribe", num_threads=THREADS)

        def run(x):
            s = rec.create_stream(); s.accept_waveform(16000, x); rec.decode_stream(s); return s.result.text
        return run
    if cand == "shruti":
        if lang not in ("ta", "hi"):
            raise ValueError("shruti: ta/hi models only")
        d = _hf("udgamlabs/Shruti-ONNX", f"{lang}/sherpa-onnx/*")
        p = os.path.join(d, lang, "sherpa-onnx") + "/"
        rec = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=p + "preprocess.onnx", encoder=p + "encode.onnx",
            uncached_decoder=p + "uncached_decode.onnx", cached_decoder=p + "cached_decode.onnx",
            tokens=p + "tokens.txt", num_threads=THREADS)

        def run(x):
            s = rec.create_stream(); s.accept_waveform(16000, x); rec.decode_stream(s); return s.result.text
        return run
    if cand == "indicconformer":
        if lang not in ("ta", "hi"):
            raise ValueError("indicconformer: Indic languages only")
        import soundfile as sf, tempfile
        from indic_asr_onnx import IndicTranscriber
        tr = IndicTranscriber()
        tmp = tempfile.mktemp(suffix=".wav")

        def run(x):
            sf.write(tmp, x, 16000); return tr.transcribe_ctc(tmp, lang)
        return run
    raise ValueError(cand)


def wer_cer(refs, hyps):
    import jiwer
    R = [normalize(r) or "-" for r in refs]
    H = [normalize(h) or "-" for h in hyps]
    return jiwer.wer(R, H), jiwer.cer(R, H)


def stt(cand, lang, items, tag):
    base = rss_mb()
    t = time.perf_counter(); run = load_stt(cand, lang); load_s = time.perf_counter() - t
    run(items[0][0])  # warm-up
    hyps, dec, aud = [], 0.0, 0.0
    for x, _ in items:
        t = time.perf_counter(); hyps.append(run(x)); dec += time.perf_counter() - t
        aud += len(x) / 16000
    w, c = wer_cer([r for _, r in items], hyps)
    res = {"kind": "stt", "test": tag, "cand": cand, "lang": lang, "n": len(items), "wer": w, "cer": c,
           "rtf": dec / aud, "load_s": load_s, "ram_mb": rss_mb("VmHWM") - base,
           "examples": [{"ref": r, "hyp": h} for (_, r), h in list(zip(items, hyps))[:3]]}
    return res


# ---------- TTS backends ----------
def load_tts(cand, lang):
    """returns synth(text) -> (float32 audio, sr)"""
    if cand == "piper":
        from huggingface_hub import list_repo_files, hf_hub_download
        from piper import PiperVoice
        prefix = {"hi": "hi/hi_IN/", "en": "en/en_US/lessac/"}.get(lang)
        if prefix is None:
            raise ValueError("no Piper voice for " + lang)
        files = [f for f in list_repo_files("rhasspy/piper-voices") if f.startswith(prefix) and f.endswith(".onnx")]
        files = sorted(files, key=lambda f: ("medium" not in f, f))
        onnx = hf_hub_download("rhasspy/piper-voices", files[0])
        hf_hub_download("rhasspy/piper-voices", files[0] + ".json")
        voice = PiperVoice.load(onnx)
        sr = voice.config.sample_rate

        def synth(text):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(text, wf)
                else:
                    voice.synthesize(text, wf)
            buf.seek(0)
            with wave.open(buf) as r:
                a = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            return a, sr
        synth.voice_name = os.path.basename(onnx)
        return synth
    if cand in ("mms", "mms-tam-ft"):
        import torch
        from transformers import VitsModel, AutoTokenizer
        torch.set_num_threads(THREADS)
        repo = "ylacombe/mms-tam-finetuned-monospeaker" if cand == "mms-tam-ft" else \
            {"ta": "facebook/mms-tts-tam", "hi": "facebook/mms-tts-hin", "en": "facebook/mms-tts-eng"}[lang]
        if cand == "mms-tam-ft" and lang != "ta":
            raise ValueError("Tamil only")
        tok = AutoTokenizer.from_pretrained(repo); model = VitsModel.from_pretrained(repo).eval()
        uroman = None
        if getattr(tok, "is_uroman", False):
            import uroman as ur
            uroman = ur.Uroman()

        def synth(text):
            if uroman is not None:
                text = uroman.romanize_string(text)
            inp = tok(text, return_tensors="pt")
            torch.manual_seed(0)
            with torch.no_grad():
                a = model(**inp).waveform[0].numpy()
            return a.astype(np.float32), model.config.sampling_rate
        synth.voice_name = repo
        return synth
    if cand == "mms-onnx":
        return load_mms_onnx(lang)
    raise ValueError(cand)


def load_mms_onnx(lang):
    """Meta MMS VITS voice exported to ONNX (fp32: int8 breaks the HiFi-GAN vocoder)."""
    import onnxruntime as ort, torch
    from transformers import AutoTokenizer, VitsModel
    repo = {"ta": "facebook/mms-tts-tam", "hi": "facebook/mms-tts-hin", "en": "facebook/mms-tts-eng"}[lang]
    tok = AutoTokenizer.from_pretrained(repo)
    path = OUT / f"mms_{lang}.onnx"
    if not path.exists():
        model = VitsModel.from_pretrained(repo).eval()

        class Wrap(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, input_ids): return self.m(input_ids=input_ids).waveform
        ids = tok("வணக்கம் இது ஒரு சோதனை" if lang == "ta" else "test sentence", return_tensors="pt")["input_ids"]
        torch.onnx.export(Wrap(model), (ids,), str(path), input_names=["input_ids"], output_names=["waveform"],
                          dynamic_axes={"input_ids": {1: "t"}, "waveform": {1: "n"}}, opset_version=17, dynamo=False)
        sr_cfg = model.config.sampling_rate
        del model
    else:
        sr_cfg = 16000
    so = ort.SessionOptions(); so.intra_op_num_threads = THREADS; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    uroman = None
    if getattr(tok, "is_uroman", False):
        import uroman as ur
        uroman = ur.Uroman()

    def synth(text):
        if uroman is not None:
            text = uroman.romanize_string(text)
        ids = tok(text, return_tensors="np")["input_ids"].astype(np.int64)
        return sess.run(None, {"input_ids": ids})[0][0].astype(np.float32), sr_cfg
    synth.voice_name = f"{repo} (ONNX fp32, {path.stat().st_size/1e6:.0f} MB)"
    return synth


def tts(cand, lang):
    import soundfile as sf
    base = rss_mb()
    t = time.perf_counter(); synth = load_tts(cand, lang); load_s = time.perf_counter() - t
    synth(SENTENCES[lang][0])
    d = OUT / "tts_audio" / f"{cand}_{lang}"; d.mkdir(parents=True, exist_ok=True)
    gen, aud = 0.0, 0.0
    for i, s in enumerate(SENTENCES[lang]):
        t = time.perf_counter(); a, sr = synth(s); gen += time.perf_counter() - t
        aud += len(a) / sr
        sf.write(d / f"{i:02d}.wav", a, sr)
    return {"kind": "tts", "cand": cand, "lang": lang, "voice": getattr(synth, "voice_name", ""),
            "rtf": gen / aud, "load_s": load_s, "ram_mb": rss_mb("VmHWM") - base,
            "audio_dir": str(d), "n": len(SENTENCES[lang])}


def roundtrip(stt_cand, lang, tts_cand):
    """Domain test: synthesize waste-assistant sentences with TTS, transcribe with STT.
    Low CER => TTS is intelligible AND STT knows the vocabulary."""
    import soundfile as sf
    d = OUT / "tts_audio" / f"{tts_cand}_{lang}"
    items = []
    for i, s in enumerate(SENTENCES[lang]):
        a, sr = sf.read(d / f"{i:02d}.wav", dtype="float32")
        items.append((to16k(a, sr), s))
    r = stt(stt_cand, lang, items, f"roundtrip:{tts_cand}")
    return r


def domain(stt_cand, lang, tts_cand):
    """Spoken waste-assistant questions -> STT -> snapping. Reports raw transcript CER and
    item/intent accuracy with fuzzy snapping vs. a plain exact-keyword baseline."""
    import importlib, sys as _s
    _s.path.insert(0, str(Path(__file__).parent))
    ds = importlib.import_module("domain_snap")
    synth = load_tts(tts_cand, lang)
    tests = ds.eval_set(lang) + [{"text": t, "item": i, "intent": None, "heldout": True} for t, i in ds.HELDOUT[lang]]
    audio = [to16k(*synth(t["text"])) for t in tests]
    run = load_stt(stt_cand, lang)
    rows = []
    for t, a in zip(tests, audio):
        hyp = run(a)
        r = ds.snap(hyp, lang)
        nh = ds.norm(hyp)
        base_item = next((c for c, by in ds.ITEMS.items() for l in ([lang, "en"]) for k in by.get(l, []) if ds.norm(k) in nh), None)
        rows.append({**t, "hyp": hyp, "snap_item": r["item"], "snap_intent": r["intent"], "base_item": base_item})
    tpl = [r for r in rows if not r.get("heldout")]; ho = [r for r in rows if r.get("heldout")]
    acc = lambda rs, f: sum(f(r) for r in rs) / max(1, len(rs))
    _, cer = wer_cer([r["text"] for r in rows], [r["hyp"] for r in rows])
    res = {"kind": "domain", "cand": stt_cand, "tts": tts_cand, "lang": lang, "n": len(tpl), "n_heldout": len(ho),
           "transcript_cer": cer,
           "item_acc_exact_keyword": acc(tpl, lambda r: r["base_item"] == r["item"]),
           "item_acc_snap": acc(tpl, lambda r: r["snap_item"] == r["item"]),
           "intent_acc_snap": acc(tpl, lambda r: r["snap_intent"] == r["intent"]),
           "both_acc_snap": acc(tpl, lambda r: r["snap_item"] == r["item"] and r["snap_intent"] == r["intent"]),
           "heldout_item_acc_snap": acc(ho, lambda r: r["snap_item"] == r["item"]),
           "failures": [{k: r[k] for k in ("text", "hyp", "item", "snap_item", "intent", "snap_intent")}
                        for r in rows if r["snap_item"] != r["item"] or (r["intent"] and r["snap_intent"] != r["intent"])][:12],
           "rows": [{k: r.get(k) for k in ("text", "hyp", "item", "intent", "heldout")} for r in rows]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "stt", "tts", "roundtrip", "domain"])
    ap.add_argument("--cand"); ap.add_argument("--lang"); ap.add_argument("--tts"); ap.add_argument("--stt")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "prep":
        return prep(a.n)
    try:
        if a.mode == "stt":
            items = pickle.load(open(CACHE / f"fleurs_{a.lang}.pkl", "rb"))
            res = stt(a.cand, a.lang, items, "fleurs")
        elif a.mode == "tts":
            res = tts(a.cand, a.lang)
        elif a.mode == "domain":
            res = domain(a.stt, a.lang, a.tts)
        else:
            res = roundtrip(a.stt, a.lang, a.tts)
    except Exception as e:  # record failures instead of crashing the sweep
        import traceback
        res = {"kind": a.mode, "cand": a.cand or a.stt, "lang": a.lang, "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-1500:]}
    name = f"{a.mode}_{(a.cand or a.stt)}_{a.lang}" + (f"_{a.tts}" if a.tts else "")
    json.dump(res, open(OUT / f"{name}.json", "w"), ensure_ascii=False, indent=2)
    short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k not in ("examples", "trace", "failures")}
    print(json.dumps(short, ensure_ascii=False))


if __name__ == "__main__":
    main()
