"""Spoken-language ID restricted to the kiosk's languages (en / hi / ta), from Whisper base's language-token logits.

Whisper's own argmax picks among ~100 languages and often says "en" for Tamil or Tanglish. Here the
probabilities of related languages are pooled (hi <- ur, mr, ne; ta <- ml, te, kn, si), only en/hi/ta are
compared, and English must clearly win before we believe it (Indian kiosk: Tamil/Hindi speakers mixing English
words are the common case). ~0.3 s per call on the laptop; ~200 MB RAM while loaded.
"""
import os
import numpy as np
import onnxruntime as ort

POOL = {"en": ["en"], "hi": ["hi", "ur", "mr", "ne"], "ta": ["ta", "ml", "te", "kn", "si"]}
EN_MARGIN = 0.20          # English is chosen only if p(en) beats the best Indic pool by this much


class LID:
    def __init__(self, model_dir, threads=4):
        o = ort.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(os.path.join(model_dir, "base-encoder.int8.onnx"), o, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(model_dir, "base-decoder.int8.onnx"), o, providers=["CPUExecutionProvider"])
        m = self.enc.get_modelmeta().custom_metadata_map
        self.sot = int(m["sot"])
        codes = m["all_language_codes"].split(","); toks = [int(t) for t in m["all_language_tokens"].split(",")]
        self.lang_tok = dict(zip(codes, toks))
        self.all_toks = np.array(toks)
        from transformers import WhisperFeatureExtractor
        self.fe = WhisperFeatureExtractor(feature_size=int(m.get("n_mels", 80)))

    def probs(self, audio16k):
        """{'en': p, 'hi': p, 'ta': p} (pooled, renormalised over the three)."""
        x = np.asarray(audio16k, np.float32)[: 16000 * 30]
        mel = self.fe(x, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
        # Encode only the speech (at least 10 s), not Whisper's 30 s of padding: Pi 5 1.12 s -> 0.23 s per call,
        # same decisions on 15/15 en/hi/ta Piper questions (and slightly higher confidence).
        frames = min(mel.shape[-1], max(1000, 2 * -(-len(x) // 320)))
        ck, cv = self.enc.run(None, {"mel": mel[:, :, :frames]})
        k = np.zeros((6, 1, 448, 512), np.float32)
        logits = self.dec.run(["logits"], {"tokens": np.array([[self.sot]], np.int64), "in_n_layer_self_k_cache": k,
                                           "in_n_layer_self_v_cache": k.copy(), "n_layer_cross_k": ck,
                                           "n_layer_cross_v": cv, "offset": np.array([0], np.int64)})[0][0, -1]
        l = logits[self.all_toks].astype(np.float64)
        p = np.exp(l - l.max()); p /= p.sum()
        pt = dict(zip(self.all_toks.tolist(), p))
        pool = {g: sum(pt.get(self.lang_tok[c], 0.0) for c in cs if c in self.lang_tok) for g, cs in POOL.items()}
        z = sum(pool.values()) or 1.0
        return {g: v / z for g, v in pool.items()}

    def decide(self, audio16k):
        """(lang, probs). English only when it clearly wins; otherwise the stronger of hi/ta."""
        p = self.probs(audio16k)
        indic = max(("hi", "ta"), key=p.get)
        lang = "en" if p["en"] - p[indic] > EN_MARGIN else indic
        return lang, p
