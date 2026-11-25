"""Shared speech post-processing (server + video voice-over builder)."""
import numpy as np


def tighten(y, sr, word_gap_ms=70, sentence_gap_ms=220, min_gap_ms=90, thresh=0.025):
    """Make TTS sound fluent: the Tamil MMS voice pauses 0.3-0.6 s between almost every word.
    Shrinks silent gaps; long ones (sentence ends) keep a little more room."""
    y = y.astype(np.float32)
    peak = float(np.abs(y).max()) or 1.0
    w = max(1, int(sr * 0.01))                          # 10 ms frames
    n = len(y) // w
    e = np.sqrt(np.mean((y[:n * w].reshape(n, w) / peak) ** 2, axis=1))
    quiet = e < thresh
    out, i = [], 0
    while i < n:
        j = i
        while j < n and quiet[j] == quiet[i]:
            j += 1
        seg = y[i * w:j * w]
        if quiet[i] and 0 < i and j < n and (j - i) * 10 >= min_gap_ms:
            keep = sentence_gap_ms if (j - i) * 10 >= 450 else word_gap_ms
            k = int(sr * keep / 1000)
            seg = np.concatenate([seg[:k // 2], seg[-(k - k // 2):]])   # keep both edges: no clicks
        out.append(seg)
        i = j
    out.append(y[n * w:])
    return np.concatenate(out)
