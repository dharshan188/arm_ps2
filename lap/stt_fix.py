"""Conservative waste-kiosk transcript repair; no models or I/O at import time."""
import re
import unicodedata
from rapidfuzz.distance import DamerauLevenshtein

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
_CONTEXT = re.compile(r"\b(waste|rubbish|garbage|trash|bin|dustbin|recy\w*|plastic\w*|batter\w*|compost\w*|kitchen|dispos\w*|wet|dry|bulb|paper|glass|metal)\b", re.I)
_LONG = ('recycle', 'recyclable', 'recycling', 'plastic', 'battery', 'batteries', 'compost')
_VALID = {'batter', 'recycled', 'recycles', 'recycler', 'compose', 'composts', 'composted', 'composting'}
# Recorded in stt_eval/results/{hi,ta}_stt_shruti_*.json(l).
# These are development-set rules, not claims about unseen speakers.
_NATIVE = {
    'hi': {'प्रास्टिक':'प्लास्टिक', 'रिसाइकल':'रीसायकल', 'रिसाइकिल':'रीसायकल',
           'री साइकिल':'रीसायकल', 'इबिस्ट':'ई-वेस्ट', 'वैस्ट':'वेस्ट',
           'बिस्ट':'वेस्ट', 'बिन्न':'बिन', 'मिटल':'मेटल', 'घीला':'गीला',
           'बेट्रियाँ':'बैटरियाँ', 'बैठरियाँ':'बैटरियाँ', 'इवेस्ट':'ई-वेस्ट', 'ivest':'e-waste',
           'पीपर':'पेपर', 'विसायकल':'रीसायकल', 'dry west':'ड्राई वेस्ट', 'drywest':'ड्राई वेस्ट'},
    'ta': {'ரீசிக்கில்':'ரீசைக்கிள்', 'ரீசிக்கல்':'ரீசைக்கிள்',
           'ரீசைக்கல்':'ரீசைக்கிள்', 'ரீசிகிள்':'ரீசைக்கிள்', 'வேஸு':'வேஸ்ட்', 'ரீசைகிள்':'ரீசைக்கிள்', 'ரேசிக்கல்':'ரீசைக்கிள்', 'ரீ சேக்கல்':'ரீசைக்கிள்',
           'இவைஸ்ட்னா':'ஈ வேஸ்ட்னா', 'இவ்வேஸ்ட்':'ஈ வேஸ்ட்', 'இவேஷ்':'ஈ வேஸ்ட்',
           'பேட்ரிகளை':'பேட்டரிகளை', 'blastic':'பிளாஸ்டிக்'},
}
_NATIVE_RX = {lang: [(re.compile(r'(?<!\w)' + re.escape(wrong) + r'(?!\w)', re.I), right)
                     for wrong, right in entries.items()] for lang, entries in _NATIVE.items()}


def correct(text, lang):
    """Repair domain homophones only with local cues; preserve unrelated prose."""
    text = unicodedata.normalize('NFC', text)
    if lang in _NATIVE:
        for pattern, right in _NATIVE_RX[lang]:
            text = pattern.sub(right, text)
        # Mixed-script "e waste" is also emitted by Shruti; retain its script.
        text = re.sub(r'\b(?:e\s+waste|ewaste)\b', 'e-waste', text, flags=re.I)
        if lang == 'hi':
            text = re.sub(r'किस (?:दिन|बीम) में (?=डाल)', 'किस बिन में ', text)
            text = re.sub(r'(ब्लू|blue) भिन्न(?=\s+में)', r'\1 बिन', text, flags=re.I)
            if re.search(r'प्लास्टिक|बोतल|अखबार|पेपर|काग[ज़ज]|बैटर|मेटल|कांच|काँच|\b(?:plastic|metal|paper|battery|glass)\b', text, re.I):
                text = re.sub(r'भी (?:साइकिल|cycle)(?=\s+(?:कर|हो))', 'रीसायकल', text)
            if re.search(r'छिलक|चाय|पत्त', text):
                text = text.replace('कम पोस्ट', 'कम्पोस्ट')
        elif lang == 'ta':
            text = re.sub(r'எந்த (?:பின்னில்|பின்னில|பீனில்|pin ல)(?=\s+போட)', 'எந்த பின்ல', text, flags=re.I)
            if re.search(r'கிண|கேன்|டப்பா|ஃபாயில்', text):
                text = text.replace('மெடல்', 'மெட்டல்')
            if re.search(r'பாட்டி|\bbottle\b', text, re.I):
                text = text.replace('ராஸ்டிக்', 'பிளாஸ்டிக்')
            if re.search(r'காய்கறி|தோல|இலை', text):
                text = re.sub(r'\bcompose\b', 'compost', text, flags=re.I)
        return correct(text, 'en')  # Shruti often code-switches loanwords to Latin.
    if lang != 'en':
        return text
    text = re.sub(r'\b(blue|green|red|yellow)bin\b', r'\1 bin', text, flags=re.I)
    text = re.sub(r'\bdrywest\b', 'dry waste', text, flags=re.I)
    context = bool(_CONTEXT.search(text))
    # Observed Whisper confusions, restricted to a bin-question frame.
    if context:
        text = re.sub(r'\b(which)\s+(?:pin|thin|then|fin|finn|spin|bend|ben)(?=\s+(?:for|does|as)\b)',
                      r'\1 bin', text, flags=re.I)
        text = re.sub(r'^(what is|is)\s+he\s+waste\b', r'\1 e-waste', text, flags=re.I)
    tokens = list(_WORD.finditer(text))
    words = [m.group().lower() for m in tokens]
    def repair(i, m):
        word = words[i]
        near = words[max(0, i-4):i] + words[i+1:i+5]
        new = word
        if word in ('west', 'waist', 'swast') and (context or any(x in near for x in ('put','throw','sort','reduce','dispose'))):
            following = words[i+1] if i+1 < len(words) else ''
            previous = words[i-1] if i else ''
            directional = {'in','from','to','toward','towards','facing'}
            geographic = word == 'west' and (following in ('bengal','coast','indies','end','side','africa','of','wind','direction')
                or previous in ('go','head','heading','north','south','sets')
                or previous in directional
                or (previous == 'the' and i >= 2 and words[i-2] in directional)
                or (i+1 < len(tokens) and tokens[i+1].group()[0].isupper()))
            anatomical = word == 'waist' and (following in ('hurts','size','pain','measurement') or previous in ('my','her','his','your','around'))
            if not geographic and not anatomical:
                new = 'waste'
        elif word in ('bean','been') and any(x in near for x in ('which','put','throw','dustbin')):
            # "I have been putting" and edible beans are not bins.
            if not (word == 'been' and i and words[i-1] in ('have','has','had')) and not any(x in near for x in ('eat','eating','coffee','kidney','green','beans')):
                new = 'bin'
        elif word == 'wait' and i+1 < len(words) and words[i+1] in ('waste','waist','west','bin','garbage'):
            new = 'wet'
        elif word == 'try' and i+1 < len(words) and words[i+1] in ('waste','bin','garbage'):
            new = 'dry'
        elif word == 'spulb' and any(x in near for x in ('dangerous','toxic','bin','recycle','dry')):
            new = 'bulb'
        elif context and word == 'plastics':
            new = 'plastic'
        elif context and len(word) >= 5 and word not in _LONG and word not in _VALID:
            candidates = [w for w in _LONG if w[0] == word[0] and DamerauLevenshtein.distance(word, w) <= 1]
            if len(candidates) == 1:
                new = candidates[0]
        if new == word:
            return m.group()
        return new.capitalize() if m.group()[0].isupper() else new
    for i in range(len(tokens)-1, -1, -1):
        m = tokens[i]
        text = text[:m.start()] + repair(i,m) + text[m.end():]
    text = re.sub(r'\b(?:e\s+waste|ewaste)\b', 'e-waste', text, flags=re.I)
    text = re.sub(r'\b(?:re cycle|re-cycle|reecycle)\b', 'recycle', text, flags=re.I) if context else text
    return text


def preprocess(audio_float32_16k):
    """DC removal, 20 ms RMS trim with 150 ms pad, peak .9, 250 ms side pads.

    Relative threshold is deliberately conservative; no denoising is attempted.
    Returns an independent mono float32 array, including for silence/empty input.
    """
    import numpy as np
    x = np.asarray(audio_float32_16k, dtype=np.float32).reshape(-1).copy()
    x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    if x.size:
        x -= np.mean(x, dtype=np.float64)
        if float(np.max(np.abs(x))) <= 1e-7:
            return np.zeros(8000, dtype=np.float32)
        n = (len(x)+319)//320
        rms = np.sqrt(np.mean(np.pad(x, (0,n*320-len(x))).reshape(n,320)**2, axis=1))
        active = np.flatnonzero(rms > max(0.003, float(rms.max())*.12))
        if active.size:
            x = x[max(0,int(active[0])*320-2400):min(len(x),(int(active[-1])+1)*320+2400)]
        peak = float(np.max(np.abs(x)))
        if peak > 1e-7:
            x *= .9/peak
    return np.pad(x, (4000,4000)).astype(np.float32, copy=False)
