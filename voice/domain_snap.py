"""Snap a noisy speech transcript to the waste assistant's known items and question types.

The assistant only needs (item, intent), not a perfect transcript. Fuzzy matching on
characters tolerates ASR spelling errors and Tamil/Hindi case endings
(e.g. பேட்டரி / பேட்டரியை / பேட்டரிகள்).

    from domain_snap import snap
    snap("இந்த பேட்டரியை எந்த தொட்டியில் போடணும்", "ta")
    -> {'item': 'battery_waste', 'intent': 'which_bin', 'item_score': 100.0, 'intent_score': 95.0, 'confident': True}

item=None means the user didn't name an item: use the camera classifier's result.
intent='other' means hand the transcript to Llama as a free question.
"""
import re
import unicodedata

from rapidfuzz import fuzz

# class -> language -> keywords (lower-case, as people say them; loanwords included in native script)
ITEMS = {
    "plastic_waste": {
        "en": ["plastic", "bottle", "polythene", "plastic bag", "wrapper", "plastic cover"],
        "ta": ["பிளாஸ்டிக்", "பாட்டில்", "பாலித்தீன்", "கவர்", "பிளாஸ்டிக் பை", "பிளாஸ்டிக் கவர்", "வாட்டர் பாட்டில்"],
        "hi": ["प्लास्टिक", "बोतल", "पॉलीथीन", "पन्नी", "थैली", "पॉलिथीन", "वॉटर बॉटल", "बॉटल"]},
    "battery_waste": {
        "en": ["battery", "batteries", "cell"],
        "ta": ["பேட்டரி", "பாட்டரி", "செல்"],
        "hi": ["बैटरी", "बेटरी", "सेल"]},
    "e_waste": {
        "en": ["phone", "mobile", "charger", "laptop", "cable", "wire", "circuit", "electronic", "e-waste", "earphone", "remote"],
        "ta": ["கைபேசி", "மொபைல்", "போன்", "சார்ஜர்", "லேப்டாப்", "வயர்", "மின்னணு", "ரிமோட்", "எலக்ட்ரானிக்", "ஹெட்போன்", "இயர்போன்", "கேபிள்"],
        "hi": ["मोबाइल", "फोन", "फ़ोन", "चार्जर", "लैपटॉप", "तार", "इलेक्ट्रॉनिक", "ई-कचरा", "रिमोट", "इलेक्ट्रॉनिक्स", "ईयरफ़ोन", "केबल", "ई-वेस्ट"]},
    "light_bulbs": {
        "en": ["bulb", "tube light", "tubelight", "lamp", "cfl"],
        "ta": ["பல்பு", "பல்ப்", "டியூப் லைட்", "விளக்கு", "லைட்", "டியூப்லைட்"],
        "hi": ["बल्ब", "ट्यूबलाइट", "ट्यूब लाइट", "लैंप", "लाइट", "ट्यूब"]},
    "glass_waste": {
        "en": ["glass", "jar", "mirror"],
        "ta": ["கண்ணாடி", "ஜாடி", "கிளாஸ்", "கண்ணாடி பாட்டில்"],
        "hi": ["कांच", "काँच", "शीशा", "शीशे", "जार", "ग्लास"]},
    "metal_waste": {
        "en": ["metal", "tin", "aluminium", "aluminum", "steel", "iron", "tin can", "soda can"],
        "ta": ["உலோக", "டப்பா", "டின்", "அலுமினியம்", "இரும்பு", "ஸ்டீல்", "மெட்டல்", "கேன்", "அலுமினிய"],
        "hi": ["धातु", "टिन", "कैन", "एल्युमिनियम", "लोहा", "स्टील", "मेटल"]},
    "organic_waste": {
        "en": ["food", "vegetable", "fruit", "peel", "leftover", "leaves", "organic", "wet waste", "kitchen waste", "compost"],
        "ta": ["உணவு", "காய்கறி", "பழம்", "பழத்தோல்", "இலை", "சாப்பாடு", "ஃபுட்", "ஃபுட் வேஸ்ட்", "வேஸ்ட் ஃபுட்", "தோல்", "வெட் வேஸ்ட்", "ஈரக் குப்பை", "கிச்சன் வேஸ்ட்"],
        "hi": ["खाना", "खाने", "सब्ज़ी", "सब्जी", "फल", "छिलका", "छिलके", "पत्ते", "जैविक", "फ़ूड", "फूड", "फ़ूड वेस्ट", "गीला कचरा", "वेट वेस्ट", "किचन वेस्ट"]},
    "paper_waste": {
        "en": ["paper", "cardboard", "newspaper", "carton"],
        "ta": ["காகிதம்", "காகித", "பேப்பர்", "அட்டை", "செய்தித்தாள்", "கார்ட்போர்டு", "பாக்ஸ்"],
        "hi": ["कागज़", "कागज", "अखबार", "गत्ता", "कार्डबोर्ड", "पेपर", "न्यूज़पेपर"]},
    "clothes": {
        "en": ["clothes", "cloth", "shirt", "dress", "fabric", "saree"],
        "ta": ["துணி", "ஆடை", "சட்டை", "புடவை", "டிரஸ்", "ட்ரெஸ்", "ஷர்ட்", "சட்ட"],
        "hi": ["कपड़े", "कपड़ा", "कपडे", "कमीज़", "साड़ी", "ड्रेस", "शर्ट"]},
}

# intent -> language -> cue phrases. Order matters only for ties (earlier wins).
INTENTS = {
    "which_bin": {
        "en": ["which bin", "which dustbin", "where do i throw", "where should i put", "goes in", "go in",
               # romanized Tanglish / Hinglish (what English STT writes when a code-mixed question is heard as English)
               "entha bin", "endha bin", "enga podanum", "enga podurathu", "bin la podanum",
               "kis bin", "kaunsa bin", "konsa bin", "kahan dalu", "kahan daalu", "kahan dalna", "bin mein dalu"],
        "ta": ["எந்த தொட்டி", "எந்த தொட்டியில்", "எங்கே போட", "போட வேண்டும்", "போடணும்", "எந்த பின்", "எந்த பின்ல", "எங்க போடணும்", "எங்க போடுறது", "எங்க போடலாம்", "எதுல போடணும்"],
        "hi": ["किस डिब्बे", "कौन से डिब्बे", "कहाँ डालें", "कहां डालें", "कहाँ फेंकें", "डालूं", "किस बिन", "कौन से बिन", "कहाँ डालूँ", "कहां डालूं", "कहाँ फेंकूँ", "कहाँ डालना है"]},
    "recyclable": {
        "en": ["recycle", "recyclable", "reuse"],
        "ta": ["மறுசுழற்சி", "ரீசைக்கிள்", "ரீசைக்கிளிங்"],
        "hi": ["रीसायकल", "रिसायकल", "रीसाइकिल", "पुनर्चक्रण", "रीसायकल हो", "रिसाइकिल"]},
    "hazardous": {
        "en": ["dangerous", "harmful", "toxic", "safe"],
        "ta": ["ஆபத்தான", "ஆபத்து", "தீங்கு", "பாதுகாப்பான", "டேஞ்சரஸ்", "டேஞ்சர்", "சேஃப்"],
        "hi": ["खतरनाक", "ख़तरनाक", "हानिकारक", "ज़हरीला", "सुरक्षित", "डेंजरस", "सेफ़", "सेफ"]},
    "where_to_give": {
        "en": ["where should i give", "where can i give", "collection centre", "collection center", "where to give", "donate",
               "enga kudukanum", "enga kodukanum", "yaar kitta kudukanum", "kahan doon", "kahan du", "kisko doon"],
        "ta": ["எங்கே கொடுக்க", "சேகரிப்பு மையம்", "யாரிடம் கொடுக்க", "தானம்", "எங்க குடுக்கணும்", "எங்க குடுக்கலாம்", "யார்கிட்ட குடுக்கணும்", "கலெக்ஷன் சென்டர்"],
        "hi": ["कहाँ दें", "कहां दें", "संग्रह केंद्र", "किसे दें", "दान", "कहाँ दूँ", "किसको दूँ", "कलेक्शन सेंटर"]},
    "repeat": {
        "en": ["repeat", "say again", "once more"],
        "ta": ["மீண்டும் சொல்", "திரும்ப சொல்", "இன்னொரு தடவ சொல்லுங்க", "திருப்பி சொல்லுங்க", "மறுபடியும் சொல்லுங்க"],
        "hi": ["दोबारा बोलो", "दोबारा बताओ", "फिर से बोलो", "फिर से बताओ", "एक बार और"]},
}

ITEM_MIN, INTENT_MIN = 82, 80

# Phonetic folding: sounds that speech recognisers (and speakers) commonly confuse map to one letter.
_TA_FOLD = str.maketrans({"வ": "ப", "ட": "த", "ற": "ர", "ள": "ல", "ழ": "ல", "ண": "ன", "ந": "ன", "ஜ": "ச", "ஸ": "ச",
                          "ஷ": "ச", "ஹ": "க", "ீ": "ி", "ூ": "ு", "ே": "ெ", "ோ": "ொ", "ஈ": "இ", "ஊ": "உ", "ஏ": "எ", "ஓ": "ஒ"})
_HI_FOLD = str.maketrans({"ख": "क", "घ": "ग", "छ": "च", "झ": "ज", "ठ": "ट", "ढ": "ड", "थ": "त", "ध": "द", "फ": "प",
                          "भ": "ब", "व": "ब", "श": "स", "ष": "स", "ी": "ि", "ू": "ु", "ै": "े", "ौ": "ो", "ँ": "ं",
                          "\u093c": None, "\u094d": None})   # nukta, halant
# English nouns that only make sense as items inside a Tamil/Hindi sentence (code-mixing)
CODEMIX_ITEMS = {"metal_waste": ["can"], "e_waste": ["charger", "mobile", "phone"], "battery_waste": ["battery"]}


def fold(s):
    s = unicodedata.normalize("NFD", s)
    s = unicodedata.normalize("NFC", s.translate(_HI_FOLD))
    return s.translate(_TA_FOLD)


def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" and c != "-" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def _ngrams(words, n_max=3):
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            yield " ".join(words[i:i + n])


# Everyday words that must never fuzzy-match an intent keyword: fold() turns வீட்டுல ("at home") into "பித்துல",
# whose prefix scored 83 against ஆபத்து ("danger"), so "how can I reduce plastic at home?" got the hazard rule.
_COMMON = {"வீட்டுல", "வீட்டில்", "வீட்ல", "வீட்டிலே", "வீடு", "எப்படி", "எப்டி", "எப்புடி",
           "घर", "घरमें", "कैसे", "how", "home"}


def _score(keyword, text, words):
    """Best similarity of keyword against any 1-3 word span. Long keywords may match a word
    prefix (Tamil/Hindi suffixes); keywords of <=3 characters must match a whole word exactly."""
    if len(keyword) <= 3:
        return 100.0 if keyword in words else 0.0
    head, _, tail = keyword.partition("-")
    if tail and len(head) <= 2:      # "e-waste": fuzzy matching would also fire on plain "waste" - require the prefix
        joined = " ".join(words)
        return 100.0 if any(f in joined or f in words for f in (keyword, head + tail, head + " " + tail)) else 0.0
    best = 0.0
    nk = len(keyword.split())
    stem = keyword[:-1] if len(keyword) >= 4 else None   # inflection often changes the final vowel
    spans = list(_ngrams(words, max(1, nk + 1)))
    spans += [sp.replace(" ", "") for sp in spans if " " in sp]       # "கை வேசியை" -> "கைவேசியை"
    common = {fold(norm(w)) for w in _COMMON}
    for span in spans:
        if keyword not in common and any(span == c or span.startswith(c + " ") or (" " not in span and span.startswith(c)
                                                                                  and span[len(c):] in words) for c in common):
            continue
        s = fuzz.ratio(keyword, span)
        if len(span) > len(keyword):      # allow inflectional suffix: compare against the span's prefix
            s = max(s, fuzz.ratio(keyword, span[:len(keyword)]), fuzz.ratio(keyword, span[:len(keyword) + 1]) - 2)
        if stem and span.startswith(stem):
            s = max(s, 95.0)
        best = max(best, s)
        if best == 100:
            break
    return best


def snap(transcript, lang):
    text = fold(norm(transcript))
    words = text.split()
    langs = [lang, "en"] if lang != "en" else ["en"]   # people mix English words into Tamil/Hindi

    def best_of(table, threshold):
        top, top_s, top_len = None, 0.0, 0
        for label, by_lang in table.items():
            kws = [kw for l in langs for kw in by_lang.get(l, [])]
            if table is ITEMS and lang != "en":
                kws += CODEMIX_ITEMS.get(label, [])
            for kw in kws:
                if True:
                    k = fold(norm(kw))
                    s = _score(k, text, words)
                    if (s, len(k)) > (top_s, top_len):   # ties go to the more specific (longer) keyword
                        top, top_s, top_len = label, s, len(k)
        return (top, top_s) if top_s >= threshold else (None, top_s)

    item, item_s = best_of(ITEMS, ITEM_MIN)
    intent, intent_s = best_of(INTENTS, INTENT_MIN)
    return {"item": item, "intent": intent or "other", "item_score": item_s, "intent_score": intent_s,
            "confident": item is not None and intent is not None}


# ---------------- evaluation set: 9 classes x 2 spoken names x 4 question types ----------------
SPOKEN = {  # names used in test utterances (some deliberately NOT identical to the lexicon form)
    "plastic_waste": {"en": ["this plastic bottle", "the polythene bag"], "ta": ["இந்த பிளாஸ்டிக் பாட்டிலை", "இந்த பாலித்தீன் கவரை"], "hi": ["इस प्लास्टिक की बोतल", "इस पॉलीथीन"]},
    "battery_waste": {"en": ["this old battery", "these batteries"], "ta": ["இந்த பழைய பேட்டரியை", "இந்த பேட்டரிகளை"], "hi": ["इस पुरानी बैटरी", "इन बैटरियों"]},
    "e_waste": {"en": ["my broken phone", "this old charger"], "ta": ["உடைந்த கைபேசியை", "இந்த பழைய சார்ஜரை"], "hi": ["टूटा हुआ मोबाइल", "पुराना चार्जर"]},
    "light_bulbs": {"en": ["this bulb", "the old tube light"], "ta": ["இந்த பல்பை", "பழைய டியூப் லைட்டை"], "hi": ["इस बल्ब", "पुरानी ट्यूबलाइट"]},
    "glass_waste": {"en": ["this glass jar", "the broken glass"], "ta": ["இந்த கண்ணாடி ஜாடியை", "உடைந்த கண்ணாடியை"], "hi": ["इस कांच के जार", "टूटा हुआ कांच"]},
    "metal_waste": {"en": ["this tin can", "the aluminium foil"], "ta": ["இந்த டின் டப்பாவை", "இந்த அலுமினியம் தாளை"], "hi": ["इस टिन के कैन", "एल्युमिनियम की पन्नी"]},
    "organic_waste": {"en": ["these vegetable peels", "leftover food"], "ta": ["இந்த காய்கறி தோலை", "மீதமான உணவை"], "hi": ["सब्ज़ी के छिलके", "बचा हुआ खाना"]},
    "paper_waste": {"en": ["this newspaper", "the cardboard box"], "ta": ["இந்த செய்தித்தாளை", "இந்த அட்டை பெட்டியை"], "hi": ["इस अखबार", "गत्ते का डिब्बा"]},
    "clothes": {"en": ["my old clothes", "this torn shirt"], "ta": ["பழைய துணிகளை", "கிழிந்த சட்டையை"], "hi": ["पुराने कपड़े", "फटी हुई कमीज़"]},
}
TEMPLATES = {
    "which_bin": {"en": "Which bin does {x} go in?", "ta": "{x} எந்த தொட்டியில் போட வேண்டும்?", "hi": "{x} को किस डिब्बे में डालें?"},
    "recyclable": {"en": "Can I recycle {x}?", "ta": "{x} மறுசுழற்சி செய்யலாமா?", "hi": "क्या {x} को रीसायकल कर सकते हैं?"},
    "hazardous": {"en": "Is {x} dangerous?", "ta": "{x} வைத்திருப்பது ஆபத்தானதா?", "hi": "क्या {x} खतरनाक है?"},
    "where_to_give": {"en": "Where should I give {x}?", "ta": "{x} எங்கே கொடுக்க வேண்டும்?", "hi": "{x} कहाँ दें?"},
}


def eval_set(lang):
    out = []
    for item, names in SPOKEN.items():
        for name in names[lang]:
            for intent, t in TEMPLATES.items():
                out.append({"text": t[lang].format(x=name), "item": item, "intent": intent})
    return out


if __name__ == "__main__":
    for lang in ("ta", "hi", "en"):
        es = eval_set(lang)
        ok = sum(snap(e["text"], lang)["item"] == e["item"] and snap(e["text"], lang)["intent"] == e["intent"] for e in es)
        print(lang, f"clean-text accuracy {ok}/{len(es)}")


# Held-out set, written AFTER the matcher rules were fixed: new item names and new phrasings.
HELDOUT = {
    "ta": [("இந்த சோடா டின்னை குப்பையில் போடலாமா?", "metal_waste"), ("பழைய லேப்டாப்பை என்ன செய்வது?", "e_waste"),
           ("இந்த வாழைப்பழத் தோல் எங்கே போகும்?", "organic_waste"), ("இந்த அட்டைப்பெட்டியை என்ன செய்யலாம்?", "paper_waste"),
           ("உடைந்த பல்பை தொட்டலாமா?", "light_bulbs"), ("இந்த செல் பேட்டரி ஆபத்தா?", "battery_waste"),
           ("தண்ணீர் பாட்டிலை எங்கே போடுவது?", "plastic_waste"), ("பழைய புடவையை என்ன செய்வது?", "clothes"),
           ("உடைந்த கண்ணாடி டம்ளர் எங்கே போடுவது?", "glass_waste")],
    "hi": [("सोडा का टिन कहाँ फेंकूं?", "metal_waste"), ("पुराने लैपटॉप का क्या करें?", "e_waste"),
           ("केले का छिलका कहाँ जाएगा?", "organic_waste"), ("इस गत्ते का क्या करें?", "paper_waste"),
           ("टूटा बल्ब छू सकते हैं क्या?", "light_bulbs"), ("क्या यह सेल खतरनाक है?", "battery_waste"),
           ("पानी की बोतल कहाँ डालूं?", "plastic_waste"), ("पुरानी साड़ी का क्या करें?", "clothes"),
           ("टूटा कांच का गिलास कहाँ फेंकें?", "glass_waste")],
    "en": [("Where do I throw this soda can?", "metal_waste"), ("What do I do with an old laptop?", "e_waste"),
           ("Where does a banana peel go?", "organic_waste"), ("What can I do with this carton?", "paper_waste"),
           ("Is it safe to touch a broken bulb?", "light_bulbs"), ("Is this cell dangerous?", "battery_waste"),
           ("Where should the water bottle go?", "plastic_waste"), ("What should I do with an old saree?", "clothes"),
           ("Where do I put a broken glass tumbler?", "glass_waste")],
}
