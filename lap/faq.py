"""Offline answer bank + tiny retrieval + answer guard for Sorti's free-form questions (intent 'other').

Order in server.free_form_en():
  1. match()      a verified answer from faq_bank.json (en/hi/ta written by hand) when the question clearly asks it
                  -> instant, and for hi/ta no Llama + NLLB round trip (was 19-23 s on the Pi)
  2. off_topic()  no waste/item word and nothing to refer back to -> a polite redirect, no Llama
  3. Llama        grounded with the best partial bank match as a "Fact:" line
  4. guard()      the 1B model's answer is checked (unsafe advice, format echo, cut-off, too long); if it fails,
                  the visitor gets the best bank answer or the item's fact from content.py instead

The bank never decides a bin and names no bin colours; bins come only from content.py + stream_map.
Matching is on the English question (after NLLB for hi/ta), with stemmed content words + rapidfuzz for ASR typos.
"""
import json, os, re

from rapidfuzz import fuzz

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = json.load(open(os.path.join(HERE, "faq_bank.json"), encoding="utf-8"))

STOP = set("""a an the is are am was were be been being do does did doing i me my we our you your it its this that these
those there here of to in on at for with by from as and or but if so can could should would will shall may might must
what which who whom whose why how when where whats hows please tell me about some any one thing things just also
really very much many more most then than too have has had get got go goes going put take make made let lets like
okay ok yes no not dont old new don't can't cant isn't
up down out off away all still even into over""".split())
# The question words that decide the answer type; kept although they are common (e.g. "why tape" vs "store").
KEEP = {"why", "how", "long", "make", "most", "who", "what", "made", "do", "take", "not", "use", "made"}
GENERIC = {"what", "how", "do", "use", "not", "who", "take", "made"}   # too vague to anchor a follow-up
FOLLOW_UP = re.compile(r"^\s*(and|but|so|what about|how about|why|how long|that|it|then|also|ok so)\b", re.I)


# Everyday / NLLB-translation variants folded onto the bank's words ("fertilizer" is how NLLB renders உரம் / खाद).
SYNONYM = {"garbage": "waste", "trash": "waste", "rubbish": "waste", "kachra": "waste", "household": "home",
           "house": "home", "fertilizer": "compost", "fertiliser": "compost", "manure": "compost",
           "newsprint": "newspaper"}


def stem(w):
    w = w.lower().replace("’", "'")
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def words(text, keep=()):
    out = []
    for w in re.findall(r"[a-z0-9']+", (text or "").lower().replace("e-waste", "ewaste").replace("e waste", "ewaste")):
        w = w.strip("'")
        if w and (w not in STOP or w in keep):
            out.append(SYNONYM.get(stem(w), stem(w)))
    return out


def _has(word, qwords):
    if word in qwords:
        return True
    return len(word) >= 5 and any(len(q) >= 5 and fuzz.ratio(word, q) >= 86 for q in qwords)


for e in BANK:
    e["_keys"] = [words(k, KEEP) for k in e["keys"]]


def match(q_en, item=None, scene=(), prev_q=None, prev_a=None):
    """Best bank entry for the question: (entry, score, direct). direct=True: answer with it verbatim.
    item: class the visitor named (domain_snap) or the single item in front of the camera; scene: classes on camera.
    Follow-ups ("and at school?", "why?") borrow words from the previous exchange to find the topic, but a verbatim
    answer still needs one of its key words in the new question itself."""
    cur = words(q_en, KEEP)
    follow = bool(prev_q) and (FOLLOW_UP.search(q_en or "") is not None or len(words(q_en)) <= 2)
    pool = cur + (words(prev_q, KEEP) + words(prev_a) if follow else [])
    content = [w for w in set(cur) if w not in KEEP]
    cur_cues = set(cur) - GENERIC
    best = (None, 0.0, False)
    for e in BANK:
        cls = e.get("cls")
        if cls and cls != item and cls not in scene:
            continue                               # item-specific answer for a different item
        for kw in e["_keys"]:
            if not kw:
                continue
            hit = [w for w in kw if _has(w, pool)]
            key_cov = len(hit) / len(kw)
            if content:
                q_cov = sum(_has(w, kw) for w in content) / len(content)
            else:   # "why?", "what can I make?": only answerable from the previous exchange or the item on camera
                q_cov = 1.0 if follow or (cls and cls == item) else 0.0
            score = 0.75 * key_cov + 0.25 * q_cov + 0.01 * len(hit) + (0.02 if cls and cls == item else 0)
            direct = key_cov == 1 and q_cov >= 0.5 and (not follow or any(_has(w, cur_cues) for w in kw))
            if (direct, score) > (best[2], best[1]):
                best = (e, score, direct)
    return best


# ------------------------------------------------------------------ off-topic guard (before Llama)
DOMAIN = re.compile(r"\b(waste|garbage|trash|rubbish|kachra|kuppai|dustbin|bin|bins|recycl\w*|compost\w*|reus\w*|"
                    r"reduc\w*|throw\w*|dispos\w*|dump\w*|sort\w*|segregat\w*|separat\w*|collect\w*|scrap|kabadi\w*|"
                    r"landfill\w*|upcycl\w*|environment\w*|planet|earth|nature|pollut\w*|climate|eco\w*|green|clean\w*|"
                    r"toxic|hazard\w*|danger\w*|harm\w*|safe\w*|burn\w*|drain|donat\w*|sell|repair|plastic|bottle|"
                    r"bag|cover|wrapper|batter\w*|cell|phone|mobile|charger|laptop|cable|wire|electronic\w*|remote|"
                    r"bulb|tube|cfl|led|lamp|glass|jar|mirror|metal|tin|can|cans|alumin\w*|steel|iron|vessel\w*|"
                    r"food|peel\w*|vegetable|fruit|leftover\w*|leaves|leaf|kitchen|organic|wet|dry|paper|cardboard|"
                    r"newspaper|carton|box|cloth\w*|shirt|dress|fabric|saree|fashion|oil|paint|medicine\w*|pad|"
                    r"diaper|mask|flower\w*|thermocol|aerosol|spray|chemical\w*|water|energy|tree\w*|item\w*|stuff|"
                    r"sorti|you|yourself|camera)\b", re.I)
REFERS = re.compile(r"\b(this|these|that|those|it|them|they|one|ones)\b", re.I)
OFF_TOPIC_REPLY = {
    "en": "I'm only good with waste, sorry! Show me an item, or ask me how to reuse, recycle or throw something away.",
    "hi": "सॉरी, मैं बस कचरे के बारे में जानता हूँ! कोई चीज़ दिखाओ, या पूछो उसे कैसे रीयूज़, रीसायकल या सही से फेंकें।",
    "ta": "சாரி, எனக்கு குப்பை விஷயம் மட்டும் தான் தெரியும்! ஏதாவது பொருள காட்டுங்க, இல்ல அத எப்படி ரீயூஸ், ரீசைக்கிள் பண்றது, எப்படி போடுறதுன்னு கேளுங்க.",
}


def off_topic(q_en, has_scene=False, has_prev=False):
    """True when the question has nothing to do with waste and cannot refer to an item or the last answer."""
    if DOMAIN.search(q_en or ""):
        return False
    if (has_scene or has_prev) and REFERS.search(q_en or ""):
        return False
    if has_prev and FOLLOW_UP.search(q_en or ""):
        return False
    return True


# ------------------------------------------------------------------ guard (after Llama)
UNSAFE = [re.compile(p, re.I) for p in (
    r"\bburn(?:ing|t)?\b(?! out)", r"\b(?:set|light)\b[^.!?]{0,12}\bfire\b",
    r"\b(?:open|cut|break|smash|puncture|pierce|crush|dismantle|take apart)\b[^.!?]{0,25}"
    r"\b(?:batter(?:y|ies)|bulbs?|cfls?|tube ?lights?|aerosol|spray can|cells?)\b",
    r"\b(?:eat|drink|swallow|taste)\b", r"\bdye\b",
    r"\b(?:pour|throw|flush|dump)\b[^.!?]{0,30}\b(?:drains?|river|lake|sea|toilet|sink|gutter|nala|canal|pond)\b",
    r"\bbury\b", r"\bmicrowave\b", r"\bvacuum\b", r"\bbare hands?\b",
    r"\bmix\b[^.!?]{0,25}\b(?:bleach|acid|chemicals?|phenyl)\b", r"\bas (?:fuel|firewood)\b",
    r"\bswollen\b[^.!?]{0,30}\b(?:charg|keep using)", r"\bnon-?toxic\b")]
NEG = re.compile(r"\b(never|don'?t|do not|doesn'?t|avoid|not|no|without|can'?t|cannot|shouldn'?t|instead of)\b", re.I)
WARN = re.compile(r"(?<!non-)\b(unsafe|dangerous|danger|illegal|toxic|harmful|poison\w*|fumes?|pollut\w*|hazard\w*)\b", re.I)
ECHO = re.compile(r"\b(?:Visitor|Sorti|Items|Fact(?: about)?|Previous question|Your previous answer|Question|Examples?|"
                  r"User|Assistant)\s*:|<\|", re.I)
MAX_WORDS = 35
NEGATED_STEP = re.compile(r"^\W*no\b[,.!]?\s+(?:just\s+)?(?:pour|burn|throw|open|break|flush|dump|mix|bury|crush|eat)\b", re.I)


def sentences(text):
    return [p for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p]


def unsafe(text):
    for s in sentences(text):
        for rx in UNSAFE:
            m = rx.search(s)
            if m and not NEG.search(s[:m.start()]) and not WARN.search(s):
                return m.group(0)
    return None


def tidy(text):
    """Cut what follows a format echo, keep whole sentences only, at most MAX_WORDS words."""
    text = ECHO.split(text or "")[0].strip().strip('"“”').strip()
    keep, n = [], 0
    for s in sentences(text):
        if not s.endswith((".", "!", "?")):
            break                                   # cut off by the token cap
        w = len(s.split())
        if keep and n + w > MAX_WORDS:
            break
        keep.append(s); n += w
    return " ".join(keep)


def guard(text):
    """(clean_text, problem). problem is None when the answer may be spoken."""
    t = tidy(text)
    if len(t.split()) < 2:
        return t, "empty or cut off"
    if len(t.split()) > MAX_WORDS + 10:
        return t, "too long"
    bad = unsafe(t)
    if bad:
        return t, f"unsafe: {bad}"
    if NEGATED_STEP.search(t):          # "No, pour it down the drain" - says no, then gives the unsafe step anyway
        return t, "contradiction"
    if re.search(r"\b(i don'?t see|i can'?t see|nothing in front|wait,)\b", t, re.I):
        return t, "camera confusion"
    return t, None
