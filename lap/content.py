"""Spoken text for the assistant, in English / Hindi / Tamil.

Tone: a friend talking, not a textbook. Tamil is modern spoken (Chennai) Tamil and Hindi is everyday
Hinglish, both with common English words mixed in. English words are written in the native script
(யெல்லோ பின், रीसायकल) because the Tamil MMS voice cannot read Latin letters, and Tanglish in Tamil
script is how people actually text.

Bin facts come ONLY from here + vision/labels.json stream_map (Llama never decides a bin).
"""
import re

LANGS = {"en": "English", "hi": "हिन्दी", "ta": "தமிழ்"}

NAMES = {
    "battery_waste": {"en": "a battery", "hi": "बैटरी", "ta": "பேட்டரி"},
    "clothes": {"en": "old clothes", "hi": "पुराने कपड़े", "ta": "பழைய டிரஸ்"},
    "e_waste": {"en": "an electronic item", "hi": "इलेक्ट्रॉनिक आइटम", "ta": "எலக்ட்ரானிக் ஐட்டம்"},
    "glass_waste": {"en": "glass", "hi": "ग्लास", "ta": "கிளாஸ்"},
    "light_bulbs": {"en": "a light bulb", "hi": "बल्ब", "ta": "பல்பு"},
    "metal_waste": {"en": "metal", "hi": "मेटल", "ta": "மெட்டல்"},
    "organic_waste": {"en": "food waste", "hi": "खाने का कचरा", "ta": "ஃபுட் வேஸ்ட்"},
    "paper_waste": {"en": "paper", "hi": "पेपर", "ta": "பேப்பர்"},
    "plastic_waste": {"en": "plastic", "hi": "प्लास्टिक", "ta": "பிளாஸ்டிக்"},
}

# Short UI labels (title case) for the item card.
TITLES = {
    "battery_waste": {"en": "Battery", "hi": "बैटरी", "ta": "பேட்டரி"},
    "clothes": {"en": "Clothes", "hi": "कपड़े", "ta": "டிரஸ்"},
    "e_waste": {"en": "E-waste", "hi": "ई-वेस्ट", "ta": "ஈ-வேஸ்ட்"},
    "glass_waste": {"en": "Glass", "hi": "ग्लास", "ta": "கிளாஸ்"},
    "light_bulbs": {"en": "Light bulb", "hi": "बल्ब", "ta": "பல்பு"},
    "metal_waste": {"en": "Metal", "hi": "मेटल", "ta": "மெட்டல்"},
    "organic_waste": {"en": "Food waste", "hi": "फ़ूड वेस्ट", "ta": "ஃபுட் வேஸ்ட்"},
    "paper_waste": {"en": "Paper", "hi": "पेपर", "ta": "பேப்பர்"},
    "plastic_waste": {"en": "Plastic", "hi": "प्लास्टिक", "ta": "பிளாஸ்டிக்"},
}

BINS = {  # stream -> colour token for the UI + what to do with it (a full spoken sentence)
    "biodegradable": {"color": "green",
                      "en": "Put it in the green bin, that's the one for biodegradable waste.",
                      "hi": "इसे ग्रीन बिन में डालो, वो बायोडिग्रेडेबल कचरे वाला है।",
                      "ta": "இத பச்சை பின்ல போடுங்க, அது மக்குற குப்பைக்கு."},
    "recyclable": {"color": "yellow",
                   "en": "Put it in the yellow bin, that's dry waste and it gets recycled.",
                   "hi": "इसे येलो बिन में डालो, ये ड्राई वेस्ट है, रीसायकल हो जाएगा।",
                   "ta": "இத யெல்லோ பின்ல போடுங்க, இது டிரை வேஸ்ட், ரீசைக்கிள் ஆயிடும்."},
    "hazardous": {"color": "red",
                  "en": "Put it in the red bin. It's hazardous, so keep it separate.",
                  "hi": "इसे रेड बिन में डालो। ये खतरनाक है, अलग रखना।",
                  "ta": "இத ரெட் பின்ல போடுங்க. இது டேஞ்சரஸ், தனியா வைங்க."},
    "e_waste": {"color": "grey",
                "en": "Don't put it in any bin. Give it to an e-waste collection centre.",
                "hi": "इसे किसी बिन में मत डालो, ई-वेस्ट कलेक्शन सेंटर पे दे दो।",
                "ta": "இத எந்த பின்லயும் போடாதீங்க, ஈ-வேஸ்ட் கலெக்ஷன் சென்டர்ல குடுங்க."},
}
BIN_SHORT = {
    "biodegradable": {"en": "Green bin", "hi": "ग्रीन बिन", "ta": "பச்சை பின்"},
    "recyclable": {"en": "Yellow bin", "hi": "येलो बिन", "ta": "யெல்லோ பின்"},
    "hazardous": {"en": "Red bin", "hi": "रेड बिन", "ta": "ரெட் பின்"},
    "e_waste": {"en": "E-waste centre", "hi": "ई-वेस्ट सेंटर", "ta": "ஈ-வேஸ்ட் சென்டர்"},
}

# One fact per item, spoken after the bin.
FACTS = {
    "battery_waste": {
        "en": "Batteries have lead, mercury and acid inside. Mixed with normal trash they leak into the soil and water, so keep them aside and tape the ends.",
        "hi": "बैटरी में लेड, मरकरी और एसिड होता है। नॉर्मल कचरे में फेंकोगे तो मिट्टी और पानी में मिल जाता है, तो अलग रखो और दोनों तरफ़ टेप लगा दो।",
        "ta": "பேட்டரில லெட், மெர்க்குரி, ஆசிட் எல்லாம் இருக்கு. சாதா குப்பையோட போட்டா மண்ணுலயும் தண்ணிலயும் கலந்துடும். அதனால தனியா வைங்க, ரெண்டு பக்கமும் டேப் ஒட்டிடுங்க."},
    "clothes": {
        "en": "If they're still wearable, donate them. Torn ones get recycled into rags and fibre. Just keep them dry.",
        "hi": "पहनने लायक हैं तो किसी को डोनेट कर दो। फटे कपड़े पोछे या फ़ाइबर में रीसायकल हो जाते हैं। बस सूखे रखना।",
        "ta": "டிரஸ் இன்னும் நல்லா இருந்தா யாருக்காவது டொனேட் பண்ணுங்க. கிழிஞ்சு போனது துணி, ஃபைபரா ரீசைக்கிள் ஆயிடும். ட்ரையா வெச்சா போதும்."},
    "e_waste": {
        "en": "Electronics have valuable metals like copper and gold, plus some toxic parts. A proper e-waste collector gets them out safely.",
        "hi": "इलेक्ट्रॉनिक्स में कॉपर और गोल्ड जैसे कीमती मेटल होते हैं, और कुछ टॉक्सिक पार्ट्स भी। सही ई-वेस्ट कलेक्टर इन्हें सेफ़ली निकाल लेता है।",
        "ta": "எலக்ட்ரானிக்ஸ்ல காப்பர், கோல்டு மாதிரி காஸ்ட்லி மெட்டல்ஸ் இருக்கு, கூடவே டாக்ஸிக் பார்ட்ஸும் இருக்கு. சரியான ஈ-வேஸ்ட் கலெக்டர் அத சேஃபா எடுத்துடுவாங்க."},
    "glass_waste": {
        "en": "Glass can be recycled again and again without losing quality. Wrap broken glass in paper so nobody cuts their hand.",
        "hi": "ग्लास को कितनी भी बार रीसायकल कर सकते हो, क्वालिटी कम नहीं होती। टूटा ग्लास पेपर में लपेट के डालो ताकि सफ़ाई वाले भैया को चोट न लगे।",
        "ta": "கிளாஸ எத்தன தடவ வேணும்னாலும் ரீசைக்கிள் பண்ணலாம், குவாலிட்டி குறையாது. உடைஞ்ச கிளாஸ பேப்பர்ல சுத்தி போடுங்க, யாருக்கும் கை கிழியாது."},
    "light_bulbs": {
        "en": "CFLs and tube lights have a little mercury in them. Don't break them, and keep them away from kids.",
        "hi": "सीएफएल और ट्यूबलाइट में थोड़ा मरकरी होता है। तोड़ना मत, और बच्चों से दूर रखना।",
        "ta": "சிஎஃப்எல், டியூப் லைட்ல கொஞ்சம் மெர்க்குரி இருக்கு. உடைக்காதீங்க, குழந்தைங்க கைக்கு எட்டாம வைங்க."},
    "metal_waste": {
        "en": "Recycling aluminium saves about ninety five percent of the energy needed to make new metal. Give the cans a quick rinse first.",
        "hi": "एल्युमिनियम रीसायकल करने से नया मेटल बनाने की लगभग पंचानवे परसेंट एनर्जी बचती है। कैन को थोड़ा धो के डालो।",
        "ta": "அலுமினியம் ரீசைக்கிள் பண்ணா, புது மெட்டல் செய்ய ஆகுற எனர்ஜில கிட்டத்தட்ட தொண்ணூத்தஞ்சு பர்சன்ட் மிச்சம். கேன கொஞ்சம் கழுவிட்டு போடுங்க."},
    "organic_waste": {
        "en": "Food scraps and leaves turn into compost in a few weeks. Mixed with dry waste they rot and make methane, which is really bad for the climate.",
        "hi": "खाने का कचरा और पत्ते कुछ हफ़्तों में खाद बन जाते हैं। ड्राई वेस्ट में मिलाओगे तो सड़ के मीथेन गैस बनती है, जो क्लाइमेट के लिए बहुत बुरी है।",
        "ta": "ஃபுட் வேஸ்ட், இலை எல்லாம் சில வாரத்துல உரமா மாறிடும். டிரை வேஸ்ட்டோட கலந்துட்டா அழுகி மீத்தேன் கேஸ் வரும், அது க்ளைமேட்டுக்கு ரொம்ப கெட்டது."},
    "paper_waste": {
        "en": "Paper is biodegradable, it breaks down in a few weeks. Tear it into small pieces and keep plastic tape off it so it composts faster.",
        "hi": "पेपर बायोडिग्रेडेबल है, कुछ हफ़्तों में गल जाता है। छोटे टुकड़े कर दो और प्लास्टिक टेप हटा दो, जल्दी खाद बनेगा।",
        "ta": "பேப்பர் மக்குற பொருள், சில வாரத்துல மக்கிடும். சின்ன துண்டா கிழிச்சு, பிளாஸ்டிக் டேப்ப எடுத்துடுங்க, சீக்கிரம் உரமாகும்."},
    "plastic_waste": {
        "en": "Plastic takes hundreds of years to break down. Give bottles and covers a quick rinse so they get recycled instead of blocking drains.",
        "hi": "प्लास्टिक को गलने में सैकड़ों साल लगते हैं। बोतल और पन्नी थोड़ा धो के डालो, तो रीसायकल हो जाएगी, नाली में जाके जाम नहीं करेगी।",
        "ta": "பிளாஸ்டிக் மக்கி போக நூறு வருஷத்துக்கு மேல ஆகும். பாட்டில், கவர் எல்லாம் கொஞ்சம் கழுவி போட்டா ரீசைக்கிள் ஆயிடும், சாக்கடைல போய் அடைக்காது."},
}

PHRASES = {  # {name} = item, {do} = the bin sentence from BINS
    "intro": {"en": "Hey, that looks like {name}! {do}",
              "hi": "अरे, ये {name} लग रहा है! {do}",
              "ta": "ஹே, இது {name} மாதிரி இருக்கு! {do}"},
    "ask": {"en": "Got any questions? Just press the button and ask me.",
            "hi": "कोई डाउट है? बटन दबाओ और पूछ लो।",
            "ta": "ஏதாவது டவுட் இருக்கா? பட்டன அழுத்தி கேளுங்க."},
    "which_bin": {"en": "{Name}? {do}", "hi": "{name}? {do}", "ta": "{name} தானே? {do}"},
    "recyclable_yes": {"en": "Yep, {name} can be recycled. {do}", "hi": "हाँ, {name} रीसायकल हो जाता है। {do}",
                       "ta": "ஆமா, {name} ரீசைக்கிள் ஆகும். {do}"},
    "recyclable_no": {"en": "Nope, {name} doesn't go with normal recycling. {do}", "hi": "नहीं, {name} नॉर्मल रीसायकलिंग में नहीं जाता। {do}",
                      "ta": "இல்ல, {name} நார்மல் ரீசைக்கிளிங்ல போகாது. {do}"},
    "hazardous_yes": {"en": "Yeah, {name} is hazardous, so be careful. {do}", "hi": "हाँ, {name} खतरनाक है, ध्यान रखना। {do}",
                      "ta": "ஆமா, {name} டேஞ்சரஸ் தான், கவனமா இருங்க. {do}"},
    "hazardous_no": {"en": "Nah, {name} isn't hazardous. {do}", "hi": "नहीं, {name} खतरनाक नहीं है। {do}",
                     "ta": "இல்ல, {name} டேஞ்சரஸ் இல்ல. {do}"},
    "where_to_give": {"en": "{do} Your local collection point or the scrap dealer will take it too.",
                      "hi": "{do} अपने एरिया के कलेक्शन पॉइंट या कबाड़ीवाले को भी दे सकते हो।",
                      "ta": "{do} உங்க ஏரியா கலெக்ஷன் பாயிண்ட் இல்ல பழைய பேப்பர் கடைக்காரரும் வாங்கிக்குவாங்க."},
    "no_item": {"en": "Show me the thing in front of the camera first, or just tell me what it is.",
                "hi": "पहले चीज़ कैमरे के सामने दिखाओ, या बस बता दो क्या है।",
                "ta": "முதல்ல அந்த பொருள கேமரா முன்னாடி காட்டுங்க, இல்ல அது என்னன்னு சொல்லுங்க."},
    "bins_tip": {"en": "Quick guide: food, peels and paper go in the green bin, dry stuff like plastic, glass and metal in the yellow bin, batteries and bulbs in the red bin, and old electronics to an e-waste centre. Show me the thing and I'll tell you exactly!",
                 "hi": "सिंपल है: खाना, छिलके और पेपर ग्रीन बिन में, प्लास्टिक, ग्लास, मेटल जैसा सूखा कचरा येलो बिन में, बैटरी और बल्ब रेड बिन में, और पुराने इलेक्ट्रॉनिक्स ई-वेस्ट सेंटर पे। चीज़ दिखाओ, मैं एकदम सही बता दूँगा!",
                 "ta": "சிம்பிள் தான்: சாப்பாடு, தோல், பேப்பர் எல்லாம் கிரீன் பின்ல, பிளாஸ்டிக், கிளாஸ், மெட்டல் மாதிரி காய்ஞ்ச குப்பை யெல்லோ பின்ல, பேட்டரி, பல்பு ரெட் பின்ல, பழைய எலக்ட்ரானிக்ஸ் ஈ-வேஸ்ட் சென்டர்ல. பொருள காட்டுங்க, கரெக்டா சொல்றேன்!"},
    "low_memory": {"en": "My memory's a bit full right now, but I can still tell you which bin anything goes in.",
                   "hi": "अभी मेरी मेमोरी थोड़ी फुल है, लेकिन कौन सी चीज़ किस बिन में जाएगी, ये अभी भी बता सकता हूँ।",
                   "ta": "இப்போ என் மெமரி கொஞ்சம் ஃபுல்லா இருக்கு. ஆனா எது எந்த பின்னுன்னு இப்பவும் சொல்லுவேன்."},
    "not_heard": {"en": "Sorry, didn't catch that. Press the button and try again?",
                  "hi": "सॉरी, ठीक से सुना नहीं। एक बार फिर बटन दबा के पूछो।",
                  "ta": "சாரி, சரியா கேக்கல. இன்னொரு தடவ பட்டன அழுத்தி கேளுங்க."},
}


def bin_for(stream, lang):
    return BINS[stream][lang]


def explain(cls, stream, lang):
    """Spoken explanation when an item is detected: (intro + fact, ask-for-questions)."""
    text = PHRASES["intro"][lang].format(name=NAMES[cls][lang], do=BINS[stream][lang])
    return text + " " + FACTS[cls][lang], PHRASES["ask"][lang]


def rule_answer(intent, cls, stream, lang):
    """Answer a matched intent from rules (never from Llama)."""
    name = NAMES[cls][lang]
    fmt = dict(name=name, Name=name[:1].upper() + name[1:], do=BINS[stream][lang])
    if intent == "which_bin":
        return PHRASES["which_bin"][lang].format(**fmt)
    if intent == "recyclable":
        return PHRASES["recyclable_yes" if stream == "recyclable" else "recyclable_no"][lang].format(**fmt)
    if intent == "hazardous":
        return PHRASES["hazardous_yes" if stream in ("hazardous", "e_waste") else "hazardous_no"][lang].format(**fmt)
    if intent == "where_to_give":
        return PHRASES["where_to_give"][lang].format(**fmt)
    return None


# ---------------------------------------------------------------- Llama answers: make the translation sound spoken
# NLLB translates into formal written Tamil/Hindi. These swaps turn the most common bookish words into
# what people actually say (longest match first). English loanwords stay in native script for the TTS.
CASUAL = {
    "ta": [
        ("மறுசுழற்சி செய்யப்படும்", "ரீசைக்கிள் ஆயிடும்"), ("மறுசுழற்சி செய்யலாம்", "ரீசைக்கிள் பண்ணலாம்"),
        ("மறுசுழற்சி செய்ய", "ரீசைக்கிள் பண்ண"), ("மறுசுழற்சி", "ரீசைக்கிள்"),
        ("மீண்டும் பயன்படுத்தக்கூடிய", "ரீயூஸ் பண்ணக்கூடிய"), ("ஒருமுறை பயன்படுத்தக்கூடிய", "சிங்கிள் யூஸ்"),
        ("பயன்படுத்துங்கள்", "யூஸ் பண்ணுங்க"), ("பயன்படுத்தவும்", "யூஸ் பண்ணுங்க"), ("பயன்படுத்தலாம்", "யூஸ் பண்ணலாம்"),
        ("பயன்படுத்தி", "யூஸ் பண்ணி"), ("பயன்படுத்த", "யூஸ் பண்ண"), ("பயன்பாட்டை", "யூஸ"), ("பயன்பாடு", "யூஸ்"),
        ("செய்யுங்கள்", "பண்ணுங்க"), ("செய்யலாம்", "பண்ணலாம்"), ("செய்யவும்", "பண்ணுங்க"), ("செய்ய வேண்டும்", "பண்ணணும்"),
        ("தொடங்குங்கள்", "ஸ்டார்ட் பண்ணுங்க"), ("தேர்ந்தெடுங்கள்", "செலக்ட் பண்ணுங்க"),
        ("வேண்டும்", "வேணும்"), ("இருக்கிறது", "இருக்கு"), ("உள்ளது", "இருக்கு"), ("உள்ளன", "இருக்கு"),
        ("இருக்கிறீர்கள்", "இருக்கீங்க"), ("முடியும்", "முடியும்"), ("நீங்கள்", "நீங்க"), ("உங்கள்", "உங்க"),
        ("மற்றும்", ","), ("அல்லது", "இல்லனா"), ("எஃகு", "ஸ்டீல்"), ("கொள்கலன்", "கண்டெய்னர்"),
        ("பேக்கேஜிங்", "பேக்கிங்"), ("தயாரிப்புகள்", "ப்ராடக்ட்ஸ்"), ("பொருட்களை", "பொருள"), ("இப்போது", "இப்போ"),
        ("எங்கே", "எங்க"), ("ஏனெனில்", "ஏன்னா"), ("ஆனால்", "ஆனா"), ("அதனால்", "அதனால"), ("மிகவும்", "ரொம்ப"),
        ("ங்கள்.", "ங்க."), ("ங்கள்,", "ங்க,"), ("ங்கள் ", "ங்க "),
    ],
    "hi": [
        ("एकल उपयोग वाले", "सिंगल-यूज़"), ("पुनः प्रयोज्य", "रीयूज़ेबल"), ("पुनर्चक्रण", "रीसायकलिंग"),
        ("पुनर्नवीनीकरण", "रीसायकल"), ("उपयोग", "यूज़"), ("प्रयोग", "यूज़"), ("विकल्पों", "ऑप्शन्स"), ("विकल्प", "ऑप्शन"),
        ("उत्पादों", "प्रोडक्ट्स"), ("उत्पाद", "प्रोडक्ट"), ("अपशिष्ट", "वेस्ट"), ("जैविक", "ऑर्गेनिक"),
        ("करें", "करो"), ("चुनें", "चुनो"), ("रखें", "रखो"), ("डालें", "डालो"), ("दें", "दो"), ("लें", "लो"),
        ("सकते हैं", "सकते हो"), ("अत्यधिक", "बहुत"), ("अधिक", "ज़्यादा"), ("किंतु", "लेकिन"), ("परंतु", "लेकिन"),
        ("तथा", "और"), ("एवं", "और"), ("यदि", "अगर"), ("कृपया", "प्लीज़"),
    ],
}


def casual(text, lang):
    for a, b in CASUAL.get(lang, []):
        text = text.replace(a, b)
    return re.sub(r"\s+,", ",", text).replace(",,", ",").strip()


# ---------------------------------------------------------------- welcome: ask the visitor's language by voice
WELCOME = {  # played one after another (en, hi, ta) from cached wavs, then the mic listens
    "en": "Hey there! Which language are you comfortable with? Just say something in your language.",
    "hi": "हाय! आपको कौन सी लैंग्वेज कम्फ़र्टेबल है? अपनी भाषा में कुछ भी बोलो।",
    "ta": "ஹாய்! உங்களுக்கு எந்த லாங்குவேஜ் கம்ஃபர்டபிள்? உங்க பாஷைல ஏதாவது பேசுங்க.",
}
READY = {  # spoken in the chosen language once it is detected
    "en": "Hi, I'm Sorti! Show your waste to the camera and I'll tell you which bin each item goes in.",
    "hi": "बढ़िया, हिंदी में बात करते हैं! अपना कचरा कैमरा को दिखाओ, मैं बता दूँगा हर चीज़ किस बिन में जाएगी।",
    "ta": "சூப்பர், தமிழ்லயே பேசலாம்! உங்க குப்பைய கேமராவுக்கு காட்டுங்க, ஒவ்வொண்ணும் எந்த பின்ல போடணும்னு சொல்றேன்.",
}
# language names as Whisper may write them (it sometimes writes Hindi in Urdu script)
LANG_WORDS = {
    "ta": ["tamil", "thamizh", "tamizh", "தமிழ்", "தமிழ", "தமி", "तमिल", "تامل", "تمل"],
    "hi": ["hindi", "हिन्दी", "हिंदी", "हिन्दि", "ہندی", "ہندي", "ہنڈی"],
    "en": ["english", "इंग्लिश", "अंग्रेज़ी", "अंग्रेजी", "ஆங்கில", "இங்கிலீஷ்", "انگلش", "انگریزی"],
}
# Whisper language codes -> our 3 (close neighbours map to the nearest supported one)
LID_MAP = {"en": "en", "hi": "hi", "ur": "hi", "mr": "hi", "ne": "hi", "pa": "hi",
           "ta": "ta", "ml": "ta", "te": "ta", "kn": "ta", "si": "ta"}


def pick_lang(text, whisper_lang):
    """Spoken reply -> 'en'|'hi'|'ta'|None. A language name said aloud wins over Whisper's guess."""
    t = (text or "").lower()
    hits = [(t.find(w), l) for l, ws in LANG_WORDS.items() for w in ws if w in t]
    if hits:
        return min(hits)[1]
    return LID_MAP.get((whisper_lang or "").strip("<|>"))

# The v2 classifier also recognises non-recyclable general waste. No new bin colour
# is assumed: the existing stream_map calls this "general".
NAMES['general_waste'] = {'en': 'general waste', 'hi': 'सामान्य कचरा', 'ta': 'சாதா குப்பை'}
TITLES['general_waste'] = {'en': 'General waste', 'hi': 'सामान्य कचरा', 'ta': 'சாதா குப்பை'}
BINS['general'] = {
    'color': 'grey',
    'en': 'Keep it separate and follow your local general-waste collection guidance.',
    'hi': 'इसे अलग रखो और अपने इलाके के सामान्य कचरा कलेक्शन के नियम मानो।',
    'ta': 'இத தனியா வைங்க, உங்க ஏரியா சாதா குப்பை கலெக்ஷன் விதிப்படி குடுங்க.'}
BIN_SHORT['general'] = {'en': 'Local general-waste collection', 'hi': 'स्थानीय सामान्य कचरा कलेक्शन',
                        'ta': 'உள்ளூர் சாதா குப்பை கலெக்ஷன்'}
FACTS['general_waste'] = {'en': 'Keep this out of clean recyclables.',
                          'hi': 'इसे साफ़ रीसायकल वाले कचरे से अलग रखो।',
                          'ta': 'இத சுத்தமான ரீசைக்கிள் குப்பையோட கலக்காதீங்க.'}


def grouped_reply(objects, stream_map, lang):
    """Group confirmed instances by Sorti class, without claiming COCO item identity."""
    from collections import Counter
    counts = Counter(o['cls'] for o in objects if o.get('confirmed') and o['cls'] != 'background')
    phrases = []
    for cls, count in sorted(counts.items()):
        stream = stream_map[cls]
        title = TITLES[cls][lang]
        dest = BIN_SHORT[stream][lang]
        if lang == 'en':
            noun = title.lower() + (' item' if count == 1 else ' items')
            phrases.append(f'{count} {noun}: {dest}.')
        elif lang == 'hi':
            phrases.append(f'{count} {title}: {dest} में दो।')
        else:
            phrases.append(f'{count} {title}: {dest}க்கு குடுங்க.')
    return ' '.join(phrases) if phrases else PHRASES['no_item'][lang]


# ---------------------------------------------------------------- fixed demo answers (owner's 3 questions)
# Each FAQ requires both a topic and the intent of the demo question.
FAQ = [
    {"id": "recycle_steps",
     "match": [["recycle", "recycling", "ரீசைக்கிள்", "ரிசைக்கிள்", "மறுசுழற்சி", "रीसायकल", "रिसायकल", "रीसाइकिल", "रिसाइकल", "पुनर्चक्रण"],
               ["step", "steps", "how to", "how do", "how does", "how can", "how should", "process", "procedure",
                "ஸ்டெப்", "படி", "எப்படி", "முறை", "स्टेप", "कैसे", "तरीका", "प्रक्रिया", "चरण"]],
     "en": "Here are the steps. One, keep dry waste separate from wet waste. Two, rinse bottles, cans and containers. "
           "Three, dry them and flatten them. Four, put plastic, glass and metal in the yellow bin. "
           "Five, give them to your waste collector or a recycling centre.",
     "hi": "स्टेप्स ये हैं। एक, सूखा कचरा गीले कचरे से अलग रखो। दो, बोतल, कैन और डिब्बे धो लो। तीन, उन्हें सुखाकर दबा दो। "
           "चार, प्लास्टिक, ग्लास और मेटल येलो बिन में डालो। पाँच, उन्हें कचरा वाले भैया या रीसायकल सेंटर को दे दो।",
     "ta": "ஸ்டெப்ஸ் இதோ. ஒண்ணு, காய்ஞ்ச குப்பைய ஈர குப்பையில இருந்து தனியா வைங்க. ரெண்டு, பாட்டில், கேன், டப்பா எல்லாம் கழுவுங்க. "
           "மூணு, காய வச்சு அமுக்கி வைங்க. நாலு, பிளாஸ்டிக், கிளாஸ், மெட்டல் எல்லாம் யெல்லோ பின்ல போடுங்க. "
           "அஞ்சு, குப்பை எடுக்கிறவங்க கிட்ட இல்ல ரீசைக்கிள் சென்டர்ல குடுங்க."},
    {"id": "ewaste_types",
     "match": [["e waste", "ewaste", "electronic waste", "ஈ-வேஸ்ட்", "ஈ வேஸ்ட்", "இ-வேஸ்ட்", "இ வேஸ்ட்", "ஈவேஸ்ட்", "இவேஸ்ட்",
                "எலக்ட்ரானிக்", "மின்னணு", "ई-वेस्ट", "ई वेस्ट", "ईवेस्ट", "इलेक्ट्रॉनिक", "इलेक्ट्रोनिक"],
               ["type", "types", "kind", "kinds", "example", "examples", "what else", "what other",
                "வகை", "டைப்", "வேற", "என்னென்ன", "என்ன என்ன", "மத்த", "மற்ற",
                "प्रकार", "टाइप", "किस्म", "और कौन", "कौन कौन", "कौन-कौन", "दूसरे"]],
     "en": "Besides phones, e-waste includes chargers and cables, earphones, laptops and keyboards, TVs and remotes, "
           "and old fans, mixers and irons. Give all of them to an e-waste collection centre.",
     "hi": "फ़ोन के अलावा चार्जर, केबल, ईयरफ़ोन, लैपटॉप, कीबोर्ड, टीवी, रिमोट, और पुराने पंखे, मिक्सी और प्रेस भी ई-वेस्ट हैं। "
           "ये सब ई-वेस्ट कलेक्शन सेंटर पे दे दो।",
     "ta": "மொபைல் தவிர சார்ஜர், கேபிள், இயர்ஃபோன், லேப்டாப், கீபோர்டு, டிவி, ரிமோட், அப்புறம் பழைய ஃபேன், மிக்ஸி, அயர்ன் பாக்ஸ் "
           "எல்லாமே ஈ-வேஸ்ட் தான். இதெல்லாம் ஈ-வேஸ்ட் கலெக்ஷன் சென்டர்ல குடுங்க."},
    {"id": "hazard_effect",
     "match": [["hazardous waste", "dangerous waste", "toxic waste", "ஹசார்டஸ்", "ஹசார்ட்", "ஆபத்தான", "ஆபத்து", "டேஞ்சரஸ்", "நச்சு",
                "खतरनाक", "ख़तरनाक", "हैज़र्डस", "हैजर्डस", "हज़ार्डस", "हानिकारक", "ज़हरीला", "जहरीला", "डेंजरस"],
               ["effect", "effects", "impact", "impacts", "harm", "damage", "problem", "problems",
                "விளைவு", "எஃபெக்ட்", "பாதிப்பு", "பிரச்சனை",
                "असर", "प्रभाव", "नुकसान", "इफेक्ट", "इफ़ेक्ट", "सबसे बड़ा", "परिणाम"]],
     "en": "The biggest effect is poison in our soil and water. Batteries, bulbs and chemicals leak lead, mercury and acid, "
           "which reach groundwater and our food, and can harm the brain, kidneys and children's growth. Always put them in the red bin.",
     "hi": "सबसे बड़ा असर है मिट्टी और पानी में ज़हर। बैटरी, बल्ब और केमिकल से लेड, मरकरी और एसिड निकलता है, जो ज़मीन के पानी और "
           "हमारे खाने तक पहुँचता है, और दिमाग, किडनी और बच्चों की ग्रोथ को नुकसान करता है। इन्हें हमेशा रेड बिन में डालो।",
     "ta": "பெரிய விளைவு மண்ணும் தண்ணியும் விஷமாகுறது தான். பேட்டரி, பல்பு, கெமிக்கல்ல இருந்து லெட், மெர்க்குரி, ஆசிட் கசியும். "
           "அது நிலத்தடி நீருக்கும் சாப்பாட்டுக்கும் போயி மூளை, கிட்னி, குழந்தைங்க வளர்ச்சிய பாதிக்கும். இத எப்பவும் ரெட் பின்ல போடுங்க."},
]


def faq_answer(text, lang):
    """Fixed answer for one of the owner's demo questions, in `lang`, or None."""
    t = (text or "").lower()
    english = " ".join(re.findall(r"[a-z0-9]+", t))
    for f in FAQ:
        if all(any((f" {w} " in f" {english} ") if w.isascii() else (w in t)
                   for w in group) for group in f["match"]):
            return f["id"], f[lang]
    return None
