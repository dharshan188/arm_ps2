"""Hand-authored parallel kiosk questions and semantic gold (not snap-generated)."""
EN = '''Where do I put this waste?
Which bin for a battery?
Is plastic recyclable?
What is e-waste?
How to reduce kitchen waste?
Can I put a pizza box in the blue bin?
Where should I put wet waste?
Which bin does dry waste go in?
Can I recycle this glass jar?
Where do I throw this metal can?
Which bin does this paper go in?
Is this bulb dangerous?
Can I compost vegetable peels?
Where should I give old batteries?
Can I recycle this plastic bottle?
Is broken glass safe to touch?
Where do I throw leftover food?
Which bin for a broken phone?
Can I recycle a newspaper?
Where can I give old clothes?
Can I put a bulb in the dry bin?
How do I make compost at home?
What is the difference between wet and dry waste?
Can I recycle a battery?
Which bin does a plastic bag go in?
Is e-waste harmful?
Where should I give this charger?
Can I recycle metal foil?
Which bin does a cardboard box go in?
Where should I put fruit peels?
Can I recycle a glass bottle?
Is a leaking battery toxic?
How do I clean a plastic bottle before recycling?
Where is the paper collection centre?
Can I put food in the blue bin?
Why should we separate waste?
Please repeat that once more.
Can I reuse this paper bag?
Where do I throw a broken tube light?
Can I compost tea leaves?'''.splitlines()
HI = '''ये वेस्ट कहाँ डालूँ?
बैटरी किस बिन में डालूँ?
क्या प्लास्टिक रीसायकल हो सकता है?
ई-वेस्ट क्या होता है?
किचन का वेस्ट कैसे कम करें?
पिज़्ज़ा बॉक्स ब्लू बिन में डाल सकते हैं क्या?
गीला वेस्ट कहाँ डालना है?
ड्राई वेस्ट किस बिन में डालूँ?
क्या कांच का जार रीसायकल कर सकते हैं?
मेटल का कैन कहाँ फेंकूँ?
ये पेपर किस बिन में डालूँ?
क्या ये बल्ब खतरनाक है?
सब्जी के छिलकों से कम्पोस्ट बना सकते हैं?
पुरानी बैटरियाँ कहाँ दूँ?
प्लास्टिक की बोतल रीसायकल कर सकते हैं क्या?
टूटा कांच छूना सेफ है क्या?
बचा हुआ खाना कहाँ डालूँ?
टूटा फोन किस बिन में डालूँ?
अखबार रीसायकल होता है क्या?
पुराने कपड़े कहाँ दूँ?
बल्ब ड्राई बिन में डाल सकते हैं क्या?
घर पे कम्पोस्ट कैसे बनाऊँ?
वेट और ड्राई वेस्ट में क्या फर्क है?
बैटरी रीसायकल हो सकती है क्या?
प्लास्टिक की थैली किस बिन में डालूँ?
ई-वेस्ट हानिकारक है क्या?
ये चार्जर कहाँ दूँ?
मेटल की फॉइल रीसायकल कर सकते हैं?
कार्डबोर्ड बॉक्स किस बिन में डालूँ?
फल के छिलके कहाँ डालना है?
कांच की बोतल रीसायकल होती है क्या?
लीक होती बैटरी खतरनाक है क्या?
रीसायकल करने से पहले प्लास्टिक बोतल कैसे साफ करें?
पेपर का कलेक्शन सेंटर कहाँ है?
खाना ब्लू बिन में डाल सकते हैं क्या?
वेस्ट अलग अलग क्यों रखना चाहिए?
प्लीज़ एक बार और बताओ।
क्या ये पेपर बैग रीसायकल कर सकते हैं?
टूटी ट्यूब लाइट कहाँ फेंकूँ?
चाय की पत्तियों से कम्पोस्ट बना सकते हैं?'''.splitlines()
TA = '''இந்த வேஸ்ட் எங்க போடணும்?
பேட்டரியை எந்த பின்ல போடணும்?
பிளாஸ்டிக் ரீசைக்கிள் பண்ணலாமா?
ஈ வேஸ்ட்னா என்ன?
கிச்சன் வேஸ்ட் எப்படி குறைக்கிறது?
பீட்சா பாக்ஸை ப்ளூ பின்ல போடலாமா?
ஈரக் குப்பை எங்க போடணும்?
ட்ரை வேஸ்ட் எந்த பின்ல போடணும்?
கண்ணாடி ஜாடியை ரீசைக்கிள் பண்ணலாமா?
மெட்டல் கேனை எங்க போடணும்?
இந்த பேப்பர் எந்த பின்ல போடணும்?
இந்த பல்பு ஆபத்தானதா?
காய்கறி தோலை கம்போஸ்ட் பண்ணலாமா?
பழைய பேட்டரிகளை எங்க குடுக்கணும்?
பிளாஸ்டிக் பாட்டிலை ரீசைக்கிள் பண்ணலாமா?
உடைஞ்ச கண்ணாடியை தொடுறது சேஃப்பா?
மிச்ச சாப்பாடு எங்க போடணும்?
உடைஞ்ச போனை எந்த பின்ல போடணும்?
நியூஸ் பேப்பரை ரீசைக்கிள் பண்ணலாமா?
பழைய துணிகளை எங்க குடுக்கலாம்?
பல்பை ட்ரை பின்ல போடலாமா?
வீட்டுல கம்போஸ்ட் எப்படி பண்றது?
வெட் வேஸ்ட் ட்ரை வேஸ்ட் என்ன வித்தியாசம்?
பேட்டரியை ரீசைக்கிள் பண்ணலாமா?
பிளாஸ்டிக் பையை எந்த பின்ல போடணும்?
ஈ வேஸ்ட் ஆபத்தானதா?
இந்த சார்ஜரை எங்க குடுக்கணும்?
மெட்டல் ஃபாயிலை ரீசைக்கிள் பண்ணலாமா?
கார்ட்போர்டு பாக்ஸை எந்த பின்ல போடணும்?
பழத்தோலை எங்க போடணும்?
கண்ணாடி பாட்டிலை ரீசைக்கிள் பண்ணலாமா?
லீக் ஆகுற பேட்டரி ஆபத்தானதா?
ரீசைக்கிள் பண்றதுக்கு முன்னாடி பிளாஸ்டிக் பாட்டிலை எப்படி கழுவணும்?
பேப்பர் கலெக்ஷன் சென்டர் எங்க இருக்கு?
சாப்பாட்டை ப்ளூ பின்ல போடலாமா?
வேஸ்ட்டை ஏன் தனித்தனியா பிரிக்கணும்?
இன்னொரு தடவ சொல்லுங்க.
இந்த பேப்பர் பையை ரீசைக்கிள் பண்ணலாமா?
உடைஞ்ச டியூப் லைட்டை எங்க போடணும்?
டீ இலைகளை கம்போஸ்ட் பண்ணலாமா?'''.splitlines()
ITEMS = [None,'battery_waste','plastic_waste','e_waste','organic_waste','paper_waste',None,None,'glass_waste','metal_waste','paper_waste','light_bulbs','organic_waste','battery_waste','plastic_waste','glass_waste','organic_waste','e_waste','paper_waste','clothes','light_bulbs',None,None,'battery_waste','plastic_waste','e_waste','e_waste','metal_waste','paper_waste','organic_waste','glass_waste','battery_waste','plastic_waste','paper_waste','organic_waste',None,None,'paper_waste','light_bulbs','organic_waste']
INTENTS = ['which_bin','which_bin','recyclable','other','other','which_bin','which_bin','which_bin','recyclable','which_bin','which_bin','hazardous','other','where_to_give','recyclable','hazardous','which_bin','which_bin','recyclable','where_to_give','which_bin','other','other','recyclable','which_bin','hazardous','where_to_give','recyclable','which_bin','which_bin','recyclable','hazardous','other','where_to_give','which_bin','other','repeat','recyclable','which_bin','other']
QUESTIONS = {l:[dict(text=t,item=ITEMS[i],intent=INTENTS[i],question_id=f'{l}_{i:02}') for i,t in enumerate(ts)] for l,ts in [('en',EN),('hi',HI),('ta',TA)]}
assert all(len(q)==40 for q in QUESTIONS.values())
