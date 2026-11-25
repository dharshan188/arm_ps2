import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stt_fix import correct, preprocess

class FixTest(unittest.TestCase):
    def test_repairs(self):
        for a,b in [('where do I put this west','where do I put this waste'),('which bean for a battery','which bin for a battery'),('put it in the been','put it in the bin'),('wait waste goes where','wet waste goes where'),('try waste bin','dry waste bin'),('is e waste recyclable','is e-waste recyclable'),('re cycle plastic','recycle plastic'),('which bin for a batery','which bin for a battery'),('can I compostt kitchen waste','can I compost kitchen waste'),('plastics in which bin','plastic in which bin')]:
            self.assertEqual(correct(a,'en'),b)
    def test_preserve(self):
        for a in ['west Bengal','I want to wait','try again','I have been putting plastic away','I eat a green bean','go west','my waist hurts','the west coast has waste bins','I want to wait before recycling','put the coffee bean away','my battery is dry','the bean is food']:
            self.assertEqual(correct(a,'en'),a)
    def test_idempotent(self):
        for lang,text in [('en','Which bin for a battery?'),('hi','बैटरी किस बिन में डालूँ?'),('ta','பேட்டரியை எந்த பின்ல போடணும்?')]:
            self.assertEqual(correct(correct(text,lang),lang),correct(text,lang))
    def test_context_safeguards(self):
        for text in ['my waist hurts while sorting waste', 'go west to the recycling centre', 'put pancake batter in the bin', 'I recycled the plastic', 'the west side has a bin', 'compose music about waste', 'the recycler takes plastic', 'recycle plastic in west London', 'the sun sets in the west near a waste bin']:
            self.assertEqual(correct(text,'en'),text)
        for text,expected in [('which pin for a battery','which bin for a battery'),('what is he waste','what is e-waste')]:
            self.assertEqual(correct(text,'en'),expected)
    def test_audio(self):
        import numpy as np
        for x in [np.array([],dtype=np.float32),np.zeros(16000),np.ones(16000),np.array([np.nan,np.inf])]:
            y=preprocess(x); self.assertEqual(y.dtype,np.float32); self.assertTrue(np.isfinite(y).all()); self.assertTrue(np.max(np.abs(y))<=.90001)
            self.assertEqual(len(y),8000)
        signal=.2*np.sin(np.arange(8000)*.17)
        x=np.concatenate([np.zeros(16000),signal,np.zeros(16000)])+.05
        y=preprocess(x)
        self.assertLess(len(y),len(x)); self.assertAlmostEqual(float(np.max(np.abs(y))),.9,places=5)
        self.assertTrue(np.all(y[:4000]==0)); self.assertTrue(np.all(y[-4000:]==0))
    def test_hindi(self):
        for a,b in [('प्रास्टिक रिसाइकल हो सकता है','प्लास्टिक रीसायकल हो सकता है'),('पुरानी बेट्रियाँ कहाँ दूँ','पुरानी बैटरियाँ कहाँ दूँ'),('इवेस्ट क्या है','ई-वेस्ट क्या है'),('किस दिन में डालें','किस बिन में डालें'),('dry west किस बिन में डालूँ','ड्राई वेस्ट किस बिन में डालूँ'),('छिलकों से कम पोस्ट बना सकते हैं','छिलकों से कम्पोस्ट बना सकते हैं')]:
            self.assertEqual(correct(a,'hi'),b)
        for text in ['किस दिन बैटरी बदलें', 'मैं भी साइकिल चलाता हूँ', 'सोशल मीडिया पर कम पोस्ट करो', 'west Bengal में रहता हूँ', 'बैठ कर बात करो']:
            self.assertEqual(correct(text,'hi'),text)
    def test_tamil(self):
        for a,b in [('plastic ரீசிக்கில் பண்ணலாமா','plastic ரீசைக்கிள் பண்ணலாமா'),('பழைய பேட்ரிகளை எங்க குடுக்கணும்','பழைய பேட்டரிகளை எங்க குடுக்கணும்'),('battery எந்த பின்னில் போடணும்','battery எந்த பின்ல போடணும்'),('இவைஸ்ட்னா என்ன','ஈ வேஸ்ட்னா என்ன'),('காய்கறி தோலை compose பண்ணலாமா','காய்கறி தோலை compost பண்ணலாமா'),('blastic bottle recycle பண்ணலாமா','பிளாஸ்டிக் bottle recycle பண்ணலாமா')]:
            self.assertEqual(correct(a,'ta'),b)
        for text in ['வீட்டுக்கு பின்னில் இருக்கு', 'என் வயசு என்ன', 'எனக்கு மெடல் கிடைச்சது', 'சைக்கிள் ஓட்டலாமா', 'வீட்டுல compose எப்படி பண்றது']:
            self.assertEqual(correct(text,'ta'),text)

if __name__=='__main__': unittest.main()
