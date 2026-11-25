"""Metric correctness: symmetric spelling rules must preserve semantic distinctions."""
import unittest
import eval as harness


class IndicMetricTests(unittest.TestCase):
    def test_unicode_marks_and_punctuation(self):
        self.assertEqual(harness.norm('  குப்பை!  कहाँ? '), 'குப்பை कहाँ')
        self.assertNotEqual(harness.norm('குப்பை'), harness.norm('குப்ப'))

    def test_symmetric_spelling_normalization(self):
        for a,b,lang in [('काँच रीसाइकिल','कांच रीसायकल','hi'),
                         ('எங்கே போடவேண்டும்','எங்க போடணும்','ta'),
                         ('கொடுக்கலாம் சொல்லுங்கள்','குடுக்கலாம் சொல்லுங்க','ta')]:
            self.assertEqual(harness.normalized_text(a,lang),harness.normalized_text(b,lang))

    def test_no_domain_or_negation_collapsing(self):
        for a,b,lang in [('गीला','ड्राई','hi'),('பேட்டரி','பிளாஸ்டிக்','ta'),
                         ('பண்ணலாமா','பண்ணக்கூடாது','ta'),('है','नहीं है','hi')]:
            self.assertNotEqual(harness.normalized_text(a,lang),harness.normalized_text(b,lang))

    def test_nukta(self):
        self.assertEqual(harness.normalized_text('पिज़्ज़ा प्लीज़','hi'),
                         harness.normalized_text('पिज्जा प्लीज','hi'))

    def test_faster_whisper_generator_is_consumed(self):
        from types import SimpleNamespace
        class FakeWhisper:
            def transcribe(self,audio,**kwargs):
                assert kwargs['language']=='ta'
                assert kwargs['beam_size']==1
                assert kwargs['temperature']==0
                assert not kwargs['condition_on_previous_text']
                return (SimpleNamespace(text=s) for s in [' வணக்கம்',' உலகம்']), None
        self.assertEqual(harness.transcribe(FakeWhisper(),None,'ta','faster-whisper'),'வணக்கம் உலகம்')

    def test_strict_and_normalized_scores(self):
        # Distance is aggregated over tokens; empty hypotheses retain deletions.
        self.assertEqual(harness.distance('a b'.split(), []), 2)
        self.assertEqual(harness.distance(['a'], ['a','b']), 1)
        self.assertEqual(harness.distance(harness.normalized_text('काँच','hi').split(),
                                          harness.normalized_text('कांच','hi').split()), 0)

if __name__=='__main__': unittest.main()
