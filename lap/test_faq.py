"""Answer bank, off-topic guard and answer guard for free-form questions (no models loaded)."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import faq
import server as s


class BankTests(unittest.TestCase):
    def test_every_entry_has_all_languages_and_no_bin_colour(self):
        for e in faq.BANK:
            for lang in ("en", "hi", "ta"):
                self.assertTrue(e[lang].strip(), (e["id"], lang))
            self.assertIsNone(s.BIN_COLOUR.search(e["en"]), e["id"])
            self.assertLessEqual(len(e["en"].split()), 36, e["id"])
        self.assertEqual(len({e["id"] for e in faq.BANK}), len(faq.BANK))

    def test_direct_match_respects_the_item(self):
        e, _, direct = faq.match("What can I make from this?", "plastic_waste", ["plastic_waste"])
        self.assertTrue(direct); self.assertEqual(e["id"], "plastic_bottle_diy")
        e, _, direct = faq.match("What can I make from this?", "paper_waste", ["paper_waste"])
        self.assertEqual(e["id"], "newspaper_diy")

    def test_follow_up_uses_previous_exchange(self):
        e, _, direct = faq.match("How long does that take?", "organic_waste", ["organic_waste"],
                                 "How do I compost peels at home?", "Mix the peels with dry leaves.")
        self.assertTrue(direct); self.assertEqual(e["id"], "compost_time")

    def test_off_topic(self):
        self.assertTrue(faq.off_topic("Who won the cricket match yesterday?"))
        self.assertFalse(faq.off_topic("How do landfills affect people nearby?"))
        self.assertFalse(faq.off_topic("Should I wash this?", has_scene=True))

    def test_guard(self):
        self.assertIsNotNone(faq.guard("Tear it up and use it as a natural dye for fabric.")[1])
        self.assertIsNotNone(faq.guard("Pour it down the toilet, as it's non-toxic.")[1])
        self.assertIsNone(faq.guard("Never burn plastic, the smoke is toxic.")[1])
        self.assertEqual(faq.guard("Rinse it first. Then dry it. Then flatten it")[0], "Rinse it first. Then dry it.")
        self.assertEqual(faq.guard("Carry a cloth bag.\nVisitor: and?")[0], "Carry a cloth bag.")


class FreeFormTests(unittest.TestCase):
    def setUp(self):
        self.old = s.S, s.VIS
        s.S = s.State(); s.S.chosen = True
        s.VIS = SimpleNamespace(meta={'stream_map': {'battery_waste': 'hazardous', 'paper_waste': 'biodegradable'}})
        s.S.objects = [dict(cls='battery_waste', stream='hazardous', confirmed=True)]
        self.say = patch.object(s.SP, 'say').start()
        patch.object(s, 'maybe_video', return_value=False).start()

    def tearDown(self):
        patch.stopall(); s.S, s.VIS = self.old

    def test_bank_answer_skips_llama_and_translation_in_tamil(self):
        s.S.lang = 'ta'
        with patch.object(s.LL, 'ask') as ask, patch.object(s, 'TR', return_value='My battery is swollen, what should I do?') as tr:
            s.answer_question(None, typed='என் பேட்டரி உப்பிடுச்சு, என்ன பண்றது?')
        ask.assert_not_called()
        self.assertEqual(tr.call_count, 1)                         # only the question is translated
        entry = next(e for e in faq.BANK if e['id'] == 'battery_swollen')
        self.assertEqual(s.S.messages[-1]['text'], entry['ta'])

    def test_unsafe_llama_answer_falls_back_to_verified_text(self):
        s.S.objects = [dict(cls='paper_waste', stream='biodegradable', confirmed=True)]
        with patch.object(s, 'snap', return_value={'intent': 'other', 'item': None}), \
                patch.object(s.LL, 'ask', return_value='Use it as a natural dye for fabric.'):
            s.answer_question(None, typed='Is this paper good for anything at all in the garden shed?')
        self.assertNotIn('dye', s.S.messages[-1]['text'])
        self.assertEqual(s.S.messages[-1]['source'], 'rules')

    def test_off_topic_gets_redirect_without_llama(self):
        s.S.objects = []
        with patch.object(s.LL, 'ask') as ask:
            s.answer_question(None, typed='Who won the cricket match yesterday?')
        ask.assert_not_called()
        self.assertEqual(s.S.messages[-1]['text'], faq.OFF_TOPIC_REPLY['en'])


if __name__ == '__main__':
    unittest.main()
