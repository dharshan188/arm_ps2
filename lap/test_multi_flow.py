"""Rules and announcement integration without loading speech models or using audio devices."""
import sys
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as s


class MultiFlowTests(unittest.TestCase):
    def setUp(self):
        self.old_state, self.old_vis = s.S, s.VIS
        s.S = s.State(); s.S.chosen = True; s.S.status = 'watching'
        s.VIS = SimpleNamespace(meta={'stream_map': {'plastic_waste': 'recyclable', 'battery_waste': 'hazardous'}})
        s.S.objects = [dict(cls=c, stream=st, confirmed=True, color=s.C.BINS[st]['color'])
                       for c, st in [('plastic_waste', 'recyclable'), ('plastic_waste', 'recyclable'),
                                     ('battery_waste', 'hazardous')]]
        self.say = patch.object(s.SP, 'say').start()
        self.video = patch.object(s, 'maybe_video', return_value=False).start()
        # These tests pin down the Llama prompt path; the answer bank (lap/faq.py) would answer some of the
        # questions verbatim before Llama, so it is switched off here and tested in test_faq.py.
        patch.object(s, 'FAQ_DIRECT', False).start()

    def tearDown(self):
        patch.stopall(); s.S, s.VIS = self.old_state, self.old_vis

    def test_itemless_questions_cover_all_current_objects(self):
        for lang, text in [('en', 'which bin?'), ('hi', 'किस बिन में डालूँ?'), ('ta', 'எந்த பின்ல போடணும்?')]:
            s.S.lang = lang
            s.answer_question(None, typed=text)
            message = s.S.messages[-1]
            self.assertEqual(message['source'], 'rules')
            self.assertEqual(len(message['objects']), 3)
            self.assertIn('2', message['text'])
            self.assertIn(s.C.BIN_SHORT['hazardous'][lang], message['text'])

    def test_video_follows_the_reply(self):
        s.answer_question(None, typed='which bin?')
        self.video.assert_called_once_with(['plastic_waste', 'plastic_waste', 'battery_waste'], 'en')
        self.video.reset_mock()
        s.answer_question(None, typed='which bin for a battery?')
        self.video.assert_called_once_with(['battery_waste'], 'en')

    def test_video_counts_as_shown_only_when_watched_to_the_end(self):
        patch.stopall()
        clip = {'id': 'x', 'video': 'x.mp4', 'poster': 'x.jpg', 'credit': '', 'vo': {'en': 'v'}, 'vo_files': {'en': 'x.wav'}}
        with patch.object(s.SP, 'no_speaker', True), patch.object(s, 'pick_video', return_value=clip):
            s.VIDEO_SHOWN.clear()
            threading.Timer(.1, s.video_skip).start()          # ask button: interrupted
            self.assertTrue(s.maybe_video(['battery_waste'], 'en'))
            self.assertNotIn('battery_waste', s.VIDEO_SHOWN)
            threading.Timer(.1, s.video_done).start()          # browser: played to the end
            self.assertTrue(s.maybe_video(['battery_waste'], 'en'))
            self.assertIn('battery_waste', s.VIDEO_SHOWN)
            self.assertFalse(s.maybe_video(['battery_waste'], 'en'))   # not again within 5 minutes

    def test_explicit_item_keeps_existing_rule_flow(self):
        s.answer_question(None, typed='which bin for a battery?')
        self.assertEqual(s.S.messages[-1]['item'], 'battery_waste')
        self.assertNotIn('objects', s.S.messages[-1])

    def test_multi_announce_and_single_dispatch(self):
        s.announce_objects(s.S.objects)
        self.assertEqual(s.S.announced_signature, ('battery_waste', 'plastic_waste', 'plastic_waste'))
        self.assertEqual(len(s.S.item['objects']), 3)
        with patch.object(s, 'announce') as announce:
            s.announce_objects(s.S.objects[:1])
            announce.assert_called_once_with('plastic_waste', 'recyclable')
        self.assertEqual(len(s.predictions()), 3)

    def test_no_current_objects_does_not_use_previous_item(self):
        s.S.objects = []; s.S.item = {'cls': 'battery_waste'}
        s.answer_question(None, typed='which bin?')
        self.assertEqual(s.S.messages[-1]['text'], s.C.PHRASES['no_item']['en'])

    def test_free_form_uses_all_confirmed_categories_and_counts(self):
        s.S.item = {'cls': 'clothes', 'stream': 'recyclable'}
        s.S.objects.append(dict(cls='clothes', confirmed=False))
        question = 'How can I reduce waste at home?'
        with patch.object(s, 'snap', return_value={'intent': 'other', 'item': None}):
            with patch.object(s.LL, 'ask', return_value='Reuse containers.') as ask:
                s.answer_question(None, typed=question)
        prompt = ask.call_args.args[0]
        self.assertIn('2 x Plastic', prompt)          # every confirmed category, its count and fixed bin
        self.assertIn('Battery (Red bin)', prompt)
        self.assertIn(s.C.FACTS['battery_waste']['en'], prompt)  # grounded with the item's real fact
        self.assertIn(question, prompt)
        self.assertNotIn('clothes', prompt.lower())
        self.assertEqual(s.S.messages[-1]['source'], 'llama')

    def test_running_recording_is_never_hidden_by_a_finishing_video(self):
        with patch.object(s.REC, 'proc', object()):
            s.S.set(status='listening')
            s.S.set(status='watching')        # what a finishing video/speech used to do -> mic looked off, got stuck
            self.assertEqual(s.S.status, 'listening')
            self.assertEqual(s.ask_start(), {'ok': True, 'recording': True})
        s.S.set(status='watching')
        self.assertEqual(s.S.status, 'watching')

    def test_greeting_is_answered_instantly_without_llama(self):
        with patch.object(s.LL, 'ask') as ask:
            s.answer_question(None, typed='Hello one two three')
        ask.assert_not_called()
        self.assertEqual(s.S.messages[-1]['text'], s.GREET_REPLY['en'])

    def test_follow_up_gets_the_previous_exchange(self):
        with patch.object(s, 'snap', return_value={'intent': 'other', 'item': None}):
            with patch.object(s.LL, 'ask', return_value='Use a cloth bag.'):
                s.answer_question(None, typed='How can I cut plastic use?')
            with patch.object(s.LL, 'ask', return_value='Yes.') as ask:
                s.answer_question(None, typed='And at school?')
        prompt = ask.call_args.args[0]
        self.assertIn('Previous question: How can I cut plastic use?', prompt)
        self.assertIn('Your previous answer: Use a cloth bag.', prompt)

    def test_missing_llama_falls_back_to_rules_for_current_scene(self):
        with patch.object(s, 'snap', return_value={'intent': 'other', 'item': None}):
            with patch.object(s.LL, 'ask', side_effect=FileNotFoundError):
                s.answer_question(None, typed='How can I reduce waste at home?')
        self.assertEqual(s.S.messages[-1]['source'], 'rules')
        self.assertEqual(s.S.messages[-1]['text'], s.C.grouped_reply(s.S.objects, s.VIS.meta['stream_map'], 'en'))

    def test_scene_diy_uses_rules_for_non_recyclable_items(self):
        with patch.object(s.LL, 'ask', return_value='Try making a phone stand.') as ask:
            with patch.object(s, 'snap', return_value={'intent': 'other', 'item': None}):
                s.answer_question(None, typed='What can I make from each of these items at home?')
        prompt = ask.call_args.args[0]
        self.assertIn('2 x Plastic', prompt)
        self.assertNotIn('battery', prompt.lower())
        self.assertIn(s.C.BIN_SHORT['hazardous']['en'], s.S.messages[-1]['text'])


if __name__ == '__main__':
    unittest.main()
