"""Deterministic checks for decoding, tracking, fallback and multilingual grouping.
Run: python -m unittest vision.test_multi
"""
import unittest
from unittest.mock import patch
import numpy as np
from vision.detector import decode
from vision.multi import MultiVision, square_crop, signature, C


class FakeDetector:
    boxes = [(10, 20, 50, 60, .9), (100, 20, 150, 60, .8)]
    def detect(self, rgb):
        return self.boxes


class FakeClassifier:
    classes = ['plastic_waste', 'metal_waste', 'background']
    meta = dict(classes=classes, stream_map=dict(plastic_waste='recyclable', metal_waste='recyclable', background='none'))
    p = [.9, .05, .05]
    def probs(self, rgb):
        return np.array(self.p)


class MultiTests(unittest.TestCase):
    def setUp(self):
        self.d = FakeDetector(); self.c = FakeClassifier()
        self.m = MultiVision(detector=self.d, classifier=self.c, confirm_frames=8, confirm_conf=.70, centre_fallback=True, hold_frames=0, smooth=0)
        self.rgb = np.zeros((100, 200, 3), np.uint8)

    def test_confirmation_counts_and_reset(self):
        for _ in range(7):
            self.assertEqual(signature(self.m.process(self.rgb)), ())
        a = self.m.process(self.rgb)
        self.assertEqual(signature(a), ('plastic_waste', 'plastic_waste'))
        for lang in ('en', 'hi', 'ta'):
            reply = C.grouped_reply(a, self.c.meta['stream_map'], lang)
            self.assertIn('2', reply)
            self.assertEqual(reply.count(C.BIN_SHORT['recyclable'][lang]), 1)
        self.c.p = [.45, .3, .25]   # below the steady-rule floor (confirm_conf - 0.2) -> reset
        self.assertEqual(signature(self.m.process(self.rgb)), ())
        self.c.p = [.9, .05, .05]
        self.assertFalse(any(o['confirmed'] for o in self.m.process(self.rgb)))
        self.c.p = [.05, .9, .05]
        self.assertEqual(self.m.process(self.rgb)[0]['streak'], 1)
        self.c.p = [.01, .01, .98]
        self.assertEqual(self.m.process(self.rgb), [])

    def test_miss_resets_and_fallback_is_separate(self):
        a = self.m.process(self.rgb); ids = {o['id'] for o in a}
        self.d.boxes = []
        fallback = self.m.process(self.rgb)
        self.assertTrue(fallback[0]['fallback'])
        self.assertEqual(fallback[0]['box'], [70., 20., 130., 80.])
        self.d.boxes = FakeDetector.boxes
        a = self.m.process(self.rgb)
        self.assertEqual({o['id'] for o in a}, ids)
        self.assertTrue(all(o['streak'] == 1 for o in a))

    def test_background_boxes_do_not_block_fallback(self):
        # Two room detections classify as background, then the centre contains waste.
        with patch.object(self.c, 'probs', side_effect=[np.array(p) for p in
                          ([[.01, .01, .98], [.01, .01, .98], [.9, .05, .05]] * 8)]):
            for _ in range(8):
                objects = self.m.process(self.rgb)
        self.assertEqual(len(objects), 1)
        self.assertTrue(objects[0]['fallback'])
        self.assertTrue(objects[0]['confirmed'])

    def test_square_crop_at_edge(self):
        crop = square_crop(self.rgb, [0, 0, 20, 60])
        self.assertEqual(crop.shape, (72, 72, 3))

    def test_decode_exclusions_nms_and_letterbox(self):
        # RGB 200x100 -> 320x160, letterbox top 80.
        rows = np.array([[80, 160, 80, 80, .01, .9], [82, 160, 80, 80, .01, .8],
                         [240, 160, 60, 60, .99, .01]], np.float32)
        raw = rows.T[None]
        out = decode(raw, (100, 200, 3), (320, 160), (0, 80))
        self.assertEqual(len(out), 1)
        np.testing.assert_allclose(out[0][:4], [25, 25, 75, 75])
        # Waste detector class 0 is allowed when explicitly configured.
        self.assertEqual(len(decode(raw, (100, 200, 3), (320, 160), (0, 80), exclude=[])), 2)


if __name__ == '__main__':
    unittest.main()
