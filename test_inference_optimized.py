import unittest
from inference_optimized import cached_backbone, PromptDetector


class Backbone:
    def __init__(self):
        self.calls = 0
    def forward(self, value):
        self.calls += 1
        return [(value, 'mask')], ['position']


class CacheTests(unittest.TestCase):
    def test_extra_levels_do_not_accumulate_and_frame_cache_expires(self):
        b = Backbone()
        original = b.forward
        with cached_backbone(b):
            features, positions = b.forward('frame1')
            features.append(('extra', 'mask'))
            positions.append('extra')
            self.assertEqual(b.forward('frame1'), ([('frame1', 'mask')], ['position']))
            self.assertEqual(b.calls, 1)
        self.assertEqual(b.forward, original)
        self.assertNotIn('forward', b.__dict__)
        with cached_backbone(b):
            self.assertEqual(b.forward('frame2')[0][0][0], 'frame2')
        self.assertEqual(b.calls, 2)

    def test_exception_restores_model(self):
        b = Backbone()
        original = b.forward
        with self.assertRaises(RuntimeError):
            with cached_backbone(b):
                b.forward('frame')
                raise RuntimeError('inference failed')
        self.assertEqual(b.forward, original)

    def test_existing_instance_override_is_preserved(self):
        b = Backbone()
        override = lambda value: ([(value, 'mask')], ['position'])
        b.forward = override
        with cached_backbone(b):
            b.forward('frame')
        self.assertIs(b.forward, override)

    def test_unknown_model_uses_original_detector(self):
        called = []
        def fallback(image, loaded, prompt, threshold):
            called.append(prompt)
            return [prompt]
        detector = PromptDetector((None, object(), 'cpu'), fallback)
        self.assertEqual(detector.detect_groups(None, {'a': 'one', 'b': 'two'}, .28),
                         {'a': ['one'], 'b': ['two']})
        self.assertEqual(called, ['one', 'two'])


if __name__ == '__main__':
    unittest.main()
