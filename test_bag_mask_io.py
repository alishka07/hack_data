import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from bag_mask_io import BagMaskStore


class MaskProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        Image.new('RGB', (32, 40), 'green').save(self.root / 'source.jpg')
        mask = Image.new('L', (32, 40)); mask.paste(255, (10, 10, 20, 30))
        mask.save(self.root / 'mask.png')
        self.image = {'file_name': 'source.jpg', 'width': 32, 'height': 40}
        self.entry = {'file_name': 'source.jpg', 'review_status': 'accepted', 'mask_file': 'mask.png',
                      'source_sha256': hashlib.sha256((self.root/'source.jpg').read_bytes()).hexdigest(),
                      'mask_sha256': hashlib.sha256((self.root/'mask.png').read_bytes()).hexdigest()}

    def store(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps({'images': [self.entry]}), encoding='utf-8')
        return BagMaskStore(path, self.root)

    def test_only_reviewed_source_matched_masks_are_loaded(self):
        mask = self.store().get(self.image)
        self.assertEqual(mask.size, (32, 40))
        self.entry['review_status'] = 'excluded'
        self.assertIsNone(self.store().get(self.image))

    def test_source_change_cannot_reuse_old_mask(self):
        Image.new('RGB', (32, 40), 'red').save(self.root / 'source.jpg')
        with self.assertRaisesRegex(ValueError, 'different source'):
            self.store().get(self.image)

    def test_mask_change_after_visual_review_is_rejected(self):
        Image.new('L', (32, 40), 255).save(self.root / 'mask.png')
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.store().get(self.image)


if __name__ == '__main__':
    unittest.main()
