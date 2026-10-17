"""Load only reviewed, source-matched TRAIN bag color masks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_STORES = {}


class BagMaskStore:
    def __init__(self, manifest_path, images_root):
        self.manifest_path = Path(manifest_path).resolve()
        self.images_root = Path(images_root).resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding='utf-8-sig'))
        self.entries = {}
        self.cache = {}
        for entry in manifest.get('images', []):
            if entry.get('review_status') != 'accepted':
                continue
            filename = entry['file_name']
            if filename in self.entries:
                raise ValueError('Duplicate reviewed bag mask: ' + filename)
            self.entries[filename] = entry

    def get(self, image_item):
        from PIL import Image
        filename = image_item['file_name']
        entry = self.entries.get(filename)
        if entry is None:
            return None
        if filename not in self.cache:
            source = (self.images_root / filename).resolve()
            if not source.is_relative_to(self.images_root):
                raise ValueError('Bag source image outside dataset')
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry['source_sha256']:
                raise ValueError('Bag mask belongs to different source pixels: ' + filename)
            mask_path = (self.manifest_path.parent / entry['mask_file']).resolve()
            if not mask_path.is_relative_to(self.manifest_path.parent):
                raise ValueError('Bag mask outside reviewed mask directory')
            if entry.get('mask_sha256') and hashlib.sha256(mask_path.read_bytes()).hexdigest() != entry['mask_sha256']:
                raise ValueError('Reviewed mask file was modified: ' + filename)
            with Image.open(mask_path) as image:
                mask = image.convert('L')
            if mask.size != (image_item['width'], image_item['height']) or not mask.getbbox():
                raise ValueError('Invalid reviewed bag mask dimensions or empty mask: ' + filename)
            self.cache[filename] = mask
        return self.cache[filename].copy()


def load_bag_mask(manifest_path, image_item, images_root):
    path = Path(manifest_path).resolve()
    if not path.is_file():
        return None
    key = (str(path), str(Path(images_root).resolve()), path.stat().st_mtime_ns)
    if key not in _STORES:
        _STORES[key] = BagMaskStore(path, images_root)
    return _STORES[key].get(image_item)
