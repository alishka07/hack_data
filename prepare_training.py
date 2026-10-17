"""Merge visually reviewed equipment and bag annotations into a video-split COCO pilot."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / 'training'
CATEGORIES = [{'id':1,'name':'safe'}, {'id':2,'name':'banknote_counter'}, {'id':3,'name':'laptop'}, {'id':4,'name':'bag'}]
VALIDATION_VIDEOS = {2, 14, 17, 20, 34, 38}


def main():
    from PIL import Image
    equipment = json.loads((TRAINING/'equipment_review.json').read_text(encoding='utf-8'))
    bags = json.loads((TRAINING/'bag_review.json').read_text(encoding='utf-8'))
    key = lambda row: (row['video'], int(row['source_frame']))
    bag_map = {key(row):row for row in bags['images']}
    if len(bag_map) != len(bags['images']):
        raise ValueError('Duplicate bag-review image keys')
    output = TRAINING/'dataset'
    (output/'images').mkdir(parents=True, exist_ok=True)
    datasets = {split:{'info':{'description':'Visually reviewed local pilot; annotation by assistant, safe identity confirmed by user'},
                       'images':[], 'annotations':[], 'categories':CATEGORIES} for split in ('train','val')}
    categories = {c['name']:c['id'] for c in CATEGORIES}
    manifest = {'annotation_source':'assistant_visual_review_of_frames; user_confirmation_of_safe',
                'automated_pseudolabels_used':False, 'split_method':'whole_video',
                'validation_video_numbers':sorted(VALIDATION_VIDEOS), 'images':[], 'excluded':[],
                'limitations':['Only one physical safe and counter, two bag appearances, and one staged room.',
                               'Validation videos are separate, but room, actors and hardware are similar.',
                               'Locks inform annotation; no separate lock detector is trained.',
                               'Bag presence does not establish contents or a banking violation.']}
    image_hashes = {}
    annotation_id = 0
    for image_id, row in enumerate(equipment['images'], 1):
        item_key = key(row)
        if item_key not in bag_map:
            raise ValueError(f'Missing bag review: {item_key}; unlabeled positives must not become negatives')
        bag = bag_map[item_key]
        if bag.get('review_status') != 'accepted':
            manifest['excluded'].append({'video':row['video'],'source_frame':row['source_frame'],'reason':bag.get('exclude_reason','ambiguous bag review')})
            continue
        if not bag.get('bag_review_complete'):
            raise ValueError(f'Incomplete bag review: {item_key}')
        source = Path(row['image_path'])
        if source.read_bytes() != Path(bag['image_path']).read_bytes():
            raise ValueError(f'Reviewers used different images: {item_key}')
        number = int(re.search(r'\d+', row['video']).group())
        split = 'val' if number in VALIDATION_VIDEOS else 'train'
        with Image.open(source) as im:
            width, height = im.size
            pixel_hash = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        if (width,height) != (row['width'],row['height']):
            raise ValueError(f'Dimension mismatch: {item_key}')
        if pixel_hash in image_hashes:
            raise ValueError(f'Duplicate pixels: {item_key} and {image_hashes[pixel_hash]}')
        image_hashes[pixel_hash] = item_key
        filename = f'images/X{number}_f{row["source_frame"]:06d}.jpg'
        shutil.copy2(source, output/filename)
        image = {'id':image_id, 'file_name':filename, 'width':width, 'height':height,
                 'source_video':row['video'], 'source_frame':row['source_frame'], 'timestamp_sec':row['timestamp_sec']}
        datasets[split]['images'].append(image)
        annotations = [*row['annotations'], *bag['annotations']]
        for ann in annotations:
            x1,y1,x2,y2 = ann['bbox_xyxy']
            if not (0<=x1<x2<=width and 0<=y1<y2<=height):
                raise ValueError(f'Invalid reviewed rectangle: {item_key}, {ann}')
            annotation_id += 1
            datasets[split]['annotations'].append({'id':annotation_id, 'image_id':image_id,
                'category_id':categories[ann['class_name']], 'bbox':[x1,y1,x2-x1,y2-y1],
                'area':(x2-x1)*(y2-y1), 'iscrowd':0, 'review_evidence':ann.get('evidence',''),
                'bag_context':ann.get('context')})
        manifest['images'].append(dict(image, split=split, sha256_pixels=pixel_hash,
                                     classes=[a['class_name'] for a in annotations]))
    manifest['counts'] = {split:{'images':len(data['images']), 'boxes':len(data['annotations']),
                         'videos':sorted({im['source_video'] for im in data['images']})} for split,data in datasets.items()}
    for split,data in datasets.items():
        (output/f'{split}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest['counts'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
