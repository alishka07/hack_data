"""Offline Grounding DINO experiments. Detections are candidates, not violations."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['USE_TF'] = '0'

import argparse
import json
from pathlib import Path
import sys
import time

# Keep the user's global packages intact; use the version verified for these weights.
ROOT = Path(__file__).resolve().parent
if (ROOT / 'compat').is_dir():
    sys.path.insert(0, str(ROOT / 'compat'))

import cv2
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers.utils import logging as transformers_logging
transformers_logging.disable_progress_bar()

PROMPTS = {
    'equipment': 'a metal safe. a computer monitor. a laptop. a desktop computer tower. a banknote counting machine. a door.',
    'cash_bags': 'a plastic bag. a transparent plastic bag. a bag. banknotes. a stack of money.',
    'desk_items': 'a smartphone. a usb flash drive. a notebook. a paper document. an identity card. a keyboard. a computer mouse.',
    'safe': 'a safe.',
    'plastic_bag': 'a plastic bag.',
}


def load_model():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(ROOT / 'weights', local_files_only=True, use_fast=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        ROOT / 'weights', local_files_only=True, disable_custom_kernels=True,
        attn_implementation='eager').eval().to(device)
    return processor, model, device


@torch.inference_mode()
def detect(image, loaded, prompt, threshold=0.3):
    processor, model, device = loaded
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device)
    outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, text_threshold=0.25,
        target_sizes=[(image.height, image.width)])[0]
    detections = []
    for label, box, score in zip(result['text_labels'], result['boxes'], result['scores']):
        detections.append({'label':label, 'box':box.cpu().tolist(), 'score':float(score)})
    return detections


def render(image, detections):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for i, item in enumerate(detections):
        color = ['#ffde00', '#00ffb3', '#ff8cdb', '#00d9ff'][i % 4]
        box = item['box']
        draw.rectangle(box, outline=color, width=2)
        label = f"{item['label']} {item['score']:.2f}"
        x, y = max(0, box[0]), max(0, box[1] - 12)
        draw.text((x, y), label, fill=color, stroke_width=1, stroke_fill='black')
    return image


def read_frame(video, second):
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot read {video} at {second}s')
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp
    finally:
        cap.release()


def run_frame(video, second, output, loaded=None, groups=None, threshold=0.3, crop=None):
    loaded = loaded or load_model()
    groups = groups or ['equipment', 'cash_bags', 'desk_items', 'safe']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image, timestamp = read_frame(video, second)
    if crop is not None:
        image = image.crop(tuple(crop))
    rows = []
    for group in groups:
        started = time.perf_counter()
        detections = detect(image, loaded, PROMPTS[group], threshold)
        name = f'{Path(video).stem}_{timestamp:.2f}_{group}' + ('_crop' if crop else '')
        picture = output / f'{name}.jpg'
        render(image, detections).save(picture, quality=95)
        row = dict(video=str(video), timestamp_sec=timestamp, group=group,
                   prompt=PROMPTS[group], threshold=threshold, detections=detections,
                   inference_seconds=time.perf_counter()-started, device=loaded[2],
                   image=picture.name, interpretation='unverified_object_candidates')
        row['crop_xyxy'] = crop
        (output / f'{name}.json').write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--second', type=float, default=10)
    parser.add_argument('--output', required=True)
    parser.add_argument('--groups', nargs='+', choices=list(PROMPTS))
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--crop', nargs=4, type=int)
    args = parser.parse_args()
    run_frame(args.video, args.second, args.output, groups=args.groups, threshold=args.threshold, crop=args.crop)
