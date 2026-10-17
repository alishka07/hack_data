"""Local RT-DETRv2-L image inference; usable from the global Jupyter kernel."""
from pathlib import Path
import argparse
import json
import time

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, ToTensor

from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_label2category, mscoco_category2name

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'weights/rtdetrv2_r50vd_6x_coco_ema.pth'


def load_model(device=None):
    device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    cfg = YAMLConfig(str(ROOT / 'configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml'))
    # The complete checkpoint already contains the backbone weights.
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    model = cfg.model
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    model = model.deploy().eval().to(device)
    postprocessor = cfg.postprocessor.deploy().to(device)
    return model, postprocessor, device


@torch.inference_mode()
def predict(image, loaded, threshold=0.5):
    model, postprocessor, device = loaded
    image = image.convert('RGB')
    data = Compose([Resize((640, 640)), ToTensor()])(image).unsqueeze(0).to(device)
    size = torch.tensor([image.size], device=device)
    labels, boxes, scores = postprocessor(model(data), size)
    detections = []
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    for label, box, score in zip(labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()):
        if float(score) < threshold:
            continue
        name = mscoco_category2name[mscoco_label2category[int(label)]]
        coords = box.tolist()
        detections.append(dict(label=name, score=float(score), box=coords))
        draw.rectangle(coords, outline='red', width=2)
        draw.text((max(0, coords[0]), max(0, coords[1])), f'{name} {float(score):.2f}', fill='red')
    return rendered, detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    loaded = load_model(args.device)
    print(f'Model loaded on {loaded[2]}', flush=True)
    start = time.perf_counter()
    rendered, detections = predict(Image.open(args.image), loaded)
    elapsed = time.perf_counter() - start
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output)
    report = dict(device=str(loaded[2]), inference_seconds=elapsed, detections=detections)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
