"""Train a real RT-DETRv2 detector on local, reviewed COCO annotations.

Uses the existing global Python and downloaded RT-DETR weights. No downloads,
cloud calls, virtual environment, or TensorBoard are needed. Run --help first.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "runtime" / "RT-DETR-main" / "rtdetrv2_pytorch"
BASE_WEIGHTS = REPO / "weights" / "rtdetrv2_r50vd_6x_coco_ema.pth"


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_coco(path, images_root):
    path, images_root = Path(path), Path(images_root).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    categories = sorted(data["categories"], key=lambda item: item["id"])
    if not categories or len({c["id"] for c in categories}) != len(categories):
        raise ValueError(f"Missing or duplicate categories: {path}")
    if len({c["name"] for c in categories}) != len(categories):
        raise ValueError(f"Category names must be unique: {path}")
    images = {item["id"]: item for item in data["images"]}
    if not images or len(images) != len(data["images"]):
        raise ValueError(f"Missing or duplicate images: {path}")
    category_ids = {c["id"] for c in categories}
    counts = Counter()
    annotations = defaultdict(list)
    seen_ids = set()
    for item in images.values():
        image_path = (images_root / item["file_name"]).resolve()
        if not image_path.is_relative_to(images_root):
            raise ValueError(f"Image outside images-root: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if item["width"] <= 0 or item["height"] <= 0:
            raise ValueError(f"Invalid image size: {item['file_name']}")
    for ann in data["annotations"]:
        if ann["id"] in seen_ids:
            raise ValueError(f"Duplicate annotation id {ann['id']}: {path}")
        seen_ids.add(ann["id"])
        if ann["image_id"] not in images or ann["category_id"] not in category_ids:
            raise ValueError(f"Unknown image/category in annotation {ann['id']}")
        x, y, width, height = map(float, ann["bbox"])
        im = images[ann["image_id"]]
        if not all(math.isfinite(v) for v in [x, y, width, height]):
            raise ValueError(f"Non-finite box in annotation {ann['id']}")
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > im["width"] + 1 or y + height > im["height"] + 1:
            raise ValueError(f"Invalid/outside-image box in annotation {ann['id']}")
        if ann.get("iscrowd", 0):
            raise ValueError("This small training runner requires ordinary boxes, not iscrowd annotations")
        annotations[ann["image_id"]].append(ann)
        counts[ann["category_id"]] += 1
    return data, categories, annotations, counts


def check_data(train_path, val_path, images_root):
    train = read_coco(train_path, images_root)
    val = read_coco(val_path, images_root)
    signature = lambda cats: [(c["id"], c["name"]) for c in cats]
    if signature(train[1]) != signature(val[1]):
        raise ValueError("Train/val category IDs and names differ")
    train_files = {item["file_name"] for item in train[0]["images"]}
    val_files = {item["file_name"] for item in val[0]["images"]}
    if train_files & val_files:
        raise ValueError("Train/val contain the same image file")
    video_sets = [set(str(i["source_video"]) for i in d[0]["images"] if i.get("source_video")) for d in [train, val]]
    if video_sets[0] & video_sets[1]:
        raise ValueError(f"Train/val share source videos: {sorted(video_sets[0] & video_sets[1])}")
    # Filenames alone do not separate duplicate exports of the same recording.
    # Legacy datasets have neither field; retain their behaviour while making
    # declared hash/group contracts complete across both splits.
    source_group_checks = {}
    all_images = [item for data in (train, val) for item in data[0]["images"]]
    for field in ("source_sha256", "source_group"):
        if not any(field in item for item in all_images):
            source_group_checks[field] = False
            continue
        normalized = []
        for data in (train, val):
            values = set()
            for item in data[0]["images"]:
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"All images must declare nonempty {field} when this metadata is used: {item['file_name']}")
                value = value.strip()
                if field == "source_sha256":
                    value = value.lower()
                    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                        raise ValueError(f"Invalid source_sha256: {item['file_name']}")
                elif value.lower().startswith("sha256:"):
                    value = value.lower()
                values.add(value)
            normalized.append(values)
        overlap = normalized[0] & normalized[1]
        if overlap:
            raise ValueError(f"Train/val share {field} source groups: {sorted(overlap)}")
        source_group_checks[field] = True
    missing = [c["name"] for c in train[1] if not train[3][c["id"]]]
    if missing:
        raise ValueError(f"No training boxes for categories: {missing}")
    return {
        "train_images": len(train_files), "val_images": len(val_files),
        "categories": train[1],
        "train_boxes": {c["name"]: train[3][c["id"]] for c in train[1]},
        "val_boxes": {c["name"]: val[3][c["id"]] for c in val[1]},
        "video_split_verified": all(video_sets) and all(all(i.get("source_video") for i in d[0]["images"]) for d in [train, val]),
        "source_group_split_checks": source_group_checks,
        "annotation_sha256": {"train": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(), "val": hashlib.sha256(Path(val_path).read_bytes()).hexdigest()},
    }


def classification_initialization(source_metadata, target_categories):
    """Describe head reuse and reject a silent same-size label remapping."""
    target_names = [item["name"] for item in target_categories]
    source_categories = (source_metadata or {}).get("categories")
    if source_categories is None:
        return {"label_source": "pretrained_checkpoint_without_category_metadata",
                "source_category_names": None, "target_category_names": target_names,
                "mode": "shape_compatible_pretrained_initialization"}
    if (not isinstance(source_categories, list) or not source_categories
            or any(not isinstance(item, dict) or not isinstance(item.get("name"), str)
                   or not item["name"].strip() for item in source_categories)):
        raise ValueError("Custom checkpoint category metadata is invalid")
    source_names = [item["name"] for item in source_categories]
    if len(set(source_names)) != len(source_names):
        raise ValueError("Custom checkpoint category names must be unique")
    same_count = len(source_names) == len(target_names)
    if same_count and source_names != target_names:
        raise ValueError("Custom checkpoint category names/order differ at the same class count; "
                         "refusing incorrect classification-head reuse")
    return {"label_source": "custom_checkpoint_category_metadata",
            "source_category_names": source_names, "target_category_names": target_names,
            "mode": "reuse_matching_class_order" if same_count else "reset_changed_class_count"}


def runtime():
    os.environ["USE_TF"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import torch
    from src.core import YAMLConfig
    return torch, YAMLConfig


def build_model(num_classes, image_size=416, queries=100, denoising=20):
    torch, YAMLConfig = runtime()
    config = YAMLConfig(str(REPO / "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"))
    config.yaml_cfg.update(num_classes=num_classes, remap_mscoco_category=False, eval_spatial_size=[image_size, image_size])
    config.yaml_cfg["PResNet"]["pretrained"] = False
    config.yaml_cfg["RTDETRTransformerv2"].update(num_queries=queries, num_denoising=denoising)
    config.yaml_cfg["RTDETRPostProcessor"]["num_top_queries"] = min(100, queries * num_classes)
    return config.model, config.criterion, config.postprocessor


class LocalCocoDataset:
    def __init__(self, annotation_path, images_root, image_size, augment=False, augmentation=None, bag_masks=None):
        self.data, self.categories, self.annotations, _ = read_coco(annotation_path, images_root)
        self.items = self.data["images"]
        self.category_to_label = {c["id"]: index for index, c in enumerate(self.categories)}
        self.images_root = Path(images_root)
        self.image_size = image_size
        self.augment = augment
        self.augmentation = (augmentation or 'legacy') if augment else 'none'
        if self.augmentation not in {'none', 'legacy', 'standard', 'bag_focus'}:
            raise ValueError('Unknown augmentation profile: ' + str(self.augmentation))
        self.bag_label = next((self.category_to_label[c['id']] for c in self.categories if c['name'] == 'bag'), None)
        self.bag_masks = Path(bag_masks or ROOT / 'training/bag_color_masks/manifest.json')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF
        item = self.items[index]
        with Image.open(self.images_root / item["file_name"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        if (width, height) != (item["width"], item["height"]):
            raise ValueError(f"Image dimensions differ from annotations: {item['file_name']}")
        boxes, labels = [], []
        for ann in self.annotations[item["id"]]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_to_label[ann["category_id"]])
        if self.augmentation == 'legacy':
            if random.random() < 0.5:
                image = TF.hflip(image)
                boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
        elif self.augmentation != 'none':
            from object_augmentation import augment_sample
            from bag_mask_io import load_bag_mask
            mask = load_bag_mask(self.bag_masks, item, self.images_root) if self.augmentation == 'bag_focus' else None
            image, boxes, labels, _ = augment_sample(image, boxes, labels, bag_label=self.bag_label,
                                                    rng=random, profile=self.augmentation, bag_mask=mask)
        boxes = torch.tensor([[(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height]
                              for x1, y1, x2, y2 in boxes], dtype=torch.float32).reshape(-1, 4)
        image = TF.to_tensor(TF.resize(image, [self.image_size, self.image_size], antialias=True))
        target = {"labels": torch.tensor(labels, dtype=torch.long), "boxes": boxes,
                  "image_id": torch.tensor([item["id"]]), "orig_size": torch.tensor([width, height])}
        return image, target


def collate(items):
    import torch
    images, targets = zip(*items)
    return torch.stack(images), list(targets)


def float_outputs(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: float_outputs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [float_outputs(v) for v in value]
    return value


def evaluate(model, postprocessor, loader, categories, annotation_path, device, predictions_path):
    import torch
    from faster_coco_eval import COCO, COCOeval_faster
    model.eval()
    predictions = []
    with torch.inference_mode():
        for images, targets in loader:
            output = model(images.to(device))
            sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = postprocessor(output, sizes)
            for target, result in zip(targets, results):
                width, height = target["orig_size"].tolist()
                for label, box, score in zip(result["labels"].cpu(), result["boxes"].cpu(), result["scores"].cpu()):
                    x1, y1, x2, y2 = box.tolist()
                    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                    if x2 > x1 and y2 > y1:
                        predictions.append({"image_id": int(target["image_id"].item()), "category_id": categories[int(label)]["id"],
                                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    write_json(predictions_path, predictions)
    if not predictions:
        return {"map_50_95": 0.0, "map_50": 0.0, "per_class_ap50": {c["name"]: 0.0 for c in categories}}
    with contextlib.redirect_stdout(io.StringIO()):
        # The library opens filenames with the Windows locale encoding.
        # Decode our UTF-8 JSON explicitly so Russian metadata remains valid.
        truth = COCO(json.loads(Path(annotation_path).read_text(encoding="utf-8-sig")))
        truth.dataset.setdefault("info", {})
        detection = truth.loadRes(predictions)
        evaluator = COCOeval_faster(truth, detection, "bbox")
        evaluator.params.catIds = [c["id"] for c in categories]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    per_class = {}
    for index, category in enumerate(categories):
        values = precision[0, :, index, 0, -1]
        valid = values[values > -1]
        per_class[category["name"]] = float(valid.mean()) if len(valid) else None
    return {"map_50_95": float(evaluator.stats[0]), "map_50": float(evaluator.stats[1]), "per_class_ap50": per_class}


def save_checkpoint(path, model, metadata):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "metadata": metadata}, temporary)
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2**attempt, .5))


def load_trained(checkpoint_path, device=None):
    """Return model, postprocessor, categories, device, image_size for inference."""
    torch, _ = runtime()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    meta = checkpoint["metadata"]
    model, _, postprocessor = build_model(len(meta["categories"]), meta["image_size"], meta["queries"], meta["denoising"])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    return model.eval().to(device), postprocessor.eval().to(device), meta["categories"], device, meta["image_size"]


def predict_image(image, loaded, threshold=0.35):
    import torch
    from torchvision.transforms import functional as TF
    model, postprocessor, categories, device, size = loaded
    image = image.convert("RGB")
    tensor = TF.to_tensor(TF.resize(image, [size, size], antialias=True)).unsqueeze(0).to(device)
    with torch.inference_mode():
        results = postprocessor(model(tensor), torch.tensor([image.size], device=device))[0]
    return [{"label": categories[int(label)]["name"], "score": float(score), "box": box.tolist()}
            for label, box, score in zip(results["labels"].cpu(), results["boxes"].cpu(), results["scores"].cpu()) if float(score) >= threshold]


def train(args):
    summary = check_data(args.train, args.val, args.images_root)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.check_data:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pth").exists() and not args.overwrite:
        raise FileExistsError(f"Existing training at {output}. Choose another --output or explicitly use --overwrite")
    torch, _ = runtime()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    # Augmentation uses OpenCV as well as Torch. Bound both CPU pools so a
    # small-RAM Windows machine does not create a second large worker pool.
    import cv2
    cv2.setNumThreads(1)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    model, criterion, postprocessor = build_model(len(summary["categories"]), args.image_size, args.queries, args.denoising)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    head_initialization = classification_initialization(checkpoint.get("metadata"), summary["categories"])
    source = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    target = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed = ("decoder.enc_score_head.", "decoder.dec_score_head.", "decoder.denoising_class_embed.")
    # These deterministic buffers are rebuilt for the selected image size.
    spatial_buffers = {"decoder.anchors", "decoder.valid_mask"}
    bad_missing = [key for key in missing if not key.startswith(allowed) and key not in spatial_buffers]
    if bad_missing or unexpected:
        raise RuntimeError(f"Unexpected pretrained mismatch: missing={bad_missing}; unexpected={unexpected}")
    del checkpoint, source, target, compatible
    for module in [model.backbone, model.encoder]:
        module.requires_grad_(False)
    model.to(device)
    criterion.to(device)
    postprocessor.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    train_set = LocalCocoDataset(args.train, args.images_root, args.image_size, augment=True, augmentation=args.augmentation, bag_masks=args.bag_masks)
    val_set = LocalCocoDataset(args.val, args.images_root, args.image_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    metadata = {"architecture": "RT-DETRv2-R50", "categories": summary["categories"], "image_size": args.image_size,
                "queries": args.queries, "denoising": args.denoising, "frozen": ["backbone", "encoder"],
                "new_classification_parameters": [key for key in missing if key.startswith(allowed)],
                "classification_initialization": head_initialization,
                "regenerated_spatial_buffers": [key for key in missing if key in spatial_buffers],
                "dataset": summary, "arguments": vars(args), "augmentation": args.augmentation,
                "source_checkpoint_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
                "bag_masks_manifest_sha256": hashlib.sha256(Path(args.bag_masks).read_bytes()).hexdigest() if args.augmentation == 'bag_focus' and Path(args.bag_masks).is_file() else None,
                "evaluation_caveat": "Small staged local set; these metrics do not establish accuracy on other cameras or banking violations."}
    if args.augmentation not in {'none', 'legacy'}:
        from object_augmentation import describe_profile
        metadata['augmentation_details'] = describe_profile(args.augmentation)
        metadata['augmentation_code_sha256'] = hashlib.sha256((ROOT / 'object_augmentation.py').read_bytes()).hexdigest()
    write_json(output / "training_setup.json", metadata)
    print(f"Device={device}; trainable={sum(p.numel() for p in parameters):,}; epochs={args.epochs}; images={len(train_set)}", flush=True)
    history, best, started = [], -1.0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        model.backbone.eval()
        model.encoder.eval()
        loss_sum, samples = 0.0, 0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                with torch.no_grad():
                    features = model.encoder(model.backbone(images))
                outputs = model.decoder(features, targets)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = criterion(float_outputs(outputs), targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 0.1)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(targets)
            samples += len(targets)
            if step == 1 or step % 10 == 0 or step == len(train_loader):
                print(f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} loss={float(loss.detach()):.4f}", flush=True)
            del images, outputs, losses, loss, features, targets
        metrics = evaluate(model, postprocessor, val_loader, summary["categories"], args.val, device, output / "val_predictions.json")
        row = {"epoch": epoch, "train_loss": loss_sum / samples, "seconds": time.perf_counter() - epoch_start, **metrics}
        history.append(row)
        metadata.update(epoch=epoch, metrics=metrics)
        save_checkpoint(output / "last.pth", model, metadata)
        if metrics["map_50_95"] > best:
            best = metrics["map_50_95"]
            save_checkpoint(output / "best.pth", model, metadata)
            write_json(output / "best_metrics.json", row)
            (output / "best_val_predictions.json").write_bytes((output / "val_predictions.json").read_bytes())
        write_json(output / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {"status": "completed", "seconds": time.perf_counter() - started, "epochs": args.epochs, "best_map_50_95": best,
              "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
              "checkpoint": str((output / "best.pth").resolve()), "history": history, "dataset": summary,
              "caveat": metadata["evaluation_caveat"]}
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", type=str, default=str(ROOT / "training/dataset/train.json"))
    parser.add_argument("--val", type=str, default=str(ROOT / "training/dataset/val.json"))
    parser.add_argument("--images-root", type=str, default=str(ROOT / "training/dataset"))
    parser.add_argument("--output", type=str, default=str(ROOT / "training/rtdetr_custom"))
    parser.add_argument("--weights", type=str, default=str(BASE_WEIGHTS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--denoising", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation", choices=['none', 'legacy', 'standard', 'bag_focus'], default='bag_focus',
                        help="Training only: stronger bag variation with bag_focus; validation is always unchanged")
    parser.add_argument("--bag-masks", type=str, default=str(ROOT / 'training/bag_color_masks/manifest.json'),
                        help="Optional reviewed TRAIN foreground masks for targeted bag recoloring")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="Validate COCO files without importing torch or using GPU")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing an existing output run")
    args = parser.parse_args()
    if args.image_size < 128 or args.image_size % 32:
        parser.error("--image-size must be a multiple of 32 and at least 128")
    if min(args.epochs, args.batch_size, args.queries, args.cpu_threads) <= 0 or args.denoising < 0 or args.lr <= 0:
        parser.error("Epochs, batch size, queries, threads, and learning rate must be positive; denoising >= 0")
    train(args)


if __name__ == "__main__":
    main()
