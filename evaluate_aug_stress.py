"""Synthetic validation robustness diagnostic, never a new independent test.

Creates three fixed variants of each ORIGINAL validation image in a separate
directory. These files must not enter training or checkpoint/threshold selection.
Use --prepare-only for the CPU-only export; evaluation runs models sequentially.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import time

import cv2
from PIL import Image, ImageDraw, ImageOps

from calibrate_threshold import sha256_file
from compare_augmentation import category_signature, fixed_metrics, metric_delta, training_details
from object_augmentation import _adjust_hue, affine_matrix, apply_affine, horizontal_flip_boxes
from train_objects import LocalCocoDataset, collate, evaluate, load_trained, read_coco, write_json

ROOT = Path(__file__).resolve().parent
VARIANTS = ("hue_shift", "hflip", "scale_translate")
CAVEAT = (
    "Синтетическая диагностика устойчивости: три заранее заданных преобразования "
    "исходной validation. Это производные тех же кадров, а не новые наблюдения "
    "или независимый тест. Результаты нельзя объединять с исходной validation "
    "как дополнительные независимые примеры. Эти изображения и результаты не "
    "используются для обучения, выбора эпохи или подбора порогов. Изменение "
    "оттенка применяется ко всему кадру, а не только к мешку."
)


def make_variant(image, boxes, variant):
    if variant == "hue_shift":
        return _adjust_hue(image, 0.25), [list(box) for box in boxes], {"hue_turns": 0.25, "scope": "whole_image"}
    if variant == "hflip":
        return ImageOps.mirror(image), horizontal_flip_boxes(boxes, image.width), {"horizontal_flip": True}
    if variant == "scale_translate":
        matrix = affine_matrix(image.width, image.height, scale=0.82,
                               translate_x=0.05 * image.width, translate_y=-0.03 * image.height)
        transformed, changed, detail = apply_affine(image, boxes, matrix, min_visibility=1.0)
        if not detail["accepted"]:
            raise ValueError(f"Fixed stress geometry unexpectedly rejected: {detail}")
        return transformed, changed, {"scale": 0.82, "translate_fractions": [0.05, -0.03], **detail}
    raise ValueError(variant)


def prepare(data_root):
    cv2.setNumThreads(1)
    source_root = ROOT / "training/dataset"
    source_annotations = source_root / "val.json"
    data, categories, by_image, _ = read_coco(source_annotations, source_root)
    source_hash = sha256_file(source_annotations)
    training_hash = sha256_file(source_root / "train.json")
    data_root = Path(data_root).resolve()
    if data_root == source_root.resolve() or source_root.resolve() in data_root.parents:
        raise ValueError("Stress outputs must stay outside the original dataset directory")
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_annotation_sha256") != source_hash or manifest.get("variant_version") != 1:
            raise ValueError("Existing stress export has different source/settings; choose a fresh directory")
        if sha256_file(data_root / "annotations.json") != manifest["annotation_sha256"]:
            raise ValueError("Existing stress annotations were modified")
        for record in manifest["images"]:
            if sha256_file(source_root / record["source_file_name"]) != record["source_sha256"]:
                raise ValueError("An original validation image changed")
            if sha256_file(data_root / record["file_name"]) != record["sha256"]:
                raise ValueError("A synthetic stress image changed")
        return data_root / "annotations.json", manifest
    if data_root.exists() and any(data_root.iterdir()):
        raise FileExistsError("Unrecognized nonempty stress directory; choose a fresh directory")
    (data_root / "images").mkdir(parents=True, exist_ok=True)
    output = {"info": {"description": "SYNTHETIC VALIDATION DIAGNOSTIC ONLY; DO NOT TRAIN OR SELECT MODELS", "caveat": CAVEAT},
              "images": [], "annotations": [], "categories": categories}
    records, preview_tiles = [], []
    next_annotation = 1
    for source_index, item in enumerate(data["images"]):
        source = source_root / item["file_name"]
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        annotations = by_image[item["id"]]
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in annotations]
        for variant_index, variant in enumerate(VARIANTS):
            transformed, changed, settings = make_variant(image, boxes, variant)
            if len(changed) != len(annotations):
                raise ValueError("Stress transform dropped labels")
            image_id = 10000 + source_index * len(VARIANTS) + variant_index
            filename = f"images/{Path(item['file_name']).stem}__{variant}.png"
            destination = data_root / filename
            transformed.save(destination)
            output["images"].append({**item, "id": image_id, "file_name": filename,
                                     "source_image_id": item["id"], "stress_variant": variant,
                                     "synthetic": True, "not_for_training": True})
            for annotation, (x1, y1, x2, y2) in zip(annotations, changed):
                output["annotations"].append({"id": next_annotation, "image_id": image_id,
                    "category_id": annotation["category_id"], "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                    "source_annotation_id": annotation["id"]})
                next_annotation += 1
            records.append({"image_id": image_id, "source_image_id": item["id"], "file_name": filename,
                            "source_file_name": item["file_name"], "source_video": item.get("source_video"),
                            "source_sha256": sha256_file(source), "sha256": sha256_file(destination),
                            "variant": variant, "settings": settings})
            if source_index in (0, 5, 10):
                annotated = transformed.copy()
                draw = ImageDraw.Draw(annotated)
                for box, annotation in zip(changed, annotations):
                    draw.rectangle(tuple(box), outline="#43e087" if annotation["category_id"] == 4 else "#ffb547", width=2)
                annotated.thumbnail((232, 424))
                tile = Image.new("RGB", (252, 456), "white")
                tile.paste(annotated, (10, 25))
                ImageDraw.Draw(tile).text((7, 7), f"{Path(item['file_name']).stem} {variant}", fill="black")
                preview_tiles.append(tile)
    annotation_path = data_root / "annotations.json"
    write_json(annotation_path, output)
    read_coco(annotation_path, data_root)
    manifest = {"variant_version": 1, "synthetic": True, "independent_test": False,
                "for_training": False, "for_model_selection": False, "for_threshold_fitting": False,
                "caveat": CAVEAT, "source_annotations": str(source_annotations),
                "source_annotation_sha256": source_hash, "original_train_annotation_sha256": training_hash,
                "annotation_sha256": sha256_file(annotation_path), "source_images": len(data["images"]),
                "derived_images": len(records), "derived_boxes": len(output["annotations"]),
                "variants": list(VARIANTS), "images": records}
    write_json(manifest_path, manifest)
    preview = Image.new("RGB", (252 * 3, 456 * 3), "#cccccc")
    for index, tile in enumerate(preview_tiles):
        preview.paste(tile, ((index % 3) * 252, (index // 3) * 456))
    preview.save(data_root / "stress_preview.jpg", quality=92)
    if sha256_file(source_annotations) != source_hash or sha256_file(source_root / "train.json") != training_hash:
        raise RuntimeError("Original annotations changed during stress export")
    print(f"Prepared {len(records)} synthetic frames; original train/val untouched: {data_root}", flush=True)
    return annotation_path, manifest


def subset_metrics(data, predictions, thresholds):
    result = {}
    for variant in VARIANTS:
        images = [item for item in data["images"] if item["stress_variant"] == variant]
        ids = {item["id"] for item in images}
        subset = {"categories": data["categories"], "images": images,
                  "annotations": [item for item in data["annotations"] if item["image_id"] in ids]}
        result[variant] = fixed_metrics(subset, [item for item in predictions if item["image_id"] in ids], thresholds)
    return result


def run(args):
    annotation_path, manifest = prepare(args.data_root)
    if args.prepare_only:
        return
    checkpoints = {role: Path(getattr(args, role)).resolve() for role in ("baseline", "control", "augmented")}
    calibration_path = ROOT / "training/rtdetr_custom_final/calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(checkpoints["baseline"]) != calibration["checkpoint_sha256"]:
        raise ValueError("Threshold calibration does not match baseline")
    if calibration["validation_annotation_sha256"] != manifest["source_annotation_sha256"]:
        raise ValueError("Stress source is not the original calibration validation")
    data, categories, _, _ = read_coco(annotation_path, args.data_root)
    thresholds = calibration["thresholds"]
    output = Path(args.output or args.data_root / "evaluation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "stress_comparison.json").exists() and not args.overwrite:
        raise FileExistsError("Stress results already exist; choose --output or explicitly --overwrite")
    import torch
    torch.set_num_threads(2)
    report = {"status": "running", "caveat": CAVEAT, "independent_test": False, "for_model_selection": False,
              "source_manifest": str(Path(args.data_root).resolve() / "manifest.json"),
              "source_manifest_sha256": sha256_file(Path(args.data_root) / "manifest.json"),
              "annotation_sha256": sha256_file(annotation_path), "derived_images": len(data["images"]),
              "fixed_thresholds": thresholds, "match_iou_gte": 0.5, "same_class_nms_iou_gte": 0.6, "models": {}}
    started = time.perf_counter()
    for role, checkpoint in checkpoints.items():
        print(f"Synthetic diagnostic: {role}", flush=True)
        cpu = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if category_signature(cpu["metadata"]["categories"]) != category_signature(categories):
            raise ValueError("Checkpoint/stress categories differ")
        details = training_details(cpu["metadata"])
        del cpu
        gc.collect()
        model, postprocessor, loaded_categories, device, size = load_trained(checkpoint, args.device)
        dataset = LocalCocoDataset(annotation_path, args.data_root, size, augment=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
        prediction_path = output / f"{role}_predictions.json"
        metrics = evaluate(model, postprocessor, loader, loaded_categories, annotation_path, device, prediction_path)
        del model, postprocessor, dataset, loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        report["models"][role] = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "training": details, "metrics": metrics, "predictions_sha256": sha256_file(prediction_path),
            "fixed_threshold_metrics": fixed_metrics(data, predictions, thresholds),
            "by_variant_fixed_threshold_metrics": subset_metrics(data, predictions, thresholds)}
        write_json(output / "stress_comparison.partial.json", report)
        print(json.dumps({"role": role, **metrics, "bag": report["models"][role]["fixed_threshold_metrics"].get("bag")}), flush=True)
    report["deltas"] = {"augmented_minus_baseline": metric_delta(report["models"]["baseline"], report["models"]["augmented"]),
                        "augmented_minus_control": metric_delta(report["models"]["control"], report["models"]["augmented"])}
    report.update(status="completed", seconds=time.perf_counter() - started)
    write_json(output / "stress_comparison.json", report)
    lines = ["Синтетическая проверка устойчивости к преобразованиям", "", CAVEAT, "",
             "48 производных кадров: 16 исходных × 3 варианта. Исходные train/val сохранены.",
             "Варианты: оттенок всего кадра +0.25 оборота; горизонтальное отражение; масштаб 0.82 и сдвиг +5% ширины / -3% высоты.", ""]
    for role, result in report["models"].items():
        lines.append(f"{role}: mAP50={result['metrics']['map_50']:.4f}; mAP50:95={result['metrics']['map_50_95']:.4f}")
        for variant, fixed in result["by_variant_fixed_threshold_metrics"].items():
            bag = fixed["bag"]
            lines.append(f"  {variant}, bag: TP={bag['tp']}, FP={bag['fp']}, FN={bag['fn']}; precision={bag['precision']:.4f}; recall={bag['recall']:.4f}; F1={bag['f1']:.4f}")
        lines.append("")
    lines += ["AP и метрики всех классов сохранены в stress_comparison.json.",
              "Проверяется обнаружение объектов, а не нарушения или содержимое мешка."]
    (output / "stress_comparison_RU.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Completed synthetic diagnostic: {output}", flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "training/augmentation_stress")
    parser.add_argument("--baseline", type=Path, default=ROOT / "training/rtdetr_custom_final/best.pth")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--augmented", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and (not args.control or not args.augmented):
        parser.error("Evaluation requires --control and --augmented checkpoints")
    run(args)


if __name__ == "__main__":
    main()
