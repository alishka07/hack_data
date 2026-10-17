"""Compare three local detectors on the unchanged, clean validation set.

Runs the models sequentially. Does not train, recalibrate thresholds, augment
validation images, download anything, or change production checkpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time

from calibrate_threshold import counts_at_threshold, nms, sha256_file, xywh_to_xyxy
from train_objects import LocalCocoDataset, collate, evaluate, load_trained, read_coco, write_json

ROOT = Path(__file__).resolve().parent
CAVEAT = (
    "Это повторная проверка на исходной validation, которая уже использовалась "
    "для выбора модели и настройки исходных порогов. Это не независимый тест. "
    "Всего 16 исходных кадров одного помещения; результаты не устанавливают "
    "качество на новых камерах или точность выявления нарушений. Пороги для всех "
    "сравниваемых моделей зафиксированы по исходной модели и заново не подбираются."
)


def category_signature(categories):
    return [(item["id"], item["name"]) for item in categories]


def fixed_metrics(annotations, predictions, thresholds):
    """Use the project's existing matching/NMS, without fitting a threshold."""
    categories = sorted(annotations["categories"], key=lambda item: item["id"])
    category_ids = {item["id"] for item in categories}
    image_ids = {item["id"] for item in annotations["images"]}
    if set(thresholds) != {item["name"] for item in categories}:
        raise ValueError("Fixed thresholds must match the validation categories exactly")
    for value in thresholds.values():
        if not math.isfinite(float(value)) or not 0 <= value <= 1:
            raise ValueError("Fixed thresholds must be finite and in [0, 1]")
    truth = defaultdict(lambda: defaultdict(list))
    predicted = defaultdict(lambda: defaultdict(list))
    for item in annotations["annotations"]:
        if item["category_id"] not in category_ids or item["image_id"] not in image_ids:
            raise ValueError("Unknown annotation category/image")
        if item.get("iscrowd", 0):
            raise ValueError("Crowd matching is unsupported")
        truth[item["category_id"]][item["image_id"]].append(xywh_to_xyxy(item["bbox"]))
    for item in predictions:
        if item["category_id"] not in category_ids or item["image_id"] not in image_ids:
            raise ValueError("Prediction refers to an unknown category/image")
        score = float(item["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction scores must be finite and in [0, 1]")
        predicted[item["category_id"]][item["image_id"]].append(
            {"box": xywh_to_xyxy(item["bbox"]), "score": score}
        )
    result = {}
    for category in categories:
        cid, name = category["id"], category["name"]
        by_image = {image_id: nms(items, nms_iou=0.6) for image_id, items in predicted[cid].items()}
        result[name] = {
            "threshold": float(thresholds[name]),
            "ground_truth_boxes": sum(map(len, truth[cid].values())),
            **counts_at_threshold(by_image, truth[cid], float(thresholds[name]), match_iou=0.5),
        }
    return result


def metric_delta(first, second):
    """Second minus first; positive AP/precision/recall means an increase."""
    result = {key: second["metrics"][key] - first["metrics"][key] for key in ("map_50", "map_50_95")}
    result["per_class_ap50"] = {
        name: None if value is None or first["metrics"]["per_class_ap50"][name] is None
        else value - first["metrics"]["per_class_ap50"][name]
        for name, value in second["metrics"]["per_class_ap50"].items()
    }
    result["per_class_fixed_threshold"] = {
        name: {key: values[key] - first["fixed_threshold_metrics"][name][key]
               for key in ("tp", "fp", "fn", "precision", "recall", "f1")}
        for name, values in second["fixed_threshold_metrics"].items()
    }
    return result


def training_details(metadata):
    arguments = metadata.get("arguments", {})
    return {
        "selected_epoch": metadata.get("epoch"),
        "configured_epochs": arguments.get("epochs"),
        "arguments": arguments,
        "dataset": metadata.get("dataset"),
        "augmentation": metadata.get("augmentation"),
        "image_size": metadata.get("image_size"),
        "queries": metadata.get("queries"),
        "denoising": metadata.get("denoising"),
        "frozen": metadata.get("frozen"),
    }


def fairness_warnings(models):
    warnings = []
    control = models["control"]["training"]["arguments"]
    augmented = models["augmented"]["training"]["arguments"]
    for key in ("epochs", "image_size", "queries", "denoising", "batch_size", "lr", "seed", "no_amp"):
        if control.get(key) != augmented.get(key):
            warnings.append(f"Control/augmented differ in {key}: {control.get(key)!r} / {augmented.get(key)!r}")
    for role in ("control", "augmented"):
        source = models[role]["training"]["arguments"].get("weights")
        if source:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            if source_path.is_file():
                source_hash = sha256_file(source_path)
                models[role]["warm_start_checkpoint_sha256"] = source_hash
                if source_hash != models["baseline"]["checkpoint_sha256"]:
                    warnings.append(f"{role} warm start differs from the baseline checkpoint")
            else:
                warnings.append(f"Cannot verify {role} warm-start file: {source_path}")
        else:
            warnings.append(f"No warm-start checkpoint recorded for {role}")
    control_data = models["control"]["training"].get("dataset") or {}
    augmented_data = models["augmented"]["training"].get("dataset") or {}
    if control_data.get("train_images") != augmented_data.get("train_images"):
        warnings.append("Control/augmented training image counts differ; equal epochs do not imply equal optimizer steps")
    return warnings


def russian_report(summary):
    lines = ["Сравнение аугментации объектов кассы", "", summary["caveat"], ""]
    names = {"baseline": "Исходная модель", "control": "Контрольное дообучение", "augmented": "Дообучение с расширенной аугментацией"}
    for role, result in summary["models"].items():
        metrics = result["metrics"]
        lines += [names[role], f"  Веса: {result['checkpoint']}",
                  f"  Выбранная эпоха: {result['training']['selected_epoch']}",
                  f"  mAP50: {metrics['map_50']:.4f}; mAP50:95: {metrics['map_50_95']:.4f}"]
        for name, fixed in result["fixed_threshold_metrics"].items():
            ap = metrics["per_class_ap50"][name]
            ap_text = "нет положительных примеров" if ap is None else f"{ap:.4f}"
            lines.append(f"  {name}: AP50={ap_text}; порог={fixed['threshold']:.2f}; "
                         f"TP={fixed['tp']}, FP={fixed['fp']}, FN={fixed['fn']}; "
                         f"precision={fixed['precision']:.4f}; recall={fixed['recall']:.4f}; F1={fixed['f1']:.4f}")
        lines.append("")
    lines += ["Разница: расширенная аугментация минус контрольное дообучение"]
    delta = summary["deltas"]["augmented_minus_control"]
    lines += [f"  mAP50: {delta['map_50']:+.4f}; mAP50:95: {delta['map_50_95']:+.4f}"]
    if "bag" in delta["per_class_fixed_threshold"]:
        bag = delta["per_class_fixed_threshold"]["bag"]
        ap = delta["per_class_ap50"]["bag"]
        lines += [f"  bag AP50: {ap:+.4f}" if ap is not None else "  bag AP50: недостаточно данных",
                  f"  bag TP: {bag['tp']:+d}; FP: {bag['fp']:+d}; FN: {bag['fn']:+d}; "
                  f"precision: {bag['precision']:+.4f}; recall: {bag['recall']:+.4f}; F1: {bag['f1']:+.4f}"]
    lines += ["", "Условия сравнения: исходная validation без преобразований; "
              "IoU сопоставления >= 0.5, NMS одного класса >= 0.6, фиксированные исходные пороги.",
              "Все рамки каждого класса сопоставляются один к одному. Ложные срабатывания учитываются и на кадрах без объекта.",
              "AP вычисляется по всем предсказаниям стандартным COCO evaluator; precision/recall — отдельно при фиксированных порогах.",
              "Проверяется распознавание объектов, а не действия сотрудников или наличие денег внутри мешка.",
              "Для решения о замене модели также нужна визуальная проверка ошибок на новых видео."]
    if summary["fairness_warnings"]:
        lines += ["", "Ограничения сопоставимости эксперимента:"] + ["  " + item for item in summary["fairness_warnings"]]
    return "\n".join(lines) + "\n"


def compare(args):
    # Validate files and provenance before creating any GPU model.
    checkpoints = {role: Path(getattr(args, role)).resolve() for role in ("baseline", "control", "augmented")}
    annotation_path, images_root, calibration_path = map(Path, (args.annotations, args.images_root, args.calibration))
    for path in [*checkpoints.values(), annotation_path, calibration_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    data, categories, _, counts = read_coco(annotation_path, images_root)
    annotation_hash = sha256_file(annotation_path)
    checkpoint_hashes = {role: sha256_file(path) for role, path in checkpoints.items()}
    calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    if calibration.get("checkpoint_sha256") != checkpoint_hashes["baseline"]:
        raise ValueError("Calibration is not from the specified baseline checkpoint")
    if calibration.get("validation_annotation_sha256") != annotation_hash:
        raise ValueError("Calibration/validation annotation hash mismatch; use the unchanged original validation")
    thresholds = calibration["thresholds"]
    # This validates fixed threshold keys and values, without any inference.
    fixed_metrics(data, [], thresholds)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists() and not args.overwrite:
        raise FileExistsError(f"Comparison already exists at {output}; use a fresh output or --overwrite")
    image_hashes = [{"image_id": item["id"], "file_name": item["file_name"],
                     "sha256": sha256_file(images_root / item["file_name"])} for item in data["images"]]

    import torch
    torch.set_num_threads(args.cpu_threads)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "independent_test": False, "clean_validation": True, "validation_augmented": False,
        "thresholds_refitted": False, "caveat": CAVEAT.replace("16 исходных кадров", f"{len(data['images'])} исходных кадров"),
        "validation": {"annotations": str(annotation_path.resolve()), "annotation_sha256": annotation_hash,
                       "images_root": str(images_root.resolve()), "images": len(data["images"]),
                       "boxes": {category["name"]: counts[category["id"]] for category in categories},
                       "source_videos": sorted({item.get("source_video", "") for item in data["images"]}),
                       "image_file_hashes": image_hashes},
        "calibration": {"path": str(calibration_path.resolve()), "sha256": sha256_file(calibration_path)},
        "method": {"match_iou_gte": 0.5, "same_class_nms_iou_gte": 0.6, "score_comparison": ">=",
                   "thresholds": thresholds, "matching": "greedy_descending_score_per_image_per_class",
                   "precision_recall_on_all_validation_images": True},
        "models": {},
    }
    started = time.perf_counter()
    for role, checkpoint_path in checkpoints.items():
        print(f"Evaluating {role}: {checkpoint_path}", flush=True)
        cpu_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = cpu_checkpoint["metadata"]
        if category_signature(metadata["categories"]) != category_signature(categories):
            raise ValueError(f"Checkpoint categories/order do not match clean validation: {role}")
        trained_val_hash = (metadata.get("dataset", {}).get("annotation_sha256", {}).get("val"))
        if trained_val_hash and trained_val_hash != annotation_hash:
            raise ValueError(f"Checkpoint {role} was selected using different validation annotations")
        details = training_details(metadata)
        del cpu_checkpoint, metadata
        gc.collect()
        model, postprocessor, loaded_categories, device, image_size = load_trained(checkpoint_path, args.device)
        dataset = LocalCocoDataset(annotation_path, images_root, image_size, augment=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
        prediction_path = output / f"{role}_val_predictions.json"
        inference_started = time.perf_counter()
        metrics = evaluate(model, postprocessor, loader, loaded_categories, annotation_path, device, prediction_path)
        elapsed = time.perf_counter() - inference_started
        del model, postprocessor, dataset, loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        summary["models"][role] = {
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hashes[role],
            "predictions": str(prediction_path), "predictions_sha256": sha256_file(prediction_path),
            "training": details, "metrics": metrics,
            "fixed_threshold_metrics": fixed_metrics(data, predictions, thresholds),
            "evaluation_seconds": elapsed,
        }
        write_json(output / "comparison.partial.json", summary)
        print(json.dumps({"role": role, **metrics, "bag": summary["models"][role]["fixed_threshold_metrics"].get("bag")}), flush=True)
    summary["fairness_warnings"] = fairness_warnings(summary["models"])
    summary["deltas"] = {
        "augmented_minus_baseline": metric_delta(summary["models"]["baseline"], summary["models"]["augmented"]),
        "control_minus_baseline": metric_delta(summary["models"]["baseline"], summary["models"]["control"]),
        "augmented_minus_control": metric_delta(summary["models"]["control"], summary["models"]["augmented"]),
    }
    summary.update(status="completed", seconds=time.perf_counter() - started)
    write_json(output / "comparison.json", summary)
    (output / "comparison_RU.txt").write_text(russian_report(summary), encoding="utf-8-sig")
    print(f"Completed comparison: {output / 'comparison_RU.txt'}", flush=True)
    return summary


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=ROOT / "training/rtdetr_custom_final/best.pth")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=ROOT / "training/dataset/val.json")
    parser.add_argument("--images-root", type=Path, default=ROOT / "training/dataset")
    parser.add_argument("--calibration", type=Path, default=ROOT / "training/rtdetr_custom_final/calibration.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    compare(args)


if __name__ == "__main__":
    main()
