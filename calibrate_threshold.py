"""Fit per-class confidence thresholds on validation predictions (CPU only).

This is threshold fitting on validation data, NOT an independent test. Uses
only Python's standard library; does not import torch or allocate GPU memory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(box):
    if len(box) != 4:
        raise ValueError("COCO bbox must contain exactly four numbers")
    x, y, width, height = map(float, box)
    if not all(math.isfinite(v) for v in [x, y, width, height]) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid COCO bbox: {box}")
    return (x, y, x + width, y + height)


def iou(first, second):
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def nms(predictions, nms_iou=0.6):
    """Stable same-class, same-image NMS; callers must group accordingly."""
    kept = []
    for item in sorted(predictions, key=lambda p: p["score"], reverse=True):
        if not any(iou(item["box"], other["box"]) >= nms_iou for other in kept):
            kept.append(item)
    return kept


def counts_at_threshold(predictions_by_image, truth_by_image, threshold, match_iou=0.5):
    true_positive = false_positive = false_negative = 0
    for image_id in set(predictions_by_image) | set(truth_by_image):
        truth = truth_by_image.get(image_id, [])
        available = set(range(len(truth)))
        predictions = sorted(predictions_by_image.get(image_id, []), key=lambda p: p["score"], reverse=True)
        for pred in predictions:
            if pred["score"] < threshold:
                continue
            best_index, best_iou = None, -1.0
            for index in sorted(available):
                overlap = iou(pred["box"], truth[index])
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index is not None and best_iou >= match_iou:
                true_positive += 1
                available.remove(best_index)
            else:
                false_positive += 1
        false_negative += len(available)
    return metrics_from_counts(true_positive, false_positive, false_negative)


def metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def fit_thresholds(annotations, predictions, thresholds=None, match_iou=0.5, nms_iou=0.6):
    thresholds = list(thresholds if thresholds is not None else [step / 100 for step in range(2, 91)])
    if not thresholds or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in thresholds):
        raise ValueError("Thresholds must be finite numbers in [0, 1]")
    categories = sorted(annotations["categories"], key=lambda category: category["id"])
    if not categories or len({c["id"] for c in categories}) != len(categories) or len({c["name"] for c in categories}) != len(categories):
        raise ValueError("Categories must be present with unique IDs and names")
    category_ids = {category["id"] for category in categories}
    image_ids = {image["id"] for image in annotations["images"]}
    if not image_ids or len(image_ids) != len(annotations["images"]):
        raise ValueError("Validation images must have unique IDs")
    truth = defaultdict(lambda: defaultdict(list))
    predicted = defaultdict(lambda: defaultdict(list))
    for item in annotations["annotations"]:
        if item["category_id"] not in category_ids or item["image_id"] not in image_ids:
            raise ValueError("Annotation refers to an unknown category or image")
        if item.get("iscrowd", 0):
            raise ValueError("Crowd/ignore matching is unsupported; use ordinary reviewed boxes")
        truth[item["category_id"]][item["image_id"]].append(xywh_to_xyxy(item["bbox"]))
    for item in predictions:
        if item["category_id"] not in category_ids or item["image_id"] not in image_ids:
            raise ValueError("Prediction refers to an unknown validation category or image")
        score = float(item["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid prediction score: {score}")
        predicted[item["category_id"]][item["image_id"]].append({"score": score, "box": xywh_to_xyxy(item["bbox"])})
    class_reports, selected_thresholds = {}, {}
    all_tp = all_fp = all_fn = 0
    for category in categories:
        category_id, name = category["id"], category["name"]
        by_image = {image_id: nms(items, nms_iou) for image_id, items in predicted[category_id].items()}
        grid = [{"threshold": float(threshold), **counts_at_threshold(by_image, truth[category_id], threshold, match_iou)} for threshold in thresholds]
        # Round only the comparison key to avoid binary float noise in exact ties.
        best = max(grid, key=lambda row: (round(row["f1"], 12), round(row["precision"], 12), row["threshold"]))
        gt_count = sum(map(len, truth[category_id].values()))
        class_reports[name] = {
            "category_id": category_id, **best,
            "ground_truth_boxes": gt_count,
            "raw_predictions": sum(map(len, predicted[category_id].values())),
            "predictions_after_nms": sum(map(len, by_image.values())),
            "support_status": "validation_supported" if gt_count else "no_positive_validation_examples",
            "threshold_scan": grid,
        }
        selected_thresholds[name] = best["threshold"]
        all_tp += best["tp"]
        all_fp += best["fp"]
        all_fn += best["fn"]
    return {
        "fit_type": "validation_threshold_fitting",
        "independent_test": False,
        "caveat": "Пороги подобраны по validation. Метрики посчитаны на тех же данных и не являются результатом независимого теста; требуется проверка на новых видео и ракурсах.",
        "method": {"matching": "greedy_descending_score_per_image_per_class", "match_iou_gte": match_iou,
                   "same_class_nms_iou_gte": nms_iou, "score_comparison": ">=",
                   "threshold_grid": thresholds, "selection": ["maximum_f1", "maximum_precision", "higher_threshold"],
                   "zero_denominator_metric_value": 0.0,
                   "additional_runtime_rules_applied": False},
        "thresholds": selected_thresholds,
        "classes": class_reports,
        "combined_validation_fit_metrics": metrics_from_counts(all_tp, all_fp, all_fn),
        "validation_images": len(image_ids),
    }


def calibrate(checkpoint, annotations_path, predictions_path, output):
    checkpoint, annotations_path, predictions_path, output = map(Path, [checkpoint, annotations_path, predictions_path, output])
    for path in [checkpoint, annotations_path, predictions_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    annotations = json.loads(annotations_path.read_text(encoding="utf-8-sig"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8-sig"))
    result = fit_thresholds(annotations, predictions)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["inputs"] = {
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "validation_annotations": {"path": str(annotations_path.resolve()), "sha256": sha256_file(annotations_path)},
        "validation_predictions": {"path": str(predictions_path.resolve()), "sha256": sha256_file(predictions_path)},
    }
    result["checkpoint_sha256"] = result["inputs"]["checkpoint"]["sha256"]
    result["validation_annotation_sha256"] = result["inputs"]["validation_annotations"]["sha256"]
    result["validation_predictions_sha256"] = result["inputs"]["validation_predictions"]["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output.resolve()), "fit_type": result["fit_type"], "independent_test": False,
                      "thresholds": result["thresholds"], "combined_validation_fit_metrics": result["combined_validation_fit_metrics"]}, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "training/rtdetr_custom/best.pth"))
    parser.add_argument("--annotations", default=str(ROOT / "training/dataset/val.json"))
    parser.add_argument("--predictions", default=str(ROOT / "training/rtdetr_custom/best_val_predictions.json"))
    parser.add_argument("--output", default=str(ROOT / "training/rtdetr_custom/calibration.json"))
    args = parser.parse_args()
    calibrate(args.checkpoint, args.annotations, args.predictions, args.output)


if __name__ == "__main__":
    main()
