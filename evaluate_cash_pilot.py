"""Evaluate a cash-only ROI pilot against local Grounding DINO, without training.

All validation inference and threshold fitting finish before test inference.
The small assistant-reviewed, same-camera holdout is not independent ground truth.
No reviewed-object templates, human reference corrections, or training labels
are used by prediction. Original images, source annotations and weights are read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

from calibrate_threshold import (counts_at_threshold, fit_thresholds, nms,
                                 sha256_file, xywh_to_xyxy)
from candidate_filters import canonical_label

ROOT = Path(__file__).resolve().parent
DINO_ROOT = ROOT.parent / 'GroundingDINO-local'
VERSION = '1.1'
# Newly initialized cash heads can produce useful rankings below .05. Keep the
# same low floor for all models; operating thresholds are still fitted on val.
RAW_FLOOR = .001
THRESHOLD_GRID = [step / 1000 for step in range(1, 100)] + [step / 100 for step in range(10, 91)]
NMS_IOU = .6
MATCH_IOU = .5
CAVEAT = (
    'Малый отложенный набор из той же камеры с предварительной разметкой ассистента. '
    'Разделение выполнено по исходным видео и их SHA-256, но это не независимая '
    'проверенная человеком эталонная разметка. Метрики относятся к рамкам банкнот '
    'в заданной области стола, а не к нарушениям и не к другим камерам. '
    'Пороги выбраны только на validation и зафиксированы до запуска test.'
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def inside_image_path(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Image outside dataset: {filename}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def source_groups(data):
    """Only image provenance is consumed; training boxes never reach inference."""
    images = data['images']
    if not images:
        raise ValueError('A split has no images')
    result = {key: set() for key in ('source_sha256', 'source_video', 'source_frame_path')}
    for item in images:
        for key in result:
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Missing {key}: {item.get("file_name")}')
            value = value.strip()
            if key == 'source_sha256':
                value = value.lower()
                if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Invalid source_sha256')
            elif key == 'source_frame_path':
                value = str(Path(value).resolve()).casefold()
            else:
                value = value.casefold().replace('х', 'x')
            result[key].add(value)
    return result


def check_split_leakage(splits):
    groups = {name: source_groups(data) for name, data in splits.items()}
    names = list(groups)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            for field in groups[first]:
                shared = groups[first][field] & groups[second][field]
                if shared:
                    raise ValueError(f'{first}/{second} share {field}: {sorted(shared)}')
    return {name: {key: sorted(values) for key, values in fields.items()}
            for name, fields in groups.items()}


def validate_dataset(data, root):
    """Validate ROI geometry and cash-only evaluation labels before model loads."""
    from PIL import Image
    categories = data.get('categories', [])
    if len(categories) != 1 or categories[0]['name'] != 'cash':
        raise ValueError('This pilot evaluator requires exactly one category named cash')
    image_map = {item['id']: item for item in data['images']}
    if len(image_map) != len(data['images']) or not image_map:
        raise ValueError('Image IDs must be unique and nonempty')
    cid = categories[0]['id']
    for item in data['images']:
        if item.get('complete_for_cash_only') is not True:
            raise ValueError('Every evaluation ROI must declare complete_for_cash_only=true; missing boxes are not known negatives')
        path = inside_image_path(root, item['file_name'])
        source = Path(item['source_frame_path'])
        if not source.is_absolute() or not source.is_file():
            raise ValueError(f'Original full frame missing: {source}')
        roi = item.get('roi_xyxy', [])
        if len(roi) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or int(x) != x for x in roi):
            raise ValueError('ROI must contain four integer pixel coordinates')
        x1, y1, x2, y2 = roi
        with Image.open(source) as full:
            if not (0 <= x1 < x2 <= full.width and 0 <= y1 < y2 <= full.height):
                raise ValueError('ROI is outside original frame')
        with Image.open(path) as crop:
            if crop.size != (item['width'], item['height']) or crop.size != (x2 - x1, y2 - y1):
                raise ValueError('Saved ROI dimensions differ from annotation/source ROI')
    ids = set()
    for ann in data['annotations']:
        if ann['id'] in ids or ann['image_id'] not in image_map or ann['category_id'] != cid or ann.get('iscrowd', 0):
            raise ValueError('Invalid annotation id, image, category or crowd flag')
        ids.add(ann['id'])
        x1, y1, x2, y2 = xywh_to_xyxy(ann['bbox'])
        item = image_map[ann['image_id']]
        if not (0 <= x1 < x2 <= item['width'] + 1e-5 and 0 <= y1 < y2 <= item['height'] + 1e-5):
            raise ValueError('Annotation outside ROI')
    return data


def prediction_images(data):
    """Explicit allowlist: no ground-truth labels/boxes are passed to predictors."""
    fields = ('id', 'file_name', 'width', 'height', 'source_frame_path', 'roi_xyxy',
              'source_sha256', 'source_video')
    return [{key: item[key] for key in fields} for item in data['images']]


def cash_predictions(rows, item, category_id, mode):
    """Map raw predictions into ROI coordinates; full-frame centers must be inside."""
    if mode not in ('rtdetr_roi', 'gdino_roi', 'gdino_fullframe'):
        raise ValueError(mode)
    if mode == 'gdino_fullframe':
        left, top, right, bottom = map(float, item['roi_xyxy'])
    else:
        left, top, right, bottom = 0., 0., float(item['width']), float(item['height'])
    result = []
    for row in rows:
        if canonical_label(row.get('label')) != 'cash':
            continue
        score = float(row['score'])
        box = list(map(float, row['box']))
        if len(box) != 4 or not all(math.isfinite(x) for x in [*box, score]) or not 0 <= score <= 1:
            raise ValueError('Non-finite or malformed prediction')
        x1, y1, x2, y2 = box
        if score < RAW_FLOOR or x2 <= x1 or y2 <= y1:
            continue
        if not (left <= (x1 + x2) / 2 < right and top <= (y1 + y2) / 2 < bottom):
            continue
        x1, x2 = max(x1, left), min(x2, right)
        y1, y2 = max(y1, top), min(y2, bottom)
        if x2 > x1 and y2 > y1:
            result.append({'image_id': item['id'], 'category_id': category_id,
                           'bbox': [x1 - left, y1 - top, x2 - x1, y2 - y1], 'score': score})
    return result


def validate_predictions(data, predictions):
    images = {item['id']: item for item in data['images']}
    cid = data['categories'][0]['id']
    for pred in predictions:
        if pred['image_id'] not in images or pred['category_id'] != cid:
            raise ValueError('Prediction image/category does not belong to split')
        box = xywh_to_xyxy(pred['bbox'])
        item = images[pred['image_id']]
        score = float(pred['score'])
        if not math.isfinite(score) or not RAW_FLOOR <= score <= 1:
            raise ValueError('Prediction score outside saved raw score range')
        if not (0 <= box[0] < box[2] <= item['width'] + 1e-5 and 0 <= box[1] < box[3] <= item['height'] + 1e-5):
            raise ValueError('Prediction outside ROI')


def fixed_metrics(data, predictions, threshold):
    """Use an already locked threshold. This function never fits or scans it."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid fixed threshold')
    validate_predictions(data, predictions)
    truth, grouped = defaultdict(list), defaultdict(list)
    for ann in data['annotations']:
        truth[ann['image_id']].append(xywh_to_xyxy(ann['bbox']))
    for pred in predictions:
        grouped[pred['image_id']].append({'score': pred['score'], 'box': xywh_to_xyxy(pred['bbox'])})
    kept = {image_id: nms(rows, NMS_IOU) for image_id, rows in grouped.items()}
    negatives = {item['id'] for item in data['images'] if not truth[item['id']]}
    negative_fps = {str(image_id): sum(row['score'] >= threshold for row in kept.get(image_id, []))
                    for image_id in sorted(negatives)}
    return {'threshold': threshold, **counts_at_threshold(kept, truth, threshold, MATCH_IOU),
            'ground_truth_boxes': len(data['annotations']), 'images': len(data['images']),
            'annotated_negative_images': len(negatives),
            'negative_images_with_false_positive': sum(count > 0 for count in negative_fps.values()),
            'false_positives_on_negative_images': sum(negative_fps.values()),
            'negative_image_false_positives': negative_fps,
            'negative_definition': 'No cash annotation in this reviewed ROI; label quality remains provisional'}


def raw_ap(data, predictions):
    """Standard COCO AP before NMS/operating threshold, with documented .001 floor."""
    validate_predictions(data, predictions)
    base = {'prediction_floor': RAW_FLOOR, 'nms_applied': False,
            'max_detections_per_image': 100, 'raw_predictions': len(predictions)}
    if not data['annotations']:
        return {**base, 'map_50': None, 'map_50_95': None, 'reason': 'No positive reference boxes'}
    if not predictions:
        return {**base, 'map_50': 0., 'map_50_95': 0.}
    from faster_coco_eval import COCO, COCOeval_faster
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO({**data, 'info': data.get('info', {})})
        detection = truth.loadRes(predictions)
        evaluator = COCOeval_faster(truth, detection, 'bbox')
        evaluator.params.catIds = [data['categories'][0]['id']]
        evaluator.params.imgIds = [item['id'] for item in data['images']]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {**base, 'map_50': float(evaluator.stats[1]), 'map_50_95': float(evaluator.stats[0])}


def fit_validation(data, predictions):
    validate_predictions(data, predictions)
    result = fit_thresholds(data, predictions, thresholds=THRESHOLD_GRID,
                            match_iou=MATCH_IOU, nms_iou=NMS_IOU)
    result['fit_split'] = 'val'
    result['test_labels_used'] = False
    return result


def cached_inference(images, category_id, root, mode, identity, output, predictor=None):
    """Atomically checkpoint each image; reuse only matching inputs and payload hash."""
    output = Path(output)
    records = []
    for item in images:
        records.append({**item, 'roi_file_sha256': sha256_file(inside_image_path(root, item['file_name'])),
                        'source_frame_file_sha256': sha256_file(item['source_frame_path'])})
    signature = {'version': VERSION, 'mode': mode, 'raw_score_floor': RAW_FLOOR,
                 'category_id': category_id, 'identity': identity, 'inputs': records}
    existing = {}
    if output.is_file():
        cache = read_json(output)
        if cache.get('signature') != signature:
            raise ValueError(f'Prediction cache inputs changed; choose a fresh output directory: {output}')
        if cache.get('records_sha256') != fingerprint(cache.get('records')):
            raise ValueError(f'Prediction cache payload hash mismatch: {output}')
        existing = {row['image_id']: row for row in cache['records']}
        if len(existing) != len(cache['records']) or not set(existing).issubset({x['id'] for x in images}):
            raise ValueError('Cache contains duplicate/unknown image IDs')
    missing = [item for item in images if item['id'] not in existing]
    if missing and predictor is None:
        return None
    if missing:
        from PIL import Image
        for item in missing:
            filename = Path(item['source_frame_path']) if mode == 'gdino_fullframe' else inside_image_path(root, item['file_name'])
            started = time.perf_counter()
            with Image.open(filename) as image:
                rows = predictor(image.convert('RGB'))
            converted = cash_predictions(rows, item, category_id, mode)
            existing[item['id']] = {'image_id': item['id'], 'predictions': converted,
                                    'inference_seconds': time.perf_counter() - started}
            rows = [existing[image['id']] for image in images if image['id'] in existing]
            write_json(output, {'signature': signature, 'records': rows, 'records_sha256': fingerprint(rows),
                                'complete': len(rows) == len(images)})
            print(f'{mode}: {len(rows)}/{len(images)} images ({output.stem})', flush=True)
    return [pred for item in images for pred in existing[item['id']]['predictions']]


def release_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_split(data, root, split, identities, output, weights, device=None, dino_root=DINO_ROOT):
    """Load only one model at a time; callbacks receive pixels, never annotations."""
    images = prediction_images(data)
    cid = data['categories'][0]['id']
    result, pending = {}, []
    for mode, identity in identities.items():
        rows = cached_inference(images, cid, root, mode, identity, output / f'{mode}_{split}_predictions.json')
        if rows is None:
            pending.append(mode)
        else:
            result[mode] = rows
    if 'rtdetr_roi' in pending:
        from train_objects import load_trained, predict_image
        loaded = load_trained(weights, device)
        if [(c['id'], c['name']) for c in loaded[2]] != [(c['id'], c['name']) for c in data['categories']]:
            raise ValueError('Checkpoint category metadata differs from cash dataset')
        result['rtdetr_roi'] = cached_inference(images, cid, root, 'rtdetr_roi', identities['rtdetr_roi'],
            output / f'rtdetr_roi_{split}_predictions.json', lambda image: predict_image(image, loaded, RAW_FLOOR))
        del loaded
        release_gpu()
    dino_modes = [mode for mode in pending if mode.startswith('gdino_')]
    if dino_modes:
        if str(dino_root) not in sys.path:
            sys.path.insert(0, str(dino_root))
        from detect_local import load_model, detect
        from run import PROMPTS
        loaded = load_model()
        if device is not None and str(loaded[2]) != str(device):
            processor, model, _ = loaded
            model.to(device)
            loaded = processor, model, device
            del processor, model
        for mode in dino_modes:
            result[mode] = cached_inference(images, cid, root, mode, identities[mode],
                output / f'{mode}_{split}_predictions.json', lambda image: detect(image, loaded, PROMPTS['cash'], RAW_FLOOR))
        del loaded
        release_gpu()
    for predictions in result.values():
        validate_predictions(data, predictions)
    return result


def build_identities(weights, dino_root, skip_baseline):
    common = {name: sha256_file(ROOT / name) for name in
              ('evaluate_cash_pilot.py', 'calibrate_threshold.py', 'candidate_filters.py')}
    identities = {'rtdetr_roi': {'checkpoint': str(Path(weights).resolve()),
                 'checkpoint_sha256': sha256_file(weights), 'code': {**common,
                 'train_objects.py': sha256_file(ROOT / 'train_objects.py')}}}
    if not skip_baseline:
        from run import PROMPTS
        files = [path for path in (dino_root / 'weights').iterdir() if path.is_file() and path.suffix in ('.json', '.txt', '.bin', '.safetensors')]
        if not any(path.suffix in ('.bin', '.safetensors') for path in files):
            raise FileNotFoundError('Local Grounding DINO weights missing')
        identity = {'weights': {path.name: sha256_file(path) for path in sorted(files)},
                    'code': {**common, 'detect_local.py': sha256_file(dino_root / 'detect_local.py'),
                             'run.py': sha256_file(ROOT / 'run.py')},
                    'prompt': PROMPTS['cash'], 'text_threshold': .25,
                    'human_reference_corrections': False, 'candidate_filters': 'canonical_label only'}
        identities.update(gdino_fullframe=identity, gdino_roi=identity)
    return identities


def run(args):
    root, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    weights, dino_root = Path(args.weights).resolve(), Path(args.dino_root).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Evaluation outputs must stay outside the dataset')
    val = validate_dataset(read_json(root / 'val.json'), root)
    test = validate_dataset(read_json(root / 'test.json'), root)
    if val['categories'] != test['categories']:
        raise ValueError('Validation and test category metadata differ')
    # Training image provenance is checked for leakage; training boxes are unused.
    training_provenance = {'images': read_json(root / 'train.json')['images']}
    separation = check_split_leakage({'train': training_provenance, 'val': val, 'test': test})
    del training_provenance
    identities = build_identities(weights, dino_root, args.skip_baseline)
    signature = {'version': VERSION, 'models': identities, 'split_sha256': {
                    name: sha256_file(root / f'{name}.json') for name in ('train', 'val', 'test')},
                 'device_override': args.device, 'prediction_floor': RAW_FLOOR,
                 'grid': THRESHOLD_GRID, 'match_iou': MATCH_IOU, 'nms_iou': NMS_IOU}
    identity_path = output / 'run_identity.json'
    if identity_path.is_file() and read_json(identity_path) != signature:
        raise ValueError('Evaluation inputs differ from existing run; choose a fresh --output')
    write_json(identity_path, signature)
    import torch
    torch.set_num_threads(2)
    started = time.perf_counter()
    print('Validation inference; no test predictions or threshold fitting on test.', flush=True)
    val_predictions = infer_split(val, root, 'val', identities, output, weights, args.device, dino_root)
    fits = {mode: fit_validation(val, rows) for mode, rows in val_predictions.items()}
    locked = {'fit_split': 'val', 'test_labels_used': False,
              'validation_annotation_sha256': signature['split_sha256']['val'],
              'validation_predictions_sha256': {mode: fingerprint(rows) for mode, rows in val_predictions.items()},
              'models': {mode: {'threshold': fit['thresholds']['cash'], 'fit': fit} for mode, fit in fits.items()}}
    lock_path = output / 'locked_thresholds.json'
    if lock_path.is_file() and read_json(lock_path) != locked:
        raise ValueError('Locked thresholds differ; refusing to overwrite test operating points')
    write_json(lock_path, locked)
    lock_sha = sha256_file(lock_path)
    calibration = {
        'checkpoint_sha256': identities['rtdetr_roi']['checkpoint_sha256'],
        'thresholds': {'cash': locked['models']['rtdetr_roi']['threshold']},
        'fit_split': 'val', 'test_labels_used': False,
        'validation_annotation_sha256': signature['split_sha256']['val'],
        'validation_predictions_sha256': locked['validation_predictions_sha256']['rtdetr_roi'],
        'threshold_lock_sha256': lock_sha,
        'raw_prediction_floor': RAW_FLOOR,
        'independent_ground_truth': False,
        'caveat': CAVEAT,
    }
    calibration_path = output / 'calibration.json'
    if calibration_path.is_file() and read_json(calibration_path) != calibration:
        raise ValueError('Runtime calibration differs from locked validation threshold')
    write_json(calibration_path, calibration)
    print('Validation thresholds locked. Starting test inference.', flush=True)
    test_predictions = infer_split(test, root, 'test', identities, output, weights, args.device, dino_root)
    if sha256_file(lock_path) != lock_sha:
        raise RuntimeError('Threshold lock changed during test inference')
    report = {'status': 'completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'evaluation_type': 'same_camera_assistant_reviewed_video_disjoint_pilot',
              'independent_ground_truth': False, 'banking_violation_evaluation': False,
              'caveat': CAVEAT, 'source_split_groups': separation, 'signature': signature,
              'threshold_lock_sha256': lock_sha, 'thresholds_locked_before_test_inference': True,
              'inference_inputs_include_ground_truth_boxes': False, 'human_reference_corrections': False,
              'models': {}, 'seconds': time.perf_counter() - started}
    lines = ['Пилот распознавания банкнот в области стола', '', CAVEAT, '',
             f'Validation: {len(val["images"])} кадров; test: {len(test["images"])} кадров.',
             f'AP: исходные предсказания с score >= {RAW_FLOOR:g}, до NMS. Рабочие метрики: NMS 0.6, IoU >= 0.5.', '']
    for mode in identities:
        threshold = locked['models'][mode]['threshold']
        row = {'threshold': threshold, 'validation': {'raw_ap': raw_ap(val, val_predictions[mode]),
                 'fixed_threshold': fixed_metrics(val, val_predictions[mode], threshold)},
               'test': {'raw_ap': raw_ap(test, test_predictions[mode]),
                 'fixed_threshold': fixed_metrics(test, test_predictions[mode], threshold)}}
        report['models'][mode] = row
        ap, metrics = row['test']['raw_ap'], row['test']['fixed_threshold']
        lines += [mode, f'  Порог по validation: {threshold:.3f}',
                  f'  Test AP50: {ap["map_50"]}; AP50:95: {ap["map_50_95"]}',
                  f'  TP={metrics["tp"]}, FP={metrics["fp"]}, FN={metrics["fn"]}; precision={metrics["precision"]:.4f}, recall={metrics["recall"]:.4f}',
                  f'  Кадры без размеченных денег: {metrics["negative_images_with_false_positive"]}/{metrics["annotated_negative_images"]} с ложными рамками; всего таких рамок: {metrics["false_positives_on_negative_images"]}.', '']
    write_json(output / 'cash_pilot_evaluation.json', report)
    (output / 'cash_pilot_evaluation.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dino-root', type=Path, default=DINO_ROOT)
    parser.add_argument('--device', default=None)
    parser.add_argument('--skip-baseline', action='store_true', help='Debug candidate only; no comparative claim')
    return run(parser.parse_args())


if __name__ == '__main__':
    main()
