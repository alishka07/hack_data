"""Offline dense cash/card observation experiment; never a violation detector.

This CLI writes only its separate experiment directory. Imports of this module
do not load Torch, Grounding DINO, download weights or start inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / 'GroundingDINO-local'
WORK_ROOT = ROOT / 'work/interaction_refinement'
VERSION = '1.0'


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_number(x) for x in value):
        raise ValueError('A box must contain four finite numbers')
    if value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError('A box must have positive width and height')
    return list(map(float, value))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def select_samples(timestamps, start, end, step_frames=3, coarse_step_frames=15):
    """Select [start, end] inclusively by original frame index, not inferred time.

    The first actual source frame in the requested window is the anchor for both
    grids. Coarse samples must be a subset of dense samples to isolate frequency.
    """
    if not _number(start) or not _number(end) or start < 0 or end < start:
        raise ValueError('Require finite 0 <= start <= end')
    if type(step_frames) is not int or step_frames < 1 or type(coarse_step_frames) is not int or coarse_step_frames < step_frames or coarse_step_frames % step_frames:
        raise ValueError('Coarse step must be a positive multiple of dense step')
    # Some valid video containers have a negative preroll PTS. Preserve it in
    # the source timeline; nonnegative requested windows naturally exclude it.
    if any(not _number(t) for t in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Source timestamps must be finite and strictly increasing')
    in_window = [index for index, value in enumerate(timestamps) if start - 1e-9 <= value <= end + 1e-9]
    if not in_window:
        raise ValueError('No source frames fall in the requested time window')
    anchor = in_window[0]
    return [{'source_frame': index, 'timestamp_sec': float(timestamps[index]), 'timestamp_source': 'video_pts',
             'coarse_sample': (index - anchor) % coarse_step_frames == 0}
            for index in in_window if (index - anchor) % step_frames == 0]


def resolve_crop(request, width, height):
    """Return integer pixel-edge crop coordinates; explicit crops are not clipped."""
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('Invalid source dimensions')
    if request == 'full':
        return [0, 0, width, height]
    if isinstance(request, str):
        try:
            request = [float(value.strip()) for value in request.split(',')]
        except ValueError:
            raise ValueError('Crop must be auto, full or x1,y1,x2,y2') from None
    box = _box(request)
    if any(value != int(value) for value in box) or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError('Explicit crop must use integer pixel edges inside the original frame')
    return list(map(int, box))


def crop_box_to_source(box, crop, width, height):
    """Map detector pixel coordinates from a crop back to original pixel edges."""
    crop = resolve_crop(crop, width, height)
    raw = _box(box)
    crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
    clipped = [max(0., min(crop_width, raw[0])), max(0., min(crop_height, raw[1])),
               max(0., min(crop_width, raw[2])), max(0., min(crop_height, raw[3]))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError('Detection does not intersect crop')
    return [clipped[0] + crop[0], clipped[1] + crop[1], clipped[2] + crop[0], clipped[3] + crop[1]]


def _iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def choose_auto_crop(anchor_candidates, width, height):
    """Conservative table ROI proposal. Ambiguity falls back to a full frame."""
    full = [0, 0, width, height]
    provenance = {'method': 'three_anchor_table_candidates', 'anchor_candidates': anchor_candidates,
                  'status': 'fallback_full_frame', 'stability_iou': .5, 'context_margin_fraction': .2,
                  'note': 'A technical ROI proposal, not a confirmed table boundary or a bank rule.'}
    if len(anchor_candidates) != 3 or any(len(items) != 1 for items in anchor_candidates):
        return full, {**provenance, 'reason': 'Each of three anchors must have exactly one table candidate'}
    boxes = [_box(items[0]['box']) for items in anchor_candidates]
    if any(_iou(a, b) < .5 for i, a in enumerate(boxes) for b in boxes[i+1:]):
        return full, {**provenance, 'reason': 'Table candidates are not spatially stable'}
    union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    dx, dy = .2 * (union[2] - union[0]), .2 * (union[3] - union[1])
    crop = [max(0, math.floor(union[0] - dx)), max(0, math.floor(union[1] - dy)),
            min(width, math.ceil(union[2] + dx)), min(height, math.ceil(union[3] + dy))]
    return crop, {**provenance, 'status': 'model_proposed', 'reason': 'Stable table projection enlarged for surrounding motion'}


def detect_sample(image, sample, crop, detector, prompts, threshold):
    """Run an injected detector, preserving empty observations and crop provenance."""
    from candidate_filters import filter_candidates
    crop = resolve_crop(crop, image.width, image.height)
    cropped = image.crop(crop)
    raw, rejected = [], []
    started = time.perf_counter()
    for group, prompt in prompts.items():
        for detection in detector(cropped, prompt, threshold):
            try:
                raw_box = _box(detection['box'])
                box = crop_box_to_source(raw_box, crop, image.width, image.height)
                score = detection.get('score')
                if not _number(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid model score')
                raw.append({**detection, 'box': box, 'query': group, 'model_box_crop': raw_box,
                            'crop_xyxy': crop, 'coordinate_space': 'original_frame_pixels'})
            except (ValueError, KeyError, TypeError) as error:
                rejected.append({'query': group, 'reason': str(error)})
    filtered = filter_candidates(raw, image_size=(image.width, image.height))
    return {**sample, 'source_dimensions': [image.width, image.height], 'crop_xyxy': crop,
            'model_image_dimensions': list(cropped.size), 'coordinate_space': 'original_frame_pixels',
            'raw_detections': raw, **filtered, 'invalid_detections': rejected,
            'inference_seconds': round(time.perf_counter() - started, 6),
            'action': 'unknown', 'interpretation': 'unverified_object_observations'}


def compare_sampling(records):
    """Observation diagnostics only; no false TP/FP or ground-truth assumptions."""
    kinds = {'cash', 'identity_card', 'card', 'document'}
    def summarize(rows):
        counts = Counter()
        frames_with_any = 0
        classes_by_frame = []
        for row in rows:
            observed = [item['label'] for item in row.get('candidates', []) if item.get('label') in kinds]
            reviewed = [item['label'] for item in row.get('review', []) if item.get('label') in kinds]
            counts.update(observed)
            frames_with_any += bool(observed or reviewed)
            classes_by_frame.append({'source_frame': row['source_frame'], 'timestamp_sec': row['timestamp_sec'],
                                     'candidate_classes': sorted(set(observed)), 'review_classes': sorted(set(reviewed))})
        gaps = [b['timestamp_sec'] - a['timestamp_sec'] for a, b in zip(rows, rows[1:])]
        return {'samples': len(rows), 'frames_with_target_candidates_or_review': frames_with_any,
                'candidate_observations_by_class': dict(counts), 'maximum_sample_gap_sec': max(gaps) if gaps else None,
                'classes_by_frame': classes_by_frame}
    coarse = [row for row in records if row['coarse_sample']]
    additional = [row for row in records if not row['coarse_sample']]
    return {'comparison_kind': 'same_inferences_same_crop_frequency_subset', 'dense': summarize(records),
            'coarse': summarize(coarse), 'additional_between_coarse_frames': summarize(additional),
            'quality': 'not_measured', 'action': 'unknown',
            'limitations': ['Repeated detector boxes are not counts of real objects.',
                            'More detections do not establish better precision or recall.',
                            'Coarse is an exact subset of the same crop/model/prompt results; this isolates sampling, not cropping.',
                            'The selected window and timing are known development examples, not an independent test.',
                            'Manual reference labels are required to assess visibility, cash/card confusion and transfer recall.']}


def sequence_diagnostics(records, plan, step_frames):
    """Reuse the conservative spatial linker; no person/action assertions."""
    from interaction_events import analyze_interactions
    normalized = [{'time': row['timestamp_sec'], 'frame': row['source_frame'], 'mode': 'refinement',
                   'playback_sec': row['timestamp_sec'], 'objects': row['candidates']} for row in records]
    modes = {'refinement': {'records': normalized, 'times': [row['time'] for row in normalized],
                            'interval': step_frames / plan['fps']}}
    job = {'video': {'fps': plan['fps'], 'frames': plan['source_frames'], 'width': plan['width'], 'height': plan['height']}}
    analyzed = analyze_interactions(modes, {}, {}, job)
    chains = analyzed['sequences']
    return {'sequence_count': len(chains), 'sequences_with_two_or_more_observations': sum(len(s['observation_ids']) >= 2 for s in chains),
            'maximum_sequence_observations': max([len(s['observation_ids']) for s in chains], default=0),
            'maximum_sequence_span_sec': max([s['end_sec'] - s['start_sec'] for s in chains], default=0),
            'interpretation': 'unverified_same_class_box_overlap_sequences', 'action': 'unknown',
            'limitations': ['A repeated false detection also forms a long spatial chain.',
                            'No hand, possession, transfer or counting model is used.']}


def inspect_video(video, start, end, step_frames, coarse_step_frames):
    import cv2
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError('Cannot open source video')
    timestamps = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not _number(fps) or fps <= 0 or min(width, height) <= 0 or expected <= 0:
            raise ValueError('Source video metadata is incomplete')
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[:2] != (height, width):
                raise ValueError('Source dimensions change during decoding')
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not _number(timestamp) or timestamps and timestamp <= timestamps[-1]:
                raise ValueError('Video PTS missing or nonmonotonic; no inferred action timing will be substituted')
            timestamps.append(timestamp)
    finally:
        capture.release()
    if len(timestamps) != expected:
        raise ValueError(f'Incomplete source decode: {len(timestamps)} frames, expected {expected}')
    duration = timestamps[-1] + 1 / fps
    if end > duration + 1e-6:
        raise ValueError(f'Requested end {end} exceeds source duration {duration}')
    samples = select_samples(timestamps, start, end, step_frames, coarse_step_frames)
    return {'width': width, 'height': height, 'fps': fps, 'source_frames': len(timestamps),
            'metadata_frames': expected, 'frame_count_matches': True, 'duration_sec': duration,
            'source_first_pts_sec': timestamps[0], 'negative_preroll_frames': sum(t < 0 for t in timestamps),
            'start_sec': start, 'end_sec': end, 'step_frames': step_frames, 'coarse_step_frames': coarse_step_frames,
            'samples': samples}


def iter_selected_frames(video, samples):
    import cv2
    from PIL import Image
    selected = {row['source_frame']: row for row in samples}
    last = max(selected)
    capture = cv2.VideoCapture(str(video))
    delivered = 0
    try:
        for index in range(last + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Second decode stopped at frame {index}')
            if index not in selected:
                continue
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not _number(timestamp) or abs(timestamp - selected[index]['timestamp_sec']) > 1e-6:
                raise ValueError('Source PTS changed between decoding passes')
            delivered += 1
            yield selected[index], Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if delivered != len(samples):
        raise ValueError('Not all planned frames were decoded')


def _preview(image, record, destination):
    from PIL import ImageDraw
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(record['crop_xyxy'], outline='yellow', width=2)
    draw.text((5, 5), f"frame {record['source_frame']} / {record['timestamp_sec']:.3f}s / action unknown", fill='white', stroke_width=1, stroke_fill='black')
    for status, color in [('candidates', 'lime'), ('review', 'orange')]:
        for row in record[status]:
            draw.rectangle(row['box'], outline=color, width=1)
            draw.text((row['box'][0], max(0, row['box'][1] - 11)), f"{row['label']} {row['score']:.2f}", fill=color, stroke_width=1, stroke_fill='black')
    canvas.save(destination, quality=94)


def run(args):
    video = args.video.resolve()
    if not video.is_file():
        raise ValueError('Source video not found')
    output = args.output.resolve()
    if not output.is_relative_to(WORK_ROOT.resolve()) or output == WORK_ROOT.resolve():
        raise ValueError(f'Experiment output must be a child directory of {WORK_ROOT}')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is nonempty; choose a new experiment name')
    if not _number(args.threshold) or not 0 < args.threshold < 1:
        raise ValueError('Threshold must be between 0 and 1')
    source_hash = sha256(video)
    plan = inspect_video(video, args.start, args.end, args.step_frames, args.coarse_step_frames)
    output.mkdir(parents=True, exist_ok=True)
    from run import PROMPTS
    prompts = {key: PROMPTS[key] for key in ('desk', 'cash')}
    plan.update(video=str(video), source_sha256=source_hash, crop_request=args.crop, threshold=args.threshold,
                prompts=prompts, model_root=str(MODEL_ROOT), version=VERSION,
                script_sha256=sha256(Path(__file__)), action='unknown')
    write_json(output / 'plan.json', plan)
    if args.plan_only:
        write_json(output / 'summary.json', {'status': 'planned_only', 'model_loaded': False,
                    'planned_samples': len(plan['samples']), 'source_sha256': source_hash, 'action': 'unknown'})
        return {'status': 'planned_only', 'samples': len(plan['samples']), 'output': str(output)}
    records = []
    started = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(2)
        sys.path.insert(0, str(MODEL_ROOT))
        from detect_local import load_model, detect
        from candidate_filters import filter_candidates
        loaded = load_model()
        detector = lambda image, prompt, threshold: detect(image, loaded, prompt, threshold)
        anchors = []
        if args.crop == 'auto':
            chosen = [plan['samples'][0], plan['samples'][len(plan['samples']) // 2], plan['samples'][-1]]
            if len({row['source_frame'] for row in chosen}) != 3:
                crop, crop_provenance = choose_auto_crop([], plan['width'], plan['height'])
            else:
                for sample, image in iter_selected_frames(video, chosen):
                    raw = detector(image, PROMPTS['workspace'], args.threshold)
                    filtered = filter_candidates(raw, (image.width, image.height))
                    tables = [row for row in filtered['candidates'] if row['label'] == 'work_table']
                    anchors.append({'source_frame': sample['source_frame'], 'timestamp_sec': sample['timestamp_sec'],
                                    'raw_detections': raw, 'tables': tables})
                crop, crop_provenance = choose_auto_crop([row['tables'] for row in anchors], plan['width'], plan['height'])
        else:
            crop = resolve_crop(args.crop, plan['width'], plan['height'])
            crop_provenance = {'method': 'explicit' if args.crop != 'full' else 'full_frame', 'status': 'requested', 'reason': 'CLI setting'}
        write_json(output / 'crop.json', {'crop_xyxy': crop, 'provenance': crop_provenance, 'anchors': anchors})
        pictures = output / 'frames'
        pictures.mkdir()
        with (output / 'frames.jsonl').open('w', encoding='utf-8') as stream:
            for sample, image in iter_selected_frames(video, plan['samples']):
                record = detect_sample(image, sample, crop, detector, prompts, args.threshold)
                record['preview_file'] = f"frames/frame_{sample['source_frame']:06}.jpg"
                _preview(image, record, output / record['preview_file'])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                records.append(record)
                print(f"sample {len(records)}/{len(plan['samples'])}; frame={sample['source_frame']}; t={sample['timestamp_sec']:.3f}; action=unknown", flush=True)
        if len(records) != len(plan['samples']) or sha256(video) != source_hash:
            raise ValueError('Planned frame count or source SHA-256 changed')
        comparison = compare_sampling(records)
        comparison['spatial_chains'] = {
            'dense': sequence_diagnostics(records, plan, args.step_frames),
            'coarse': sequence_diagnostics([row for row in records if row['coarse_sample']], plan, args.coarse_step_frames)}
        write_json(output / 'sampling_comparison.json', comparison)
        summary = {'version': VERSION, 'status': 'completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'source_sha256': source_hash, 'video': str(video), 'device': str(loaded[2]),
                   'plan_sha256': sha256(output / 'plan.json'), 'jsonl_sha256': sha256(output / 'frames.jsonl'),
                   'jsonl_rows': len(records), 'planned_samples': len(plan['samples']), 'planned_frames_match': True,
                   'source_frame_count_matches': plan['frame_count_matches'], 'crop_xyxy': crop, 'crop_provenance': crop_provenance,
                   'first_timestamp_sec': records[0]['timestamp_sec'], 'last_timestamp_sec': records[-1]['timestamp_sec'],
                   'maximum_sample_gap_sec': comparison['dense']['maximum_sample_gap_sec'],
                   'processing_seconds': round(time.perf_counter() - started, 3),
                   'action': 'unknown', 'violation_events': [], 'quality': 'not_measured',
                   'limitations': comparison['limitations'] + ['Only this requested window was analyzed; no full-video action coverage.',
                      'The 16x11-pixel card seen in earlier X18 probes may remain ambiguous; scaling cannot restore missing visual detail.']}
        write_json(output / 'summary.json', summary)
        return {'status': 'completed', 'samples': len(records), 'output': str(output), 'seconds': summary['processing_seconds']}
    except Exception as error:
        write_json(output / 'summary.json', {'version': VERSION, 'status': 'failed', 'error': str(error),
                    'source_sha256': source_hash, 'completed_samples': len(records), 'planned_samples': len(plan['samples']),
                    'action': 'unknown', 'quality': 'not_measured', 'violation_events': []})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--step-frames', type=int, default=3)
    parser.add_argument('--coarse-step-frames', type=int, default=15)
    parser.add_argument('--crop', default='auto', help='auto, full or integer original-pixel x1,y1,x2,y2')
    parser.add_argument('--threshold', type=float, default=.28)
    parser.add_argument('--output', type=Path, default=WORK_ROOT / 'experiment')
    parser.add_argument('--plan-only', action='store_true', help='CPU decode and sample plan only; never load a model')
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == '__main__':
    main()
