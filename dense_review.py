"""Bounded, automatic close-up review of cash/card candidates, never verdicts.

Only raw model detections and raw tracker output propose where/when to look.
Human labels, Google notes, reviewed templates and event answers are not read.
Importing this module neither imports Torch nor loads/downloads a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

from interaction_refinement import crop_box_to_source, resolve_crop, sha256, write_json

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / 'runtime' / 'GroundingDINO-local'
VERSION = 'dense-roi-1'
PROMPTS = {
    'targets': 'banknotes. an identity card. a paper document.',
    'confounders': 'a chocolate bar. a plastic bag. a paper cup.',
}
LIMITATIONS = [
    'Only selected windows and a model-proposed table region are analyzed densely.',
    'No detection does not establish absence; sampled frames can miss a short action.',
    'Raw tracking may include reflections; motion only proposes review, not a person role.',
    'Object boxes do not establish possession, transfer, counting, authenticity or a violation.',
    'This optional close-up stage has not been validated for precision or recall.',
]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(_finite(v) for v in value) and value[2] > value[0] and value[3] > value[1])


def _iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _inside(box, region):
    x, y = _center(box)
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _raw_frames(path, width, height):
    """Re-filter only original detections; ignore annotations and corrected results."""
    from candidate_filters import filter_candidates
    rows = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            t = row.get('timestamp_sec')
            if not _finite(t) or t < 0:
                continue
            # A reference match must not become a proposal even if a producer
            # accidentally puts it in raw_detections in a future format.
            raw = [d for d in row.get('raw_detections', [])
                   if d.get('source', 'model') == 'model']
            filtered = filter_candidates(raw, image_size=(width, height))
            rows.append({'timestamp_sec': t, 'source_frame': row.get('source_frame'),
                         'candidates': filtered['candidates'], 'review': filtered['review']})
    return sorted(rows, key=lambda row: row['timestamp_sec'])


def _stable_table(rows, width, height):
    """One stable table over the observed timeline; reject competing/moving tables."""
    observations = []
    usable_times = sorted({r['timestamp_sec'] for r in rows})
    for row in rows:
        tables = [d for d in row['candidates'] if d['label'] == 'work_table'
                  and d.get('score', 0) >= .35 and _box(d.get('box'))]
        if len(tables) > 1:
            return None, {'reason': 'multiple_table_candidates', 'anchor_count': 0}
        if len(tables) == 1:
            observations.append((row['timestamp_sec'], tables[0]['box']))
    if len(observations) < 3 or not usable_times or observations[-1][0] - observations[0][0] < 1:
        return None, {'reason': 'fewer_than_three_stable_table_observations', 'anchor_count': len(observations)}
    median = [statistics.median(b[k] for _, b in observations) for k in range(4)]
    if any(_iou(b, median) < .55 for _, b in observations):
        return None, {'reason': 'table_position_changed', 'anchor_count': len(observations)}
    if len(observations) / len(usable_times) < .6:
        return None, {'reason': 'table_insufficiently_visible', 'anchor_count': len(observations)}
    # Include hands above the table; this is a region proposal, not its surface.
    dx = .15 * (median[2] - median[0])
    dy = .25 * (median[3] - median[1])
    crop = [max(0, math.floor(median[0] - dx)), max(0, math.floor(median[1] - dy)),
            min(width, math.ceil(median[2] + dx)), min(height, math.ceil(median[3] + .10*(median[3]-median[1])))]
    if (crop[2]-crop[0])*(crop[3]-crop[1]) > .80*width*height:
        return None, {'reason': 'table_roi_not_a_closeup', 'anchor_count': len(observations)}
    return crop, {'method': 'raw_model_stable_table_median', 'table_box': median,
                  'anchor_count': len(observations), 'observed_times': [t for t, _ in observations],
                  'minimum_iou': .55, 'minimum_visibility_fraction': .6,
                  'surface_or_hand_contact_confirmed': False}


def _complete_stage(job, key):
    return any(s.get('key') == key and s.get('status') == 'completed' for s in job.get('stages', []))


def _select_windows(triggers, duration, budget):
    """Rank proposals, avoid duplicate overlap, then spend the bounded time budget."""
    chosen = []
    spent = 0.
    for trigger in sorted(triggers, key=lambda t: (-t['priority'], t['timestamp_sec'])):
        if spent >= budget - 1e-8:
            break
        t = trigger['timestamp_sec']
        length = min(3., budget - spent, duration)
        if length < .4:
            break
        start = max(0., min(t - length/2, duration - length))
        end = start + length
        if any(min(end, w['end_sec']) > max(start, w['start_sec']) for w in chosen):
            continue
        chosen.append({'start_sec': round(start, 6), 'end_sec': round(end, 6),
                       'trigger': trigger})
        spent += length
    return sorted(chosen, key=lambda w: w['start_sec'])


def plan_dense_windows(jobfolder, job, *, max_seconds=12., sample_fps=5.):
    """Return a deterministic bounded plan using raw job outputs, never labels.

    A not_checked plan is a valid outcome and must not be rendered as compliance.
    The source SHA binds the plan; cached producer input files have their own hashes.
    """
    if not _finite(max_seconds) or not .4 <= max_seconds <= 60 or not _finite(sample_fps) or not 1 <= sample_fps <= 15:
        raise ValueError('Require budget .4..60 seconds and sampling 1..15 FPS')
    folder = Path(jobfolder).resolve()
    source = (folder / job.get('_source_file', 'input.mov')).resolve()
    if not source.is_file() or not source.is_relative_to(folder):
        raise ValueError('Job source must be an existing file inside the job directory')
    metadata = job.get('video', {})
    width, height = metadata.get('width'), metadata.get('height')
    duration = metadata.get('duration', metadata.get('duration_sec'))
    if type(width) is not int or type(height) is not int or min(width, height) <= 0 or not _finite(duration) or duration <= 0:
        raise ValueError('Invalid job video dimensions/duration')
    plan = {'version': VERSION, 'status': 'not_checked', 'source_sha256': sha256(source),
            'source_file': str(source), 'source_dimensions': [width, height], 'duration_sec': duration,
            'sample_fps': sample_fps, 'budget_seconds': max_seconds,
            'maximum_samples': int(math.ceil(max_seconds * sample_fps)), 'windows': [],
            'input_files': {}, 'proposal_inputs': 'raw_model_outputs_only',
            'human_labels_used': False, 'action': 'unknown', 'violations': [],
            'limitations': list(LIMITATIONS)}
    if not _complete_stage(job, 'desk'):
        return {**plan, 'reason': 'desk_stage_not_completed'}
    desk_dir = folder / 'desk' / source.stem
    summary_path, frames_path = desk_dir / 'summary.json', desk_dir / 'frames.jsonl'
    if not summary_path.is_file() or not frames_path.is_file():
        return {**plan, 'reason': 'desk_outputs_missing'}
    summary = _read_json(summary_path)
    if summary.get('status') != 'completed' or summary.get('frame_count_matches') is not True or summary.get('decoding_warning'):
        return {**plan, 'reason': 'desk_decode_incomplete'}
    signature = summary.get('signature', {})
    if (signature.get('video_size') != source.stat().st_size
            or signature.get('video_mtime') != source.stat().st_mtime_ns
            or summary.get('image_size') != [width, height]
            or summary.get('decoded_frames') != metadata.get('frames')):
        return {**plan, 'reason': 'desk_source_metadata_mismatch'}
    plan['producer_source_binding'] = 'legacy_size_mtime_dimensions_framecount; current_source_sha256_recorded'
    plan['input_files'] = {str(path.relative_to(folder)): sha256(path) for path in (summary_path, frames_path)}
    rows = _raw_frames(frames_path, width, height)
    crop, roi = _stable_table(rows, width, height)
    plan.update(crop_xyxy=crop, roi_provenance=roi)
    if crop is None:
        return {**plan, 'reason': roi['reason']}
    triggers = []
    previous = {}
    for row in rows:
        t = row['timestamp_sec']
        if t > duration:
            continue
        for d in row['candidates'] + row['review']:
            if d['label'] not in {'cash', 'identity_card'} or not _box(d.get('box')) or not _inside(d['box'], crop):
                continue
            label = d['label']
            last = previous.get(label)
            moved = (last and 0 < t-last[0] <= 1.1
                     and min(math.dist(_center(b), _center(d['box'])) for b in last[1]) >= 10)
            fresh = last is None or t-last[0] > 1.1
            triggers.append({'timestamp_sec': t, 'priority': 3 if moved else 2 if fresh else 1,
                             'reason': 'object_position_change_candidate' if moved else 'object_appearance_candidate' if fresh else 'object_presence_candidate',
                             'label': label, 'box': d['box'], 'source_frame': row['source_frame']})
        for label in ('cash', 'identity_card'):
            boxes = [d['box'] for d in row['candidates'] + row['review']
                     if d['label'] == label and _box(d.get('box')) and _inside(d['box'], crop)]
            if boxes:
                previous[label] = (t, boxes)
    # Raw tracks are ONLY proposals: reflected people must not create violations.
    people_csv, people_summary = folder/'people/review.csv', folder/'people/review.json'
    if _complete_stage(job, 'people') and people_csv.is_file() and people_summary.is_file():
        info = _read_json(people_summary)
        if info.get('frame_count_matches') is True and not info.get('decoding_warning'):
            plan['input_files'].update({str(p.relative_to(folder)): sha256(p) for p in (people_csv, people_summary)})
            prior = {}
            with people_csv.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    try:
                        t, track = float(row['timestamp_sec']), row['track_id']
                        box = [float(row[k]) for k in ('x1', 'y1', 'x2', 'y2')]
                    except (ValueError, KeyError):
                        continue
                    if not _finite(t) or not 0 <= t <= duration or not _box(box) or _iou(box, crop) <= 0:
                        continue
                    last = prior.get(track)
                    if last and .2 <= t-last[0] <= .8:
                        if math.dist(_center(box), _center(last[1])) >= max(12., .04*math.hypot(width, height)):
                            triggers.append({'timestamp_sec': t, 'priority': 2, 'reason': 'raw_person_motion_near_table', 'track_id': track})
                        prior[track] = (t, box)
                    elif last is None or t-last[0] > .8:
                        prior[track] = (t, box)
    # Require nearby table support; do not use a stable ROI across a long unseen gap.
    anchors = roi['observed_times']
    triggers = [t for t in triggers if min(abs(t['timestamp_sec']-a) for a in anchors) <= 1.1]
    windows = _select_windows(triggers, duration, max_seconds)
    if not windows:
        return {**plan, 'reason': 'no_automatic_cash_card_or_motion_proposals'}
    covered = sum(w['end_sec']-w['start_sec'] for w in windows)
    return {**plan, 'status': 'planned', 'reason': 'automatic_candidates_for_closeup_review',
            'windows': windows, 'planned_window_seconds': round(covered, 6),
            'window_coverage_fraction': min(1., covered/duration), 'trigger_count': len(triggers)}


def select_dense_samples(timestamps, windows, fps=5., max_samples=60):
    """Use actual source PTS, half-open windows and an explicit inference budget."""
    if any(not _finite(t) for t in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Source PTS must be finite and strictly increasing')
    if not _finite(fps) or fps <= 0 or type(max_samples) is not int or max_samples < 1:
        raise ValueError('Invalid dense sampling parameters')
    if any(not _finite(w.get('start_sec')) or not _finite(w.get('end_sec'))
           or not 0 <= w['start_sec'] < w['end_sec'] for w in windows):
        raise ValueError('Invalid sample window')
    if any(a['end_sec'] > b['start_sec'] for a, b in zip(windows, windows[1:])):
        raise ValueError('Sample windows must be ordered and nonoverlapping')
    result = []
    for wi, window in enumerate(windows):
        next_time = window['start_sec']
        for index, t in enumerate(timestamps):
            if not window['start_sec'] <= t < window['end_sec'] or t + 1e-8 < next_time:
                continue
            if len(result) == max_samples:
                return result
            result.append({'source_frame': index, 'timestamp_sec': t, 'timestamp_source': 'video_pts', 'window_index': wi})
            next_time = t + 1/fps
    return result


def inspect_source(video):
    """Decode the complete source to validate PTS and detect truncated reading."""
    import cv2
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(video))
    timestamps = []
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open source video')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not _finite(fps) or fps <= 0 or expected <= 0 or min(width, height) <= 0:
            raise ValueError('Missing source metadata')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            if frame.shape[:2] != (height, width) or not _finite(t) or timestamps and t <= timestamps[-1]:
                raise ValueError('Source dimensions changed or nonmonotonic/missing PTS')
            timestamps.append(t)
    finally:
        capture.release()
    if len(timestamps) != expected:
        raise ValueError(f'Incomplete source decode: {len(timestamps)} of {expected} frames')
    return timestamps, {'decoded_frames': len(timestamps), 'metadata_frames': expected,
                        'frame_count_matches': True, 'source_fps': fps,
                        'source_dimensions': [width, height], 'first_timestamp_sec': timestamps[0],
                        'last_timestamp_sec': timestamps[-1], 'negative_preroll_frames': sum(t < 0 for t in timestamps)}


def detect_roi_sample(image, sample, crop, detector, threshold=.28):
    """Injected grouped detector: crop pixel boxes -> full source pixel boxes."""
    from candidate_filters import filter_candidates
    crop = resolve_crop(crop, image.width, image.height)
    cropped = image.crop(crop)
    grouped = detector.detect_groups(cropped, PROMPTS, threshold)
    raw, invalid = [], []
    for group, detections in grouped.items():
        for d in detections:
            try:
                if not _finite(d.get('score')) or not 0 <= d['score'] <= 1:
                    raise ValueError('Invalid model score')
                box = crop_box_to_source(d['box'], crop, image.width, image.height)
                raw.append({**d, 'box': box, 'query': group, 'model_box_crop': d['box'],
                            'crop_xyxy': crop, 'coordinate_space': 'original_frame_pixels',
                            'source': 'model', 'score_kind': 'model_score_not_probability'})
            except (ValueError, KeyError, TypeError) as error:
                invalid.append({'query': group, 'reason': str(error)})
    filtered = filter_candidates(raw, image_size=(image.width, image.height))
    return {**sample, 'crop_xyxy': crop, 'source_dimensions': list(image.size),
            'coordinate_space': 'original_frame_pixels', 'raw_detections': raw, **filtered,
            'invalid_detections': invalid, 'action': 'unknown', 'violations': [],
            'interpretation': 'unverified_closeup_object_observations'}


def _model_signature():
    weight_files = ['model.safetensors', 'config.json', 'preprocessor_config.json',
                    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                    'added_tokens.json', 'vocab.txt']
    files = [*(MODEL_ROOT/'weights'/name for name in weight_files),
             MODEL_ROOT/'detect_local.py', ROOT/'candidate_filters.py', ROOT/'inference_optimized.py',
             ROOT/'interaction_refinement.py', Path(__file__)]
    return {str(path): sha256(path) for path in files}


def run_stage(video, job_dir, output, *, max_seconds=12., sample_fps=5., threshold=.28,
              plan_only=False, detector=None):
    """Write a new independent output directory; optional detector is for testing."""
    if not _finite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must be between zero and one')
    folder, video, output = Path(job_dir).resolve(), Path(video).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Dense output is nonempty; use a new directory')
    job = _read_json(folder/'job.json')
    if video != (folder/job['_source_file']).resolve():
        raise ValueError('Explicit source video must match job source')
    plan = plan_dense_windows(folder, job, max_seconds=max_seconds, sample_fps=sample_fps)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'plan.json', plan)
    base = {'version': VERSION, 'source_sha256': plan['source_sha256'], 'action': 'unknown', 'violations': [],
            'human_labels_used': False, 'interpretation': 'unverified_closeup_object_observations',
            'planned_window_seconds': plan.get('planned_window_seconds', 0),
            'window_coverage_fraction': plan.get('window_coverage_fraction', 0),
            'limitations': list(LIMITATIONS)}
    if plan['status'] != 'planned' or plan_only:
        summary = {**base, 'status': 'planned_only' if plan_only and plan['status'] == 'planned' else 'not_checked',
                   'reason': plan['reason'], 'model_loaded': False, 'analyzed_samples': 0}
        write_json(output/'summary.json', summary)
        return summary
    records, started, model_loaded = [], time.perf_counter(), False
    try:
        timestamps, decode = inspect_source(video)
        if decode['source_dimensions'] != plan['source_dimensions']:
            raise ValueError('Source dimensions do not match job')
        samples = select_dense_samples(timestamps, plan['windows'], sample_fps, plan['maximum_samples'])
        if not samples:
            raise ValueError('Planned windows contain no source samples')
        signature = {'version': VERSION, 'source_sha256': plan['source_sha256'],
                     'input_files': plan['input_files'], 'model_and_code': _model_signature(),
                     'prompts': PROMPTS, 'threshold': threshold, 'plan_sha256': sha256(output/'plan.json'),
                     'inference_kind': 'injected_test_detector' if detector is not None else 'local_grounding_dino'}
        write_json(output/'signature.json', signature)
        if detector is None:
            sys.path.insert(0, str(MODEL_ROOT))
            from detect_local import load_model, detect
            from inference_optimized import PromptDetector
            import torch
            torch.set_num_threads(2)
            detector = PromptDetector(load_model(), detect)
            model_loaded = True
        from interaction_refinement import iter_selected_frames
        with (output/'observations.jsonl').open('w', encoding='utf-8') as stream:
            for sample, image in iter_selected_frames(video, samples):
                record = detect_roi_sample(image, sample, plan['crop_xyxy'], detector, threshold)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
                records.append(record)
                print(f"DENSE {len(records)}/{len(samples)} at {sample['timestamp_sec']:.3f}s", flush=True)
        counts = Counter(d['label'] for row in records for d in row['candidates'])
        summary = {**base, **decode, 'status': 'completed', 'model_loaded': model_loaded,
                   'planned_samples': len(samples), 'analyzed_samples': len(records),
                   'all_planned_samples_analyzed': len(records) == len(samples),
                   'candidate_observations_by_class': dict(counts),
                   'review_observations': sum(len(r['review']) for r in records),
                   'sampling_complete_for_selected_windows_only': True,
                   'inference_kind': signature['inference_kind'],
                   'signature_sha256': sha256(output/'signature.json')}
    except Exception as error:
        summary = {**base, 'status': 'partial' if records else 'not_checked', 'reason': str(error),
                   'model_loaded': model_loaded, 'analyzed_samples': len(records),
                   'all_planned_samples_analyzed': False}
    summary['processing_seconds'] = round(time.perf_counter()-started, 3)
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=12.)
    parser.add_argument('--sample-fps', type=float, default=5.)
    parser.add_argument('--threshold', type=float, default=.28)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    summary = run_stage(args.video, args.job_dir, args.output, max_seconds=args.max_seconds,
                        sample_fps=args.sample_fps, threshold=args.threshold, plan_only=args.plan_only)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if summary['status'] == 'partial' else 0


if __name__ == '__main__':
    raise SystemExit(main())
