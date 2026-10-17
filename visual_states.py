"""Conservative CPU reference classifier for visible door and safe-door states.

This is calibrated to reviewed views of one room, not a general door detector. Static
background feature matches register a frame to the reviewed reference. Crops
are compared with reviewed state examples; ambiguous or unfamiliar appearances
produce unknown. A closed safe door never implies that its lock is engaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from rule_settings import DOOR_OPEN_THRESHOLD_SEC

ROOT = Path(__file__).resolve().parent
BASELINE_MODEL = ROOT / 'training' / 'visual_states'
MULTIVIEW_MODEL = ROOT / 'training' / 'visual_states_v2'
DEFAULT_MODEL = MULTIVIEW_MODEL
STATES = ('closed', 'open', 'unknown')
ROIS = {'door': [24, 20, 136, 185], 'safe': [181, 274, 253, 365]}
# These are static scene landmarks, not people, moving door, or safe door.
ANCHORS = [[137, 28, 246, 140], [270, 110, 443, 310], [340, 470, 455, 680]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def patch_feature(bgr: np.ndarray) -> np.ndarray:
    """Appearance + edges; no learned network and no unsafe pickled model."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 40), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    # Moderate exposure normalisation keeps material/shape differences visible.
    norm = (small - small.mean()) / max(float(small.std()), .12)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3) * .25
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3) * .25
    color = cv2.resize(bgr, (16, 20), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    return np.concatenate([norm.ravel() * .22, gx.ravel() * .6, gy.ravel() * .6, color.ravel() * .55])


class SceneRegistration:
    def __init__(self, reference: np.ndarray, anchors=None):
        if reference is None or reference.shape[:2] != (848, 464):
            raise ValueError('Reference must be the reviewed 464 x 848 scene.')
        self.reference = reference
        self.height, self.width = reference.shape[:2]
        mask = np.zeros((self.height, self.width), np.uint8)
        for x1, y1, x2, y2 in (anchors if anchors is not None else ANCHORS):
            mask[y1:y2, x1:x2] = 255
        self.sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=.015)
        self.kp, self.desc = self.sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), mask)
        self.matcher = cv2.BFMatcher()

    def query_features(self, frame: np.ndarray):
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        kq, dq = self.sift.detectAndCompute(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), None)
        return resized, kq, dq

    def align(self, frame: np.ndarray, query=None):
        height, width = frame.shape[:2]
        aspect_error = abs((width / height) / (self.width / self.height) - 1)
        if aspect_error > .04:
            return None, None, {'matched': False, 'reason': 'different_aspect_ratio'}
        resized, kq, dq = self.query_features(frame) if query is None else query
        if dq is None or self.desc is None:
            return None, None, {'matched': False, 'reason': 'no_scene_features'}
        pairs = self.matcher.knnMatch(dq, self.desc, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
        if len(matches) < 15:
            return None, None, {'matched': False, 'reason': 'too_few_landmarks', 'matches': len(matches)}
        source = np.float32([kq[m.queryIdx].pt for m in matches])
        target = np.float32([self.kp[m.trainIdx].pt for m in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2)
        if matrix is None or inliers is None:
            return None, None, {'matched': False, 'reason': 'registration_failed'}
        mask = inliers.ravel().astype(bool)
        inlier_count = int(mask.sum())
        ratio = inlier_count / len(matches)
        scale = float(math.hypot(matrix[0, 0], matrix[0, 1]))
        rotation = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
        area = float(cv2.contourArea(cv2.convexHull(target[mask]))) if inlier_count >= 3 else 0.
        error = np.linalg.norm(cv2.transform(source[None, :, :], matrix)[0] - target, axis=1)
        median_error = float(np.median(error[mask]))
        info = dict(matched=False, inliers=inlier_count, matches=len(matches), inlier_ratio=round(ratio, 4),
                    landmark_area=round(area, 2), reprojection_error_px=round(median_error, 3),
                    scale=round(scale, 4), rotation_deg=round(rotation, 3),
                    translation_x=round(float(matrix[0, 2]), 3), translation_y=round(float(matrix[1, 2]), 3))
        # Narrow viewpoint range is deliberate. Matching the room is insufficient
        # to claim a calibrated state from a substantially different camera pose.
        if (inlier_count < 18 or ratio < .30 or area < 2000 or median_error > 1.1
                or not .82 <= scale <= 1.20 or abs(rotation) > 6
                or abs(matrix[0, 2]) > 65 or abs(matrix[1, 2]) > 85):
            info['reason'] = 'unverified_viewpoint'
            return None, None, info
        aligned = cv2.warpAffine(resized, matrix, (self.width, self.height))
        # Avoid analysing padded pixels if an object moves outside the camera.
        valid = cv2.warpAffine(np.full((self.height, self.width), 255, np.uint8), matrix,
                               (self.width, self.height), flags=cv2.INTER_NEAREST)
        info.update(matched=True, reason='reviewed_scene_landmarks_match')
        full_matrix = matrix @ np.diag([self.width / width, self.height / height, 1.])
        return aligned, (full_matrix, valid), info


class VisualStateDetector:
    def __init__(self, model_dir: Path = DEFAULT_MODEL):
        self.model_dir = Path(model_dir)
        self.metadata = json.loads((self.model_dir / 'model.json').read_text(encoding='utf-8'))
        self.views = []
        for view in self.metadata.get('views', [{'view_id': 'legacy', 'model_file': 'model.json'}]):
            path = self.model_dir / view['model_file']
            if view.get('model_sha256') and sha256(path) != view['model_sha256']:
                raise ValueError('Visual-state view metadata checksum mismatch.')
            metadata = json.loads(path.read_text(encoding='utf-8'))
            reference_path = path.parent / metadata['reference_file']
            examples_path = path.parent / metadata['examples_file']
            if sha256(reference_path) != metadata['reference_sha256'] or sha256(examples_path) != metadata['examples_sha256']:
                raise ValueError('Visual-state reference/model checksum mismatch.')
            registration = SceneRegistration(cv2.imread(str(reference_path)), metadata.get('anchor_regions'))
            with np.load(examples_path, allow_pickle=False) as data:
                examples = {kind: (data[kind + '_features'].copy(), data[kind + '_labels'].copy())
                            for kind in metadata['rois'] if kind + '_features' in data}
            self.views.append(dict(view_id=view['view_id'], metadata=metadata, registration=registration, examples=examples))
        # Kept for consumers which inspect the original registration.
        self.registration = self.views[0]['registration']
        self.examples = self.views[0]['examples']

    def _registered_view(self, frame):
        query = self.registration.query_features(frame)
        candidates, failures = [], []
        for view in self.views:
            aligned, transform, scene = view['registration'].align(frame, query=query)
            scene['view_id'] = view['view_id']
            if aligned is None:
                failures.append(scene)
                continue
            # Prefer the calibrated camera pose with the strongest spatial fit;
            # never select a camera because its classifier produces a label.
            cost = (1 + abs(math.log(scene['scale'])) * 4
                    + abs(scene['translation_x']) / 100 + abs(scene['translation_y']) / 100)
            quality = scene['inliers'] * scene['inlier_ratio'] / ((1 + scene['reprojection_error_px']) * cost)
            scene['registration_quality'] = round(quality, 3)
            candidates.append((quality, view, aligned, transform, scene))
        if not candidates:
            failure = max(failures, key=lambda x: x.get('inliers', 0)) if failures else {'matched': False, 'reason': 'no_reviewed_views'}
            failure['regions'] = {'reflections': []}
            return None, None, None, failure
        _, view, aligned, transform, scene = max(candidates, key=lambda c: c[0])
        scene['regions'] = self._project_regions(view, transform[0], frame.shape)
        return view, aligned, transform, scene

    @staticmethod
    def _project_regions(view, matrix, shape):
        height, width = shape[:2]
        inverse = cv2.invertAffineTransform(matrix)
        regions = {'reflections': []}
        for region in view['metadata'].get('reflection_regions', []):
            polygon = cv2.transform(np.float32([region['polygon']]), inverse)[0]
            # Clipping a region to the frame may include an unrelated visible
            # surface; suppress it when a corner is outside the image instead.
            if (polygon[:, 0].min() < 0 or polygon[:, 1].min() < 0
                    or polygon[:, 0].max() > width or polygon[:, 1].max() > height):
                continue
            points = [[round(float(x), 2), round(float(y), 2)] for x, y in polygon]
            regions['reflections'].append(dict(region_id=region['region_id'], polygon=points,
                box=[round(float(polygon[:, 0].min()), 2), round(float(polygon[:, 1].min()), 2),
                     round(float(polygon[:, 0].max()), 2), round(float(polygon[:, 1].max()), 2)],
                source='reviewed_reflection_region', view_id=view['view_id'],
                interpretation='context_only_not_a_person_exclusion_mask'))
        return regions

    def project_regions(self, frame: np.ndarray) -> dict:
        """Registered glass context. Presence in a region is NOT proof of reflection."""
        scene = self.predict(frame)['scene']
        return {'scene': scene, 'regions': scene['regions']}

    def predict(self, frame: np.ndarray) -> dict:
        view, aligned, transform, scene = self._registered_view(frame)
        result = {'scene': scene}
        for kind in ROIS:
            item = {'state': 'unknown', 'score': None, 'score_kind': 'reference_similarity_not_probability',
                    'box': None, 'reason': 'unverified_viewpoint'}
            if kind == 'safe':
                item['locked'] = None
            if aligned is None:
                result[kind] = item
                continue
            box = view['metadata']['rois'].get(kind)
            if box is None or kind not in view['examples']:
                item['reason'] = 'no_reviewed_state_examples_for_view'
                result[kind] = item
                continue
            matrix, valid = transform
            x1, y1, x2, y2 = box
            inverse = cv2.invertAffineTransform(matrix)
            corners = cv2.transform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), inverse)[0]
            height, width = frame.shape[:2]
            item['box'] = [round(float(np.clip(corners[:, 0].min(), 0, width)), 2),
                           round(float(np.clip(corners[:, 1].min(), 0, height)), 2),
                           round(float(np.clip(corners[:, 0].max(), 0, width)), 2),
                           round(float(np.clip(corners[:, 1].max(), 0, height)), 2)]
            if np.mean(valid[y1:y2, x1:x2] == 255) < .98:
                item['reason'] = 'object_outside_frame'
                result[kind] = item
                continue
            feature = patch_feature(aligned[y1:y2, x1:x2])
            features, labels = view['examples'][kind]
            distances = np.sqrt(np.mean((features - feature[None]) ** 2, axis=1))
            best = int(np.argmin(distances))
            nearest_state = str(labels[best])
            distance = float(distances[best])
            competing = distances[labels != nearest_state]
            other_distance = float(competing.min()) if len(competing) else 1.
            margin = (other_distance - distance) / max(other_distance, 1e-9)
            limits = view['metadata']['thresholds'][kind]
            item.update(score=round(max(0., 1 - distance / limits['max_distance']), 4),
                        reference_distance=round(distance, 5), reference_margin=round(margin, 5))
            if nearest_state == 'unknown':
                item['reason'] = 'occluded_or_ambiguous_reference'
            elif distance > limits['max_distance']:
                item['reason'] = 'unfamiliar_object_appearance_or_occlusion'
            elif margin < limits['min_margin']:
                item['reason'] = 'ambiguous_state'
            else:
                item.update(state=nearest_state, reason='reviewed_state_reference_match')
            result[kind] = item
        # These reviewed reflection locations are on the moving glass door.
        # Its open/uncertain position invalidates the closed-plane projection.
        # In particular, never offer a doorway exclusion hint during entry.
        if result['door']['state'] != 'closed':
            result['scene']['regions']['reflections'] = []
            result['scene']['reflection_region_status'] = 'door_plane_not_confirmed_closed'
        else:
            result['scene']['reflection_region_status'] = 'closed_glass_context_only'
        return result


def build_intervals(observations: list[dict], sample_seconds: float, fps: float | None = None) -> list[dict]:
    """Never bridge unknown/closed observations or missing sample gaps."""
    if not math.isfinite(sample_seconds) or sample_seconds <= 0:
        raise ValueError('Positive sample_seconds required.')
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError('Positive FPS required.')
    max_frame_gap = max(1, math.ceil(sample_seconds * fps - 1e-6)) if fps is not None else None
    max_gap = (max_frame_gap / fps if fps is not None else sample_seconds) + 2e-6
    times = [float(row['timestamp_sec']) for row in observations]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Observation timestamps must be finite and strictly increasing.')
    intervals = []
    for kind in ROIS:
        active = []
        def finish():
            if not active:
                return
            start, end = active[0], active[-1]
            evidence = active[len(active) // 2]
            span = max(0., float(end['timestamp_sec']) - float(start['timestamp_sec']))
            intervals.append(dict(object=kind, state='open', start_sec=start['timestamp_sec'], end_sec=end['timestamp_sec'],
                                  observed_duration_sec=round(span, 6), sample_count=len(active),
                                  evidence_frame=evidence['frame_index'], evidence_timestamp_sec=evidence['timestamp_sec'],
                                  **({'threshold_sec': DOOR_OPEN_THRESHOLD_SEC,
                                      'threshold_reached': span >= DOOR_OPEN_THRESHOLD_SEC - 1e-6}
                                     if kind == 'door' else {}),
                                  interpretation='visible_open_state_only_not_a_confirmed_violation'))
            active.clear()
        for row in observations:
            if active and (float(row['timestamp_sec']) - float(active[-1]['timestamp_sec']) > max_gap
                           or (max_frame_gap is not None and row['frame_index'] - active[-1]['frame_index'] > max_frame_gap)):
                finish()
            if row.get(kind, {}).get('state') == 'open':
                active.append(row)
            else:
                finish()
        finish()
    return sorted(intervals, key=lambda row: (row['start_sec'], row['object']))


def detect_video(video: Path, output: Path | None = None, sample_seconds: float = .25,
                 model_dir: Path = DEFAULT_MODEL) -> dict:
    if not math.isfinite(sample_seconds) or not .05 <= sample_seconds <= .5:
        raise ValueError('Sampling must be between 0.05 and 0.5 seconds for visible-state checks.')
    video = Path(video).resolve()
    cv2.setNumThreads(1)
    detector = VisualStateDetector(model_dir)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video has no valid FPS.')
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    observations, frame_index, next_time, last_time = [], 0, 0., -1.
    timestamp_fallbacks = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= last_time:
                timestamp = frame_index / fps
                timestamp_fallbacks += 1
            if timestamp <= last_time:
                raise ValueError('Cannot establish monotonic video timestamps.')
            last_time = timestamp
            if timestamp + 1e-6 >= next_time:
                prediction = detector.predict(frame)
                observations.append(dict(timestamp_sec=round(timestamp, 6), frame_index=frame_index,
                                         source_frame=frame_index, **prediction))
                next_time = timestamp + sample_seconds
            frame_index += 1
    finally:
        cap.release()
    complete = expected > 0 and frame_index == expected
    intervals = build_intervals(observations, sample_seconds, fps=fps)
    result = dict(schema_version=2, status='completed' if complete else 'partial',
                  video=dict(path=str(video), name=video.name, sha256=sha256(video), fps=fps, width=width, height=height,
                             duration_sec=round(frame_index / fps, 6)),
                  decoded_frames=frame_index, expected_frames=expected, frame_count_matches=complete,
                  sample_seconds=sample_seconds, timestamp_fallbacks=timestamp_fallbacks,
                  first_timestamp_sec=observations[0]['timestamp_sec'] if observations else None,
                  last_timestamp_sec=round(last_time, 6) if frame_index else None,
                  elapsed_seconds=round(time.monotonic() - started, 3),
                  observations=observations, intervals=intervals,
                  rule_thresholds={'door_open_sec': DOOR_OPEN_THRESHOLD_SEC},
                  model=dict(type='registered_local_state_references', model_sha256=sha256(Path(model_dir) / 'model.json'),
                             inference_version='multiview_2_closed_plane_reflection_context',
                             calibrated_views=[view['view_id'] for view in detector.views],
                             validation_file=str(Path(model_dir) / 'validation_report.json')),
                  capabilities=dict(
                      door=dict(status='partial', output='visible_open_closed_unknown', general_detector=False,
                                note='Video-disjoint sparse validation exists; unknown gaps can cause missed or fragmented events.'),
                      safe=dict(status='experimental', output='visible_open_closed_unknown', locked_state_supported=False,
                                note='Held-out safe-open recall was zero in the small initial test; not a validated violation detector.')),
                  limitations=[
                      'Prototype calibrated to reviewed views of one room; unknown means not checked.',
                      'New door views use the door/frame seam; partly closed transitions can still be missed.',
                      'Reflection regions are offered only with a recognised closed door and are not exclusion masks.',
                      'Open/closed describes visible door position, never the locked state.',
                      'A safe being open alone is not classified as a violation.',
                      'Short occlusions and unknown observations break intervals; duration is the span between samples.',
                      'Matching decoded frame count does not exclude every concealed codec error.',
                      'This visual stage does not identify cashier roles or establish whether a safe is unattended.'])
    if output is not None:
        output = Path(output)
        path = output if output.suffix.lower() == '.json' else output / 'observations.json'
        atomic_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-seconds', type=float, default=.25)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    report = detect_video(args.video, args.output, args.sample_seconds, args.model_dir)
    print(json.dumps({key: report[key] for key in ['status', 'decoded_frames', 'expected_frames', 'frame_count_matches', 'elapsed_seconds']}, ensure_ascii=False))
    print('observations', len(report['observations']), 'open_intervals', len(report['intervals']))
    print('door_threshold_sec', DOOR_OPEN_THRESHOLD_SEC, 'door_intervals_reaching_threshold',
          sum(row['object'] == 'door' and row['threshold_reached'] for row in report['intervals']))
    if report['status'] != 'completed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
