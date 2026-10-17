"""Conservative, source-bound matching of objects explicitly reviewed by a human.

This is evidence reuse within the exact reviewed video, not a trained detector.
Raw detector results belong to the caller and are never mutated by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from candidate_filters import canonical_label


DEFAULT_ANNOTATIONS = Path(__file__).parent / 'annotations/cash_review_2026_10_17.json'
CORRECTED_CLASSES = {'chocolate': 'food', 'food': 'food'}
CASH_REFERENCE_CONFLICTS = {'cash', 'food', 'drink', 'identity_card', 'document', 'notebook'}


def _reference_label(row):
    """Load supported source-bound corrections with their required approvals."""
    if not isinstance(row, dict) or not row.get('correction_id'):
        return None
    confirmed_label = row.get('confirmed_label')
    if not isinstance(confirmed_label, str):
        return None
    negatives = row.get('negative_for_labels', [])
    if (row.get('label_status') == 'human_corrected_class'
            and confirmed_label in CORRECTED_CLASSES
            and isinstance(negatives, (list, tuple)) and 'cash' in negatives):
        return CORRECTED_CLASSES[confirmed_label]
    if (row.get('label_status') == 'human_confirmed_class'
            and confirmed_label == 'cash'
            and row.get('reference_matching_enabled') is True):
        return 'cash'
    if (row.get('label_status') == 'human_corrected_class'
            and confirmed_label == 'bag'
            and row.get('reference_matching_enabled') is True
            and isinstance(negatives, (list, tuple))
            and 'cash' in negatives and 'food' in negatives):
        return 'bag'
    return None


def _iou(first, second):
    try:
        a = [float(value) for value in first]
        b = [float(value) for value in second]
        intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
        return intersection / union if union > 0 else 0.0
    except (ValueError, TypeError, IndexError):
        return 0.0


def _same_cash_object(candidate_box, reference_box):
    """Reject large containers; tolerate slightly shifted small-object boxes."""
    try:
        candidate = [float(value) for value in candidate_box]
        reference = [float(value) for value in reference_box]
        if len(candidate) != 4 or len(reference) != 4:
            return False
        if not all(np.isfinite(value) for value in candidate + reference):
            return False
        candidate_width, candidate_height = candidate[2]-candidate[0], candidate[3]-candidate[1]
        reference_width, reference_height = reference[2]-reference[0], reference[3]-reference[1]
        candidate_area = max(0.0, candidate_width) * max(0.0, candidate_height)
        reference_area = max(0.0, reference_width) * max(0.0, reference_height)
        if min(candidate_area, reference_area) <= 0 or candidate_area > 3.0 * reference_area:
            return False
        intersection = max(0.0, min(candidate[2], reference[2])-max(candidate[0], reference[0])) * max(0.0, min(candidate[3], reference[3])-max(candidate[1], reference[1]))
        if candidate_area > 2.0 * reference_area:
            # Small detector boxes can include an extra margin around a distant
            # banknote. Accept this bounded case only when centered on nearly
            # the entire reviewed object, never an elongated or broad container.
            centered = (abs((candidate[0]+candidate[2]-reference[0]-reference[2])/2) <= reference_width*.25
                        and abs((candidate[1]+candidate[3]-reference[1]-reference[3])/2) <= reference_height*.25)
            return (intersection / reference_area >= .9 and centered
                    and candidate_width <= 2.0 * reference_width
                    and candidate_height <= 2.0 * reference_height)
        return _iou(candidate, reference) >= 0.45 or intersection / reference_area >= 0.7
    except (ValueError, TypeError, IndexError):
        return False


def _box(values, width, height):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        coordinates = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in coordinates):
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in coordinates]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class _Reference:
    label: str
    confirmed_class: str
    frame: int
    frame_size: tuple[int, int]
    box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    template: np.ndarray
    gray: np.ndarray
    annotation_number: int
    correction_id: str
    rejected_cash_regions: tuple[tuple[int, int, int, int], ...] = ()


class ReviewedObjectMatcher:
    """Match corrected food or explicitly enabled cash/bags in their source video.

    ``for_video`` returns ``None`` when the optional reference is unavailable.
    ``apply`` returns ``(new_detections, correction_notes)``. A matched reference
    observation uses correlation as the pipeline's numeric ``score`` with an
    explicit similarity type, never as a model probability.
    """

    def __init__(self, source_sha256: str, references: list[_Reference]):
        self.source_sha256 = source_sha256
        self.references = references
        self.search_radius = 20
        self.min_correlation = 0.90
        self.max_color_mae = 0.08

    @classmethod
    def for_video(cls, path, annotations_path=DEFAULT_ANNOTATIONS):
        """Read/hash once during setup; every failure disables this optional aid."""
        capture = None
        try:
            annotation = json.loads(Path(annotations_path).read_text(encoding='utf-8-sig'))
            candidates = [row for row in annotation.get('candidates', [])
                          if _reference_label(row) is not None]
            if not candidates:
                return None
            source = Path(path)
            with source.open('rb') as stream:
                source_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            candidates = [row for row in candidates if row.get('source_video_sha256') == source_sha256]
            if not candidates:
                return None
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                return None
            # Decode sequentially to bind exact zero-based frame numbers, avoiding
            # codec-dependent seeking to a nearby keyframe.
            frame_numbers = sorted({int(row['source_frame_zero_based']) for row in candidates})
            if frame_numbers[0] < 0:
                return None
            reference_frames = {}
            for index in range(frame_numbers[-1] + 1):
                ok, frame = capture.read()
                if not ok:
                    return None
                if index in frame_numbers:
                    reference_frames[index] = frame
            references = []
            for row in candidates:
                index = int(row['source_frame_zero_based'])
                frame = reference_frames[index]
                height, width = frame.shape[:2]
                if list(row.get('frame_size', [])) != [width, height]:
                    continue
                box = _box(row.get('integer_box_xyxy', row.get('detector_box_xyxy')), width, height)
                crop_box = _box(row.get('crop_box_xyxy'), width, height)
                if box is None or crop_box is None:
                    continue
                x1, y1, x2, y2 = box
                cx1, cy1, cx2, cy2 = crop_box
                if not (cx1 <= x1 < x2 <= cx2 and cy1 <= y1 < y2 <= cy2):
                    continue
                # Use the annotated crop but compare its object interior. Hands
                # moving in the crop margin must not manufacture an object match.
                reviewed_crop = frame[cy1:cy2, cx1:cx2]
                template = reviewed_crop[y1-cy1:y2-cy1, x1-cx1:x2-cx1].copy()
                gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                if min(gray.shape) < 8 or gray.size < 128 or float(gray.std()) < 8.0:
                    continue
                if float(cv2.Laplacian(gray, cv2.CV_32F).std()) < 5.0:
                    continue
                rejected_cash_regions = []
                if _reference_label(row) == 'bag':
                    # A bag can contain genuine money. Only a separately reviewed
                    # negative region may suppress a smaller cash box inside it.
                    regions = row.get('rejected_class_regions', [])
                    if not isinstance(regions, (list, tuple)):
                        regions = []
                    for region in regions:
                        if not isinstance(region, dict) or region.get('label') != 'cash':
                            continue
                        rejected_box = _box(region.get('box_xyxy'), width, height)
                        if rejected_box is None:
                            continue
                        rx1, ry1, rx2, ry2 = rejected_box
                        if x1 <= rx1 < rx2 <= x2 and y1 <= ry1 < ry2 <= y2:
                            rejected_cash_regions.append(rejected_box)
                references.append(_Reference(
                    _reference_label(row), row['confirmed_label'],
                    index, (width, height), box, crop_box, template, gray,
                    int(row['number']), str(row['correction_id']),
                    tuple(rejected_cash_regions)))
            return cls(source_sha256, references) if references else None
        except Exception:
            return None
        finally:
            if capture is not None:
                capture.release()

    def apply(self, frame, detections):
        """Return matched human-reference evidence without changing raw inputs."""
        result = copy.deepcopy(list(detections))
        notes: list[dict[str, Any]] = []
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return result, notes
        height, width = frame.shape[:2]
        for reference in self.references:
            if reference.frame_size != (width, height):
                continue
            try:
                x1, y1, x2, y2 = reference.box
                left, top = max(0, x1-self.search_radius), max(0, y1-self.search_radius)
                right, bottom = min(width, x2+self.search_radius), min(height, y2+self.search_radius)
                search = frame[top:bottom, left:right]
                search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
                if search_gray.shape[0] < reference.gray.shape[0] or search_gray.shape[1] < reference.gray.shape[1]:
                    continue
                response = cv2.matchTemplate(search_gray, reference.gray, cv2.TM_CCOEFF_NORMED)
                _, correlation, _, location = cv2.minMaxLoc(response)
                if not np.isfinite(correlation) or correlation < self.min_correlation:
                    continue
                match_x, match_y = left+location[0], top+location[1]
                matched_box = [match_x, match_y, match_x+x2-x1, match_y+y2-y1]
                patch = frame[matched_box[1]:matched_box[3], matched_box[0]:matched_box[2]]
                color_mae = float(np.mean(np.abs(patch.astype(np.float32)-reference.template.astype(np.float32))) / 255.0)
                if not np.isfinite(color_mae) or color_mae > self.max_color_mae:
                    continue
                if float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std()) < 8.0:
                    continue
                evidence = {
                    'source': 'human_confirmed_reference_match',
                    'confirmed_class': reference.confirmed_class,
                    'reference_frame_zero_based': reference.frame,
                    'reference_video_sha256': self.source_sha256,
                    'reference_box_xyxy': list(reference.box),
                    'reference_crop_box_xyxy': list(reference.crop_box),
                    'annotation_number': reference.annotation_number,
                    'correction_id': reference.correction_id,
                    'correlation': round(float(correlation), 6),
                    'color_mae': round(color_mae, 6),
                    'score_type': 'reference_match_not_model_probability',
                    'score_kind': 'reference_similarity_not_probability',
                    'score': round(float(correlation), 6),
                }
                shifted_negative_boxes = [
                    [rx1 + match_x - x1, ry1 + match_y - y1,
                     rx2 + match_x - x1, ry2 + match_y - y1]
                    for rx1, ry1, rx2, ry2 in reference.rejected_cash_regions
                ]
                if reference.label == 'bag':
                    evidence['reference_rejected_class_regions'] = [
                        {'label': 'cash', 'box_xyxy': list(box)}
                        for box in reference.rejected_cash_regions
                    ]
                    evidence['matched_rejected_class_regions'] = [
                        {'label': 'cash', 'box_xyxy': box}
                        for box in shifted_negative_boxes
                    ]
                def replaces(item):
                    label = canonical_label(item.get('label', ''))
                    if reference.label == 'cash':
                        return label in CASH_REFERENCE_CONFLICTS and _same_cash_object(item.get('box', []), matched_box)
                    if reference.label == 'bag':
                        candidate_box = item.get('box', [])
                        if (label in {'bag', 'cash', 'food', 'drink'}
                                and _same_cash_object(candidate_box, matched_box)):
                            return True
                        return label == 'cash' and any(
                            _iou(candidate_box, negative_box) >= 0.45
                            for negative_box in shifted_negative_boxes)
                    # Preserve the existing chocolate correction scope.
                    return label in {'cash', 'food'} and _iou(item.get('box', []), matched_box) > 0.45

                suppressed = [item for item in result if replaces(item)]
                result = [item for item in result if not replaces(item)]
                result.append({'label': reference.label, 'box': matched_box, **evidence})
                suppressed_labels = [canonical_label(item.get('label', '')) for item in suppressed]
                notes.append({'action': 'apply_human_confirmed_reference' if reference.label == 'cash' else 'apply_human_corrected_reference',
                              'box': matched_box,
                              'suppressed_cash_count': suppressed_labels.count('cash'),
                              'suppressed_conflicting_count': len(suppressed),
                              'suppressed_conflicting_labels': sorted(set(suppressed_labels)), **evidence})
            except (ValueError, TypeError, cv2.error):
                # Missing or malformed optional evidence must not stop analysis.
                continue
        return result, notes
