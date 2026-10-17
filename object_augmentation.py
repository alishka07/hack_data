"""Label-preserving, CPU-only augmentation for the local banking detector.

All image coordinates are pixel-edge XYXY coordinates (right/bottom may equal
the image width/height). The public entry point needs only Pillow, NumPy and
OpenCV; it does not load a model or touch source files.

Bag annotations alone are boxes, not segmentation masks. Without a separate
reviewed foreground mask, appearance changes affect the WHOLE image. Recoloring
a rectangular bag crop would also recolor a hand, banknotes and the background.
An optional reviewed mask enables foreground-only bag recoloring. Geometry
moves/scales the complete scene instead of pasting floating bags.

Use this only for the training split. Validation frames must stay original.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


VERSION = "1.1"
PROFILES = ("none", "standard", "bag_focus")
MIN_VISIBILITY = 0.60
MIN_BOX_SIDE = 4.0


def describe_profile(profile: str = "bag_focus") -> dict:
    """Return JSON-safe settings/provenance suitable for a training report."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown augmentation profile {profile!r}; choose {PROFILES}")
    return {
        "version": VERSION,
        "profile": profile,
        "enabled": profile != "none",
        "training_only": True,
        "horizontal_flip_probability": 0.5,
        "affine_probability": 0.85,
        "ordinary_scale_range": [0.88, 1.12],
        "bag_frame_scale_range": [0.76, 1.24] if profile == "bag_focus" else [0.88, 1.12],
        "max_translation_fraction": 0.10 if profile == "bag_focus" else 0.07,
        "max_rotation_degrees": 5.0,
        "minimum_box_visibility": MIN_VISIBILITY,
        "minimum_transformed_box_side_pixels": MIN_BOX_SIDE,
        "maximum_geometry_attempts": 8,
        "bag_frame_hue_turn_range": [-0.18, 0.18] if profile == "bag_focus" else [-0.04, 0.04],
        "bag_frame_saturation_range": [0.40, 1.70] if profile == "bag_focus" else [0.80, 1.20],
        "degradations": ["mild_gaussian_blur", "jpeg_compression", "sensor_noise", "reduced_resolution"],
        "appearance_scope": "whole_image unless a reviewed foreground bag mask is supplied; masked bag recoloring plus mild whole-image appearance changes when present",
        "optional_bag_mask": "Reviewed foreground only, same image dimensions; constrained to annotated bag boxes. A bounding-box rectangle is not a valid foreground segmentation.",
        "masked_bag_hue_turn_range": [-0.42, 0.42] if profile == "bag_focus" else [-0.12, 0.12],
        "masked_bag_saturation_range": [0.30, 1.80] if profile == "bag_focus" else [0.75, 1.25],
        "geometry_scope": "whole_image; no copy-paste, mosaic or erased objects",
        "border_mode": "constant edge median; no reflected or duplicated objects",
        "box_policy": "Reject the entire geometry proposal if any annotation becomes too small or loses over 40% of its area; never drop labels.",
    }


def _validated_boxes(boxes: Sequence[Sequence[float]], size: tuple[int, int]) -> list[list[float]]:
    width, height = size
    validated = []
    for raw in boxes:
        if len(raw) != 4:
            raise ValueError("Every box must contain four pixel XYXY coordinates")
        box = list(map(float, raw))
        if not all(math.isfinite(v) for v in box):
            raise ValueError("Box coordinates must be finite")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Input box {box} is empty or outside image size {size}")
        validated.append(box)
    return validated


def horizontal_flip_boxes(boxes: Sequence[Sequence[float]], width: int) -> list[list[float]]:
    """Match PIL ImageOps.mirror using pixel-edge, rather than pixel-center, boxes."""
    return [[float(width - x2), float(y1), float(width - x1), float(y2)]
            for x1, y1, x2, y2 in boxes]


def affine_matrix(width: int, height: int, *, scale: float = 1.0,
                  angle_degrees: float = 0.0, translate_x: float = 0.0,
                  translate_y: float = 0.0) -> np.ndarray:
    """Create a forward affine matrix in pixel-edge coordinates around image center."""
    if not all(math.isfinite(v) for v in (scale, angle_degrees, translate_x, translate_y)) or scale <= 0:
        raise ValueError("Affine arguments must be finite and scale must be positive")
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, scale)
    matrix[:, 2] += [translate_x, translate_y]
    return matrix


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1]
                   - points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points)))) / 2.0


def _clip_polygon(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    """Sutherland-Hodgman clipping: visibility uses rotated area, not inflated AABB."""
    for axis, boundary, keep_greater in ((0, 0.0, True), (0, float(width), False),
                                         (1, 0.0, True), (1, float(height), False)):
        if not points:
            break
        output = []
        previous = points[-1]
        previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        for current in points:
            inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if inside != previous_inside:
                delta = current[axis] - previous[axis]
                ratio = (boundary - previous[axis]) / delta
                intersection = [previous[0] + ratio * (current[0] - previous[0]),
                                previous[1] + ratio * (current[1] - previous[1])]
                intersection[axis] = boundary
                output.append(intersection)
            if inside:
                output.append(current)
            previous, previous_inside = current, inside
        points = output
    return points


def transform_boxes(boxes: Sequence[Sequence[float]], matrix: np.ndarray,
                    size: tuple[int, int], *, min_visibility: float = MIN_VISIBILITY,
                    min_side: float = MIN_BOX_SIDE) -> tuple[list[list[float]] | None, dict]:
    """Transform and clip every box, or reject the entire proposal.

    No box is silently removed: this prevents keeping visible object remnants
    without annotations after a crop. Rotation uses the visible polygon area.
    """
    width, height = size
    source = _validated_boxes(boxes, size)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 2x3 forward affine matrix")
    if not 0 < min_visibility <= 1 or min_side <= 0:
        raise ValueError("Invalid visibility or minimum-side threshold")
    output, visibility = [], []
    for index, (x1, y1, x2, y2) in enumerate(source):
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64)
        polygon = (corners @ matrix.T).tolist()
        area = _polygon_area(polygon)
        clipped = _clip_polygon(polygon, width, height)
        fraction = _polygon_area(clipped) / area if area > 1e-12 else 0.0
        visibility.append(float(min(1.0, max(0.0, fraction))))
        if not clipped or fraction + 1e-9 < min_visibility:
            return None, {"accepted": False, "reason": "box_visibility", "box_index": index,
                          "visibility": visibility}
        array = np.asarray(clipped)
        box = [float(max(0, array[:, 0].min())), float(max(0, array[:, 1].min())),
               float(min(width, array[:, 0].max())), float(min(height, array[:, 1].max()))]
        if min(box[2] - box[0], box[3] - box[1]) + 1e-9 < min_side:
            return None, {"accepted": False, "reason": "box_too_small", "box_index": index,
                          "visibility": visibility}
        output.append(box)
    return output, {"accepted": True, "visibility": visibility}


def _border_color(image: np.ndarray) -> tuple[int, int, int]:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)
    return tuple(int(v) for v in np.median(border, axis=0))


def apply_affine(image: Image.Image, boxes: Sequence[Sequence[float]], matrix: np.ndarray,
                 *, min_visibility: float = MIN_VISIBILITY,
                 min_side: float = MIN_BOX_SIDE) -> tuple[Image.Image, list[list[float]], dict]:
    """Apply safe geometry; rejected proposals return a copy of the original sample."""
    source = image.convert("RGB")
    source_boxes = _validated_boxes(boxes, source.size)
    transformed, metadata = transform_boxes(source_boxes, matrix, source.size,
                                            min_visibility=min_visibility, min_side=min_side)
    metadata["matrix"] = np.asarray(matrix, dtype=np.float64).tolist()
    if transformed is None:
        return source.copy(), source_boxes, metadata
    array = np.asarray(source)
    # OpenCV operates on pixel centers; annotations use edges. Conjugating the
    # matrix by a half-pixel translation keeps raster pixels and boxes aligned.
    raster_matrix = np.asarray(matrix, dtype=np.float64).copy()
    raster_matrix[:, 2] += raster_matrix[:, :2] @ np.array([0.5, 0.5]) - 0.5
    fill = _border_color(array)
    result = cv2.warpAffine(array, raster_matrix, source.size, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    metadata["border_rgb"] = list(fill)
    return Image.fromarray(result), transformed, metadata


def _adjust_hue(image: Image.Image, turns: float) -> Image.Image:
    hsv = np.asarray(image.convert("HSV")).copy()
    hue = hsv[:, :, 0].astype(np.int16)
    hsv[:, :, 0] = ((hue + round(turns * 256)) % 256).astype(np.uint8)
    return Image.frombytes("HSV", image.size, hsv.tobytes()).convert("RGB")


def _validated_mask(mask: Image.Image | np.ndarray, size: tuple[int, int]) -> Image.Image:
    if isinstance(mask, Image.Image):
        if mask.mode not in ("L", "1"):
            raise ValueError("A bag mask must be a grayscale/binary foreground mask")
        if mask.size != size:
            raise ValueError("Bag mask dimensions must exactly match the source image")
        result = mask.convert("L")
    else:
        pixels = np.asarray(mask)
        if pixels.shape != (size[1], size[0]) or pixels.dtype != np.uint8:
            raise ValueError("A NumPy bag mask must have shape (height, width) and uint8 values 0..255")
        result = Image.fromarray(pixels)
    if result.getbbox() is None:
        raise ValueError("Bag mask contains no foreground pixels")
    return result


def recolor_masked_bag(image: Image.Image, bag_mask: Image.Image | np.ndarray,
                       *, hue_turns: float, saturation: float = 1.0,
                       brightness: float = 1.0) -> Image.Image:
    """Recolor reviewed foreground only; all zero-mask pixels remain bit-exact.

    This helper does not create or validate semantic segmentations. Callers must
    supply a reviewed foreground mask which excludes hands and background.
    """
    if not all(math.isfinite(v) for v in (hue_turns, saturation, brightness)) or min(saturation, brightness) < 0:
        raise ValueError("Color parameters must be finite and factors nonnegative")
    source = image.convert("RGB")
    mask = _validated_mask(bag_mask, source.size)
    changed = ImageEnhance.Color(source).enhance(saturation)
    changed = _adjust_hue(changed, hue_turns)
    changed = ImageEnhance.Brightness(changed).enhance(brightness)
    return Image.composite(changed, source, mask)


def augment_sample(image: Image.Image, boxes_xyxy: Sequence[Sequence[float]],
                   labels: Sequence[int], bag_label: int | None = None,
                   rng=random, profile: str = "bag_focus", *,
                   bag_mask: Image.Image | np.ndarray | None = None) -> tuple[Image.Image, list[list[float]], list[int], dict]:
    """Return an augmented RGB image, pixel XYXY boxes, unchanged labels and audit data.

    Pass a seeded ``random.Random`` for deterministic previews/tests. Training
    may pass the seeded global ``random`` module. No NumPy global RNG is used.
    ``bag_focus`` is stronger only on frames that contain the supplied bag label.
    With no reviewed foreground mask this changes the entire scene, not just the
    bag rectangle. With ``bag_mask`` the bag also receives targeted recoloring
    before geometry; other objects still receive mild camera/color variation.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown augmentation profile {profile!r}; choose {PROFILES}")
    result = image.convert("RGB").copy()
    boxes = _validated_boxes(boxes_xyxy, result.size)
    labels = list(labels)
    if len(boxes) != len(labels):
        raise ValueError("The number of boxes and labels must match")
    bag_frame = bag_label is not None and bag_label in labels
    stronger = profile == "bag_focus" and bag_frame
    metadata = {"version": VERSION, "profile": profile, "bag_frame": bag_frame,
                "appearance_scope": "whole_image", "transforms": [],
                "input_boxes": len(boxes), "output_boxes": len(boxes),
                "bag_mask_supplied": bag_mask is not None, "bag_mask_applied": False}
    if profile == "none":
        return result, boxes, labels, metadata
    width, height = result.size
    if bag_mask is not None:
        if not bag_frame:
            raise ValueError("A bag mask requires at least one annotation with bag_label")
        mask = _validated_mask(bag_mask, result.size)
        allowed = np.zeros((height, width), dtype=bool)
        for box, label in zip(boxes, labels):
            if label == bag_label:
                x1, y1, x2, y2 = box
                allowed[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)] = True
        pixels = np.asarray(mask)
        if np.any((pixels > 0) & ~allowed):
            raise ValueError("Bag-mask foreground extends outside annotated bag boxes")
        metadata["bag_mask_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        metadata["bag_mask_pixels"] = int(np.count_nonzero(pixels))
        metadata["appearance_scope"] = "masked_bag_plus_mild_whole_image"
        if rng.random() < 0.90:
            hue = rng.uniform(-0.42, 0.42) if stronger else rng.uniform(-0.12, 0.12)
            saturation = rng.uniform(0.30, 1.80) if stronger else rng.uniform(0.75, 1.25)
            brightness = rng.uniform(0.78, 1.22)
            result = recolor_masked_bag(result, mask, hue_turns=hue, saturation=saturation,
                                       brightness=brightness)
            metadata["bag_mask_applied"] = True
            metadata["transforms"].append({"name": "masked_bag_color", "hue_turns": hue,
                                           "saturation": saturation, "brightness": brightness,
                                           "scope": "reviewed_bag_foreground_only"})
    if rng.random() < 0.5:
        result = ImageOps.mirror(result)
        boxes = horizontal_flip_boxes(boxes, width)
        metadata["transforms"].append({"name": "horizontal_flip"})
    if rng.random() < 0.85:
        attempts = []
        for _ in range(8):
            scale = rng.uniform(*( (0.76, 1.24) if stronger else (0.88, 1.12) ))
            angle = rng.uniform(-5.0, 5.0) if rng.random() < 0.4 else 0.0
            fraction = 0.10 if stronger else 0.07
            tx, ty = rng.uniform(-fraction, fraction) * width, rng.uniform(-fraction, fraction) * height
            matrix = affine_matrix(width, height, scale=scale, angle_degrees=angle,
                                   translate_x=tx, translate_y=ty)
            proposal, transformed, detail = apply_affine(result, boxes, matrix)
            attempts.append(detail)
            if detail["accepted"]:
                result, boxes = proposal, transformed
                metadata["transforms"].append({"name": "affine", "scale": scale,
                                               "angle_degrees": angle, "translate_pixels": [tx, ty],
                                               **detail})
                break
        metadata["geometry_attempts"] = len(attempts)
        metadata["geometry_rejections"] = [item for item in attempts if not item["accepted"]]
    if rng.random() < 0.90:
        stronger_global = stronger and bag_mask is None
        brightness = rng.uniform(0.78, 1.24) if stronger_global else rng.uniform(0.87, 1.15)
        contrast = rng.uniform(0.80, 1.25) if stronger_global else rng.uniform(0.87, 1.15)
        saturation = rng.uniform(0.40, 1.70) if stronger_global else rng.uniform(0.80, 1.20)
        hue = rng.uniform(-0.18, 0.18) if stronger_global else rng.uniform(-0.04, 0.04)
        result = ImageEnhance.Brightness(result).enhance(brightness)
        result = ImageEnhance.Contrast(result).enhance(contrast)
        result = ImageEnhance.Color(result).enhance(saturation)
        result = _adjust_hue(result, hue)
        metadata["transforms"].append({"name": "color", "brightness": brightness,
                                       "contrast": contrast, "saturation": saturation, "hue_turns": hue,
                                       "scope": "whole_image"})
    if rng.random() < (0.08 if stronger else 0.03):
        result = ImageOps.grayscale(result).convert("RGB")
        metadata["transforms"].append({"name": "grayscale"})
    # One mild camera degradation per sample: avoid stacking several severe
    # distortions which could erase already small counter/bag details.
    degradation = rng.random()
    if degradation < 0.15:
        radius = rng.uniform(0.25, 0.90)
        result = result.filter(ImageFilter.GaussianBlur(radius))
        metadata["transforms"].append({"name": "gaussian_blur", "radius": radius})
    elif degradation < 0.35:
        quality = rng.randint(50, 92)
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            result = compressed.convert("RGB")
        metadata["transforms"].append({"name": "jpeg_compression", "quality": quality})
    elif degradation < 0.50:
        sigma = rng.uniform(1.0, 5.0)
        noise_seed = rng.getrandbits(64)
        pixels = np.asarray(result).astype(np.float32)
        noise = np.random.default_rng(noise_seed).normal(0, sigma, pixels.shape)
        result = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
        metadata["transforms"].append({"name": "sensor_noise", "sigma": sigma, "seed": noise_seed})
    elif degradation < 0.65:
        factor = rng.uniform(0.68, 0.92)
        small = (max(1, round(width * factor)), max(1, round(height * factor)))
        result = result.resize(small, Image.Resampling.BILINEAR).resize((width, height), Image.Resampling.BILINEAR)
        metadata["transforms"].append({"name": "reduced_resolution", "factor": factor})
    return result, boxes, labels, metadata
