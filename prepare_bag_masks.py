"""Build conservative TRAIN-only bag-fabric masks for color augmentation.

These are partial color masks, not whole-object segmentation or copy-paste
masks. Acceptance requires a separate visual review decision tied to both
source-image and generated-mask hashes. Validation is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
METHOD = "conservative_green_hsv_grabcut_v2"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_mask(rgb, boxes):
    """Return only confidently green fabric inside labelled bag rectangles."""
    height, width = rgb.shape[:2]
    result = np.zeros((height, width), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Skin, pink sleeves, wood, white paper and black handles are excluded.
    color = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 85)
    color &= (hsv[:, :, 1] >= 24) & (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] <= 242)
    for x, y, box_width, box_height in boxes:
        left, top = max(0, int(np.ceil(x))), max(0, int(np.ceil(y)))
        right, bottom = min(width, int(np.floor(x + box_width))), min(height, int(np.floor(y + box_height)))
        if right - left < 5 or bottom - top < 5:
            continue
        context = 8
        cx1, cy1 = max(0, left - context), max(0, top - context)
        cx2, cy2 = min(width, right + context), min(height, bottom + context)
        crop = bgr[cy1:cy2, cx1:cx2]
        interior = np.zeros(crop.shape[:2], dtype=np.uint8)
        interior[top - cy1 + 1:bottom - cy1 - 1, left - cx1 + 1:right - cx1 - 1] = 1
        color_candidate = (color[cy1:cy2, cx1:cx2] & (interior != 0)).astype(np.uint8)
        color_candidate = cv2.morphologyEx(color_candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        seeds = cv2.erode(color_candidate, np.ones((3, 3), np.uint8))
        if int(seeds.sum()) < 8:
            continue
        gc_mask = np.full(crop.shape[:2], cv2.GC_BGD, dtype=np.uint8)
        gc_mask[interior != 0] = cv2.GC_PR_BGD
        gc_mask[color_candidate != 0] = cv2.GC_PR_FGD
        gc_mask[seeds != 0] = cv2.GC_FGD
        background = np.zeros((1, 65), dtype=np.float64)
        foreground = np.zeros((1, 65), dtype=np.float64)
        cv2.setRNGSeed(20261017)
        cv2.grabCut(crop, gc_mask, None, background, foreground, 2, cv2.GC_INIT_WITH_MASK)
        foreground_mask = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)) & (color_candidate != 0)
        # Keep an extra pixel away from boundaries: conservative recoloring is
        # preferable to moving the hue of fingers or nearby equipment.
        foreground_mask = cv2.erode(foreground_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
        clean = np.zeros_like(foreground_mask)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= max(10, largest * 0.05):
                clean[labels == label] = 255
        result[cy1:cy2, cx1:cx2] = np.maximum(result[cy1:cy2, cx1:cx2], clean)
    return result


def recolor_preview(rgb, mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    changed = hsv.copy()
    selected = mask > 0
    changed[selected, 0] = 125
    changed[selected, 1] = np.maximum(changed[selected, 1], 105)
    recolored = cv2.cvtColor(changed, cv2.COLOR_HSV2RGB)
    # Preserve all unmasked pixels exactly, including round-trip rounding.
    output = rgb.copy()
    output[selected] = recolored[selected]
    return output


def make_tile(rgb, mask, boxes, title):
    x1 = max(0, int(min(b[0] for b in boxes)) - 12)
    y1 = max(0, int(min(b[1] for b in boxes)) - 12)
    x2 = min(rgb.shape[1], int(max(b[0] + b[2] for b in boxes)) + 12)
    y2 = min(rgb.shape[0], int(max(b[1] + b[3] for b in boxes)) + 12)
    overlay = rgb.copy()
    selected = mask > 0
    overlay[selected] = (rgb[selected].astype(np.float32) * 0.25 + np.array([255, 20, 200]) * 0.75).astype(np.uint8)
    preview = recolor_preview(rgb, mask)
    tile = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((8, 5), title, fill="black")
    for index, (array, label) in enumerate([(rgb, "original"), (overlay, "mask overlay"), (preview, "blue recolor")]):
        crop = Image.fromarray(array[y1:y2, x1:x2])
        ratio = min(190 / crop.width, 190 / crop.height)
        crop = crop.resize((max(1, round(crop.width * ratio)), max(1, round(crop.height * ratio))), Image.Resampling.NEAREST)
        tile.paste(crop, (index * 200 + (200 - crop.width) // 2, 32 + (190 - crop.height) // 2))
        draw.text((index * 200 + 6, 224), label, fill="black")
    return tile


def prepare(output):
    cv2.setNumThreads(1)
    images_root = ROOT / "training/dataset"
    annotation_path = images_root / "train.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    bag_ids = {category["id"] for category in data["categories"] if category["name"] == "bag"}
    images = {item["id"]: item for item in data["images"]}
    by_image = {}
    for annotation in data["annotations"]:
        if annotation["category_id"] in bag_ids:
            by_image.setdefault(annotation["image_id"], []).append(annotation["bbox"])
    output.mkdir(parents=True, exist_ok=True)
    decision_path = output / "review_decisions.json"
    decisions = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    rows, tiles = [], []
    for image_id, boxes in by_image.items():
        item = images[image_id]
        path = (images_root / item["file_name"]).resolve()
        if not path.is_relative_to(images_root.resolve()):
            raise ValueError("Training image escapes images_root")
        with Image.open(path) as source:
            rgb = np.array(source.convert("RGB"))
        if rgb.shape[:2] != (item["height"], item["width"]):
            raise ValueError(f"Image dimensions mismatch: {path}")
        # These six bags are plaid/beige; their green/background pixels do not
        # define bag fabric. This exclusion comes from reviewing TRAIN crops.
        plaid = Path(item["file_name"]).stem.startswith("X40_")
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8) if plaid else build_mask(rgb, boxes)
        filename = Path(item["file_name"]).stem + "_color_mask.png"
        mask_path = output / filename
        Image.fromarray(mask).save(mask_path)
        row = {"image_id": image_id, "file_name": item["file_name"], "source_video": item.get("source_video"),
               "source_sha256": sha256(path), "mask_file": filename, "mask_sha256": sha256(mask_path),
               "mask_pixels": int(np.count_nonzero(mask)), "bag_boxes_xywh": boxes,
               "width": item["width"], "height": item["height"],
               "review_status": "excluded", "reason": "Awaiting visual review of mask overlay and recolor preview"}
        if plaid:
            row["reason"] = "Plaid/beige bag excluded: conservative green-fabric segmentation is inappropriate"
        elif row["mask_pixels"] < 30:
            row["reason"] = "Too few confidently segmented fabric pixels"
        else:
            decision = decisions.get(item["file_name"], {})
            if decision.get("source_sha256") == row["source_sha256"] and decision.get("mask_sha256") == row["mask_sha256"]:
                if decision.get("review_status") == "accepted":
                    row.update(review_status="accepted", reason=decision["reason"])
                elif decision.get("review_status") == "excluded":
                    row["reason"] = decision.get("reason", "Excluded after visual review")
        rows.append(row)
        tiles.append(make_tile(rgb, mask, boxes, f"{Path(item['file_name']).name} | pixels={row['mask_pixels']} | {row['review_status']}"))
    for part, start in enumerate(range(0, len(tiles), 8), start=1):
        batch = tiles[start:start + 8]
        sheet = Image.new("RGB", (1200, 240 * ((len(batch) + 1) // 2)), (210, 210, 210))
        for index, tile in enumerate(batch):
            sheet.paste(tile, ((index % 2) * 600, (index // 2) * 240))
        sheet.save(output / f"mask_review_{part:02d}.jpg", quality=95)
    manifest = {
        "method": METHOD, "purpose": "partial_foreground_color_augmentation_only_not_copy_paste",
        "training_only": True, "validation_loaded": False,
        "annotation_file": str(annotation_path), "annotation_sha256": sha256(annotation_path),
        "images_root": str(images_root), "mask_root": str(output.resolve()),
        "review": "Acceptance is tied to both source-image and mask SHA256 after visual inspection",
        "accepted": sum(row["review_status"] == "accepted" for row in rows),
        "excluded": sum(row["review_status"] == "excluded" for row in rows), "images": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "accepted": manifest["accepted"], "excluded": manifest["excluded"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "training/bag_color_masks")
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
