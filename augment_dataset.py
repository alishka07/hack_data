"""Export reproducible, box-aware training augmentation; validation stays original.

CPU only. Source images and annotations are never modified. The destination must
be empty: each export is a separate dataset that can be reviewed before use.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
COLORS = {"safe": "#ffb547", "banknote_counter": "#35c9ee", "laptop": "#b59aff", "bag": "#51e397"}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_path(root: Path, name: str) -> Path:
    result = (root / name).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Image path escapes dataset: {name}")
    return result


def validate_split(root: Path, filename: str):
    path = root / filename
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    images = {item["id"]: item for item in data["images"]}
    categories = {item["id"]: item["name"] for item in data["categories"]}
    if not images or len(images) != len(data["images"]):
        raise ValueError(f"Empty images or duplicate image IDs in {filename}")
    if not categories or len(categories) != len(data["categories"]):
        raise ValueError(f"Empty categories or duplicate category IDs in {filename}")
    if len({item["file_name"] for item in images.values()}) != len(images):
        raise ValueError(f"Duplicate filenames in {filename}")
    annotations = defaultdict(list)
    seen_ann = set()
    hashes = {}
    for image in images.values():
        source = image_path(root, image["file_name"])
        with Image.open(source) as opened:
            if opened.size != (image["width"], image["height"]):
                raise ValueError(f"Image dimensions disagree: {source}")
            opened.verify()
        hashes[image["file_name"]] = digest(source)
    for ann in data["annotations"]:
        if ann["id"] in seen_ann or ann["image_id"] not in images or ann["category_id"] not in categories:
            raise ValueError(f"Invalid annotation ID/image/category in {filename}: {ann}")
        seen_ann.add(ann["id"])
        if ann.get("segmentation") or ann.get("keypoints") or ann.get("iscrowd", 0):
            raise ValueError("Exporter supports bounding boxes only, without masks, keypoints or crowds")
        x, y, width, height = map(float, ann["bbox"])
        image = images[ann["image_id"]]
        if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid bounding box: {ann['id']}")
        if x < 0 or y < 0 or x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6:
            raise ValueError(f"Bounding box outside image: {ann['id']}")
        annotations[ann["image_id"]].append(ann)
    return data, images, categories, annotations, hashes


def source_checks(source: Path):
    train = validate_split(source, "train.json")
    val = validate_split(source, "val.json")
    if train[2] != val[2]:
        raise ValueError("Train and validation categories differ")
    if set(train[4]) & set(val[4]):
        raise ValueError("Train and validation share image filenames")
    if set(train[4].values()) & set(val[4].values()):
        raise ValueError("Train and validation share identical image file contents")
    for split in (train, val):
        if any(not image.get("source_video") for image in split[1].values()):
            raise ValueError("source_video is required for every image to verify video-separated splits")
    train_videos = {item["source_video"] for item in train[1].values()}
    val_videos = {item["source_video"] for item in val[1].values()}
    if train_videos & val_videos:
        raise ValueError(f"Train and validation share videos: {sorted(train_videos & val_videos)}")
    return train, val, {
        "source_video_separated": True,
        "identical_image_files_shared": False,
        "train_videos": sorted(train_videos),
        "validation_videos": sorted(val_videos),
    }


def sample_seed(seed: int, image_id: int, variant: int) -> int:
    payload = f"augmentation-v1:{seed}:{image_id}:{variant}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def font(size: int):
    for path in (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def create_preview(output: Path, train: dict, originals: list, variants: dict, categories: dict, max_sources: int):
    ann_by_image = defaultdict(list)
    for ann in train["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)
    # Show visible bags from different videos rather than several tiny bags.
    def bag_fraction(item):
        return max((ann["area"] / (item["width"] * item["height"])
                    for ann in ann_by_image[item["id"]] if categories[ann["category_id"]] == "bag"), default=0)
    originals = sorted(originals, key=lambda item: (-bag_fraction(item), item["id"]))
    selected, seen = [], set()
    for item in originals:
        video = item.get("source_video")
        if video not in seen and len(selected) < max_sources:
            selected.append(item)
            seen.add(video)
    for item in originals:
        if item not in selected and len(selected) < max_sources:
            selected.append(item)
    if not selected:
        return None
    columns = min(7, max(1 + len(variants[item["id"]]) for item in selected))
    cell_w, cell_h, top = 226, 472, 72
    sheet = Image.new("RGB", (columns * cell_w + 24, len(selected) * cell_h + top + 24), "#eef2f6")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), "Аугментация обучающей выборки: исходник и варианты", font=font(24), fill="#202d43")
    draw.text((16, 44), "Рамки изменяются вместе с изображением. Проверочная выборка не менялась.", font=font(16), fill="#4b596b")
    gallery_rows = []
    for row_index, original in enumerate(selected):
        items = [original] + variants[original["id"]][:columns - 1]
        gallery_cells = []
        for column, item in enumerate(items):
            im = Image.open(output / item["file_name"]).convert("RGB")
            im.thumbnail((cell_w - 12, cell_h - 78), Image.Resampling.LANCZOS)
            scale_x, scale_y = im.width / item["width"], im.height / item["height"]
            local = ImageDraw.Draw(im)
            for ann in ann_by_image[item["id"]]:
                x, y, w, h = ann["bbox"]
                label = categories[ann["category_id"]]
                box = [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y]
                local.rectangle(box, outline=COLORS.get(label, "white"), width=2)
                display_label = "counter" if label == "banknote_counter" else label
                local.text((box[0] + 2, max(0, box[1] - 15)), display_label, font=font(12),
                           fill=COLORS.get(label, "white"), stroke_width=1, stroke_fill="#10201a")
            left, upper = 12 + column * cell_w, top + row_index * cell_h
            draw.rounded_rectangle((left, upper, left + cell_w - 8, upper + cell_h - 10), 8, fill="white")
            sheet.paste(im, (left + 4 + (cell_w - 12 - im.width) // 2, upper + 35))
            title = "Исходник" if column == 0 else f"Вариант {column}"
            draw.text((left + 8, upper + 8), title, font=font(17), fill="#202d43")
            draw.text((left + 8, upper + cell_h - 34), str(original.get("source_video", "")), font=font(14), fill="#4b596b")
            file_ref = html.escape(item["file_name"], quote=True)
            metadata = item.get("augmentation", {})
            details = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))
            gallery_cells.append(f'<article><h3>{html.escape(title)}</h3><a href="{file_ref}"><img src="{file_ref}" loading="lazy" alt="{html.escape(title)}"></a><details><summary>Преобразования</summary><pre>{details}</pre></details></article>')
        gallery_rows.append(f'<section><h2>{html.escape(str(original.get("source_video", "")))} · {html.escape(str(original.get("timestamp_sec", "")))} с</h2><div class="row">{"".join(gallery_cells)}</div></section>')
    sheet.save(output / "preview.jpg", quality=94, subsampling=0)
    (output / "gallery.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Аугментация обучающей выборки</title><style>body{font:16px Segoe UI,Arial,sans-serif;background:#eef2f6;color:#202d43;margin:24px}h1{font-size:28px}.row{display:flex;gap:12px;overflow-x:auto}article{background:white;padding:12px;min-width:210px;width:210px;border-radius:8px}img{max-width:100%;max-height:390px}pre{font-size:12px;white-space:pre-wrap;word-break:break-word}h3{margin:0 0 12px}summary{cursor:pointer}a{color:#9b214b}</style><h1>Аугментация обучающей выборки</h1><p>Исходники и варианты из train. Вся проверочная выборка сохранена без изменений. Общие изменения цвета применяются ко всему кадру; адресное перекрашивание ткани дополнительно используется только с проверенной маской мешка.</p><p><a href="preview.jpg">Обзор с рамками объектов</a> · <a href="manifest.json">Происхождение и параметры</a></p>' + "".join(gallery_rows) + "</html>", encoding="utf-8")
    return {"image": "preview.jpg", "gallery": "gallery.html", "source_images": len(selected)}


def export(args):
    from object_augmentation import augment_sample, describe_profile

    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Output must be separate from the source dataset and its parent directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Destination is not empty. Choose a new --output directory: {output}")
    train, val, split_checks = source_checks(source)
    bag_ids = [key for key, name in train[2].items() if name.casefold() in {"bag", "bags"}]
    if len(bag_ids) != 1:
        raise ValueError("Dataset must have exactly one category named bag")
    bag_id = bag_ids[0]
    train_data, train_images, categories, annotations, train_hashes = train
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "export_status.json", {"status": "in_progress", "seed": args.seed})
    for original in list(train_images.values()) + list(val[1].values()):
        destination = image_path(output, original["file_name"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path(source, original["file_name"]), destination)
    shutil.copyfile(source / "train.json", output / "train_original.json")
    shutil.copyfile(source / "val.json", output / "val.json")
    if (source / "manifest.json").is_file():
        shutil.copyfile(source / "manifest.json", output / "source_manifest.json")
    augmented = copy.deepcopy(train_data)
    augmented.setdefault("info", {})["augmentation"] = {
        "profile": args.profile, "seed": args.seed, "source": "train_original.json",
        "validation_unchanged": True, "purpose": "reusable static augmentation; do not stack strong online augmentation",
    }
    next_image_id = max(list(train_images) + list(val[1])) + 1
    next_annotation_id = max([ann["id"] for ann in train_data["annotations"] + val[0]["annotations"]], default=0) + 1
    variants = defaultdict(list)
    provenance = []
    bag_sources = []
    mask_manifest = args.bag_masks.resolve() if args.bag_masks else None
    mask_manifest_exists = mask_manifest is not None and mask_manifest.is_file()
    mask_sources = []
    mask_hashes = {}
    if mask_manifest_exists:
        from bag_mask_io import load_bag_mask
    for original in train_data["images"]:
        source_annotations = annotations[original["id"]]
        boxes = [[ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]] for ann in source_annotations]
        labels = [ann["category_id"] for ann in source_annotations]
        has_bag = bag_id in labels
        if has_bag:
            bag_sources.append(original)
        count = args.bag_variants if has_bag else args.other_variants
        with Image.open(image_path(source, original["file_name"])) as opened:
            original_pixels = opened.convert("RGB")
        bag_mask = load_bag_mask(mask_manifest, original, source) if mask_manifest_exists and has_bag else None
        if bag_mask is not None:
            mask_sources.append(original["id"])
            mask_hashes[str(original["id"])] = hashlib.sha256(bag_mask.tobytes()).hexdigest()
        for variant in range(1, count + 1):
            seed = sample_seed(args.seed, original["id"], variant)
            kwargs = {"profile": args.profile}
            if bag_mask is not None:
                kwargs["bag_mask"] = bag_mask.copy()
            transformed, new_boxes, new_labels, metadata = augment_sample(original_pixels.copy(), copy.deepcopy(boxes), list(labels), bag_id, random.Random(seed), **kwargs)
            if list(new_labels) != labels or len(new_boxes) != len(boxes):
                raise ValueError(f"Augmentation removed/reordered annotations for image {original['id']}")
            if transformed.mode != "RGB":
                transformed = transformed.convert("RGB")
            w, h = transformed.size
            for box in new_boxes:
                if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                    raise ValueError("Augmentation returned an invalid box")
                x1, y1, x2, y2 = map(float, box)
                if not (0 <= x1 < x2 <= w + 1e-6 and 0 <= y1 < y2 <= h + 1e-6):
                    raise ValueError(f"Augmentation returned a box outside the image: {box}")
            relative = f"images/aug_{original['id']:06d}_{variant:02d}.jpg"
            destination = image_path(output, relative)
            if destination.exists():
                raise ValueError(f"Generated filename would overwrite an original image: {relative}")
            transformed.save(destination, quality=95, subsampling=0)
            record = copy.deepcopy(original)
            record.update({"id": next_image_id, "file_name": relative, "width": w, "height": h,
                           "augmentation": {"source_image_id": original["id"], "source_file": original["file_name"],
                                            "source_video": original["source_video"], "variant": variant, "seed": seed,
                                            "profile": args.profile, "transforms": metadata}})
            augmented["images"].append(record)
            variants[original["id"]].append(record)
            for source_ann, box, label in zip(source_annotations, new_boxes, new_labels):
                x1, y1, x2, y2 = map(float, box)
                new_ann = copy.deepcopy(source_ann)
                new_ann.update({"id": next_annotation_id, "image_id": next_image_id, "category_id": label,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                                "source_annotation_id": source_ann["id"]})
                augmented["annotations"].append(new_ann)
                next_annotation_id += 1
            provenance.append({"image_id": next_image_id, "file_name": relative, "sha256": digest(destination),
                               **record["augmentation"], "source_image_sha256": train_hashes[original["file_name"]]})
            next_image_id += 1
    write_json(output / "train.json", augmented)
    validation_unchanged = digest(source / "val.json") == digest(output / "val.json")
    if not validation_unchanged:
        raise ValueError("Validation annotations changed unexpectedly")
    output_train, output_val, output_split_checks = source_checks(output)
    if output_val[4] != val[4]:
        raise ValueError("Validation image files changed unexpectedly")
    if any(digest(image_path(output, name)) != sha for name, sha in train_hashes.items()):
        raise ValueError("Original training image files changed unexpectedly")
    counts = Counter(categories[ann["category_id"]] for ann in augmented["annotations"])
    preview = create_preview(output, augmented, bag_sources, variants, categories, args.preview_sources)
    core_path = ROOT / "object_augmentation.py"
    try:
        profile_description = describe_profile(args.profile)
    except TypeError:
        profile_description = describe_profile()
    manifest = {
        "format_version": 1, "status": "completed", "seed": args.seed, "profile": args.profile,
        "profile_description": profile_description,
        "source_annotation_sha256": {name: digest(source / name) for name in ("train.json", "val.json")},
        "source_manifest_sha256": digest(source / "manifest.json") if (source / "manifest.json").is_file() else None,
        "augmentation_code_sha256": digest(core_path), "exporter_code_sha256": digest(Path(__file__)),
        "bag_mask_loader_code_sha256": digest(ROOT / "bag_mask_io.py") if mask_manifest_exists else None,
        "runtime": {"python": sys.version.split()[0], "pillow": importlib.metadata.version("Pillow"),
                    "numpy": importlib.metadata.version("numpy")},
        "output_annotation_sha256": {name: digest(output / name) for name in ("train.json", "val.json")},
        "source_images_sha256": {"train": train[4], "val": val[4]},
        "reviewed_bag_masks": {"manifest_sha256": digest(mask_manifest) if mask_manifest_exists else None,
                               "source_train_image_ids": mask_sources, "mask_pixels_sha256_by_image_id": mask_hashes},
        "counts": {"original_train_images": len(train_images), "original_train_annotations": len(train_data["annotations"]),
                   "source_train_images_with_bag": len(bag_sources), "augmented_images": len(provenance),
                   "source_train_images_with_reviewed_bag_mask": len(mask_sources),
                   "targeted_bag_recolor_variants": sum(bool(item["transforms"].get("bag_mask_applied")) for item in provenance),
                   "total_train_images": len(augmented["images"]), "total_train_annotations": len(augmented["annotations"]),
                   "validation_images": len(val[1]), "validation_annotations": len(val[0]["annotations"]),
                   "training_boxes_by_category": dict(counts)},
        "variants": {"per_bag_image": args.bag_variants, "per_other_image": args.other_variants},
        "split_checks": {**split_checks, "verified_again_after_export": output_split_checks["source_video_separated"],
                         "validation_json_byte_identical": validation_unchanged, "validation_images_byte_identical": True,
                         "original_training_json_byte_identical": digest(output / "train_original.json") == digest(source / "train.json"),
                         "original_training_images_byte_identical": True, "validation_images_used_for_augmentation": False},
        "preview": preview, "provenance": provenance,
        "limitations": ["Synthetic variants do not add independent cameras, people or rooms.",
                        "Color transforms affect the whole image; targeted fabric recoloring additionally applies only where a verified bag mask exists.",
                        "This export changes training data only; it does not train or replace model weights.",
                        "For this static dataset use --augmentation none during training to avoid stacking strong transforms."],
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "export_status.json", {"status": "completed", "seed": args.seed, "counts": manifest["counts"]})
    return {"output": str(output), "manifest": str(output / "manifest.json"), "preview": str(output / "preview.jpg") if preview else None,
            "counts": manifest["counts"], "split_checks": manifest["split_checks"]}


def nonnegative(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "training" / "dataset", help="Original COCO dataset containing train.json, val.json and images/")
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "dataset_augmented", help="New empty directory; existing exports are never overwritten")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--profile", choices=["bag_focus"], default="bag_focus")
    parser.add_argument("--bag-masks", type=Path, default=ROOT / "training" / "bag_color_masks" / "manifest.json", help="Optional reviewed bag-mask manifest; missing default means global appearance transforms only")
    parser.add_argument("--bag-variants", type=nonnegative, default=6, help="Extra variants of each training image with a bag")
    parser.add_argument("--other-variants", type=nonnegative, default=2, help="Extra variants of other training images")
    parser.add_argument("--preview-sources", type=nonnegative, default=3, help="Training bag examples in the contact sheet and gallery")
    args = parser.parse_args()
    try:
        report = export(args)
    except (ValueError, OSError, ImportError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
