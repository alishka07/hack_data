"""Offline object candidates for recorded videos. Run with the existing global Python."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.mov', '.mp4', '.mkv', '.avi', '.m4v'}
PROMPTS = {
    'equipment': 'a banknote counting machine. a power strip. an electrical outlet. a laptop. a computer monitor. a desktop computer tower. a safe.',
    'bags': 'a plastic bag. a banknote counting machine. a power strip. an office chair.',
    'desk': 'a smartphone. a usb flash drive. a notebook. a paper document. an identity card. a keyboard. a computer mouse.',
    'cash': 'banknotes. money. a paper document.',
    'workspace': 'an office desk. a table. a computer monitor.',
    'food_drink': 'a snack. a paper cup.',
}
VERSION = '1.3'
RENDER_FOOTER = 190


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(8):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2**attempt, .5))


def discover(folder):
    import re
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p: (int(re.search(r'\d+', p.stem).group()) if re.search(r'\d+', p.stem) else 10**9, p.name))


def resolve_video(value, folder=ROOT / 'videos'):
    value, folder = Path(value), Path(folder)
    for candidate in (value, folder / value):
        if candidate.is_file():
            return candidate.resolve()
    # Several supplied files use Cyrillic Х; accept a typed Latin X for filenames.
    if value.parent == Path('.'):
        normalize = lambda name: name.casefold().replace('х', 'x')
        matches = [p for p in discover(folder) if normalize(p.name) == normalize(value.name)]
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            raise ValueError(f'Ambiguous Latin/Cyrillic filename: {value}; use the exact path')
    raise FileNotFoundError(f'Video not found: {value}')


def read_regions(video, width, height):
    path = ROOT / 'scenes' / (video.stem + '.json')
    if not path.exists():
        return [], 'not_calibrated'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('video_name') != video.name or data.get('video_size_bytes') != video.stat().st_size:
        raise ValueError(f'Scene calibration does not match the video: {path}')
    if data.get('image_size') != [width, height]:
        raise ValueError(f'Scene dimensions changed: {path}. Recalibrate this video.')
    return data['regions'], 'manual_regions_present'


def draw_frame(frame, result, timestamp, interval):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    names = {'banknote_counter':'Счётчик банкнот', 'plastic_bag':'Пакет', 'bag':'Мешок/сумка',
             'safe':'Сейф', 'laptop':'Ноутбук', 'computer_monitor':'Экран', 'desktop_computer':'Системный блок',
             'cash':'Банкноты', 'smartphone':'Телефон', 'document':'Документ', 'identity_card':'Удостоверение',
             'food':'Еда/шоколад', 'drink':'Чашка/напиток',
             'keyboard':'Клавиатура', 'mouse':'Мышь', 'notebook':'Блокнот', 'usb_drive':'USB-накопитель',
             'electrical_outlet':'Розетка', 'power_strip':'Блок розеток', 'chair':'Стул', 'work_table':'Стол (гипотеза)'}
    background = {'electrical_outlet', 'power_strip', 'chair'}
    groups = []
    for category in ('review', 'candidates'):
        for row in result[category]:
            if row['label'] in background:
                continue  # These negative classes remain in JSON/CSV and participate in filtering.
            box = row['box']
            group = None
            for existing in groups:
                other = existing['box']
                inter = max(0, min(box[2], other[2])-max(box[0], other[0])) * max(0, min(box[3], other[3])-max(box[1], other[1]))
                union = (box[2]-box[0])*(box[3]-box[1]) + (other[2]-other[0])*(other[3]-other[1]) - inter
                if union > 0 and inter / union >= .55:
                    group = existing
                    break
            if group is None:
                group = {'box':box, 'labels':[], 'review':False, 'reviewed_reference':False}
                groups.append(group)
            group['review'] |= category == 'review'
            group['reviewed_reference'] |= row.get('source') == 'human_confirmed_reference_match'
            for label in [row['label'], *[c['label'] for c in row.get('conflicts', [])]]:
                if label not in group['labels']:
                    group['labels'].append(label)
    height, width = frame.shape[:2]
    canvas = Image.new('RGB', (width, height + RENDER_FOOTER), '#192230')
    canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    draw.text((7, height+8), f'{timestamp:.3f} с · шаг анализа {interval:g} с · кандидаты модели', font=font, fill='white')
    for number, group in enumerate(groups[:8], 1):
        color = '#ffcc55' if group['review'] else '#70e6a2'
        x1, y1, x2, y2 = group['box']
        draw.rectangle(group['box'], outline=color, width=2)
        draw.rectangle((x1, y1, min(width,x1+22), y1+19), fill='#192230')
        draw.text((x1+3,y1+1), str(number), font=font, fill=color)
        label = ' / '.join(names.get(x,x) for x in group['labels'])
        prefix = 'Проверить: ' if group['review'] else 'По уточнению: ' if group.get('reviewed_reference') else ''
        line = f'{number}. {prefix}{label}'
        while draw.textbbox((0,0),line,font=font)[2] > width-16 and len(line)>8:
            line = line[:-4] + '…'
        draw.text((7, height+30+(number-1)*18), line, font=font, fill=color)
    footer = 'Все рамки — в JSON/CSV. Это не заключение о нарушении.'
    draw.text((7, height+174), footer, font=font, fill='#c1cbd8')
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def analyze(video, args, loaded, detect):
    import cv2
    import numpy as np
    from PIL import Image
    from candidate_filters import filter_candidates
    from inference_optimized import PromptDetector, VERSION as CACHE_VERSION
    from reviewed_objects import ReviewedObjectMatcher, DEFAULT_ANNOTATIONS
    prompt_detector = None if getattr(args, 'no_feature_cache', False) else PromptDetector(loaded, detect)
    reviewed_enabled = bool({'cash', 'food_drink'}.intersection(args.groups))
    reviewed_matcher = ReviewedObjectMatcher.for_video(video) if reviewed_enabled else None
    output = args.output / video.stem
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(version=VERSION, video_size=video.stat().st_size, video_mtime=video.stat().st_mtime_ns,
                     interval=args.interval, groups=args.groups, threshold=args.threshold,
                     max_samples=args.max_samples, video_preview=not args.no_video,
                     filter_hash=hashlib.sha256((ROOT / 'candidate_filters.py').read_bytes()).hexdigest(),
                     prompts={k: PROMPTS[k] for k in args.groups})
    signature['inference_engine'] = CACHE_VERSION if prompt_detector and prompt_detector.supported else 'original'
    signature['engine_hash'] = hashlib.sha256((ROOT/'inference_optimized.py').read_bytes()).hexdigest()
    signature['reviewed_objects_hash'] = hashlib.sha256((ROOT/'reviewed_objects.py').read_bytes()).hexdigest()
    signature['reviewed_annotations_hash'] = hashlib.sha256(DEFAULT_ANNOTATIONS.read_bytes()).hexdigest() if DEFAULT_ANNOTATIONS.is_file() and reviewed_enabled else None
    scene_path = ROOT / 'scenes' / f'{video.stem}.json'
    signature['scene_hash'] = hashlib.sha256(scene_path.read_bytes()).hexdigest() if scene_path.exists() else None
    previous = output / 'summary.json'
    if args.resume and previous.exists():
        cached = json.loads(previous.read_text(encoding='utf-8'))
        if cached.get('signature') == signature and cached.get('status') == 'completed':
            print(f'SKIP completed: {video.name}', flush=True)
            return cached
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError('Missing video metadata')
    try:
        regions, calibration = read_regions(video, width, height)
    except Exception:
        cap.release()
        raise
    preview_writer = None
    if not args.no_video:
        import imageio_ffmpeg
        # A clearly labelled sampled review, not the original continuous video.
        preview_writer = imageio_ffmpeg.write_frames(str(output / 'review.mp4'), (width, height + RENDER_FOOTER),
                         fps=max(.1, 1 / args.interval), codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                         macro_block_size=1, ffmpeg_log_level='error',
                         output_params=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart', '-crf', '21'])
        try:
            preview_writer.send(None)
        except Exception:
            cap.release()
            raise
    frames = samples = 0
    next_sample = 0.0
    last_pts = None
    pts_problem = fallback_times = 0
    counts = Counter()
    thumbnails = []
    started = time.perf_counter()
    try:
        with (output / 'frames.jsonl').open('w', encoding='utf-8') as jsonl, (output / 'detections.csv').open('w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp_sec', 'source_frame', 'status', 'label', 'score', 'x1', 'y1', 'x2', 'y2', 'reasons', 'source', 'score_kind', 'reference_frame'])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = frames
                frames += 1
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                timestamp_source = 'video_pts'
                if not np.isfinite(pts) or pts < 0 or (last_pts is not None and pts <= last_pts):
                    pts_problem += 1
                    pts = frame_index / fps
                    timestamp_source = 'frame_index_fps_estimate'
                    fallback_times += 1
                last_pts = pts
                if pts + 1e-6 < next_sample or (args.max_samples and samples >= args.max_samples):
                    continue
                next_sample = pts + args.interval
                raw = []
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                grouped = (prompt_detector.detect_groups(pil_image, {key: PROMPTS[key] for key in args.groups}, args.threshold)
                           if prompt_detector else {key: detect(pil_image, loaded, PROMPTS[key], args.threshold) for key in args.groups})
                for group in args.groups:
                    for row in grouped[group]:
                        raw.append(dict(row, query=group))
                effective, corrections = reviewed_matcher.apply(frame, raw) if reviewed_matcher else (raw, [])
                filtered = filter_candidates(effective, image_size=(width, height), regions=regions)
                record = dict(source_frame=frame_index, timestamp_sec=pts, timestamp_source=timestamp_source,
                              raw_detections=raw, reviewed_object_corrections=corrections, **filtered)
                jsonl.write(json.dumps(record, ensure_ascii=False) + '\n')
                jsonl.flush()
                for status, rows in filtered.items():
                    counts[status] += len(rows)
                    for row in rows:
                        writer.writerow([pts, frame_index, status, row.get('label', ''), row.get('score'), *(row.get('box') or [None]*4), '|'.join(row.get('reasons', [])), row.get('source', 'model'), row.get('score_kind', 'model_score'), row.get('reference_frame_zero_based')])
                annotated = draw_frame(frame, filtered, pts, args.interval)
                if preview_writer is not None:
                    preview_writer.send(np.ascontiguousarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))
                if samples < 6:
                    filename = f'sample_{samples:03d}_{pts:.3f}s.jpg'
                    cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(output / filename))
                    thumbnails.append(dict(file=filename, timestamp_sec=pts))
                samples += 1
                if samples == 1 or samples % 10 == 0:
                    print(f'{video.name}: t={pts:.2f}s; {samples} samples; review={counts["review"]}', flush=True)
    finally:
        cap.release()
        if preview_writer is not None:
            preview_writer.close()
    if frames == 0 or samples == 0:
        raise RuntimeError('No frames / inference samples were produced')
    decode_warning = bool(expected > 0 and frames != expected)
    summary = dict(status='completed_with_warnings' if decode_warning else 'completed', video=str(video), signature=signature,
                   decoded_frames=frames, metadata_frames=expected, frame_count_matches=(frames == expected),
                   last_timestamp_sec=last_pts, source_fps=fps, image_size=[width, height],
                   analyzed_samples=samples, sample_interval_sec=args.interval, timestamp_problems=pts_problem,
                   fallback_timestamp_count=fallback_times, calibration=calibration,
                   processing_seconds=round(time.perf_counter() - started, 2), device=str(loaded[2]),
                   reviewed_object_reference_enabled=reviewed_matcher is not None,
                   reviewed_reference_scope='exact_source_sha256_only',
                   counts=dict(counts), thumbnails=thumbnails,
                   video_preview='review.mp4' if preview_writer is not None else None,
                   verdict='not_evaluated', violations=[],
                   decoding_warning='Decoded frame count differs from container metadata; check the source.' if decode_warning else None,
                   limitations=['Object candidates are not confirmed objects or violations.',
                                'Frames between inference samples are decoded but not analyzed by the model.',
                                'A bag does not imply that it contains money.',
                                'No safe/door state, counting action, identity check or client departure classifier.',
                                'No detection does not prove that an object is absent.'])
    write_json(previous, summary)
    print(f'DONE {video.name}: decoded={frames}, analyzed={samples}, seconds={summary["processing_seconds"]}', flush=True)
    return summary


def make_index(output, rows):
    cards = []
    for row in rows:
        name = Path(row['video']).name
        stem = Path(row['video']).stem
        if row.get('status') not in {'completed', 'completed_with_warnings'}:
            cards.append(f'<article><h2>{html.escape(name)}</h2><p class="error">{html.escape(row.get("error", "error"))}</p></article>')
            continue
        counts = row.get('counts', {})
        pictures = ''.join(f'<a href="{html.escape(stem)}/{html.escape(p["file"])}"><img loading="lazy" src="{html.escape(stem)}/{html.escape(p["file"])}"></a>' for p in row['thumbnails'][:3])
        link = f'<a href="{html.escape(stem)}/review.mp4">Видео по выбранным кадрам</a> · ' if row['video_preview'] else ''
        cards.append(f'<article><h2>{html.escape(name)}</h2><p>Прочитано кадров: {row["decoded_frames"]}. Проанализировано моделью: {row["analyzed_samples"]}. Шаг: {row["sample_interval_sec"]} с.</p><p>Кандидаты: {counts.get("candidates",0)} · Неоднозначно: {counts.get("review",0)} · Отсеяно: {counts.get("suppressed",0)}. Это число рамок, не число предметов.</p><p>{link}<a href="{html.escape(stem)}/detections.csv">Таблица</a> · <a href="{html.escape(stem)}/summary.json">Сводка</a></p>{pictures}</article>')
    page = '''<!doctype html><meta charset="utf-8"><title>Проверка видео</title><style>body{font:16px system-ui;background:#eef2f6;color:#192230;max-width:1100px;margin:30px auto;padding:15px}article{background:white;padding:22px;border-radius:12px;margin:20px 0}img{width:29%;max-height:420px;object-fit:contain;margin:1%}a{color:#155b99}.error{color:#a32626}</style><h1>Объекты на банковских видео</h1><p>Зелёные рамки — кандидаты модели. Жёлтые — неоднозначные объекты для ручной проверки. Скрытые рамки и исходные ответы сохранены в frames.jsonl. Автоматическое заключение о нарушениях не выносится.</p><p>Все кадры декодируются, модель анализирует кадры с указанным шагом. Видеопредпросмотр состоит из этих кадров; плавное исходное видео находится в папке videos. Порог двери по вашему правилу — 2 секунды; распознавание её состояния ещё не подключено.</p>'''
    (output / 'index.html').write_text(page + ''.join(cards), encoding='utf-8')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--video', type=Path, help='A file path, or a filename in videos/')
    selection.add_argument('--all', action='store_true', help='All videos (also the default)')
    parser.add_argument('--videos', type=Path, default=ROOT / 'videos')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'objects')
    parser.add_argument('--interval', type=float, default=1.0, help='Seconds between object inference samples; default 1')
    parser.add_argument('--threshold', type=float, default=.28)
    parser.add_argument('--groups', nargs='+', choices=list(PROMPTS), default=['equipment', 'bags'])
    parser.add_argument('--max-samples', type=int, default=0, help='Test only the first N samples; still decode whole video. 0 = no limit')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--no-feature-cache', action='store_true', help='Use original per-prompt image encoding for comparison')
    parser.add_argument('--resume', action='store_true', help='Skip completed videos with exactly matching settings')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < .04 or not 0 < args.threshold < 1 or args.max_samples < 0:
        parser.error('interval must be >= .04, threshold in (0,1), max-samples >= 0')
    args.output = args.output.resolve()
    if args.video:
        try:
            videos = [resolve_video(args.video, args.videos)]
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    else:
        videos = discover(args.videos)
    if not videos or any(not p.is_file() for p in videos):
        parser.error('No valid video files found')
    backend = ROOT / 'runtime' / 'GroundingDINO-local'
    if not (backend / 'weights' / 'model.safetensors').is_file():
        parser.error(f'Missing local Grounding DINO weights: {backend / "weights"}')
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(backend))
    print('Loading existing local model; no video is sent to an external service...', flush=True)
    from detect_local import load_model, detect
    import torch
    torch.set_num_threads(2)
    import transformers
    if transformers.__version__ != '4.51.3':
        parser.error('Use a fresh Python process. The tested project-local transformers version is 4.51.3.')
    loaded = load_model()
    print(f'Model ready on {loaded[2]}; videos={len(videos)}', flush=True)
    rows = []
    for video in videos:
        try:
            rows.append(analyze(video, args, loaded, detect))
        except Exception as error:
            import traceback
            traceback.print_exc()
            rows.append(dict(status='error', video=str(video), error=str(error)))
        write_json(args.output / 'batch_summary.json', rows)
        make_index(args.output, rows)
    errors = sum(r['status'] == 'error' for r in rows)
    print(f'FINISHED: {len(rows)-errors}/{len(rows)} videos. Open {args.output / "index.html"}', flush=True)
    return int(errors > 0)


if __name__ == '__main__':
    raise SystemExit(main())
