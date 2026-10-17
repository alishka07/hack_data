"""CPU-only annotated people preview; never rewrites primary tracker evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

VERSION = 'people-roles-preview-2-segment-reflections'
_LOCK = threading.Lock()


def _signature(folder, job, report):
    folder = Path(folder)
    csv_path = folder / 'people' / 'review.csv'
    source_name = job.get('_source_file', 'input.mov')
    source = folder / source_name
    if source.parent.resolve() != folder.resolve() or not source.is_file():
        raise ValueError('Expected a local source video in this job folder.')
    with csv_path.open('rb') as stream:
        csv_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    stat = source.stat()
    payload = {'version': VERSION, 'csv_sha256': csv_sha,
               'video': job.get('video', {}), 'source_name': source_name,
               'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
               'role_summary': report.get('service', {}).get('role_summary', {}),
               'reflections': report.get('reflections', {})}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')).encode('utf-8')).hexdigest(), csv_sha, source


def existing_people_preview(folder, job, report):
    """Lightweight fingerprint check; does not decode video or render anything."""
    try:
        folder = Path(folder)
        signature, _, _ = _signature(folder, job, report)
        manifest = json.loads((folder / 'people' / 'roles.json').read_text(encoding='utf-8'))
        target = folder / 'people' / 'roles.webm'
        if (manifest.get('status') == 'completed' and manifest.get('signature') == signature
                and manifest.get('frames') == job.get('video', {}).get('frames')
                and manifest.get('file_size') == target.stat().st_size and target.stat().st_size > 0):
            return target
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _cancelled(callback):
    return bool(callback() if callable(callback) else callback.is_set() if callback is not None else False)


def _labels(report):
    from reflection_tracks import exclusion_predicate
    role = report.get('service', {}).get('role_summary', {})
    cashier = role.get('cashier_track_id') if role.get('status') == 'candidate' else None
    visitors = set(role.get('visitor_track_ids', []))
    reflected_at = exclusion_predicate(report.get('reflections', {}))
    def label(identity, frame=None):
        if reflected_at(identity, frame):
            return f'Отражение · ID {identity}', (155, 160, 169)
        if identity == cashier:
            return f'Кассир (кандидат) · ID {identity}', (59, 210, 143)
        if identity in visitors:
            return f'Посетитель (кандидат) · ID {identity}', (74, 175, 255)
        return f'Человек · ID {identity}', (245, 195, 70)
    return label


def _font(size):
    from PIL import ImageFont
    for name in ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.truetype('DejaVuSans.ttf', size=size)


def _draw_frame(frame, rows, get_label, font, timestamp):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for row in rows:
        raw = row.get('box', [])
        if len(raw) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in raw):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in raw]
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        text, color = get_label(row['id'], row.get('frame'))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        bounds = draw.textbbox((0, 0), text, font=font)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        left = max(0, min(x1, width - tw - 8))
        top = max(0, y1 - th - 8)
        draw.rectangle((left, top, min(width - 1, left + tw + 8), top + th + 8), fill=(25, 31, 38))
        draw.text((left + 4, top + 3 - bounds[1]), text, font=font, fill=color)
    stamp = f'{timestamp:.2f} с'
    bounds = draw.textbbox((0, 0), stamp, font=font)
    draw.rectangle((0, 0, bounds[2] + 12, bounds[3] - bounds[1] + 10), fill=(25, 31, 38))
    draw.text((6, 4 - bounds[1]), stamp, font=font, fill=(255, 255, 255))
    return np.asarray(canvas)


def _replace(source, target):
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.04 * (attempt + 1))


def prepare_people_preview(folder, job, report, cancelled=None):
    """Render a complete source-length annotated WebM or keep the old preview.

    Returns metadata with status completed/cancelled/unavailable/failed. Failure
    does not change report verdicts or model stage status. The original CSV,
    review.mp4 and source video are read-only inputs.
    """
    from rule_engine import _load_people
    import cv2
    import imageio_ffmpeg

    folder = Path(folder)
    if _cancelled(cancelled):
        return {'status': 'cancelled', 'reason': 'Обновление подписей отменено.'}
    with _LOCK:
        cached = existing_people_preview(folder, job, report)
        if cached:
            result = json.loads(cached.with_suffix('.json').read_text(encoding='utf-8'))
            return {**result, 'cached': True, 'path': str(cached)}
        people = _load_people(folder, job)
        if not people.get('complete'):
            return {'status': 'unavailable', 'reason': people.get('reason', 'Нет проверенного журнала людей.')}
        target = folder / 'people' / 'roles.webm'
        temporary = target.with_name('roles.encoding.webm')
        manifest_path = target.with_suffix('.json')
        manifest_temp = target.with_name('roles.encoding.json')
        backup = target.with_name('roles.previous.' + uuid.uuid4().hex + '.webm')
        error_log = target.with_name('roles.encoding.log')
        capture, process = None, None
        backed_up = published = committed = False
        started = time.monotonic()
        try:
            signature, csv_sha, source = _signature(folder, job, report)
            video = job['video']
            width, height, expected, fps = (video[k] for k in ('width', 'height', 'frames', 'fps'))
            if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (width, height, expected, fps)):
                raise ValueError('Некорректные размеры исходного видео.')
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise ValueError('Не удалось открыть исходное видео.')
            font = _font(max(12, min(18, round(width / 38))))
            get_label = _labels(report)
            command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
                       '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
                       '-an', '-c:v', 'libvpx-vp9', '-deadline', 'realtime', '-cpu-used', '8', '-threads', '2',
                       '-crf', '34', '-b:v', '0', '-pix_fmt', 'yuv420p', str(temporary)]
            frames = 0
            with error_log.open('wb') as error_stream:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_stream,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                while True:
                    if _cancelled(cancelled):
                        raise InterruptedError('Обновление подписей отменено.')
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (height, width):
                        raise ValueError('Размер кадра изменился внутри видео.')
                    annotated = _draw_frame(frame, people.get('by_frame', {}).get(frames, []), get_label, font, frames / fps)
                    process.stdin.write(annotated.tobytes())
                    frames += 1
                process.stdin.close()
                process.stdin = None
                code = process.wait(timeout=60)
                if code != 0:
                    raise RuntimeError('Не удалось закодировать видео с ролями.')
            if frames != expected or not temporary.is_file() or temporary.stat().st_size == 0:
                raise ValueError('Не удалось прочитать и сохранить все исходные кадры.')
            if _cancelled(cancelled):
                raise InterruptedError('Обновление подписей отменено.')
            # Check the encoded stream before replacing the previous preview.
            verify = cv2.VideoCapture(str(temporary))
            encoded_frames = int(round(verify.get(cv2.CAP_PROP_FRAME_COUNT)))
            encoded_fps = verify.get(cv2.CAP_PROP_FPS)
            first_ok, _ = verify.read()
            verify.release()
            if not first_ok or encoded_frames != expected or abs(encoded_fps - fps) > .1:
                raise ValueError('Проверка сохранённого видео не подтвердила число кадров и частоту.')
            metadata = {'status': 'completed', 'version': VERSION, 'signature': signature,
                        'csv_sha256': csv_sha, 'frames': frames, 'fps': fps, 'width': width, 'height': height,
                        'file_size': temporary.stat().st_size, 'elapsed_sec': round(time.monotonic() - started, 2),
                        'source_frames_annotated': True, 'original_tracker_journal_unchanged': True,
                        'excluded_reflection_track_ids': report.get('reflections', {}).get('excluded_track_ids', []),
                        'reflection_observation_exclusions': report.get('reflections', {}).get('observation_exclusions', []),
                        'partially_excluded_reflection_track_ids': report.get('reflections', {}).get('partially_excluded_track_ids', []),
                        'legend': ['Кассир и посетитель — позиционные кандидаты.',
                                   'Серым отмечены только подтверждённые участки отражений; остальные кадры того же ID сохраняются как человек.']}
            manifest_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            if target.exists():
                _replace(target, backup)
                backed_up = True
            _replace(temporary, target)
            published = True
            _replace(manifest_temp, manifest_path)
            committed = True
            return {**metadata, 'cached': False, 'path': str(target)}
        except Exception as error:
            try:
                if published:
                    target.unlink(missing_ok=True)
                if backed_up and backup.exists():
                    _replace(backup, target)
            except OSError as restore_error:
                return {'status': 'failed', 'reason': str(error) + '; ' + str(restore_error),
                        'recoverable_previous_preview': str(backup)}
            return {'status': 'cancelled' if isinstance(error, InterruptedError) else 'failed', 'reason': str(error)}
        finally:
            if capture is not None:
                capture.release()
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            for path in (temporary, manifest_temp, *([backup] if committed else [])):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
