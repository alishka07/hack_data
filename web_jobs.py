"""Local, serial video-analysis jobs for the web interface.

Models always run in fresh subprocesses, with the current Python interpreter.
This module does not import Torch or keep GPU state in the web server.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from urllib.parse import quote
import uuid
import zipfile

CASE = Path(__file__).resolve().parent
TERMINAL = {'completed', 'partial', 'failed', 'cancelled', 'interrupted'}
STAGE_LABELS = {
    'trained': 'Контекст кассы',
    'desk': 'Проверка рабочего места',
    'people': 'Присутствие и перемещения людей',
    'visual_states': 'Состояния двери и сейфа',
    'dense_review': 'Уточнение мелких предметов',
}
LABELS_RU = {
    'safe': 'Сейф', 'banknote_counter': 'Счётчик банкнот', 'laptop': 'Ноутбук',
    'bag': 'Мешок / сумка', 'plastic_bag': 'Пакет', 'computer_monitor': 'Монитор',
    'desktop_computer': 'Системный блок', 'cash': 'Банкноты',
    'food': 'Еда / шоколад', 'drink': 'Чашка / напиток',
    'smartphone': 'Смартфон', 'document': 'Документ', 'identity_card': 'Удостоверение',
    'keyboard': 'Клавиатура', 'mouse': 'Мышь', 'notebook': 'Блокнот',
    'usb_drive': 'Флешка', 'electrical_outlet': 'Розетка',
    'power_strip': 'Блок розеток', 'chair': 'Стул', 'person': 'Человек', 'work_table': 'Рабочий стол (гипотеза)',
}
BACKGROUND = {'electrical_outlet', 'power_strip', 'chair'}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _atomic_json(path: Path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        # Windows indexers/antivirus may briefly open a newly written file without
        # FILE_SHARE_DELETE. Keep the old complete snapshot until replacement works.
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.05 * 2**attempt, .5))
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback


def _dense_review_result(folder, job, stage):
    """Keep optional close-up diagnostics separate from authoritative rule inputs."""
    summary = _read_json(folder / 'dense_review/summary.json', {})
    if not isinstance(summary, dict):
        summary = {}
    result = dict(summary)
    result['observations'] = []
    status = summary.get('status', 'not_checked')
    rows, journal_error = [], False
    journal = folder / 'dense_review/observations.jsonl'
    if journal.is_file() and status in {'completed', 'partial'}:
        try:
            with journal.open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('Invalid close-up observation')
                    rows.append(row)
        except (OSError, ValueError):
            journal_error = True
    count = summary.get('analyzed_samples', 0)
    planned = summary.get('planned_samples', 0)
    valid_count = type(count) is int and count >= 0
    complete = (status == 'completed' and summary.get('all_planned_samples_analyzed') is True
                and summary.get('frame_count_matches') is True
                and summary.get('decoded_frames') == job.get('video', {}).get('frames')
                and type(summary.get('decoded_frames')) is int and summary['decoded_frames'] > 0
                and valid_count and type(planned) is int and count == planned == len(rows)
                and count > 0 and not journal_error)
    if complete:
        source_frames = [r.get('source_frame') for r in rows]
        timestamps = [r.get('timestamp_sec') for r in rows]
        complete = (all(type(i) is int and 0 <= i < summary['decoded_frames'] for i in source_frames)
                    and all(isinstance(t, (int, float)) and not isinstance(t, bool)
                            and math.isfinite(t) and t >= 0 for t in timestamps)
                    and all(a < b for a, b in zip(source_frames, source_frames[1:]))
                    and all(a < b for a, b in zip(timestamps, timestamps[1:])))
    if complete and stage.get('status') in {'completed', 'running'}:
        result.update(status='completed', observations=rows,
                      title_ru=f'Уточнение мелких предметов: {count} кадров',
                      note_ru='Увеличенные области проверены чаще только в выбранных интервалах. Наблюдения требуют просмотра и пока не используются как доказательство действий.')
    elif status == 'partial' or (status == 'completed' and not complete):
        result.update(status='partial', observations=[],
                      title_ru='Уточнение мелких предметов завершено частично',
                      note_ru='Полнота выбранных кадров не подтверждена. Частичный журнал доступен для диагностики и не включён в выводы о действиях.')
        if status == 'completed':
            result['reason'] = 'incomplete_or_invalid_closeup_journal'
    else:
        result.update(status='not_checked', observations=[], analyzed_samples=0,
                      title_ru='Уточнение мелких предметов не выполнено',
                      note_ru='Дополнительная проверка увеличенных областей не выполнена. Это не означает отсутствие предметов или нарушений.')
    result['stage_status'] = stage.get('status')
    result['action'] = 'unknown'
    result['violations'] = []
    result['used_for_rule_verdicts'] = False
    return result


def _video_metadata(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Не удалось открыть видео. Выберите другой файл или сохраните его в MP4.')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        readable, frame = cap.read()
        if not readable or frame is None or frames <= 0 or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Видео не содержит читаемых кадров или корректных данных о длительности.')
        return {'width': width, 'height': height, 'fps': fps, 'duration': frames / fps, 'frames': frames}
    finally:
        cap.release()


def _browser_preview_command(source: Path, target: Path):
    import imageio_ffmpeg
    return [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
            '-i', str(source), '-an', '-c:v', 'libvpx-vp9', '-deadline', 'realtime',
            '-cpu-used', '5', '-crf', '34', '-b:v', '0', '-threads', '2', '-row-mt', '1',
            '-pix_fmt', 'yuv420p', str(target)]


def _stage_video(folder: Path, job: dict, key: str):
    if key == 'people':
        return folder / 'people/review.mp4'
    return folder / key / Path(job['_source_file']).stem / 'review.mp4'


def _people_decode_warning(report: dict, metadata: dict):
    """A successful process exit is not proof that OpenCV decoded the full file."""
    processed = int(report.get('frames', 0))
    expected = int(metadata.get('frames', 0))
    fps = float(metadata.get('fps', 0))
    timestamp = report.get('last_timestamp_sec')
    reasons = []
    incomplete = expected > 0 and processed < expected
    if expected > 0 and processed != expected:
        reasons.append(f'обработано {processed} из {expected} кадров')
    if expected > 0 and math.isfinite(fps) and fps > 0:
        expected_last = (expected - 1) / fps
        tolerance = max(.15, 2 / fps)
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            reasons.append('нет корректной временной отметки последнего кадра')
        elif abs(timestamp - expected_last) > tolerance:
            reasons.append(f'последний кадр на {timestamp:.2f} с при ожидаемых {expected_last:.2f} с')
            incomplete |= timestamp < expected_last - tolerance
    if report.get('decoding_warning') and not reasons:
        reasons.append('обработчик сообщил о проблеме чтения исходного видео')
    if not reasons:
        return None
    prefix = 'Обработана только часть видео' if incomplete else 'Полнота обработки видео требует проверки'
    return prefix + ': ' + '; '.join(reasons) + '. Результаты относятся только к прочитанным кадрам.'


class JobManager:
    """One worker, one owned model process at a time, persistent job snapshots."""

    def __init__(self, root: Path = CASE / 'web_data/jobs'):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._termination_lock = threading.Lock()
        self._queue = queue.Queue()
        self._jobs = {}
        self._cancellations = {}
        self._process = None
        self._process_job = None
        self._closed = False
        for folder in self.root.iterdir():
            if not folder.is_dir() or not re.fullmatch(r'[a-f0-9]{32}', folder.name):
                continue
            job = _read_json(folder / 'job.json')
            if not isinstance(job, dict) or job.get('id') != folder.name:
                continue
            if job.get('status') in {'queued', 'running'}:
                job.update(status='interrupted', stage='Обработка прервана перезапуском приложения', updated_at=_now())
                for stage in job.get('stages', []):
                    if stage.get('status') in {'pending', 'running'}:
                        stage['status'] = 'cancelled'
                _atomic_json(folder / 'job.json', job)
            self._jobs[folder.name] = job
        self._worker = threading.Thread(target=self._work, name='video-analysis-worker', daemon=True)
        self._worker.start()

    @staticmethod
    def _public(job):
        return copy.deepcopy({k: v for k, v in job.items() if not k.startswith('_')})

    def _save(self, job):
        job['updated_at'] = _now()
        _atomic_json(self.root / job['id'] / 'job.json', job)

    def enqueue(self, job_id: str, source_path: Path, original_name: str, options: dict) -> dict:
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise ValueError('Некорректный номер задания.')
        source = Path(source_path).resolve()
        folder = self.root / job_id
        if source.parent != folder or not source.is_file():
            raise ValueError('Файл видео не найден в папке задания.')
        selected = {key: bool(options.get(key, False)) for key in ('trained', 'desk', 'people')}
        if not any(selected.values()):
            raise ValueError('Выберите хотя бы один вид анализа.')
        selected['visual_states'] = True
        # The measured close-up experiment was expensive and did not yet improve
        # cash detection. Keep it opt-in until quality/speed have been validated.
        selected['dense_review'] = selected['desk'] and os.environ.get('CASE_ENABLE_DENSE_REVIEW') == '1'
        interval = float(options.get('interval', 1))
        if not math.isfinite(interval) or not .04 <= interval <= 60:
            raise ValueError('Шаг анализа должен быть от 0,04 до 60 секунд.')
        selected['interval'] = interval
        metadata = _video_metadata(source)
        with self._lock:
            if self._closed:
                raise RuntimeError('Приложение завершает работу. Запустите его заново.')
            if job_id in self._jobs:
                raise ValueError('Такое задание уже существует.')
            job = {
                'id': job_id, 'name': str(original_name), 'status': 'queued',
                'created_at': _now(), 'updated_at': _now(), 'options': selected,
                'progress': 0.0, 'stage': 'В очереди', 'video': metadata,
                'stages': [{'key': key, 'status': 'pending'} for key in STAGE_LABELS if selected[key]],
                '_source_file': source.name,
            }
            self._jobs[job_id] = job
            self._cancellations[job_id] = threading.Event()
            try:
                self._save(job)
            except Exception:
                self._jobs.pop(job_id, None)
                self._cancellations.pop(job_id, None)
                raise
            self._queue.put(job_id)
            return self._public(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public(job) if job else None

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._public(job) for job in sorted(self._jobs.values(), key=lambda row: row['created_at'], reverse=True)]

    def _terminate(self, process):
        if process is None:
            return
        with self._termination_lock:
            if process.poll() is not None:
                return
            if os.name == 'nt':
                try:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=15, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    # The owned Python process is the GPU owner. Do not leave it
                    # running if Windows cannot complete the tree-stop command.
                    if process.poll() is None:
                        process.kill()
            else:
                import signal
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != 'nt':
                    import signal
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait(timeout=10)

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job['status'] in TERMINAL:
                return self._public(job)
            self._cancellations[job_id].set()
            job['_stop_status'] = 'cancelled'
            job['stage'] = 'Останавливаем обработку'
            process = self._process if self._process_job == job_id else None
            if job['status'] == 'queued':
                job.update(status='cancelled', stage='Отменено')
                for stage in job['stages']:
                    stage['status'] = 'cancelled'
            self._save(job)
        self._terminate(process)
        return self.get(job_id)

    def shutdown(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job_id, job in self._jobs.items():
                if job['status'] in {'queued', 'running'}:
                    self._cancellations[job_id].set()
                    job['_stop_status'] = 'interrupted'
                    job.update(status='interrupted', stage='Обработка прервана остановкой приложения')
                    for stage in job['stages']:
                        if stage['status'] in {'pending', 'running'}:
                            stage['status'] = 'cancelled'
                    self._save(job)
            process = self._process
            self._queue.put(None)
        self._terminate(process)
        self._worker.join(timeout=20)

    def _command(self, job, key):
        folder = self.root / job['id']
        source = folder / job['_source_file']
        if key == 'dense_review':
            return [sys.executable, '-u', str(CASE/'dense_review.py'), '--video', str(source),
                    '--job-dir', str(folder), '--output', str(folder/'dense_review'),
                    '--max-seconds', '12', '--sample-fps', '5']
        if key == 'visual_states':
            return [sys.executable, '-u', str(CASE / 'visual_states.py'), '--video', str(source),
                    '--output', str(folder / 'visual_states'), '--sample-seconds', '.25']
        if key == 'people':
            return [sys.executable, '-u', str(CASE / 'track_people.py'), '--video', str(source),
                    '--duration', '0', '--output', str(folder / 'people/review.mp4')]
        script = 'detect_trained.py' if key == 'trained' else 'run.py'
        command = [sys.executable, '-u', str(CASE / script), '--video', str(source),
                   '--interval', str(job['options']['interval']), '--output', str(folder / key)]
        if key == 'desk':
            command += ['--groups', 'desk', 'cash', 'workspace', 'food_drink']
        return command

    def _validate_stage(self, job, key):
        folder = self.root / job['id']
        if key == 'dense_review':
            raw_summary = _read_json(folder/'dense_review/summary.json', {})
            stage = next(s for s in job['stages'] if s['key'] == key)
            summary = _dense_review_result(folder, job, stage)
            valid = (isinstance(raw_summary, dict) and raw_summary.get('status') in {'completed', 'not_checked'}
                     and summary.get('status') in {'completed', 'not_checked'})
            for stage in job['stages']:
                if stage['key'] == key:
                    stage['coverage_status'] = summary.get('status', 'failed')
                    stage['coverage_reason'] = summary.get('reason')
                    if summary.get('status') == 'not_checked':
                        stage['warning'] = summary['note_ru']
            return valid
        if key == 'visual_states':
            summary = _read_json(folder / 'visual_states/observations.json', {})
            return bool(summary.get('status') == 'completed' and summary.get('frame_count_matches') is True
                        and summary.get('decoded_frames') == job['video']['frames']
                        and summary.get('observations'))
        if key == 'people':
            summary = _read_json(folder / 'people/review.json', {})
            valid = bool(summary.get('frames', 0) > 0 and (folder / 'people/review.mp4').is_file())
            warning = _people_decode_warning(summary, job.get('video', {})) if valid else None
            if warning:
                with self._lock:
                    for stage in job['stages']:
                        if stage['key'] == key and warning not in stage.get('warning', ''):
                            stage['warning'] = (stage.get('warning', '') + ' ' + warning).strip()
            return valid
        summary = _read_json(folder / key / Path(job['_source_file']).stem / 'summary.json', {})
        valid = summary.get('status') in {'completed', 'completed_with_warnings'} and summary.get('analyzed_samples', 0) > 0
        if valid and (summary.get('status') == 'completed_with_warnings' or summary.get('decoding_warning')):
            with self._lock:
                for stage in job['stages']:
                    if stage['key'] == key:
                        stage['warning'] = 'Количество прочитанных кадров отличается от данных видеофайла. Часть кадров могла не прочитаться.'
        return valid

    def _prepare_browser_preview(self, job_id, stage, log):
        """Encode a browser-compatible copy under the same cancellation owner."""
        job = self._jobs[job_id]
        source = _stage_video(self.root / job_id, job, stage['key'])
        target = source.with_suffix('.webm')
        temporary = source.with_name('review.encoding.webm')
        process = None
        try:
            if not source.is_file():
                raise RuntimeError('The model stage produced no MP4 preview.')
            command = _browser_preview_command(source, temporary)
            with self._lock:
                if self._cancellations[job_id].is_set():
                    return
                job['stage'] = 'Подготавливаем видео для браузера'
                self._save(job)
                process = subprocess.Popen(command, cwd=str(CASE), stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                           start_new_session=os.name != 'nt')
                self._process, self._process_job = process, job_id
            code = process.wait()
            if self._cancellations[job_id].is_set():
                return
            if code != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f'Browser preview encoder exited with code {code}.')
            temporary.replace(target)
        except Exception as error:
            log.write('\nBROWSER PREVIEW ERROR: ' + repr(error) + '\n')
            log.flush()
            with self._lock:
                if not self._cancellations[job_id].is_set():
                    warning = 'Не удалось подготовить видео для встроенного браузера. Результаты модели сохранены; доступна копия MP4.'
                    stage['warning'] = (stage.get('warning', '') + ' ' + warning).strip()
                    stage['preview_error_detail'] = str(error)[-2000:]
                    self._save(job)
        finally:
            try:
                self._terminate(process)
            finally:
                temporary.unlink(missing_ok=True)
                with self._lock:
                    if self._process is process:
                        self._process, self._process_job = None, None

    def _run_stage(self, job_id, index):
        stage_started = time.perf_counter()
        with self._lock:
            job = self._jobs[job_id]
            if self._cancellations[job_id].is_set():
                return
            stage = job['stages'][index]
            key = stage['key']
            stage['status'] = 'running'
            job.update(stage=STAGE_LABELS[key] + ' — загружаем модель', progress=index / len(job['stages']))
            self._save(job)
            command = self._command(job, key)
        log_path = self.root / job_id / (key + '.log')
        recent = deque(maxlen=12)
        process = None
        try:
            environment = os.environ.copy()
            environment.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0')
            with log_path.open('w', encoding='utf-8') as log:
                with self._lock:
                    if self._cancellations[job_id].is_set():
                        return
                    process = subprocess.Popen(command, cwd=str(CASE), env=environment,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, encoding='utf-8', errors='replace', bufsize=1,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                               start_new_session=os.name != 'nt')
                    self._process, self._process_job = process, job_id
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    recent.append(line.rstrip())
                    match = re.search(r'(?:\bt|\btime)=([0-9]+(?:\.[0-9]+)?)s', line)
                    dense_match = re.search(r'\bDENSE (\d+)/(\d+) at ', line) if key == 'dense_review' else None
                    if match or dense_match:
                        with self._lock:
                            if not self._cancellations[job_id].is_set():
                                portion = (int(dense_match.group(1)) / max(1, int(dense_match.group(2)))
                                           if dense_match else float(match.group(1)) / job['video']['duration'])
                                fraction = min(.97, max(.02, portion))
                                job['progress'] = max(job['progress'], (index + fraction) / len(job['stages']))
                                job['stage'] = STAGE_LABELS[key] + ' — обрабатываем видео'
                                self._save(job)
                returncode = process.wait()
                if self._cancellations[job_id].is_set():
                    return
                if returncode != 0 or not self._validate_stage(job, key):
                    raise RuntimeError(f'Exit code: {returncode}.\n' + '\n'.join(recent))
                if key not in {'visual_states', 'people', 'dense_review'}:
                    self._prepare_browser_preview(job_id, stage, log)
                if self._cancellations[job_id].is_set():
                    return
            with self._lock:
                stage['status'] = 'completed'
                stage['processing_seconds'] = round(time.perf_counter() - stage_started, 3)
                job['progress'] = (index + 1) / len(job['stages'])
                self._save(job)
        except Exception as error:
            with log_path.open('a', encoding='utf-8') as log:
                log.write('\nWORKER ERROR: ' + repr(error) + '\n')
            with self._lock:
                if not self._cancellations[job_id].is_set():
                    stage.update(status='failed', error='Не удалось завершить этот этап. Подробности доступны в журнале.', error_detail=str(error)[-6000:])
                    self._save(job)
        finally:
            try:
                self._terminate(process)
            finally:
                if process is not None and process.stdout is not None:
                    process.stdout.close()
                with self._lock:
                    if self._process is process:
                        self._process, self._process_job = None, None

    def _prepare_role_preview(self, job_id):
        job = self._jobs[job_id]
        if not any(stage['key'] == 'people' and stage['status'] == 'completed' for stage in job['stages']):
            return
        with self._lock:
            job['stage'] = 'Уточняем роли людей и отмечаем отражения'
            self._save(job)
        from rule_engine import build_rule_report
        from people_preview import prepare_people_preview
        folder = self.root / job_id
        report = build_rule_report(folder, job, {})
        metadata = prepare_people_preview(folder, job, report, self._cancellations[job_id])
        with self._lock:
            job['role_preview'] = {key: value for key, value in metadata.items() if key != 'path'}
            self._save(job)
        if metadata.get('status') == 'failed' and not self._cancellations[job_id].is_set():
            stage = next(stage for stage in job['stages'] if stage['key'] == 'people')
            with (folder/'people.log').open('a', encoding='utf-8') as log:
                log.write('\nROLE PREVIEW ERROR: ' + str(metadata.get('reason')) + '\n')
                self._prepare_browser_preview(job_id, stage, log)

    def _work(self):
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                with self._lock:
                    job = self._jobs[job_id]
                    if self._cancellations[job_id].is_set() or self._closed:
                        continue
                    job.update(status='running', stage='Подготавливаем видео')
                    self._save(job)
                for index in range(len(job['stages'])):
                    if self._cancellations[job_id].is_set():
                        break
                    self._run_stage(job_id, index)
                if not self._cancellations[job_id].is_set():
                    try:
                        self._prepare_role_preview(job_id)
                    except Exception as error:
                        # A presentation failure must not discard completed model results.
                        job['role_preview'] = {'status': 'failed', 'reason': str(error)}
                with self._lock:
                    if self._cancellations[job_id].is_set():
                        job['status'] = job.get('_stop_status', 'cancelled')
                        job['stage'] = 'Отменено' if job['status'] == 'cancelled' else 'Обработка прервана остановкой приложения'
                        for stage in job['stages']:
                            if stage['status'] in {'running', 'pending'}:
                                stage['status'] = 'cancelled'
                    else:
                        completed = sum(stage['status'] == 'completed' for stage in job['stages'])
                        has_warnings = any(stage.get('warning') for stage in job['stages'])
                        job['status'] = 'completed' if completed == len(job['stages']) and not has_warnings else 'partial' if completed else 'failed'
                        job['progress'] = 1.0
                        job['stage'] = {'completed': 'Готово', 'partial': 'Часть результатов готова', 'failed': 'Не удалось обработать видео'}[job['status']]
                        if completed == len(job['stages']) and has_warnings:
                            job['stage'] = 'Готово с замечаниями'
                        if job['status'] == 'failed':
                            job['error'] = 'Ни один выбранный этап не завершился. Подробности доступны в журналах.'
                    self._save(job)
            except Exception as error:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job:
                        job.update(status='failed', stage='Ошибка обработки', error='Не удалось обработать задание.', error_detail=str(error)[-6000:])
                        for stage in job.get('stages', []):
                            if stage['status'] == 'running':
                                stage['status'] = 'failed'
                        try:
                            self._save(job)
                        except OSError as save_error:
                            # A storage failure must not kill the queue worker and
                            # leave every following upload waiting indefinitely.
                            job['error_detail'] += '\nНе удалось сохранить статус: ' + str(save_error)
                            print(job['error_detail'], file=sys.stderr, flush=True)
            finally:
                self._queue.task_done()


def collect_results(job_dir: Path, job: dict) -> dict:
    """Read only finished stages; partial files are never treated as detections."""
    folder = Path(job_dir).resolve()
    job_id = job['id']

    def link(path):
        relative = Path(path).resolve().relative_to(folder).as_posix()
        return '/api/jobs/' + job_id + '/files/' + quote(relative, safe='/')

    def video_entry(video, key, **extra):
        webm = video.with_suffix('.webm')
        webm_url = link(webm) if webm.is_file() and webm.stat().st_size else None
        output['downloads'].append({'label': STAGE_LABELS[key] + ' — видео MP4', 'url': link(video)})
        if webm_url:
            output['downloads'].append({'label': STAGE_LABELS[key] + ' — видео WebM', 'url': webm_url})
        return {'key': key, 'label': STAGE_LABELS[key], 'url': webm_url or link(video),
                'mp4_url': link(video), 'webm_url': webm_url,
                'file': video.relative_to(folder).as_posix(),
                'webm_file': webm.relative_to(folder).as_posix() if webm_url else None, **extra}

    output = {
        'videos': [], 'objects': [], 'class_counts': [], 'thumbnails': [], 'people': None,
        'downloads': [], 'analyzed_samples': {}, 'source_frames': {}, 'verdict': 'not_evaluated',
        'limitations': [
            'Эпизоды показывают возможные признаки нарушения и требуют просмотра человеком. Оценки объектов не являются вероятностью нарушения.',
            'Проверка автоматически использует все доступные модели. Предметы анализируются по выборке кадров, люди — по кадрам видео; очень короткие события и мелкие детали могут быть пропущены.',
            'Число рамок и кадров с объектом не равно числу отдельных предметов.',
            'Пакет или сумка не подтверждают наличие денег внутри. Отсутствие рамки не доказывает отсутствие объекта.',
            'Состояния двери и сейфа проверяются при совпадении с изученными ракурсами. Роли предполагаются по рабочему месту и стороне сейфа; проверенные кандидаты отражений исключаются. Пересчёт, оригинальность документов и окончание операции не подтверждаются.',
            'Дообученная модель проверялась на небольшом наборе кадров. На новых камерах возможны ошибки.',
        ],
    }
    boxes = Counter()
    frame_sets = defaultdict(set)
    for stage in job.get('stages', []):
        key = stage['key']
        log = folder / (key + '.log')
        if log.is_file():
            output['downloads'].append({'label': STAGE_LABELS[key] + ' — журнал', 'url': link(log)})
        if key == 'dense_review':
            output['closeup_review'] = _dense_review_result(folder, job, stage)
            if output['closeup_review']['status'] != 'completed':
                output['limitations'].append(output['closeup_review']['note_ru'])
            for name, label in [('summary.json', 'Уточнение предметов — охват'),
                                ('plan.json', 'Уточнение предметов — интервалы'),
                                ('signature.json', 'Уточнение предметов — происхождение'),
                                ('observations.jsonl', 'Уточнение предметов — диагностические рамки')]:
                path = folder / 'dense_review' / name
                if path.is_file():
                    output['downloads'].append({'label': label, 'url': link(path)})
            continue
        if stage.get('status') != 'completed':
            continue
        if stage.get('warning'):
            output['limitations'].append(STAGE_LABELS[key] + ': ' + stage['warning'])
        if key == 'visual_states':
            path = folder / 'visual_states/observations.json'
            if path.is_file():
                output['downloads'].append({'label': 'Состояния двери и сейфа — JSON', 'url': link(path)})
            continue
        if key == 'people':
            video = folder / 'people/review.mp4'
            report = _read_json(video.with_suffix('.json'), {})
            if not report or not video.is_file():
                continue
            output['videos'].append(video_entry(video, key, sampled=False))
            output['people'] = {
                'track_ids': report.get('track_ids', []), 'frames': report.get('frames', 0),
                'source_frames': job.get('video', {}).get('frames'),
                'last_timestamp_sec': report.get('last_timestamp_sec'),
                'processing_seconds': report.get('processing_seconds'),
                'note': 'ID обозначают временные траектории. Один человек может получить несколько ID; это не личности и не точное число людей.',
            }
            # Recompute for older saved jobs, including results produced before
            # stage validation checked premature OpenCV end-of-stream.
            decode_warning = _people_decode_warning(report, job.get('video', {}))
            if decode_warning:
                output['people']['warning'] = decode_warning
                output['limitations'].append('Люди: ' + decode_warning)
            output['source_frames'][key] = report.get('frames', 0)
            output['analyzed_samples'][key] = report.get('frames', 0)
            for extension, label in [('.csv', 'Траектории людей — CSV'), ('.json', 'Люди — сводка JSON')]:
                path = video.with_suffix(extension)
                if path.is_file():
                    output['downloads'].append({'label': label, 'url': link(path)})
            continue
        # The input is uploaded with a neutral name; each CLI creates one child folder.
        summaries = sorted((folder / key).glob('*/summary.json'))
        for summary_path in summaries:
            summary = _read_json(summary_path, {})
            if summary.get('status') not in {'completed', 'completed_with_warnings'}:
                continue
            result_dir = summary_path.parent
            output['analyzed_samples'][key] = output['analyzed_samples'].get(key, 0) + summary.get('analyzed_samples', 0)
            output['source_frames'][key] = output['source_frames'].get(key, 0) + summary.get('decoded_frames', 0)
            if summary.get('decoding_warning'):
                output['limitations'].append(STAGE_LABELS[key] + ': количество прочитанных кадров отличается от данных файла. Проверьте исходное видео.')
            video = result_dir / 'review.mp4'
            if summary.get('video_preview') and video.is_file():
                output['videos'].append(video_entry(video, key, sampled=True, interval=summary.get('sample_interval_sec')))
            interval = float(summary.get('sample_interval_sec', job.get('options', {}).get('interval', 1)))
            preview_frame_duration = 1 / max(.1, 1 / interval)
            for thumb_index, thumb in enumerate(summary.get('thumbnails', [])):
                picture = result_dir / Path(thumb['file']).name
                if picture.is_file():
                    output['thumbnails'].append({'url': link(picture), 'timestamp_sec': thumb.get('timestamp_sec'), 'playback_sec': thumb_index * preview_frame_duration, 'mode': key})
            frames_path = result_dir / 'frames.jsonl'
            if frames_path.is_file():
                with frames_path.open(encoding='utf-8') as stream:
                    for record_index, line in enumerate(stream):
                        try:
                            frame = json.loads(line)
                        except ValueError:
                            continue
                        for status in ('candidates', 'review'):
                            for row in frame.get(status, []):
                                label = row.get('label', '')
                                if label in BACKGROUND:
                                    continue
                                output['objects'].append({
                                    'timestamp_sec': frame.get('timestamp_sec'), 'source_frame': frame.get('source_frame'),
                                    'playback_sec': record_index * preview_frame_duration,
                                    'label': label, 'label_ru': LABELS_RU.get(label, label), 'score': row.get('score'),
                                    'source': row.get('source', 'model'), 'score_kind': row.get('score_kind', 'model_score'),
                                    'reference_frame': row.get('reference_frame_zero_based', row.get('reference_frame')),
                                    'status': status, 'mode': key, 'box': row.get('box'), 'reasons': row.get('reasons', []),
                                })
                                boxes[label] += 1
                                frame_sets[label].add((key, frame.get('source_frame')))
            for filename, label in [('detections.csv', 'Все рамки — CSV'), ('frames.jsonl', 'Кадры и исходные ответы — JSONL'), ('summary.json', 'Сводка — JSON')]:
                path = result_dir / filename
                if path.is_file():
                    output['downloads'].append({'label': STAGE_LABELS[key] + ': ' + label, 'url': link(path)})
    output['objects'].sort(key=lambda row: (row.get('timestamp_sec') or 0, row['mode'], row['label']))
    output['class_counts'] = [{'label': key, 'label_ru': LABELS_RU.get(key, key), 'frames': len(frame_sets[key]), 'boxes': value} for key, value in boxes.most_common()]
    if job.get('options', {}).get('desk'):
        output['limitations'].append('Предметы на столе и банкноты ищет экспериментальная модель по текстовым описаниям. Маленькие флешки и документы могут распознаваться неверно.')
    if job.get('status') in {'cancelled', 'interrupted', 'partial', 'failed'}:
        output['limitations'].append('Показаны результаты только полностью завершённых этапов. Незавершённые файлы сохранены в архиве для диагностики.')
    from rule_engine import build_rule_report
    from web_report import save_rule_report
    report = build_rule_report(folder, job, output)
    from people_preview import existing_people_preview
    role_preview = existing_people_preview(folder, job, report)
    if role_preview:
        for video in output['videos']:
            if video['key'] == 'people':
                video.update(url=link(role_preview), webm_url=link(role_preview), mp4_url=None,
                             file=role_preview.relative_to(folder).as_posix(),
                             webm_file=role_preview.relative_to(folder).as_posix(), label='Люди, роли и отражения')
        output['downloads'].append({'label': 'Люди, роли и отражения — видео WebM', 'url': link(role_preview)})
    elif output['people'] and job.get('status') in TERMINAL:
        output['limitations'].append('Видеоподписи этой записи ещё не обновлены; роли и исключения отражений приведены в отчёте. Старое видео содержит исходные номера трекера.')
    if output['people']:
        excluded = set(report.get('reflections', {}).get('excluded_track_ids', []))
        raw_ids = output['people'].get('track_ids', [])
        output['people']['raw_track_ids'] = raw_ids
        output['people']['track_ids'] = [identity for identity in raw_ids if identity not in excluded]
        output['people']['excluded_reflection_track_ids'] = sorted(excluded)
    try:
        report_downloads = save_rule_report(folder, job, report)
    except OSError:
        # A locked export or full disk must not hide the already computed review.
        report_downloads = []
        output['limitations'].append('Не удалось сохранить файлы отчёта или кадры. Результаты показаны на странице; закройте открытые файлы отчёта, проверьте место на диске и обновите страницу.')
    output.update(report)
    output['downloads'] = report_downloads + output['downloads']
    return output


def prepare_existing_browser_previews(job_dir: Path):
    """CPU-only maintenance for a terminal job, run while no queue job is active."""
    folder = Path(job_dir).resolve()
    job = _read_json(folder / 'job.json')
    if not isinstance(job, dict) or job.get('status') not in TERMINAL:
        raise ValueError('Видеокопии можно подготовить только для завершённого задания.')
    results = []
    for stage in job.get('stages', []):
        if stage.get('status') != 'completed':
            continue
        key = stage['key']
        if key in {'visual_states', 'dense_review'}:
            continue  # These stages produce observations, not review videos.
        source = _stage_video(folder, job, key)
        if not source.is_file():
            continue
        target = source.with_suffix('.webm')
        temporary = source.with_name('review.encoding.webm')
        try:
            with (folder / (key + '.log')).open('a', encoding='utf-8') as log:
                subprocess.run(_browser_preview_command(source, temporary), stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                               check=True)
            if not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError('Browser preview is empty.')
            temporary.replace(target)
            results.append({'key': key, 'status': 'completed', 'file': str(target), 'bytes': target.stat().st_size})
        except Exception as error:
            results.append({'key': key, 'status': 'failed', 'error': str(error)})
        finally:
            temporary.unlink(missing_ok=True)
    return results


def _build_archive_locked(job_dir: Path) -> Path:
    """Export results and logs, excluding the user's uploaded source video."""
    folder = Path(job_dir).resolve()
    job = _read_json(folder / 'job.json')
    if not isinstance(job, dict):
        raise ValueError('Задание не найдено.')
    if job.get('status') not in TERMINAL:
        raise ValueError('Дождитесь завершения обработки перед скачиванием архива.')
    result = collect_results(folder, job)
    _atomic_json(folder / 'result.json', result)
    archive_path = folder / 'results.zip'
    temporary = folder / ('results.' + uuid.uuid4().hex + '.zip.tmp')
    source_name = job.get('_source_file', '')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or path.is_symlink() or path.name.endswith('.tmp'):
                    continue
                relative = path.relative_to(folder)
                if len(relative.parts) == 1 and (path.name == source_name or path.name == 'results.zip' or path.name.endswith('.tmp') or path.name.startswith('input.')):
                    continue
                archive.write(path, relative.as_posix())
        temporary.replace(archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return archive_path


def build_archive(job_dir: Path) -> Path:
    # GET results may prepare portable reports too. Keep their atomic writes
    # outside the archive's snapshot, including on Windows file sharing locks.
    from web_report import REPORT_LOCK
    with REPORT_LOCK:
        return _build_archive_locked(job_dir)
