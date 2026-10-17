"""Local upload API for the existing offline video models; no model imports here."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.formparsers import MultiPartException
from starlette.staticfiles import StaticFiles

from web_jobs import JobManager, collect_results, build_archive

CASE = Path(__file__).resolve().parent
MAX_UPLOAD = 512 * 1024 * 1024
EXTENSIONS = {'.mov', '.mp4', '.mkv', '.avi', '.m4v', '.webm'}
JOB_ID = re.compile(r'^[a-f0-9]{32}$')
APP_ID = 'fortebank-local-video-review'
TERMINAL = {'completed', 'partial', 'failed', 'cancelled', 'interrupted'}


class UploadSizeLimit:
    """Bound chunked uploads before the multipart parser can spool them to disk."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') != '/api/jobs':
            return await self.app(scope, receive, send)
        received = 0
        too_large = False

        async def limited_receive():
            nonlocal received, too_large
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > self.limit:
                    too_large = True
                    # MultiPartParser closes any already-spooled files on this exception.
                    raise MultiPartException('Видео слишком большое. Максимум — 512 МБ.')
            return message

        async def limited_send(message):
            if too_large and message['type'] == 'http.response.start':
                message = {**message, 'status': 413}
            await send(message)

        return await self.app(scope, limited_receive, limited_send)


def parse_options(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        raise HTTPException(422, 'Не удалось прочитать настройки анализа.')
    if not isinstance(value, dict) or set(value) - {'trained', 'desk', 'people', 'interval'}:
        raise HTTPException(422, 'Переданы неизвестные настройки анализа.')
    result = {'trained': True, 'desk': True, 'people': True, 'interval': .5}
    result.update(value)
    if any(type(result[key]) is not bool for key in ('trained', 'desk', 'people')):
        raise HTTPException(422, 'Режимы анализа должны быть включены или выключены.')
    interval = result['interval']
    if isinstance(interval, bool) or not isinstance(interval, (float, int)) or interval not in (.5, 1, 2, 5):
        raise HTTPException(422, 'Выберите шаг анализа: 0,5, 1, 2 или 5 секунд.')
    if not any(result[key] for key in ('trained', 'desk', 'people')):
        raise HTTPException(422, 'Выберите хотя бы один режим анализа.')
    # Validate legacy clients, but never silently reduce the automatic review.
    return {'trained': True, 'desk': True, 'people': True, 'interval': .5}


def capabilities():
    return {
        'trained': (CASE/'training/rtdetr_custom_final/best.pth').is_file(),
        'desk': (CASE.parent/'GroundingDINO-local/weights/model.safetensors').is_file(),
        'people': (CASE.parent/'RT-DETR-main/rtdetrv2_pytorch/weights/rtdetrv2_r50vd_6x_coco_ema.pth').is_file()
                  and (CASE.parent/'ByteTrack-main/run_rtdetr_bytetrack.py').is_file(),
    }


def create_app(data_root: Path | None = None, manager_factory=JobManager, max_upload_bytes=MAX_UPLOAD):
    storage = Path(data_root or os.environ.get('FORTE_WEB_DATA', CASE/'web_data/jobs')).resolve()
    archive_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        storage.mkdir(parents=True, exist_ok=True)
        manager = manager_factory(storage)
        app.state.manager = manager
        try:
            yield
        finally:
            await run_in_threadpool(manager.shutdown)

    app = FastAPI(title='Проверка нарушений в кассе', docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(UploadSizeLimit, limit=max_upload_bytes + 1024 * 1024)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]'])

    @app.middleware('http')
    async def local_request_guard(request: Request, call_next):
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('origin')
            if origin:
                try:
                    parsed = urlsplit(origin)
                except ValueError:
                    return JSONResponse({'detail': 'Некорректный адрес страницы.'}, status_code=403)
                if parsed.scheme not in {'http', 'https'} or parsed.netloc.lower() != request.headers.get('host', '').lower():
                    return JSONResponse({'detail': 'Откройте страницу на локальном адресе приложения.'}, status_code=403)
            length = request.headers.get('content-length')
            if length:
                try:
                    if int(length) < 0 or int(length) > max_upload_bytes + 1024 * 1024:
                        return JSONResponse({'detail': 'Видео слишком большое. Максимум — 512 МБ.'}, status_code=413)
                except ValueError:
                    return JSONResponse({'detail': 'Некорректный размер запроса.'}, status_code=400)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    def job_or_404(job_id):
        if not JOB_ID.fullmatch(job_id):
            raise HTTPException(404, 'Такой обработки нет.')
        job = app.state.manager.get(job_id)
        if job is None:
            raise HTTPException(404, 'Такой обработки нет.')
        return job

    def cleanup_new_upload(folder):
        # Only a UUID directory created by this request is eligible for cleanup.
        if folder.parent.resolve() == storage and JOB_ID.fullmatch(folder.name):
            shutil.rmtree(folder, ignore_errors=True)

    @app.get('/api/health')
    async def health():
        return {'app': APP_ID, 'status': 'ok', 'version': '1.1'}

    @app.get('/api/config')
    async def config():
        return {'version': '1.1', 'max_upload_mb': max_upload_bytes // (1024 * 1024),
                'capabilities': capabilities(), 'analysis_mode': 'automatic_rule_review',
                'defaults': {'trained': True, 'desk': True, 'people': True, 'interval': .5}}

    @app.get('/api/jobs')
    async def list_jobs():
        return {'jobs': app.state.manager.list_jobs()}

    @app.post('/api/jobs', status_code=202)
    async def upload_video(file: UploadFile = File(...), options: str = Form('{}')):
        folder = None
        accepted = False
        try:
            settings = parse_options(options)
            available = capabilities()
            if any(settings[mode] and not available[mode] for mode in available):
                raise HTTPException(409, 'Не найдены файлы моделей для автоматической проверки. Проверьте папки проекта.')
            supplied = str(file.filename or '')
            name = supplied.replace('\\', '/').rsplit('/', 1)[-1]
            name = ''.join(char for char in name if char.isprintable())[:200]
            extension = Path(name).suffix.lower()
            if extension not in EXTENSIONS:
                raise HTTPException(415, 'Выберите видео MOV, MP4, MKV, AVI, M4V или WebM.')
            job_id = uuid.uuid4().hex
            folder = storage/job_id
            folder.mkdir(exist_ok=False)
            source = folder/f'input{extension}'
            temporary = folder/'upload.part'
            size = 0
            with temporary.open('xb') as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(413, 'Видео слишком большое. Максимум — 512 МБ.')
                    await run_in_threadpool(stream.write, chunk)
            if not size:
                raise HTTPException(422, 'Файл пустой. Выберите видео с записью.')
            temporary.replace(source)
            try:
                result = await run_in_threadpool(app.state.manager.enqueue, job_id, source, name, settings)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            accepted = True
            return result
        except OSError as exc:
            raise HTTPException(507, 'Не удалось сохранить видео. Проверьте свободное место на диске.') from exc
        finally:
            await file.close()
            if folder is not None and not accepted:
                cleanup_new_upload(folder)

    @app.get('/api/jobs/{job_id}')
    async def job(job_id: str):
        return job_or_404(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    async def cancel(job_id: str):
        job_or_404(job_id)
        return await run_in_threadpool(app.state.manager.cancel, job_id)

    @app.get('/api/jobs/{job_id}/results')
    async def results(job_id: str):
        state = job_or_404(job_id)
        return await run_in_threadpool(collect_results, storage/job_id, state)

    @app.get('/api/jobs/{job_id}/archive')
    async def archive(job_id: str):
        state = job_or_404(job_id)
        if state['status'] not in TERMINAL:
            raise HTTPException(409, 'Архив будет доступен после завершения обработки.')
        def package():
            with archive_lock:
                return build_archive(storage/job_id)
        archive_path = await run_in_threadpool(package)
        return FileResponse(archive_path, media_type='application/zip', filename=f'video-results-{job_id[:8]}.zip')

    @app.get('/api/jobs/{job_id}/files/{relative:path}')
    async def result_file(job_id: str, relative: str):
        job_or_404(job_id)
        root = (storage/job_id).resolve()
        path = (root/relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise HTTPException(404, 'Файл не найден.')
        allowed = {'.mp4', '.webm', '.jpg', '.jpeg', '.png', '.json', '.jsonl', '.csv', '.txt', '.log', '.html'}
        if not path.is_file() or path.suffix.lower() not in allowed or path.name.startswith('input.') or path.name == 'job.json':
            raise HTTPException(404, 'Файл не найден.')
        media = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix.lower() == '.mp4':
            media = 'video/mp4'
        elif path.suffix.lower() == '.webm':
            media = 'video/webm'
        return FileResponse(path, media_type=media, filename=path.name, content_disposition_type='inline')

    @app.get('/')
    async def index():
        return FileResponse(CASE/'web/index.html', media_type='text/html')

    from annotation_api import register_annotation_routes
    register_annotation_routes(app, CASE)
    app.mount('/static', StaticFiles(directory=CASE/'web', check_dir=False), name='static')
    return app


app = create_app()
