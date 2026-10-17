"""CPU API boundary tests. Fake queue; actual tiny video decoding; no model runs."""
from pathlib import Path
import json
import tempfile
import unittest
import uuid

import cv2
import numpy as np
from fastapi.testclient import TestClient

from web_app import create_app
from web_jobs import _video_metadata


class FakeManager:
    def __init__(self, root):
        self.root = Path(root)
        self.jobs = {}
        self.closed = False

    def enqueue(self, job_id, source_path, original_name, options):
        metadata = _video_metadata(source_path)
        job = {'id': job_id, 'name': original_name, 'status': 'queued',
               'options': options, 'video': metadata, 'stages': []}
        self.jobs[job_id] = job
        return dict(job)

    def list_jobs(self):
        return [dict(value) for value in self.jobs.values()]

    def get(self, job_id):
        job = self.jobs.get(job_id)
        return dict(job) if job else None

    def cancel(self, job_id):
        self.jobs[job_id]['status'] = 'cancelled'
        return self.get(job_id)

    def shutdown(self):
        self.closed = True


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='web_api_')
        self.root = Path(self.temporary.name)
        source = self.root/'fixture.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 64))
        assert writer.isOpened()
        for i in range(5):
            writer.write(np.full((64, 64, 3), i*30, dtype=np.uint8))
        writer.release()
        self.payload = source.read_bytes()
        self.storage = self.root/'jobs'
        self.app = create_app(self.storage, manager_factory=FakeManager, max_upload_bytes=24*1024)
        self.client = TestClient(self.app, base_url='http://127.0.0.1:8765', raise_server_exceptions=False)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    def upload(self, filename='clip.avi', payload=None, options=None, headers=None):
        return self.client.post('/api/jobs', files={'file': (filename, self.payload if payload is None else payload, 'video/x-msvideo')},
                                data={'options': json.dumps(options or {})}, headers=headers or {})

    def accepted(self):
        response = self.upload()
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    def test_real_video_upload_sanitizes_filename_and_cancel(self):
        response = self.upload(r'C:\private\..\evil.avi')
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual(job['name'], 'evil.avi')
        self.assertEqual(job['video']['frames'], 5)
        self.assertEqual(job['options'], {'trained': True, 'desk': True, 'people': True, 'interval': .5})
        self.assertEqual([p.name for p in (self.storage/job['id']).iterdir()], ['input.avi'])
        self.assertEqual(self.client.post('/api/jobs/'+job['id']+'/cancel').json()['status'], 'cancelled')

    def test_legacy_options_cannot_disable_automatic_checks(self):
        response = self.upload(options={'trained': True, 'desk': False, 'people': False, 'interval': 5})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()['options'], {'trained': True, 'desk': True, 'people': True, 'interval': .5})

    def test_invalid_uploads_do_not_queue_or_leave_folders(self):
        for name, data, status in [('bad.exe', b'123', 415), ('empty.mp4', b'', 422), ('corrupt.mp4', b'123', 422), ('large.mp4', b'x'*(24*1024+1), 413)]:
            with self.subTest(name=name):
                response = self.upload(name, data)
                self.assertEqual(response.status_code, status, response.text)
                self.assertFalse(list(self.storage.iterdir()))
                self.assertFalse(self.app.state.manager.list_jobs())

    def test_invalid_options_reject_without_exception(self):
        for options in [{'interval': float('nan')}, {'interval': float('inf')}, {'interval': 0}, {'interval': True}, {'interval': 10**400}, {'trained': 'true'}, {'mode': 'unknown'}, {'trained': False, 'desk': False, 'people': False}]:
            with self.subTest(options=options):
                response = self.upload(options=options)
                self.assertEqual(response.status_code, 422, response.text)

    def test_chunked_upload_limit_before_spooling(self):
        boundary = 'local-test-boundary'
        def body():
            yield (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="large.mp4"\r\nContent-Type: video/mp4\r\n\r\n').encode()
            for _ in range(18):
                yield b'x' * (64 * 1024)
            yield f'\r\n--{boundary}--\r\n'.encode()
        response = self.client.post('/api/jobs', content=body(), headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        self.assertEqual(response.status_code, 413, response.text)
        self.assertFalse(list(self.storage.iterdir()))

    def test_origin_and_host_boundary(self):
        for origin in ['https://other.example', 'null', 'http://[invalid']:
            with self.subTest(origin=origin):
                response = self.upload(headers={'Origin': origin})
                self.assertEqual(response.status_code, 403, response.text)
        response = self.upload(headers={'Origin': 'http://127.0.0.1:8765'})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(self.client.get('/api/health', headers={'Host': 'outside.example'}).status_code, 400)

    def test_media_range_and_path_boundary(self):
        identity = self.accepted()
        folder = self.storage/identity
        (folder/'review.mp4').write_bytes(bytes(range(256)))
        (folder/'job.json').write_text('{}')
        (self.storage/'secret.txt').write_text('secret')
        base = f'/api/jobs/{identity}/files/'
        response = self.client.get(base+'review.mp4', headers={'Range': 'bytes=10-19'})
        self.assertEqual(response.status_code, 206, response.text)
        self.assertEqual(response.content, bytes(range(10,20)))
        self.assertEqual(response.headers['content-range'], 'bytes 10-19/256')
        for path in ['input.avi', 'job.json', '..%2fsecret.txt', '..%5csecret.txt', 'C:%5cWindows%5cwin.ini', '%00.txt']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(base+path).status_code, 404)
        self.assertEqual(self.client.get(f'/api/jobs/{identity}/archive').status_code, 409)


if __name__ == '__main__':
    unittest.main(verbosity=2)
