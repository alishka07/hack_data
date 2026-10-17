"""CPU-only regression checks for result provenance and sampled-video timing."""
import json
import queue
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile
from unittest.mock import patch

from web_jobs import (JobManager, _atomic_json, _people_decode_warning, build_archive,
                      collect_results, prepare_existing_browser_previews)


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.job = {'id': 'a' * 32, 'status': 'completed', 'options': {'interval': 2},
                    'stages': [{'key': 'trained', 'status': 'completed'}], '_source_file': 'input.mov'}
        self.result_dir = self.folder / 'trained/input'
        self.result_dir.mkdir(parents=True)
        self.summary = {'status': 'completed', 'decoded_frames': 180, 'analyzed_samples': 3,
                        'sample_interval_sec': 2, 'video_preview': 'review.mp4',
                        'thumbnails': [{'file': 'sample.jpg', 'timestamp_sec': 0}]}
        (self.result_dir / 'summary.json').write_text(json.dumps(self.summary), encoding='utf-8')
        (self.result_dir / 'review.mp4').write_bytes(b'fake-test-preview')
        (self.result_dir / 'sample.jpg').write_bytes(b'fake-test-thumbnail')
        records = []
        for index, pts in enumerate([0, 2.0333, 4.0667]):
            row = {'label': 'safe', 'score': .9, 'box': [10, 10, 30, 30]}
            records.append({'timestamp_sec': pts, 'source_frame': index * 61,
                            'candidates': [row], 'review': [],
                            'suppressed': [{'label': 'bag', 'score': .2, 'box': [1, 1, 2, 2]}]})
        (self.result_dir / 'frames.jsonl').write_text('\n'.join(json.dumps(record) for record in records), encoding='utf-8')

    def test_playback_is_sample_index_not_source_timestamp(self):
        result = collect_results(self.folder, self.job)
        self.assertEqual([row['playback_sec'] for row in result['objects']], [0, 2, 4])
        self.assertEqual([row['timestamp_sec'] for row in result['objects']], [0, 2.0333, 4.0667])
        self.assertEqual(result['source_frames'], {'trained': 180})
        self.assertEqual(result['analyzed_samples'], {'trained': 3})
        self.assertEqual(result['class_counts'], [{'label': 'safe', 'label_ru': 'Сейф', 'frames': 3, 'boxes': 3}])
        self.assertNotIn(result['verdict'], {'no_violations', 'compliant', 'passed'})
        self.assertIn('rule_coverage', result)

    def test_failed_stage_has_no_authoritative_objects(self):
        self.job['stages'][0]['status'] = 'failed'
        self.job['status'] = 'failed'
        (self.folder / 'trained.log').write_text('test error', encoding='utf-8')
        result = collect_results(self.folder, self.job)
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['videos'], [])
        self.assertTrue(any(row['url'].endswith('/trained.log') for row in result['downloads']))
        self.assertEqual(result['events'], [])
        self.assertTrue(all(row['status'] == 'not_checked' for row in result['rule_coverage']))

    def test_webm_is_preferred_and_mp4_remains_available(self):
        (self.result_dir / 'review.webm').write_bytes(b'webm-preview')
        result = collect_results(self.folder, self.job)
        video = result['videos'][0]
        self.assertTrue(video['url'].endswith('.webm'))
        self.assertEqual(video['url'], video['webm_url'])
        self.assertTrue(video['mp4_url'].endswith('.mp4'))
        self.assertTrue(any(item['url'] == video['mp4_url'] for item in result['downloads']))

    def test_export_file_lock_does_not_hide_rule_results(self):
        with patch('web_report.save_rule_report', side_effect=PermissionError('locked export')):
            result = collect_results(self.folder, self.job)
        self.assertIn('events', result)
        self.assertTrue(result['rule_coverage'])
        self.assertTrue(any('Не удалось сохранить' in row for row in result['limitations']))
        self.assertFalse(any('/files/report/' in row['url'] for row in result['downloads']))

    def test_archive_excludes_uploaded_source_and_itself(self):
        (self.folder / 'job.json').write_text(json.dumps(self.job), encoding='utf-8')
        (self.folder / 'input.mov').write_bytes(b'private-original-video')
        nested = self.folder / 'report'
        nested.mkdir(exist_ok=True)
        (nested / 'pending.json.test.tmp').write_text('incomplete')
        archive = build_archive(self.folder)
        archive = build_archive(self.folder)
        with zipfile.ZipFile(archive) as stream:
            names = stream.namelist()
            self.assertNotIn('input.mov', names)
            self.assertNotIn('results.zip', names)
            self.assertFalse(any(name.endswith('.tmp') for name in names))
            self.assertIn('trained/input/review.mp4', names)
            self.assertIn('result.json', names)

    def test_truncated_people_success_still_warns(self):
        job_id = 'b' * 32
        folder = self.folder / job_id
        people = folder / 'people'
        people.mkdir(parents=True)
        (people / 'review.mp4').write_bytes(b'partial-preview')
        (people / 'review.json').write_text(json.dumps({
            'frames': 114, 'last_timestamp_sec': 3.77, 'track_ids': [1, 2],
        }), encoding='utf-8')
        job = {'id': job_id, 'video': {'frames': 442, 'fps': 30},
               'stages': [{'key': 'people', 'status': 'completed'}]}
        manager = object.__new__(JobManager)
        manager.root = self.folder
        manager._lock = threading.RLock()
        self.assertTrue(manager._validate_stage(job, 'people'))
        self.assertIn('114 из 442', job['stages'][0]['warning'])
        result = collect_results(folder, job)
        self.assertIn('Обработана только часть', result['people']['warning'])
        self.assertTrue(any(row.startswith('Люди:') and '114 из 442' in row for row in result['limitations']))

    def test_people_timestamp_mismatch_warns_even_when_frame_count_matches(self):
        job_id = 'c' * 32
        people = self.folder / job_id / 'people'
        people.mkdir(parents=True)
        (people / 'review.mp4').write_bytes(b'preview')
        (people / 'review.json').write_text(json.dumps({'frames': 442, 'last_timestamp_sec': 3.77}), encoding='utf-8')
        job = {'id': job_id, 'video': {'frames': 442, 'fps': 30},
               'stages': [{'key': 'people', 'status': 'completed'}]}
        manager = object.__new__(JobManager)
        manager.root = self.folder
        manager._lock = threading.RLock()
        self.assertTrue(manager._validate_stage(job, 'people'))
        self.assertIn('последний кадр', job['stages'][0]['warning'])

    def test_complete_people_report_tolerates_small_pts_offset(self):
        self.assertIsNone(_people_decode_warning(
            {'frames': 442, 'last_timestamp_sec': 14.733}, {'frames': 442, 'fps': 30}))

    def test_temporary_windows_file_lock_keeps_snapshot_and_recovers(self):
        path = self.folder / 'state.json'
        path.write_text('{"old": true}', encoding='utf-8')
        replace = Path.replace
        calls = []
        def briefly_locked(source, target):
            calls.append(1)
            if len(calls) < 3:
                self.assertEqual(json.loads(path.read_text()), {'old': True})
                raise PermissionError('Temporary Windows sharing violation')
            return replace(source, target)
        with patch.object(Path, 'replace', briefly_locked), patch('web_jobs.time.sleep'):
            _atomic_json(path, {'complete': True})
        self.assertEqual(json.loads(path.read_text()), {'complete': True})
        self.assertEqual(list(self.folder.glob('state.json.*.tmp')), [])


class DenseWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job_id = 'd' * 32
        self.folder = self.root / self.job_id
        self.folder.mkdir()
        (self.folder/'input.mov').write_bytes(b'fake-source')
        self.job = {'id': self.job_id, 'status': 'completed', '_source_file': 'input.mov',
                    'video': {'frames': 600, 'fps': 30., 'duration': 20., 'width': 400, 'height': 500},
                    'options': {'desk': True, 'interval': .5},
                    'stages': [{'key': 'dense_review', 'status': 'completed'}]}
        self.dense = self.folder/'dense_review'
        self.dense.mkdir()
        self.summary = {'status': 'completed', 'decoded_frames': 600, 'frame_count_matches': True,
                        'all_planned_samples_analyzed': True, 'planned_samples': 2,
                        'analyzed_samples': 2, 'planned_window_seconds': 3.,
                        'window_coverage_fraction': .15, 'source_sha256': 'a'*64}
        self.rows = [{'source_frame': i, 'timestamp_sec': i/30, 'candidates': [
            {'label':'cash', 'box':[30,40,70,50], 'score':.7}], 'review': []} for i in (30,36)]
        self.save_dense()

    def save_dense(self):
        (self.dense/'summary.json').write_text(json.dumps(self.summary), encoding='utf-8')
        (self.dense/'observations.jsonl').write_text('\n'.join(json.dumps(r) for r in self.rows), encoding='utf-8')
        (self.folder/'job.json').write_text(json.dumps(self.job), encoding='utf-8')

    def manager(self):
        manager = object.__new__(JobManager)
        manager.root = self.root
        manager._lock = threading.RLock()
        manager._closed = False
        manager._jobs = {}
        manager._cancellations = {}
        manager._queue = queue.Queue()
        return manager

    def test_new_upload_does_not_schedule_expensive_unvalidated_stage_by_default(self):
        with patch.dict('os.environ', {'CASE_ENABLE_DENSE_REVIEW': ''}), \
             patch('web_jobs._video_metadata', return_value=self.job['video']):
            job = self.manager().enqueue(self.job_id, self.folder/'input.mov', 'X.mov',
                                         {'trained':True, 'desk':True, 'people':True})
        self.assertNotIn('dense_review', [s['key'] for s in job['stages']])
        self.assertFalse(job['options']['dense_review'])

    def test_explicit_server_opt_in_schedules_dense_after_dependencies_with_bound(self):
        with patch.dict('os.environ', {'CASE_ENABLE_DENSE_REVIEW': '1'}), \
             patch('web_jobs._video_metadata', return_value=self.job['video']):
            job = self.manager().enqueue(self.job_id, self.folder/'input.mov', 'X.mov',
                                         {'trained':True, 'desk':True, 'people':True})
        self.assertEqual(job['stages'][-1]['key'], 'dense_review')
        command = self.manager()._command(self.job, 'dense_review')
        self.assertEqual(command[command.index('--max-seconds')+1], '12')
        self.assertEqual(command[command.index('--sample-fps')+1], '5')

    def test_server_opt_in_still_requires_desk_stage(self):
        with patch.dict('os.environ', {'CASE_ENABLE_DENSE_REVIEW': '1'}), \
             patch('web_jobs._video_metadata', return_value=self.job['video']):
            job = self.manager().enqueue(self.job_id, self.folder/'input.mov', 'X.mov', {'people':True})
        self.assertFalse(job['options']['dense_review'])

    def test_complete_closeup_is_separate_from_rule_objects(self):
        result = collect_results(self.folder, self.job)
        closeup = result['closeup_review']
        self.assertEqual(closeup['status'], 'completed')
        self.assertEqual(len(closeup['observations']), 2)
        self.assertFalse(closeup['used_for_rule_verdicts'])
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['events'], [])
        self.assertNotIn(result['verdict'], {'compliant', 'no_violations', 'passed'})
        self.assertIn('только в выбранных интервалах', closeup['note_ru'])

    def test_not_checked_cannot_claim_regions_were_analyzed(self):
        self.summary = {'status':'not_checked', 'reason':'no_stable_table', 'analyzed_samples':0}
        self.save_dense()
        self.assertTrue(self.manager()._validate_stage(self.job, 'dense_review'))
        result = collect_results(self.folder, self.job)
        closeup = result['closeup_review']
        self.assertEqual(closeup['status'], 'not_checked')
        self.assertEqual(closeup['observations'], [])
        self.assertIn('не выполнена', closeup['note_ru'])
        self.assertNotIn('области проверены', closeup['note_ru'])
        self.assertIn('Это не означает', closeup['note_ru'])

    def test_partial_failed_stage_remains_explicit_diagnostic(self):
        self.job['stages'][0]['status'] = 'failed'
        self.summary['status'] = 'partial'
        self.summary['analyzed_samples'] = 1
        self.rows = self.rows[:1]
        self.save_dense()
        result = collect_results(self.folder, self.job)
        self.assertEqual(result['closeup_review']['status'], 'partial')
        self.assertIn('частично', result['closeup_review']['title_ru'])
        self.assertEqual(result['closeup_review']['observations'], [])
        self.assertEqual(result['objects'], [])
        self.assertTrue(any(d['url'].endswith('observations.jsonl') for d in result['downloads']))

    def test_malformed_journal_does_not_break_whole_report_or_claim_completion(self):
        (self.dense/'observations.jsonl').write_text('{broken', encoding='utf-8')
        self.assertFalse(self.manager()._validate_stage(self.job, 'dense_review'))
        result = collect_results(self.folder, self.job)
        self.assertEqual(result['closeup_review']['status'], 'partial')
        self.assertEqual(result['closeup_review']['observations'], [])
        self.assertIn('rule_coverage', result)

    def test_success_flags_alone_do_not_prove_completed_sampling(self):
        for change in ({'decoded_frames':599}, {'analyzed_samples':3}, {'planned_samples':3}):
            with self.subTest(change=change):
                original = dict(self.summary)
                self.summary.update(change)
                self.save_dense()
                self.assertFalse(self.manager()._validate_stage(self.job, 'dense_review'))
                self.summary = original

    def test_duplicate_or_nonmonotonic_journal_frames_fail_validation(self):
        self.rows[1]['source_frame'] = self.rows[0]['source_frame']
        self.save_dense()
        self.assertFalse(self.manager()._validate_stage(self.job, 'dense_review'))

    def test_missing_summary_does_not_pass_stage_validation(self):
        (self.dense/'summary.json').unlink()
        self.assertFalse(self.manager()._validate_stage(self.job, 'dense_review'))

    def test_browser_maintenance_never_searches_for_dense_or_state_videos(self):
        self.job['stages'].append({'key':'visual_states','status':'completed'})
        self.save_dense()
        with patch('web_jobs._stage_video', side_effect=AssertionError('No video for observations-only stage')):
            self.assertEqual(prepare_existing_browser_previews(self.folder), [])


if __name__ == '__main__':
    unittest.main()
