"""CPU evidence tests: isolated detections and incomplete tracking are not events."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rule_engine import build_rule_report, _load_modes, _food_drink_events, _object_tracks, _near_cashier_workplace


def item(label, box, score=.9):
    return {'label': label, 'box': box, 'score': score}


TABLE = item('work_table', [80, 100, 640, 400], .8)
PHONE = item('smartphone', [220, 220, 245, 255])
LAPTOP = item('laptop', [380, 200, 500, 300])
CASH = item('cash', [550, 200, 580, 220])
FOOD = item('food', [550, 200, 580, 220])
DRINK = item('drink', [350, 160, 380, 195])


class RuleEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job = {'id': 'test', 'video': {'frames': 80, 'fps': 10., 'width': 640, 'height': 480, 'duration': 8},
                    'stages': [], 'options': {'interval': 1}}

    def write_mode(self, mode, rows, review_only=False, groups=None, interval=1):
        folder = self.root / mode / 'input'
        folder.mkdir(parents=True, exist_ok=True)
        summary = {'status': 'completed', 'analyzed_samples': len(rows), 'sample_interval_sec': interval,
                   'decoded_frames': 80, 'metadata_frames': 80}
        if groups is not None:
            summary['signature'] = {'groups': groups}
        (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        records = [{'source_frame': round(index * 10 * interval), 'timestamp_sec': float(index) * interval,
                    'candidates': [] if review_only else objects,
                    'review': objects if review_only else [], 'suppressed': []} for index, objects in enumerate(rows)]
        (folder / 'frames.jsonl').write_text('\n'.join(json.dumps(record) for record in records), encoding='utf-8')
        self.job['stages'].append({'key': mode, 'status': 'completed'})

    def write_people(self, counts, report_frames=80, boxes=None, last_timestamp=None):
        folder = self.root / 'people'
        folder.mkdir(parents=True, exist_ok=True)
        ids = set()
        with (folder / 'review.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_frame', 'timestamp_sec', 'track_id', 'class', 'score', 'x1', 'y1', 'x2', 'y2'])
            for frame, count in enumerate(counts):
                for identity in range(1, count + 1):
                    ids.add(identity)
                    box = boxes(frame, identity) if boxes else [50 + identity * 140, 80, 140 + identity * 140, 350]
                    writer.writerow([frame, frame / 10, identity, 'person', .9, *box])
        report = {'frames': report_frames, 'metadata_frames': 80, 'source_fps': 10,
                  'first_timestamp_sec': 0, 'last_timestamp_sec': (report_frames - 1) / 10 if last_timestamp is None else last_timestamp,
                  'full_video_requested': True, 'frame_count_matches': report_frames == 80,
                  'track_ids': sorted(ids), 'decoding_warning': None,
                  'csv_sha256': hashlib.sha256((folder / 'review.csv').read_bytes()).hexdigest(), 'csv_rows': sum(counts)}
        (folder / 'review.json').write_text(json.dumps(report), encoding='utf-8')
        self.job['stages'].append({'key': 'people', 'status': 'completed'})

    def report(self):
        return build_rule_report(self.root, self.job, {})

    def events(self, rule_id):
        return [event for event in self.report()['events'] if event['rule_id'] == rule_id]

    def coverage(self, rule_id):
        return next(row for row in self.report()['rule_coverage'] if row['rule_id'] == rule_id)

    def test_single_phone_never_becomes_event(self):
        self.write_mode('desk', [[TABLE], [TABLE, PHONE], [TABLE], [TABLE]])
        self.assertEqual(self.events('clean_desk_banned_items'), [])

    def test_repeated_weak_usb_lock_is_not_an_event(self):
        lock = item('usb_drive', [55, 473, 85, 499], .346)
        self.write_mode('desk', [[TABLE, lock] for _ in range(8)])
        self.assertEqual(self.events('clean_desk_banned_items'), [])

    def test_stationary_phone_without_table_is_not_checked(self):
        self.write_mode('desk', [[PHONE] for _ in range(8)])
        self.assertEqual(self.events('clean_desk_banned_items'), [])
        self.assertEqual(self.coverage('clean_desk_banned_items')['status'], 'not_checked')

    def test_sustained_phone_on_candidate_table_is_one_review_episode(self):
        self.write_mode('desk', [[TABLE, PHONE] for _ in range(4)] + [[TABLE] for _ in range(4)])
        events = self.events('clean_desk_banned_items')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['start_sec'], 0)
        self.assertEqual(events[0]['end_sec'], 3)
        self.assertEqual(events[0]['confidence_label'], 'Требует проверки')
        self.assertEqual(events[0]['evidence_box'], PHONE['box'])
        self.assertEqual(len(events[0]['evidence_times']), 4)
        self.assertEqual(self.coverage('clean_desk_banned_items')['status'], 'partial')

    def test_phone_inside_other_modes_laptop_is_suppressed(self):
        phone = item('smartphone', [400, 220, 425, 255])
        self.write_mode('desk', [[TABLE, phone] for _ in range(8)])
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.assertEqual(self.events('clean_desk_banned_items'), [])

    def test_empty_sample_breaks_phone_episode(self):
        self.write_mode('desk', [[TABLE, PHONE], [TABLE], [TABLE, PHONE], [TABLE], [TABLE, PHONE]])
        self.assertEqual(self.events('clean_desk_banned_items'), [])

    def test_ambiguous_review_rows_are_not_temporal_evidence(self):
        self.write_mode('desk', [[TABLE, PHONE] for _ in range(8)], review_only=True)
        self.assertEqual(self.report()['events'], [])

    def test_legacy_desk_journal_cannot_claim_food_check(self):
        self.write_mode('desk', [[TABLE, FOOD] for _ in range(8)])
        self.assertEqual(self.events('food_drinks_at_workplace'), [])
        self.assertEqual(self.coverage('food_drinks_at_workplace')['status'], 'not_checked')

    def test_stationary_food_needs_no_cash_people_or_customer_departure(self):
        self.write_mode('desk', [[TABLE, FOOD] for _ in range(8)], groups=['desk', 'food_drink'])
        events = self.events('food_drinks_at_workplace')
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]['context']['requires_cash'])
        self.assertFalse(events[0]['context']['requires_client_departure'])
        self.assertFalse(events[0]['context']['consumption_action_checked'])
        self.assertEqual(events[0]['labels'], ['food'])
        self.assertEqual(self.coverage('food_drinks_at_workplace')['status'], 'partial')

    def test_food_presence_does_not_depend_on_client_presence(self):
        self.write_mode('desk', [[TABLE, FOOD] for _ in range(8)], groups=['food_drink'])
        self.write_people([2] * 80)
        self.assertEqual(len(self.events('food_drinks_at_workplace')), 1)
        self.assertEqual(self.events('eating_while_cash_present'), [])

    def test_empty_food_sample_breaks_temporal_evidence(self):
        self.write_mode('desk', [[TABLE, FOOD], [TABLE], [TABLE, FOOD], [TABLE], [TABLE, FOOD]], groups=['food_drink'])
        self.assertEqual(self.events('food_drinks_at_workplace'), [])

    def test_stable_service_table_bridges_missing_table_not_missing_food(self):
        self.write_mode('desk', [[TABLE, FOOD], [FOOD], [TABLE, FOOD], [FOOD], [TABLE, FOOD]], groups=['food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        service = {'role_summary': {'status': 'candidate', 'cashier_track_id': 1,
                                   'anchors': {'source': 'stable_model_candidates', 'table_box': TABLE['box']}}}
        events = _food_drink_events(_object_tracks(modes), modes, None, True, {'complete': False}, service)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['context']['workspace_source'], 'stable_model_anchor')
        self.assertEqual((events[0]['start_sec'], events[0]['end_sec']), (0, 4))
        self.assertEqual(len(events[0]['evidence_times']), 5)
        self.write_mode('desk', [[TABLE, FOOD], [TABLE], [TABLE, FOOD], [TABLE], [TABLE, FOOD]], groups=['food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        self.assertEqual(_food_drink_events(_object_tracks(modes), modes, None, True, {'complete': False}, service), [])

    def test_unknown_table_anchor_does_not_bridge_context_gaps(self):
        self.write_mode('desk', [[TABLE, FOOD], [FOOD], [TABLE, FOOD], [FOOD], [TABLE, FOOD]], groups=['food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        service = {'role_summary': {'status': 'unknown', 'anchors': {'source': 'stable_model_candidates', 'table_box': TABLE['box']}}}
        self.assertEqual(_food_drink_events(_object_tracks(modes), modes, None, True, {'complete': False}, service), [])
        service['role_summary'].update(status='candidate', anchors={'source': 'unverified', 'table_box': TABLE['box']})
        self.assertEqual(_food_drink_events(_object_tracks(modes), modes, None, True, {'complete': False}, service), [])

    def test_user_workspace_takes_priority_over_stable_model_anchor(self):
        self.write_mode('desk', [[TABLE, FOOD] for _ in range(8)], groups=['food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        service = {'role_summary': {'status': 'candidate', 'anchors': {'source': 'stable_model_candidates', 'table_box': TABLE['box']}}}
        self.assertEqual(_food_drink_events(_object_tracks(modes), modes, [100, 100, 200, 200], True, {'complete': False}, service), [])

    def test_food_three_half_second_samples_form_presence_episode(self):
        self.write_mode('desk', [[TABLE, DRINK]] * 3, groups=['food_drink'], interval=.5)
        events = self.events('food_drinks_at_workplace')
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]['start_sec'], events[0]['end_sec']), (0, 1))
        self.assertFalse(events[0]['context']['consumption_action_checked'])
        settings = self.report()['analysis_settings']
        self.assertEqual(settings['minimum_evidence_span_sec'], 2)
        self.assertEqual(settings['food_drinks_policy']['minimum_evidence_span_sec'], 1)

    def test_one_or_two_food_samples_or_gaps_do_not_form_presence_episode(self):
        for rows in ([[TABLE, DRINK]], [[TABLE, DRINK]] * 2,
                     [[TABLE, DRINK], [TABLE], [TABLE, DRINK], [TABLE], [TABLE, DRINK]]):
            self.write_mode('desk', rows, groups=['food_drink'], interval=.5)
            self.assertEqual(self.events('food_drinks_at_workplace'), [])

    def test_phone_still_requires_two_second_span(self):
        self.write_mode('desk', [[TABLE, PHONE]] * 3, groups=['desk'], interval=.5)
        self.assertEqual(self.events('clean_desk_banned_items'), [])

    def test_unlocalized_food_and_weak_food_are_not_events(self):
        self.write_mode('desk', [[TABLE, item('food', [5, 5, 20, 20]), item('drink', DRINK['box'], .27)] for _ in range(8)], groups=['food_drink'])
        self.assertEqual(self.events('food_drinks_at_workplace'), [])

    def test_repeated_small_cup_uses_separate_score_threshold_and_needs_time(self):
        cup = item('drink', FOOD['box'], .41)
        self.write_mode('desk', [[TABLE, cup] for _ in range(8)], groups=['food_drink'])
        self.assertEqual(len(self.events('food_drinks_at_workplace')), 1)
        self.write_mode('desk', [[TABLE, item('drink', FOOD['box'], .39)] for _ in range(8)], groups=['food_drink'])
        self.assertEqual(self.events('food_drinks_at_workplace'), [])
        self.write_mode('desk', [[TABLE, cup], [TABLE, cup]], groups=['food_drink'])
        self.assertEqual(self.events('food_drinks_at_workplace'), [])

    def test_food_competing_class_removes_cash_for_all_downstream_rules(self):
        self.write_mode('desk', [[TABLE, CASH, FOOD] for _ in range(8)], groups=['cash', 'food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        self.assertTrue(all(all(obj['label'] != 'cash' for obj in row['objects']) for row in modes['desk']['records']))
        self.assertEqual(len(modes['desk']['cash_food_conflicts']), 8)
        self.assertEqual(self.events('food_drinks_at_workplace'), [])
        self.assertTrue(self.report()['data_quality']['cash_food_class_conflicts'])
        self.assertTrue(self.report()['data_quality']['food_cash_class_conflicts'])

    def test_partial_cash_food_overlap_cannot_turn_ambiguity_into_food(self):
        # IoU=.429: older candidate filters accepted both (their cutoff was .55).
        cash = item('cash', [200, 200, 220, 220], .8)
        food = item('food', [208, 200, 228, 220], .45)
        self.write_mode('desk', [[TABLE, cash, food] for _ in range(8)], groups=['cash', 'food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        self.assertTrue(all(all(obj['label'] not in {'cash', 'food'} for obj in row['objects']) for row in modes['desk']['records']))
        self.assertEqual(self.events('food_drinks_at_workplace'), [])

    def test_review_cash_also_blocks_overlapping_model_food(self):
        self.write_mode('desk', [[TABLE, FOOD] for _ in range(8)], groups=['food_drink'])
        path = self.root / 'desk' / 'input' / 'frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            row['review'] = [CASH]
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        self.assertEqual(self.events('food_drinks_at_workplace'), [])

    def test_cross_mode_cash_food_ambiguity_blocks_both_hypotheses(self):
        self.write_mode('trained', [[CASH] for _ in range(8)])
        self.write_mode('desk', [[TABLE, FOOD] for _ in range(8)], groups=['food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        self.assertFalse(any(obj['label'] in {'cash', 'food'} for data in modes.values() for row in data['records'] for obj in row['objects']))
        self.assertEqual(self.events('food_drinks_at_workplace'), [])

    def test_human_reference_resolves_cash_food_conflict_without_rewriting_raw(self):
        reviewed = dict(FOOD, source='human_confirmed_reference_match', score_kind='reference_similarity_not_probability')
        self.write_mode('desk', [[TABLE, CASH, reviewed] for _ in range(8)], groups=['cash', 'food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        self.assertFalse(any(obj['label'] == 'cash' for row in modes['desk']['records'] for obj in row['objects']))
        self.assertEqual(len(self.events('food_drinks_at_workplace')), 1)
        self.assertEqual(self.events('food_drinks_at_workplace')[0]['context']['object_evidence_source'], 'human_confirmed_reference_match')
        source = json.loads((self.root / 'desk' / 'input' / 'frames.jsonl').read_text(encoding='utf-8').splitlines()[0])
        self.assertIn('cash', [obj['label'] for obj in source['candidates']])

    def test_mixed_food_episode_keeps_per_frame_evidence_sources(self):
        reviewed = dict(FOOD, source='human_confirmed_reference_match', score_kind='reference_similarity_not_probability')
        self.write_mode('desk', [[TABLE, FOOD], [TABLE, reviewed], [TABLE, reviewed], [TABLE, FOOD]], groups=['food_drink'])
        event = self.events('food_drinks_at_workplace')[0]
        self.assertIn('В части кадров', event['description'])
        self.assertEqual(event['context']['object_evidence_source'], 'mixed_model_and_human_reference')
        self.assertEqual(event['context']['reference_match_observations'], 2)
        self.assertEqual(event['context']['model_observations'], 2)
        self.assertEqual([row['source'] for row in event['evidence_times']], ['model', 'human_confirmed_reference_match', 'human_confirmed_reference_match', 'model'])
        timed = [step for step in event['steps'] if 'timestamp_sec' in step]
        self.assertTrue(all('класс предложен моделью' in step['detail'] for step in timed))

    def test_ambiguous_food_can_reject_cash_without_becoming_food_event(self):
        self.write_mode('desk', [[TABLE, CASH] for _ in range(8)], groups=['cash', 'food_drink'])
        path = self.root / 'desk' / 'input' / 'frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            row['review'] = [item('food', FOOD['box'], .31)]
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        modes, _ = _load_modes(self.root, self.job)
        self.assertTrue(all(all(obj['label'] != 'cash' for obj in row['objects']) for row in modes['desk']['records']))
        self.assertEqual(self.events('food_drinks_at_workplace'), [])

    def test_cup_near_cashier_is_presence_not_drinking(self):
        table = item('work_table', [80, 200, 640, 400])
        cup = item('drink', [220, 160, 245, 190])
        self.write_mode('desk', [[table, cup] for _ in range(8)], groups=['food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        people = {'complete': True, 'by_frame': {frame: [{'id': 1, 'box': [180, 70, 280, 350]}] for frame in range(80)}}
        service = {'role_summary': {'status': 'candidate', 'cashier_track_id': 1}}
        events = _food_drink_events(_object_tracks(modes), modes, None, True, people, service)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['context']['spatial_relations'], ['cashier_near_table'])
        self.assertFalse(events[0]['context']['consumption_action_checked'])
        self.assertEqual(events[0]['context']['cashier_track_id'], 1)

    def test_client_cup_outside_table_is_not_cashier_food_event(self):
        table = item('work_table', [80, 200, 640, 400])
        cup = item('drink', [450, 160, 475, 190])
        self.write_mode('desk', [[table, cup] for _ in range(8)], groups=['food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        people = {'complete': True, 'by_frame': {frame: [{'id': 1, 'box': [180, 70, 280, 350]}, {'id': 2, 'box': [400, 70, 500, 350]}] for frame in range(80)}}
        service = {'role_summary': {'status': 'candidate', 'cashier_track_id': 1}}
        self.assertEqual(_food_drink_events(_object_tracks(modes), modes, None, True, people, service), [])

    def test_ambiguous_overlapping_people_cannot_assign_held_cup(self):
        self.write_mode('desk', [[TABLE, DRINK] for _ in range(8)], groups=['food_drink'])
        modes, _ = _load_modes(self.root, self.job)
        observation = dict(DRINK, time=0, frame=0)
        people = {'complete': True, 'by_frame': {0: [{'id': 1, 'box': [300, 70, 400, 350]}, {'id': 2, 'box': [320, 70, 420, 350]}]}}
        service = {'role_summary': {'status': 'candidate', 'cashier_track_id': 1}}
        self.assertFalse(_near_cashier_workplace(observation, modes, None, people, service))

    def test_truncated_people_suppresses_unattended_alert(self):
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.write_people([1] * 20 + [0] * 10, report_frames=30)
        self.assertEqual(self.events('unattended_assets'), [])
        self.assertEqual(self.coverage('unattended_assets')['status'], 'not_checked')
        self.assertFalse(self.report()['data_quality']['people_complete'])

    def short_tail_fixture(self, counts=None, cash_rows=None, people_frames=80):
        safe = item('safe', [300, 10, 420, 80])
        if cash_rows is None:
            cash_rows = [[CASH] for _ in range(16)]
        self.write_mode('desk', [[TABLE, safe, *objects] for objects in cash_rows], interval=.5)
        self.write_people(counts if counts is not None else [1] * 63 + [0] * 17,
                          report_frames=people_frames, boxes=lambda frame, identity: [190, 40, 280, 220])

    def short_tail_events(self):
        return [event for event in self.events('unattended_assets')
                if event['context'].get('recording_ends_during_absence')]

    def test_short_eof_tail_is_separate_cash_review_without_asserting_any_exit(self):
        cash = dict(CASH, source='human_confirmed_reference_match', score_kind='reference_similarity_not_probability')
        self.short_tail_fixture(cash_rows=[[cash] for _ in range(16)])
        events = self.short_tail_events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event['start_sec'], event['end_sec']), (6.5, 7.5))
        self.assertEqual(event['labels'], ['cash'])
        self.assertEqual(event['source_frame'], 65)
        self.assertAlmostEqual(event['context']['absence_duration_sec'], 1.7)
        self.assertFalse(event['context']['cashier_exit_observed'])
        self.assertFalse(event['context']['visitor_exit_observed'])
        self.assertTrue(any(row['source_frame'] == 62 and row['labels'] == ['person'] for row in event['evidence_times']))
        self.assertTrue(all(row['source'] == 'human_confirmed_reference_match'
                            for row in event['evidence_times'] if row['labels'] == ['cash']))
        self.assertEqual(self.events('cash_after_client'), [])
        self.assertEqual(self.events('unattended_room'), [])
        settings = self.report()['analysis_settings']
        self.assertEqual(settings['minimum_evidence_span_sec'], 2)
        self.assertEqual(settings['short_clip_cash_review']['minimum_absence_sec'], 1)
        self.assertTrue(any(check['check_id'] == 'cash_at_truncated_recording_end' and check['status'] == 'observed'
                            for check in self.coverage('unattended_assets')['checks']))

    def test_short_tail_requires_a_full_second_and_never_handles_internal_gap(self):
        for counts in ([1] * 71 + [0] * 9, [1] * 40 + [0] * 17 + [1] * 23):
            self.short_tail_fixture(counts)
            self.assertEqual(self.short_tail_events(), [])

    def test_short_tail_does_not_duplicate_normal_two_second_absence(self):
        self.short_tail_fixture([1] * 40 + [0] * 40)
        self.assertEqual(self.short_tail_events(), [])
        self.assertEqual(len(self.events('unattended_assets')), 1)

    def test_short_tail_rejects_initial_empty_video_and_unknown_role(self):
        self.short_tail_fixture([0] * 80)
        self.assertEqual(self.short_tail_events(), [])
        self.write_mode('desk', [[TABLE, CASH] for _ in range(16)], interval=.5)
        self.write_people([1] * 63 + [0] * 17)
        self.assertEqual(self.short_tail_events(), [])

    def test_short_tail_rejects_incomplete_decode_and_malformed_people(self):
        self.short_tail_fixture([1] * 63 + [0] * 10, people_frames=73)
        self.assertEqual(self.short_tail_events(), [])
        self.short_tail_fixture()
        path = self.root / 'people/review.json'
        path.write_text('{', encoding='utf-8')
        self.assertEqual(self.short_tail_events(), [])

    def test_short_tail_rejects_tampered_truncated_csv(self):
        self.short_tail_fixture()
        path = self.root / 'people/review.csv'
        path.write_text('\n'.join(path.read_text(encoding='utf-8').splitlines()[:-2]), encoding='utf-8')
        self.assertEqual(self.short_tail_events(), [])

    def test_short_tail_requires_three_distinct_cash_samples_before_and_after(self):
        for missing in ((14,), (9, 10, 11), tuple(range(16))):
            self.short_tail_fixture(cash_rows=[[] if index in missing else [CASH] for index in range(16)])
            self.assertEqual(self.short_tail_events(), [])

    def test_short_tail_cash_ambiguity_and_shifted_object_do_not_create_event(self):
        self.short_tail_fixture(cash_rows=[[CASH, FOOD] for _ in range(16)])
        self.assertEqual(self.short_tail_events(), [])
        shifted = item('cash', [590, 200, 620, 220])
        self.short_tail_fixture(cash_rows=[[CASH] if index < 13 else [shifted] for index in range(16)])
        self.assertEqual(self.short_tail_events(), [])

    def test_short_tail_needs_visible_table_at_every_evidence_sample(self):
        self.short_tail_fixture()
        path = self.root / 'desk/input/frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[14]['candidates'] = [obj for obj in rows[14]['candidates'] if obj['label'] != 'work_table']
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        self.assertEqual(self.short_tail_events(), [])

    def test_short_tail_duplicate_samples_do_not_replace_missing_cash_observations(self):
        self.short_tail_fixture(cash_rows=[[CASH] if index < 14 else [] for index in range(16)])
        path = self.root / 'desk/input/frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[13]['candidates'].extend([CASH, CASH])
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        self.assertEqual(self.short_tail_events(), [])

    def test_short_tail_another_person_at_last_visible_frame_blocks_review(self):
        self.short_tail_fixture([1] * 62 + [2] + [0] * 17)
        self.assertEqual(self.short_tail_events(), [])

    def test_early_last_timestamp_suppresses_unattended_alert(self):
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.write_people([1] * 20 + [0] * 60, last_timestamp=3.77)
        self.assertEqual(self.events('unattended_assets'), [])

    def test_absence_from_video_start_does_not_imply_abandonment(self):
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.write_people([0] * 80)
        self.assertEqual(self.events('unattended_assets'), [])

    def test_short_person_dropout_is_not_unattended(self):
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.write_people([1] * 30 + [0] * 10 + [1] * 40)
        self.assertEqual(self.events('unattended_assets'), [])

    def test_complete_people_gap_and_persistent_laptop_create_review_episode(self):
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.write_people([1] * 30 + [0] * 50)
        events = self.events('unattended_assets')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['start_sec'], 3)
        self.assertEqual(events[0]['labels'], ['laptop'])
        self.assertTrue(events[0]['context']['people_decode_complete'])
        self.assertEqual(events[0]['source_frame'], 30)

    def test_cash_inside_bag_does_not_become_cash_left_candidate(self):
        bag = item('bag', [530, 175, 610, 270])
        self.write_mode('trained', [[bag] for _ in range(8)])
        self.write_mode('desk', [[TABLE, CASH] for _ in range(8)])
        self.write_people([1] * 30 + [0] * 50)
        events = self.events('unattended_assets')
        self.assertTrue(events)
        self.assertTrue(all('cash' not in event['labels'] for event in events))
        self.assertEqual(self.events('cash_after_client'), [])

    def test_global_count_drop_without_edge_exit_does_not_mean_client_left(self):
        self.write_mode('desk', [[TABLE, CASH] for _ in range(8)])
        self.write_people([2] * 30 + [1] * 50)
        self.assertEqual(self.events('cash_after_client'), [])
        self.assertEqual(self.coverage('cash_after_client')['status'], 'not_checked')
        self.assertEqual(self.coverage('one_client_at_a_time')['status'], 'not_checked')

    def test_outward_exit_without_cashier_visitor_roles_does_not_mean_client_left(self):
        self.write_mode('desk', [[TABLE, CASH] for _ in range(8)])
        def boxes(frame, identity):
            if identity == 1:
                return [100, 80, 180, 380]
            x = 450 + frame * 5
            return [x, 120, min(640, x + 80), 350]
        self.write_people([2] * 30 + [1] * 50, boxes=boxes)
        events = self.events('cash_after_client')
        self.assertEqual(events, [])
        self.assertEqual(self.coverage('cash_after_client')['assessment_status'], 'insufficient_evidence')
        self.assertTrue(self.report()['actions']['checks'])

    def test_missing_people_csv_is_not_an_empty_room(self):
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.write_people([1] * 30 + [0] * 50)
        (self.root / 'people/review.csv').unlink()
        self.assertEqual(self.events('unattended_assets'), [])
        self.assertFalse(self.report()['data_quality']['people_complete'])

    def test_truncated_csv_with_same_track_ids_is_not_an_empty_room(self):
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.write_people([1] * 80)
        path = self.root / 'people/review.csv'
        lines = path.read_text(encoding='utf-8').splitlines()
        path.write_text('\n'.join(lines[:21]) + '\n', encoding='utf-8')
        self.assertEqual(self.events('unattended_assets'), [])
        self.assertFalse(self.report()['data_quality']['people_complete'])
        self.assertIn('Контрольная сумма', self.report()['data_quality']['people_detail'])

    def test_old_people_report_without_csv_integrity_is_not_checked(self):
        self.write_mode('trained', [[LAPTOP] for _ in range(8)])
        self.write_people([1] * 30 + [0] * 50)
        path = self.root / 'people/review.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        del report['csv_sha256']
        path.write_text(json.dumps(report), encoding='utf-8')
        self.assertEqual(self.events('unattended_assets'), [])
        self.assertEqual(self.coverage('unattended_assets')['status'], 'not_checked')

    def test_door_and_action_rules_are_explicitly_not_checked(self):
        report = self.report()
        coverage = {row['rule_id']: row for row in report['rule_coverage']}
        self.assertEqual(coverage['door_open']['status'], 'not_checked')
        self.assertEqual(coverage['door_open']['user_threshold_sec'], 3.5)
        self.assertEqual(coverage['eating_while_cash_present']['implementation_status'], 'not_implemented')
        self.assertFalse(report['analysis_settings']['eating_while_cash_present']['implemented'])
        for name in ['safe_closed', 'cash_counted_visible', 'change_through_counter', 'original_id_verification',
                     'previous_client_documents', 'peripheral_test', 'one_client_at_a_time']:
            self.assertEqual(coverage[name]['status'], 'not_checked')
        self.assertEqual(report['verdict'], 'not_evaluated')


if __name__ == '__main__':
    unittest.main()
