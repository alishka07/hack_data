"""Semantic checks: explanations must never strengthen available evidence."""
from copy import deepcopy
import unittest

from episode_steps import attach_episode_steps, build_event_steps


def evidence(time, label, frame=None):
    return {'timestamp_sec': time, 'source_frame': round(time * 10) if frame is None else frame,
            'mode': 'visual_states' if label in {'door', 'safe'} else 'desk', 'labels': [label]}


class EpisodeStepsTests(unittest.TestCase):
    def text(self, steps):
        return ' '.join(item['title'] + ' ' + item['detail'] for item in steps)

    def test_door_boundary_uses_current_threshold_and_real_evidence(self):
        for duration, reached in [(3.49, False), (3.5, True), (3.51, True)]:
            with self.subTest(duration=duration):
                event = {'rule_id': 'door_open', 'start_sec': 0, 'end_sec': 100,
                         'context': {'duration_basis': 'observed_sample_span'},
                         'evidence_times': [evidence(4 + duration, 'door'), evidence(4, 'door')]}
                steps = build_event_steps(event)
                rule = next(item for item in steps if item['kind'] == 'rule')
                self.assertIn('3,5 секунды', rule['detail'])
                self.assertIn('порог достигнут.' if reached else 'порог не достигнут.', rule['detail'])
                self.assertEqual([item['timestamp_sec'] for item in steps if 'timestamp_sec' in item], [4, 4 + duration])
                self.assertNotIn('100.00', self.text(steps))

    def test_door_without_interval_provenance_does_not_claim_continuity(self):
        steps = build_event_steps({'rule_id': 'door_open', 'evidence_times': [evidence(0, 'door'), evidence(6, 'door')]})
        self.assertNotIn('порог достигнут', self.text(steps))
        self.assertNotIn('открытие продолжается', self.text(steps))

    def test_cash_requires_before_exit_and_after_cash(self):
        rows = [evidence(6, 'cash'), evidence(2, 'cash'), evidence(4, 'person')]
        event = {'rule_id': 'cash_after_client', 'context': {'visitor_exit_sec': 4, 'same_projected_location': True}, 'evidence_times': rows}
        steps = build_event_steps(event)
        timed = [step for step in steps if 'timestamp_sec' in step]
        self.assertEqual([step['timestamp_sec'] for step in timed], [2, 4, 6])
        self.assertEqual(timed[1]['evidence']['labels'], ['person'])
        self.assertIn('при обслуживании нарушением не считается', self.text(steps))
        self.assertEqual(steps[-1]['kind'], 'limitation')

    def test_cash_presence_without_exit_cannot_become_departure(self):
        for context in [{}, {'visitor_exit_sec': 4, 'same_projected_location': True}]:
            steps = build_event_steps({'rule_id': 'cash_after_client', 'context': context,
                                       'evidence_times': [evidence(5, 'cash'), evidence(6, 'cash')]})
            self.assertFalse(any(step['title'] == 'Посетитель уходит из видимой зоны' for step in steps))
            self.assertIn('Один кассир в кадре и деньги на столе не доказывают', self.text(steps))

    def test_person_evidence_cannot_replace_cash_after_exit(self):
        steps = build_event_steps({'rule_id': 'cash_after_client',
                                  'context': {'visitor_exit_sec': 4, 'same_projected_location': True},
                                  'evidence_times': [evidence(2, 'cash'), evidence(4, 'person'), evidence(6, 'person')]})
        self.assertFalse(any(step['title'] == 'Предмет остаётся после ухода' for step in steps))

    def test_invalid_timestamps_are_not_printed_or_used(self):
        steps = build_event_steps({'rule_id': 'cash_after_client', 'context': {'visitor_exit_sec': True},
                                  'evidence_times': [evidence(float('nan'), 'cash', 2), evidence(-1, 'cash', 3), evidence(True, 'cash', 4)]})
        self.assertFalse(any('timestamp_sec' in step for step in steps))

    def test_generic_first_frame_does_not_borrow_later_object(self):
        steps = build_event_steps({'rule_id': 'unattended_assets',
                                  'evidence_times': [evidence(1, 'person'), evidence(4, 'cash')]})
        self.assertIn('траектория человека', steps[0]['detail'])
        self.assertNotIn('банкноты', steps[0]['detail'])

    def test_safe_does_not_prove_locked_or_receipt(self):
        steps = build_event_steps({'rule_id': 'safe_closed', 'context': {'cashier_departure': {'exit_sec': 1}},
                                  'evidence_times': [evidence(2, 'safe'), evidence(4, 'safe')]})
        self.assertEqual([step['timestamp_sec'] for step in steps if 'timestamp_sec' in step], [1, 2, 4])
        self.assertIn('Не установлены получение денег', self.text(steps))
        self.assertIn('Закрытая дверца не означает запертый', self.text(steps))

    def test_food_rule_explains_presence_without_claiming_eating_or_cash(self):
        steps = build_event_steps({'rule_id': 'food_drinks_at_workplace',
                                  'context': {'spatial_relations': ['cashier_near_table']},
                                  'evidence_times': [evidence(1, 'drink'), evidence(4, 'drink')]})
        self.assertEqual([step['timestamp_sec'] for step in steps if 'timestamp_sec' in step], [1, 4])
        self.assertIn('независимо от наличия денег и ухода клиента', self.text(steps))
        self.assertIn('удержание рукой этим не доказано', self.text(steps))
        self.assertIn('Не установлены приём пищи или питьё', self.text(steps))
        self.assertNotIn('пьёт кофе', self.text(steps))

    def test_food_procedure_is_implemented_even_if_eating_is_planned(self):
        result = attach_episode_steps({'events': [], 'rule_coverage': [
            {'rule_id': 'food_drinks_at_workplace', 'status': 'partial'},
            {'rule_id': 'eating_while_cash_present', 'status': 'not_checked', 'implementation_status': 'not_implemented'}]})
        self.assertEqual([row['procedure_status'] for row in result['rule_coverage']], ['method', 'planned'])
        self.assertIn('Шоколадка', result['rule_coverage'][1]['procedure_steps'][0]['detail'])

    def test_food_step_source_comes_from_its_frame_not_event_summary(self):
        model = dict(evidence(1, 'food'), source='model')
        matched = dict(evidence(4, 'food'), source='human_confirmed_reference_match')
        for rows in ([model, matched], [dict(matched, timestamp_sec=1), dict(model, timestamp_sec=4)]):
            steps = build_event_steps({'rule_id': 'food_drinks_at_workplace',
                                       'context': {'object_evidence_source': 'human_confirmed_reference_match'},
                                       'evidence_times': rows})
            timed = [step for step in steps if 'timestamp_sec' in step]
            self.assertEqual(len(timed), 2)
            for step in timed:
                if step['evidence']['source'] == 'model':
                    self.assertIn('класс предложен моделью', step['detail'])
                    self.assertNotIn('найдено совпадение', step['detail'])
                else:
                    self.assertIn('найдено совпадение', step['detail'])
                    self.assertIn('не новое подтверждение человеком', step['detail'])

    def test_food_step_without_source_does_not_inherit_human_confirmation(self):
        steps = build_event_steps({'rule_id': 'food_drinks_at_workplace',
                                   'context': {'object_evidence_source': 'human_confirmed_reference_match'},
                                   'evidence_times': [evidence(1, 'food'), evidence(4, 'food')]})
        timed = [step for step in steps if 'timestamp_sec' in step]
        self.assertTrue(all('Источник классификации этого кадра не указан' in step['detail'] for step in timed))
        self.assertTrue(all('найдено совпадение' not in step['detail'] for step in timed))

    def test_change_path_is_not_counting_or_return_proof(self):
        steps = build_event_steps({'rule_id': 'change_through_counter',
                                  'context': {'counter_observed_throughout_chain': True, 'counter_approach_observed_in_chain': False},
                                  'evidence_times': [evidence(1, 'cash'), evidence(3, 'cash')]})
        self.assertIn('в этой цепочке не наблюдалось', self.text(steps))
        self.assertIn('пересчёт до начала цепочки', self.text(steps))
        self.assertNotIn('деньги не пересчитаны', self.text(steps))

    def test_short_clip_cash_steps_describe_absence_without_inventing_exit(self):
        cash_before = dict(evidence(5, 'cash'), source='human_confirmed_reference_match')
        cash_first = dict(evidence(6.5, 'cash'), source='human_confirmed_reference_match')
        cash_last = dict(evidence(7.5, 'cash'), source='model')
        person = dict(evidence(6.2, 'person'), mode='people', source='tracker')
        event = {'rule_id': 'unattended_assets', 'start_sec': 6.5, 'end_sec': 7.5,
                 'context': {'recording_ends_during_absence': True, 'absence': {'start_sec': 6.3}},
                 'evidence_times': [cash_last, person, cash_first, cash_before]}
        steps = build_event_steps(event)
        self.assertEqual([step['timestamp_sec'] for step in steps if 'timestamp_sec' in step], [5, 6.2, 6.5, 7.5])
        text = self.text(steps)
        self.assertIn('Запись заканчивается', text)
        self.assertIn('не разрешённое время', text)
        self.assertIn('Не установлен выход кассира или клиента', text)
        self.assertNotIn('достигает края', text)
        for step in steps:
            if step.get('evidence', {}).get('source') == 'human_confirmed_reference_match':
                self.assertIn('класс которого уточнён пользователем', step['detail'])
                self.assertIn('не новое подтверждение человеком', step['detail'])
            if step.get('evidence', {}).get('source') == 'model':
                self.assertIn('предложен моделью', step['detail'])
                self.assertNotIn('найдено совпадение', step['detail'])

    def test_short_clip_steps_do_not_inherit_confirmation_or_invent_missing_context(self):
        event = {'rule_id': 'unattended_assets',
                 'context': {'recording_ends_during_absence': True, 'absence': {'start_sec': 6.3},
                             'object_evidence_source': 'human_confirmed_reference_match'},
                 'evidence_times': [evidence(5, 'cash'), evidence(6.5, 'cash'), evidence(7.5, 'cash')]}
        steps = build_event_steps(event)
        self.assertIn('Источник классификации этого кадра не указан', self.text(steps))
        self.assertNotIn('найдено совпадение', self.text(steps))
        event['context']['absence'] = None
        steps = build_event_steps(event)
        self.assertNotIn('Последнее наблюдение кассира', self.text(steps))
        self.assertIn('Не установлен выход кассира или клиента', self.text(steps))

    def test_attach_preserves_input_and_verdicts_and_separates_methods(self):
        original = {'events': [{'rule_id': 'door_open', 'status': 'review', 'evidence_times': []}],
                    'rule_coverage': [{'rule_id': 'cash_after_client', 'status': 'partial', 'event_count': 0},
                                      {'rule_id': 'door_open', 'status': 'not_checked'},
                                      {'rule_id': 'peripheral_test', 'status': 'not_checked', 'implementation_status': 'not_implemented'}]}
        baseline = deepcopy(original)
        result = attach_episode_steps(original)
        self.assertEqual(original, baseline)
        self.assertEqual(result['events'][0]['status'], 'review')
        rows = result['rule_coverage']
        self.assertEqual([row['procedure_status'] for row in rows], ['method', 'insufficient_evidence', 'planned'])
        self.assertEqual(rows[0]['event_count'], 0)
        self.assertTrue(all(step['kind'] == 'procedure' for row in rows for step in row['procedure_steps']))
        self.assertTrue(all(step['status'] == 'planned' for step in rows[-1]['procedure_steps']))
        self.assertIn('ещё не реализован', rows[-1]['procedure_note'])


if __name__ == '__main__':
    unittest.main()
