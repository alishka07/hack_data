import unittest
from check_coverage import describe_coverage


class CoverageTests(unittest.TestCase):
    def test_no_candidate_does_not_mean_unimplemented_or_full_compliance(self):
        result = describe_coverage([dict(rule_id='door_open', status='partial', event_count=0, detail='Door samples')],
                                   {}, {'state_counts': {'door': {'closed': 12, 'unknown': 2}}}, {})[0]
        self.assertEqual(result['assessment_status'], 'partial_scope')
        self.assertTrue(result['checks'][0]['available'])
        self.assertEqual(result['checks'][0]['status'], 'analyzed')
        self.assertIn('12', result['detail'])

    def test_unimplemented_and_missing_inputs_are_different(self):
        results = describe_coverage([dict(rule_id=key, status='not_checked', detail='')
                                     for key in ('peripheral_test', 'cash_after_client')], {}, {}, {})
        self.assertEqual([row['assessment_status'] for row in results], ['not_implemented', 'insufficient_evidence'])


if __name__ == '__main__': unittest.main()
