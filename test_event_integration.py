"""Keep independent asset evidence while removing duplicate review episodes."""
import unittest

from rule_engine import _deduplicate_absence_events, _event


def episode(objects, source='trained'):
    observations = [dict(time=float(t), frame=t * 30, mode=source,
                         playback_sec=float(t), box=box, label=label)
                    for label, box in objects for t in (2, 3, 4)]
    return _event('unattended_assets', 'review', observations, 'observed absence', [], {})


class DuplicateAbsenceTests(unittest.TestCase):
    def test_same_object_and_times_prefers_role_episode(self):
        laptop = [('laptop', [10, 10, 30, 30])]
        self.assertEqual(_deduplicate_absence_events([episode(laptop)], [episode(laptop)]), [])

    def test_other_asset_in_same_legacy_episode_survives(self):
        laptop, bag = ('laptop', [10, 10, 30, 30]), ('bag', [60, 10, 90, 40])
        result = _deduplicate_absence_events([episode([laptop, bag])], [episode([laptop])])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['labels'], ['bag'])
        self.assertEqual(result[0]['evidence_box'], bag[1])
        self.assertEqual(len(result[0]['evidence_times']), 3)

    def test_same_label_at_another_location_is_independent(self):
        old = episode([('laptop', [10, 10, 30, 30])])
        new = episode([('laptop', [70, 10, 90, 30])])
        self.assertEqual(_deduplicate_absence_events([old], [new]), [old])

    def test_partially_overlapping_track_is_not_silently_shortened(self):
        old = episode([('laptop', [10, 10, 30, 30])])
        new = episode([('laptop', [10, 10, 30, 30])])
        new['evidence_times'] = new['evidence_times'][1:]
        self.assertEqual(_deduplicate_absence_events([old], [new]), [old])


if __name__ == '__main__':
    unittest.main()
