import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visual_states import DEFAULT_MODEL, MULTIVIEW_MODEL, ROIS, SceneRegistration, VisualStateDetector, build_intervals, detect_video
from rule_settings import DOOR_OPEN_THRESHOLD_SEC


def observations(states, times=None):
    times = times or [i * .25 for i in range(len(states))]
    return [dict(timestamp_sec=t, frame_index=round(t*30), door={'state': state}, safe={'state': 'unknown'})
            for t,state in zip(times,states)]


class IntervalTests(unittest.TestCase):
    def test_failed_calibration_does_not_accept_an_arbitrary_default(self):
        from prepare_visual_states import choose_thresholds
        # Indistinguishable appearances with contradictory labels must abstain.
        features=np.zeros((2,5),np.float32)
        limits,report=choose_thresholds(features,np.array(['closed','open']),[1,2])
        self.assertEqual(report['accepted'],0)
        self.assertTrue(limits['max_distance']<=.005)

    def test_exactly_three_point_five_observed_seconds_reaches_threshold(self):
        rows=observations(['open']*15)
        interval=build_intervals(rows,.25)[0]
        self.assertEqual(interval['observed_duration_sec'],3.5)
        self.assertEqual(interval['threshold_sec'], DOOR_OPEN_THRESHOLD_SEC)
        self.assertTrue(interval['threshold_reached'])
        self.assertFalse(build_intervals(rows[:-1],.25)[0]['threshold_reached'])
        self.assertNotIn('reaches_two_seconds', interval)

    def test_threshold_boundary_349_350_351(self):
        for end, expected in [(3.49, False), (3.5, True), (3.51, True)]:
            with self.subTest(end=end):
                times = [i * .25 for i in range(14)] + [end]
                interval = build_intervals(observations(['open'] * len(times), times), .3)[0]
                self.assertEqual(interval['observed_duration_sec'], end)
                self.assertEqual(interval['threshold_reached'], expected)

    def test_sample_count_does_not_add_an_extra_interval_to_duration(self):
        # Fourteen .25-second samples span 3.25 seconds, not 3.5 seconds.
        interval = build_intervals(observations(['open'] * 14), .25)[0]
        self.assertEqual(interval['sample_count'], 14)
        self.assertEqual(interval['observed_duration_sec'], 3.25)
        self.assertFalse(interval['threshold_reached'])

    def test_unknown_cannot_bridge_door_threshold(self):
        intervals = build_intervals(observations(['open'] * 8 + ['unknown'] + ['open'] * 8), .25)
        self.assertEqual(len(intervals), 2)
        self.assertTrue(all(not interval['threshold_reached'] for interval in intervals))

    def test_missing_sample_at_low_fps_breaks_interval(self):
        rows = observations(['open'] * 17)
        for row in rows:
            row['frame_index'] = round(row['timestamp_sec'] * 4)
        rows.pop(8)
        intervals = build_intervals(rows, .25, fps=4)
        self.assertEqual(len(intervals), 2)
        self.assertTrue(all(not interval['threshold_reached'] for interval in intervals))

    def test_unknown_and_closed_split_intervals(self):
        for state in ['unknown','closed']:
            intervals=build_intervals(observations(['open']*5+[state]+['open']*5),.25)
            self.assertEqual(len(intervals),2)
            self.assertTrue(all(x['observed_duration_sec']==1 for x in intervals))

    def test_missing_observations_split_intervals(self):
        intervals=build_intervals(observations(['open']*4,[0,.25,2,2.25]),.25)
        self.assertEqual(len(intervals),2)
        self.assertTrue(all(x['observed_duration_sec']==.25 for x in intervals))

    def test_one_observation_is_zero_duration(self):
        self.assertEqual(build_intervals(observations(['open']),.25)[0]['observed_duration_sec'],0)

    def test_timestamps_must_be_strictly_increasing(self):
        for times in [[0,0],[1,0],[0,float('nan')]]:
            with self.assertRaises(ValueError):build_intervals(observations(['open']*2,times),.25)

    def test_safe_open_does_not_claim_unattended_or_unlocked(self):
        rows=observations(['closed']*9)
        for row in rows:row['safe']={'state':'open','locked':None}
        interval=build_intervals(rows,.25)[0]
        self.assertEqual(interval['object'],'safe')
        self.assertIn('not_a_confirmed_violation',interval['interpretation'])
        self.assertNotIn('unattended',interval)
        self.assertNotIn('threshold_sec',interval)
        self.assertNotIn('threshold_reached',interval)


@unittest.skipUnless((DEFAULT_MODEL/'model.json').exists(),'Reviewed state model has not been prepared.')
class ActualReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.detector=VisualStateDetector()

    def test_blank_scene_unknown(self):
        result=self.detector.predict(np.zeros((848,464,3),np.uint8))
        self.assertTrue(all(result[k]['state']=='unknown' for k in ROIS))
        self.assertIsNone(result['safe']['locked'])

    def test_different_aspect_ratio_unknown(self):
        result=self.detector.predict(np.zeros((400,640,3),np.uint8))
        self.assertFalse(result['scene']['matched'])
        self.assertTrue(all(result[k]['state']=='unknown' for k in ROIS))

    def test_fully_occluded_state_region_unknown(self):
        ref=cv2.imread(str(DEFAULT_MODEL/'reference.jpg'))
        for kind,(x1,y1,x2,y2) in ROIS.items():
            for color in [(0,0,0),(240,240,240),(150,90,180)]:
                frame=ref.copy();frame[y1:y2,x1:x2]=color
                self.assertEqual(self.detector.predict(frame)[kind]['state'],'unknown',(kind,color))

    def test_video_integrity_and_unknown_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'unfamiliar.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20.,(64,116))
            self.assertTrue(writer.isOpened())
            for _ in range(30):writer.write(np.zeros((116,64,3),np.uint8))
            writer.release()
            report=detect_video(path,Path(folder)/'result',.25)
            self.assertEqual(report['status'],'completed')
            self.assertTrue(report['frame_count_matches'])
            self.assertEqual(report['decoded_frames'],30)
            self.assertEqual(report['intervals'],[])
            self.assertEqual(len(report['video']['sha256']),64)
            self.assertTrue(all(row['door']['state']=='unknown' for row in report['observations']))
            self.assertTrue((Path(folder)/'result/observations.json').exists())


@unittest.skipUnless((MULTIVIEW_MODEL/'model.json').exists(),'Reviewed multiview model has not been prepared.')
class MultiViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.detector=VisualStateDetector(MULTIVIEW_MODEL)

    def test_view_selection_uses_scene_not_filenames(self):
        for view_id in ['desk_back_left','desk_back_right','desk_front_white']:
            frame=cv2.imread(str(MULTIVIEW_MODEL/'views'/view_id/'reference.jpg'))
            result=self.detector.predict(frame)
            self.assertTrue(result['scene']['matched'])
            self.assertEqual(result['scene']['view_id'],view_id)
            self.assertEqual(result['door']['state'],'closed')

    def test_missing_safe_examples_do_not_invent_a_closed_safe(self):
        frame=cv2.imread(str(MULTIVIEW_MODEL/'views/desk_back_left/reference.jpg'))
        result=self.detector.predict(frame)
        self.assertEqual(result['safe']['state'],'unknown')
        self.assertEqual(result['safe']['reason'],'no_reviewed_state_examples_for_view')
        self.assertIsNone(result['safe']['locked'])

    def test_reflection_context_is_registered_and_not_a_mask(self):
        frame=cv2.imread(str(MULTIVIEW_MODEL/'views/desk_front_white/reference.jpg'))
        result=self.detector.project_regions(frame)
        regions=result['regions']['reflections']
        self.assertEqual(len(regions),1)
        self.assertEqual(regions[0]['view_id'],'desk_front_white')
        self.assertEqual(regions[0]['interpretation'],'context_only_not_a_person_exclusion_mask')
        self.assertEqual(len(regions[0]['polygon']),4)
        for actual,expected in zip(regions[0]['box'],[48,210,120,278]):
            self.assertAlmostEqual(actual,expected,delta=1)

    def test_unfamiliar_scene_has_no_projected_reflection_regions(self):
        result=self.detector.project_regions(np.zeros((848,464,3),np.uint8))
        self.assertFalse(result['scene']['matched'])
        self.assertEqual(result['regions']['reflections'],[])

    def test_open_door_does_not_supply_closed_plane_reflection_regions(self):
        root=MULTIVIEW_MODEL.parent.parent
        frame=cv2.imread(str(root/'work/states_v2/frames/X1_002.0.jpg'))
        self.assertIsNotNone(frame)
        result=self.detector.predict(frame)
        self.assertEqual(result['door']['state'],'open')
        self.assertEqual(result['scene']['regions']['reflections'],[])

    def test_occluded_new_view_door_is_unknown(self):
        for view in self.detector.views[1:]:
            frame=view['registration'].reference.copy()
            x1,y1,x2,y2=view['metadata']['rois']['door']
            frame[y1:y2,x1:x2]=0
            self.assertEqual(self.detector.predict(frame)['door']['state'],'unknown',view['view_id'])


if __name__=='__main__':unittest.main()
