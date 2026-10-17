import importlib.util
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('portable_bbox',ROOT/'runtime/ByteTrack-main/yolox/tracker/bbox.py')
bbox=importlib.util.module_from_spec(spec);spec.loader.exec_module(bbox)
class PortableRuntimeTests(unittest.TestCase):
    def test_overlap_preserves_inclusive_pixel_convention(self):
        result=bbox.bbox_overlaps([[0,0,9,9]],[[0,0,9,9],[5,5,9,9],[20,20,30,30]])
        np.testing.assert_allclose(result,[[1,.25,0]])
    def test_empty_boxes(self):
        self.assertEqual(bbox.bbox_overlaps([],[[0,0,9,9]]).shape,(0,1))
    def test_model_paths_are_inside_project(self):
        from train_objects import REPO,BASE_WEIGHTS
        self.assertTrue(REPO.is_relative_to(ROOT));self.assertTrue(BASE_WEIGHTS.is_relative_to(ROOT))
if __name__=='__main__':unittest.main()
