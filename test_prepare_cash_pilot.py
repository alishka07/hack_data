import unittest
from prepare_cash_pilot import validate_rows,checked_box
from detect_cash_roi import map_cash_predictions

class CashPilotTests(unittest.TestCase):
    def rows(self):
        return {'frames':[{'source_sha256':s,'source_frame':0,'split':p,'review_status':'assistant_reviewed_for_cash_only','complete_for_cash_only':True} for s,p in [('a','train'),('b','val'),('c','heldout')]]}
    def test_duplicate_video_across_splits_rejected_even_different_names(self):
        r=self.rows();r['frames'][2]['source_sha256']='a';r['frames'][2]['source_frame']=5
        with self.assertRaises(ValueError):validate_rows(r)
    def test_unknown_completeness_never_becomes_negative_training_frame(self):
        r=self.rows();r['frames'].append({'review_status':'uncertain','complete_for_cash_only':False})
        self.assertEqual(len(validate_rows(r)),3)
    def test_heldout_is_separate_test(self):
        self.assertEqual(validate_rows(self.rows())[2]['split'],'test')
    def test_roi_box_outside_image_rejected(self):
        with self.assertRaises(ValueError):checked_box([20,10,105,20],100,100)
    def test_crop_predictions_mapped_once_and_clipped_to_source_region(self):
        rows=map_cash_predictions([{'label':'cash','box':[-3,5,40,35],'score':.8}], [20,280,400,520],464,848)
        self.assertEqual(rows[0]['box'],[20,285,60,315])
        self.assertFalse(rows[0]['violation_established'])
        self.assertEqual(map_cash_predictions([{'label':'cash','box':[500,1,520,10],'score':.8}],[20,280,400,520],464,848),[])

if __name__=='__main__':unittest.main()
