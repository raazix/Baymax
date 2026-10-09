import unittest

from core.upload_inspection import _apply_severity, _merge_detections
from vision.severity import classify_severity


class SeverityRulesV3Tests(unittest.TestCase):
    def test_class_sets_base_and_size_escalates(self):
        self.assertEqual(classify_severity('scratches', length_mm=3)['level'], 'low')
        self.assertEqual(classify_severity('scratches', length_mm=6)['level'], 'medium')
        self.assertEqual(classify_severity('inclusion', length_fraction=.05)['level'], 'medium')
        self.assertEqual(classify_severity('inclusion', length_fraction=.2)['level'], 'high')

    def test_only_crack_like_defects_reach_critical(self):
        self.assertEqual(classify_severity('crazing', length_mm=18, zone='rim')['level'], 'critical')
        scratch = classify_severity('scratches', length_mm=40, finding_count=5)
        self.assertEqual(scratch['level'], 'high')
        self.assertIn('capped', scratch['reason'])

    def test_critical_needs_physical_evidence(self):
        pixels_only = classify_severity('crazing', length_fraction=.6, finding_count=4)
        self.assertEqual(pixels_only['level'], 'high')
        self.assertIn('measured size', pixels_only['reason'])
        self.assertEqual(classify_severity('crazing', length_mm=20)['level'], 'critical')
        self.assertEqual(classify_severity('crazing', length_fraction=.05, zone='bore / hub')['level'], 'critical')

    def test_unclassified_anomaly_never_automatically_critical(self):
        result = classify_severity('anomaly_unclassified', length_fraction=.6, zone='rim', finding_count=4)
        self.assertEqual(result['level'], 'high')
        self.assertTrue(result['review_required'])
        self.assertEqual(classify_severity('anomaly_unclassified', length_fraction=.05)['level'], 'medium')

    def test_location_matters_for_cracks_not_cosmetics(self):
        self.assertEqual(classify_severity('crazing', length_fraction=.05, zone='mid-radius')['level'], 'high')
        self.assertEqual(classify_severity('crazing', length_fraction=.05, zone='rim')['level'], 'critical')
        self.assertEqual(classify_severity('scratches', length_fraction=.05, zone='rim')['level'], 'low')

    def test_confidence_and_score_are_not_inputs(self):
        with self.assertRaises(TypeError):
            classify_severity('crazing', confidence=.99)
        self.assertIn('not severity inputs', classify_severity('crazing')['inputs_excluded'])


class SpatialMergeTests(unittest.TestCase):
    def defect(self):
        return {'label': 'anomaly_unclassified', 'region_box_px': [100, 100, 200, 200], 'length_px': 100, 'zone': 'rim'}

    def test_box_inside_flagged_region_names_the_anomaly(self):
        defects = [self.defect()]
        merge = _merge_detections(defects, {'model_sha256': 'x', 'detections': [
            {'label': 'crazing', 'confidence': .6, 'bbox_xyxy_px': [110, 110, 180, 170]},
            {'label': 'scratches', 'confidence': .9, 'bbox_xyxy_px': [300, 300, 380, 380]}]})
        self.assertEqual(defects[0]['class_hint']['label'], 'crazing')
        self.assertEqual((merge['merged'], merge['unconfirmed']), (1, 1))
        _apply_severity(defects, None, (512, 512))
        self.assertEqual(defects[0]['severity']['level'], 'critical')    # crack-like hint + 20% size + rim
        self.assertIn('class hint', defects[0]['severity']['factors'][0]['source'])

    def test_unlocalised_region_names_nothing(self):
        defects = [{'label': 'anomaly_unclassified', 'region_box_px': [0, 0, 500, 500], 'length_px': 500, 'zone': 'rim'}]
        merge = _merge_detections(defects, {'detections': [{'label': 'crazing', 'confidence': .9, 'bbox_xyxy_px': [100, 100, 150, 150]}]}, (512, 512))
        self.assertNotIn('class_hint', defects[0])
        self.assertIn('covers most of the image', defects[0]['class_hint_skipped'])
        self.assertEqual(merge['merged'], 0)

    def test_large_or_weak_boxes_are_not_merged(self):
        defects = [self.defect()]
        merge = _merge_detections(defects, {'detections': [
            {'label': 'inclusion', 'confidence': .8, 'bbox_xyxy_px': [0, 0, 400, 400]},     # mostly outside the region
            {'label': 'crazing', 'confidence': .2, 'bbox_xyxy_px': [120, 120, 160, 160]}]})  # below detector cut-off
        self.assertNotIn('class_hint', defects[0])
        self.assertEqual(merge['merged'], 0)
        _apply_severity(defects, None, (512, 512))
        self.assertEqual(defects[0]['severity']['level'], 'high')         # unclassified stays capped at high


if __name__ == '__main__':
    unittest.main()
